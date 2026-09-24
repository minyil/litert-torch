# Copyright 2026 The LiteRT Torch Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Exportable module for the Qwen3-VL vision encoder.

The HF vision tower takes dynamically sized, pre-patchified pixel values plus a
`grid_thw` tensor. For LiteRT we export one signature per fixed image size
instead:

  * Input `images` is a [1, H, W, 3] float image in [0, 1] (what the LiteRT-LM
    runtime's stb preprocessor produces). Normalization and patchification
    happen in-graph.
  * A still image is fed to the Conv3D patch embedding as two identical
    temporal frames, so the Conv3D is folded into an equivalent Conv2D.
  * Everything that depends only on the grid (interpolated learned position
    embeddings, 2D vision RoPE) is precomputed as constants.
  * All patches belong to a single image, so vision attention is plain
    unmasked attention.

Outputs, with N = (H / 32) * (W / 32) image tokens:
  * `features`: [1, N, text_hidden] merged image embeddings.
  * `deepstack_features`: [1, N, num_deepstack, text_hidden], added to the
    decoder hidden states after decoder layers 0..num_deepstack-1.
  * `mrope_offsets`: [1, N, 3] float (t, h, w) offsets of each image token
    relative to the image's first M-RoPE position.
"""

from litert_torch.generative.export_hf.core import exportable_module as exportable_module_base
import torch
import torch.nn.functional as F
from transformers import vision_utils

_IMAGE_SIZES_KWARG = 'qwen3_vl_image_sizes'
_DEFAULT_IMAGE_SIZES = '448'


def parse_image_sizes(spec) -> list[tuple[int, int]]:
  """Parses e.g. "448,672,896x512" into [(448, 448), (672, 672), (896, 512)]."""
  if isinstance(spec, (list, tuple)):
    spec = ','.join(str(s) for s in spec)
  sizes = []
  for item in str(spec).split(','):
    item = item.strip()
    if not item:
      continue
    if 'x' in item:
      h, w = item.split('x')
      sizes.append((int(h), int(w)))
    else:
      sizes.append((int(item), int(item)))
  return sizes


def _rotate_half(x):
  x1 = x[..., : x.shape[-1] // 2]
  x2 = x[..., x.shape[-1] // 2 :]
  return torch.cat((-x2, x1), dim=-1)


class LiteRTExportableModuleForQwen3VLVisionEncoder(
    exportable_module_base.ExportableModuleBase
):
  """Fixed-size Qwen3-VL vision encoder with DeepStack and M-RoPE outputs."""

  def __init__(self, model: torch.nn.Module, export_config):
    super().__init__(export_config)
    self.visual = model.model.visual
    cfg = self.visual.config
    self.patch_size = cfg.patch_size
    self.merge_size = cfg.spatial_merge_size
    self.num_heads = cfg.num_heads
    self.deepstack_visual_indexes = list(cfg.deepstack_visual_indexes)
    # Qwen3-VL uses mean = std = 0.5 for every channel.
    self.image_mean = 0.5
    self.image_std = 0.5

    # Two identical temporal frames through a Conv3D == one frame through the
    # Conv2D whose kernel is the sum over the temporal kernel axis.
    proj = self.visual.patch_embed.proj
    self.register_buffer(
        'patch_weight', proj.weight.detach().sum(dim=2), persistent=False
    )
    self.register_buffer(
        'patch_bias', proj.bias.detach().clone(), persistent=False
    )
    self._grid_constants = {}

  def _grid(self, height: int, width: int) -> tuple[int, int]:
    unit = self.patch_size * self.merge_size
    if height % unit or width % unit:
      raise ValueError(
          f'Image size {height}x{width} must be a multiple of {unit}.'
      )
    return height // self.patch_size, width // self.patch_size

  @torch.no_grad()
  def _constants(self, grid_h: int, grid_w: int):
    """Grid-dependent constants, in HF's merge-block patch order."""
    key = (grid_h, grid_w)
    if key in self._grid_constants:
      return self._grid_constants[key]
    visual = self.visual
    device = visual.pos_embed.weight.device
    grid_thw = torch.tensor([[1, grid_h, grid_w]], device=device)
    interp_indices, interp_weights = (
        vision_utils.get_vision_interpolation_indices_and_weights(
            grid_thw,
            num_grid_per_side=visual.num_grid_per_side,
            mode=visual.interpolation_mode,
            align_corners=visual.interpolation_align_corners,
            spatial_merge_size=self.merge_size,
            kwargs={},
        )
    )
    pos_embeds = (
        visual.pos_embed(interp_indices) * interp_weights[:, :, None]
    ).sum(1)
    position_ids = vision_utils.get_vision_position_ids(
        grid_thw, self.merge_size, kwargs={}
    )
    cos, sin = visual.rotary_pos_emb(pos_embeds, position_ids)

    merged_h, merged_w = grid_h // self.merge_size, grid_w // self.merge_size
    hh, ww = torch.meshgrid(
        torch.arange(merged_h, device=device),
        torch.arange(merged_w, device=device),
        indexing='ij',
    )
    mrope_offsets = torch.stack(
        [torch.zeros_like(hh), hh, ww], dim=-1
    ).reshape(1, -1, 3)

    consts = (
        pos_embeds.float(),
        cos.float()[:, None, :],
        sin.float()[:, None, :],
        mrope_offsets.float(),
    )
    self._grid_constants[key] = consts
    return consts

  def _attention(self, blk, hidden_states, cos, sin):
    attn = blk.attn
    seq_len = hidden_states.shape[0]
    q, k, v = (
        attn.qkv(hidden_states)
        .reshape(seq_len, 3, self.num_heads, -1)
        .permute(1, 0, 2, 3)
        .unbind(0)
    )
    q = q * cos + _rotate_half(q) * sin
    k = k * cos + _rotate_half(k) * sin
    # [seq, heads, dim] -> [1, heads, seq, dim]
    q, k, v = (t.transpose(0, 1).unsqueeze(0) for t in (q, k, v))
    out = F.scaled_dot_product_attention(q, k, v, scale=attn.scaling)
    out = out.squeeze(0).transpose(0, 1).reshape(seq_len, -1)
    return attn.proj(out)

  def forward(self, images):
    _, height, width, _ = images.shape
    grid_h, grid_w = self._grid(height, width)
    pos_embeds, cos, sin, mrope_offsets = self._constants(grid_h, grid_w)
    m = self.merge_size

    x = (images - self.image_mean) / self.image_std
    x = x.permute(0, 3, 1, 2)
    x = F.conv2d(x, self.patch_weight, self.patch_bias, stride=self.patch_size)
    # [1, D, gh, gw] -> merge-block order [gh/m, gw/m, m, m] -> [gh*gw, D]
    dim = x.shape[1]
    x = x.reshape(dim, grid_h // m, m, grid_w // m, m)
    x = x.permute(1, 3, 2, 4, 0).reshape(grid_h * grid_w, dim)
    hidden_states = x + pos_embeds

    deepstack = []
    for layer_num, blk in enumerate(self.visual.blocks):
      hidden_states = hidden_states + self._attention(
          blk, blk.norm1(hidden_states), cos, sin
      )
      hidden_states = hidden_states + blk.mlp(blk.norm2(hidden_states))
      if layer_num in self.deepstack_visual_indexes:
        merger = self.visual.deepstack_merger_list[
            self.deepstack_visual_indexes.index(layer_num)
        ]
        deepstack.append(merger(hidden_states))

    features = self.visual.merger(hidden_states)
    return {
        'features': features.unsqueeze(0),
        'deepstack_features': torch.stack(deepstack, dim=1).unsqueeze(0),
        'mrope_offsets': mrope_offsets,
    }

  def get_sample_inputs(
      self, model_config, **kwargs
  ) -> dict[str, tuple[dict[str, torch.Tensor], dict[str, torch.export.Dim]]]:
    """One signature per configured image size, named vision_{H}x{W}."""
    sizes = parse_image_sizes(
        kwargs.get(_IMAGE_SIZES_KWARG, _DEFAULT_IMAGE_SIZES)
    )
    ret = {}
    for height, width in sizes:
      self._grid(height, width)
      ret[f'vision_{height}x{width}'] = (
          {'images': torch.zeros((1, height, width, 3), dtype=torch.float32)},
          {},
      )
    return ret

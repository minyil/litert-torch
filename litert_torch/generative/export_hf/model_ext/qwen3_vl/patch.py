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
"""Export patches for the Qwen3-VL text decoder."""

import contextlib

from litert_torch.generative.export_hf.model_ext import patches as patches_lib
import torch
from transformers.models.qwen3_vl import modeling_qwen3_vl


def _dense_deepstack_process(self, hidden_states, visual_pos_masks, visual_embeds):
  """DeepStack injection without boolean indexing.

  The exported decoder receives DeepStack features for every token of the
  chunk (zeros at text positions), so the injection is a plain add. Boolean
  mask indexing has a data-dependent shape and cannot be exported.
  """
  if visual_pos_masks is not None:
    return _ORIGINAL_DEEPSTACK_PROCESS(
        self, hidden_states, visual_pos_masks, visual_embeds
    )
  return hidden_states + visual_embeds.to(hidden_states.dtype)


def _select_recomposition_frequencies(self, freq):
  """Interleaved M-RoPE recomposition via elementwise select.

  freq: [3, B, T, head_dim // 2] (t, h, w). Frequency index i takes the h
  angle if i % 3 == 1 and i < 3 * section_h, the w angle if i % 3 == 2 and
  i < 3 * section_w, and the t angle otherwise. Same result as the slice
  assignment in HF, but lowers to SELECT instead of scatter ops.
  """
  half = freq.shape[-1]
  idx = torch.arange(half, device=freq.device)
  use_h = (idx % 3 == 1) & (idx < 3 * self.mrope_section[1])
  use_w = (idx % 3 == 2) & (idx < 3 * self.mrope_section[2])
  freqs_thw = torch.where(
      use_h, freq[1], torch.where(use_w, freq[2], freq[0])
  )
  return torch.cat((freqs_thw, freqs_thw), dim=-1)


_ORIGINAL_DEEPSTACK_PROCESS = modeling_qwen3_vl.Qwen3VLTextModel._deepstack_process


@contextlib.contextmanager
def _patch_classes():
  text_model = modeling_qwen3_vl.Qwen3VLTextModel
  rotary = modeling_qwen3_vl.Qwen3VLTextRotaryEmbedding
  orig_rotary = rotary.recomposition_frequencies
  text_model._deepstack_process = _dense_deepstack_process
  rotary.recomposition_frequencies = _select_recomposition_frequencies
  try:
    yield
  finally:
    text_model._deepstack_process = _ORIGINAL_DEEPSTACK_PROCESS
    rotary.recomposition_frequencies = orig_rotary


@patches_lib.register_patch(['qwen3_vl', 'qwen3_vl_text'])
def qwen3_vl_load_patch():
  return contextlib.nullcontext()


@patches_lib.register_model_patch(['qwen3_vl', 'qwen3_vl_text'])
def qwen3_vl_model_patch(model, export_config):
  del model, export_config  # Class-level patches only.
  return _patch_classes()

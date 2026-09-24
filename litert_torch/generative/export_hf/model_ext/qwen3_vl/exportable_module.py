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
"""Prefill/decode exportables for the Qwen3-VL text decoder.

Compared to the generic external-embedder decoder, these take two extra
inputs:
  * `mrope_pos` int32 [3, T]: M-RoPE (t, h, w) positions. `input_pos` stays the
    KV-cache slot index; for Qwen3-VL the two diverge after an image.
  * `deepstack_embeddings` float32 [1, T, num_deepstack, hidden] (prefill
    only): added to the hidden states after decoder layers
    0..num_deepstack-1. Zeros at text positions. Decode only ever sees text
    tokens, so it has no DeepStack input.
"""

from litert_torch.generative.export_hf.core.external_emb import exportable_module as external_emb_exportable
import torch


def _text_model_and_head(module):
  full = module.source_model_artifacts.model
  return full.model.language_model, full.lm_head


def _num_deepstack(module) -> int:
  full = module.source_model_artifacts.model
  return len(full.config.vision_config.deepstack_visual_indexes)


def _position_ids(input_pos, mrope_pos):
  # HF Qwen3-VL takes [4, B, T]: row 0 is the text position (mask creation
  # only), rows 1..3 are the t/h/w M-RoPE positions.
  return torch.cat(
      [input_pos.view(1, 1, -1), mrope_pos.unsqueeze(1)], dim=0
  )


class LiteRTExportableModuleForQwen3VLPrefill(
    external_emb_exportable.LiteRTExportableModuleForDecoderOnlyLMPrefillExternalEmbedder
):
  """Prefill with external embeddings, M-RoPE positions and DeepStack."""

  # pylint: disable=arguments-renamed
  def forward(  # pyrefly: ignore[bad-override]
      self,
      embeddings,
      deepstack_embeddings,
      mrope_pos,
      input_pos,
      kv_cache,
      mask,
  ):
    inputs = self.adapt_inputs(
        None,
        embeddings,
        input_pos,
        kv_cache,
        mask,
        use_bool_mask=self.export_config.extra_kwargs.get(
            'use_bool_mask', False
        ),
    )
    inputs['position_ids'] = _position_ids(input_pos, mrope_pos)
    inputs['visual_pos_masks'] = None
    inputs['deepstack_visual_embeds'] = list(deepstack_embeddings.unbind(2))
    inputs |= self.attention_kwargs()
    text_model, _ = _text_model_and_head(self)
    output = text_model(**inputs)
    return {'kv_cache': output.past_key_values}

  def _get_input(
      self, batch_size, prefill_length, prefill_length_dim, model_config
  ):
    inputs, dynamic_shapes = super()._get_input(
        batch_size, prefill_length, prefill_length_dim, model_config
    )
    inputs['deepstack_embeddings'] = torch.zeros(
        (
            batch_size,
            prefill_length,
            _num_deepstack(self),
            model_config.hidden_size,
        ),
        dtype=torch.float32,
    )
    inputs['mrope_pos'] = torch.zeros((3, prefill_length), dtype=torch.int32)
    if prefill_length_dim:
      dynamic_shapes['deepstack_embeddings'] = {1: prefill_length_dim}
      dynamic_shapes['mrope_pos'] = {1: prefill_length_dim}
    return inputs, dynamic_shapes


class LiteRTExportableModuleForQwen3VLGenerate(
    external_emb_exportable.LiteRTExportableModuleForDecoderOnlyLMGenerateExternalEmbedder
):
  """Decode with external embeddings and M-RoPE positions."""

  # pylint: disable=arguments-renamed
  def forward(  # pyrefly: ignore[bad-override]
      self,
      embeddings,
      mrope_pos,
      input_pos,
      kv_cache,
      mask,
  ):
    inputs = self.adapt_inputs(
        None,
        embeddings,
        input_pos,
        kv_cache,
        mask,
        use_bool_mask=self.export_config.extra_kwargs.get(
            'use_bool_mask', False
        ),
    )
    inputs['position_ids'] = _position_ids(input_pos, mrope_pos)
    inputs |= self.attention_kwargs()
    text_model, lm_head = _text_model_and_head(self)
    output = text_model(**inputs)
    logits = lm_head(output.last_hidden_state)
    return {'kv_cache': output.past_key_values, 'logits': logits}

  def _get_input(
      self, batch_size, decode_length, decode_length_dim, model_config
  ):
    inputs, dynamic_shapes = super()._get_input(
        batch_size, decode_length, decode_length_dim, model_config
    )
    inputs['mrope_pos'] = torch.zeros((3, decode_length), dtype=torch.int32)
    if decode_length_dim:
      dynamic_shapes['mrope_pos'] = None
    return inputs, dynamic_shapes

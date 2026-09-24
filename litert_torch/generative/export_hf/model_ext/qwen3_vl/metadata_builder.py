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
"""Metadata builder for Qwen3-VL.

Qwen3-VL runs as a GenericModel with image input. The HF chat template renders
an image as `<|vision_start|><|image_pad|><|vision_end|>`; the runtime replaces
that placeholder with the vision tokens, keeping the start/end markers.
"""

import re

from litert_torch.generative.export_hf.core import export_lib
from litert_torch.generative.export_hf.core import exportable_module
from litert_torch.generative.export_hf.model_ext.qwen3_vl import vision_exportable

from litert_lm_builder.runtime.proto import llm_metadata_pb2
from litert_lm_builder.runtime.proto import llm_model_type_pb2

_VISION_START = '<|vision_start|>'
_IMAGE_PAD = '<|image_pad|>'
_VISION_END = '<|vision_end|>'
_IM_END = '<|im_end|>'


def build_llm_metadata(
    source_model_artifacts: export_lib.SourceModelArtifacts,
    export_config: exportable_module.ExportableModuleConfig,
    exported_model_artifacts: export_lib.ExportedModelArtifacts,
    llm_metadata: llm_metadata_pb2.LlmMetadata,
) -> llm_metadata_pb2.LlmMetadata:
  """Builds LLM metadata."""
  del source_model_artifacts  # Unused.
  if export_config.task != 'image_text_to_text':
    return llm_metadata
  if not exported_model_artifacts.vision_encoder_model_path:
    return llm_metadata
  sizes = vision_exportable.parse_image_sizes(
      export_config.extra_kwargs.get(
          'qwen3_vl_image_sizes', vision_exportable._DEFAULT_IMAGE_SIZES  # pylint: disable=protected-access
      )
  )
  # The runtime resizes each image to the candidate size that fits it best and
  # runs the matching `vision_{H}x{W}` encoder signature. The first size is
  # also the default target for runtimes without candidate size support.
  height, width = sizes[0]

  placeholder = re.escape(_VISION_START + _IMAGE_PAD + _VISION_END)
  generic = llm_model_type_pb2.GenericModel()
  generic.image_enabled = True
  generic.delimiter_regex = f'({placeholder})'
  generic.image_token_regex = placeholder
  generic.image_prefix = _VISION_START
  generic.image_suffix = _VISION_END
  generic.image_tensor_height = height
  generic.image_tensor_width = width
  if len(sizes) > 1:
    for size_height, size_width in sizes:
      generic.image_tensor_sizes.add(height=size_height, width=size_width)
  llm_metadata.llm_model_type.CopyFrom(
      llm_model_type_pb2.LlmModelType(generic_model=generic)
  )
  # The chat turn ends with <|im_end|>; generation_config only lists
  # <|endoftext|>, which the model emits a few tokens later.
  if not any(t.token_str == _IM_END for t in llm_metadata.stop_tokens):
    llm_metadata.stop_tokens.add(token_str=_IM_END)
  return llm_metadata

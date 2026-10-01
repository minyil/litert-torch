#!/bin/bash
# Exports a Qwen3-VL / Qwen3.5 (image + text) checkpoint to .litertlm.
#
# Usage: export_qwen3_vl.sh MODEL OUTPUT_DIR
#   MODEL       HF id or local dir, e.g. Qwen/Qwen3.5-4B, Qwen/Qwen3.5-2B,
#               Qwen/Qwen3-VL-4B-Instruct
# Environment overrides (defaults in brackets):
#   IMAGE_SIZES        [448,672,896,448x672,672x448] vision encoder sizes (HxW)
#   QUANT              [dynamic_wi8_afp32]           decoder/embedder recipe
#   VISION_QUANT       [weight_only_wi8_afp32]       vision encoder recipe
#   PREFILL_LENGTHS    [[256]]
#   CACHE_LENGTH       [4096]
#
# Needs ~50 GB of RAM for a 4B model (fp32 weights during conversion).
set -euo pipefail

MODEL="${1:?usage: $0 MODEL OUTPUT_DIR}"
OUT="${2:?usage: $0 MODEL OUTPUT_DIR}"
mkdir -p "${OUT}"

python3 -m litert_torch.generative.export_hf \
  --model="${MODEL}" \
  --output_dir="${OUT}" \
  --task=image_text_to_text \
  --prefill_lengths="${PREFILL_LENGTHS:-[256]}" \
  --cache_length="${CACHE_LENGTH:-4096}" \
  --quantization_recipe="${QUANT:-dynamic_wi8_afp32}" \
  --vision_encoder_quantization_recipe="${VISION_QUANT:-weight_only_wi8_afp32}" \
  --experimental_lightweight_conversion=True \
  --qwen3_vl_image_sizes="${IMAGE_SIZES:-448,672,896,448x672,672x448}"

ls -la "${OUT}/model.litertlm"

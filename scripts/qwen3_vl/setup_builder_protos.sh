#!/bin/bash
# Makes the installed litert-lm-builder understand the LiteRT-LM fork's protos
# (e.g. GenericModel.image_tensor_sizes), without building its wheel with Bazel.
#
# Usage: setup_builder_protos.sh [LITERT_LM_REPO] [BRANCH]
#   defaults: minyil/LiteRT-LM qwen3-vl
set -euo pipefail

REPO="${1:-minyil/LiteRT-LM}"
BRANCH="${2:-qwen3-vl}"
WORK="$(mktemp -d)"
trap 'rm -rf "${WORK}"' EXIT

pip install -q litert-lm-builder grpcio-tools "protobuf>=6.31"

mkdir -p "${WORK}/src/runtime/proto" "${WORK}/out"
names=$(curl -fsSL "https://api.github.com/repos/${REPO}/contents/runtime/proto?ref=${BRANCH}" |
  python3 -c "import sys, json; print(' '.join(x['name'] for x in json.load(sys.stdin) if x['name'].endswith('.proto')))")
for name in ${names}; do
  curl -fsSL -o "${WORK}/src/runtime/proto/${name}" \
    "https://raw.githubusercontent.com/${REPO}/${BRANCH}/runtime/proto/${name}"
done

(cd "${WORK}/src" && python3 -m grpc_tools.protoc -I. --python_out="${WORK}/out" runtime/proto/*.proto)

PKG="$(python3 -c "import importlib.util, os; print(os.path.dirname(importlib.util.find_spec('litert_lm_builder').origin))")"
cp "${WORK}"/out/runtime/proto/*_pb2.py "${PKG}/runtime/proto/"
# protoc emits absolute "runtime.proto" imports; the package nests them.
sed -i 's/^from runtime\.proto import/from litert_lm_builder.runtime.proto import/' "${PKG}"/runtime/proto/*_pb2.py

python3 -c "
from litert_lm_builder.runtime.proto import llm_model_type_pb2 as p
p.GenericModel().image_tensor_sizes.add(height=1, width=2)
print('litert-lm-builder protos updated from ${REPO}@${BRANCH}')"

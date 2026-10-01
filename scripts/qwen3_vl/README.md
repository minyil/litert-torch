# Qwen3.5 / Qwen3-VL → `.litertlm`

把 Qwen3.5（2B / 4B，圖文）與 Qwen3-VL 轉成 LiteRT-LM 的 `.litertlm`，並用錶頭照片評估讀數準確度。

需要兩個 fork 的 `qwen3-vl` 分支：

| repo | 修改內容 |
|---|---|
| [minyil/litert-torch](https://github.com/minyil/litert-torch/tree/qwen3-vl)（本 repo） | 匯出器：視覺編碼器（多種固定尺寸）、M-RoPE、DeepStack（Qwen3-VL）、Qwen3.5 多模態 decoder、metadata |
| [minyil/LiteRT-LM](https://github.com/minyil/LiteRT-LM/tree/qwen3-vl) | runtime：M-RoPE 位置、DeepStack 注入、依影像長寬比挑選編碼器尺寸、CLI `--enable_thinking` |

原版 LiteRT-LM 無法執行這裡匯出的模型（缺 `mrope_pos` 等輸入的處理），必須用上面 fork 編出的 runtime。

## 1. 環境（建議 Colab L4 執行階段：12 核、52 GB RAM）

```bash
git clone -b qwen3-vl https://github.com/minyil/litert-torch.git
pip install -e litert-torch
pip install -U "typing-extensions>=4.15" "protobuf>=6.31" "transformers==5.17.0"
bash litert-torch/scripts/qwen3_vl/setup_builder_protos.sh   # 讓 litert-lm-builder 認得 fork 的 proto 欄位
```

## 2. 轉檔

```bash
bash litert-torch/scripts/qwen3_vl/export_qwen3_vl.sh Qwen/Qwen3.5-4B export_q35_4b   # 約 15 分鐘
bash litert-torch/scripts/qwen3_vl/export_qwen3_vl.sh Qwen/Qwen3.5-2B export_q35_2b   # 約 10 分鐘
```

預設：decoder/embedder `dynamic_wi8_afp32`、視覺編碼器 `weight_only_wi8_afp32`、影像尺寸 `448,672,896,448x672,672x448`、prefill 256、context 4096。可用環境變數覆寫，見腳本開頭說明。

4B 需約 50 GB RAM；不加 `--experimental_lightweight_conversion`（腳本已加）會 OOM。

## 3. 編譯 runtime CLI

```bash
apt-get install -y clang lld git-lfs
git clone --depth 1 -b qwen3-vl https://github.com/minyil/LiteRT-LM.git && cd LiteRT-LM
git lfs install --local && git lfs pull --include="prebuilt/linux_x86_64/*"
CC=clang CXX=clang++ bazelisk build -c opt //runtime/engine:litert_lm_advanced_main
```

要用 `litert_lm_advanced_main`；`litert_lm_main` 會忽略圖片。單獨拷貝執行檔時，同目錄需要 `prebuilt/linux_x86_64/libGemmaModelConstraintProvider.so`（以 `LD_LIBRARY_PATH` 指定）。

```bash
bazel-bin/runtime/engine/litert_lm_advanced_main --backend=cpu --vision_backend=cpu \
  --model_path=model.litertlm --enable_thinking=false --max_num_images=1 \
  --input_prompt="[image:/path/to/photo.jpg] 讀出錶頭數字"
```

## 4. 評估錶頭讀數

```bash
python3 litert-torch/scripts/qwen3_vl/eval_meter.py \
  --model export_q35_4b/model.litertlm \
  --cli LiteRT-LM/bazel-bin/runtime/engine/litert_lm_advanced_main \
  --data-root tainan_eval            # 含 results.csv、台南廠提示詞_最終.csv 與照片
```

篩選：剪裁、`failure_reasons` 空白、`--filter`（預設無濾鏡）。逐張輸出存成 `eval_raw_<tag>.jsonl`，中斷後重跑會接續。

## 結果（台南廠 2026-09-16，剪裁＋無 failure_reasons＋無濾鏡，78 張，Colab CPU）

| 模型 | 成功率 | 大小 | 每張（含載入模型） |
|---|---|---|---|
| Qwen3.5-4B（int8） | 87.2%（68/78） | 5.27 GB | 39 秒 |
| Qwen3.5-2B（int8） | 80.8%（63/78） | 2.8 GB | 21 秒 |
| Ollama qwen3-vl:4b（對照，先前地端紀錄） | 80.8% | — | — |

取樣為 greedy（runtime 預設 top_k=1；改 temperature=0 結果逐字相同）。

## 注意事項

- **思考模式**：runtime 預設開啟，讀數任務請加 `--enable_thinking=false`。
- **GPU**：Colab 容器沒有 Vulkan 圖形權限，LiteRT 的 GPU 後端（WebGPU/Vulkan）在 Colab 上無法使用；要在有 GPU 驅動的實機或手機上測。
- **padding**：Qwen3.5 的 GatedDeltaNet 依「`input_pos` 遞增」判斷 padding，prefill 中未使用的位置 `input_pos` 必須為 0（runtime 已如此處理；自行驗證 TFLite 時要注意）。
- **驗證**：匯出器在 PyTorch 層級與 HF 原版比對過（視覺特徵、M-RoPE 位置、decoder logits，cos≈1.0）。

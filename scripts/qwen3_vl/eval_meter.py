#!/usr/bin/env python3
"""Meter-reading accuracy of a .litertlm model via the LiteRT-LM CLI.

Photos are selected from results.csv (crop == 剪裁, empty failure_reasons,
filter == --filter), prompted per meter type from the prompts CSV, and scored
as in 錶頭測試/run_tainan_eval_qwen.py (numeric equality after removing
units/leading zeros; any "?" counts as wrong). Raw outputs go to a JSONL file,
so an interrupted run resumes where it stopped.

Example:
  python3 eval_meter.py --model qwen3.5-4b.litertlm --data-root tainan_eval \\
      --cli LiteRT-LM/bazel-bin/runtime/engine/litert_lm_advanced_main
"""

import argparse
import csv
import json
import os
import re
import subprocess
import time


def answer_text(stdout):
  """The CLI prints its config (ending with "patch_num_shrink_factor: N"),
  then the answer, then BenchmarkInfo."""
  if "patch_num_shrink_factor:" in stdout:
    stdout = stdout.split("patch_num_shrink_factor:", 1)[1].split("\n", 1)[-1]
  return stdout.split("BenchmarkInfo:")[0].strip()


def parse_reading(text):
  """The "reading" of the first JSON object, else the first number."""
  m = re.search(r"\{.*?\}", text, re.S)
  if m:
    try:
      obj = json.loads(m.group(0))
      r = obj.get("reading")
      return (None if r is None else str(r).strip()), obj.get("status")
    except json.JSONDecodeError:
      pass
  m = re.search(r"[\d?]+(?:\.[\d?]+)?", text)
  return (m.group(0) if m else None), None


def clean(v):
  if v is None:
    return None
  s = re.sub(r"[^\d.?]", "", str(v))
  return s or None


def same_value(pred, gt):
  p, g = clean(pred), clean(gt)
  if not p or not g or "?" in p:
    return False
  try:
    return float(p) == float(g)
  except ValueError:
    return p.lstrip("0") == g.lstrip("0")


def main():
  ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
  ap.add_argument("--model", required=True, help=".litertlm file")
  ap.add_argument("--cli", required=True, help="litert_lm_advanced_main binary")
  ap.add_argument("--data-root", required=True,
                  help="folder with results.csv, the prompts CSV and the photos")
  ap.add_argument("--prompts", default="台南廠提示詞_最終.csv")
  ap.add_argument("--filter", default="無濾鏡", help="photo filter to evaluate")
  ap.add_argument("--tag", default=None, help="name for the raw JSONL output")
  ap.add_argument("--backend", default="cpu")
  ap.add_argument("--lib-dir", default=None,
                  help="LD_LIBRARY_PATH for the CLI's shared libraries")
  ap.add_argument("--max-output-tokens", type=int, default=160)
  args = ap.parse_args()

  tag = args.tag or os.path.splitext(os.path.basename(args.model))[0]
  raw_path = f"eval_raw_{tag}.jsonl"
  env = dict(os.environ)
  if args.lib_dir:
    env["LD_LIBRARY_PATH"] = args.lib_dir

  root = args.data_root
  prompts = {
      r["meter_type"].strip(): r["hint"]
      for r in csv.DictReader(
          open(os.path.join(root, args.prompts), encoding="utf-8-sig"))
  }
  rows = list(csv.DictReader(
      open(os.path.join(root, "results.csv"), encoding="utf-8-sig")))
  sel = [r for r in rows
         if r["crop"] == "剪裁"
         and not (r["failure_reasons"] or "").strip()
         and r["filter"] == args.filter]
  missing = sorted({r["device"] for r in sel} - set(prompts))
  if missing:
    raise SystemExit(f"提示詞檔缺少這些錶頭類型: {missing}")

  done = {}
  if os.path.exists(raw_path):
    for line in open(raw_path, encoding="utf-8"):
      d = json.loads(line)
      done[d["image_path"]] = d
  todo = [r for r in sel if r["image_path"] not in done]
  print(f"selected {len(sel)}, done {len(done)}, todo {len(todo)}", flush=True)

  with open(raw_path, "a", encoding="utf-8") as f:
    for i, r in enumerate(todo, 1):
      img = os.path.join(root, r["image_path"])
      t0 = time.time()
      p = subprocess.run(
          [args.cli, f"--backend={args.backend}",
           f"--vision_backend={args.backend}", f"--model_path={args.model}",
           "--max_num_images=1",
           f"--max_output_tokens={args.max_output_tokens}",
           "--enable_thinking=false",
           f"--input_prompt=[image:{img}] {prompts[r['device']]}"],
          capture_output=True, text=True, timeout=900, env=env)
      text = answer_text(p.stdout)
      reading, status = parse_reading(text)
      d = {
          "image_path": r["image_path"], "device": r["device"],
          "raw": text[:600], "reading": reading, "model_status": status,
          "ground_truth": r["ground_truth"],
          "ms": int((time.time() - t0) * 1000),
          "error": "" if p.returncode == 0 else (p.stderr or p.stdout)[-400:],
      }
      done[r["image_path"]] = d
      f.write(json.dumps(d, ensure_ascii=False) + "\n")
      f.flush()
      ok = same_value(reading, r["ground_truth"])
      print(f"[{i}/{len(todo)}] {r['device']:9s} gt={r['ground_truth']:>14s} "
            f"pred={reading}  {'✓' if ok else '✗'}  {d['ms'] / 1000:.0f}s",
            flush=True)

  per_device = {}
  ok_n = 0
  for r in sel:
    ok = same_value(done[r["image_path"]]["reading"], r["ground_truth"])
    ok_n += ok
    a, b = per_device.get(r["device"], (0, 0))
    per_device[r["device"]] = (a + ok, b + 1)
  print(f"\n模型 {os.path.basename(args.model)} ({tag})")
  print(f"辨識成功：{ok_n}/{len(sel)} ({ok_n / len(sel) * 100:.1f}%)")
  for dev, (a, b) in sorted(per_device.items(), key=lambda x: -x[1][1]):
    print(f"  {dev:10s} {a:3d}/{b:3d} ({a / b * 100:5.1f}%)")


if __name__ == "__main__":
  main()

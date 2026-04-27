#!/usr/bin/env bash
set -euo pipefail

# 学習前にJSONLを作り直し、対応するtranscript jsonが揃っているか確認する
UV_CACHE_DIR="${UV_CACHE_DIR:-.uv-cache}" \
uv run --project moshi-finetune python scripts/prepareDatasetJsonl.py \
  --audio-dir datasets/stereo \
  --output datasets/train.jsonl \
  --require-transcript

# 既存のLoRA出力がある場合は消さずにタイムスタンプ付きで退避する
run_dir="${RUN_DIR:-loras/llmJpMoshiV1}"
if [ -e "$run_dir" ]; then
  archived_run_dir="${run_dir}.previous.$(date +%Y%m%d-%H%M%S)"
  echo "Run dir already exists: $run_dir"
  echo "Moving it to: $archived_run_dir"
  mv "$run_dir" "$archived_run_dir"
fi

# llm-jp/llm-jp-moshi-v1をLoRAで学習する
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}" \
NO_TORCH_COMPILE="${NO_TORCH_COMPILE:-1}" \
HF_HOME="$PWD/models/huggingface" \
uv run --project moshi-finetune torchrun \
  --nproc-per-node 1 \
  moshi-finetune/train.py \
  config/llmJpMoshiLora.yaml

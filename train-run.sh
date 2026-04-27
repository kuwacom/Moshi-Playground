#!/usr/bin/env bash
set -euo pipefail

# 学習前にJSONLを作り直し、対応するtranscript jsonが揃っているか確認する
UV_CACHE_DIR="${UV_CACHE_DIR:-.uv-cache}" \
uv run --project moshi-finetune python scripts/prepareDatasetJsonl.py \
  --audio-dir datasets/stereo \
  --output datasets/train.jsonl \
  --require-transcript

# 既存の出力先がある場合は消さずにタイムスタンプ付きで退避する
run_dir="${RUN_DIR:-loras/llmJpMoshiV1}"
if [ -e "$run_dir" ]; then
  archived_run_dir="${run_dir}.previous.$(date +%Y%m%d-%H%M%S)"
  echo "Run dir already exists: $run_dir"
  echo "Moving it to: $archived_run_dir"
  mv "$run_dir" "$archived_run_dir"
fi

gpu_ids="${CUDA_VISIBLE_DEVICES:-0}"
IFS=',' read -ra gpu_id_list <<< "$gpu_ids"
nproc_per_node="${NPROC_PER_NODE:-${#gpu_id_list[@]}}"
master_port="${MASTER_PORT:-29501}"
config_path="${CONFIG_PATH:-config/llmJpMoshiLora.yaml}"

echo "Training configuration"
echo "  CUDA_VISIBLE_DEVICES:  $gpu_ids"
echo "  nproc_per_node:       $nproc_per_node"
echo "  master_port:          $master_port"
echo "  run_dir:              $run_dir"
echo "  config_path:          $config_path"

# llm-jp/llm-jp-moshi-v1をLoRAで学習する
CUDA_VISIBLE_DEVICES="$gpu_ids" \
NO_TORCH_COMPILE="${NO_TORCH_COMPILE:-1}" \
HF_HOME="$PWD/models/huggingface" \
uv run --project moshi-finetune torchrun \
  --nproc-per-node "$nproc_per_node" \
  --master_port "$master_port" \
  moshi-finetune/train.py \
  "$config_path"

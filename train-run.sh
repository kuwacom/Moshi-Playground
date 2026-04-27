#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"
source scripts/env/loadLoraEnv.sh

# 学習前にJSONLを作り直し、対応するtranscript jsonが揃っているか確認する
UV_CACHE_DIR="${UV_CACHE_DIR:-.uv-cache}" \
uv run --project moshi-finetune python -m scripts.dataset.prepareDatasetJsonl \
  --audio-dir "$DATASET_STEREO_DIR" \
  --output "$TRAIN_JSONL" \
  --require-transcript

if [ "${CONFIG_PATH:-}" != "" ]; then
  config_path="$CONFIG_PATH"
  should_render_config="${RENDER_TRAIN_CONFIG:-0}"
else
  config_path="$TRAIN_CONFIG_PATH"
  should_render_config="${RENDER_TRAIN_CONFIG:-1}"
fi

if [ "$should_render_config" = "1" ]; then
  UV_CACHE_DIR="${UV_CACHE_DIR:-.uv-cache}" \
  uv run --project moshi-finetune python -m scripts.train.renderTrainConfig \
    --template "$TRAIN_CONFIG_TEMPLATE_PATH" \
    --output "$config_path" \
    --train-data "$TRAIN_JSONL" \
    --run-dir "$RUN_DIR"
fi

# 既存の出力先がある場合は消さずにタイムスタンプ付きで退避する
run_dir="$RUN_DIR"
if [ -e "$run_dir" ]; then
  archived_run_dir="${run_dir}.previous.$(date +%Y%m%d-%H%M%S)"
  echo "Run dir already exists: $run_dir"
  echo "Moving it to: $archived_run_dir"
  mv "$run_dir" "$archived_run_dir"
fi

gpu_ids="${CUDA_VISIBLE_DEVICES:-0}"
IFS=',' read -ra gpu_id_list <<< "$gpu_ids"
gpu_count="${#gpu_id_list[@]}"
nproc_per_node="${NPROC_PER_NODE:-auto}"
if [ "$nproc_per_node" = "auto" ] || [ "$nproc_per_node" = "" ]; then
  nproc_per_node="$gpu_count"
fi
if ! [[ "$nproc_per_node" =~ ^[0-9]+$ ]] || [ "$nproc_per_node" -lt 1 ]; then
  echo "NPROC_PER_NODE must be a positive integer or auto: $nproc_per_node" >&2
  exit 1
fi
master_port="${MASTER_PORT:-29501}"

echo "Training configuration"
echo "  LORA_NAME:           $LORA_NAME"
echo "  CUDA_VISIBLE_DEVICES:  $gpu_ids"
echo "  nproc_per_node:       $nproc_per_node"
echo "  master_port:          $master_port"
echo "  train_jsonl:          $TRAIN_JSONL"
echo "  run_dir:              $run_dir"
echo "  config_path:          $config_path"
if [ "$gpu_count" -gt 1 ] && [ "$nproc_per_node" -eq 1 ]; then
  echo "Warning: multiple GPUs are visible, but nproc_per_node is 1. Only one training process will run."
fi

# llm-jp/llm-jp-moshi-v1をLoRAで学習する
CUDA_VISIBLE_DEVICES="$gpu_ids" \
NO_TORCH_COMPILE="${NO_TORCH_COMPILE:-1}" \
HF_HOME="$PWD/models/huggingface" \
uv run --project moshi-finetune torchrun \
  --nproc-per-node "$nproc_per_node" \
  --master_port "$master_port" \
  moshi-finetune/train.py \
  "$config_path"

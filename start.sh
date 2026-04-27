#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

run_dir="${RUN_DIR:-loras/llmJpMoshiV1}"
latest_dir="${LORA_LATEST_DIR:-$run_dir/latest}"
hf_repo="${HF_REPO:-llm-jp/llm-jp-moshi-v1}"

if [ "${SKIP_EXPORT_LATEST:-0}" != "1" ]; then
  UV_CACHE_DIR="${UV_CACHE_DIR:-.uv-cache}" \
    uv run --project moshi-finetune python scripts/exportLatestLora.py \
      --run-dir "$run_dir" \
      --output-dir "$latest_dir"
fi

lora_weight="${LORA_WEIGHT:-$latest_dir/lora.safetensors}"
config_path="${CONFIG_PATH:-$latest_dir/config.json}"

if [ ! -f "$lora_weight" ]; then
  echo "LoRA weight not found: $lora_weight" >&2
  echo "Run training first, or set LORA_WEIGHT to a checkpoint lora.safetensors." >&2
  exit 1
fi

if [ ! -f "$config_path" ]; then
  echo "Config not found: $config_path" >&2
  echo "Use the config.json from the same checkpoint as the LoRA weight." >&2
  exit 1
fi

echo "Starting Moshi server"
echo "  repo:   $hf_repo"
echo "  lora:   $lora_weight"
echo "  config: $config_path"
echo "  url:    http://localhost:8998"

CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}" \
HF_HOME="${HF_HOME:-$PWD/models/huggingface}" \
NO_TORCH_COMPILE="${NO_TORCH_COMPILE:-1}" \
UV_CACHE_DIR="${UV_CACHE_DIR:-.uv-cache}" \
  uv run --project moshi-finetune python -m moshi.server \
    --hf-repo "$hf_repo" \
    --lora-weight "$lora_weight" \
    --config-path "$config_path"

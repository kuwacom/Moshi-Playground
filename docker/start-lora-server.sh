#!/usr/bin/env bash
set -euo pipefail

lora_weight="${LORA_WEIGHT:-/opt/moshi/lora/lora.safetensors}"
config_path="${MOSHI_CONFIG_PATH:-${CONFIG_PATH:-/opt/moshi/lora/config.json}}"
hf_repo="${HF_REPO:-llm-jp/llm-jp-moshi-v1}"
host="${HOST:-0.0.0.0}"
port="${PORT:-8998}"

if [ ! -f "$lora_weight" ]; then
  echo "LoRA weight not found: $lora_weight" >&2
  exit 1
fi

if [ ! -f "$config_path" ]; then
  echo "Config not found: $config_path" >&2
  exit 1
fi

export HF_HOME="${HF_HOME:-/opt/moshi/models/huggingface}"
export NO_TORCH_COMPILE="${NO_TORCH_COMPILE:-1}"

echo "Starting Moshi LoRA server"
echo "  lora_name: ${LORA_NAME:-unknown}"
echo "  repo:      $hf_repo"
echo "  lora:      $lora_weight"
echo "  config:    $config_path"
echo "  hf_home:   $HF_HOME"
echo "  url:       http://$host:$port"

server_args=(--host "$host" --port "$port")
server_args+=("$@")

exec python -m moshi.server \
  --hf-repo "$hf_repo" \
  --lora-weight "$lora_weight" \
  --config-path "$config_path" \
  "${server_args[@]}"

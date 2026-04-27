#!/usr/bin/env bash

# 各bashからsourceしてLoRAごとの作業パスを揃える
repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
env_file="${ENV_FILE:-$repo_root/.env}"

if [ -f "$env_file" ]; then
  had_nounset=0
  case "$-" in
    *u*)
      had_nounset=1
      set +u
      ;;
  esac

  set -a
  # shellcheck source=/dev/null
  source "$env_file"
  set +a

  if [ "$had_nounset" = "1" ]; then
    set -u
  fi
fi

: "${LORA_NAME:=shigureui1}"

if [[ ! "$LORA_NAME" =~ ^[A-Za-z0-9._-]+$ ]]; then
  echo "LORA_NAME may only contain letters, numbers, dot, underscore, and hyphen: $LORA_NAME" >&2
  return 1 2>/dev/null || exit 1
fi

: "${DATASET_ROOT:=datasets}"
: "${DATASET_RAW_DIR:=$DATASET_ROOT/raw/$LORA_NAME}"
: "${DATASET_STEREO_DIR:=$DATASET_ROOT/stereo/$LORA_NAME}"
: "${DATASET_CACHE_DIR:=$DATASET_ROOT/cache/$LORA_NAME}"
: "${DATASET_TTS_DIR:=$DATASET_ROOT/tts/$LORA_NAME}"
: "${TRAIN_JSONL:=$DATASET_ROOT/$LORA_NAME.jsonl}"
: "${RUN_DIR:=loras/$LORA_NAME}"
: "${LORA_LATEST_DIR:=$RUN_DIR/latest}"
: "${TRAIN_CONFIG_TEMPLATE_PATH:=config/llmJpMoshiLora.example.yaml}"
: "${TRAIN_CONFIG_PATH:=config/$LORA_NAME.yaml}"
: "${HF_REPO:=llm-jp/llm-jp-moshi-v1}"

export ENV_FILE="$env_file"
export LORA_NAME
export DATASET_ROOT
export DATASET_RAW_DIR
export DATASET_STEREO_DIR
export DATASET_CACHE_DIR
export DATASET_TTS_DIR
export TRAIN_JSONL
export RUN_DIR
export LORA_LATEST_DIR
export TRAIN_CONFIG_TEMPLATE_PATH
export TRAIN_CONFIG_PATH
export HF_REPO

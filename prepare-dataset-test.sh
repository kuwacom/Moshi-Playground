#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"
source scripts/loadLoraEnv.sh

mkdir -p "$DATASET_RAW_DIR" "$DATASET_STEREO_DIR" "$DATASET_CACHE_DIR"

uv run --project moshi-finetune python scripts/processRawToStereo.py \
  --input-dir "$DATASET_RAW_DIR" \
  --output-dir "$DATASET_STEREO_DIR" \
  --cache-dir "$DATASET_CACHE_DIR" \
  --env-file "$ENV_FILE" \
  --limit 1 \
  --max-segments 60 \
  --max-response-chars 40 \
  --response-delay-sec 0.5 \
  --tts-speed 1.05 \
  --semantic-gap-insert-mode auto \
  --semantic-gap-max-count 2

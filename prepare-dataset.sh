#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"
source scripts/env/loadLoraEnv.sh

mkdir -p "$DATASET_RAW_DIR" "$DATASET_STEREO_DIR" "$DATASET_CACHE_DIR"

echo "Dataset configuration"
echo "  LORA_NAME:       $LORA_NAME"
echo "  raw_dir:         $DATASET_RAW_DIR"
echo "  stereo_dir:      $DATASET_STEREO_DIR"
echo "  cache_dir:       $DATASET_CACHE_DIR"
echo "  train_jsonl:     $TRAIN_JSONL"

# raw音声を順番に処理してMoshi学習用のstereo音声を作る
uv run --project moshi-finetune python -m scripts.dataset.processRawToStereo \
  --input-dir "$DATASET_RAW_DIR" \
  --output-dir "$DATASET_STEREO_DIR" \
  --cache-dir "$DATASET_CACHE_DIR" \
  --env-file "$ENV_FILE" \
  --response-delay-sec 0.5 \
  --tts-speed 1.05

# stereo音声一覧から学習用JSONLを作る
uv run --project moshi-finetune python -m scripts.dataset.prepareDatasetJsonl \
  --audio-dir "$DATASET_STEREO_DIR" \
  --output "$TRAIN_JSONL"

# Moshiが読む単語タイムスタンプ付きアノテーションJSONを作る
uv run --project moshi-finetune python -m scripts.dataset.annotateDataset \
  "$TRAIN_JSONL" \
  --lang ja \
  --whisper-model large-v3 \
  --whisper-cache-dir models \
  --overwrite-existing

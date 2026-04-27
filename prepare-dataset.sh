#!/usr/bin/env bash
set -euo pipefail

# raw音声を順番に処理してMoshi学習用のstereo音声を作る
uv run --project moshi-finetune python scripts/processRawToStereo.py \
  --response-delay-sec 0.5 \
  --tts-speed 1.05

# stereo音声一覧から学習用JSONLを作る
uv run --project moshi-finetune python scripts/prepareDatasetJsonl.py \
  --audio-dir datasets/stereo \
  --output datasets/train.jsonl

# Moshiが読む単語タイムスタンプ付きアノテーションJSONを作る
uv run --project moshi-finetune python scripts/annotateDataset.py \
  datasets/train.jsonl \
  --lang ja \
  --whisper-model large-v3 \
  --whisper-cache-dir models/whisper \
  --overwrite-existing

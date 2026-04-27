uv run --project moshi-finetune python scripts/processRawToStereo.py \
  --response-delay-sec 0.5 \
  --tts-speed 1.05

uv run --project moshi-finetune python scripts/prepareDatasetJsonl.py \
  --audio-dir datasets/stereo \
  --output datasets/train.jsonl
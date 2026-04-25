uv run --project moshi-finetune python scripts/processRawToStereo.py \
  --limit 2 \
  --max-segments 60 \
  --max-response-chars 40 \
  --response-delay-sec 0.5 \
  --tts-speed 1.05
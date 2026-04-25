#!/usr/bin/env bash
set -euo pipefail

export UV_LINK_MODE="${UV_LINK_MODE:-copy}"

if [ ! -x "moshi-finetune/.venv/bin/python" ]; then
  uv venv --project moshi-finetune
fi

uv sync --project moshi-finetune

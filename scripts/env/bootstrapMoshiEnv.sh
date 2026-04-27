#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$repo_root"

export UV_LINK_MODE="${UV_LINK_MODE:-copy}"

if [ ! -x "moshi-finetune/.venv/bin/python" ]; then
  uv venv --project moshi-finetune
fi

uv sync --project moshi-finetune

from __future__ import annotations

import argparse
from pathlib import Path

from huggingface_hub import snapshot_download

from progressUtils import console, status


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download a Hugging Face model snapshot under models"
    )
    parser.add_argument("--repo-id", default="llm-jp/llm-jp-moshi-v1")
    parser.add_argument("--output", type=Path, default=Path("models/llm-jp-moshi-v1"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    with status(f"Downloading model [bold]{args.repo_id}[/bold] to {args.output}"):
        local_dir = snapshot_download(repo_id=args.repo_id, local_dir=args.output)
    console.print(f"[green]Downloaded[/green] {args.repo_id} to {local_dir}")


if __name__ == "__main__":
    main()

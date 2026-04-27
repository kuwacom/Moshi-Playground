from __future__ import annotations

import argparse
from pathlib import Path

from huggingface_hub import snapshot_download

from progressUtils import console, status


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download a Hugging Face model snapshot into the shared HF cache"
    )
    parser.add_argument("--repo-id", default="llm-jp/llm-jp-moshi-v1")
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=Path("models/huggingface/hub"),
        help="Hugging Face Hub cache directory. Matches HF_HOME=models/huggingface.",
    )
    parser.add_argument(
        "--local-dir",
        type=Path,
        help="Optional full local copy. Usually unnecessary and uses extra disk.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.cache_dir.mkdir(parents=True, exist_ok=True)
    with status(
        f"Downloading model [bold]{args.repo_id}[/bold] to HF cache {args.cache_dir}"
    ):
        local_dir = snapshot_download(
            repo_id=args.repo_id,
            cache_dir=args.cache_dir,
            local_dir=args.local_dir,
        )
    console.print(f"[green]Downloaded[/green] {args.repo_id} snapshot at {local_dir}")


if __name__ == "__main__":
    main()

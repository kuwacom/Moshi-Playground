from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

from progressUtils import console, status


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Annotate Moshi dataset audio with local Whisper large-v3"
    )
    parser.add_argument(
        "jsonl",
        type=Path,
        nargs="?",
        default=Path("datasets/train.jsonl"),
    )
    parser.add_argument("--lang", default="ja")
    parser.add_argument("--whisper-model", default="large-v3")
    parser.add_argument("--cache-root", type=Path, default=Path("models"))
    parser.add_argument("--rerun-errors", action="store_true")
    parser.add_argument("--verbose", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    command = [
        sys.executable,
        "moshi-finetune/annotate.py",
        str(args.jsonl),
        "--local",
        "--lang",
        args.lang,
        "--whisper_model",
        args.whisper_model,
    ]
    if args.rerun_errors:
        command.append("--rerun_errors")
    if args.verbose:
        command.append("--verbose")

    env = os.environ.copy()
    env["XDG_CACHE_HOME"] = str(args.cache_root.resolve())
    with status(
        f"Annotating [bold]{args.jsonl}[/bold] with Whisper {args.whisper_model}"
    ):
        subprocess.run(command, check=True, env=env)
    console.print(f"[green]Annotated[/green] dataset listed in {args.jsonl}")


if __name__ == "__main__":
    main()

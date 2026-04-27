from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

from scripts.common.datasetPaths import trainJsonlPath
from scripts.common.progressUtils import console, status


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Annotate Moshi dataset audio with local Whisper large-v3"
    )
    parser.add_argument(
        "jsonl",
        type=Path,
        nargs="?",
        default=trainJsonlPath(),
    )
    parser.add_argument("--lang", default="ja")
    parser.add_argument("--whisper-model", default="large-v3")
    parser.add_argument("--whisper-cache-dir", type=Path, default=Path("models"))
    parser.add_argument("--overwrite-existing", action="store_true")
    parser.add_argument("--rerun-errors", action="store_true")
    parser.add_argument("--verbose", action="store_true")
    return parser.parse_args()


def remove_existing_annotations(jsonl_path: Path) -> int:
    removed = 0
    for line in jsonl_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        data = json.loads(line)
        audio_path = Path(str(data["path"]))
        if not audio_path.is_absolute():
            audio_path = jsonl_path.parent / audio_path
        annotation_path = audio_path.with_suffix(".json")
        error_path = audio_path.with_suffix(".json.err")
        for path in (annotation_path, error_path):
            if path.exists():
                path.unlink()
                removed += 1
    return removed


def main() -> None:
    args = parse_args()
    if args.overwrite_existing:
        removed = remove_existing_annotations(args.jsonl)
        console.print(f"[yellow]Removed[/yellow] {removed} existing annotation files")

    command = [
        sys.executable,
        "moshi-finetune/annotate.py",
        str(args.jsonl),
        "--local",
        "--lang",
        args.lang,
        "--whisper_model",
        args.whisper_model,
        "--whisper_download_root",
        str(args.whisper_cache_dir.resolve()),
    ]
    if args.rerun_errors:
        command.append("--rerun_errors")
    if args.verbose:
        command.append("--verbose")

    env = os.environ.copy()
    with status(
        f"Annotating [bold]{args.jsonl}[/bold] with Whisper {args.whisper_model}"
    ):
        subprocess.run(command, check=True, env=env)
    console.print(f"[green]Annotated[/green] dataset listed in {args.jsonl}")


if __name__ == "__main__":
    main()

from __future__ import annotations

import argparse
import json
from pathlib import Path

import sphn

from progressUtils import console, create_progress, status


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create a Moshi dataset JSONL from stereo wav files"
    )
    parser.add_argument("--audio-dir", type=Path, default=Path("datasets/stereo"))
    parser.add_argument("--output", type=Path, default=Path("datasets/train.jsonl"))
    parser.add_argument("--pattern", default="*.wav")
    parser.add_argument("--require-transcript", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    paths = sorted(args.audio_dir.glob(args.pattern))
    if not paths:
        raise FileNotFoundError(f"No wav files found in {args.audio_dir}")

    if args.require_transcript:
        missing = [
            path.with_suffix(".json")
            for path in paths
            if not path.with_suffix(".json").exists()
        ]
        if missing:
            missing_list = "\n".join(str(path) for path in missing[:20])
            raise FileNotFoundError(f"Missing transcript json files:\n{missing_list}")

    with status(f"Reading durations for [bold]{len(paths)}[/bold] audio files"):
        durations = sphn.durations([str(path) for path in paths])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    written = 0
    with args.output.open("w", encoding="utf-8") as output_file:
        with create_progress() as progress:
            task = progress.add_task("Writing dataset JSONL", total=len(paths))
            for path, duration in zip(paths, durations, strict=True):
                if duration is not None:
                    json.dump(
                        {"path": str(path), "duration": float(duration)},
                        output_file,
                        ensure_ascii=False,
                    )
                    output_file.write("\n")
                    written += 1
                progress.advance(task)

    console.print(f"[green]Wrote[/green] {written} items to {args.output}")


if __name__ == "__main__":
    main()

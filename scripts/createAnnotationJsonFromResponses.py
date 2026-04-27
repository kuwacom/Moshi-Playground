from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any

from datasetPaths import trainJsonlPath
from progressUtils import console, create_progress


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Create approximate Moshi transcript json files from existing "
            "*.responses.json metadata"
        )
    )
    parser.add_argument("jsonl", type=Path, nargs="?", default=trainJsonlPath())
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--chunk-chars", type=int, default=8)
    return parser.parse_args()


def load_dataset_paths(jsonl_path: Path) -> list[Path]:
    paths: list[Path] = []
    for line in jsonl_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        data = json.loads(line)
        path = Path(str(data["path"]))
        if not path.is_absolute():
            path = jsonl_path.parent / path
        paths.append(path)
    return paths


def split_text(text: str, chunk_chars: int) -> list[str]:
    normalized = re.sub(r"\s+", " ", text).strip()
    if not normalized:
        return []
    parts = [
        part.strip()
        for part in re.split(r"([。！？!?、,])", normalized)
        if part.strip()
    ]
    merged: list[str] = []
    buffer = ""
    for part in parts:
        if re.fullmatch(r"[。！？!?、,]", part):
            buffer += part
            continue
        if buffer:
            merged.append(buffer)
        buffer = part
        while len(buffer) > chunk_chars:
            merged.append(buffer[:chunk_chars])
            buffer = buffer[chunk_chars:]
    if buffer:
        merged.append(buffer)
    return [item for item in merged if item.strip()]


def segment_to_alignments(
    segment: dict[str, Any],
    chunk_chars: int,
) -> list[list[Any]]:
    text = str(segment.get("text", "")).strip()
    start = float(segment["start"])
    end = float(segment["end"])
    chunks = split_text(text, chunk_chars)
    if not chunks or end <= start:
        return []

    total_chars = sum(max(1, len(chunk)) for chunk in chunks)
    cursor = start
    alignments: list[list[Any]] = []
    for index, chunk in enumerate(chunks):
        if index == len(chunks) - 1:
            chunk_end = end
        else:
            ratio = max(1, len(chunk)) / total_chars
            chunk_end = min(end, cursor + (end - start) * ratio)
        if chunk_end > cursor:
            alignments.append([chunk, [cursor, chunk_end], "SPEAKER_MAIN"])
        cursor = chunk_end
    return alignments


def create_annotation(audio_path: Path, chunk_chars: int, overwrite: bool) -> bool:
    output_path = audio_path.with_suffix(".json")
    if output_path.exists() and not overwrite:
        return False

    metadata_path = audio_path.with_name(f"{audio_path.stem}.responses.json")
    if not metadata_path.exists():
        raise FileNotFoundError(f"Missing responses metadata: {metadata_path}")

    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    transcript = metadata.get("transcript")
    if not isinstance(transcript, list):
        raise RuntimeError(f"No transcript list in {metadata_path}")

    alignments: list[list[Any]] = []
    for segment in transcript:
        if isinstance(segment, dict):
            alignments.extend(segment_to_alignments(segment, chunk_chars))

    output_path.write_text(
        json.dumps({"alignments": alignments}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return True


def main() -> None:
    args = parse_args()
    paths = load_dataset_paths(args.jsonl)
    written = 0
    with create_progress() as progress:
        task = progress.add_task("Creating approximate annotation json", total=len(paths))
        for path in paths:
            if create_annotation(path, args.chunk_chars, args.overwrite):
                written += 1
            progress.advance(task)
    console.print(f"[green]Wrote[/green] {written} annotation json files")


if __name__ == "__main__":
    main()

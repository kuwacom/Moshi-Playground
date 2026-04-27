from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path

from datasetPaths import datasetRawDir
from progressUtils import console, create_progress, status


AUDIO_EXTENSIONS = {".wav", ".mp3", ".flac", ".m4a", ".ogg", ".aac"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Extract vocal stems with Demucs and copy them into the active raw dataset"
    )
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=datasetRawDir() / "demucsVocals",
    )
    parser.add_argument(
        "--work-dir",
        type=Path,
        default=Path("datasets/demucsSeparated"),
    )
    parser.add_argument("--model", default="htdemucs")
    parser.add_argument("--device", choices=["auto", "cuda", "cpu"], default="auto")
    parser.add_argument("--segment", type=float)
    parser.add_argument("--jobs", type=int)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def collect_audio_paths(input_path: Path) -> list[Path]:
    if input_path.is_file():
        return [input_path]
    if not input_path.is_dir():
        raise FileNotFoundError(f"Input does not exist: {input_path}")
    paths = [
        path
        for path in sorted(input_path.rglob("*"))
        if path.is_file() and path.suffix.lower() in AUDIO_EXTENSIONS
    ]
    if not paths:
        raise FileNotFoundError(f"No audio files found in {input_path}")
    return paths


def build_demucs_command(args: argparse.Namespace, audio_paths: list[Path]) -> list[str]:
    command = [
        sys.executable,
        "-m",
        "demucs.separate",
        "--two-stems",
        "vocals",
        "-n",
        args.model,
        "-o",
        str(args.work_dir),
    ]
    if args.device != "auto":
        command.extend(["-d", args.device])
    if args.segment is not None:
        command.extend(["--segment", str(args.segment)])
    if args.jobs is not None:
        command.extend(["-j", str(args.jobs)])
    command.extend(str(path) for path in audio_paths)
    return command


def copy_vocal_outputs(args: argparse.Namespace, audio_paths: list[Path]) -> None:
    args.output_dir.mkdir(parents=True, exist_ok=True)
    with create_progress() as progress:
        task = progress.add_task("Copying vocal stems", total=len(audio_paths))
        for audio_path in audio_paths:
            vocal_path = args.work_dir / args.model / audio_path.stem / "vocals.wav"
            if not vocal_path.exists():
                raise FileNotFoundError(
                    f"Demucs did not create expected file: {vocal_path}"
                )
            output_path = args.output_dir / f"{audio_path.stem}_vocals.wav"
            if output_path.exists() and not args.overwrite:
                raise FileExistsError(
                    f"{output_path} already exists. Use --overwrite to replace it."
                )
            shutil.copy2(vocal_path, output_path)
            console.print(f"[green]Copied[/green] vocal stem to {output_path}")
            progress.advance(task)


def main() -> None:
    args = parse_args()
    audio_paths = collect_audio_paths(args.input)
    args.work_dir.mkdir(parents=True, exist_ok=True)
    command = build_demucs_command(args, audio_paths)
    console.print(f"[bold]Demucs input files:[/bold] {len(audio_paths)}")
    try:
        with status("Running Demucs vocal extraction"):
            subprocess.run(command, check=True)
    except ModuleNotFoundError as error:
        raise RuntimeError(
            "Demucs is not installed. Run with: "
            "uv run --project moshi-finetune --with demucs python "
            "scripts/extractVocalsDemucs.py ..."
        ) from error
    except subprocess.CalledProcessError as error:
        raise RuntimeError(f"Demucs failed with exit code {error.returncode}") from error
    copy_vocal_outputs(args, audio_paths)


if __name__ == "__main__":
    main()

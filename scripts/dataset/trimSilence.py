from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import torch
import torchaudio
import torchaudio.functional as audio_functional

from scripts.common.progressUtils import console, create_progress


AUDIO_EXTENSIONS = {".wav", ".mp3", ".flac", ".m4a", ".ogg", ".aac"}


@dataclass(frozen=True)
class KeptRegion:
    startFrame: int
    endFrame: int


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Trim long silent regions while preserving short natural pauses"
    )
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--sample-rate", type=int, default=24_000)
    parser.add_argument("--threshold-db", type=float, default=-45.0)
    parser.add_argument("--frame-ms", type=float, default=20.0)
    parser.add_argument("--min-silence-sec", type=float, default=0.8)
    parser.add_argument("--keep-silence-sec", type=float, default=0.25)
    parser.add_argument("--min-voice-sec", type=float, default=0.15)
    parser.add_argument("--recursive", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def collect_audio_paths(input_path: Path, recursive: bool) -> list[Path]:
    if input_path.is_file():
        return [input_path]
    if not input_path.is_dir():
        raise FileNotFoundError(f"Input does not exist: {input_path}")
    iterator = input_path.rglob("*") if recursive else input_path.glob("*")
    paths = [
        path
        for path in sorted(iterator)
        if path.is_file() and path.suffix.lower() in AUDIO_EXTENSIONS
    ]
    if not paths:
        raise FileNotFoundError(f"No audio files found in {input_path}")
    return paths


def load_audio(path: Path, sample_rate: int) -> torch.Tensor:
    waveform, input_sample_rate = torchaudio.load(path)
    if input_sample_rate != sample_rate:
        waveform = audio_functional.resample(waveform, input_sample_rate, sample_rate)
    return waveform


def frame_voice_mask(
    waveform: torch.Tensor,
    sample_rate: int,
    frame_ms: float,
    threshold_db: float,
) -> tuple[torch.Tensor, int]:
    frame_size = max(1, int(sample_rate * frame_ms / 1000))
    total_frames = waveform.shape[-1]
    padded_frames = ((total_frames + frame_size - 1) // frame_size) * frame_size
    padded = torch.nn.functional.pad(waveform, (0, padded_frames - total_frames))
    frames = padded.unfold(dimension=-1, size=frame_size, step=frame_size)
    rms = torch.sqrt(torch.mean(frames.pow(2), dim=(0, 2)).clamp_min(1e-12))
    db = 20 * torch.log10(rms)
    return db > threshold_db, frame_size


def smooth_voice_mask(
    voice_mask: torch.Tensor,
    sample_rate: int,
    frame_size: int,
    min_voice_sec: float,
) -> torch.Tensor:
    min_voice_frames = max(1, int(min_voice_sec * sample_rate / frame_size))
    smoothed = voice_mask.clone()
    start = 0
    while start < len(smoothed):
        value = bool(smoothed[start].item())
        end = start + 1
        while end < len(smoothed) and bool(smoothed[end].item()) == value:
            end += 1
        if value and end - start < min_voice_frames:
            smoothed[start:end] = False
        start = end
    return smoothed


def build_kept_regions(
    voice_mask: torch.Tensor,
    frame_size: int,
    total_frames: int,
    sample_rate: int,
    min_silence_sec: float,
    keep_silence_sec: float,
) -> list[KeptRegion]:
    min_silence_frames = max(1, int(min_silence_sec * sample_rate / frame_size))
    keep_frames = max(0, int(keep_silence_sec * sample_rate))
    regions: list[KeptRegion] = []
    voice_indices = torch.nonzero(voice_mask, as_tuple=False).flatten().tolist()
    if not voice_indices:
        return [KeptRegion(0, min(total_frames, keep_frames * 2))]

    voice_start = int(voice_indices[0]) * frame_size
    current_start = max(0, voice_start - keep_frames)
    previous_voice_frame = int(voice_indices[0])

    for frame_index in voice_indices[1:]:
        silence_frames = int(frame_index) - previous_voice_frame - 1
        if silence_frames >= min_silence_frames:
            current_end = min(total_frames, (previous_voice_frame + 1) * frame_size + keep_frames)
            regions.append(KeptRegion(current_start, current_end))
            current_start = max(0, int(frame_index) * frame_size - keep_frames)
        previous_voice_frame = int(frame_index)

    final_end = min(total_frames, (previous_voice_frame + 1) * frame_size + keep_frames)
    regions.append(KeptRegion(current_start, final_end))
    return merge_regions(regions)


def merge_regions(regions: list[KeptRegion]) -> list[KeptRegion]:
    if not regions:
        return []
    merged = [regions[0]]
    for region in regions[1:]:
        previous = merged[-1]
        if region.startFrame <= previous.endFrame:
            merged[-1] = KeptRegion(previous.startFrame, max(previous.endFrame, region.endFrame))
        else:
            merged.append(region)
    return merged


def trim_waveform(waveform: torch.Tensor, args: argparse.Namespace) -> torch.Tensor:
    voice_mask, frame_size = frame_voice_mask(
        waveform,
        args.sample_rate,
        args.frame_ms,
        args.threshold_db,
    )
    voice_mask = smooth_voice_mask(
        voice_mask,
        args.sample_rate,
        frame_size,
        args.min_voice_sec,
    )
    regions = build_kept_regions(
        voice_mask,
        frame_size,
        waveform.shape[-1],
        args.sample_rate,
        args.min_silence_sec,
        args.keep_silence_sec,
    )
    clips = [waveform[:, region.startFrame : region.endFrame] for region in regions]
    if not clips:
        return waveform[:, :0]
    return torch.cat(clips, dim=-1)


def build_output_path(input_path: Path, output_root: Path, base_input: Path) -> Path:
    if base_input.is_file():
        return output_root
    relative = input_path.relative_to(base_input)
    return output_root / relative.with_suffix(".wav")


def main() -> None:
    args = parse_args()
    input_paths = collect_audio_paths(args.input, args.recursive)
    with create_progress() as progress:
        task = progress.add_task("Trimming silence", total=len(input_paths))
        for input_path in input_paths:
            output_path = build_output_path(input_path, args.output, args.input)
            if output_path.exists() and not args.overwrite:
                raise FileExistsError(f"{output_path} already exists. Use --overwrite.")
            waveform = load_audio(input_path, args.sample_rate)
            trimmed = trim_waveform(waveform, args)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            torchaudio.save(output_path, trimmed, args.sample_rate)
            before_sec = waveform.shape[-1] / args.sample_rate
            after_sec = trimmed.shape[-1] / args.sample_rate
            console.print(
                f"{input_path} -> {output_path}: "
                f"{before_sec:.2f}s -> {after_sec:.2f}s"
            )
            progress.advance(task)


if __name__ == "__main__":
    main()

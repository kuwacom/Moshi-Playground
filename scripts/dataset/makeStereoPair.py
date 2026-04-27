from __future__ import annotations

import argparse
from pathlib import Path

import torch
import torchaudio
import torchaudio.functional as audio_functional

from scripts.common.progressUtils import console, status


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create a two-channel Moshi training wav")
    parser.add_argument("--left", type=Path, required=True)
    parser.add_argument("--right", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--sample-rate", type=int, default=24_000)
    return parser.parse_args()


def load_mono(path: Path, target_sample_rate: int) -> torch.Tensor:
    waveform, sample_rate = torchaudio.load(path)
    if sample_rate != target_sample_rate:
        waveform = audio_functional.resample(waveform, sample_rate, target_sample_rate)
    return waveform.mean(dim=0, keepdim=True)


def main() -> None:
    args = parse_args()
    with status("Loading and resampling channels"):
        left = load_mono(args.left, args.sample_rate)
        right = load_mono(args.right, args.sample_rate)
    with status(f"Writing stereo wav to [bold]{args.output}[/bold]"):
        max_frames = max(left.shape[-1], right.shape[-1])
        left = torch.nn.functional.pad(left, (0, max_frames - left.shape[-1]))
        right = torch.nn.functional.pad(right, (0, max_frames - right.shape[-1]))
        stereo = torch.cat([left, right], dim=0)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        torchaudio.save(args.output, stereo, args.sample_rate)
    console.print(f"[green]Wrote[/green] stereo wav to {args.output}")


if __name__ == "__main__":
    main()

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

from progressUtils import console, status


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Copy the latest LoRA adapter to a stable path"
    )
    parser.add_argument("--run-dir", type=Path, default=Path("loras/llmJpMoshiV1"))
    parser.add_argument("--output-dir", type=Path, default=Path("loras/llmJpMoshiV1/latest"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    checkpoint_dirs = sorted(
        (args.run_dir / "checkpoints").glob("checkpoint_*/consolidated")
    )
    if not checkpoint_dirs:
        raise FileNotFoundError(f"No consolidated checkpoints found under {args.run_dir}")

    latest = checkpoint_dirs[-1]
    lora_path = latest / "lora.safetensors"
    config_path = latest / "config.json"
    if not lora_path.exists() or not config_path.exists():
        raise FileNotFoundError(f"Missing lora.safetensors or config.json in {latest}")

    with status(f"Exporting latest LoRA from [bold]{latest}[/bold]"):
        args.output_dir.mkdir(parents=True, exist_ok=True)
        shutil.copy2(lora_path, args.output_dir / "lora.safetensors")
        shutil.copy2(config_path, args.output_dir / "config.json")
    console.print(f"[green]Exported[/green] latest LoRA from {latest} to {args.output_dir}")


if __name__ == "__main__":
    main()

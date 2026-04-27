from __future__ import annotations

import argparse
from pathlib import Path
from typing import TypedDict, cast

import yaml


class TrainDataConfig(TypedDict, total=False):
    train_data: str
    eval_data: str


class TrainConfig(TypedDict, total=False):
    data: TrainDataConfig
    run_dir: str


def parseArgs() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Render a Moshi training config from the tracked example config"
    )
    parser.add_argument("--template", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--train-data", required=True)
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--eval-data")
    return parser.parse_args()


def loadConfig(path: Path) -> TrainConfig:
    loaded = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(loaded, dict):
        raise RuntimeError(f"Config template must be a mapping: {path}")
    return cast(TrainConfig, loaded)


def renderConfig(args: argparse.Namespace) -> None:
    config = loadConfig(args.template)
    data = config.setdefault("data", {})
    data["train_data"] = args.train_data
    if args.eval_data is not None:
        data["eval_data"] = args.eval_data
    config["run_dir"] = args.run_dir

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        yaml.safe_dump(config, allow_unicode=True, sort_keys=False),
        encoding="utf-8",
    )


def main() -> None:
    renderConfig(parseArgs())


if __name__ == "__main__":
    main()

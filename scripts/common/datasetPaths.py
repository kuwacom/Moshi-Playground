from __future__ import annotations

import os
from pathlib import Path


def loraName() -> str:
    return os.environ.get("LORA_NAME", "shigureui1")


def datasetRoot() -> Path:
    return Path(os.environ.get("DATASET_ROOT", "datasets"))


def datasetRawDir() -> Path:
    default = datasetRoot() / "raw" / loraName()
    return Path(os.environ.get("DATASET_RAW_DIR", str(default)))


def datasetStereoDir() -> Path:
    default = datasetRoot() / "stereo" / loraName()
    return Path(os.environ.get("DATASET_STEREO_DIR", str(default)))


def datasetCacheDir() -> Path:
    default = datasetRoot() / "cache" / loraName()
    return Path(os.environ.get("DATASET_CACHE_DIR", str(default)))


def datasetTtsDir() -> Path:
    default = datasetRoot() / "tts" / loraName()
    return Path(os.environ.get("DATASET_TTS_DIR", str(default)))


def trainJsonlPath() -> Path:
    default = datasetRoot() / f"{loraName()}.jsonl"
    return Path(os.environ.get("TRAIN_JSONL", str(default)))

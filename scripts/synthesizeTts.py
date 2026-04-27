from __future__ import annotations

import argparse
import os
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

from datasetPaths import datasetTtsDir
from progressUtils import console, status


DEFAULT_TTS_TYPE = "10"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Synthesize speech with the Kuwa CapCut TTS API"
    )
    parser.add_argument("text")
    parser.add_argument("--type")
    parser.add_argument("--url")
    parser.add_argument("--env-file", type=Path, default=Path(".env"))
    parser.add_argument("--output", type=Path, default=datasetTtsDir() / "response.wav")
    parser.add_argument("--timeout", type=float, default=120.0)
    return parser.parse_args()


def load_dotenv(path: Path) -> None:
    if not path.exists():
        return
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip().strip("'").strip('"'))


def main() -> None:
    args = parse_args()
    load_dotenv(args.env_file)
    tts_url = (args.url or os.environ.get("KUWA_TTS_URL", "")).strip()
    if not tts_url:
        raise RuntimeError(
            "KUWA_TTS_URL is required. Set it in .env or pass --url."
        )
    tts_type = args.type or os.environ.get("KUWA_TTS_TYPE", DEFAULT_TTS_TYPE)
    query = urllib.parse.urlencode({"text": args.text, "type": tts_type})
    request = urllib.request.Request(
        f"{tts_url.rstrip('/')}?{query}",
        headers={
            "User-Agent": "curl/8.5.0",
            "Accept": "audio/wav,*/*",
        },
        method="GET",
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    try:
        with status(f"Synthesizing TTS to [bold]{args.output}[/bold]"):
            with urllib.request.urlopen(request, timeout=args.timeout) as response:
                args.output.write_bytes(response.read())
    except urllib.error.HTTPError as error:
        body = error.read().decode("utf-8", errors="replace")
        raise RuntimeError(
            f"TTS request failed with HTTP {error.code}: {body[:500]}"
        ) from error
    console.print(f"[green]Wrote[/green] TTS audio to {args.output}")


if __name__ == "__main__":
    main()

from __future__ import annotations

import argparse
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

from progressUtils import console, status


BASE_URL = "https://api.kuwa.app/v1/capcut/synthesize"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Synthesize speech with the Kuwa CapCut TTS API"
    )
    parser.add_argument("text")
    parser.add_argument("--type", default="10")
    parser.add_argument("--output", type=Path, default=Path("datasets/tts/response.wav"))
    parser.add_argument("--timeout", type=float, default=120.0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    query = urllib.parse.urlencode({"text": args.text, "type": args.type})
    request = urllib.request.Request(
        f"{BASE_URL}?{query}",
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

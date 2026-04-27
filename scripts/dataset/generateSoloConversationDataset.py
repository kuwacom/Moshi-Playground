from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import shutil
import subprocess
import urllib.error
import urllib.parse
import urllib.request
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
import torchaudio
import torchaudio.functional as audio_functional
import whisper_timestamped as whisper
from openai import OpenAI
from openai import APIStatusError, OpenAIError

from scripts.common.datasetPaths import datasetCacheDir
from scripts.common.progressUtils import console, create_progress, status


DEFAULT_OPENAI_MODEL = "anthropic/claude-sonnet-4.6"
DEFAULT_TTS_TYPE = "10"
BALANCE_FILL_INDEX_BASE = 10_000_000
SEMANTIC_GAP_INDEX_BASE = 20_000_000
SEMANTIC_GAP_RESPONSE_KIND = "semantic_gap_insert"


@dataclass(frozen=True)
class EnvConfig:
    openaiBaseUrl: str
    openaiApiKey: str
    openaiModel: str
    ttsUrl: str
    ttsType: str


@dataclass(frozen=True)
class TranscriptSegment:
    index: int
    start: float
    end: float
    text: str


@dataclass(frozen=True)
class GeneratedResponse:
    index: int
    promptStart: float
    promptEnd: float
    promptText: str
    responseText: str
    responseKind: str
    responseStart: float
    responseEnd: float
    ttsPath: str


@dataclass(frozen=True)
class ConversationTurn:
    segment: TranscriptSegment
    previousEnd: float | None
    nextStart: float | None


@dataclass(frozen=True)
class ResponsePlacement:
    start: float
    end: float


@dataclass(frozen=True)
class BalanceFillCandidate:
    turn: ConversationTurn
    nextTurn: ConversationTurn | None
    fillIndex: int
    preferredStart: float
    windowStart: float
    windowEnd: float | None
    placementKind: str


@dataclass(frozen=True)
class SemanticGapPoint:
    pointId: str
    turn: ConversationTurn
    insertAt: float
    beforeText: str
    afterText: str


@dataclass(frozen=True)
class SemanticGapChoice:
    pointId: str
    responseText: str
    responseKind: str


@dataclass(frozen=True)
class TimelineInsertion:
    responseIndex: int
    insertAt: float
    gapSec: float
    responseOffsetSec: float


def transcript_segment_from_dict(data: dict[str, Any]) -> TranscriptSegment:
    return TranscriptSegment(
        index=int(data["index"]),
        start=float(data["start"]),
        end=float(data["end"]),
        text=str(data["text"]),
    )


def generated_response_from_dict(data: dict[str, Any]) -> GeneratedResponse:
    return GeneratedResponse(
        index=int(data["index"]),
        promptStart=float(data["promptStart"]),
        promptEnd=float(data["promptEnd"]),
        promptText=str(data["promptText"]),
        responseText=str(data["responseText"]),
        responseKind=str(data.get("responseKind", "reply")),
        responseStart=float(data["responseStart"]),
        responseEnd=float(data["responseEnd"]),
        ttsPath=str(data["ttsPath"]),
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create stereo Moshi data from solo audio using local Whisper, an OpenAI-compatible API, and TTS"
    )
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--metadata-output",
        type=Path,
        help="Defaults to the output wav path with a .responses.json suffix",
    )
    parser.add_argument("--env-file", type=Path, default=Path(".env"))
    parser.add_argument("--language", default="ja")
    parser.add_argument("--whisper-model", default="large-v3")
    parser.add_argument("--sample-rate", type=int, default=24_000)
    parser.add_argument("--response-delay-sec", type=float, default=0.35)
    parser.add_argument("--response-margin-sec", type=float, default=0.25)
    parser.add_argument("--timing-jitter-sec", type=float, default=0.18)
    parser.add_argument("--response-placement-retries", type=int, default=1)
    parser.add_argument("--merge-gap-sec", type=float, default=0.8)
    parser.add_argument(
        "--interaction-mode",
        choices=["auto", "reply", "pre-question"],
        default="auto",
    )
    parser.add_argument("--min-insert-gap-sec", type=float, default=0.8)
    parser.add_argument("--pre-question-gap-sec", type=float, default=1.6)
    parser.add_argument("--short-gap-sec", type=float, default=2.0)
    parser.add_argument("--long-gap-sec", type=float, default=5.0)
    parser.add_argument("--min-response-chars", type=int, default=12)
    parser.add_argument("--max-response-chars", type=int, default=48)
    parser.add_argument("--tts-chars-per-sec", type=float, default=8.0)
    parser.add_argument("--tts-speed", type=float, default=1.2)
    parser.add_argument(
        "--balance-fill-mode",
        choices=["auto", "always", "off"],
        default="auto",
    )
    parser.add_argument("--target-right-ratio", type=float, default=0.4)
    parser.add_argument("--max-right-ratio", type=float, default=0.5)
    parser.add_argument("--long-turn-fill-sec", type=float, default=12.0)
    parser.add_argument("--fill-interval-sec", type=float, default=6.0)
    parser.add_argument("--fill-max-chars", type=int, default=14)
    parser.add_argument("--allow-left-overlap-fill", action="store_true")
    parser.add_argument(
        "--semantic-gap-insert-mode",
        choices=["off", "auto", "always"],
        default="off",
    )
    parser.add_argument("--semantic-gap-max-count", type=int, default=2)
    parser.add_argument("--semantic-gap-min-turn-sec", type=float, default=8.0)
    parser.add_argument("--semantic-gap-max-chars", type=int, default=36)
    parser.add_argument("--min-segment-sec", type=float, default=0.4)
    parser.add_argument("--max-segments", type=int)
    parser.add_argument("--keep-tts-dir", type=Path)
    parser.add_argument("--cache-dir", type=Path, default=datasetCacheDir())
    parser.add_argument("--refresh-transcript", action="store_true")
    parser.add_argument("--refresh-responses", action="store_true")
    parser.add_argument("--tts-timeout", type=float, default=120.0)
    parser.add_argument("--llm-timeout", type=float, default=120.0)
    return parser.parse_args()


def load_dotenv(path: Path) -> None:
    if not path.exists():
        return
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip("'").strip('"')
        os.environ.setdefault(key, value)


def load_env_config(env_file: Path) -> EnvConfig:
    load_dotenv(env_file)
    api_key = os.environ.get("OPENAI_API_KEY", "").strip()
    if not api_key:
        api_key = os.environ.get("LITELLM_API_KEY", "").strip()
    if not api_key:
        raise RuntimeError(
            "OPENAI_API_KEY is required. Copy .env.example to .env and set the key."
        )
    openai_base_url = os.environ.get(
        "OPENAI_BASE_URL",
        os.environ.get("LITELLM_BASE_URL", ""),
    ).strip()
    if not openai_base_url:
        raise RuntimeError(
            "OPENAI_BASE_URL is required. Copy .env.example to .env and set the API base URL."
        )
    tts_url = os.environ.get("KUWA_TTS_URL", "").strip()
    if not tts_url:
        raise RuntimeError(
            "KUWA_TTS_URL is required. Copy .env.example to .env and set the TTS URL."
        )
    return EnvConfig(
        openaiBaseUrl=openai_base_url.rstrip("/"),
        openaiApiKey=api_key,
        openaiModel=os.environ.get(
            "OPENAI_MODEL",
            os.environ.get("LITELLM_MODEL", DEFAULT_OPENAI_MODEL),
        ),
        ttsUrl=tts_url.rstrip("/"),
        ttsType=os.environ.get("KUWA_TTS_TYPE", DEFAULT_TTS_TYPE),
    )


def create_openai_client(config: EnvConfig) -> OpenAI:
    return OpenAI(
        api_key=config.openaiApiKey,
        base_url=config.openaiBaseUrl,
    )


def load_mono_audio(path: Path, sample_rate: int) -> torch.Tensor:
    waveform, input_sample_rate = torchaudio.load(path)
    if input_sample_rate != sample_rate:
        waveform = audio_functional.resample(waveform, input_sample_rate, sample_rate)
    return waveform.mean(dim=0, keepdim=True)


def transcribe_audio(
    path: Path,
    language: str,
    whisper_model: str,
    min_segment_sec: float,
    max_segments: int | None,
) -> list[TranscriptSegment]:
    model = load_whisper_model(whisper_model)
    return transcribe_audio_with_model(
        model,
        path,
        language,
        min_segment_sec,
        max_segments,
    )


def load_whisper_model(whisper_model: str) -> Any:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    with status(f"Loading Whisper model [bold]{whisper_model}[/bold] on {device}"):
        model = whisper.load_model(
            whisper_model,
            device=device,
            # whisper_timestamped 側で whisper/ が足されるため root は models にする
            download_root="models",
        )
    return model


def transcribe_audio_with_model(
    model: Any,
    path: Path,
    language: str,
    min_segment_sec: float,
    max_segments: int | None,
) -> list[TranscriptSegment]:
    with status(f"Transcribing [bold]{path}[/bold]"):
        result = whisper.transcribe(
            model,
            str(path),
            language=language,
            best_of=5,
            beam_size=5,
            temperature=(0.0, 0.2, 0.4, 0.6, 0.8, 1.0),
            verbose=None,
        )
    segments: list[TranscriptSegment] = []
    for index, segment in enumerate(result.get("segments", [])):
        text = str(segment.get("text", "")).strip()
        start = float(segment.get("start", 0.0))
        end = float(segment.get("end", start))
        if not text or end - start < min_segment_sec:
            continue
        segments.append(
            TranscriptSegment(index=len(segments), start=start, end=end, text=text)
        )
        if max_segments is not None and len(segments) >= max_segments:
            break
    return segments


def build_conversation_turns(
    segments: list[TranscriptSegment],
    merge_gap_sec: float,
) -> list[ConversationTurn]:
    if not segments:
        return []

    grouped: list[TranscriptSegment] = []
    current_start = segments[0].start
    current_end = segments[0].end
    current_texts = [segments[0].text]

    for segment in segments[1:]:
        gap = segment.start - current_end
        if gap <= merge_gap_sec:
            current_end = segment.end
            current_texts.append(segment.text)
            continue
        grouped.append(
            TranscriptSegment(
                index=len(grouped),
                start=current_start,
                end=current_end,
                text="\n".join(current_texts),
            )
        )
        current_start = segment.start
        current_end = segment.end
        current_texts = [segment.text]

    grouped.append(
        TranscriptSegment(
            index=len(grouped),
            start=current_start,
            end=current_end,
            text="\n".join(current_texts),
        )
    )

    turns: list[ConversationTurn] = []
    for index, segment in enumerate(grouped):
        previous_end = grouped[index - 1].end if index > 0 else None
        next_start = grouped[index + 1].start if index + 1 < len(grouped) else None
        turns.append(
            ConversationTurn(
                segment=segment,
                previousEnd=previous_end,
                nextStart=next_start,
            )
        )
    return turns


def response_char_limit_for_gap(
    gap_sec: float | None,
    min_response_chars: int,
    max_response_chars: int,
    response_delay_sec: float,
    response_margin_sec: float,
    short_gap_sec: float,
    long_gap_sec: float,
    tts_chars_per_sec: float,
    tts_speed: float,
) -> int:
    if gap_sec is None:
        return max_response_chars

    gap_sec = max(0.0, gap_sec)
    if gap_sec >= long_gap_sec:
        return max_response_chars
    if gap_sec <= short_gap_sec:
        return min(max_response_chars, min_response_chars)

    available_sec = max(0.0, gap_sec - response_delay_sec - response_margin_sec)
    fitted_chars = int(available_sec * tts_chars_per_sec * max(0.1, tts_speed))
    return max(min_response_chars, min(max_response_chars, fitted_chars))


def response_char_limit_for_turn(
    turn: ConversationTurn,
    min_response_chars: int,
    max_response_chars: int,
    response_delay_sec: float,
    response_margin_sec: float,
    short_gap_sec: float,
    long_gap_sec: float,
    tts_chars_per_sec: float,
    tts_speed: float,
) -> int:
    gap_sec = None if turn.nextStart is None else turn.nextStart - turn.segment.end
    return response_char_limit_for_gap(
        gap_sec,
        min_response_chars,
        max_response_chars,
        response_delay_sec,
        response_margin_sec,
        short_gap_sec,
        long_gap_sec,
        tts_chars_per_sec,
        tts_speed,
    )


def pre_question_char_limit_for_turn(
    turn: ConversationTurn,
    min_response_chars: int,
    max_response_chars: int,
    response_delay_sec: float,
    response_margin_sec: float,
    short_gap_sec: float,
    long_gap_sec: float,
    tts_chars_per_sec: float,
    tts_speed: float,
) -> int:
    gap_sec = None if turn.previousEnd is None else turn.segment.start - turn.previousEnd
    return response_char_limit_for_gap(
        gap_sec,
        min_response_chars,
        max_response_chars,
        response_delay_sec,
        response_margin_sec,
        short_gap_sec,
        long_gap_sec,
        tts_chars_per_sec,
        tts_speed,
    )


def stable_fraction(*parts: object) -> float:
    key = "\0".join(str(part) for part in parts).encode("utf-8")
    digest = hashlib.blake2b(key, digest_size=4).digest()
    return int.from_bytes(digest, byteorder="big") / 2**32


def stable_jitter_sec(
    segment_index: int,
    response_kind: str,
    timing_jitter_sec: float,
) -> float:
    if timing_jitter_sec <= 0.0:
        return 0.0
    return stable_fraction(segment_index, response_kind, "timing") * timing_jitter_sec


def normalize_response_text(text: str) -> str:
    return "".join(
        char
        for char in text.strip().lower()
        if not char.isspace() and char not in "。、，,.！？!?「」『』（）()[]【】"
    )


def response_family(text: str) -> str | None:
    normalized = normalize_response_text(text)
    families = {
        "なるほど": ["なるほど", "なるほどですね", "そうなんですね", "そうなんだ"],
        "たしかに": ["たしかに", "確かに", "たしかにね"],
        "うん": ["うん", "うんうん", "はい", "そうですね", "そうだね"],
        "きになる": ["気になります", "そこ気になります", "もう少し聞きたいです"],
    }
    for family, markers in families.items():
        if any(normalized == normalize_response_text(marker) for marker in markers):
            return family
    return None


def response_variant_candidates(response_kind: str) -> list[str]:
    if response_kind == "pre_question":
        return [
            "それって何がきっかけですか",
            "そこもう少し聞きたいです",
            "具体的にはどんな感じですか",
            "それはどういう意味ですか",
            "何が一番大きかったんですか",
        ]
    if response_kind == "balance_fill_llm":
        return [
            "たしかに",
            "うんうん",
            "そこ気になります",
            "それは大きいですね",
            "なるほど、面白いです",
            "わかります",
        ]
    if response_kind == SEMANTIC_GAP_RESPONSE_KIND:
        return [
            "そこ、もう少し聞きたいです",
            "それって何がきっかけですか",
            "たしかに、そこ大事ですね",
            "具体的にはどんな感じですか",
            "なるほど、続き気になります",
            "それはかなり大きいですね",
        ]
    return [
        "たしかに、そこ大事ですね",
        "なるほど、そういう見方なんですね",
        "それは気になります",
        "もう少し聞きたいです",
        "そこは面白いですね",
        "たしかにありそうです",
    ]


def diversify_response_text(
    text: str,
    response_kind: str,
    avoid_texts: list[str] | None,
    max_chars: int,
    seed_parts: tuple[object, ...],
) -> str:
    normalized = normalize_response_text(text)
    avoid_texts = avoid_texts or []
    avoid_normalized = {normalize_response_text(item) for item in avoid_texts}
    avoid_families = {
        family
        for item in avoid_texts
        if (family := response_family(item)) is not None
    }
    current_family = response_family(text)
    should_replace = (
        normalized in avoid_normalized
        or (current_family is not None and current_family in avoid_families)
    )
    if not should_replace:
        return text[:max_chars].strip()

    candidates = [
        candidate
        for candidate in response_variant_candidates(response_kind)
        if normalize_response_text(candidate) not in avoid_normalized
        and response_family(candidate) not in avoid_families
    ]
    if not candidates:
        return text[:max_chars].strip()
    index = int(stable_fraction(*seed_parts, response_kind, text) * len(candidates))
    return candidates[index][:max_chars].strip()


def recent_response_texts(
    responses: list[GeneratedResponse],
    limit: int = 8,
) -> list[str]:
    sorted_responses = sorted(responses, key=lambda response: response.responseStart)
    return [
        response.responseText
        for response in sorted_responses[-limit:]
        if response.responseText.strip()
    ]


def responses_before_turn(
    responses: list[GeneratedResponse],
    turn: ConversationTurn,
    exclude_index: int,
) -> list[GeneratedResponse]:
    return [
        response
        for response in responses
        if response.index != exclude_index
        and response.responseStart < turn.segment.end
    ]


def avoid_texts_instruction(avoid_texts: list[str] | None) -> str:
    if not avoid_texts:
        return ""
    lines = "\n".join(f"- {text}" for text in avoid_texts[-8:])
    return f"\n最近使った右ch発話。意味の薄い反復になるので同じ言い回しは禁止:\n{lines}\n"


def looks_like_comment_answer(text: str) -> bool:
    normalized = text.replace(" ", "").replace("\n", "")
    markers = [
        "コメント",
        "質問",
        "それは",
        "それって",
        "これは",
        "これって",
        "あれは",
        "そうですね",
        "そうだね",
        "いや",
        "違う",
        "たしかに",
        "確かに",
        "なるほど",
        "ありがとう",
        "助かる",
        "いい質問",
        "どうなんだろう",
        "なんでか",
        "なんで言うと",
        "なぜかというと",
        "理由",
        "結論",
        "個人的には",
        "答えると",
        "聞かれる",
        "聞かれた",
        "コメントで",
        "コメント欄",
        "っていう",
        "というと",
    ]
    return any(marker in normalized for marker in markers)


def allowed_interaction_kinds(
    interaction_mode: str,
    turn: ConversationTurn,
    min_insert_gap_sec: float,
    pre_question_gap_sec: float,
) -> list[str]:
    reply_gap = None if turn.nextStart is None else turn.nextStart - turn.segment.end
    previous_gap = (
        None if turn.previousEnd is None else turn.segment.start - turn.previousEnd
    )
    can_reply = reply_gap is None or reply_gap >= min_insert_gap_sec
    can_pre_question = (
        previous_gap is not None
        and previous_gap >= pre_question_gap_sec
        and looks_like_comment_answer(turn.segment.text)
    )

    if interaction_mode == "reply":
        return ["reply"]
    if interaction_mode == "pre-question":
        return ["pre_question"] if can_pre_question else []
    if interaction_mode == "auto":
        allowed: list[str] = []
        if can_pre_question:
            allowed.append("pre_question")
        if can_reply:
            allowed.append("reply")
        return allowed
    return ["reply"]


def load_or_create_transcript(
    transcript_path: Path,
    transcribe: Callable[[], list[TranscriptSegment]],
    refresh: bool,
) -> list[TranscriptSegment]:
    if transcript_path.exists() and not refresh:
        data = json.loads(transcript_path.read_text(encoding="utf-8"))
        console.print(f"[cyan]Loaded cached transcript[/cyan] {transcript_path}")
        return [transcript_segment_from_dict(item) for item in data["segments"]]

    transcript = transcribe()
    transcript_path.parent.mkdir(parents=True, exist_ok=True)
    transcript_path.write_text(
        json.dumps(
            {"segments": [segment.__dict__ for segment in transcript]},
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    console.print(f"[green]Cached transcript[/green] {transcript_path}")
    return transcript


def load_cached_responses(responses_path: Path, refresh: bool) -> dict[int, GeneratedResponse]:
    if refresh or not responses_path.exists():
        return {}
    data = json.loads(responses_path.read_text(encoding="utf-8"))
    responses = {
        int(item["index"]): generated_response_from_dict(item)
        for item in data.get("responses", [])
    }
    console.print(
        f"[cyan]Loaded cached responses[/cyan] {responses_path} ({len(responses)} items)"
    )
    return responses


def save_cached_responses(
    responses_path: Path,
    responses: list[GeneratedResponse],
) -> None:
    responses_path.parent.mkdir(parents=True, exist_ok=True)
    responses_path.write_text(
        json.dumps(
            {"responses": [response.__dict__ for response in responses]},
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )


def parse_interaction_response(
    content: str,
    allowed_kinds: list[str],
    fallback_kind: str,
    max_chars_by_kind: dict[str, int],
) -> tuple[str, str]:
    stripped = content.strip()
    parsed: dict[str, Any] | None = None
    if stripped.startswith("{") and stripped.endswith("}"):
        try:
            data = json.loads(stripped)
            if isinstance(data, dict):
                parsed = data
        except json.JSONDecodeError:
            parsed = None

    if parsed is None:
        start = stripped.find("{")
        end = stripped.rfind("}")
        if start >= 0 and end > start:
            try:
                data = json.loads(stripped[start : end + 1])
                if isinstance(data, dict):
                    parsed = data
            except json.JSONDecodeError:
                parsed = None

    if parsed is None:
        kind = fallback_kind
        text = stripped
    else:
        kind = str(parsed.get("kind", fallback_kind)).strip()
        text_value = (
            parsed.get("text")
            or parsed.get("content")
            or parsed.get("message")
            or parsed.get("utterance")
            or ""
        )
        text = str(text_value).strip()

    if kind in {"pre-question", "question", "before", "before_question"}:
        kind = "pre_question"
    if kind not in allowed_kinds:
        kind = fallback_kind
    if not text:
        text = "それってどういうことですか" if kind == "pre_question" else "なるほど"

    text = text.replace("\n", " ").strip()
    return kind, text[: max_chars_by_kind[kind]].strip()


def request_interaction_completion(
    client: OpenAI,
    config: EnvConfig,
    transcript: list[TranscriptSegment],
    segment: TranscriptSegment,
    allowed_kinds: list[str],
    max_chars_by_kind: dict[str, int],
    timeout: float,
    time_until_next_sec: float | None = None,
    time_since_previous_sec: float | None = None,
    avoid_texts: list[str] | None = None,
) -> tuple[str, str]:
    context = build_recent_context(transcript, segment.index)
    next_timing_instruction = (
        "次の発言まで十分な間があります。自然に少し具体的に返してください。"
        if time_until_next_sec is None
        else f"次の発言まで約{time_until_next_sec:.1f}秒です。間に合う短さで返してください。"
    )
    previous_timing_instruction = (
        "直前の発言からの間隔は不明です。"
        if time_since_previous_sec is None
        else f"直前の発言から約{time_since_previous_sec:.1f}秒空いています。"
    )
    kind_lines = []
    if "reply" in allowed_kinds:
        kind_lines.append(
            f"- reply: 今の発言の後ろに置く自然な返事。{max_chars_by_kind['reply']}文字以内。"
        )
    if "pre_question" in allowed_kinds:
        kind_lines.append(
            "- pre_question: 今の発言の手前にあったら自然なリスナーの質問やコメント。"
            f"{max_chars_by_kind['pre_question']}文字以内。"
            "配信者がコメントに答えていると見える場合だけ選ぶ。"
        )
    fallback_kind = allowed_kinds[0]
    response = client.chat.completions.create(
        model=config.openaiModel,
        temperature=0.8,
        max_tokens=80,
        timeout=timeout,
        messages=[
            {
                "role": "system",
                "content": (
                    "あなたは自然な雑談相手です。"
                    "一人配信の音声を、リスナーと会話しているような学習データに整えます。"
                    "コメント読みや質問回答に見える発話なら手前のリスナー発言を作り、"
                    "そうでない独立した説明や雑談なら発話後の返事を作ってください。"
                    "同じ相槌や薄い同意を繰り返さず、直前の内容に一歩だけ具体的に触れてください。"
                    "音声合成するため、括弧書き、絵文字、長い説明、メタ発言は避けてください。"
                    "出力は必ずJSONだけにしてください。"
                ),
            },
            {
                "role": "user",
                "content": (
                    f"直近の文脈:\n{context}\n\n"
                    f"今の発言:\n{segment.text}\n\n"
                    f"{previous_timing_instruction}\n"
                    f"{next_timing_instruction}\n\n"
                    f"{avoid_texts_instruction(avoid_texts)}"
                    "選べる種類:\n"
                    f"{chr(10).join(kind_lines)}\n\n"
                    'JSON形式: {"kind":"reply","text":"短い発話"}'
                ),
            },
        ],
    )
    content = response.choices[0].message.content
    if not content:
        raise RuntimeError(f"LLM returned an empty response: {response}")
    response_kind, response_text = parse_interaction_response(
        content,
        allowed_kinds,
        fallback_kind,
        max_chars_by_kind,
    )
    return response_kind, diversify_response_text(
        response_text,
        response_kind,
        avoid_texts,
        max_chars_by_kind[response_kind],
        (segment.index, segment.start, segment.end),
    )


def request_interaction_completion_checked(
    client: OpenAI,
    config: EnvConfig,
    transcript: list[TranscriptSegment],
    segment: TranscriptSegment,
    allowed_kinds: list[str],
    max_chars_by_kind: dict[str, int],
    timeout: float,
    time_until_next_sec: float | None = None,
    time_since_previous_sec: float | None = None,
    avoid_texts: list[str] | None = None,
) -> tuple[str, str]:
    try:
        return request_interaction_completion(
            client,
            config,
            transcript,
            segment,
            allowed_kinds,
            max_chars_by_kind,
            timeout,
            time_until_next_sec,
            time_since_previous_sec,
            avoid_texts,
        )
    except APIStatusError as error:
        raise RuntimeError(
            f"LLM request failed with HTTP {error.status_code}: {error.response.text[:1000]}"
        ) from error
    except OpenAIError as error:
        raise RuntimeError(
            f"LLM request failed: {error}"
        ) from error


def request_chat_completion_checked(
    client: OpenAI,
    config: EnvConfig,
    transcript: list[TranscriptSegment],
    segment: TranscriptSegment,
    max_response_chars: int,
    timeout: float,
    time_until_next_sec: float | None = None,
) -> str:
    _, text = request_interaction_completion_checked(
        client,
        config,
        transcript,
        segment,
        ["reply"],
        {"reply": max_response_chars},
        timeout,
        time_until_next_sec,
        None,
    )
    return text


def request_balance_fill_completion(
    client: OpenAI,
    config: EnvConfig,
    transcript: list[TranscriptSegment],
    candidate: BalanceFillCandidate,
    max_chars: int,
    timeout: float,
    avoid_texts: list[str] | None = None,
) -> str:
    turn = candidate.turn
    context = build_recent_context(transcript, turn.segment.index)
    offset_sec = max(0.0, candidate.preferredStart - turn.segment.start)
    if candidate.placementKind == "gap_bridge":
        next_text = "なし" if candidate.nextTurn is None else candidate.nextTurn.segment.text
        system_prompt = (
            "あなたは一人配信を自然な双方向会話データへ整える編集者です。"
            "配信者が話し終えた後の無音に、右チャンネルの短い返事か自然な質問を入れます。"
            "直前の発話と次の発話をつなぎ、同じ相槌の反復や話題転換を避けてください。"
            "括弧書き、絵文字、説明、メタ発言は禁止です。"
        )
        user_prompt = (
            f"直近の文脈:\n{context}\n\n"
            f"直前の配信者発話:\n{turn.segment.text}\n\n"
            f"次の配信者発話:\n{next_text}\n\n"
            f"差し込み可能な無音: 約"
            f"{max(0.0, (candidate.windowEnd or candidate.windowStart) - candidate.windowStart):.1f}秒\n"
            f"{avoid_texts_instruction(avoid_texts)}"
            f"{max_chars}文字以内で、自然な短い右ch発話を1つだけ返してください。"
        )
    else:
        system_prompt = (
            "あなたは一人配信を自然な双方向会話データへ整える編集者です。"
            "配信者が長く話している途中に、右チャンネルへ短いリスナー音声を差し込みます。"
            "話を遮らず、内容に軽く合う相槌、驚き、短い促しを作ってください。"
            "質問で話題を変えないでください。括弧書き、絵文字、説明、メタ発言は禁止です。"
        )
        user_prompt = (
            f"直近の文脈:\n{context}\n\n"
            f"長い発話:\n{turn.segment.text}\n\n"
            f"差し込み位置: この発話の開始から約{offset_sec:.1f}秒後\n"
            f"{avoid_texts_instruction(avoid_texts)}"
            f"{max_chars}文字以内で、自然な短いリスナー発話を1つだけ返してください。"
        )
    response = client.chat.completions.create(
        model=config.openaiModel,
        temperature=0.85,
        max_tokens=60,
        timeout=timeout,
        messages=[
            {
                "role": "system",
                "content": system_prompt,
            },
            {
                "role": "user",
                "content": user_prompt,
            },
        ],
    )
    content = response.choices[0].message.content
    if not content:
        raise RuntimeError(f"LLM returned an empty balance-fill response: {response}")
    text = content.replace("\n", " ").strip()[:max_chars]
    return diversify_response_text(
        text,
        "balance_fill_llm",
        avoid_texts,
        max_chars,
        (
            turn.segment.index,
            candidate.fillIndex,
            candidate.preferredStart,
            candidate.placementKind,
        ),
    )


def request_balance_fill_completion_checked(
    client: OpenAI,
    config: EnvConfig,
    transcript: list[TranscriptSegment],
    candidate: BalanceFillCandidate,
    max_chars: int,
    timeout: float,
    avoid_texts: list[str] | None = None,
) -> str:
    try:
        return request_balance_fill_completion(
            client,
            config,
            transcript,
            candidate,
            max_chars,
            timeout,
            avoid_texts,
        )
    except APIStatusError as error:
        raise RuntimeError(
            f"LLM balance-fill request failed with HTTP {error.status_code}: "
            f"{error.response.text[:1000]}"
        ) from error
    except OpenAIError as error:
        raise RuntimeError(f"LLM balance-fill request failed: {error}") from error


def compact_prompt_text(text: str, max_chars: int = 180) -> str:
    normalized = " ".join(text.split())
    if len(normalized) <= max_chars:
        return normalized
    return f"{normalized[: max_chars - 1]}…"


def parse_semantic_gap_choice(
    content: str,
    points: list[SemanticGapPoint],
    max_chars: int,
    avoid_texts: list[str] | None,
) -> SemanticGapChoice:
    stripped = content.strip()
    parsed: dict[str, Any] | None = None
    if stripped.startswith("{") and stripped.endswith("}"):
        try:
            data = json.loads(stripped)
            if isinstance(data, dict):
                parsed = data
        except json.JSONDecodeError:
            parsed = None
    if parsed is None:
        start = stripped.find("{")
        end = stripped.rfind("}")
        if start >= 0 and end > start:
            try:
                data = json.loads(stripped[start : end + 1])
                if isinstance(data, dict):
                    parsed = data
            except json.JSONDecodeError:
                parsed = None

    point_ids = {point.pointId for point in points}
    fallback_point_id = points[0].pointId
    if parsed is None:
        point_id = fallback_point_id
        response_kind = SEMANTIC_GAP_RESPONSE_KIND
        text = stripped or "そこ、もう少し聞きたいです"
    else:
        point_id = str(parsed.get("pointId", fallback_point_id)).strip()
        response_kind = str(parsed.get("kind", SEMANTIC_GAP_RESPONSE_KIND)).strip()
        text_value = (
            parsed.get("text")
            or parsed.get("content")
            or parsed.get("message")
            or parsed.get("utterance")
            or ""
        )
        text = str(text_value).strip()

    if point_id not in point_ids:
        point_id = fallback_point_id
    if response_kind in {"question", "pre_question", "pre-question"}:
        response_kind = SEMANTIC_GAP_RESPONSE_KIND
    if response_kind not in {"reply", SEMANTIC_GAP_RESPONSE_KIND}:
        response_kind = SEMANTIC_GAP_RESPONSE_KIND
    if not text:
        text = "そこ、もう少し聞きたいです"
    text = text.replace("\n", " ").strip()
    text = diversify_response_text(
        text,
        SEMANTIC_GAP_RESPONSE_KIND,
        avoid_texts,
        max_chars,
        (point_id, text),
    )
    return SemanticGapChoice(
        pointId=point_id,
        responseText=text[:max_chars].strip(),
        responseKind=SEMANTIC_GAP_RESPONSE_KIND,
    )


def request_semantic_gap_choice(
    client: OpenAI,
    config: EnvConfig,
    transcript: list[TranscriptSegment],
    points: list[SemanticGapPoint],
    max_chars: int,
    timeout: float,
    avoid_texts: list[str] | None = None,
) -> SemanticGapChoice:
    first_turn_index = points[0].turn.segment.index
    context = build_recent_context(transcript, first_turn_index)
    point_lines = []
    for point in points:
        point_lines.append(
            "- "
            f'pointId="{point.pointId}" '
            f"時刻={point.insertAt:.1f}s "
            f"直前={compact_prompt_text(point.beforeText, 120)} "
            f"直後={compact_prompt_text(point.afterText, 120)}"
        )
    response = client.chat.completions.create(
        model=config.openaiModel,
        temperature=0.8,
        max_tokens=100,
        timeout=timeout,
        messages=[
            {
                "role": "system",
                "content": (
                    "あなたは一人配信を自然な双方向会話データへ整える編集者です。"
                    "無音が少ない配信に、会話として自然な短い間を1つ追加します。"
                    "候補地点から、配信者が一まとまり言い切った直後、質問や短い応答を挟んでも"
                    "次の発話につながる場所を選んでください。"
                    "文の途中、固有名詞の途中、強い話題転換になる場所は避けます。"
                    "同じ相槌の反復、括弧書き、絵文字、説明、メタ発言は禁止です。"
                    "出力は必ずJSONだけにしてください。"
                ),
            },
            {
                "role": "user",
                "content": (
                    f"直近の文脈:\n{context}\n\n"
                    "候補地点:\n"
                    f"{chr(10).join(point_lines)}\n\n"
                    f"{avoid_texts_instruction(avoid_texts)}"
                    f"{max_chars}文字以内で、自然な右ch発話を1つ作ってください。\n"
                    'JSON形式: {"pointId":"候補ID","kind":"reply","text":"短い発話"}'
                ),
            },
        ],
    )
    content = response.choices[0].message.content
    if not content:
        raise RuntimeError(f"LLM returned an empty semantic-gap response: {response}")
    return parse_semantic_gap_choice(content, points, max_chars, avoid_texts)


def request_semantic_gap_choice_checked(
    client: OpenAI,
    config: EnvConfig,
    transcript: list[TranscriptSegment],
    points: list[SemanticGapPoint],
    max_chars: int,
    timeout: float,
    avoid_texts: list[str] | None = None,
) -> SemanticGapChoice:
    try:
        return request_semantic_gap_choice(
            client,
            config,
            transcript,
            points,
            max_chars,
            timeout,
            avoid_texts,
        )
    except APIStatusError as error:
        raise RuntimeError(
            f"LLM semantic-gap request failed with HTTP {error.status_code}: "
            f"{error.response.text[:1000]}"
        ) from error
    except OpenAIError as error:
        raise RuntimeError(f"LLM semantic-gap request failed: {error}") from error


def build_recent_context(
    transcript: list[TranscriptSegment],
    current_index: int,
    max_items: int = 4,
) -> str:
    start_index = max(0, current_index - max_items)
    items = transcript[start_index:current_index]
    if not items:
        return "なし"
    return "\n".join(f"- {item.text}" for item in items)


def extract_chat_content(data: dict[str, Any]) -> str:
    choices = data.get("choices")
    if not isinstance(choices, list) or not choices:
        raise RuntimeError(f"Unexpected LLM response: {data}")
    first_choice = choices[0]
    if not isinstance(first_choice, dict):
        raise RuntimeError(f"Unexpected LLM choice: {first_choice}")
    message = first_choice.get("message")
    if not isinstance(message, dict):
        raise RuntimeError(f"Unexpected LLM message: {message}")
    content = message.get("content")
    if not isinstance(content, str):
        raise RuntimeError(f"Unexpected LLM content: {content}")
    return content


def synthesize_tts(
    config: EnvConfig,
    text: str,
    output: Path,
    timeout: float,
) -> None:
    query = urllib.parse.urlencode({"text": text, "type": config.ttsType})
    request = urllib.request.Request(
        f"{config.ttsUrl}?{query}",
        headers={
            "User-Agent": "curl/8.5.0",
            "Accept": "audio/wav,*/*",
        },
        method="GET",
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            output.write_bytes(response.read())
    except urllib.error.HTTPError as error:
        body = error.read().decode("utf-8", errors="replace")
        raise RuntimeError(
            f"TTS request failed with HTTP {error.code}: {body[:500]}"
        ) from error


def apply_tts_speed(
    input_path: Path,
    output_path: Path,
    speed: float,
) -> None:
    if abs(speed - 1.0) < 1e-6:
        if input_path != output_path:
            shutil.copy2(input_path, output_path)
        return
    ffmpeg_path = shutil.which("ffmpeg")
    if ffmpeg_path is None:
        raise RuntimeError(
            "ffmpeg is required for pitch-preserving TTS speed adjustment. "
            "Install ffmpeg or run with --tts-speed 1.0."
        )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = output_path.with_suffix(f"{output_path.suffix}.tmp.wav")
    command = [
        ffmpeg_path,
        "-y",
        "-hide_banner",
        "-loglevel",
        "error",
        "-i",
        str(input_path),
        "-filter:a",
        f"atempo={speed}",
        "-vn",
        str(tmp_path),
    ]
    try:
        subprocess.run(command, check=True)
        tmp_path.replace(output_path)
    except subprocess.CalledProcessError as error:
        raise RuntimeError(f"ffmpeg failed while adjusting TTS speed: {error}") from error
    finally:
        if tmp_path.exists():
            tmp_path.unlink()


def synthesize_tts_with_speed(
    config: EnvConfig,
    text: str,
    raw_output: Path,
    output: Path,
    timeout: float,
    speed: float,
) -> None:
    if not raw_output.exists():
        synthesize_tts(config, text, raw_output, timeout)
    apply_tts_speed(raw_output, output, speed)


def response_start_for_kind(
    response_kind: str,
    turn: ConversationTurn,
    tts_duration_sec: float,
    response_delay_sec: float,
    response_margin_sec: float,
    next_available_start: float,
    timing_jitter_sec: float = 0.0,
) -> float:
    jitter_sec = stable_jitter_sec(
        turn.segment.index,
        response_kind,
        timing_jitter_sec,
    )
    if response_kind == "pre_question":
        desired_start = max(
            0.0,
            turn.segment.start - response_margin_sec - tts_duration_sec,
        )
        if turn.previousEnd is None:
            return desired_start
        earliest_start = turn.previousEnd + response_delay_sec + jitter_sec
        return max(earliest_start, desired_start)
    return max(
        turn.segment.end + response_delay_sec + jitter_sec,
        next_available_start,
    )


def response_window_for_kind(
    response_kind: str,
    turn: ConversationTurn,
    response_delay_sec: float,
    response_margin_sec: float,
    next_available_start: float,
    timing_jitter_sec: float,
) -> tuple[float, float | None]:
    jitter_sec = stable_jitter_sec(
        turn.segment.index,
        response_kind,
        timing_jitter_sec,
    )
    if response_kind == "pre_question":
        window_start = 0.0
        if turn.previousEnd is not None:
            window_start = turn.previousEnd + response_delay_sec + jitter_sec
        return window_start, max(0.0, turn.segment.start - response_margin_sec)

    window_start = max(
        turn.segment.end + response_delay_sec + jitter_sec,
        next_available_start,
    )
    window_end = None if turn.nextStart is None else turn.nextStart - response_margin_sec
    return window_start, window_end


def find_available_response_start(
    window_start: float,
    window_end: float | None,
    duration_sec: float,
    existing_responses: list[GeneratedResponse],
    margin_sec: float,
    prefer_latest: bool,
) -> float | None:
    if duration_sec <= 0.0:
        return None
    if window_end is not None and window_end - window_start < duration_sec:
        return None

    relevant_responses = sorted(
        existing_responses,
        key=lambda response: response.responseStart,
    )
    if prefer_latest:
        if window_end is None:
            return None
        candidate_start = window_end - duration_sec
        for response in reversed(relevant_responses):
            if response.responseEnd + margin_sec <= window_start:
                continue
            if response.responseStart - margin_sec >= candidate_start + duration_sec:
                continue
            if (
                candidate_start < response.responseEnd + margin_sec
                and candidate_start + duration_sec > response.responseStart - margin_sec
            ):
                candidate_start = response.responseStart - margin_sec - duration_sec
        if candidate_start >= window_start:
            return candidate_start
        return None

    candidate_start = window_start
    for response in relevant_responses:
        if response.responseEnd + margin_sec <= candidate_start:
            continue
        if (
            window_end is not None
            and response.responseStart - margin_sec >= window_end
        ):
            break
        if candidate_start + duration_sec <= response.responseStart - margin_sec:
            return candidate_start
        if candidate_start < response.responseEnd + margin_sec:
            candidate_start = response.responseEnd + margin_sec

    if window_end is None or candidate_start + duration_sec <= window_end:
        return candidate_start
    return None


def place_response_for_kind(
    response_kind: str,
    turn: ConversationTurn,
    tts_duration_sec: float,
    response_delay_sec: float,
    response_margin_sec: float,
    next_available_start: float,
    timing_jitter_sec: float,
    existing_responses: list[GeneratedResponse],
) -> ResponsePlacement | None:
    window_start, window_end = response_window_for_kind(
        response_kind,
        turn,
        response_delay_sec,
        response_margin_sec,
        next_available_start,
        timing_jitter_sec,
    )
    start_sec = find_available_response_start(
        window_start,
        window_end,
        tts_duration_sec,
        existing_responses,
        response_margin_sec,
        response_kind == "pre_question",
    )
    if start_sec is None:
        return None
    return ResponsePlacement(start=start_sec, end=start_sec + tts_duration_sec)


def available_response_window_sec(
    response_kind: str,
    turn: ConversationTurn,
    response_delay_sec: float,
    response_margin_sec: float,
    next_available_start: float,
    timing_jitter_sec: float,
    existing_responses: list[GeneratedResponse],
) -> float:
    window_start, window_end = response_window_for_kind(
        response_kind,
        turn,
        response_delay_sec,
        response_margin_sec,
        next_available_start,
        timing_jitter_sec,
    )
    if window_end is None:
        return 999.0
    best_duration = 0.0
    cursor = window_start
    for response in sorted(existing_responses, key=lambda item: item.responseStart):
        if response.responseEnd + response_margin_sec <= cursor:
            continue
        if response.responseStart - response_margin_sec >= window_end:
            break
        best_duration = max(
            best_duration,
            max(0.0, response.responseStart - response_margin_sec - cursor),
        )
        cursor = max(cursor, response.responseEnd + response_margin_sec)
    best_duration = max(best_duration, max(0.0, window_end - cursor))
    return best_duration


def tighten_char_limits_for_placement(
    allowed_kinds: list[str],
    turn: ConversationTurn,
    current_limits: dict[str, int],
    response_delay_sec: float,
    response_margin_sec: float,
    next_available_start: float,
    timing_jitter_sec: float,
    existing_responses: list[GeneratedResponse],
    tts_chars_per_sec: float,
    tts_speed: float,
) -> dict[str, int]:
    tightened: dict[str, int] = {}
    for response_kind in allowed_kinds:
        available_sec = available_response_window_sec(
            response_kind,
            turn,
            response_delay_sec,
            response_margin_sec,
            next_available_start,
            timing_jitter_sec,
            existing_responses,
        )
        fitted_chars = int(
            max(0.0, available_sec - 0.15)
            * tts_chars_per_sec
            * max(0.1, tts_speed)
        )
        if fitted_chars <= 0:
            continue
        tightened[response_kind] = max(4, min(current_limits[response_kind], fitted_chars))
    return tightened


def response_audio_cache_key(
    config: EnvConfig,
    text: str,
    tts_speed: float,
) -> str:
    key = f"{config.ttsUrl}\0{config.ttsType}\0{tts_speed:.4f}\0{text}".encode(
        "utf-8"
    )
    return hashlib.blake2b(key, digest_size=5).hexdigest()


def tts_paths_for_text(
    tts_root: Path,
    stem: str,
    config: EnvConfig,
    text: str,
    tts_speed: float,
) -> tuple[Path, Path]:
    cache_key = response_audio_cache_key(config, text, tts_speed)
    tts_path = tts_root / f"{stem}_{cache_key}.wav"
    raw_tts_path = tts_root / "raw" / f"{stem}_{cache_key}.wav"
    return tts_path, raw_tts_path


def balance_fill_enabled(balance_fill_mode: str, interaction_mode: str) -> bool:
    if balance_fill_mode == "off":
        return False
    if balance_fill_mode == "always":
        return True
    return interaction_mode == "auto"


def balance_fill_index(segment_index: int, fill_index: int) -> int:
    return BALANCE_FILL_INDEX_BASE + segment_index * 1000 + fill_index


def semantic_gap_index(segment_index: int, insert_index: int) -> int:
    return SEMANTIC_GAP_INDEX_BASE + segment_index * 1000 + insert_index


def response_belongs_to_turns(
    response: GeneratedResponse,
    turn_indexes: set[int],
) -> bool:
    if response.index in turn_indexes:
        return True
    if response.index >= SEMANTIC_GAP_INDEX_BASE:
        segment_index = (response.index - SEMANTIC_GAP_INDEX_BASE) // 1000
        return segment_index in turn_indexes
    if response.index < BALANCE_FILL_INDEX_BASE:
        return False
    segment_index = (response.index - BALANCE_FILL_INDEX_BASE) // 1000
    return segment_index in turn_indexes


def semantic_gap_insertions_enabled(mode: str) -> bool:
    return mode != "off"


def semantic_gap_insert_at(response: GeneratedResponse) -> float:
    return response.promptEnd


def semantic_gap_duration_sec(
    response: GeneratedResponse,
    response_delay_sec: float,
    response_margin_sec: float,
) -> float:
    return (
        response_delay_sec
        + max(0.0, response.responseEnd - response.responseStart)
        + response_margin_sec
    )


def timeline_insertions_from_responses(
    responses: list[GeneratedResponse],
    response_delay_sec: float,
    response_margin_sec: float,
) -> list[TimelineInsertion]:
    insertions: list[TimelineInsertion] = []
    for response in responses:
        if response.responseKind != SEMANTIC_GAP_RESPONSE_KIND:
            continue
        insert_at = semantic_gap_insert_at(response)
        insertions.append(
            TimelineInsertion(
                responseIndex=response.index,
                insertAt=insert_at,
                gapSec=semantic_gap_duration_sec(
                    response,
                    response_delay_sec,
                    response_margin_sec,
                ),
                responseOffsetSec=max(0.0, response.responseStart - insert_at),
            )
        )
    return sorted(insertions, key=lambda item: item.insertAt)


def shift_at_time(
    time_sec: float,
    insertions: list[TimelineInsertion],
    include_current: bool = True,
) -> float:
    shift_sec = 0.0
    for insertion in insertions:
        if insertion.insertAt < time_sec or (
            include_current and abs(insertion.insertAt - time_sec) < 1e-6
        ):
            shift_sec += insertion.gapSec
    return shift_sec


def shift_generated_response_for_timeline(
    response: GeneratedResponse,
    insertions: list[TimelineInsertion],
) -> GeneratedResponse:
    if response.responseKind == SEMANTIC_GAP_RESPONSE_KIND:
        insertion = next(
            item for item in insertions if item.responseIndex == response.index
        )
        shift_sec = shift_at_time(insertion.insertAt, insertions, False)
        response_start = insertion.insertAt + shift_sec + insertion.responseOffsetSec
        response_end = response_start + max(0.0, response.responseEnd - response.responseStart)
        return GeneratedResponse(
            index=response.index,
            promptStart=response.promptStart + shift_at_time(
                response.promptStart,
                insertions,
                True,
            ),
            promptEnd=insertion.insertAt + shift_sec,
            promptText=response.promptText,
            responseText=response.responseText,
            responseKind=response.responseKind,
            responseStart=response_start,
            responseEnd=response_end,
            ttsPath=response.ttsPath,
        )

    return GeneratedResponse(
        index=response.index,
        promptStart=response.promptStart + shift_at_time(
            response.promptStart,
            insertions,
            True,
        ),
        promptEnd=response.promptEnd + shift_at_time(
            response.promptEnd,
            insertions,
            False,
        ),
        promptText=response.promptText,
        responseText=response.responseText,
        responseKind=response.responseKind,
        responseStart=response.responseStart + shift_at_time(
            response.responseStart,
            insertions,
            True,
        ),
        responseEnd=response.responseEnd + shift_at_time(
            response.responseEnd,
            insertions,
            False,
        ),
        ttsPath=response.ttsPath,
    )


def shift_responses_for_timeline(
    responses: list[GeneratedResponse],
    response_delay_sec: float,
    response_margin_sec: float,
) -> list[GeneratedResponse]:
    insertions = timeline_insertions_from_responses(
        responses,
        response_delay_sec,
        response_margin_sec,
    )
    if not insertions:
        return responses
    return [
        shift_generated_response_for_timeline(response, insertions)
        for response in responses
    ]


def shift_transcript_for_timeline(
    transcript: list[TranscriptSegment],
    responses: list[GeneratedResponse],
    response_delay_sec: float,
    response_margin_sec: float,
) -> list[TranscriptSegment]:
    insertions = timeline_insertions_from_responses(
        responses,
        response_delay_sec,
        response_margin_sec,
    )
    if not insertions:
        return transcript
    shifted: list[TranscriptSegment] = []
    for segment in transcript:
        shifted.append(
            TranscriptSegment(
                index=segment.index,
                start=segment.start + shift_at_time(segment.start, insertions, True),
                end=segment.end + shift_at_time(segment.end, insertions, False),
                text=segment.text,
            )
        )
    return shifted


def speech_seconds_for_turns(turns: list[ConversationTurn]) -> float:
    return sum(max(0.0, turn.segment.end - turn.segment.start) for turn in turns)


def speech_seconds_for_responses(responses: list[GeneratedResponse]) -> float:
    return sum(max(0.0, item.responseEnd - item.responseStart) for item in responses)


def right_speech_ratio(
    left_speech_sec: float,
    right_speech_sec: float,
) -> float:
    total = left_speech_sec + right_speech_sec
    if total <= 0.0:
        return 0.0
    return right_speech_sec / total


def response_interval_overlaps(
    start_sec: float,
    end_sec: float,
    responses: list[GeneratedResponse],
    margin_sec: float,
) -> bool:
    for response in responses:
        if start_sec < response.responseEnd + margin_sec and end_sec > response.responseStart - margin_sec:
            return True
    return False


def response_overlaps_left_channel(
    response: GeneratedResponse,
    turns: list[ConversationTurn],
    margin_sec: float = 0.0,
) -> bool:
    for turn in turns:
        if (
            response.responseStart < turn.segment.end + margin_sec
            and response.responseEnd > turn.segment.start - margin_sec
        ):
            return True
    return False


def cached_response_is_usable(
    response: GeneratedResponse,
    turns: list[ConversationTurn],
    expected_prompt: str,
    allowed_kinds: list[str],
    existing_responses: list[GeneratedResponse],
    response_margin_sec: float,
) -> bool:
    if response.promptText != expected_prompt or response.responseKind not in allowed_kinds:
        return False
    if not Path(response.ttsPath).exists():
        return False
    if response_overlaps_left_channel(response, turns, response_margin_sec):
        return False
    other_responses = [
        item for item in existing_responses if item.index != response.index
    ]
    return not response_interval_overlaps(
        response.responseStart,
        response.responseEnd,
        other_responses,
        response_margin_sec,
    )


def remove_left_overlapping_balance_fills(
    generated_by_index: dict[int, GeneratedResponse],
    turns: list[ConversationTurn],
) -> None:
    for index, response in list(generated_by_index.items()):
        if response.responseKind != "balance_fill_llm":
            continue
        if response_overlaps_left_channel(response, turns):
            generated_by_index.pop(index, None)


def remove_conflicting_balance_fills(
    generated_by_index: dict[int, GeneratedResponse],
    response_margin_sec: float,
) -> None:
    anchored_responses = [
        response
        for response in generated_by_index.values()
        if response.responseKind != "balance_fill_llm"
    ]
    kept_balance_fills: list[GeneratedResponse] = []
    balance_items = sorted(
        [
            (index, response)
            for index, response in generated_by_index.items()
            if response.responseKind == "balance_fill_llm"
        ],
        key=lambda item: item[1].responseStart,
    )
    for index, response in balance_items:
        if response_interval_overlaps(
            response.responseStart,
            response.responseEnd,
            anchored_responses + kept_balance_fills,
            response_margin_sec,
        ):
            generated_by_index.pop(index, None)
            continue
        kept_balance_fills.append(response)


def build_balance_fill_candidates(
    turns: list[ConversationTurn],
    response_delay_sec: float,
    response_margin_sec: float,
    min_insert_gap_sec: float,
    long_turn_fill_sec: float,
    fill_interval_sec: float,
    allow_left_overlap_fill: bool,
) -> list[BalanceFillCandidate]:
    candidates: list[BalanceFillCandidate] = []
    interval_sec = max(min_insert_gap_sec, fill_interval_sec)
    for index, turn in enumerate(turns[:-1]):
        next_turn = turns[index + 1]
        window_start = turn.segment.end + response_delay_sec
        window_end = next_turn.segment.start - response_margin_sec
        if window_end - window_start < min_insert_gap_sec:
            continue
        fill_index = 0
        cursor_sec = window_start
        while window_end - cursor_sec >= min_insert_gap_sec:
            candidates.append(
                BalanceFillCandidate(
                    turn=turn,
                    nextTurn=next_turn,
                    fillIndex=fill_index,
                    preferredStart=cursor_sec,
                    windowStart=cursor_sec,
                    windowEnd=window_end,
                    placementKind="gap_bridge",
                )
            )
            fill_index += 1
            cursor_sec += interval_sec

    if not allow_left_overlap_fill:
        return candidates

    for turn in turns:
        duration_sec = turn.segment.end - turn.segment.start
        if duration_sec < long_turn_fill_sec:
            continue
        fill_index = 500
        cursor_sec = turn.segment.start + fill_interval_sec
        latest_start_sec = turn.segment.end - response_margin_sec
        while cursor_sec < latest_start_sec:
            candidates.append(
                BalanceFillCandidate(
                    turn=turn,
                    nextTurn=None,
                    fillIndex=fill_index,
                    preferredStart=cursor_sec,
                    windowStart=cursor_sec,
                    windowEnd=latest_start_sec,
                    placementKind="left_overlap",
                )
            )
            fill_index += 1
            cursor_sec += fill_interval_sec
    return candidates


def transcript_segments_for_turn(
    transcript: list[TranscriptSegment],
    turn: ConversationTurn,
) -> list[TranscriptSegment]:
    return [
        segment
        for segment in transcript
        if segment.start >= turn.segment.start - 0.05
        and segment.end <= turn.segment.end + 0.05
    ]


def existing_semantic_gap_times(
    responses: list[GeneratedResponse],
) -> list[float]:
    return [
        semantic_gap_insert_at(response)
        for response in responses
        if response.responseKind == SEMANTIC_GAP_RESPONSE_KIND
    ]


def build_semantic_gap_points(
    turns: list[ConversationTurn],
    transcript: list[TranscriptSegment],
    responses: list[GeneratedResponse],
    min_turn_sec: float,
    fill_interval_sec: float,
    max_points: int = 18,
) -> list[SemanticGapPoint]:
    existing_times = existing_semantic_gap_times(responses)
    points: list[SemanticGapPoint] = []
    min_edge_sec = 0.75
    min_distance_sec = max(2.0, fill_interval_sec * 0.8)
    for turn in turns:
        turn_duration_sec = turn.segment.end - turn.segment.start
        if turn_duration_sec < min_turn_sec:
            continue
        segments = transcript_segments_for_turn(transcript, turn)
        for segment_index, segment in enumerate(segments[:-1]):
            insert_at = segment.end
            if insert_at - turn.segment.start < min_edge_sec:
                continue
            if turn.segment.end - insert_at < min_edge_sec:
                continue
            if any(abs(insert_at - time_sec) < min_distance_sec for time_sec in existing_times):
                continue
            next_segment = segments[segment_index + 1]
            point_id = f"{turn.segment.index}:{segment_index}"
            points.append(
                SemanticGapPoint(
                    pointId=point_id,
                    turn=turn,
                    insertAt=insert_at,
                    beforeText=segment.text,
                    afterText=next_segment.text,
                )
            )

        if len(segments) > 1:
            continue
        insert_at = turn.segment.start + turn_duration_sec * 0.55
        if any(abs(insert_at - time_sec) < min_distance_sec for time_sec in existing_times):
            continue
        point_id = f"{turn.segment.index}:fallback"
        points.append(
            SemanticGapPoint(
                pointId=point_id,
                turn=turn,
                insertAt=insert_at,
                beforeText=turn.segment.text,
                afterText="この後も同じ話題が続く",
            )
        )

    return points[:max_points]


def next_semantic_gap_insert_index(
    generated_by_index: dict[int, GeneratedResponse],
    segment_index: int,
) -> int:
    used_indexes = set(generated_by_index)
    insert_index = 0
    while semantic_gap_index(segment_index, insert_index) in used_indexes:
        insert_index += 1
    return insert_index


def semantic_gap_should_run(
    mode: str,
    current_ratio: float,
    target_right_ratio: float,
    max_right_ratio: float,
) -> bool:
    if mode == "off":
        return False
    if mode == "always":
        return current_ratio < max_right_ratio
    return current_ratio < target_right_ratio


def add_semantic_gap_insertions(
    client: OpenAI,
    config: EnvConfig,
    turns: list[ConversationTurn],
    transcript: list[TranscriptSegment],
    generated_by_index: dict[int, GeneratedResponse],
    responses_cache_path: Path,
    tts_root: Path,
    sample_rate: int,
    tts_timeout: float,
    tts_speed: float,
    response_delay_sec: float,
    response_margin_sec: float,
    target_right_ratio: float,
    max_right_ratio: float,
    fill_interval_sec: float,
    semantic_gap_mode: str,
    semantic_gap_max_count: int,
    semantic_gap_min_turn_sec: float,
    semantic_gap_max_chars: int,
    llm_timeout: float,
) -> dict[str, float]:
    left_speech_sec = speech_seconds_for_turns(turns)
    right_speech_sec = speech_seconds_for_responses(list(generated_by_index.values()))
    initial_ratio = right_speech_ratio(left_speech_sec, right_speech_sec)
    if semantic_gap_max_count <= 0 or not semantic_gap_should_run(
        semantic_gap_mode,
        initial_ratio,
        target_right_ratio,
        max_right_ratio,
    ):
        return {
            "leftSpeechSec": left_speech_sec,
            "rightSpeechSec": right_speech_sec,
            "rightRatio": initial_ratio,
            "addedGapCount": 0,
            "insertedGapSec": 0.0,
        }

    added_count = 0
    inserted_gap_sec = 0.0
    with create_progress() as progress:
        task = progress.add_task(
            "Inserting semantic conversation gaps",
            total=semantic_gap_max_count,
        )
        while added_count < semantic_gap_max_count:
            current_ratio = right_speech_ratio(left_speech_sec, right_speech_sec)
            if not semantic_gap_should_run(
                semantic_gap_mode,
                current_ratio,
                target_right_ratio,
                max_right_ratio,
            ):
                break
            points = build_semantic_gap_points(
                turns,
                transcript,
                list(generated_by_index.values()),
                semantic_gap_min_turn_sec,
                fill_interval_sec,
            )
            if not points:
                break

            avoid_texts = recent_response_texts(list(generated_by_index.values()))
            choice = request_semantic_gap_choice_checked(
                client,
                config,
                transcript,
                points,
                semantic_gap_max_chars,
                llm_timeout,
                avoid_texts,
            )
            point_by_id = {point.pointId: point for point in points}
            point = point_by_id[choice.pointId]
            tts_path, raw_tts_path = tts_paths_for_text(
                tts_root,
                f"semanticGap{point.turn.segment.index:04d}_{added_count:02d}",
                config,
                choice.responseText,
                tts_speed,
            )
            synthesize_tts_with_speed(
                config,
                choice.responseText,
                raw_tts_path,
                tts_path,
                tts_timeout,
                tts_speed,
            )
            tts_audio = load_mono_audio(tts_path, sample_rate)
            tts_duration_sec = tts_audio.shape[-1] / sample_rate
            new_right_speech_sec = right_speech_sec + tts_duration_sec
            new_ratio = right_speech_ratio(left_speech_sec, new_right_speech_sec)
            if new_ratio > max_right_ratio:
                progress.advance(task)
                break

            insert_index = next_semantic_gap_insert_index(
                generated_by_index,
                point.turn.segment.index,
            )
            response_index = semantic_gap_index(
                point.turn.segment.index,
                insert_index,
            )
            response_start = point.insertAt + response_delay_sec
            response_end = response_start + tts_duration_sec
            generated_by_index[response_index] = GeneratedResponse(
                index=response_index,
                promptStart=point.turn.segment.start,
                promptEnd=point.insertAt,
                promptText=point.turn.segment.text,
                responseText=choice.responseText,
                responseKind=SEMANTIC_GAP_RESPONSE_KIND,
                responseStart=response_start,
                responseEnd=response_end,
                ttsPath=str(tts_path),
            )
            gap_sec = tts_duration_sec + response_delay_sec + response_margin_sec
            right_speech_sec = new_right_speech_sec
            inserted_gap_sec += gap_sec
            added_count += 1
            save_cached_responses(
                responses_cache_path,
                [generated_by_index[index] for index in sorted(generated_by_index)],
            )
            console.print(
                f"[cyan]Semantic gap[/cyan] {point.pointId} "
                f"{point.insertAt:.2f}s +{gap_sec:.2f}s -> {choice.responseText}"
            )
            progress.advance(task)

    return {
        "leftSpeechSec": left_speech_sec,
        "rightSpeechSec": right_speech_sec,
        "rightRatio": right_speech_ratio(left_speech_sec, right_speech_sec),
        "addedGapCount": added_count,
        "insertedGapSec": inserted_gap_sec,
    }


def add_balance_fill_responses(
    client: OpenAI,
    config: EnvConfig,
    turns: list[ConversationTurn],
    transcript: list[TranscriptSegment],
    generated_by_index: dict[int, GeneratedResponse],
    responses_cache_path: Path,
    tts_root: Path,
    sample_rate: int,
    tts_timeout: float,
    tts_speed: float,
    response_delay_sec: float,
    response_margin_sec: float,
    min_insert_gap_sec: float,
    target_right_ratio: float,
    max_right_ratio: float,
    long_turn_fill_sec: float,
    fill_interval_sec: float,
    fill_max_chars: int,
    tts_chars_per_sec: float,
    allow_left_overlap_fill: bool,
    llm_timeout: float,
) -> dict[str, float]:
    if not allow_left_overlap_fill:
        remove_left_overlapping_balance_fills(generated_by_index, turns)
    left_speech_sec = speech_seconds_for_turns(turns)
    right_speech_sec = speech_seconds_for_responses(list(generated_by_index.values()))
    initial_ratio = right_speech_ratio(left_speech_sec, right_speech_sec)
    if target_right_ratio <= 0.0 or initial_ratio >= target_right_ratio:
        return {
            "leftSpeechSec": left_speech_sec,
            "rightSpeechSec": right_speech_sec,
            "rightRatio": initial_ratio,
            "addedFillCount": 0,
        }

    added_count = 0
    candidates = build_balance_fill_candidates(
        turns,
        response_delay_sec,
        response_margin_sec,
        min_insert_gap_sec,
        long_turn_fill_sec,
        fill_interval_sec,
        allow_left_overlap_fill,
    )

    if not candidates:
        return {
            "leftSpeechSec": left_speech_sec,
            "rightSpeechSec": right_speech_sec,
            "rightRatio": initial_ratio,
            "addedFillCount": 0,
        }

    with create_progress() as progress:
        task = progress.add_task("Balancing right-channel speech", total=len(candidates))
        for candidate in candidates:
            current_ratio = right_speech_ratio(left_speech_sec, right_speech_sec)
            if current_ratio >= target_right_ratio:
                break

            response_index = balance_fill_index(
                candidate.turn.segment.index,
                candidate.fillIndex,
            )
            cached_response = generated_by_index.get(response_index)
            if (
                cached_response is not None
                and cached_response.responseKind == "balance_fill_llm"
                and Path(cached_response.ttsPath).exists()
                and not (
                    not allow_left_overlap_fill
                    and response_overlaps_left_channel(cached_response, turns)
                )
            ):
                progress.advance(task)
                continue
            if cached_response is not None:
                generated_by_index.pop(response_index, None)

            window_duration_sec = (
                fill_interval_sec
                if candidate.windowEnd is None
                else max(0.0, candidate.windowEnd - candidate.windowStart)
            )
            candidate_max_chars = min(
                fill_max_chars,
                max(
                    4,
                    int(
                        max(0.0, window_duration_sec - response_margin_sec)
                        * tts_chars_per_sec
                        * max(0.1, tts_speed)
                    ),
                ),
            )
            avoid_texts = recent_response_texts(list(generated_by_index.values()))
            text = request_balance_fill_completion_checked(
                client,
                config,
                transcript,
                candidate,
                candidate_max_chars,
                llm_timeout,
                avoid_texts,
            )
            tts_path, raw_tts_path = tts_paths_for_text(
                tts_root,
                f"fill{candidate.turn.segment.index:04d}_{candidate.fillIndex:02d}",
                config,
                text,
                tts_speed,
            )
            synthesize_tts_with_speed(
                config,
                text,
                raw_tts_path,
                tts_path,
                tts_timeout,
                tts_speed,
            )
            tts_audio = load_mono_audio(tts_path, sample_rate)
            tts_duration_sec = tts_audio.shape[-1] / sample_rate
            adjusted_start_sec = find_available_response_start(
                candidate.windowStart,
                candidate.windowEnd,
                tts_duration_sec,
                list(generated_by_index.values()),
                response_margin_sec,
                False,
            )
            if adjusted_start_sec is None:
                progress.advance(task)
                continue
            end_sec = adjusted_start_sec + tts_duration_sec
            if (
                not allow_left_overlap_fill
                and response_overlaps_left_channel(
                    GeneratedResponse(
                        index=response_index,
                        promptStart=candidate.turn.segment.start,
                        promptEnd=candidate.turn.segment.end,
                        promptText=candidate.turn.segment.text,
                        responseText=text,
                        responseKind="balance_fill_llm",
                        responseStart=adjusted_start_sec,
                        responseEnd=end_sec,
                        ttsPath=str(tts_path),
                    ),
                    turns,
                )
            ):
                progress.advance(task)
                continue
            if response_interval_overlaps(
                adjusted_start_sec,
                end_sec,
                list(generated_by_index.values()),
                response_margin_sec,
            ):
                progress.advance(task)
                continue

            new_right_speech_sec = right_speech_sec + tts_duration_sec
            new_ratio = right_speech_ratio(left_speech_sec, new_right_speech_sec)
            if new_ratio > max_right_ratio:
                progress.advance(task)
                continue

            generated_by_index[response_index] = GeneratedResponse(
                index=response_index,
                promptStart=candidate.turn.segment.start,
                promptEnd=candidate.turn.segment.end,
                promptText=candidate.turn.segment.text,
                responseText=text,
                responseKind="balance_fill_llm",
                responseStart=adjusted_start_sec,
                responseEnd=end_sec,
                ttsPath=str(tts_path),
            )
            right_speech_sec = new_right_speech_sec
            added_count += 1
            save_cached_responses(
                responses_cache_path,
                [generated_by_index[index] for index in sorted(generated_by_index)],
            )
            console.print(
                f"[cyan]Balance fill[/cyan] "
                f"{candidate.turn.segment.index}:{candidate.fillIndex} "
                f"{adjusted_start_sec:.2f}s -> {text}"
            )
            progress.advance(task)

    return {
        "leftSpeechSec": left_speech_sec,
        "rightSpeechSec": right_speech_sec,
        "rightRatio": right_speech_ratio(left_speech_sec, right_speech_sec),
        "addedFillCount": added_count,
    }


def add_audio_clip(
    target: torch.Tensor,
    clip_path: Path,
    start_sec: float,
    sample_rate: int,
) -> None:
    clip = load_mono_audio(clip_path, sample_rate)
    start_frame = max(0, int(start_sec * sample_rate))
    end_frame = start_frame + clip.shape[-1]
    existing = target[:, start_frame:end_frame]
    target[:, start_frame:end_frame] = torch.clamp(existing + clip, min=-1.0, max=1.0)


def build_stereo_audio(
    input_audio: Path,
    generated_responses: list[GeneratedResponse],
    sample_rate: int,
    response_delay_sec: float = 0.0,
    response_margin_sec: float = 0.0,
) -> torch.Tensor:
    left = load_mono_audio(input_audio, sample_rate)
    insertions = timeline_insertions_from_responses(
        generated_responses,
        response_delay_sec,
        response_margin_sec,
    )
    if insertions:
        insertion_frames = [
            (
                min(left.shape[-1], max(0, int(insertion.insertAt * sample_rate))),
                max(0, math.ceil(insertion.gapSec * sample_rate)),
            )
            for insertion in insertions
        ]
        total_gap_frames = sum(gap_frames for _, gap_frames in insertion_frames)
        shifted_left = torch.zeros(
            1,
            left.shape[-1] + total_gap_frames,
            dtype=left.dtype,
        )
        input_cursor = 0
        output_cursor = 0
        for insert_frame, gap_frames in sorted(insertion_frames):
            insert_frame = max(input_cursor, insert_frame)
            segment_frames = insert_frame - input_cursor
            if segment_frames > 0:
                shifted_left[
                    :,
                    output_cursor : output_cursor + segment_frames,
                ] = left[:, input_cursor:insert_frame]
                output_cursor += segment_frames
            output_cursor += gap_frames
            input_cursor = insert_frame
        if input_cursor < left.shape[-1]:
            remaining = left.shape[-1] - input_cursor
            shifted_left[:, output_cursor : output_cursor + remaining] = left[
                :,
                input_cursor:,
            ]
        left = shifted_left

    shifted_responses = shift_responses_for_timeline(
        generated_responses,
        response_delay_sec,
        response_margin_sec,
    )
    max_frames = left.shape[-1]
    for response in shifted_responses:
        max_frames = max(max_frames, math.ceil(response.responseEnd * sample_rate) + 1)

    left = torch.nn.functional.pad(left, (0, max_frames - left.shape[-1]))
    right = torch.zeros(1, max_frames, dtype=left.dtype)
    for response in shifted_responses:
        add_audio_clip(
            right,
            Path(response.ttsPath),
            response.responseStart,
            sample_rate,
        )
    return torch.cat([left, right], dim=0)


def main() -> None:
    args = parse_args()
    config = load_env_config(args.env_file)
    openai_client = create_openai_client(config)
    metadata_output = args.metadata_output or args.output.with_name(
        f"{args.output.stem}.responses.json"
    )
    cache_root = args.cache_dir / args.output.stem
    transcript_cache_path = cache_root / "transcript.json"
    responses_cache_path = cache_root / "responses.json"
    tts_root = args.keep_tts_dir or cache_root / "tts"
    tts_root.mkdir(parents=True, exist_ok=True)

    transcript = load_or_create_transcript(
        transcript_cache_path,
        lambda: transcribe_audio(
            args.input,
            args.language,
            args.whisper_model,
            args.min_segment_sec,
            None,
        ),
        args.refresh_transcript,
    )
    if args.max_segments is not None:
        transcript = transcript[: args.max_segments]
    if not transcript:
        raise RuntimeError("No transcript segments were produced from the input audio.")

    turns = build_conversation_turns(transcript, args.merge_gap_sec)
    cached_responses = load_cached_responses(
        responses_cache_path,
        args.refresh_responses,
    )
    generated_by_index: dict[int, GeneratedResponse] = dict(cached_responses)
    next_available_start = 0.0
    with create_progress() as progress:
        task = progress.add_task("Generating LLM replies and TTS", total=len(turns))
        for turn in turns:
            segment = turn.segment
            progress.update(task, description=f"LLM/TTS turn {segment.index + 1}")
            time_until_next = (
                None if turn.nextStart is None else max(0.0, turn.nextStart - segment.end)
            )
            time_since_previous = (
                None
                if turn.previousEnd is None
                else max(0.0, segment.start - turn.previousEnd)
            )
            reply_chars = response_char_limit_for_turn(
                turn,
                args.min_response_chars,
                args.max_response_chars,
                args.response_delay_sec,
                args.response_margin_sec,
                args.short_gap_sec,
                args.long_gap_sec,
                args.tts_chars_per_sec,
                args.tts_speed,
            )
            pre_question_chars = pre_question_char_limit_for_turn(
                turn,
                args.min_response_chars,
                args.max_response_chars,
                args.response_delay_sec,
                args.response_margin_sec,
                args.short_gap_sec,
                args.long_gap_sec,
                args.tts_chars_per_sec,
                args.tts_speed,
            )
            allowed_kinds = allowed_interaction_kinds(
                args.interaction_mode,
                turn,
                args.min_insert_gap_sec,
                args.pre_question_gap_sec,
            )
            cached_response = generated_by_index.get(segment.index)
            expected_prompt = segment.text
            existing_responses = responses_before_turn(
                list(generated_by_index.values()),
                turn,
                segment.index,
            )
            if not allowed_kinds:
                generated_by_index.pop(segment.index, None)
                console.print(
                    f"[yellow]Skipping[/yellow] turn {segment.index}: "
                    "no safe gap for a right-channel response"
                )
                progress.advance(task)
                continue
            if cached_response is not None and Path(cached_response.ttsPath).exists():
                if cached_response_is_usable(
                    cached_response,
                    turns,
                    expected_prompt,
                    allowed_kinds,
                    existing_responses,
                    args.response_margin_sec,
                ):
                    console.print(f"[cyan]Using cached response[/cyan] turn {segment.index}")
                    if cached_response.responseKind == "reply":
                        next_available_start = max(
                            next_available_start,
                            cached_response.responseEnd + args.response_delay_sec,
                        )
                    progress.advance(task)
                    continue
                generated_by_index.pop(segment.index, None)

            max_chars_by_kind = {
                "reply": reply_chars,
                "pre_question": pre_question_chars,
            }
            generated_response: GeneratedResponse | None = None
            attempt_allowed_kinds = list(allowed_kinds)
            attempt_limits = dict(max_chars_by_kind)
            attempts = max(1, args.response_placement_retries + 1)
            for attempt_index in range(attempts):
                avoid_texts = recent_response_texts(existing_responses)
                response_kind, response_text = request_interaction_completion_checked(
                    openai_client,
                    config,
                    [turn.segment for turn in turns],
                    segment,
                    attempt_allowed_kinds,
                    attempt_limits,
                    args.llm_timeout,
                    time_until_next,
                    time_since_previous,
                    avoid_texts,
                )
                tts_path, raw_tts_path = tts_paths_for_text(
                    tts_root,
                    f"response{segment.index:04d}",
                    config,
                    response_text,
                    args.tts_speed,
                )
                synthesize_tts_with_speed(
                    config,
                    response_text,
                    raw_tts_path,
                    tts_path,
                    args.tts_timeout,
                    args.tts_speed,
                )
                tts_audio = load_mono_audio(tts_path, args.sample_rate)
                tts_duration_sec = tts_audio.shape[-1] / args.sample_rate
                placement = place_response_for_kind(
                    response_kind,
                    turn,
                    tts_duration_sec,
                    args.response_delay_sec,
                    args.response_margin_sec,
                    next_available_start,
                    args.timing_jitter_sec,
                    existing_responses,
                )
                if placement is not None:
                    generated_response = GeneratedResponse(
                        index=segment.index,
                        promptStart=segment.start,
                        promptEnd=segment.end,
                        promptText=segment.text,
                        responseText=response_text,
                        responseKind=response_kind,
                        responseStart=placement.start,
                        responseEnd=placement.end,
                        ttsPath=str(tts_path),
                    )
                    break
                attempt_limits = tighten_char_limits_for_placement(
                    attempt_allowed_kinds,
                    turn,
                    attempt_limits,
                    args.response_delay_sec,
                    args.response_margin_sec,
                    next_available_start,
                    args.timing_jitter_sec,
                    existing_responses,
                    args.tts_chars_per_sec,
                    args.tts_speed,
                )
                attempt_allowed_kinds = [
                    kind for kind in attempt_allowed_kinds if kind in attempt_limits
                ]
                if not attempt_allowed_kinds or attempt_index + 1 >= attempts:
                    break

            if generated_response is None:
                console.print(
                    f"[yellow]Skipping[/yellow] turn {segment.index}: "
                    "TTS response did not fit a safe gap"
                )
                progress.advance(task)
                continue

            if generated_response.responseKind == "reply":
                next_available_start = (
                    generated_response.responseEnd + args.response_delay_sec
                )
            generated_by_index[segment.index] = generated_response
            save_cached_responses(
                responses_cache_path,
                [generated_by_index[index] for index in sorted(generated_by_index)],
            )
            console.print(
                f"[{segment.index:04d}] {generated_response.responseKind} | "
                f"{segment.text} -> {generated_response.responseText}"
            )
            progress.advance(task)

    turn_indexes = {turn.segment.index for turn in turns}
    include_balance_fills = balance_fill_enabled(
        args.balance_fill_mode,
        args.interaction_mode,
    )
    include_semantic_gap_insertions = semantic_gap_insertions_enabled(
        args.semantic_gap_insert_mode,
    )
    generated_by_index = {
        index: response
        for index, response in generated_by_index.items()
        if response_belongs_to_turns(response, turn_indexes)
        and response.responseKind != "balance_fill"
        and (include_balance_fills or response.responseKind != "balance_fill_llm")
        and (
            include_semantic_gap_insertions
            or response.responseKind != SEMANTIC_GAP_RESPONSE_KIND
        )
    }
    if not args.allow_left_overlap_fill:
        remove_left_overlapping_balance_fills(generated_by_index, turns)
    remove_conflicting_balance_fills(generated_by_index, args.response_margin_sec)
    balance_stats = {
        "leftSpeechSec": speech_seconds_for_turns(turns),
        "rightSpeechSec": speech_seconds_for_responses(list(generated_by_index.values())),
        "rightRatio": right_speech_ratio(
            speech_seconds_for_turns(turns),
            speech_seconds_for_responses(list(generated_by_index.values())),
        ),
        "addedFillCount": 0,
    }
    if include_balance_fills:
        balance_stats = add_balance_fill_responses(
            openai_client,
            config,
            turns,
            [turn.segment for turn in turns],
            generated_by_index,
            responses_cache_path,
            tts_root,
            args.sample_rate,
            args.tts_timeout,
            args.tts_speed,
            args.response_delay_sec,
            args.response_margin_sec,
            args.min_insert_gap_sec,
            args.target_right_ratio,
            args.max_right_ratio,
            args.long_turn_fill_sec,
            args.fill_interval_sec,
            args.fill_max_chars,
            args.tts_chars_per_sec,
            args.allow_left_overlap_fill,
            args.llm_timeout,
        )
    semantic_gap_stats = {
        "leftSpeechSec": speech_seconds_for_turns(turns),
        "rightSpeechSec": speech_seconds_for_responses(list(generated_by_index.values())),
        "rightRatio": right_speech_ratio(
            speech_seconds_for_turns(turns),
            speech_seconds_for_responses(list(generated_by_index.values())),
        ),
        "addedGapCount": 0,
        "insertedGapSec": 0.0,
    }
    if include_semantic_gap_insertions:
        semantic_gap_stats = add_semantic_gap_insertions(
            openai_client,
            config,
            turns,
            [turn.segment for turn in turns],
            generated_by_index,
            responses_cache_path,
            tts_root,
            args.sample_rate,
            args.tts_timeout,
            args.tts_speed,
            args.response_delay_sec,
            args.response_margin_sec,
            args.target_right_ratio,
            args.max_right_ratio,
            args.fill_interval_sec,
            args.semantic_gap_insert_mode,
            args.semantic_gap_max_count,
            args.semantic_gap_min_turn_sec,
            args.semantic_gap_max_chars,
            args.llm_timeout,
        )
    generated = [generated_by_index[index] for index in sorted(generated_by_index)]
    save_cached_responses(responses_cache_path, generated)
    rendered_responses = shift_responses_for_timeline(
        generated,
        args.response_delay_sec,
        args.response_margin_sec,
    )
    rendered_transcript = shift_transcript_for_timeline(
        transcript,
        generated,
        args.response_delay_sec,
        args.response_margin_sec,
    )
    with status(f"Writing stereo wav to [bold]{args.output}[/bold]"):
        stereo = build_stereo_audio(
            args.input,
            generated,
            args.sample_rate,
            args.response_delay_sec,
            args.response_margin_sec,
        )
        args.output.parent.mkdir(parents=True, exist_ok=True)
        torchaudio.save(args.output, stereo, args.sample_rate)

    metadata_output.parent.mkdir(parents=True, exist_ok=True)
    metadata = {
        "input": str(args.input),
        "output": str(args.output),
        "language": args.language,
        "whisperModel": args.whisper_model,
        "openaiBaseUrl": config.openaiBaseUrl,
        "openaiModel": config.openaiModel,
        "ttsUrl": config.ttsUrl,
        "ttsType": config.ttsType,
        "interactionMode": args.interaction_mode,
        "balanceFillMode": args.balance_fill_mode,
        "semanticGapInsertMode": args.semantic_gap_insert_mode,
        "semanticGapMaxCount": args.semantic_gap_max_count,
        "semanticGapMinTurnSec": args.semantic_gap_min_turn_sec,
        "semanticGapMaxChars": args.semantic_gap_max_chars,
        "targetRightRatio": args.target_right_ratio,
        "maxRightRatio": args.max_right_ratio,
        "fillMaxChars": args.fill_max_chars,
        "allowLeftOverlapFill": args.allow_left_overlap_fill,
        "timingJitterSec": args.timing_jitter_sec,
        "balanceStats": balance_stats,
        "semanticGapStats": semantic_gap_stats,
        "transcript": [segment.__dict__ for segment in rendered_transcript],
        "responses": [response.__dict__ for response in rendered_responses],
    }
    metadata_output.write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    console.print(f"[green]Wrote[/green] stereo wav to {args.output}")
    console.print(f"[green]Wrote[/green] metadata to {metadata_output}")


if __name__ == "__main__":
    main()

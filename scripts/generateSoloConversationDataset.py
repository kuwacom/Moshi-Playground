from __future__ import annotations

import argparse
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

from datasetPaths import datasetCacheDir
from progressUtils import console, create_progress, status


DEFAULT_OPENAI_MODEL = "anthropic/claude-sonnet-4.6"
DEFAULT_TTS_TYPE = "10"
BALANCE_FILL_INDEX_BASE = 10_000_000


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
            download_root="models/whisper",
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
        if can_pre_question and can_reply:
            return ["pre_question", "reply"]
        if can_pre_question:
            return ["pre_question"]
        return ["reply"]
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
    return parse_interaction_response(
        content,
        allowed_kinds,
        fallback_kind,
        max_chars_by_kind,
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
    turn: ConversationTurn,
    insert_sec: float,
    max_chars: int,
    timeout: float,
) -> str:
    context = build_recent_context(transcript, turn.segment.index)
    offset_sec = max(0.0, insert_sec - turn.segment.start)
    response = client.chat.completions.create(
        model=config.openaiModel,
        temperature=0.85,
        max_tokens=60,
        timeout=timeout,
        messages=[
            {
                "role": "system",
                "content": (
                    "あなたは一人配信を自然な双方向会話データへ整える編集者です。"
                    "配信者が長く話している途中に、右チャンネルへ短いリスナー音声を差し込みます。"
                    "話を遮らず、内容に軽く合う相槌、驚き、短い促しを作ってください。"
                    "質問で話題を変えないでください。括弧書き、絵文字、説明、メタ発言は禁止です。"
                ),
            },
            {
                "role": "user",
                "content": (
                    f"直近の文脈:\n{context}\n\n"
                    f"長い発話:\n{turn.segment.text}\n\n"
                    f"差し込み位置: この発話の開始から約{offset_sec:.1f}秒後\n"
                    f"{max_chars}文字以内で、自然な短いリスナー発話を1つだけ返してください。"
                ),
            },
        ],
    )
    content = response.choices[0].message.content
    if not content:
        raise RuntimeError(f"LLM returned an empty balance-fill response: {response}")
    return content.replace("\n", " ").strip()[:max_chars]


def request_balance_fill_completion_checked(
    client: OpenAI,
    config: EnvConfig,
    transcript: list[TranscriptSegment],
    turn: ConversationTurn,
    insert_sec: float,
    max_chars: int,
    timeout: float,
) -> str:
    try:
        return request_balance_fill_completion(
            client,
            config,
            transcript,
            turn,
            insert_sec,
            max_chars,
            timeout,
        )
    except APIStatusError as error:
        raise RuntimeError(
            f"LLM balance-fill request failed with HTTP {error.status_code}: "
            f"{error.response.text[:1000]}"
        ) from error
    except OpenAIError as error:
        raise RuntimeError(f"LLM balance-fill request failed: {error}") from error


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
) -> float:
    if response_kind == "pre_question":
        desired_start = max(
            0.0,
            turn.segment.start - response_margin_sec - tts_duration_sec,
        )
        if turn.previousEnd is None:
            return desired_start
        earliest_start = turn.previousEnd + response_delay_sec
        return max(earliest_start, desired_start)
    return max(
        turn.segment.end + response_delay_sec,
        next_available_start,
    )


def balance_fill_enabled(balance_fill_mode: str, interaction_mode: str) -> bool:
    if balance_fill_mode == "off":
        return False
    if balance_fill_mode == "always":
        return True
    return interaction_mode == "auto"


def balance_fill_index(segment_index: int, fill_index: int) -> int:
    return BALANCE_FILL_INDEX_BASE + segment_index * 1000 + fill_index


def response_belongs_to_turns(
    response: GeneratedResponse,
    turn_indexes: set[int],
) -> bool:
    if response.index in turn_indexes:
        return True
    if response.index < BALANCE_FILL_INDEX_BASE:
        return False
    segment_index = (response.index - BALANCE_FILL_INDEX_BASE) // 1000
    return segment_index in turn_indexes


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
    response_margin_sec: float,
    target_right_ratio: float,
    max_right_ratio: float,
    long_turn_fill_sec: float,
    fill_interval_sec: float,
    fill_max_chars: int,
    llm_timeout: float,
) -> dict[str, float]:
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
    candidates: list[tuple[ConversationTurn, int, float]] = []
    for turn in turns:
        duration_sec = turn.segment.end - turn.segment.start
        if duration_sec < long_turn_fill_sec:
            continue
        fill_index = 0
        cursor_sec = turn.segment.start + fill_interval_sec
        latest_start_sec = turn.segment.end - response_margin_sec
        while cursor_sec < latest_start_sec:
            candidates.append((turn, fill_index, cursor_sec))
            fill_index += 1
            cursor_sec += fill_interval_sec

    if not candidates:
        return {
            "leftSpeechSec": left_speech_sec,
            "rightSpeechSec": right_speech_sec,
            "rightRatio": initial_ratio,
            "addedFillCount": 0,
        }

    with create_progress() as progress:
        task = progress.add_task("Balancing right-channel speech", total=len(candidates))
        for candidate_index, (turn, fill_index, start_sec) in enumerate(candidates):
            current_ratio = right_speech_ratio(left_speech_sec, right_speech_sec)
            if current_ratio >= target_right_ratio:
                break

            response_index = balance_fill_index(turn.segment.index, fill_index)
            cached_response = generated_by_index.get(response_index)
            if (
                cached_response is not None
                and cached_response.responseKind == "balance_fill_llm"
                and Path(cached_response.ttsPath).exists()
            ):
                progress.advance(task)
                continue
            if cached_response is not None:
                generated_by_index.pop(response_index, None)

            text = request_balance_fill_completion_checked(
                client,
                config,
                transcript,
                turn,
                start_sec,
                fill_max_chars,
                llm_timeout,
            )
            tts_path = tts_root / f"fill{turn.segment.index:04d}_{fill_index:02d}.wav"
            raw_tts_path = tts_root / "raw" / f"fill{turn.segment.index:04d}_{fill_index:02d}.wav"
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
            adjusted_start_sec = min(
                start_sec,
                max(turn.segment.start, turn.segment.end - response_margin_sec - tts_duration_sec),
            )
            end_sec = adjusted_start_sec + tts_duration_sec
            if end_sec > turn.segment.end - response_margin_sec:
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
                promptStart=turn.segment.start,
                promptEnd=turn.segment.end,
                promptText=turn.segment.text,
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
                f"[cyan]Balance fill[/cyan] {turn.segment.index}:{fill_index} "
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
) -> torch.Tensor:
    left = load_mono_audio(input_audio, sample_rate)
    max_frames = left.shape[-1]
    for response in generated_responses:
        max_frames = max(max_frames, math.ceil(response.responseEnd * sample_rate) + 1)

    left = torch.nn.functional.pad(left, (0, max_frames - left.shape[-1]))
    right = torch.zeros(1, max_frames, dtype=left.dtype)
    for response in generated_responses:
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
            if not allowed_kinds:
                generated_by_index.pop(segment.index, None)
                console.print(
                    f"[yellow]Skipping[/yellow] turn {segment.index}: "
                    "pre-question can not be placed naturally"
                )
                progress.advance(task)
                continue
            if cached_response is not None and Path(cached_response.ttsPath).exists():
                if (
                    cached_response.promptText == expected_prompt
                    and cached_response.responseKind in allowed_kinds
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

            response_kind, response_text = request_interaction_completion_checked(
                openai_client,
                config,
                [turn.segment for turn in turns],
                segment,
                allowed_kinds,
                {
                    "reply": reply_chars,
                    "pre_question": pre_question_chars,
                },
                args.llm_timeout,
                time_until_next,
                time_since_previous,
            )
            tts_path = tts_root / f"response{segment.index:04d}.wav"
            raw_tts_path = tts_root / "raw" / f"response{segment.index:04d}.wav"
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
            response_start = response_start_for_kind(
                response_kind,
                turn,
                tts_duration_sec,
                args.response_delay_sec,
                args.response_margin_sec,
                next_available_start,
            )
            response_end = response_start + tts_duration_sec
            if response_kind == "reply":
                next_available_start = response_end + args.response_delay_sec
            generated_by_index[segment.index] = GeneratedResponse(
                index=segment.index,
                promptStart=segment.start,
                promptEnd=segment.end,
                promptText=segment.text,
                responseText=response_text,
                responseKind=response_kind,
                responseStart=response_start,
                responseEnd=response_end,
                ttsPath=str(tts_path),
            )
            save_cached_responses(
                responses_cache_path,
                [generated_by_index[index] for index in sorted(generated_by_index)],
            )
            console.print(
                f"[{segment.index:04d}] {response_kind} | {segment.text} -> {response_text}"
            )
            progress.advance(task)

    turn_indexes = {turn.segment.index for turn in turns}
    generated_by_index = {
        index: response
        for index, response in generated_by_index.items()
        if response_belongs_to_turns(response, turn_indexes)
        and response.responseKind != "balance_fill"
    }
    balance_stats = {
        "leftSpeechSec": speech_seconds_for_turns(turns),
        "rightSpeechSec": speech_seconds_for_responses(list(generated_by_index.values())),
        "rightRatio": right_speech_ratio(
            speech_seconds_for_turns(turns),
            speech_seconds_for_responses(list(generated_by_index.values())),
        ),
        "addedFillCount": 0,
    }
    if balance_fill_enabled(args.balance_fill_mode, args.interaction_mode):
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
            args.response_margin_sec,
            args.target_right_ratio,
            args.max_right_ratio,
            args.long_turn_fill_sec,
            args.fill_interval_sec,
            args.fill_max_chars,
            args.llm_timeout,
        )
    generated = [generated_by_index[index] for index in sorted(generated_by_index)]
    save_cached_responses(responses_cache_path, generated)
    with status(f"Writing stereo wav to [bold]{args.output}[/bold]"):
        stereo = build_stereo_audio(args.input, generated, args.sample_rate)
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
        "targetRightRatio": args.target_right_ratio,
        "maxRightRatio": args.max_right_ratio,
        "fillMaxChars": args.fill_max_chars,
        "balanceStats": balance_stats,
        "transcript": [segment.__dict__ for segment in transcript],
        "responses": [response.__dict__ for response in generated],
    }
    metadata_output.write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    console.print(f"[green]Wrote[/green] stereo wav to {args.output}")
    console.print(f"[green]Wrote[/green] metadata to {metadata_output}")


if __name__ == "__main__":
    main()

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torchaudio

from generateSoloConversationDataset import (
    GeneratedResponse,
    build_conversation_turns,
    build_stereo_audio,
    create_openai_client,
    load_env_config,
    load_mono_audio,
    load_cached_responses,
    load_or_create_transcript,
    load_whisper_model,
    allowed_interaction_kinds,
    pre_question_char_limit_for_turn,
    request_interaction_completion_checked,
    response_char_limit_for_turn,
    response_start_for_kind,
    save_cached_responses,
    synthesize_tts_with_speed,
    transcribe_audio_with_model,
)
from progressUtils import console, create_progress, status


AUDIO_EXTENSIONS = {".wav", ".mp3", ".flac", ".m4a", ".ogg", ".aac"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Process raw solo audio files sequentially into stereo Moshi training wavs"
    )
    parser.add_argument("--input-dir", type=Path, default=Path("datasets/raw"))
    parser.add_argument("--output-dir", type=Path, default=Path("datasets/stereo"))
    parser.add_argument("--cache-dir", type=Path, default=Path("datasets/cache"))
    parser.add_argument("--env-file", type=Path, default=Path(".env"))
    parser.add_argument("--recursive", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--continue-on-error", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--limit", type=int)
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
    parser.add_argument("--min-segment-sec", type=float, default=0.4)
    parser.add_argument("--max-segments", type=int)
    parser.add_argument("--refresh-transcript", action="store_true")
    parser.add_argument("--refresh-responses", action="store_true")
    parser.add_argument("--tts-timeout", type=float, default=120.0)
    parser.add_argument("--llm-timeout", type=float, default=120.0)
    return parser.parse_args()


def collect_audio_paths(input_dir: Path, recursive: bool) -> list[Path]:
    if not input_dir.exists():
        raise FileNotFoundError(f"Input directory does not exist: {input_dir}")
    iterator = input_dir.rglob("*") if recursive else input_dir.glob("*")
    paths = [
        path
        for path in sorted(iterator)
        if path.is_file() and path.suffix.lower() in AUDIO_EXTENSIONS
    ]
    return paths


def build_output_path(input_path: Path, input_dir: Path, output_dir: Path) -> Path:
    relative = input_path.relative_to(input_dir)
    return output_dir / relative.with_suffix(".wav")


def process_one(
    args: argparse.Namespace,
    config,
    openai_client,
    whisper_model,
    input_path: Path,
) -> None:
    output_path = build_output_path(input_path, args.input_dir, args.output_dir)
    metadata_output = output_path.with_name(f"{output_path.stem}.responses.json")
    cache_root = args.cache_dir / output_path.stem
    transcript_cache_path = cache_root / "transcript.json"
    responses_cache_path = cache_root / "responses.json"
    tts_root = cache_root / "tts"

    if output_path.exists() and not args.overwrite:
        print(f"Skip existing: {output_path}")
        return

    transcript = load_or_create_transcript(
        transcript_cache_path,
        lambda: transcribe_audio_with_model(
            whisper_model,
            input_path,
            args.language,
            args.min_segment_sec,
            None,
        ),
        args.refresh_transcript,
    )
    if args.max_segments is not None:
        transcript = transcript[: args.max_segments]
    if not transcript:
        raise RuntimeError(f"No transcript segments were produced: {input_path}")

    turns = build_conversation_turns(transcript, args.merge_gap_sec)
    tts_root.mkdir(parents=True, exist_ok=True)
    cached_responses = load_cached_responses(
        responses_cache_path,
        args.refresh_responses,
    )
    generated_by_index: dict[int, GeneratedResponse] = dict(cached_responses)
    next_available_start = 0.0
    with create_progress() as progress:
        task = progress.add_task(
            f"Generating replies for {input_path.name}",
            total=len(turns),
        )
        for turn in turns:
            segment = turn.segment
            progress.update(task, description=f"{input_path.name} turn {segment.index + 1}")
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
                    cached_response.promptText == segment.text
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
                f"  [{segment.index:04d}] {response_kind} | {segment.text} -> {response_text}"
            )
            progress.advance(task)

    turn_indexes = {turn.segment.index for turn in turns}
    generated = [
        generated_by_index[index]
        for index in sorted(generated_by_index)
        if index in turn_indexes
    ]
    save_cached_responses(responses_cache_path, generated)
    with status(f"Writing stereo wav to [bold]{output_path}[/bold]"):
        stereo = build_stereo_audio(input_path, generated, args.sample_rate)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        torchaudio.save(output_path, stereo, args.sample_rate)

    metadata = {
        "input": str(input_path),
        "output": str(output_path),
        "language": args.language,
        "whisperModel": args.whisper_model,
        "openaiBaseUrl": config.openaiBaseUrl,
        "openaiModel": config.openaiModel,
        "ttsUrl": config.ttsUrl,
        "ttsType": config.ttsType,
        "interactionMode": args.interaction_mode,
        "transcript": [segment.__dict__ for segment in transcript],
        "responses": [response.__dict__ for response in generated],
        "cacheDir": str(cache_root),
    }
    metadata_output.write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    console.print(f"[green]Wrote[/green] stereo wav to {output_path}")
    console.print(f"[green]Wrote[/green] metadata to {metadata_output}")


def main() -> None:
    args = parse_args()
    audio_paths = collect_audio_paths(args.input_dir, args.recursive)
    if not audio_paths:
        console.print(f"[yellow]No audio files found[/yellow] in {args.input_dir}")
        return
    if args.limit is not None:
        audio_paths = audio_paths[: args.limit]

    console.print("[bold]Input files:[/bold]")
    for index, path in enumerate(audio_paths, start=1):
        console.print(f"  {index:04d}: {path}")
    if args.dry_run:
        return

    config = load_env_config(args.env_file)
    openai_client = create_openai_client(config)
    whisper_model = load_whisper_model(args.whisper_model)
    for index, input_path in enumerate(audio_paths, start=1):
        console.rule(f"[bold]Processing {index}/{len(audio_paths)}[/bold] {input_path.name}")
        try:
            process_one(args, config, openai_client, whisper_model, input_path)
        except Exception as error:
            if not args.continue_on_error:
                raise
            console.print(f"[red]Failed[/red]: {input_path}: {error}")


if __name__ == "__main__":
    main()

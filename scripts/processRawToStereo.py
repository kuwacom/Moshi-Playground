from __future__ import annotations

import argparse
import json
from pathlib import Path

import torchaudio

from datasetPaths import datasetCacheDir, datasetRawDir, datasetStereoDir
from generateSoloConversationDataset import (
    GeneratedResponse,
    add_balance_fill_responses,
    add_semantic_gap_insertions,
    allowed_interaction_kinds,
    balance_fill_enabled,
    build_conversation_turns,
    build_stereo_audio,
    cached_response_is_usable,
    create_openai_client,
    load_env_config,
    load_mono_audio,
    load_cached_responses,
    load_or_create_transcript,
    load_whisper_model,
    place_response_for_kind,
    pre_question_char_limit_for_turn,
    recent_response_texts,
    remove_conflicting_balance_fills,
    remove_left_overlapping_balance_fills,
    request_interaction_completion_checked,
    response_char_limit_for_turn,
    response_belongs_to_turns,
    responses_before_turn,
    save_cached_responses,
    SEMANTIC_GAP_RESPONSE_KIND,
    semantic_gap_insertions_enabled,
    shift_responses_for_timeline,
    shift_transcript_for_timeline,
    synthesize_tts_with_speed,
    tighten_char_limits_for_placement,
    transcribe_audio_with_model,
    tts_paths_for_text,
)
from progressUtils import console, create_progress, status


AUDIO_EXTENSIONS = {".wav", ".mp3", ".flac", ".m4a", ".ogg", ".aac"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Process raw solo audio files sequentially into stereo Moshi training wavs"
    )
    parser.add_argument("--input-dir", type=Path, default=datasetRawDir())
    parser.add_argument("--output-dir", type=Path, default=datasetStereoDir())
    parser.add_argument("--cache-dir", type=Path, default=datasetCacheDir())
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
        console.print(f"[cyan]Skip existing[/cyan]: {output_path}")
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
                    segment.text,
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
                f"  [{segment.index:04d}] {generated_response.responseKind} | "
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
    left_speech_sec = sum(
        max(0.0, turn.segment.end - turn.segment.start) for turn in turns
    )
    right_speech_sec = sum(
        max(0.0, response.responseEnd - response.responseStart)
        for response in generated_by_index.values()
    )
    balance_stats = {
        "leftSpeechSec": left_speech_sec,
        "rightSpeechSec": right_speech_sec,
        "rightRatio": (
            0.0
            if left_speech_sec + right_speech_sec <= 0.0
            else right_speech_sec / (left_speech_sec + right_speech_sec)
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
    semantic_right_speech_sec = sum(
        max(0.0, response.responseEnd - response.responseStart)
        for response in generated_by_index.values()
    )
    semantic_gap_stats = {
        "leftSpeechSec": left_speech_sec,
        "rightSpeechSec": semantic_right_speech_sec,
        "rightRatio": (
            0.0
            if left_speech_sec + semantic_right_speech_sec <= 0.0
            else semantic_right_speech_sec
            / (left_speech_sec + semantic_right_speech_sec)
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
    with status(f"Writing stereo wav to [bold]{output_path}[/bold]"):
        stereo = build_stereo_audio(
            input_path,
            generated,
            args.sample_rate,
            args.response_delay_sec,
            args.response_margin_sec,
        )
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

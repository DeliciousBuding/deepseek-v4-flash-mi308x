#!/usr/bin/env python3
"""Exact multi-needle recall benchmark for long-context serving profiles."""

from __future__ import annotations

import argparse
import json
import re
import urllib.request
import uuid
from dataclasses import dataclass

from bench_agent_trace import BASE, MODEL, api_key, chat_stream


DEFAULT_TARGET_TOKENS = (100_000, 256_000, 384_000, 475_000)
DEFAULT_NEEDLE_COUNT = 5
FILLER_TOKEN_ESTIMATE = 20


@dataclass(frozen=True)
class RecallResult:
    target_tokens: int
    prompt_tokens: int
    cached_tokens: int | None
    time_to_first_token_seconds: float
    total_seconds: float
    expected_values: tuple[str, ...]
    returned_values: tuple[str, ...] | None

    @property
    def passed(self) -> bool:
        return self.returned_values == self.expected_values

    @property
    def position_matches(self) -> tuple[bool, ...]:
        if self.returned_values is None:
            return tuple(False for _ in self.expected_values)
        return tuple(
            position < len(self.returned_values)
            and self.returned_values[position] == expected_value
            for position, expected_value in enumerate(self.expected_values)
        )


def parse_arguments() -> argparse.Namespace:
    argument_parser = argparse.ArgumentParser(
        description=(
            "Place random retrieval records throughout a cold long context and "
            "require an exact ordered JSON-array response."
        )
    )
    argument_parser.add_argument(
        "--target-tokens",
        type=int,
        nargs="+",
        default=list(DEFAULT_TARGET_TOKENS),
        help="approximate prompt token targets",
    )
    argument_parser.add_argument(
        "--needles",
        type=int,
        default=DEFAULT_NEEDLE_COUNT,
        help="retrieval records distributed through each document",
    )
    argument_parser.add_argument(
        "--output-tokens",
        type=int,
        default=256,
        help="maximum completion tokens",
    )
    argument_parser.add_argument(
        "--calibration-rounds",
        type=int,
        default=3,
        help="maximum /tokenize-based filler calibration rounds",
    )
    return argument_parser.parse_args()


def validate_arguments(arguments: argparse.Namespace) -> None:
    if any(target_tokens < 1 for target_tokens in arguments.target_tokens):
        raise ValueError("all --target-tokens values must be positive")
    if arguments.needles < 1:
        raise ValueError("--needles must be positive")
    if arguments.output_tokens < 1:
        raise ValueError("--output-tokens must be positive")
    if arguments.calibration_rounds < 1:
        raise ValueError("--calibration-rounds must be positive")


def create_needle_values(needle_count: int) -> tuple[str, ...]:
    return tuple(
        f"VALUE_{needle_index:02d}_{uuid.uuid4().hex[:16].upper()}"
        for needle_index in range(1, needle_count + 1)
    )


def distribute_filler_repetitions(
    filler_repetitions: int,
    segment_count: int,
) -> list[int]:
    repetitions_per_segment, remainder = divmod(
        filler_repetitions,
        segment_count,
    )
    return [
        repetitions_per_segment + (1 if segment_index < remainder else 0)
        for segment_index in range(segment_count)
    ]


def create_filler_segment(start_index: int, repetitions: int) -> str:
    return "".join(
        (
            f"Repository note {record_index:06d}: validate cancellation, "
            "backpressure, error propagation, ownership, and deterministic "
            "cleanup before merging. "
        )
        for record_index in range(start_index, start_index + repetitions)
    )


def build_recall_prompt(
    filler_repetitions: int,
    needle_values: tuple[str, ...],
    request_nonce: str,
) -> str:
    segment_repetitions = distribute_filler_repetitions(
        filler_repetitions=filler_repetitions,
        segment_count=len(needle_values) + 1,
    )
    prompt_parts = [
        f"Request nonce: {request_nonce}.\n",
        "Read the complete repository archive. Retrieval records are factual "
        "and must be returned exactly; ordinary notes are distractors.\n",
    ]
    next_record_index = 0
    for segment_index, repetitions in enumerate(segment_repetitions):
        prompt_parts.append(
            create_filler_segment(
                start_index=next_record_index,
                repetitions=repetitions,
            )
        )
        next_record_index += repetitions
        if segment_index < len(needle_values):
            prompt_parts.append(
                "\n<retrieval_record "
                f'id="needle-{segment_index + 1}">'
                f"{needle_values[segment_index]}"
                "</retrieval_record>\n"
            )

    prompt_parts.append(
        "\nReturn one JSON array containing the retrieval_record values in "
        "needle number order. Copy every value exactly. Do not use Markdown, "
        "keys, commentary, or values from ordinary repository notes."
    )
    return "".join(prompt_parts)


def tokenize_prompt(prompt: str) -> int:
    request = urllib.request.Request(
        BASE + "/tokenize",
        data=json.dumps({"model": MODEL, "prompt": prompt}).encode(),
        headers={
            "Content-Type": "application/json",
            "Authorization": "Bearer " + api_key(),
        },
    )
    with urllib.request.urlopen(request, timeout=1800) as response:
        payload = json.load(response)
    token_count = payload.get("count")
    if not isinstance(token_count, int):
        raise RuntimeError(f"unexpected /tokenize response: {payload!r}")
    return token_count


def calibrate_prompt(
    target_tokens: int,
    needle_values: tuple[str, ...],
    request_nonce: str,
    calibration_rounds: int,
) -> str:
    filler_repetitions = max(1, target_tokens // FILLER_TOKEN_ESTIMATE)
    prompt = ""
    for _ in range(calibration_rounds):
        prompt = build_recall_prompt(
            filler_repetitions=filler_repetitions,
            needle_values=needle_values,
            request_nonce=request_nonce,
        )
        measured_tokens = tokenize_prompt(prompt)
        relative_error = abs(measured_tokens - target_tokens) / target_tokens
        if relative_error <= 0.01:
            return prompt
        filler_repetitions = max(
            1,
            round(filler_repetitions * target_tokens / measured_tokens),
        )
    return prompt


def parse_json_array(content: str) -> tuple[str, ...] | None:
    json_decoder = json.JSONDecoder()
    for match in re.finditer(r"\[", content):
        try:
            decoded_value, _ = json_decoder.raw_decode(content[match.start():])
        except json.JSONDecodeError:
            continue
        if (
            isinstance(decoded_value, list)
            and all(isinstance(value, str) for value in decoded_value)
        ):
            return tuple(decoded_value)
    return None


def run_recall_case(
    target_tokens: int,
    needle_count: int,
    output_tokens: int,
    calibration_rounds: int,
) -> RecallResult:
    needle_values = create_needle_values(needle_count)
    request_nonce = uuid.uuid4().hex
    prompt = calibrate_prompt(
        target_tokens=target_tokens,
        needle_values=needle_values,
        request_nonce=request_nonce,
        calibration_rounds=calibration_rounds,
    )
    result = chat_stream(
        [
            {
                "role": "system",
                "content": (
                    "You are an exact retrieval evaluator. Follow the final "
                    "output-format instruction and copy records verbatim."
                ),
            },
            {"role": "user", "content": prompt},
        ],
        max_tokens=output_tokens,
        temperature=0.0,
        salt=f"long-recall-{uuid.uuid4().hex}",
    )
    return RecallResult(
        target_tokens=target_tokens,
        prompt_tokens=result["prompt_tokens"],
        cached_tokens=result["cached_tokens"],
        time_to_first_token_seconds=result["ttft"],
        total_seconds=result["total"],
        expected_values=needle_values,
        returned_values=parse_json_array(result["content"]),
    )


def main() -> int:
    arguments = parse_arguments()
    try:
        validate_arguments(arguments)
    except ValueError as error:
        raise SystemExit(str(error)) from error

    failed_cases = 0
    for target_tokens in arguments.target_tokens:
        recall_result = run_recall_case(
            target_tokens=target_tokens,
            needle_count=arguments.needles,
            output_tokens=arguments.output_tokens,
            calibration_rounds=arguments.calibration_rounds,
        )
        if not recall_result.passed:
            failed_cases += 1
        print(
            f"target={recall_result.target_tokens} "
            f"prompt={recall_result.prompt_tokens} "
            f"needles={len(recall_result.expected_values)} "
            f"result={'PASS' if recall_result.passed else 'FAIL'} "
            f"TTFT={recall_result.time_to_first_token_seconds:.3f}s "
            f"total={recall_result.total_seconds:.3f}s "
            f"cached={recall_result.cached_tokens}",
            flush=True,
        )
        if not recall_result.passed:
            print(
                f"  position_matches={recall_result.position_matches} "
                f"returned_count={len(recall_result.returned_values or ())} "
                f"expected_count={len(recall_result.expected_values)}",
                flush=True,
            )

    print(
        f"summary={len(arguments.target_tokens) - failed_cases}/"
        f"{len(arguments.target_tokens)} PASS",
        flush=True,
    )
    return 1 if failed_cases else 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Measure short/medium prompt cost under one configured context ceiling.

Run this script once after a clean restart for each MAX_MODEL_LEN candidate.
Every request receives a unique cache salt so automatic prefix caching cannot
turn the configured-ceiling comparison into a warm-prefix comparison.
"""

from __future__ import annotations

import argparse
import json
import statistics
import urllib.request
import uuid
from dataclasses import asdict, dataclass
from pathlib import Path

from bench_agent_trace import SYSTEM_PROMPT, chat_stream, make_repo_context


DEFAULT_PROMPT_TOKEN_TARGETS = (8_000, 32_000, 100_000)
# ``make_repo_context`` intentionally uses a cheap character-based estimate.
# This model's tokenizer measures that fixture about 1.4x larger, so compensate
# here while still reporting the authoritative response-level prompt count.
REPO_CONTEXT_TARGET_SCALE = 0.70


@dataclass(frozen=True)
class PromptSample:
    target_prompt_tokens: int
    measured_prompt_tokens: int
    cached_prompt_tokens: int | None
    completion_tokens: int
    time_to_first_token_seconds: float
    total_seconds: float
    decode_tokens_per_second: float


@dataclass(frozen=True)
class PromptSummary:
    target_prompt_tokens: int
    measured_prompt_tokens: int
    sample_count: int
    median_time_to_first_token_seconds: float
    maximum_time_to_first_token_seconds: float
    median_total_seconds: float
    median_decode_tokens_per_second: float
    cached_prompt_tokens: int


def parse_arguments() -> argparse.Namespace:
    argument_parser = argparse.ArgumentParser(
        description=(
            "Measure cold 8K/32K/100K prompt latency for the currently "
            "running MAX_MODEL_LEN profile."
        )
    )
    argument_parser.add_argument(
        "--profile-label",
        default="current",
        help="label included in output, for example 262144 or 524288",
    )
    argument_parser.add_argument(
        "--prompt-tokens",
        type=int,
        nargs="+",
        default=list(DEFAULT_PROMPT_TOKEN_TARGETS),
        help="approximate prompt token targets",
    )
    argument_parser.add_argument(
        "--samples",
        type=int,
        default=2,
        help="cold samples per prompt target",
    )
    argument_parser.add_argument(
        "--output-tokens",
        type=int,
        default=64,
        help="maximum completion tokens per request",
    )
    argument_parser.add_argument(
        "--json-output",
        type=Path,
        help="optional path for machine-readable samples and summaries",
    )
    argument_parser.add_argument(
        "--skip-warmup",
        action="store_true",
        help="skip the default isolated 8K first-request JIT warm-up",
    )
    return argument_parser.parse_args()


def validate_arguments(arguments: argparse.Namespace) -> None:
    if arguments.samples < 1:
        raise ValueError("--samples must be positive")
    if arguments.output_tokens < 1:
        raise ValueError("--output-tokens must be positive")
    if any(prompt_tokens < 1 for prompt_tokens in arguments.prompt_tokens):
        raise ValueError("all --prompt-tokens values must be positive")


def measure_prompt(
    target_prompt_tokens: int,
    output_tokens: int,
) -> PromptSample:
    estimated_repo_context_tokens = round(
        target_prompt_tokens * REPO_CONTEXT_TARGET_SCALE
    )
    prompt = (
        SYSTEM_PROMPT
        + make_repo_context(estimated_repo_context_tokens)
        + "\nReview the repository context and identify one concurrency risk."
    )
    cache_salt = f"configured-ceiling-{uuid.uuid4().hex}"
    result = chat_stream(
        [{"role": "user", "content": prompt}],
        max_tokens=output_tokens,
        temperature=0.0,
        salt=cache_salt,
    )
    return PromptSample(
        target_prompt_tokens=target_prompt_tokens,
        measured_prompt_tokens=result["prompt_tokens"],
        cached_prompt_tokens=result["cached_tokens"],
        completion_tokens=result["n_tokens"],
        time_to_first_token_seconds=result["ttft"],
        total_seconds=result["total"],
        decode_tokens_per_second=result["decode_tok_s"],
    )


def summarize_samples(samples: list[PromptSample]) -> PromptSummary:
    if not samples:
        raise ValueError("cannot summarize an empty sample list")

    measured_prompt_tokens = [sample.measured_prompt_tokens for sample in samples]
    cached_prompt_tokens = [
        sample.cached_prompt_tokens or 0
        for sample in samples
    ]
    return PromptSummary(
        target_prompt_tokens=samples[0].target_prompt_tokens,
        measured_prompt_tokens=round(statistics.median(measured_prompt_tokens)),
        sample_count=len(samples),
        median_time_to_first_token_seconds=statistics.median(
            sample.time_to_first_token_seconds for sample in samples
        ),
        maximum_time_to_first_token_seconds=max(
            sample.time_to_first_token_seconds for sample in samples
        ),
        median_total_seconds=statistics.median(
            sample.total_seconds for sample in samples
        ),
        median_decode_tokens_per_second=statistics.median(
            sample.decode_tokens_per_second for sample in samples
        ),
        cached_prompt_tokens=sum(cached_prompt_tokens),
    )


def read_preemption_count() -> float | None:
    try:
        metrics_text = urllib.request.urlopen(
            "http://127.0.0.1:8000/metrics",
            timeout=10,
        ).read().decode()
    except Exception:
        return None

    metric_prefix = "vllm:num_preemptions_total{"
    for metric_line in metrics_text.splitlines():
        if metric_line.startswith(metric_prefix):
            return float(metric_line.rsplit(maxsplit=1)[-1])
    return None


def format_summary(summary: PromptSummary) -> str:
    return (
        f"target={summary.target_prompt_tokens} "
        f"prompt={summary.measured_prompt_tokens} "
        f"samples={summary.sample_count} "
        f"TTFT_p50={summary.median_time_to_first_token_seconds:.3f}s "
        f"TTFT_max={summary.maximum_time_to_first_token_seconds:.3f}s "
        f"total_p50={summary.median_total_seconds:.3f}s "
        f"decode_p50={summary.median_decode_tokens_per_second:.1f} tok/s "
        f"cached_tokens={summary.cached_prompt_tokens}"
    )


def main() -> int:
    arguments = parse_arguments()
    try:
        validate_arguments(arguments)
    except ValueError as error:
        raise SystemExit(str(error)) from error

    if not arguments.skip_warmup:
        warmup_target_prompt_tokens = min(arguments.prompt_tokens)
        warmup_sample = measure_prompt(
            target_prompt_tokens=warmup_target_prompt_tokens,
            output_tokens=min(arguments.output_tokens, 16),
        )
        print(
            f"warmup_target={warmup_target_prompt_tokens} "
            f"prompt={warmup_sample.measured_prompt_tokens} "
            f"TTFT={warmup_sample.time_to_first_token_seconds:.3f}s",
            flush=True,
        )

    preemptions_before = read_preemption_count()
    all_samples: list[PromptSample] = []
    summaries: list[PromptSummary] = []

    print(f"profile={arguments.profile_label}", flush=True)
    for target_prompt_tokens in arguments.prompt_tokens:
        target_samples: list[PromptSample] = []
        for sample_number in range(1, arguments.samples + 1):
            sample = measure_prompt(
                target_prompt_tokens=target_prompt_tokens,
                output_tokens=arguments.output_tokens,
            )
            target_samples.append(sample)
            all_samples.append(sample)
            print(
                f"  target={target_prompt_tokens} sample={sample_number} "
                f"prompt={sample.measured_prompt_tokens} "
                f"TTFT={sample.time_to_first_token_seconds:.3f}s "
                f"total={sample.total_seconds:.3f}s "
                f"decode={sample.decode_tokens_per_second:.1f} tok/s "
                f"cached={sample.cached_prompt_tokens}",
                flush=True,
            )

        summary = summarize_samples(target_samples)
        summaries.append(summary)
        print(format_summary(summary), flush=True)

    preemptions_after = read_preemption_count()
    preemption_delta = None
    if preemptions_before is not None and preemptions_after is not None:
        preemption_delta = preemptions_after - preemptions_before
    print(f"preemption_delta={preemption_delta}", flush=True)

    if arguments.json_output:
        output_payload = {
            "profile": arguments.profile_label,
            "preemption_delta": preemption_delta,
            "samples": [asdict(sample) for sample in all_samples],
            "summaries": [asdict(summary) for summary in summaries],
        }
        arguments.json_output.write_text(
            json.dumps(output_payload, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Compare decode throughput and latency at high request concurrency.

Run this benchmark once per serving profile so speculative and native decoder
results remain isolated by a clean service restart. The default C32-C64 sweep
matches the MI308X DSpark boundary experiment documented in the tuning log.
"""
from __future__ import annotations

import argparse
import concurrent.futures
import math
import statistics
import time
from dataclasses import dataclass

DEFAULT_CONCURRENCIES = (32, 64)
DEFAULT_PROMPT = (
    "Implement a concurrency-safe bounded work queue with cancellation, "
    "backpressure, and tests."
)


@dataclass(frozen=True)
class BenchmarkResult:
    concurrency: int
    elapsed_seconds: float
    completion_tokens: int
    request_latencies: tuple[float, ...]
    successful_requests: int
    failed_requests: int
    mean_accepted_length: float | None
    acceptance_rate: float | None

    @property
    def aggregate_tokens_per_second(self) -> float:
        return self.completion_tokens / self.elapsed_seconds

    @property
    def median_latency_seconds(self) -> float:
        return statistics.median(self.request_latencies)

    @property
    def p95_latency_seconds(self) -> float:
        percentile_index = max(
            0,
            math.ceil(len(self.request_latencies) * 0.95) - 1,
        )
        return self.request_latencies[percentile_index]


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Measure high-concurrency decode throughput and latency."
    )
    parser.add_argument(
        "--concurrencies",
        type=int,
        nargs="+",
        default=list(DEFAULT_CONCURRENCIES),
        help="Concurrent request counts to test (default: 32 64).",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=256,
        help="Maximum completion tokens per request (default: 256).",
    )
    parser.add_argument(
        "--prompt",
        default=DEFAULT_PROMPT,
        help="Base prompt. A request index is appended to keep requests distinct.",
    )
    return parser.parse_args()


def validate_arguments(arguments: argparse.Namespace) -> None:
    invalid_concurrencies = [
        concurrency
        for concurrency in arguments.concurrencies
        if concurrency <= 0
    ]
    if invalid_concurrencies:
        raise ValueError(
            "Concurrency values must be positive: "
            f"{invalid_concurrencies}"
        )
    if arguments.max_tokens <= 0:
        raise ValueError("--max-tokens must be positive")


def run_request(
    request_index: int,
    prompt: str,
    max_tokens: int,
) -> dict[str, float | int | str]:
    from bench_full import chat

    request_prompt = f"{prompt} Request variant {request_index}."
    return chat(request_prompt, max_tokens=max_tokens, temperature=0.0)


def measure_concurrency(
    concurrency: int,
    prompt: str,
    max_tokens: int,
) -> BenchmarkResult:
    from bench_full import spec_metrics

    metrics_before = spec_metrics()
    started_at = time.perf_counter()
    successful_results: list[dict[str, float | int | str]] = []
    failures: list[Exception] = []

    with concurrent.futures.ThreadPoolExecutor(
        max_workers=concurrency
    ) as executor:
        request_futures = [
            executor.submit(
                run_request,
                request_index,
                prompt,
                max_tokens,
            )
            for request_index in range(concurrency)
        ]
        for request_future in concurrent.futures.as_completed(request_futures):
            try:
                successful_results.append(request_future.result())
            except Exception as error:  # noqa: BLE001 - report all client failures.
                failures.append(error)

    elapsed_seconds = time.perf_counter() - started_at
    if not successful_results:
        failure_samples = ", ".join(repr(error) for error in failures[:3])
        raise RuntimeError(
            f"C{concurrency} produced no successful requests: {failure_samples}"
        )

    metrics_after = spec_metrics()
    draft_count = metrics_after["drafts"] - metrics_before["drafts"]
    accepted_tokens = metrics_after["accepted"] - metrics_before["accepted"]
    draft_tokens = (
        metrics_after["draft_tokens"] - metrics_before["draft_tokens"]
    )
    mean_accepted_length = (
        1 + accepted_tokens / draft_count if draft_count > 0 else None
    )
    acceptance_rate = (
        accepted_tokens / draft_tokens if draft_tokens > 0 else None
    )

    return BenchmarkResult(
        concurrency=concurrency,
        elapsed_seconds=elapsed_seconds,
        completion_tokens=sum(
            int(result["completion_tokens"])
            for result in successful_results
        ),
        request_latencies=tuple(
            sorted(float(result["total_s"]) for result in successful_results)
        ),
        successful_requests=len(successful_results),
        failed_requests=len(failures),
        mean_accepted_length=mean_accepted_length,
        acceptance_rate=acceptance_rate,
    )


def format_result(result: BenchmarkResult) -> str:
    decoder_metrics = "native decoder"
    if (
        result.mean_accepted_length is not None
        and result.acceptance_rate is not None
    ):
        decoder_metrics = (
            f"mean_accepted={result.mean_accepted_length:.2f} "
            f"acceptance={result.acceptance_rate:.1%}"
        )

    return (
        f"C{result.concurrency}: "
        f"success={result.successful_requests}/"
        f"{result.successful_requests + result.failed_requests} "
        f"elapsed={result.elapsed_seconds:.3f}s "
        f"tokens={result.completion_tokens} "
        f"aggregate={result.aggregate_tokens_per_second:.1f} tok/s "
        f"latency_p50={result.median_latency_seconds:.3f}s "
        f"latency_p95={result.p95_latency_seconds:.3f}s "
        f"{decoder_metrics}"
    )


def main() -> int:
    arguments = parse_arguments()
    validate_arguments(arguments)

    total_failures = 0
    for concurrency in arguments.concurrencies:
        result = measure_concurrency(
            concurrency=concurrency,
            prompt=arguments.prompt,
            max_tokens=arguments.max_tokens,
        )
        print(format_result(result), flush=True)
        total_failures += result.failed_requests

    return 1 if total_failures else 0


if __name__ == "__main__":
    raise SystemExit(main())

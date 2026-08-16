#!/usr/bin/env python3
"""Measure short-request TTFT while a genuinely cold long prefill is running.

The long prompt gets a nonce in its first cache block by default so repeated runs
cannot be turned into fake "isolation wins" by automatic prefix caching.
"""
from __future__ import annotations

import argparse
import json
import os
import secrets
import statistics
import threading
import time
import urllib.request
from pathlib import Path

BASE = os.environ.get("VLLM_BASE_URL", "http://127.0.0.1:8000")
MODEL = "deepseek-v4-flash"


def api_key() -> str:
    configured_api_key = os.environ.get("VLLM_API_KEY")
    if configured_api_key:
        return configured_api_key

    api_key_file = os.environ.get("VLLM_API_KEY_FILE")
    if not api_key_file:
        raise RuntimeError(
            "No API key configured; set VLLM_API_KEY or VLLM_API_KEY_FILE"
        )

    try:
        resolved_api_key = Path(api_key_file).read_text(encoding="utf-8").strip()
    except FileNotFoundError as exc:
        raise RuntimeError(
            f"VLLM_API_KEY_FILE does not exist: {api_key_file}"
        ) from exc
    if not resolved_api_key:
        raise RuntimeError(f"VLLM_API_KEY_FILE is empty: {api_key_file}")
    return resolved_api_key


def stream_request(prompt: str, max_tokens: int) -> float:
    body = json.dumps({
        "model": MODEL,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "stream": True,
        "stream_options": {"include_usage": True},
    }).encode()
    req = urllib.request.Request(BASE + "/v1/chat/completions", data=body, headers={
        "Content-Type": "application/json",
        "Authorization": "Bearer " + api_key(),
    })
    t0 = time.time()
    with urllib.request.urlopen(req, timeout=1800) as resp:
        for raw in resp:
            line = raw.decode("utf-8", "ignore").strip()
            if line.startswith("data:") and line[5:].strip() != "[DONE]":
                return time.time() - t0
    raise RuntimeError("stream ended before the first response chunk")


def run_once(
    long_tokens: int,
    short_tokens: int,
    *,
    reuse_prefix: bool,
    inject_after: float,
) -> tuple[float, float, float, float]:
    unit = (
        "The software engineering coding standards mandate camelCase for "
        "function names, snake_case for variable names, explicit error "
        "handling, single responsibility, dependency injection. "
    )
    nonce = "" if reuse_prefix else f"cold-isolation-{secrets.token_hex(8)}: "
    long_prompt = nonce + unit * max(1, long_tokens // 40)

    ttft_alone = stream_request("Say ok.", short_tokens)
    result: dict[str, float] = {}

    def long_worker() -> None:
        t0 = time.time()
        stream_request(long_prompt, short_tokens)
        result["long_total"] = time.time() - t0

    thread = threading.Thread(target=long_worker)
    thread.start()
    time.sleep(inject_after)
    ttft_late = stream_request("Say ok.", short_tokens)
    thread.join()

    long_total = result["long_total"]
    overhead = ttft_late - ttft_alone
    return ttft_alone, long_total, ttft_late, overhead


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("long_tokens", nargs="?", type=int, default=200000)
    ap.add_argument("short_tokens", nargs="?", type=int, default=8)
    ap.add_argument("--rounds", type=int, default=1, help="independent cold rounds")
    ap.add_argument("--inject-after", type=float, default=1.5, metavar="SECONDS")
    ap.add_argument("--max-added-ttft", type=float, default=0.5, metavar="SECONDS")
    ap.add_argument(
        "--reuse-prefix",
        action="store_true",
        help="reuse the deterministic long prefix (cache diagnostic only)",
    )
    args = ap.parse_args()
    if args.rounds < 1:
        ap.error("--rounds must be >= 1")
    if args.inject_after < 0:
        ap.error("--inject-after must be >= 0")

    samples: list[tuple[float, float, float, float]] = []
    for idx in range(1, args.rounds + 1):
        alone, long_total, late, overhead = run_once(
            args.long_tokens,
            args.short_tokens,
            reuse_prefix=args.reuse_prefix,
            inject_after=args.inject_after,
        )
        samples.append((alone, long_total, late, overhead))
        print(
            f"round {idx:02d}: alone={alone:.2f}s long={long_total:.1f}s "
            f"late={late:.2f}s added={overhead:+.2f}s"
        )

    long_totals = [x[1] for x in samples]
    overheads = [x[3] for x in samples]
    median_overhead = statistics.median(overheads)
    max_overhead = max(overheads)
    print()
    print(f"long prefill median:     {statistics.median(long_totals):.1f}s (~{args.long_tokens} tokens)")
    print(f"isolation median added:  {median_overhead:+.2f}s")
    print(f"isolation max added:     {max_overhead:+.2f}s")
    verdict = "OK" if max_overhead <= args.max_added_ttft else "DEGRADED"
    print(
        f"verdict:                 {verdict} "
        f"(all rounds <= +{args.max_added_ttft:.2f}s)"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""bench_ttft_isolation.py — short-request TTFT during a long prefill.

Validates the `--long-prefill-token-threshold` latency isolation: while a
long prefill occupies the engine, a late short request should still get its
first token quickly instead of waiting for the whole prefill to finish.

Usage:
  python3 bench_ttft_isolation.py [long_tokens] [short_tokens]
Defaults: 200K long prefill, 8-token short request.
"""
import json
import os
import sys
import threading
import time
import urllib.request

KEY = os.environ.get("VLLM_API_KEY") or open(
    os.environ.get("VLLM_API_KEY_FILE", "/mnt/workspace/.bootstrap/vllm_api_key")
).read().strip()
BASE = os.environ.get("VLLM_BASE_URL", "http://127.0.0.1:8000")
MODEL = "deepseek-v4-flash"


def stream_request(prompt, max_tokens):
    body = json.dumps({
        "model": MODEL,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "stream": True,
        "stream_options": {"include_usage": True},
    }).encode()
    req = urllib.request.Request(BASE + "/v1/chat/completions", data=body, headers={
        "Content-Type": "application/json",
        "Authorization": "Bearer " + KEY,
    })
    t0 = time.time()
    first = None
    with urllib.request.urlopen(req, timeout=1800) as resp:
        for line in resp:
            line = line.decode("utf-8", "ignore").strip()
            if line.startswith("data:") and line[5:].strip() != "[DONE]":
                if first is None:
                    first = time.time()
                    break  # only need TTFT
    return first - t0 if first else None


def main():
    long_tokens = int(sys.argv[1]) if len(sys.argv) > 1 else 200000
    unit = ("The software engineering coding standards mandate camelCase for "
            "function names, snake_case for variable names, explicit error "
            "handling, single responsibility, dependency injection. ")
    long_prompt = unit * max(1, long_tokens // 40)

    # Baseline: short request alone (no competing prefill)
    t0 = time.time()
    ttft_alone = stream_request("Say ok.", 8)
    print(f"short TTFT alone:        {ttft_alone:.2f}s (total {time.time()-t0:.2f}s)")

    # Start the long prefill, then inject the short request 1.5s later
    result = {}

    def long_worker():
        t0 = time.time()
        stream_request(long_prompt, 8)
        result["long_total"] = time.time() - t0

    thread = threading.Thread(target=long_worker)
    thread.start()
    time.sleep(1.5)
    t0 = time.time()
    ttft_late = stream_request("Say ok.", 8)
    result["short_late"] = time.time() - t0
    thread.join()

    print(f"long prefill total:      {result['long_total']:.1f}s "
          f"(~{long_tokens} tokens)")
    print(f"short TTFT during prefill: {ttft_late:.2f}s")
    print()
    overhead = ttft_late - ttft_alone
    print(f"isolation overhead:      {overhead:+.2f}s vs alone")
    verdict = "OK" if ttft_late <= ttft_alone + 0.5 else "DEGRADED"
    print(f"verdict:                 {verdict} (threshold: <= alone + 0.5s)")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""bench_agent_trace.py — coding-agent multi-turn session benchmark.

Simulates a realistic coding-agent workload against the OpenAI-compatible
endpoint, aligned with the trace profile from the vLLM x Mooncake agentic
serving study (Codex/SWE-bench Pro corpus):

  - stable prefix: system prompt + tool schema + repo context (reused every
    turn; must hit the prefix cache)
  - per-turn delta: ~2K new tokens (user message + tool results)
  - output: coding-sized completions
  - 131:1 aggregate input-to-output ratio is typical; 94%+ hit rate is the
    reference target for a well-structured agent trace

Measures per turn: total latency, TTFT (streaming), decode tok/s, and the
session-wide prefix-cache hit rate (sampled from /metrics).

Env overrides: VLLM_API_KEY / VLLM_API_KEY_FILE, VLLM_BASE_URL.
Usage:
  python3 bench_agent_trace.py [turns] [prefix_tokens] [--salt SALT]
Defaults: 30 turns, ~20K stable prefix, output 256 tokens/turn.
"""
import argparse
import json
import os
import re
import sys
import time
import urllib.request

KEY = os.environ.get("VLLM_API_KEY") or open(
    os.environ.get("VLLM_API_KEY_FILE", "/mnt/workspace/.bootstrap/vllm_api_key")
).read().strip()
BASE = os.environ.get("VLLM_BASE_URL", "http://127.0.0.1:8000")
MODEL = "deepseek-v4-flash"

SYSTEM_PROMPT = (
    "You are an expert coding agent. Follow the repo conventions, prefer "
    "explicit error handling, and always explain non-obvious choices. "
)
TOOL_SCHEMA = json.dumps({
    "type": "function",
    "function": {
        "name": "read_file",
        "description": "Read a file from the repository",
        "parameters": {
            "type": "object",
            "properties": {"path": {"type": "string"}},
            "required": ["path"],
        },
    },
})
REPO_UNIT = (
    "File utils/parser.go: parse JSON config with strict field validation, "
    "return typed errors on unknown keys. File service/cache.go: LRU cache "
    "with per-entry TTL and single-flight refresh. File handler/auth.go: "
    "OIDC token verification with clock-skew tolerance. "
)


def make_repo_context(target_tokens: int) -> str:
    """Approximately target_tokens of stable repo-context text."""
    return REPO_UNIT * max(1, target_tokens // 40)


def cache_hit_metrics() -> tuple[int, int] | None:
    """(hits, queries) counters from /metrics."""
    try:
        txt = urllib.request.urlopen(BASE + "/metrics", timeout=10).read().decode()
    except Exception:
        return None
    def counter(name: str) -> int:
        m = re.search(r"^vllm:%s_total\S*\s([0-9.e+]+)$" % name, txt, re.M)
        return float(m.group(1)) if m else 0.0
    return counter("prefix_cache_hits"), counter("prefix_cache_queries")


def chat_stream(messages, max_tokens=256, temperature=0.0, salt=None):
    body = {
        "model": MODEL,
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": temperature,
        "stream": True,
        "stream_options": {"include_usage": True},
    }
    if salt:
        body["cache_salt"] = salt
    req = urllib.request.Request(BASE + "/v1/chat/completions",
                                 data=json.dumps(body).encode(), headers={
        "Content-Type": "application/json",
        "Authorization": "Bearer " + KEY,
    })
    t0 = time.time()
    first = None
    n_tokens = 0
    with urllib.request.urlopen(req, timeout=1800) as resp:
        for line in resp:
            line = line.decode("utf-8", "ignore").strip()
            if not line.startswith("data:"):
                continue
            data = line[5:].strip()
            if data == "[DONE]":
                break
            try:
                obj = json.loads(data)
            except Exception:
                continue
            usage = obj.get("usage")
            if usage and usage.get("completion_tokens"):
                n_tokens = usage["completion_tokens"]
            now = time.time()
            if first is None:
                first = now
    total = time.time() - t0
    ttft = (first - t0) if first else total
    decode = total - ttft
    return {
        "total": total,
        "ttft": ttft,
        "n_tokens": n_tokens,
        "decode_tok_s": n_tokens / decode if decode > 0 else 0.0,
    }


def main():
    parser = argparse.ArgumentParser(description="coding-agent multi-turn bench")
    parser.add_argument("turns", nargs="?", type=int, default=30)
    parser.add_argument("prefix_tokens", nargs="?", type=int, default=20000)
    parser.add_argument("--salt", help="per-session cache_salt (multi-tenant isolation)")
    args = parser.parse_args()

    repo_context = make_repo_context(args.prefix_tokens)
    history = [{"role": "system", "content": SYSTEM_PROMPT + TOOL_SCHEMA}]
    results = []
    hits0 = cache_hit_metrics()
    t_session = time.time()

    for turn in range(1, args.turns + 1):
        user_turn = (
            f"[turn {turn}] Analyze handlers/auth.go for a subtle token-refresh "
            f"race and propose a fix with tests."
        )
        messages = list(history) + [{"role": "user", "content": user_turn}]
        # Stable prefix stays first; growing context tail goes after it.
        messages[0]["content"] = SYSTEM_PROMPT + TOOL_SCHEMA + repo_context

        r = chat_stream(messages, max_tokens=256, salt=args.salt)
        results.append(r)

        history.append({"role": "user", "content": user_turn})
        history.append({
            "role": "assistant",
            "content": f"turn-{turn} analysis (placeholder reply)",
        })
        if turn % 4 == 0:
            history.append({
                "role": "tool",
                "content": f"read_file(handlers/auth.go) -> {turn * 37} bytes "
                           f"of source returned",
            })

        cache_state = "hot" if turn > 1 else "cold"
        print(
            f"turn {turn:2d} [{cache_state}]: total {r['total']:5.1f}s | "
            f"TTFT {r['ttft']:5.1f}s | {r['n_tokens']:3d} tok | "
            f"decode {r['decode_tok_s']:5.1f} tok/s",
            flush=True,
        )

    session_s = time.time() - t_session
    hits1 = cache_hit_metrics()
    total_tokens = sum(r["n_tokens"] for r in results)
    avg_ttft = sum(r["ttft"] for r in results[1:]) / max(1, len(results) - 1)
    avg_decode = sum(r["decode_tok_s"] for r in results) / len(results)
    speedup = results[0]["total"] / results[-1]["total"]
    print()
    print(f"=== session summary ({args.turns} turns, ~{args.prefix_tokens} "
          f"ctx prefix, salt={args.salt or 'none'}) ===")
    print(f"session total: {session_s:.1f}s, {total_tokens} completion tokens")
    print(f"avg hot TTFT: {avg_ttft:.2f}s")
    print(f"avg decode: {avg_decode:.1f} tok/s")
    print(f"hot vs cold total latency: {speedup:.1f}x")
    if hits0 and hits1:
        dh = hits1[0] - hits0[0]
        dq = hits1[1] - hits0[1]
        rate = 100 * dh / dq if dq else 0.0
        print(f"session prefix-cache hit rate: {rate:.1f}% "
              f"({dh:.0f}/{dq:.0f} queries) [reference: 94% on agent traces]")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""bench_agent_trace.py — coding-agent multi-turn session benchmark.

Simulates a realistic coding-agent workload against the OpenAI-compatible
endpoint: a large stable prefix (system prompt + tool schema + repo context)
sent every turn, with a growing conversation history and periodic tool
results appended. The stable prefix must hit the vLLM prefix cache; the
growing tail must not.

Measures per turn:
  - total latency (TTFT + decode)
  - TTFT (time to first token, via streaming)
  - decode tok/s
  - aggregate session tok/s
  - prefix-cache effectiveness (hot vs cold total latency)

Env overrides: VLLM_API_KEY / VLLM_API_KEY_FILE, VLLM_BASE_URL.
Usage:
  python3 bench_agent_trace.py [turns] [context_tokens] [tool_call_every_n]
Defaults: 12 turns, ~30K context prefix, tool call every 4th turn.
"""
import json
import os
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


def chat_stream(messages, max_tokens=512, temperature=0.0):
    body = json.dumps({
        "model": MODEL,
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": temperature,
        "stream": True,
        "stream_options": {"include_usage": True},
    }).encode()
    req = urllib.request.Request(BASE + "/v1/chat/completions", data=body, headers={
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
    turns = int(sys.argv[1]) if len(sys.argv) > 1 else 12
    context_tokens = int(sys.argv[2]) if len(sys.argv) > 2 else 30000
    tool_every_n = int(sys.argv[3]) if len(sys.argv) > 3 else 4

    repo_context = make_repo_context(context_tokens)
    history = [{"role": "system", "content": SYSTEM_PROMPT + TOOL_SCHEMA}]
    results = []
    t_session = time.time()

    for turn in range(1, turns + 1):
        user_turn = (
            f"[turn {turn}] Analyze handlers/auth.go for a subtle token-refresh "
            f"race and propose a fix with tests."
        )
        messages = list(history) + [{"role": "user", "content": user_turn}]
        # Stable prefix stays first; growing context tail goes after it.
        messages[0]["content"] = SYSTEM_PROMPT + TOOL_SCHEMA + repo_context

        r = chat_stream(messages, max_tokens=256)
        results.append(r)

        # Append assistant reply + (every tool_every_n) a simulated tool result.
        history.append({"role": "user", "content": user_turn})
        history.append({
            "role": "assistant",
            "content": f"turn-{turn} analysis (placeholder reply)",
        })
        if turn % tool_every_n == 0:
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
    total_tokens = sum(r["n_tokens"] for r in results)
    avg_ttft = sum(r["ttft"] for r in results[1:]) / max(1, len(results) - 1)
    avg_decode = sum(r["decode_tok_s"] for r in results) / len(results)
    speedup = results[0]["total"] / results[-1]["total"]
    print()
    print(f"=== session summary ({turns} turns, ~{context_tokens} ctx prefix) ===")
    print(f"session total: {session_s:.1f}s, {total_tokens} completion tokens")
    print(f"avg hot TTFT: {avg_ttft:.2f}s")
    print(f"avg decode: {avg_decode:.1f} tok/s")
    print(f"hot vs cold total latency: {speedup:.1f}x")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""bench_agent_trace.py — coding-agent multi-turn session benchmark.

Simulates a realistic coding-agent workload against the OpenAI-compatible
endpoint:

  - stable prefix: system prompt + tool schema + repo context
  - growing tail: previous user/assistant turns
  - periodic environment/tool observations folded into the next user turn
  - streaming TTFT + per-request prompt cache accounting

This benchmark intentionally does NOT emit synthetic ``role=tool`` messages.
A role=tool message is only protocol-valid when it corresponds to an earlier
assistant tool_call id. Full tool protocol correctness is exercised separately
by ``bench_tool_roundtrip.py``.

The authoritative cache metric is
``usage.prompt_tokens_details.cached_tokens`` from each request. Engine-wide
/metrics counters are retained only as a diagnostic because concurrent traffic
can make a global counter delta include other clients.

Env overrides: VLLM_API_KEY / VLLM_API_KEY_FILE, VLLM_BASE_URL, VLLM_MODEL.
Usage:
  python3 bench_agent_trace.py [turns] [prefix_tokens] [--salt SALT]
Defaults: 30 turns, ~20K stable prefix, output 256 tokens/turn.
"""
import argparse
import json
import os
import re
import time
import urllib.request

KEY = os.environ.get("VLLM_API_KEY") or open(
    os.environ.get("VLLM_API_KEY_FILE", "/mnt/workspace/.bootstrap/vllm_api_key")
).read().strip()
BASE = os.environ.get("VLLM_BASE_URL", "http://127.0.0.1:8000")
MODEL = os.environ.get("VLLM_MODEL", "deepseek-v4-flash")

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
}, separators=(",", ":"), sort_keys=True)
REPO_UNIT = (
    "File utils/parser.go: parse JSON config with strict field validation, "
    "return typed errors on unknown keys. File service/cache.go: LRU cache "
    "with per-entry TTL and single-flight refresh. File handler/auth.go: "
    "OIDC token verification with clock-skew tolerance. "
)


def make_repo_context(target_tokens: int) -> str:
    """Approximately target_tokens of stable repo-context text."""
    return REPO_UNIT * max(1, target_tokens // 40)


def engine_cache_metrics() -> tuple[float, float] | None:
    """Engine-global (hits, queries) token counters from /metrics."""
    try:
        txt = urllib.request.urlopen(BASE + "/metrics", timeout=10).read().decode()
    except Exception:
        return None

    def counter(name: str) -> float:
        m = re.search(r"^vllm:%s_total\S*\s([0-9.e+]+)$" % name, txt, re.M)
        return float(m.group(1)) if m else 0.0

    return counter("prefix_cache_hits"), counter("prefix_cache_queries")


def _semantic_delta(delta: dict) -> bool:
    """True when a streaming chunk contains model output, not role metadata."""
    return bool(
        delta.get("content")
        or delta.get("reasoning_content")
        or delta.get("tool_calls")
    )


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

    req = urllib.request.Request(
        BASE + "/v1/chat/completions",
        data=json.dumps(body).encode(),
        headers={
            "Content-Type": "application/json",
            "Authorization": "Bearer " + KEY,
        },
    )

    t0 = time.perf_counter()
    first = None
    usage_final = {}
    content_parts = []
    reasoning_parts = []

    with urllib.request.urlopen(req, timeout=1800) as resp:
        for raw in resp:
            line = raw.decode("utf-8", "ignore").strip()
            if not line.startswith("data:"):
                continue
            data = line[5:].strip()
            if data == "[DONE]":
                break
            try:
                obj = json.loads(data)
            except Exception:
                continue

            if obj.get("usage"):
                usage_final = obj["usage"]

            choices = obj.get("choices") or []
            if not choices:
                continue
            delta = choices[0].get("delta") or {}
            if first is None and _semantic_delta(delta):
                first = time.perf_counter()
            if delta.get("content"):
                content_parts.append(delta["content"])
            if delta.get("reasoning_content"):
                reasoning_parts.append(delta["reasoning_content"])

    end = time.perf_counter()
    total = end - t0
    ttft = (first - t0) if first is not None else total
    decode = max(0.0, total - ttft)

    n_tokens = int(usage_final.get("completion_tokens") or 0)
    prompt_tokens = int(usage_final.get("prompt_tokens") or 0)
    details = usage_final.get("prompt_tokens_details") or {}
    cached_raw = details.get("cached_tokens")
    cached_tokens = int(cached_raw) if cached_raw is not None else None

    return {
        "total": total,
        "ttft": ttft,
        "n_tokens": n_tokens,
        "prompt_tokens": prompt_tokens,
        "cached_tokens": cached_tokens,
        "decode_tok_s": n_tokens / decode if decode > 0 else 0.0,
        "content": "".join(content_parts),
        "reasoning": "".join(reasoning_parts),
    }


def fmt_cache(r: dict) -> str:
    cached = r["cached_tokens"]
    prompt = r["prompt_tokens"]
    if cached is None or prompt <= 0:
        return "cache n/a"
    return f"cache {cached}/{prompt} ({100.0 * cached / prompt:5.1f}%)"


def main():
    parser = argparse.ArgumentParser(description="coding-agent multi-turn bench")
    parser.add_argument("turns", nargs="?", type=int, default=30)
    parser.add_argument("prefix_tokens", nargs="?", type=int, default=20000)
    parser.add_argument("--salt", help="per-session cache_salt (multi-tenant isolation)")
    parser.add_argument("--output-tokens", type=int, default=256)
    args = parser.parse_args()

    repo_context = make_repo_context(args.prefix_tokens)
    stable_system = SYSTEM_PROMPT + TOOL_SCHEMA + repo_context
    history = [{"role": "system", "content": stable_system}]
    results = []
    global0 = engine_cache_metrics()
    t_session = time.perf_counter()
    pending_observation = ""

    for turn in range(1, args.turns + 1):
        task = (
            f"[turn {turn}] Analyze handlers/auth.go for a subtle token-refresh "
            "race and propose a fix with tests."
        )
        user_turn = (pending_observation + "\n" + task).strip()
        pending_observation = ""
        messages = list(history) + [{"role": "user", "content": user_turn}]

        r = chat_stream(
            messages,
            max_tokens=args.output_tokens,
            salt=args.salt,
        )
        results.append(r)

        history.append({"role": "user", "content": user_turn})
        # Keep the actual model answer in the growing tail. If a parser exposes
        # only reasoning text, retain that rather than a synthetic unrelated
        # reply.
        answer = r["content"] or r["reasoning"] or f"turn-{turn} completed"
        history.append({"role": "assistant", "content": answer})

        # Emulate a large read/test result without forging an invalid role=tool
        # record. It is presented as an environment observation at the start of
        # the NEXT user turn. Full role=tool semantics are tested elsewhere.
        if turn % 4 == 0:
            pending_observation = (
                "[environment observation from read_file handlers/auth.go] "
                + ("auth middleware source, stack trace, and focused test output; " * 180)
            )

        cache_state = "hot" if turn > 1 else "cold"
        print(
            f"turn {turn:2d} [{cache_state}]: total {r['total']:6.2f}s | "
            f"TTFT {r['ttft']:6.2f}s | {r['n_tokens']:4d} tok | "
            f"decode {r['decode_tok_s']:6.1f} tok/s | {fmt_cache(r)}",
            flush=True,
        )

    session_s = time.perf_counter() - t_session
    global1 = engine_cache_metrics()
    total_tokens = sum(r["n_tokens"] for r in results)
    hot = results[1:] if len(results) > 1 else results
    avg_ttft = sum(r["ttft"] for r in hot) / max(1, len(hot))
    avg_decode = sum(r["decode_tok_s"] for r in results) / max(1, len(results))
    speedup = results[0]["total"] / results[-1]["total"] if results[-1]["total"] else 0.0

    measurable = [
        r for r in results
        if r["cached_tokens"] is not None and r["prompt_tokens"] > 0
    ]
    cached_sum = sum(r["cached_tokens"] for r in measurable)
    prompt_sum = sum(r["prompt_tokens"] for r in measurable)

    print()
    print(
        f"=== session summary ({args.turns} turns, ~{args.prefix_tokens} "
        f"stable-prefix tokens, salt={args.salt or 'none'}) ==="
    )
    print(f"session total: {session_s:.1f}s, {total_tokens} completion tokens")
    print(f"avg hot TTFT: {avg_ttft:.3f}s")
    print(f"avg decode: {avg_decode:.1f} tok/s")
    print(f"hot vs cold total latency: {speedup:.1f}x")

    if prompt_sum:
        print(
            "per-request prefix-cache hit rate: "
            f"{100.0 * cached_sum / prompt_sum:.2f}% "
            f"({cached_sum}/{prompt_sum} prompt tokens cached; "
            f"{len(measurable)}/{len(results)} requests reported details)"
        )
    else:
        print(
            "per-request prefix-cache hit rate: n/a "
            "(server did not return prompt_tokens_details.cached_tokens)"
        )

    if global0 and global1:
        dh = global1[0] - global0[0]
        dq = global1[1] - global0[1]
        rate = 100.0 * dh / dq if dq else 0.0
        print(
            "engine-global cache counters (diagnostic only): "
            f"{rate:.2f}% ({dh:.0f}/{dq:.0f}); may include other clients"
        )


if __name__ == "__main__":
    main()

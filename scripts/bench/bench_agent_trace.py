#!/usr/bin/env python3
"""Coding-agent multi-turn benchmark.

Stable prefix + growing user/assistant history + periodic environment
observations. Full role=tool protocol correctness is tested separately by
bench_tool_roundtrip.py.

Authoritative cache metric:
  usage.prompt_tokens_details.cached_tokens
Engine-global cache counters are diagnostic only.
"""
import argparse
import json
import os
import re
import time
import urllib.request

BASE = os.environ.get("VLLM_BASE_URL", "http://127.0.0.1:8000")
MODEL = os.environ.get("VLLM_MODEL", "deepseek-v4-flash")

SYSTEM_PROMPT = (
    "You are an expert coding agent. Follow repository conventions, prefer "
    "explicit error handling, and explain non-obvious choices. "
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


def api_key() -> str:
    """Resolve credentials only when a request is actually sent.

    Keeping this lazy makes imports and ``--help`` safe on CI/developer hosts
    that intentionally do not have the private DSW key file.
    """
    configured_api_key = os.environ.get("VLLM_API_KEY")
    if configured_api_key:
        return configured_api_key

    api_key_file = os.environ.get("VLLM_API_KEY_FILE")
    if not api_key_file:
        raise RuntimeError(
            "No API key configured; set VLLM_API_KEY or VLLM_API_KEY_FILE"
        )

    try:
        with open(api_key_file, encoding="utf-8") as api_key_stream:
            resolved_api_key = api_key_stream.read().strip()
    except FileNotFoundError as exc:
        raise RuntimeError(
            f"VLLM_API_KEY_FILE does not exist: {api_key_file}"
        ) from exc

    if not resolved_api_key:
        raise RuntimeError(f"VLLM_API_KEY_FILE is empty: {api_key_file}")
    return resolved_api_key


def make_repo_context(target_tokens: int) -> str:
    return REPO_UNIT * max(1, target_tokens // 40)


def engine_cache_metrics() -> tuple[float, float] | None:
    try:
        txt = urllib.request.urlopen(BASE + "/metrics", timeout=10).read().decode()
    except Exception:
        return None

    def counter(name: str) -> float:
        m = re.search(r"^vllm:%s_total\S*\s([0-9.e+]+)$" % name, txt, re.M)
        return float(m.group(1)) if m else 0.0

    return counter("prefix_cache_hits"), counter("prefix_cache_queries")


def _semantic_delta(delta: dict) -> bool:
    return bool(delta.get("content") or delta.get("reasoning_content") or delta.get("tool_calls"))


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
        headers={"Content-Type": "application/json", "Authorization": "Bearer " + api_key()},
    )

    t0 = time.perf_counter()
    first = None
    usage = {}
    content_parts, reasoning_parts = [], []
    with urllib.request.urlopen(req, timeout=1800) as resp:
        for raw in resp:
            line = raw.decode("utf-8", "ignore").strip()
            if not line.startswith("data:"):
                continue
            payload = line[5:].strip()
            if payload == "[DONE]":
                break
            try:
                obj = json.loads(payload)
            except Exception:
                continue
            if obj.get("usage"):
                usage = obj["usage"]
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
    details = usage.get("prompt_tokens_details") or {}
    cached = details.get("cached_tokens")
    n_tokens = int(usage.get("completion_tokens") or 0)
    return {
        "total": total,
        "ttft": ttft,
        "n_tokens": n_tokens,
        "prompt_tokens": int(usage.get("prompt_tokens") or 0),
        "cached_tokens": int(cached) if cached is not None else None,
        "decode_tok_s": n_tokens / decode if decode > 0 else 0.0,
        "content": "".join(content_parts),
        "reasoning": "".join(reasoning_parts),
    }


def history_answer(result: dict, include_reasoning: bool, turn: int) -> str:
    """Model what an upstream harness re-submits on the next turn."""
    if include_reasoning:
        text = "\n".join(x for x in (result["reasoning"], result["content"]) if x).strip()
    else:
        text = result["content"].strip()
    return text or f"[assistant final content unavailable for turn {turn}]"


def fmt_cache(r: dict) -> str:
    if r["cached_tokens"] is None or r["prompt_tokens"] <= 0:
        return "cache n/a"
    return f"cache {r['cached_tokens']}/{r['prompt_tokens']} ({100.0*r['cached_tokens']/r['prompt_tokens']:.1f}%)"


def main():
    p = argparse.ArgumentParser()
    p.add_argument("turns", nargs="?", type=int, default=30)
    p.add_argument("prefix_tokens", nargs="?", type=int, default=20000)
    p.add_argument("--salt")
    p.add_argument("--output-tokens", type=int, default=256)
    p.add_argument(
        "--include-reasoning-history", action="store_true",
        help="replay reasoning_content on following turns (default: content-only)",
    )
    args = p.parse_args()

    stable_system = SYSTEM_PROMPT + TOOL_SCHEMA + make_repo_context(args.prefix_tokens)
    history = [{"role": "system", "content": stable_system}]
    results = []
    global0 = engine_cache_metrics()
    pending_observation = ""
    t_session = time.perf_counter()

    for turn in range(1, args.turns + 1):
        task = (
            f"[turn {turn}] Analyze handlers/auth.go for a subtle token-refresh "
            "race and propose a fix with tests."
        )
        user_turn = (pending_observation + "\n" + task).strip()
        pending_observation = ""
        messages = list(history) + [{"role": "user", "content": user_turn}]
        r = chat_stream(messages, args.output_tokens, 0.0, args.salt)
        results.append(r)

        history.append({"role": "user", "content": user_turn})
        history.append({
            "role": "assistant",
            "content": history_answer(r, args.include_reasoning_history, turn),
        })

        if turn % 4 == 0:
            pending_observation = (
                "[environment observation from read_file handlers/auth.go] "
                + ("auth middleware source, stack trace, and focused test output; " * 180)
            )

        print(
            f"turn {turn:2d} [{'hot' if turn > 1 else 'cold'}]: "
            f"total {r['total']:6.2f}s | TTFT {r['ttft']:6.2f}s | "
            f"{r['n_tokens']:4d} tok | decode {r['decode_tok_s']:6.1f} tok/s | {fmt_cache(r)}",
            flush=True,
        )

    global1 = engine_cache_metrics()
    session_s = time.perf_counter() - t_session
    hot = results[1:] if len(results) > 1 else results
    measurable = [r for r in results if r["cached_tokens"] is not None and r["prompt_tokens"] > 0]
    cached_sum = sum(r["cached_tokens"] for r in measurable)
    prompt_sum = sum(r["prompt_tokens"] for r in measurable)
    avg_ttft = sum(r["ttft"] for r in hot) / max(1, len(hot))
    avg_decode = sum(r["decode_tok_s"] for r in results) / max(1, len(results))

    print()
    print("=== session summary ===")
    print(
        f"turns={args.turns} stable_prefix~{args.prefix_tokens} "
        f"reasoning_history={args.include_reasoning_history} salt={args.salt or 'none'}"
    )
    print(f"session total: {session_s:.1f}s")
    print(f"avg hot TTFT: {avg_ttft:.3f}s")
    print(f"avg decode: {avg_decode:.1f} tok/s")
    if prompt_sum:
        print(
            f"per-request prefix-cache hit: {100.0*cached_sum/prompt_sum:.2f}% "
            f"({cached_sum}/{prompt_sum}; {len(measurable)}/{len(results)} requests reported details)"
        )
    else:
        print("per-request prefix-cache hit: n/a")
    if global0 and global1:
        dh, dq = global1[0] - global0[0], global1[1] - global0[1]
        print(
            f"engine-global cache counters (diagnostic): "
            f"{(100.0*dh/dq if dq else 0.0):.2f}% ({dh:.0f}/{dq:.0f})"
        )


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Validate DeepSeek-V4 streaming tool calls and full round trips.

Exercises:
  assistant(streamed tool_calls) -> role=tool result -> assistant final answer

Supports forced/required/auto tool selection, long stable prefixes, and
concurrent independent rounds. This directly targets parser failures that only
appear under long context or concurrent streaming load.
"""
from __future__ import annotations

import argparse
import concurrent.futures
from dataclasses import dataclass
import json
import os
import statistics
import time
import urllib.request

BASE = os.environ.get("VLLM_BASE_URL", "http://127.0.0.1:8000")
MODEL = os.environ.get("VLLM_MODEL", "deepseek-v4-flash")

TOOLS = [{
    "type": "function",
    "function": {
        "name": "read_file",
        "description": "Read a UTF-8 source file from the current repository.",
        "parameters": {
            "type": "object",
            "properties": {"path": {"type": "string"}},
            "required": ["path"],
            "additionalProperties": False,
        },
    },
}]
FORCED_TOOL = {"type": "function", "function": {"name": "read_file"}}
REPO_UNIT = (
    "Repository convention: Go services use context.Context, typed errors, "
    "table-driven tests, and explicit cancellation. handlers/auth.go validates "
    "OIDC tokens and refreshes JWKS through a single-flight cache. "
)
DSML_HINTS = ("DSML", "invoke name=", "tool_calls>", "parameter name=")


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
        with open(api_key_file, encoding="utf-8") as api_key_stream:
            resolved_api_key = api_key_stream.read().strip()
    except FileNotFoundError as exc:
        raise RuntimeError(
            f"VLLM_API_KEY_FILE does not exist: {api_key_file}"
        ) from exc

    if not resolved_api_key:
        raise RuntimeError(f"VLLM_API_KEY_FILE is empty: {api_key_file}")
    return resolved_api_key


def stable_system(prefix_tokens: int) -> str:
    repeats = max(1, prefix_tokens // 36)
    return (
        "You are a coding agent. Use tools when requested. Never invent file "
        "contents; after a tool result, answer from that result.\n" + REPO_UNIT * repeats
    )


@dataclass
class StreamResult:
    total_s: float
    ttft_s: float
    content: str
    reasoning: str
    tool_calls: list[dict]
    prompt_tokens: int
    completion_tokens: int
    cached_tokens: int | None


def semantic_delta(delta: dict) -> bool:
    return bool(delta.get("content") or delta.get("reasoning_content") or delta.get("tool_calls"))


def stream_chat(body: dict) -> StreamResult:
    body = dict(body)
    body.update({
        "model": MODEL,
        "stream": True,
        "stream_options": {"include_usage": True},
        "temperature": 0.0,
    })
    req = urllib.request.Request(
        BASE + "/v1/chat/completions",
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json", "Authorization": "Bearer " + api_key()},
    )

    t0 = time.perf_counter()
    first = None
    usage = {}
    content_parts: list[str] = []
    reasoning_parts: list[str] = []
    calls: dict[int, dict] = {}

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
            except json.JSONDecodeError:
                continue
            if obj.get("usage"):
                usage = obj["usage"]
            choices = obj.get("choices") or []
            if not choices:
                continue
            delta = choices[0].get("delta") or {}
            if first is None and semantic_delta(delta):
                first = time.perf_counter()
            if delta.get("content"):
                content_parts.append(delta["content"])
            if delta.get("reasoning_content"):
                reasoning_parts.append(delta["reasoning_content"])
            for tc in delta.get("tool_calls") or []:
                idx = int(tc.get("index", 0))
                acc = calls.setdefault(
                    idx,
                    {"id": "", "type": "function", "function": {"name": "", "arguments": ""}},
                )
                if tc.get("id"):
                    acc["id"] += tc["id"]
                fn = tc.get("function") or {}
                if fn.get("name"):
                    acc["function"]["name"] += fn["name"]
                if fn.get("arguments"):
                    acc["function"]["arguments"] += fn["arguments"]

    end = time.perf_counter()
    details = usage.get("prompt_tokens_details") or {}
    cached = details.get("cached_tokens")
    return StreamResult(
        total_s=end - t0,
        ttft_s=(first - t0) if first is not None else end - t0,
        content="".join(content_parts),
        reasoning="".join(reasoning_parts),
        tool_calls=[calls[i] for i in sorted(calls)],
        prompt_tokens=int(usage.get("prompt_tokens") or 0),
        completion_tokens=int(usage.get("completion_tokens") or 0),
        cached_tokens=int(cached) if cached is not None else None,
    )


def cache_text(r: StreamResult) -> str:
    if r.cached_tokens is None or not r.prompt_tokens:
        return "cache=n/a"
    return f"cache={100.0*r.cached_tokens/r.prompt_tokens:.1f}%"


def tool_choice(mode: str):
    return FORCED_TOOL if mode == "forced" else mode


def validate_tool_call(r: StreamResult) -> tuple[bool, str]:
    if len(r.tool_calls) != 1:
        leaked = any(x in r.content for x in DSML_HINTS)
        suffix = " (raw DSML-like content leaked)" if leaked else ""
        return False, f"expected 1 tool call, got {len(r.tool_calls)}{suffix}"
    tc = r.tool_calls[0]
    if tc["function"]["name"] != "read_file":
        return False, f"wrong function {tc['function']['name']!r}"
    try:
        args = json.loads(tc["function"]["arguments"])
    except json.JSONDecodeError as exc:
        return False, f"invalid tool JSON: {exc}"
    if args.get("path") != "handlers/auth.go":
        return False, f"wrong path {args.get('path')!r}"
    if not tc.get("id"):
        return False, "missing tool_call id"
    return True, "ok"


def run_round(i: int, system: str, mode: str, salt: str | None) -> dict:
    common = {"cache_salt": salt} if salt else {}
    request1 = {
        **common,
        "messages": [
            {"role": "system", "content": system},
            {
                "role": "user",
                "content": (
                    f"Round {i}: you must inspect handlers/auth.go before answering. "
                    "Call read_file with the exact path handlers/auth.go now. Do not "
                    "describe the call in prose; actually call the tool."
                ),
            },
        ],
        "tools": TOOLS,
        "tool_choice": tool_choice(mode),
        "max_tokens": 256,
    }

    try:
        r1 = stream_chat(request1)
    except Exception as exc:
        return {"i": i, "ok": False, "why": f"request1 error: {exc}"}
    ok, why = validate_tool_call(r1)
    if not ok:
        return {"i": i, "ok": False, "why": why, "r1": r1}

    tc = r1.tool_calls[0]
    tool_result = (
        "package handlers\n\n"
        "func VerifyToken(ctx context.Context, raw string) error {\n"
        "    keys, err := jwks.Get(ctx)\n"
        "    if err != nil { return fmt.Errorf(\"jwks: %w\", err) }\n"
        "    return verify(raw, keys)\n"
        "}\n"
    )
    request2 = {
        **common,
        "messages": request1["messages"] + [
            {"role": "assistant", "content": r1.content or None, "tool_calls": [tc]},
            {"role": "tool", "tool_call_id": tc["id"], "content": tool_result},
        ],
        "tools": TOOLS,
        "tool_choice": "none",
        "max_tokens": 256,
    }
    try:
        r2 = stream_chat(request2)
    except Exception as exc:
        return {"i": i, "ok": False, "why": f"request2 error: {exc}", "r1": r1}
    final_text = (r2.content or r2.reasoning).strip()
    if not final_text:
        return {"i": i, "ok": False, "why": "empty post-tool answer", "r1": r1, "r2": r2}
    return {"i": i, "ok": True, "why": "ok", "r1": r1, "r2": r2}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--rounds", type=int, default=10)
    ap.add_argument("--prefix-tokens", type=int, default=20000)
    ap.add_argument("--mode", choices=("forced", "required", "auto"), default="forced")
    ap.add_argument("--concurrency", type=int, default=1)
    ap.add_argument("--salt", default=None)
    args = ap.parse_args()
    if args.rounds < 1 or args.concurrency < 1:
        ap.error("rounds and concurrency must be positive")

    system = stable_system(args.prefix_tokens)
    if args.concurrency == 1:
        results = [run_round(i, system, args.mode, args.salt) for i in range(1, args.rounds + 1)]
    else:
        with concurrent.futures.ThreadPoolExecutor(max_workers=args.concurrency) as pool:
            futures = [pool.submit(run_round, i, system, args.mode, args.salt) for i in range(1, args.rounds + 1)]
            results = [f.result() for f in futures]
        results.sort(key=lambda x: x["i"])

    passed = 0
    tool_ttfts, answer_ttfts = [], []
    for x in results:
        i = x["i"]
        if not x["ok"]:
            print(f"round {i:02d}: FAIL | {x['why']}")
            continue
        passed += 1
        r1, r2 = x["r1"], x["r2"]
        tool_ttfts.append(r1.ttft_s)
        answer_ttfts.append(r2.ttft_s)
        print(
            f"round {i:02d}: PASS | tool TTFT={r1.ttft_s:.3f}s {cache_text(r1)} | "
            f"post-tool TTFT={r2.ttft_s:.3f}s {cache_text(r2)}"
        )

    print()
    print(
        f"tool round-trip survival: {passed}/{args.rounds} | mode={args.mode} "
        f"prefix~{args.prefix_tokens} concurrency={args.concurrency}"
    )
    if tool_ttfts:
        print(f"tool-call TTFT median={statistics.median(tool_ttfts):.3f}s max={max(tool_ttfts):.3f}s")
        print(f"post-tool TTFT median={statistics.median(answer_ttfts):.3f}s max={max(answer_ttfts):.3f}s")
    return 0 if passed == args.rounds else 1


if __name__ == "__main__":
    raise SystemExit(main())

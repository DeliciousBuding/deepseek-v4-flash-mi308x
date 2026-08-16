#!/usr/bin/env python3
"""Validate streaming DeepSeek-V4 tool calls and the full tool round trip.

The benchmark exercises the exact protocol shape used by coding agents:

  assistant(streamed tool_calls) -> tool result -> assistant final answer

It checks fragmented streaming tool-call deltas, JSON arguments, TTFT to the
first semantic delta, and per-request prefix-cache accounting. The default run
uses ten independent rounds sharing a stable system/repository prefix.

Usage:
  python3 scripts/bench/bench_tool_roundtrip.py [--rounds 10] [--prefix-tokens 20000]

Environment:
  VLLM_BASE_URL, VLLM_API_KEY / VLLM_API_KEY_FILE, VLLM_MODEL
"""
from __future__ import annotations

import argparse
import json
import os
import time
import urllib.request
from dataclasses import dataclass

BASE = os.environ.get("VLLM_BASE_URL", "http://127.0.0.1:8000")
MODEL = os.environ.get("VLLM_MODEL", "deepseek-v4-flash")
KEY = os.environ.get("VLLM_API_KEY") or open(
    os.environ.get("VLLM_API_KEY_FILE", "/mnt/workspace/.bootstrap/vllm_api_key")
).read().strip()

TOOLS = [
    {
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
    }
]
FORCED_TOOL = {"type": "function", "function": {"name": "read_file"}}
REPO_UNIT = (
    "Repository convention: Go services use context.Context, typed errors, "
    "table-driven tests, and explicit cancellation. handlers/auth.go validates "
    "OIDC tokens and refreshes JWKS through a single-flight cache. "
)


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
        headers={
            "Content-Type": "application/json",
            "Authorization": "Bearer " + KEY,
        },
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
    return f"cache={100.0 * r.cached_tokens / r.prompt_tokens:.1f}%"


def validate_tool_call(r: StreamResult) -> tuple[bool, str, dict | None]:
    if len(r.tool_calls) != 1:
        return False, f"expected 1 tool call, got {len(r.tool_calls)}", None
    tc = r.tool_calls[0]
    if tc["function"]["name"] != "read_file":
        return False, f"wrong function {tc['function']['name']!r}", None
    try:
        args = json.loads(tc["function"]["arguments"])
    except json.JSONDecodeError as exc:
        return False, f"invalid tool JSON: {exc}", None
    if args.get("path") != "handlers/auth.go":
        return False, f"wrong path {args.get('path')!r}", args
    if not tc.get("id"):
        return False, "missing tool_call id", args
    return True, "ok", args


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--rounds", type=int, default=10)
    ap.add_argument("--prefix-tokens", type=int, default=20000)
    ap.add_argument("--salt", default=None, help="optional cache_salt for the whole fixture")
    args = ap.parse_args()

    system = stable_system(args.prefix_tokens)
    passed = 0
    tool_ttfts: list[float] = []
    answer_ttfts: list[float] = []

    for i in range(1, args.rounds + 1):
        common = {"cache_salt": args.salt} if args.salt else {}
        request1 = {
            **common,
            "messages": [
                {"role": "system", "content": system},
                {
                    "role": "user",
                    "content": (
                        f"Round {i}: inspect handlers/auth.go. Call read_file with "
                        "that exact path before answering."
                    ),
                },
            ],
            "tools": TOOLS,
            "tool_choice": FORCED_TOOL,
            "max_tokens": 256,
        }
        r1 = stream_chat(request1)
        ok, why, _ = validate_tool_call(r1)
        if not ok:
            print(f"round {i:02d}: FAIL tool-call: {why}")
            continue

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
            "messages": request1["messages"]
            + [
                {
                    "role": "assistant",
                    "content": r1.content or None,
                    "tool_calls": [tc],
                },
                {
                    "role": "tool",
                    "tool_call_id": tc["id"],
                    "content": tool_result,
                },
            ],
            "tools": TOOLS,
            "tool_choice": "none",
            "max_tokens": 256,
        }
        r2 = stream_chat(request2)
        final_text = (r2.content or r2.reasoning).strip()
        ok2 = bool(final_text)
        if ok2:
            passed += 1
        tool_ttfts.append(r1.ttft_s)
        answer_ttfts.append(r2.ttft_s)
        print(
            f"round {i:02d}: {'PASS' if ok2 else 'FAIL'} | "
            f"tool TTFT={r1.ttft_s:.3f}s {cache_text(r1)} | "
            f"answer TTFT={r2.ttft_s:.3f}s {cache_text(r2)} | "
            f"tool={tc['function']['name']}"
        )

    print()
    print(f"tool round-trip survival: {passed}/{args.rounds}")
    if tool_ttfts:
        print(f"avg tool-call TTFT: {sum(tool_ttfts) / len(tool_ttfts):.3f}s")
        print(f"avg post-tool TTFT: {sum(answer_ttfts) / len(answer_ttfts):.3f}s")
    return 0 if passed == args.rounds else 1


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Concurrent multi-session coding-agent benchmark.

Keeps N independent growing agent histories alive for several rounds, with
periodic environment observations and tool/IO idle windows. Measures cache
retention, TTFT tails, decode rate and aggregate completion throughput.

This performance fixture never forges role=tool messages. Full tool protocol
correctness is tested by bench_tool_roundtrip.py.
"""
from __future__ import annotations

import argparse
import concurrent.futures
import statistics
import time

from bench_agent_trace import SYSTEM_PROMPT, chat_stream, history_answer, make_repo_context


def percentile(values: list[float], q: float) -> float:
    if not values:
        return 0.0
    xs = sorted(values)
    idx = max(0, min(len(xs) - 1, round((len(xs) - 1) * q)))
    return xs[idx]


def cache_ratio(result: dict) -> float | None:
    if result["cached_tokens"] is None or result["prompt_tokens"] <= 0:
        return None
    return result["cached_tokens"] / result["prompt_tokens"]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sessions", type=int, default=4)
    ap.add_argument("--rounds", type=int, default=8)
    ap.add_argument("--prefix-tokens", type=int, default=20000)
    ap.add_argument("--output-tokens", type=int, default=128)
    ap.add_argument("--tool-pause", type=float, default=0.5,
                    help="seconds between rounds, emulating tool/IO idle time")
    ap.add_argument("--isolate-cache", action="store_true",
                    help="use a distinct cache_salt per session")
    ap.add_argument("--include-reasoning-history", action="store_true",
                    help="replay reasoning_content into following prompts")
    args = ap.parse_args()

    if args.sessions < 1 or args.rounds < 1:
        ap.error("sessions and rounds must be positive")

    repo = make_repo_context(args.prefix_tokens)
    states = []
    for sid in range(args.sessions):
        system = (
            SYSTEM_PROMPT
            + repo
            + f"\nRepository/session identity: repo-{sid:02d}. "
            "Do not assume files from other repositories."
        )
        states.append({
            "history": [{"role": "system", "content": system}],
            "salt": f"agent-session-{sid:02d}" if args.isolate_cache else None,
            "pending_observation": "",
        })

    all_results: list[dict] = []
    hot_ttfts: list[float] = []
    wall0 = time.perf_counter()

    with concurrent.futures.ThreadPoolExecutor(max_workers=args.sessions) as pool:
        for turn in range(1, args.rounds + 1):
            futures = []
            pending_users = []
            for sid, state in enumerate(states):
                task = (
                    f"[repo-{sid:02d} turn {turn}] Inspect the auth/cache boundary, "
                    "identify one race or cancellation bug, and propose a focused test."
                )
                user = (state["pending_observation"] + "\n" + task).strip()
                state["pending_observation"] = ""
                messages = list(state["history"]) + [{"role": "user", "content": user}]
                pending_users.append(user)
                futures.append(pool.submit(
                    chat_stream,
                    messages,
                    args.output_tokens,
                    0.0,
                    state["salt"],
                ))

            round_results = [f.result() for f in futures]
            all_results.extend(round_results)
            round_ttft = [r["ttft"] for r in round_results]
            if turn > 1:
                hot_ttfts.extend(round_ttft)

            known = [r for r in round_results if cache_ratio(r) is not None]
            cached = sum(r["cached_tokens"] for r in known)
            prompt = sum(r["prompt_tokens"] for r in known)
            cache_text = f"{100.0*cached/prompt:.1f}%" if prompt else "n/a"
            print(
                f"round {turn:02d}: completion={sum(r['n_tokens'] for r in round_results):4d} tok | "
                f"TTFT p50={statistics.median(round_ttft):.3f}s "
                f"p95={percentile(round_ttft, 0.95):.3f}s | cache={cache_text}",
                flush=True,
            )

            for sid, (state, r) in enumerate(zip(states, round_results)):
                state["history"].append({"role": "user", "content": pending_users[sid]})
                state["history"].append({
                    "role": "assistant",
                    "content": history_answer(r, args.include_reasoning_history, turn),
                })
                if turn % 2 == 0:
                    state["pending_observation"] = (
                        f"[environment observation for repo-{sid:02d}] "
                        + ("auth.go test output, grep matches, and stack trace; " * 160)
                    )

            if args.tool_pause > 0 and turn != args.rounds:
                time.sleep(args.tool_pause)

    wall = time.perf_counter() - wall0
    completion = sum(r["n_tokens"] for r in all_results)
    measurable = [r for r in all_results if cache_ratio(r) is not None]
    cached_total = sum(r["cached_tokens"] for r in measurable)
    measured_prompt = sum(r["prompt_tokens"] for r in measurable)
    decode_rates = [r["decode_tok_s"] for r in all_results if r["decode_tok_s"] > 0]

    print()
    print("=== concurrent agent session summary ===")
    print(
        f"sessions={args.sessions} rounds={args.rounds} isolate_cache={args.isolate_cache} "
        f"reasoning_history={args.include_reasoning_history} tool_pause={args.tool_pause}s"
    )
    print(f"wall={wall:.2f}s completion={completion} tok aggregate={completion/wall:.1f} tok/s")
    if measured_prompt:
        print(
            f"per-request cache hit={100.0*cached_total/measured_prompt:.2f}% "
            f"({cached_total}/{measured_prompt}; {len(measurable)}/{len(all_results)} requests reported details)"
        )
    else:
        print("per-request cache hit=n/a")
    if hot_ttfts:
        print(
            f"hot TTFT p50={statistics.median(hot_ttfts):.3f}s "
            f"p95={percentile(hot_ttfts,0.95):.3f}s max={max(hot_ttfts):.3f}s"
        )
    if decode_rates:
        print(f"per-stream decode median={statistics.median(decode_rates):.1f} tok/s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

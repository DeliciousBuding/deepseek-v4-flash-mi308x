# Performance — DeepSeek-V4-Flash-0731 on a single MI308X

This file separates **local measurements** from **community reference numbers**.
The stable defaults remain frozen until an A/B improves the end-to-end coding-agent
workload and still passes the 500K/context + tool-call correctness gates. See
[`GPU_VALIDATION_PLAN.md`](GPU_VALIDATION_PLAN.md) for the next GPU run.

## Stable runtime provenance

```text
GPU    : MI308X / gfx942 / 192 GB class
vLLM   : 0.26.1rc1.dev306+gcb8104839.rocm723
AITER  : 0.1.19
flydsl : 0.2.4
torch  : 2.11.0+gitd0c8b1f, ROCm build
ROCm   : 7.2.3
patch source:
  ryanzhou/deepseek-v4-flash-mi300x
  012b9945c1e61ec7a7c7de12da58e8c7cafd92ab
```

As of 2026-08-16 that exact patch-source commit is also the current ryanzhou
`main` commit. The local serving runtime is nevertheless **not byte-equivalent**
to upstream production: upstream applies the overlays to vLLM `dev229`, while
this recipe ports them onto `dev306`, adds the small `activation=None` signature
compatibility edit required by the newer caller, and adds the local sandbox
`shared_offload_region.madvise-tolerant.py` patch. The full upstream SHA is still
pinned so a future branch move cannot silently change a reproducible install.

## Stable serve profile (v2)

```text
--max-model-len 524288
--kv-cache-dtype fp8_ds_mla
--block-size 256
--enable-prefix-caching
--kv-cache-memory-bytes 16000000000
--kv-offloading-size 12
--kv-offloading-backend native
--max-num-seqs 64
--max-num-batched-tokens 4096
--long-prefill-token-threshold 1024
--moe-backend triton
--enable-expert-parallel
--tokenizer-mode deepseek_v4
--reasoning-parser deepseek_v4
--tool-call-parser deepseek_v4
--enable-auto-tool-choice
--enable-prompt-tokens-details
--speculative-config {
  "method":"dspark",
  "num_speculative_tokens":7,
  "draft_sample_method":"probabilistic",
  "rejection_sample_method":"block"
}
--gpu-memory-utilization 0.95
```

The launcher exposes the main A/B knobs through environment variables without
changing those defaults: `MAX_MODEL_LEN`, `MAX_NUM_SEQS`,
`MAX_BATCHED_TOKENS`, `LONG_PREFILL_TOKEN_THRESHOLD`, `DSPARK_ENABLED`,
`DSPARK_K`, `KV_OFFLOAD_GB`, `KV_CACHE_BYTES`, `GPU_MEMORY_UTILIZATION`, and
`MOE_BACKEND`.

## Local long-context validation

| Target | Measured prompt tokens | Total time | Result |
|---:|---:|---:|---|
| 50K | 47,505 | 14.7s | pass |
| 128K | 121,605 | 27.4s | pass |
| 256K | 243,205 | 71.6s | pass |
| 384K | 364,805 | 73.4s | pass |
| 500K | 475,005 | 91.7s | **pass** |

The ladder intentionally shares a prefix, so the later rows are not fresh-prefill
benchmarks. Supporting a 524,288-token maximum does not allocate 512K KV to each
short request; nevertheless, the next GPU run explicitly A/Bs the configured
upper bound because planner/scheduler structures can still have measurable cost.

## Local prefix-cache and coding-agent results

Simple repeated-prefix fixture (~18K-token prefix):

| Scenario | Time |
|---|---:|
| cold | 8.67s |
| hot | 0.49s |
| speedup | **17.5x** |

vLLM APC matches token-prefix blocks, not a conversation ID. Keep stable system
instructions, tool schemas and repository context first; append changing history
and tool output after them.

### Cache-accounting correction

An earlier 30-turn report quoted **95.9%** using engine-global Prometheus counter
deltas. Those counters remain useful for engine trends but can include concurrent
clients. Immediately before the last disconnect, the agent trace was re-run using
each response's `usage.prompt_tokens_details.cached_tokens`: **856K / 860K prompt
tokens cached = 99.4%**, with hot TTFT around **0.21 s** in that run.

`scripts/bench/bench_agent_trace.py` now makes per-request `cached_tokens` the
authoritative metric. The updated harness also defaults to **not** replaying
`reasoning_content` into the next prompt, which better models harnesses that keep
hidden reasoning out of conversation history; `--include-reasoning-history`
provides the explicit comparison mode.

Historical coding-agent points:

| Metric | Last measured local result |
|---|---:|
| repeated-prefix speedup | 17.5x |
| per-request cached prompt tokens | **99.4% (856K/860K)** |
| hot TTFT | ~0.21–0.32s depending on trace revision / warm state |
| earlier 30-turn average decode | ~119 tok/s |
| streaming tool-call survival | **10/10** before disconnect |

The repository now contains:

- `bench_agent_trace.py` — one growing agent session;
- `bench_session_concurrency.py` — independent long-lived sessions with idle/tool windows;
- `bench_tool_roundtrip.py` — actual assistant tool call -> role=tool -> final answer,
  with forced/required/auto modes, long prefix and concurrent parser stress.

Synthetic performance traces do not forge `role=tool` messages without a matching
assistant tool-call ID.

## Local TTFT isolation

A short request injected during a 200K-token prefill received its first semantic
token in **1.96s vs 1.91s alone (+0.04s)**. The 1,024-token long-prefill cap kept
the long cold request from monopolizing the scheduler.

## Local concurrency

| Concurrency | Aggregate tok/s | Approx. per-stream tok/s |
|---:|---:|---:|
| C1 | 128.2 | 128.2 |
| C2 | 197.3 | 98.7 |
| C4 | 286.2 | 71.5 |
| C8 | **446.2** | 55.8 |

This fixed the initial regression where an early microbenchmark produced only
~63 tok/s aggregate at C2 and ~81 at C4. The v2 profile scales monotonically.

## Local single-stream decode

| Output | Total time incl. TTFT | tok/s incl. TTFT |
|---:|---:|---:|
| 128 | 1.30s | 98.2 |
| 512 | 3.83s | **133.5** |

A separate agent run measured roughly ~142 tok/s pure decode after TTFT.

## Current upstream reference (2026-08-16)

Current ryanzhou `main` is commit
`012b9945c1e61ec7a7c7de12da58e8c7cafd92ab`. Its README reports a production
runtime based on vLLM `0.26.1rc1.dev229+g124154a88.rocm723` + AITER 0.1.19:

| Metric | Current public upstream result |
|---|---:|
| uncached C1 prefill | **11.69K tok/s steady** (11.53K median) |
| static DSpark-7 C1 | 152.6 tok/s aggregate, **158.8 tok/s median per stream** |
| native non-spec C1 | 67.3 tok/s aggregate |
| C64 burst | 1,278 tok/s aggregate (K7) |
| context | **384K validated** (architecture supports 1M) |
| GPU KV | 16 GB fp8_ds_mla + 96 GiB native CPU tier |
| scheduler | 4,096-token budget; 384 reserved for speculative work, up to 3,712 ordinary prefill |

The same source overlays can behave differently because the runtime base differs.
Our project also has a 16 GB `/dev/shm` sandbox constraint, only a 12 GB CPU-KV
tier, and a required ~500K context ceiling.

## Tuning history

### v1 -> v2

Major changes:

- pinned 16 GB GPU KV pool + 12 GB native CPU KV tier;
- `--max-num-seqs 64` (was 8);
- stale `/dev/shm/vllm_offload_*.mmap` cleanup before launch;
- correct DeepSeek V4 tokenizer/reasoning/tool parser path;
- DSpark probabilistic drafting + block rejection;
- Triton MoE + AITER tuning tables;
- `--long-prefill-token-threshold 1024`;
- sandbox `MADV_POPULATE_WRITE` fallback patch.

Measured direction: decode-512 90.6 -> 133.5, C1 76 -> 128,
C4 177 -> 286, C8 372 -> 446, repeated-prefix speedup ~14x -> 17.5x.

### Rejected experiments on dev306

| Experiment | Result | Verdict |
|---|---|---|
| FULL_AND_PIECEWISE graph capture through 3712 | cold prefill 2509 vs 3222 tok/s; decode flat; ~+10 GB HBM; slower startup | reject |
| same capture extended through 3840/4096 | cold prefill 2504 tok/s; decode flat | reject |
| DSpark K=5 vs K=7 | 121.7 vs 133.5 tok/s at decode-512 | keep K=7 for C1 latency |
| AITER M=1/2 source tuning | tuning harness absent from shipped wheel | defer to separate source/CK environment |

Do not repeat the two graph experiments without changing the runtime base or
kernel stack.

## Next GPU run

The ordered matrix is in [`GPU_VALIDATION_PLAN.md`](GPU_VALIDATION_PLAN.md):

1. runtime/patch audit and default-profile correctness regression;
2. short/100K/200K + concurrent streaming tool parser gates;
3. reasoning-history policy comparison;
4. scheduler budget 2048/3072/4096/8192 A/B;
5. static DSpark K7 vs native decode at C1/C2/C4/C8, with acceptance metrics;
6. DSpark x CPU-KV prefix-cache matrix;
7. 256K/384K/512K configured-bound overhead check;
8. long-context recall, then a separate dev229-runtime experiment if useful.

The winning configuration is the one that improves a complete multi-turn coding
agent session while preserving 500K correctness, streaming tool calls, cache
retention, tail TTFT and restart reproducibility — not whichever isolated kernel
prints the largest tok/s number.

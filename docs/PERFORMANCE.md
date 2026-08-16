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
| 50K | 47,505 | 16.7s | pass |
| 128K | 121,605 | 25.3s | pass |
| 256K | 243,205 | 52.3s | pass |
| 384K | 364,805 | 66.2s | pass |
| 500K | 475,005 | **75.3s** | **pass** |

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

Engine-global Prometheus deltas are diagnostic only because they can include
concurrent clients. The current 30-turn gate uses each response's
`usage.prompt_tokens_details.cached_tokens`: **1,030,144 / 1,079,154 = 95.46%**.
Ordinary hot turns were roughly **0.20–0.36s TTFT**; turns that deliberately append
a large new environment/tool observation rise to ~0.8–0.9s and then recover.
A previous warm-state trace reached 99.4%; both measurements use the same
per-request accounting, but the current 30-turn run is the promotion headline.

`scripts/bench/bench_agent_trace.py` now makes per-request `cached_tokens` the
authoritative metric. The updated harness also defaults to **not** replaying
`reasoning_content` into the next prompt, which better models harnesses that keep
hidden reasoning out of conversation history; `--include-reasoning-history`
provides the explicit comparison mode.

Current coding-agent gate summary:

| Metric | Current production result |
|---|---:|
| 30-turn per-request cached prompt tokens | **95.46%** |
| ordinary hot TTFT | **~0.20–0.36s** |
| average hot-trace decode | **167.3 tok/s** |
| auto tool-call survival | **100K 5/5; 200K 3/3** |
| true-cold 200K isolation | **DEGRADED: +~2.1s short TTFT** |

The repository now contains:

- `bench_agent_trace.py` — one growing agent session;
- `bench_session_concurrency.py` — independent long-lived sessions with idle/tool windows;
- `bench_tool_roundtrip.py` — actual assistant tool call -> role=tool -> final answer,
  with forced/required/auto modes, long prefix and concurrent parser stress.

Synthetic performance traces do not forge `role=tool` messages without a matching
assistant tool-call ID.

## Local TTFT isolation

The old `+0.04s` result was **cache-contaminated**: the benchmark reused the same
deterministic 200K prefix, so an immediate rerun could turn the long request from
~45s into ~0.6s via APC. `bench_ttft_isolation.py` now prepends a random nonce by
default, forcing a genuinely cold prefix (`--reuse-prefix` is diagnostic only).

With the production scheduler and the promoted 80-CU tuning tables, a true-cold
200K run measured **0.05s alone vs 2.12s during prefill (+2.07s)**. This remains a
known tail-latency issue. Two attempted fixes were rejected: forcing the 1,024
chunk cap even for a solo long request made the long request slower and the short
overhead ~+3.08s; adding extra M~1024/1031/1032 GEMM tuning did not solve the
end-to-end scheduler isolation problem.

## Local concurrency

| Concurrency | Aggregate tok/s | Approx. per-stream tok/s |
|---:|---:|---:|
| C1 | **128.9** | 128.9 |
| C2 | **235.8** | 117.9 |
| C4 | **375.3** | 93.8 |
| C8 | **548.3** | 68.5 |

This fixed the initial regression where an early microbenchmark produced only
~63 tok/s aggregate at C2 and ~81 at C4. The v2 profile scales monotonically.

## Local single-stream decode

| Output | Total time incl. TTFT | tok/s incl. TTFT |
|---:|---:|---:|
| 128 | ~1.1s | 119.1 |
| 512 | ~3.6s | **140.8** |

A fixed 512-token fixture repeated three more times at **141.4 / 141.4 / 141.3
tok/s**, with identical DSpark acceptance, so ~141 tok/s is the current stable
low-concurrency control rather than a one-off wall-clock spike.

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

## MI308X 80-CU AITER tuning

The largest new finding in this GPU session was hardware-key mismatch rather than
a vLLM scheduler bug. `torch`, `rocminfo` and AITER all report this MI308X as
`gfx942` with **80 compute units**. The inherited ryanzhou MI300X tuning CSVs are
keyed with `cu_num=304`; because AITER lookup includes both `gfx` and `cu_num`,
those rows silently missed and the service was largely running default A8W8 GEMM
kernels.

The repository now carries two measured tables:

- `tuning/dsv4-mi308x-80cu-a8w8-blockscale-bpreshuffle.csv` — **13 rows**;
- `tuning/dsv4-mi308x-80cu-a8w8-blockscale.csv` — **24 rows**.

They cover the production C1 (`M=7/8`), C8 (`M=56/64`) and long-prefill
(`M=4096`) shapes that won production-operator A/Bs with numerical checks. Typical
operator gains included ~59–63% latency reduction for key C1 bpreshuffle GEMMs,
~16–21% for C1 standard GEMMs, and ~45–78% for several C8/prefill standard
GEMMs. End-to-end gains are smaller because GEMM is only part of the request:
~141.4 tok/s single-stream and 548.3 tok/s C8 are the measured service-level
results.

AITER 0.1.19's tuner compare path has a same-process native-extension reload
trap: after rebuilding a candidate `.so`, the Python process may still hold the
old extension registry. Candidate rows that reported "kernel not present" were
therefore revalidated in a **fresh Python process** before promotion. Only rows
with numerical PASS and measured production-operator benefit remain in the
production CSVs.

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
| Force 1024 cap for solo long prefill | 200K total ~62s and short overhead ~+3.08s | reject; production scheduler unchanged |
| MI308X 1K-shape expansion | operator wins, but true-cold 200K isolation did not improve (+2.94s in one run) | keep out of production table |

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

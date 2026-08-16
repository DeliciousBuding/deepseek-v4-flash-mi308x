# GPU validation and tuning plan

This document is the runbook for the next MI300X/MI308X GPU session. The
current production defaults are intentionally frozen until a controlled A/B
beats them. Do **not** migrate patch stacks, change the pinned runtime, or turn
on graph capture before the baseline gates below pass.

## Stable baseline to preserve

The last validated MI308X profile is:

```text
vLLM                 0.26.1rc1.dev306+gcb8104839.rocm723
AITER                0.1.19
flydsl               0.2.4
max_model_len        524288
GPU KV pool          16 GB pinned
native CPU KV tier   12 GB
block size           256
prefix caching       enabled
max_num_seqs         64
max_batched_tokens   4096
long prefill cap     1024
DSpark                K=7, probabilistic draft, block rejection
cudagraph             disabled
```

Last measured gates: 512-token single-stream decode 133.5 tok/s including
TTFT, C8 446 tok/s aggregate, 50K→500K context ladder all passing, and a short
request injected during a 200K prefill paying only +0.04 s TTFT.

## Phase 0 — restart/runtime integrity

Run before serving traffic:

```bash
python3 scripts/audit_runtime.py
```

The audit must pass. It checks:

- pinned vLLM/AITER/flydsl versions;
- every declared Python overlay against its installed target;
- topk-tiebreak binary and sparse-prefill module;
- JIT/AITER cache and persistent environment snapshots;
- recipe and external patch-repository revisions.

A failure is a recovery problem, not a performance-tuning opportunity. Restore
the declared runtime first.

## Phase 1 — default-profile regression gate

Start the server with **no tuning overrides**. Warm the kernels with one
uncached prefill before recording numbers, then run:

```bash
python3 scripts/bench/bench_full.py all
python3 scripts/bench/bench_agent_trace.py 30 20000
python3 scripts/bench/bench_tool_roundtrip.py --rounds 10 --prefix-tokens 20000
python3 scripts/bench/bench_ttft_isolation.py 200000
python3 scripts/bench/bench_session_concurrency.py --sessions 4 --rounds 8
```

Minimum regression gates:

| Gate | Requirement |
|---|---|
| engine correctness | no crash / no EngineCore death |
| long context | 50K, 128K, 256K, 384K, 500K all complete |
| streaming tool calls | 10/10 complete tool round trips |
| per-request cache accounting | `cached_tokens` reported on warm agent turns |
| warm agent cache | >= 95% session prompt tokens cached after warm-up |
| 512-token single stream | >= 126 tok/s including TTFT (within ~5% of 133.5 baseline) |
| C8 aggregate | >= 424 tok/s (within ~5% of 446 baseline) |
| long-prefill isolation | short-request TTFT overhead <= 0.25 s |

Do not proceed to upstream/runtime migration if these gates fail.

## Phase 2 — scheduler budget Pareto sweep

vLLM's chunked-prefill budget is explicitly a latency/throughput knob. The
community MI300X production profile uses 2,048 tokens for latency isolation,
while larger budgets improve fresh-prefill throughput. Our current local
Pareto point is 4,096, so compare rather than copy.

For each value below, restart, perform one warm-up prefill, and run the same
C1/C4/C8 + cold-prefill + TTFT-isolation subset:

```text
MAX_BATCHED_TOKENS=2048
MAX_BATCHED_TOKENS=3072
MAX_BATCHED_TOKENS=4096   # current default/control
MAX_BATCHED_TOKENS=8192
```

Keep these fixed for the sweep:

```text
LONG_PREFILL_TOKEN_THRESHOLD=1024
MAX_NUM_SEQS=64
DSPARK_ENABLED=1
DSPARK_K=7
KV_OFFLOAD_GB=12
MAX_MODEL_LEN=524288
```

Record at least:

- fresh prefill tok/s;
- warm-prefix TTFT;
- short-request TTFT during a long prefill;
- C1/C4/C8 aggregate and per-stream decode;
- DSpark acceptance metrics;
- HBM, CPU-KV and preemption pressure.

The winner is the agent-session Pareto point, not simply the highest fresh
prefill number.

## Phase 3 — speculative-decoding concurrency sweep

Static K=7 is proven best for the single-stream path on this stack, but
speculative verification can become throughput-negative as batch size grows.
Use the new switch to compare the exact same requests:

```bash
DSPARK_ENABLED=1 DSPARK_K=7 ...   # control
DSPARK_ENABLED=0 ...              # native decode
```

Measure C1/C2/C4/C8. Also capture:

```text
spec_decode_num_accepted_tokens_total
spec_decode_num_draft_tokens_total
spec_decode_num_drafts
per-position acceptance
```

A dynamic K schedule is intentionally **not** a production option yet. Only
try it in a separate experiment after confirming the pinned dev306 DSpark
implementation accepts and correctly applies the relevant configuration.

## Phase 4 — 512K upper-bound overhead check

A 512K `max_model_len` does not allocate 512K KV to every short request, but a
larger configured bound can still affect planner/capture/scheduler structures.
Quantify it rather than assuming zero overhead:

```text
MAX_MODEL_LEN=262144
MAX_MODEL_LEN=393216
MAX_MODEL_LEN=524288   # production requirement
```

Use identical 8K/32K/100K requests and compare TTFT, decode rate, HBM high-water
mark, admitted concurrency and startup time. Keep 524288 unless a measurable
regression is demonstrated; the product requirement remains >=500K context.

## Phase 5 — CPU-KV and prefix-cache correctness

The stable profile uses a 16 GB GPU pool plus a 12 GB native CPU tier because
the host sandbox exposes only 16 GB `/dev/shm`. Validate the native tier under
real agent pressure rather than disabling it pre-emptively:

```text
KV_OFFLOAD_GB=12   # control
KV_OFFLOAD_GB=0    # GPU-only A/B
```

Run both the 500K ladder and concurrent-session benchmark. Compare cache hit
retention, preemptions, TTFT and throughput. Prefix-cache behavior must be
measured with per-request `prompt_tokens_details.cached_tokens`; engine-global
Prometheus counters are only diagnostic under concurrent traffic.

## Phase 6 — upstream-stack experiment (separate environment only)

The current recipe is **not byte-equivalent** to the latest public
`ryanzhou/deepseek-v4-flash-mi300x` stack. The local stable runtime is dev306
plus the historical overlay set; the current upstream production image is
based on another pinned nightly and now carries a smaller/newer overlay set,
including fused fast routing and `BLOCK_H=64` sparse-prefill changes.

Do not overwrite the production venv. Build a second experimental venv and
compare:

1. exact runtime revision;
2. exact overlay inventory and SHA-256 values;
3. model correctness (tool calls + long-context recall) before speed;
4. fresh prefill, C1/C8 decode and agent-session latency.

Only migrate if the full validation suite passes and the improvement survives
multiple warm runs.

## Current external comparison target

The current public MI300X production reference reports roughly:

- 168.6 tok/s median single-stream decode;
- 542 tok/s aggregate at C8;
- 6.99–7.02K tok/s fresh prefill in its 2,048-token shipping profile;
- ~7.9–8.5K tok/s with larger scheduler budgets;
- 256K context validated in that stack.

Those are comparison points, not acceptance requirements for this recipe: our
hardware/runtime constraints differ, and this project separately requires a
verified 500K-class context window.

## Decision rule

Keep the current defaults unless a candidate configuration:

1. passes correctness and 500K gates;
2. does not regress C1/C8 or hot-agent TTFT beyond the regression bands;
3. improves the end-to-end multi-turn agent trace, not just an isolated kernel
   or one-shot throughput number;
4. remains reproducible after a restart/runtime audit.

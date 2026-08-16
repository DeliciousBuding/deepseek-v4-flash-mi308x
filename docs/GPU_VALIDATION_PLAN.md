# GPU validation and tuning plan

This is the ordered runbook for the next MI300X/MI308X GPU session. The
validated defaults stay frozen until a controlled candidate improves the
**end-to-end coding-agent workload** while preserving correctness, 500K-class
context, cache retention, tool protocol, and restart reproducibility.

## Stable control profile

```text
vLLM                 0.26.1rc1.dev306+gcb8104839.rocm723
AITER                0.1.19
flydsl               0.2.4
patch source          ryanzhou commit 012b9945c1e61ec7a7c7de12da58e8c7cafd92ab
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

As of 2026-08-16, the pinned patch-source commit is also the current upstream
`main` commit. The important runtime difference is that upstream production uses
vLLM `dev229`, while this recipe ports the same source overlays onto `dev306`
plus a small compatibility edit and the local sandbox `madvise` patch. Therefore
same patch source != byte-identical serving runtime.

Last local control measurements:

```text
512-token generation incl. TTFT    133.5 tok/s
C1/C2/C4/C8 aggregate              128 / 197 / 286 / 446 tok/s
50K -> 500K ladder                 all pass
repeated-prefix speedup            17.5x
last per-request cached tokens     99.4% (856K / 860K)
hot agent TTFT                     ~0.21-0.32s
streaming tool roundtrip           10/10 before disconnect
200K-prefill short-request cost    +0.04s TTFT
```

## Phase 0 — runtime integrity before serving

After the human switches to the GPU instance and runs the private bootstrap
once, the agent must **not reinstall first**.

```bash
cd /mnt/workspace/vllm-rocm-dsv4-flash
git pull --ff-only
python3 scripts/audit_runtime.py
```

The audit must pass. It verifies the pinned vLLM/AITER/flydsl versions, upstream
patch commit, installed overlay hashes, top-k binary, sparse-prefill module,
JIT source/cache, and persistent venv/cache snapshots.

If this fails, treat it as recovery/provenance work. Do not start tuning on an
unknown mixed runtime.

## Phase 1 — default-profile correctness + regression gate

Start with **no environment overrides**. Warm the runtime once, then run:

```bash
python3 scripts/bench/bench_full.py all
python3 scripts/bench/bench_agent_trace.py 30 20000
python3 scripts/bench/bench_session_concurrency.py --sessions 4 --rounds 8
python3 scripts/bench/bench_ttft_isolation.py 200000

# tool protocol: short prefix / forced tool
python3 scripts/bench/bench_tool_roundtrip.py --rounds 10 --mode forced --prefix-tokens 20000

# tool parser stress: auto selection + long context
python3 scripts/bench/bench_tool_roundtrip.py --rounds 10 --mode auto --prefix-tokens 100000
python3 scripts/bench/bench_tool_roundtrip.py --rounds 8 --mode auto --prefix-tokens 200000

# parser state under concurrent streaming
python3 scripts/bench/bench_tool_roundtrip.py --rounds 16 --mode auto --prefix-tokens 100000 --concurrency 4
```

Minimum promotion gates:

| Gate | Requirement |
|---|---|
| engine | no EngineCore death / restart |
| long context | 50K, 128K, 256K, 384K, 500K complete |
| 512-token single stream | >= 126 tok/s incl. TTFT (~5% band from 133.5) |
| C8 aggregate | >= 424 tok/s (~5% band from 446) |
| warm agent cache | >= 95% by **per-request** cached prompt tokens |
| tool short-prefix | 10/10 valid round trips |
| tool 100K auto | 10/10, no raw DSML in content |
| tool 200K auto | 8/8, no raw DSML in content |
| concurrent tool parser | 16/16, no malformed/leaked markers |
| long-prefill isolation | added short-request TTFT <= 0.25s |

Why the tool gates are strict: current vLLM issue reports include a DeepSeek-V4
long-context case where the model may omit the DSML tool-call START wrapper and
raw invoke text is returned as ordinary assistant content, plus reports of
parser/tag corruption under concurrent load. A coding agent that loses a tool
call is broken even if tok/s improves.

## Phase 2 — reason-history policy A/B

Many coding harnesses do **not** submit hidden reasoning back on the next turn.
That can affect both prompt growth and prefix-cache behavior. Compare:

```bash
python3 scripts/bench/bench_agent_trace.py 30 20000
python3 scripts/bench/bench_agent_trace.py 30 20000 --include-reasoning-history

python3 scripts/bench/bench_session_concurrency.py --sessions 4 --rounds 8
python3 scripts/bench/bench_session_concurrency.py --sessions 4 --rounds 8 --include-reasoning-history
```

The default/content-only trace is the primary coding-agent metric unless the
actual upstream harness explicitly replays reasoning tokens.

## Phase 3 — scheduler-budget Pareto sweep

vLLM's current optimization guide treats `max_num_batched_tokens` as a real
latency/throughput tradeoff: smaller budgets generally protect decode/ITL,
while larger budgets give prefill more work per iteration and can improve TTFT
or throughput. Upstream ryanzhou production itself currently uses a 4,096-token
budget with capacity reserved for speculative work, so our 4,096 default is not
an obvious mismatch.

Compare:

```text
MAX_BATCHED_TOKENS=2048
MAX_BATCHED_TOKENS=3072
MAX_BATCHED_TOKENS=4096   # control
MAX_BATCHED_TOKENS=8192
```

Keep fixed:

```text
MAX_MODEL_LEN=524288
MAX_NUM_SEQS=64
LONG_PREFILL_TOKEN_THRESHOLD=1024
DSPARK_ENABLED=1
DSPARK_K=7
KV_OFFLOAD_GB=12
```

For every restart record fresh-prefill tok/s, hot-prefix TTFT, 200K-prefill
isolation, C1/C4/C8 aggregate, per-stream decode, DSpark acceptance metrics,
preemptions, HBM high-water and CPU-KV pressure.

Do not choose the winner by fresh prefill alone. Choose the agent-session Pareto
point.

## Phase 4 — DSpark vs native decode under concurrency

Static K=7 is the proven local single-stream winner, but speculative verification
can become throughput-negative as runtime batch size rises. A current vLLM issue
shows DSpark losing badly to native decoding on a saturated high-batch workload
even with reasonable acceptance.

Use the same prompts/profile:

```bash
DSPARK_ENABLED=1 DSPARK_K=7 bash scripts/02_serve_vllm.sh dsflash
DSPARK_ENABLED=0            bash scripts/02_serve_vllm.sh dsflash
```

Measure C1/C2/C4/C8 and capture at minimum:

```text
vllm:spec_decode_num_accepted_tokens_total
vllm:spec_decode_num_draft_tokens_total
vllm:spec_decode_num_drafts
per-position acceptance
```

A dynamic batch-size -> K schedule is an **experimental** follow-up only. Current
vLLM supports `num_speculative_tokens_per_batch_size`, but its documentation says
dynamic speculative decoding is tested with Eagle/Eagle3/DFlash; other methods
may not work out of the box. The pinned dev306 DSpark path must be checked before
using dynamic K in production.

## Phase 5 — prefix cache matrix: DSpark x CPU-KV

There are open vLLM bug reports around DeepSeek V4 + DSpark hybrid KV groups
vetoing prefix reuse, and separate external-offload paths reporting zero external
prefix-cache hits. Those reports are not identical to this project's patched
native CPU-KV path, and our previous local cache numbers are strong, but the
interaction must be tested explicitly after restart.

Run the agent trace and 4-session trace for all four combinations:

| DSpark | CPU KV | Purpose |
|---|---|---|
| on | 12 GB | production control |
| off | 12 GB | isolate speculative KV groups |
| on | off | isolate CPU tier |
| off | off | simplest GPU-only/native baseline |

Use per-request `prompt_tokens_details.cached_tokens` as the authoritative
request-level cache measurement. Global Prometheus cache counters are diagnostic
only under concurrent traffic.

Also watch preemptions, HBM high-water, restored/offloaded bytes and any engine
error. Keep the CPU tier only if it improves long-context/multi-session capacity
without correctness or latency regression.

## Phase 6 — 512K configured-ceiling overhead

Paged KV means a short request does not reserve 512K token blocks just because
`max_model_len` is 524,288. But the larger configured bound can still affect
planner/scheduler tables, admission math or startup structures. Measure instead
of assuming zero cost:

```text
MAX_MODEL_LEN=262144
MAX_MODEL_LEN=393216
MAX_MODEL_LEN=524288   # required production ceiling
```

Use identical 8K / 32K / 100K prompts and compare startup time, TTFT, decode,
HBM high-water and admitted concurrency. Keep 524,288 unless it produces a real
short-request regression that is large enough to justify a more complex serving
profile; the product requirement remains approximately 500K context.

## Phase 7 — long-context correctness, not just survival

A request reaching 500K without crashing is necessary but insufficient. Add
needle/retrieval fixtures at ~100K / 256K / 384K / 475K and verify exact recall
of multiple separated values. Run both native and K7 for at least the longest
case.

This follows the upstream production discipline: its current correctness report
promotes long-context recall and tool rounds alongside throughput, not after it.

## Phase 8 — upstream production-runtime experiment

The **patch source commit is currently the same upstream `main` commit** we pin.
The major stack difference is runtime base:

```text
upstream production: vLLM dev229 + AITER 0.1.19
local stable port:   vLLM dev306 + AITER 0.1.19
                     + mxfp4 activation signature compatibility edit
                     + local sandbox madvise patch
```

If the local control is healthy, build a **second venv** reproducing upstream's
actual dev229 runtime as closely as a Docker-less host permits. Do not overwrite
`/root/.venvs/vllm`.

Compare correctness first, then fresh prefill, C1/C8 decode, 500K viability,
agent-session TTFT/cache and tool protocol. Promote only if the alternate runtime
wins the whole matrix and survives restart/audit.

## Current upstream comparison (2026-08-16)

Current ryanzhou `main` / commit
`012b9945c1e61ec7a7c7de12da58e8c7cafd92ab` reports:

```text
vLLM production base       dev229
AITER                       0.1.19
uncached C1 prefill         11.69K tok/s steady (11.53K median)
static DSpark-7 C1          152.6 aggregate / 158.8 median per-stream
native C1                   67.3 aggregate
C64 K7 burst                1,278 aggregate
context                     384K validated (architecture supports 1M)
GPU KV                      16 GB fp8_ds_mla
CPU KV                      96 GiB native tier
scheduler                   4096 budget; 384 reserved for spec, up to 3712 ordinary prefill
```

These are reference points, not acceptance requirements. Our host has a much
smaller CPU-KV tier and this project requires ~500K context.

## Decision rule

Keep the current defaults unless a candidate:

1. passes runtime provenance and all long-context/tool correctness gates;
2. preserves the ~500K requirement;
3. stays within regression bands for C1/C8 and hot-agent TTFT;
4. improves the complete multi-turn agent trace or a clearly defined production
   objective rather than one isolated microbenchmark;
5. remains reproducible after restart and `audit_runtime.py`.

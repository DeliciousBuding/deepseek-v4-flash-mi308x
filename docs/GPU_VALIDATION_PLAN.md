# GPU validation and tuning plan

This is the ordered runbook for future MI300X/MI308X GPU experiments. The
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
max_batched_tokens   3072
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
512-token generation incl. TTFT    141.8 tok/s
C1/C2/C4/C8 aggregate              129.2 / 235.8 / 375.0 / 549.6 tok/s
50K -> 500K ladder                 all pass (500K: 77.3s at 3072; 75.3s at 4096)
30-turn per-request cached tokens  95.46% (1,027,596 / 1,076,518)
hot agent TTFT                     ~0.23-0.38s ordinary hot turns
auto tool roundtrip                100K 5/5; 200K 3/3; 100K C4 16/16
true-cold 200K isolation           KNOWN DEGRADED: +1.30 / +1.33 / +1.41s at 3072
high-batch K7 C32/C64              730.2 / 914.6 tok/s; native reaches parity only at C64
```

## Phase 0 — runtime integrity before serving

After the operator switches to the GPU instance and completes that deployment's
bootstrap, do **not reinstall first**.

```bash
cd "${RECIPE_ROOT:?set RECIPE_ROOT to the deepseek-v4-flash-mi308x checkout}"
git pull --ff-only
python3 scripts/audit_runtime.py
```

The audit must have **zero correctness/provenance failures**. It verifies the
pinned vLLM/AITER/flydsl versions, upstream patch commit, installed overlay
hashes, top-k binary, sparse-prefill module, JIT source, and the persistent venv
snapshot. Runtime-generated AITER/torch/ROCm-COMGR/Triton caches are warm-start accelerators: a
fresh host may warn when they are absent, but that does not block the first
controlled warm-up.

If the audit reports a correctness/provenance failure, treat it as recovery
work. Do not start tuning on an unknown mixed runtime.

## Phase 1 — default-profile correctness + regression gate

Start with **no environment overrides**. Warm the runtime once, then run:

```bash
python3 scripts/bench/bench_full.py all
python3 scripts/bench/bench_agent_trace.py 30 20000
python3 scripts/bench/bench_session_concurrency.py --sessions 4 --rounds 8
python3 scripts/bench/bench_ttft_isolation.py 200000 --rounds 3  # nonce-forced true-cold samples

# tool protocol: short prefix / forced tool
python3 scripts/bench/bench_tool_roundtrip.py --rounds 10 --mode forced --prefix-tokens 20000

# tool parser stress: auto selection + long context
python3 scripts/bench/bench_tool_roundtrip.py --rounds 10 --mode auto --prefix-tokens 100000
python3 scripts/bench/bench_tool_roundtrip.py --rounds 8 --mode auto --prefix-tokens 200000

# parser state under concurrent streaming
python3 scripts/bench/bench_tool_roundtrip.py --rounds 16 --mode auto --prefix-tokens 100000 --concurrency 4

# Cover first-use Triton/DSpark paths, then atomically persist the validated
# production venv plus AITER/torch/COMGR/Triton caches.
SNAPSHOT_AFTER_WARMUP=1 bash scripts/warmup_runtime.sh
python3 scripts/audit_runtime.py
```

Minimum promotion gates:

| Gate | Requirement |
| --- | --- |
| engine | no EngineCore death / restart |
| long context | 50K, 128K, 256K, 384K, 500K complete |
| 512-token single stream | >= 134 tok/s incl. TTFT (~5% band from 141.8) |
| C8 aggregate | >= 522 tok/s (~5% band from 549.6) |
| warm agent cache | >= 95% by **per-request** cached prompt tokens |
| tool short-prefix | 10/10 valid round trips |
| tool 100K auto | 10/10, no raw DSML in content |
| tool 200K auto | 8/8, no raw DSML in content |
| concurrent tool parser | 16/16, no malformed/leaked markers |
| long-prefill isolation | **open issue** until nonce-forced cold added TTFT <= 0.5s |

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

The 2026-08-16 sweep already promoted **3,072** as the coding-agent default.
4,096 remains the measured throughput profile; 2,048 was rejected. Re-run this
phase only after a material runtime/kernel change, using the same multi-round
true-cold isolation method.

```text
MAX_BATCHED_TOKENS=3072   # current default, ordinary DSpark budget 2688
MAX_BATCHED_TOKENS=4096   # throughput profile, ordinary DSpark budget 3712
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

## Phase 4 — DSpark vs native decode under concurrency (completed)

The 2026-08-16 clean-restart comparison found that K7 remains the clear winner
through C32. Native reaches aggregate parity only at full C64 admission:

```text
                         warm decode-512    C32 aggregate    C64 aggregate
DSpark K7                        141.7           730.2            914.6 tok/s
native                            33.4           570.4            921.6 tok/s
```

K7 retains the lower C64 median latency and remains the production default.
Re-run this phase only after a material runtime or DSpark implementation change.
Use the same prompts/profile with separate clean restarts and warm-ups:

```bash
DSPARK_ENABLED=1 DSPARK_K=7 bash scripts/02_serve_vllm.sh dsflash
DSPARK_ENABLED=0            bash scripts/02_serve_vllm.sh dsflash

python3 scripts/bench/bench_high_concurrency.py --concurrencies 32 64
```

Capture at minimum:

```text
vllm:spec_decode_num_accepted_tokens_total
vllm:spec_decode_num_draft_tokens_total
vllm:spec_decode_num_drafts
per-position acceptance
```

A dynamic batch-size -> K schedule remains an **experimental** follow-up only.
Current vLLM supports `num_speculative_tokens_per_batch_size`, but its
documentation says dynamic speculative decoding is tested with
Eagle/Eagle3/DFlash; other methods may not work out of the box. The pinned
dev306 DSpark path must be checked before using dynamic K in production.

## Phase 5 — prefix cache matrix: DSpark x CPU-KV (completed)

There are open vLLM bug reports around DeepSeek V4 + DSpark hybrid KV groups
vetoing prefix reuse, and separate external-offload paths reporting zero external
prefix-cache hits. Those reports are not identical to this project's patched
native CPU-KV path, and our previous local cache numbers are strong, but the
interaction must be tested explicitly after restart.

The 2026-08-16 clean-restart control ran the 4-session trace for all four
combinations, plus fixed decode and representative 30-turn traces:

| DSpark | CPU KV | Purpose |
| --- | --- | --- |
| on | 12 GB | production control |
| off | 12 GB | isolate speculative KV groups |
| on | off | isolate CPU tier |
| off | off | simplest GPU-only/native baseline |

```text
                         decode-512    4x8 cache    hot TTFT p95    stream decode median
K7 + CPU KV 12 GB             141.9       89.40%          2.904s                  79.3 tok/s
K7 + GPU-only                 142.0       89.45%          4.463s                  80.2 tok/s
native + CPU KV 12 GB          33.5       89.56%          2.863s                  30.5 tok/s
native + GPU-only              33.4       89.63%          2.814s                  29.5 tok/s
```

All four 4x8 cells completed 32/32 requests with zero preemptions. Natural EOS
made aggregate completion counts differ, so the decision uses per-stream decode,
TTFT and cache rather than incomparable aggregate throughput. The CPU tier had
no fixed-decode tax, while cumulative counters at the end of K7 control sampling
reported 13.06 GB moved to CPU and 74.2 MB restored. GPU-only K7 passed 500K but
did not improve decode and had the worse 4-session tail. Keep K7 + 12 GB CPU-KV.

Use per-request `prompt_tokens_details.cached_tokens` as the authoritative
request-level cache measurement. Global Prometheus cache counters are diagnostic
only under concurrent traffic.

Re-run this phase only after a material scheduler, KV-offload or DSpark change.
Continue to watch preemptions, HBM high-water, restored/offloaded bytes and any
engine error.

## Phase 6 — 512K configured-ceiling overhead (completed)

Paged KV means a short request does not reserve 512K token blocks just because
`max_model_len` is 524,288. But the larger configured bound can still affect
planner/scheduler tables, admission math or startup structures. Measure instead
of assuming zero cost:

```text
MAX_MODEL_LEN=262144
MAX_MODEL_LEN=393216
MAX_MODEL_LEN=524288   # required production ceiling
```

After each clean restart, run the cache-isolated fixture with the matching
profile label:

```bash
python3 scripts/bench/bench_configured_ceiling.py \
  --profile-label "$MAX_MODEL_LEN" \
  --prompt-tokens 8000 32000 100000 \
  --samples 2
```

Use identical 8K / 32K / 100K prompts and compare startup time, TTFT, decode,
HBM high-water and admitted concurrency. Keep 524,288 unless it produces a real
short-request regression that is large enough to justify a more complex serving
profile; the product requirement remains approximately 500K context.

The clean-restart result retained 524,288:

```text
MAX_MODEL_LEN        actual 7,879    actual 31,399    actual 98,039    VRAM used
524288                    2.096s            8.852s           30.316s     ~194.26 GB
393216                    2.096s            8.849s           30.328s     ~193.19 GB
262144                    2.096s            8.860s           30.354s     ~192.75 GB
```

All formal requests had zero cached tokens and zero preemptions. Smaller ceilings
saved only about 1.1-1.5 GB and did not improve TTFT, so losing 500K-class
admission has no compensating benefit. Re-run only after a material planner,
scheduler or KV-allocation change.

## Phase 7 — long-context correctness, not just survival

A request reaching 500K without crashing is necessary but insufficient. The
cache-isolated five-needle fixture produced this result:

```text
decoder    prompt tokens    result
K7               100,069    PASS (5/5 exact)
K7               256,057    PASS (5/5 exact)
K7               384,053    FAIL twice (needle 4 mismatch)
K7               475,068    PASS (5/5 exact)
native           384,053    PASS (5/5 exact)
```

The repeated K7-only 384K failure is an open DSpark correctness risk. It is not
a simple context cutoff because the sampled 475K case passed, but these results
must not be presented as universal long-context recall.

```bash
python3 scripts/bench/bench_long_context_recall.py \
  --target-tokens 100000 256000 384000 475000 \
  --needles 5
```

The harness uses `/tokenize` to calibrate document size, places random exact
values throughout the document, forces a cold cache with a first-block nonce and
unique cache salt, and accepts only the ordered JSON array in final content.

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
`$VLLM_VENV`.

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

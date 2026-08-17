# GPU tuning log — 2026-08-16 MI308X session

This log records the actual tuning path used to move the local
DeepSeek-V4-Flash-0731 service from the inherited MI300X-oriented configuration
to the current MI308X production profile. It intentionally includes failed
experiments and benchmark corrections so later work does not repeat them.

The short version is in [`PERFORMANCE.md`](PERFORMANCE.md). This file is the
experiment notebook and decision trail.

## 1. Starting point and invariants

The GPU session started only after the CPU preflight passed:

- model weights: 48/48 shards, about 156 GiB on persistent storage;
- runtime: vLLM `0.26.1rc1.dev306+gcb8104839.rocm723`;
- AITER `0.1.19`, flydsl `0.2.4`;
- upstream patch source pinned to
  `ryanzhou/deepseek-v4-flash-mi300x@012b9945c1e61ec7a7c7de12da58e8c7cafd92ab`;
- configured context ceiling: 524,288 tokens;
- GPU KV: 16 GB pinned, native CPU-KV tier: 12 GB;
- DSpark: K=7, probabilistic drafting, block rejection;
- scheduler budget: 4,096 batched tokens, long-prefill threshold 1,024;
- max sequences: 64.

The GPU reports `gfx942`, but **80 compute units**. `torch`, `rocminfo` and AITER
agreed on the 80-CU value. This detail became the main tuning finding.

## 2. Runtime recovery and cold-start observations

The recovered venv passed the runtime provenance audit. The first GPU start was
expensive because runtime-generated compiler caches were absent.

Observed first-start components:

| Component | First observed time |
| --- | ---: |
| target model weight load | ~104.3s |
| DSpark draft weight load | ~61.8s |
| total model load | ~169.7s |
| graph capture | ~104s |
| engine profile + KV + warm-up | ~283.5s |

The first run also generated substantial caches outside `$HOME/.aiter`:

- `$HOME/.cache/torch_extensions`: custom HIP/C++ extension cache;
- `$HOME/.cache/comgr`: ROCm compiler/code-object cache, about 612 MB in this run.

The torch-extension cache was about 17 MB. The repository therefore persists
both caches in addition to the optional legacy AITER cache. After warm-up, graph
capture fell to roughly the 25-second range and engine profile/KV/warm-up was
observed around 55.9s.

## 3. Baseline before MI308X-specific tuning

Before changing AITER tables, the service was healthy and passed basic API and
long-context checks. Representative same-session measurements were:

| Metric | Untuned control |
| --- | ---: |
| decode-512 | ~128.9-139.3 tok/s depending on fixture/warm state |
| C1 aggregate | 117.7 tok/s |
| C2 aggregate | 235.5 tok/s |
| C4 aggregate | 375.0 tok/s |
| C8 aggregate | 534.4 tok/s |

The important point is not the small fixture variance. Logs showed AITER saying
that shapes already present in the inherited tuning CSV were still "not found".

## 4. Root cause: gfx942 is not one tuning key

Inspection of the inherited ryanzhou CSVs showed `cu_num=304`. AITER 0.1.19
looks up A8W8 configs using a key that includes:

```text
(gfx, cu_num, M, N, K)
```

This host is `gfx942,80`, so the MI300X `gfx942,304` rows do not match. The
service had the patch stack, but much of its A8W8 GEMM path was falling back to
default kernels.

This is why the production repository does **not** rewrite `304` to `80`. The
80-CU rows were tuned on the actual MI308X and numerically checked.

## 5. Pilot tuning: prove the opportunity before expanding

Two real runtime shapes were used as pilots.

### 5.1 Target verify / bpreshuffle pilot

Shape:

```text
M=8, N=32768, K=1024
```

Production-operator result:

```text
default: 37.02 us
tuned:   14.42 us
speedup: 2.57x
latency reduction: ~61.0%
numerical check: PASS, err_ratio=0.0
```

### 5.2 DSpark drafter / standard A8W8 pilot

Shape:

```text
M=7, N=32768, K=1024
```

Production-operator result:

```text
default: 31.97 us
tuned:   25.31 us
speedup: 1.26x
latency reduction: ~20.8%
numerical check: PASS
```

Both target and drafter paths improved, so the tuning work expanded to actual
runtime shapes rather than synthetic arbitrary sizes.

## 6. C1 shape tuning

Runtime logs were used to extract the low-concurrency C1 shapes. The first batch
focused on `M=7/8`.

### 6.1 bpreshuffle examples

| Shape `(M,N,K)` | Default | Tuned | Result |
| --- | ---: | ---: | ---: |
| `(8,4096,4096)` | 31.11 us | 12.13 us | ~61.0% lower latency |
| `(8,4096,8192)` | 58.31 us | 21.59 us | ~63.0% lower latency |
| `(8,32768,1024)` | 36.06 us | 14.87 us | ~58.8% lower latency |
| `(8,1536,4096)` | faster on default | slower candidate | **rejected** |

The process deliberately rejected the last row rather than assuming every tuner
winner should be promoted.

### 6.2 standard A8W8

Nine C1 standard/drafter shapes were evaluated. Eight showed roughly 16-21%
production-operator improvement with numerical PASS. A candidate involving
`M=7,N=8192,K=1024` did not meet the production-registry verification path at
that stage and was not promoted from that batch.

### 6.3 C1-only service A/B

After installing only the C1 winners, end-to-end service results moved as
expected: low concurrency improved while higher concurrency stayed effectively
flat.

| Metric | Before | C1-tuned |
| --- | ---: | ---: |
| decode-128 | 113.9 | 118.8 tok/s |
| decode-512 | 128.9 | 140.7 tok/s |
| C1 | 117.7 | 128.3 tok/s |
| C2 | 235.5 | 234.9 tok/s |
| C4 | 375.0 | 373.5 tok/s |
| C8 | 534.4 | 534.1 tok/s |

A fixed decode-512 fixture was then repeated three times at **141.4 / 141.4 /
141.3 tok/s**, with identical DSpark acceptance. That established that the
single-stream gain was not a one-shot wall-clock fluctuation.

## 7. AITER tuner verification trap

When tuning larger C8/prefill shapes, the AITER compare harness produced an
important false-negative pattern:

1. the tuner found and compiled a candidate kernel;
2. generated lookup/header artifacts contained the candidate;
3. the same Python process then reported that the kernel was not present in the
   production registry.

The rebuilt `.so` was correct. The problem was same-process native-extension
reload behavior: the Python process could retain the previously loaded extension
object/registry even after the file on disk had been rebuilt.

The mitigation used for promotion was:

- keep the candidate CSV;
- start a **fresh Python process**;
- load the newly built production operator and candidate config there;
- require numerical PASS and real production-operator timing improvement.

Previously reported "kernel not present" large-M candidates then validated
successfully in the fresh process. This is why the checked-in tuning tables are
not based only on the tuner's same-process summary.

## 8. C8 and long-prefill tuning

Runtime shapes `M=56/64` were selected for C8 throughput and `M=4096` for the
large prefill path.

Representative bpreshuffle fresh-process results:

| Shape | Default | Tuned | Approx. improvement |
| --- | ---: | ---: | ---: |
| `M56,4096x4096` | 34.45 us | 22.02 us | ~36% |
| `M56,4096x8192` | 64.34 us | 39.54 us | ~39% |
| `M64,4096x8192` | 43.38 us | 38.39 us | ~11.5% |
| `M4096,4096x8192` | 1275 us | 1025 us | ~19.6% |
| `M4096,32768x1024` | 1556 us | 1317 us | ~15.4% |

Representative standard A8W8 results were larger:

| Shape | Default | Tuned | Approx. improvement |
| --- | ---: | ---: | ---: |
| `M4096,1536x4096` | 941 us | 222 us | ~76% |
| `M4096,4096x2048` | 1308 us | 302 us | ~77% |
| `M4096,4096x12288` | 7128 us | 1569 us | ~78% |
| `M4096,8192x1024` | 1443 us | 368 us | ~74% |

For C8, the main standard GEMMs at `M=56` improved about 42-54%; `M=64`
commonly improved about 45-49% in the operator benchmark. A tiny
`M=8,8192x1024` candidate measured 9.09 -> 9.17 us and was rejected.

## 9. Final promoted 80-CU tables

Only rows with a numerical PASS and useful production-operator result were kept:

```text
tuning/dsv4-mi308x-80cu-a8w8-blockscale-bpreshuffle.csv  13 rows
tuning/dsv4-mi308x-80cu-a8w8-blockscale.csv              24 rows
```

Promoted M coverage:

- bpreshuffle: `M=8,56,64,4096`;
- standard: `M=7,8,56,64,4096`.

The launcher detects `gfx942/80-CU` and selects these tables automatically.
Explicit AITER config environment variables remain an override.

## 10. End-to-end result with the promoted tables

The full service profile produced:

| Metric | Result |
| --- | ---: |
| decode-128 | 119.1 tok/s |
| decode-512 | 140.8 tok/s |
| repeated fixed decode-512 | 141.4 / 141.4 / 141.3 tok/s |
| C1 | 128.9 tok/s |
| C2 | 235.8 tok/s |
| C4 | 375.3 tok/s |
| C8 | **548.3 tok/s** |

The C8 improvement from roughly 534.4 to 548.3 tok/s is ~2.6%. The operator
benchmarks can improve far more than the whole service because attention, MoE,
scheduling, Python/runtime overhead and speculative-decoding behavior remain in
the end-to-end path.

## 11. Long-context ladder

The promoted profile passed the complete context ladder:

| Target | Actual prompt tokens | Wall time | Result |
| ---: | ---: | ---: | --- |
| 50K | 47,505 | 16.7s | PASS |
| 128K | 121,605 | 25.3s | PASS |
| 256K | 243,205 | 52.3s | PASS |
| 384K | 364,805 | 66.2s | PASS |
| 500K | 475,005 | **75.3s** | PASS |

The ladder shares prefixes, so these wall times are not equivalent to independent
cold-prefill measurements. Engine logging separately showed prompt-throughput
samples around 12.18K tok/s in the long-prompt path.

## 12. Coding-agent and tool correctness gates

### 12.1 30-turn agent trace

Pre-promotion 4,096 throughput-profile result, retained for comparison:

```text
session total: 26.8s
avg hot TTFT (including deliberate new-observation turns): 0.442s
ordinary hot TTFT: roughly 0.20-0.36s
avg decode: 167.3 tok/s
per-request prefix-cache hit: 95.46%
  1,030,144 / 1,079,154 prompt tokens
```

Every fourth-style environment/tool growth event intentionally adds a large new
suffix, briefly reducing the request's cache percentage into the low/mid 90s;
subsequent turns return to roughly 99%.

The promoted 3,072 production-profile rerun supersedes those headline values:

```text
session total: 24.8s
avg decode: 167.0 tok/s
per-request prefix-cache hit: 95.46%
  1,027,596 / 1,076,518 prompt tokens
```

### 12.2 auto tool calls

100K auto mode:

```text
5/5 PASS
first cold tool TTFT ~33.0s
hot tool TTFT ~0.53s
post-tool TTFT ~0.49-0.51s
no raw DSML leakage observed
```

200K auto mode:

```text
3/3 PASS
first partially cached tool TTFT ~45.1s
hot tool TTFT ~0.83-0.88s
post-tool TTFT ~0.87-0.89s
no raw DSML leakage observed
```

The protocol gates were kept separate from synthetic performance traces so a
benchmark never fabricates an unmatched `role=tool` message.

## 13. Model staging and restart cost

Persistent model weights live on NFS. A complete ephemeral copy was staged to:

```text
$HOME/models/deepseek-ai/DeepSeek-V4-Flash-0731
```

The staging script copies to a temporary directory, compares a complete filename + size inventory, checks all 48 shards and key metadata, then renames atomically.
The launcher only auto-selects the hot copy when it is complete.

Observed model-load progression:

| State | Target | Draft | Total model load |
| --- | ---: | ---: | ---: |
| original NFS | ~104.3s | ~61.8s | ~169.7s |
| first local staged run | 97.5s | 20.1s | 121.1s |
| warmer local/page-cache run | 84.8s | 3.69s | 92.0s |
| hottest same-instance run | 35.86s | 3.62s | **43.0s** |

The 43s result is a same-instance warm/page-cache result, not a promise that a
fresh GPU VM will always load the model in 43 seconds. The persistent source
remains NFS; the local copy is deliberately disposable.

## 14. TTFT isolation benchmark correction

The original isolation fixture had a measurement flaw: it reused a deterministic
200K prefix. An immediate rerun could therefore turn the long request from about
45 seconds into **~0.6s** through automatic prefix caching, making the short
request appear perfectly isolated.

The benchmark now places a random nonce in the **first block** by default.
`--reuse-prefix` exists only to reproduce/cache-diagnose the old behavior.

With the production scheduler and true-cold 200K prefix:

```text
short alone:            ~0.05s TTFT
long prefill total:     ~44.6s
short during prefill:   ~2.12s TTFT
added short latency:    ~+2.07s
verdict:                DEGRADED
```

This is an open production-quality issue, not hidden by the tuning headline.

## 15. Rejected scheduler / 1K-shape experiments

### 15.1 always cap solo long prefill at 1,024

A local experimental scheduler switch kept the 1,024-token long-prefill cap
active even when the long request was the only request in the engine. The
original measurement reported `~+3.08s`, but that was later shown to be the
**first-round JIT/capture tax**, not steady-state behavior (see the 2026-08-16
re-test below).

```text
200K long total:        ~62.3s
short alone:            ~0.08s
short during prefill:   ~3.16s
added short latency:    ~+3.08s   (first round, JIT-contaminated)
```

**2026-08-16 re-test, warm/JIT-stable (4 rounds, round 1 discarded):**

```text
round 02/03/04 added:   +0.98 / +0.99 / +0.90s   (median +0.98s)
200K long total:        ~59.5s
```

So the *steady-state* always-1,024 cap is not a regression: it improves the
late-short isolation from ~+1.30s to ~+0.98s at the cost of ~+13.5s on the 200K
prefill (59.5s vs 46.1s, ~29% prefill-time increase / ~22% throughput loss). The
original `+3.08s` verdict was a cold-start artifact. The cap was still removed
because the isolation gain (+0.32s) does not justify that prefill cost, and it
still misses the +0.5s gate. The production scheduler source is unchanged.

### 15.2 extra M~1024/1031/1032 tuning

The experimental cap exposed many untuned shapes near M=1024. Tuning them showed
large operator wins; for the standard A8W8 path, 12/12 tested shapes passed and
many dropped roughly 66-72% in operator latency. Fresh-process verification also
rescued bpreshuffle candidates affected by the native-extension reload issue.

However, with the **normal production scheduler**, adding these extra rows did
not solve the actual cold-isolation gate. A measured run was:

```text
long prefill total:     ~47.3s
short alone:            ~0.05s
short during prefill:   ~2.99s
added short latency:    ~+2.94s
```

The extra 1K rows are therefore **not in the production CSVs**. Operator speedup
alone is insufficient reason to enlarge the production configuration when the
end-to-end objective does not improve.

## 16. Scheduler-budget sweep and the 3,072 promotion

The next controlled variable was `max_num_batched_tokens`. DSpark K=7 reserves
384 token slots, so the effective ordinary scheduled-token budgets were 3,712,
2,688 and 1,664 for the 4,096 / 3,072 / 2,048 launcher settings respectively.
All other production variables stayed fixed.

### 16.1 4,096 control

The clean repository-default service (before promotion) reproduced the tuned
throughput baseline:

```text
decode-512              141.8 tok/s
C1/C2/C4/C8             129.3 / 235.8 / 375.1 / 549.6 tok/s
```

Three true-cold 200K isolation samples in the same session were:

```text
sample 1: long 47.4s, added short TTFT +2.95s  (outlier)
sample 2: long 44.6s, added short TTFT +2.06s
sample 3: long 44.5s, added short TTFT +2.08s
```

The two stable samples confirm the earlier ~+2.1s finding; the outlier also shows
why a single injection timing is not a sufficient scheduler gate.

### 16.2 3,072 candidate

The first decode measurement after the 3,072 restart was **not** a valid steady-
state result: decode-128 / decode-512 fell to 7.3 / 40.4 tok/s while the EngineCore
log explicitly reported inference-time Triton JIT for sparse/indexing/DSpark
kernels. The immediate repeat was 124.9 / **141.8 tok/s**, and C8 was **549.6
tok/s**. This directly motivated the post-start warm-up and Triton cache snapshot
added later in the session.

Three independent true-cold 200K isolation samples were tightly grouped:

```text
long 46.1s, added short TTFT +1.33s
long 46.1s, added short TTFT +1.41s
long 46.1s, added short TTFT +1.30s
```

The full context ladder at 3,072 was:

| Target | Prompt tokens | Wall time | Result |
| ---: | ---: | ---: | --- |
| 50K | 47,505 | 15.0s | PASS |
| 128K | 121,605 | 26.4s | PASS |
| 256K | 243,205 | 54.5s | PASS |
| 384K | 364,805 | 71.6s | PASS |
| 500K | 475,005 | **77.3s** | PASS |

Compared with the prior 75.3s 4,096 endpoint, the measured 500K cost is ~2.7%,
while the stable late-short-request penalty falls by roughly one third. The
interactive coding-agent profile therefore prefers 3,072; 4,096 remains useful
as an explicit throughput profile.

The 3,072 candidate then passed the higher-level promotion gates:

```text
30-turn cache          95.46% = 1,027,596 / 1,076,518 prompt tokens
100K auto tool         5/5 PASS; hot median tool TTFT 0.527s
200K auto tool         3/3 PASS; hot median tool TTFT 0.863s
100K auto, C=4         16/16 PASS; tool TTFT median 0.796s
```

No raw DSML/tool-parser leakage was observed.

### 16.3 2,048 candidate

Steady decode/C8 again stayed flat (141.8 / 549.7 tok/s), but the cold long request
stretched to 54.1s and 51.4s in two samples. Added short-request latency was
+2.98s and +1.16s: both slower overall and much less stable than 3,072. It was
rejected without spending more GPU time trying to make the matrix look prettier.

### 16.4 Why no prompt-specific 2,688 GEMM table was added

The 3,072 setting yields an ordinary scheduler budget of 2,688, but runtime AITER
miss logs did **not** show one stable `M=2688` GEMM. The larger misses were routed
MoE-dependent M values such as 1,131 / 1,216 / 1,669 / 1,809. Tuning those exact
sizes from one synthetic prompt would overfit the benchmark rather than the
production workload, so no such rows were promoted.

## 17. First-request JIT and restart-state cleanup

The scheduler sweep exposed compiler state that vLLM's built-in warm-up did not
cover. After a representative first request, the runtime had:

```text
$HOME/.triton                 ~185 MB
$HOME/.cache/comgr            ~612 MB
$HOME/.cache/torch_extensions ~17 MB
$HOME/.aiter                  tiny compatibility cache
```

The recovery design was expanded accordingly:

- `warmup_runtime.sh` exercises representative DSpark decode plus a short forced
  tool/prefill path after `/health` is ready;
- `snapshot_runtime_caches.sh` now persists Triton + COMGR + torch extensions +
  optional AITER cache;
- `snapshot_runtime_state.sh` atomically snapshots the **current production
  venv** and then all runtime caches, so a tuned AITER registry is not replaced
  on reboot by an older experimental snapshot.

This is operational tuning, not just startup cosmetics: the first 3,072 decode
measurement demonstrated that missing first-use JIT can turn a normal 141 tok/s
request into a temporary 40 tok/s request.

## 18. What remains worth tuning

The next useful work is above the already-promoted GEMM layer:

1. investigate why a late short request can still wait ~1.3s behind an already
   dispatched 3,072-profile cold-prefill iteration;
2. profile attention/MoE/scheduler time after the large GEMMs became much faster;
3. avoid exact-M overfitting to prompt-dependent MoE routed sizes unless a broad
   real workload proves that they recur;
4. keep tool/parser, 500K, cache and restart gates mandatory for every candidate.

### 18.1 Where the ~1.3s actually goes (scheduler trace, 2026-08-16)

A per-step scheduler trace (temporary `SCHEDTRACE` instrumentation, reproduced
added short TTFT +1.31s) resolves the open question in item 1 with direct
evidence. The steady-state (post-warm-up) timeline:

- a solo long-prefill chunk is **2,688 tokens** (3,072 minus DSpark K7's 384
  reservation) and executes in **~0.73s** per step;
- the moment the short request is admitted, the long prefill drops to 1,024-token
  chunks and the short request is scheduled in the **same step** as the long
  chunk (so admission/interleaving works as intended);
- the short request's penalty is dominated by **queue wait for the in-flight
  2,688-token chunk to finish** (~0.7s), plus its own batched prefill/decode
  (~0.4s), plus a small contention residue.

Correction to the earlier Prometheus reading: `request_queue_time_seconds` /
`request_inference_time_seconds` are only emitted for non-streaming finished
requests, so they silently exclude this benchmark's streaming requests and the
previous "~7ms queue time" conclusion was a measurement artifact. The
`iteration_tokens_total` histogram counts per-step computed tokens and does not
contradict the 2,688-token chunk assertion in the scheduler.

Note on restart contamination: the *first* 2,688-token prefill chunk after a
vLLM restart can take ~3.0s (inference-time JIT/capture), so an isolation run
immediately after restart reports a much larger penalty than steady state. The
warm-up (`warmup_runtime.sh`) mitigates this, but a restart re-introduces a
first-prefill tax.

Conclusion: the fix belongs in the **in-flight chunk granularity** — a late
short request waits ~0.7s for a 2,688-token chunk it cannot interrupt. Reducing
the solo chunk size shortens that wait but raises total prefill time (the
always-1,024-cap experiment in §15.1 regressed to +3.08s). The next levers are
adaptive chunk sizing (small chunks early / near the interactive window) or
preempting the in-flight prefill chunk; more GEMM rows are irrelevant here.

## 19. Promotion policy used in this session

A tuning row or runtime change was promoted only if the relevant checks passed:

- exact hardware key (`gfx942`, `cu_num=80`);
- numerical correctness through the production operator;
- fresh-process verification when AITER's same-process extension reload was
  ambiguous;
- end-to-end service improvement for the intended workload;
- no regression in 500K context, DSpark, prefix caching or tool protocol.

This policy is intentionally conservative. The repository keeps a smaller table
that is explainable and measured instead of a large collection of tuner output.

## 20. DSpark K7 high-batch boundary

The production DSpark K7 profile was compared with the native decoder while
keeping the model, 3,072-token scheduler profile, GPU/CPU KV configuration,
prompt family and 256-token concurrent outputs fixed. Each profile received a
clean service restart and representative warm-up. Measurements that could have
overlapped an interrupted client sweep were discarded; the C32/C64 values below
are clean, standalone reruns from an idle engine. The decode-512 row uses the
existing warm-up fixture, not the indexed high-concurrency prompt.

The reproducible entry point is:

```bash
python3 scripts/bench/bench_high_concurrency.py \
  --concurrencies 32 64 --max-tokens 256
```

Run the command separately against each serving profile. Do not launch native
and speculative sweeps against the same live engine.

| Decoder | warm-up decode-512 | C32 aggregate | C64 aggregate |
| --- | ---: | ---: | ---: |
| DSpark K7 | **141.7** | **730.2** | 914.6 |
| native | 33.4 | 570.4 | **921.6** |

All throughput values are aggregate completion tok/s and all tested requests
completed successfully. The clean high-batch latency details were:

```text
DSpark K7 C32: p50 10.402s, p95 11.095s, acceptance 47.1%
native    C32: p50 14.345s, p95 14.354s

DSpark K7 C64: p50 16.947s, p95 17.799s, acceptance 47.1%
native    C64: p50 17.709s, p95 17.754s
```

Native decode only catches the speculative path at the fully occupied C64
boundary: its measured aggregate advantage there is about 0.8%, small enough to
be treated as parity rather than a new default. K7 remains about 4.2x faster for
decode-512, about 28% faster at C32, and retains the lower C64 median request
latency. The service therefore keeps DSpark K7 as the production default; native
decode remains a controlled extreme-throughput comparison only.

## 21. DSpark x CPU-KV four-cell control

The next clean-restart matrix held the model, 524,288-token ceiling, 3,072-token
scheduler profile and agent fixtures fixed while changing only the decoder and
native CPU-KV tier:

| Decoder | CPU KV | decode-512 | 4x8 cache | hot TTFT p95 | stream decode median |
| --- | --- | ---: | ---: | ---: | ---: |
| DSpark K7 | 12 GB | **141.9** | 89.40% | **2.904s** | 79.3 |
| DSpark K7 | off | 142.0 | 89.45% | 4.463s | **80.2** |
| native | 12 GB | 33.5 | 89.56% | 2.863s | **30.5** |
| native | off | 33.4 | **89.63%** | **2.814s** | 29.5 |

Throughput columns are tok/s. Every 4x8 cell completed 32/32 requests with zero
preemptions. Aggregate session throughput is intentionally omitted because the
fixture allows natural EOS and completion lengths differed between runs.

The K7 + CPU-KV control's 30-turn trace reproduced the promoted 95.46% cache hit,
0.388s average hot TTFT and 167.7 tok/s average decode. Native + CPU-KV retained
95.93% with 0.455s hot TTFT but decoded at only 32.4 tok/s. A K7 GPU-only
30-turn run started with an already-hot prompt after the context ladder, so its
headline cache result was rejected as cold-start evidence rather than promoted.

GPU-only K7 preserved the full 50K-500K context ladder, including a 79.0s 500K
endpoint, but did not improve fixed decode and increased the measured 4-session
TTFT p95. At the end of control sampling, vLLM's cumulative counters reported
13.06 GB GPU-to-CPU movement and 74.2 MB restored to GPU. The tier therefore
supplies real spill capacity without a measurable fixed-decode tax.

The production decision remains K7 + 12 GB native CPU-KV. After the matrix, the
service was restored to that profile and passed the 18/18 runtime audit, health,
141.9 tok/s decode-512, 129.2/236.0/374.8/549.2 tok/s C1-C8 and zero-preemption
checks.

## 22. Configured context-ceiling overhead

The next control tested whether the required 524,288-token configured ceiling
adds measurable cost to ordinary prompts. The service received clean restarts at
262,144, 393,216 and 524,288 while keeping K7, CPU-KV 12 GB, the pinned 16 GB GPU
pool and the 3,072 scheduler profile fixed.

`bench_configured_ceiling.py` generated approximately 8K/32K/100K prompts and
assigned every request a unique cache salt. The first candidate run exposed a
separate first-request JIT tax: the first 8K request took about 5.64s TTFT and
the next took about 2.10s. That sample was rejected from the ceiling comparison,
and the harness now performs an isolated warm-up before formal measurements.

Warm-runtime medians from two cold-cache samples per prompt were:

| Ceiling | actual 7,879 | actual 31,399 | actual 98,039 | observed VRAM used |
| ---: | ---: | ---: | ---: | ---: |
| **524,288** | **2.096s** | 8.852s | **30.316s** | ~194.26 GB |
| 393,216 | **2.096s** | **8.849s** | 30.328s | ~193.19 GB |
| 262,144 | **2.096s** | 8.860s | 30.354s | ~192.75 GB |

Every formal request reported zero cached tokens and the preemption delta stayed
zero. Decode medians were also flat within fixture variance. Reducing the ceiling
saved only about 1.1-1.5 GB of observed VRAM and did not improve TTFT; the largest
measured short/medium-prompt change was about 0.13% and was in the slower
direction. Because the smaller candidates also remove the required 500K-class
admission, 524,288 remains the production ceiling.

After restoration with no environment overrides, the service again passed the
18/18 runtime audit, health, 141.8 tok/s decode-512, 129.3/233.1/376.1/549.2
tok/s C1-C8 and zero-preemption checks. The C2 point is normal single-run fixture
variance and remains within the established production band.

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
|---|---:|
| target model weight load | ~104.3s |
| DSpark draft weight load | ~61.8s |
| total model load | ~169.7s |
| graph capture | ~104s |
| engine profile + KV + warm-up | ~283.5s |

The first run also generated substantial caches outside `/root/.aiter`:

- `/root/.cache/torch_extensions`: custom HIP/C++ extension cache;
- `/root/.cache/comgr`: ROCm compiler/code-object cache, about 612 MB in this run.

The torch-extension cache was about 17 MB. The repository therefore persists
both caches in addition to the optional legacy AITER cache. After warm-up, graph
capture fell to roughly the 25-second range and engine profile/KV/warm-up was
observed around 55.9s.

## 3. Baseline before MI308X-specific tuning

Before changing AITER tables, the service was healthy and passed basic API and
long-context checks. Representative same-session measurements were:

| Metric | Untuned control |
|---|---:|
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
|---|---:|---:|---:|
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
|---|---:|---:|
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
|---|---:|---:|---:|
| `M56,4096x4096` | 34.45 us | 22.02 us | ~36% |
| `M56,4096x8192` | 64.34 us | 39.54 us | ~39% |
| `M64,4096x8192` | 43.38 us | 38.39 us | ~11.5% |
| `M4096,4096x8192` | 1275 us | 1025 us | ~19.6% |
| `M4096,32768x1024` | 1556 us | 1317 us | ~15.4% |

Representative standard A8W8 results were larger:

| Shape | Default | Tuned | Approx. improvement |
|---|---:|---:|---:|
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
|---|---:|
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
|---:|---:|---:|---|
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

Current 30-turn result:

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
/root/models/deepseek-ai/DeepSeek-V4-Flash-0731
```

The staging script copies to a temporary directory, compares a complete filename
+ size inventory, checks all 48 shards and key metadata, then renames atomically.
The launcher only auto-selects the hot copy when it is complete.

Observed model-load progression:

| State | Target | Draft | Total model load |
|---|---:|---:|---:|
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
active even when the long request was the only request in the engine. With the
then-current table coverage:

```text
200K long total:        ~62.3s
short alone:            ~0.08s
short during prefill:   ~3.16s
added short latency:    ~+3.08s
```

This was worse than production and was removed. The production scheduler source
is unchanged.

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

## 16. What remains worth tuning

The next useful work is above the already-promoted GEMM layer:

1. scheduler-budget A/B (`4096`, `3072`, `2048`) using the nonce-forced cold
   isolation benchmark, while also recording C1/C8 and long-prefill throughput;
2. investigate why a late short request can still wait ~2s behind an already
   dispatched cold-prefill chunk;
3. profile attention/MoE/scheduler time after the M=4096 GEMMs became much
   faster; the long-prompt engine throughput is now limited elsewhere;
4. keep tool/parser, 500K, cache and restart gates mandatory for every candidate.

## 17. Promotion policy used in this session

A tuning row or runtime change was promoted only if the relevant checks passed:

- exact hardware key (`gfx942`, `cu_num=80`);
- numerical correctness through the production operator;
- fresh-process verification when AITER's same-process extension reload was
  ambiguous;
- end-to-end service improvement for the intended workload;
- no regression in 500K context, DSpark, prefix caching or tool protocol.

This policy is intentionally conservative. The repository keeps a smaller table
that is explainable and measured instead of a large collection of tuner output.

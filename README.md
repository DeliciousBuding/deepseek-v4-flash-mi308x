# vllm-rocm-dsv4-flash

<div align="center">

**DeepSeek-V4-Flash-0731 on a single AMD Instinct MI300X / MI308X (gfx942), served with native vLLM on ROCm.**

512K configured context · 500K validated · DSpark · prefix caching · native CPU-KV tier · coding-agent benchmarks · no Docker required

[![License](https://img.shields.io/badge/License-Apache--2.0-blue.svg)](LICENSE)
[![ROCm](https://img.shields.io/badge/ROCm-7.2-red)](https://rocm.docs.amd.com/)
[![vLLM](https://img.shields.io/badge/vLLM-dev306-4B32C3)](https://github.com/vllm-project/vllm)
[![GPU](https://img.shields.io/badge/GPU-gfx942%20%7C%20192GB-ED1C24)]()

</div>

This repository is a **reproducible serving recipe and benchmark harness** for
running `deepseek-ai/DeepSeek-V4-Flash-0731` on one 192 GB-class gfx942 GPU.
The primary workload is not a one-shot chatbot: it is a **long-lived coding
agent** with a large stable prefix, growing multi-turn history, tool calls, and
mixed cold/hot requests.

The project intentionally pins the runtime and the historical upstream patch
source that produced the validated MI308X baseline. Current upstream development
is tracked and compared, but it is never silently mixed into the stable venv.

## What is validated

Measured on the local MI308X profile documented in
[`docs/PERFORMANCE.md`](docs/PERFORMANCE.md):

| Metric | Local result |
|---|---:|
| Configured context ceiling | **524,288 tokens** |
| Long-context ladder | **50K / 128K / 256K / 384K / 500K all pass** |
| 512-token single-stream generation, incl. TTFT | **133.5 tok/s** |
| C1 / C2 / C4 / C8 aggregate | **128 / 197 / 286 / 446 tok/s** |
| Repeated-prefix latency | **8.67s -> 0.49s (17.5x)** |
| Last per-request cached-token trace | **99.4% (856K / 860K prompt tokens)** |
| Hot coding-agent TTFT | **~0.21–0.32s** across trace revisions / warm states |
| Short request during 200K prefill | **+0.04s TTFT overhead** |
| Streaming tool-call validation before last disconnect | **10/10 round trips** |

Those values are **local measurements**, not copied community numbers. The next
GPU session re-runs the full regression suite with the updated benchmark harness
before any runtime migration.

## Stable runtime

```text
GPU        MI308X / MI300X class, gfx942, 192 GB
ROCm       7.2.3
Python     3.12
Torch      2.11.0+gitd0c8b1f (platform ROCm build, reused)
vLLM       0.26.1rc1.dev306+gcb8104839.rocm723
AITER      0.1.19
flydsl     0.2.4
patch src  ryanzhou/deepseek-v4-flash-mi300x
           012b9945c1e61ec7a7c7de12da58e8c7cafd92ab
```

The stable profile uses the historical upstream overlay set at that exact
commit, plus one local sandbox compatibility overlay. It is **not** equivalent
to the current public ryanzhou production main, which pins another vLLM nightly
and a changed kernel/overlay set. Any upstream migration is therefore tested in
a second environment rather than overwriting the known-good runtime.

## Why the extra work is necessary on gfx942

DeepSeek-V4-Flash combines sparse MLA, MXFP4 experts, FP8/FNUZ-sensitive paths,
DeepSeek-specific speculative decoding, and unusual long-context cache behavior.
The validated gfx942 stack includes fixes/tuning for deterministic sparse top-k,
ROCm DSpark verification, expert routing / SwiGLU behavior, sparse prefill,
AITER GEMM shapes, block-table overhead, and native CPU-KV synchronization.

The public production work by
[ryanzhou/deepseek-v4-flash-mi300x](https://github.com/ryanzhou/deepseek-v4-flash-mi300x)
is the main upstream reference for this recipe; this repository ports the
validated historical stack to a **native, Docker-less venv deployment** and adds
sandbox recovery, long-context, coding-agent, cache and protocol validation.

## Quick start

### 1. Clone and prepare the exact patch source

```bash
git clone https://github.com/DeliciousBuding/vllm-rocm-dsv4-flash.git
cd vllm-rocm-dsv4-flash
bash scripts/prepare_patch_repo.sh
```

`prepare_patch_repo.sh` fetches and checks out the exact historical upstream
commit used by the stable profile. It does not follow upstream `main`.

### 2. Create the isolated venv

```bash
bash scripts/env_setup.sh
```

The venv uses `--system-site-packages` so the platform ROCm Torch build is reused
rather than replaced. System Python/Torch remain untouched.

### 3. Install the pinned runtime

Place the pinned vLLM, AITER and flydsl wheels in `${WHEELS:-/mnt/workspace/wheels}`,
then run:

```bash
bash scripts/install_vllm_nightly.sh
```

The installer refuses to continue if the external patch checkout is not at the
expected full commit SHA. It applies the overlays, validates artifacts, and
persists the venv/AITER cache snapshots for restart recovery.

### 4. Download the model

```bash
bash scripts/01_download_model.sh dsflash
```

The serve path checks for all 48 weight shards before launch.

### 5. Audit and serve

```bash
python3 scripts/audit_runtime.py
bash scripts/02_serve_vllm.sh dsflash
```

`audit_runtime.py` verifies versions, external patch provenance, installed
overlays, the patched C++ extension, sparse-prefill artifact and persistent
restart snapshots.

## CPU-instance preparation before paying for GPU time

For environments where persistent storage is shared between CPU and GPU
instances, run:

```bash
bash scripts/preflight_cpu.sh
```

The CPU preflight performs every useful non-GPU check first:

- shell and Python benchmark syntax;
- exact historical patch checkout;
- all 48 model shards and metadata;
- exact vLLM/AITER/flydsl wheel inventory + ZIP integrity;
- venv / AITER / JIT persistent tarball readability;
- persistent-disk headroom.

A failed CPU preflight is a storage/source problem and should be fixed before
switching to a GPU instance.

## Production launch profile

The stable `dsflash` defaults are:

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
DSpark: K=7, probabilistic drafting, block rejection
--gpu-memory-utilization 0.95
```

The 12 GB CPU-KV tier is a **host constraint**, not a general recommendation:
the validated sandbox exposes only a 16 GB `/dev/shm`. Bare-metal systems may
support a much larger native CPU tier.

### A/B without editing the launcher

The performance-sensitive knobs are environment variables:

```bash
MAX_MODEL_LEN=524288
MAX_NUM_SEQS=64
MAX_BATCHED_TOKENS=4096
LONG_PREFILL_TOKEN_THRESHOLD=1024
DSPARK_ENABLED=1
DSPARK_K=7
KV_OFFLOAD_GB=12
KV_CACHE_BYTES=16000000000
GPU_MEMORY_UTILIZATION=0.95
MOE_BACKEND=triton
```

Example native-decoder baseline:

```bash
DSPARK_ENABLED=0 bash scripts/02_serve_vllm.sh dsflash
```

Example scheduler-budget A/B:

```bash
MAX_BATCHED_TOKENS=2048 bash scripts/02_serve_vllm.sh dsflash
```

The defaults remain unchanged until a candidate wins the complete agent workload
and correctness gates.

## Coding-agent benchmark suite

The benchmark directory covers different failure/performance modes instead of
collapsing everything into a single tokens/s number:

```text
scripts/bench/
├── bench_full.py                  general decode/prefill/cache/context suite
├── bench_latency.py               TTFT / decode latency fixture
├── bench_agent_trace.py           long-lived single agent, per-request cache accounting
├── bench_session_concurrency.py   N independent growing agent histories
├── bench_tool_roundtrip.py        streamed tool-call -> tool-result -> final answer
├── bench_ttft_isolation.py        short request injected during long prefill
└── collect_shapes.py              GEMM / runtime-shape inspection helper
```

The multi-turn cache metric uses each response's
`usage.prompt_tokens_details.cached_tokens`. Engine-global Prometheus counter
deltas are diagnostic only because they can include unrelated concurrent traffic.

Full tool protocol testing is intentionally separate from the synthetic
performance trace: a `role=tool` message is only emitted when it has a matching
assistant tool-call ID.

## 500K context and "adaptive" usage

`--max-model-len 524288` is an admission/configuration ceiling, not a command to
pre-allocate 512K KV for every short request. Paged KV/cache blocks are consumed
as requests actually grow. A larger configured ceiling can still influence some
planner/scheduler structures, so the next validation plan explicitly compares
256K / 384K / 512K ceilings on identical short requests rather than assuming the
upper bound is free.

The product requirement remains: **allow approximately 500K context while
preserving short-request latency and concurrency as much as possible.**

## Prefix caching for agents

Cache identity is based on token-prefix blocks, **not conversation IDs**. For a
coding harness, keep the stable material first:

```text
system instructions
-> stable tool schemas
-> repository / AGENTS.md / policy context
-> growing conversation
-> newest tool output / user turn
```

Avoid injecting timestamps, random IDs, unstable JSON key order, or reordered
tool schemas near the front of the prompt. If several untrusted users/agents
share one endpoint, use a per-trust-group `cache_salt` so prefix reuse is
isolated appropriately.

## Rejected experiments on the pinned dev306 runtime

Two FULL_AND_PIECEWISE graph-capture experiments were slower on this stack —
including a second pass extended through the 4096-token chunk size — while
consuming additional HBM and startup time. They remain disabled by default.
DSpark K=5 also lost to K=7 for the measured single-stream latency path.

Do not repeat those exact experiments unless the runtime/kernel stack changes.
Details and numbers are in [`docs/PERFORMANCE.md`](docs/PERFORMANCE.md).

## Next GPU session

The ordered test matrix is in
[`docs/GPU_VALIDATION_PLAN.md`](docs/GPU_VALIDATION_PLAN.md). It starts with a
runtime audit and default-profile regression, then measures scheduler budget,
DSpark vs native decoding, configured context-ceiling overhead, CPU-KV behavior,
and only afterward an isolated experiment against current upstream main.

vLLM's current tuning guidance makes `max_num_batched_tokens` a workload tradeoff:
smaller budgets favor decode/ITL while larger budgets favor prefill/TTFT and
throughput. The correct value here is therefore selected from the coding-agent
trace, not copied blindly from another deployment.

## Repository layout

```text
.
├── README.md
├── LICENSE
├── CONTRIBUTING.md
├── docs/
│   ├── PERFORMANCE.md
│   └── GPU_VALIDATION_PLAN.md
├── patches/
│   └── shared_offload_region.madvise-tolerant.py
└── scripts/
    ├── 00_check_env.sh
    ├── 01_download_model.sh
    ├── 02_serve_vllm.sh
    ├── 03_benchmark.sh
    ├── 04_bench_decode.py
    ├── env_setup.sh
    ├── prepare_patch_repo.sh
    ├── preflight_cpu.sh
    ├── install_vllm_nightly.sh
    ├── audit_runtime.py
    └── bench/
```

## Upstream reference snapshot

At the time of this update, the current public ryanzhou MI300X production README
reports a pinned `dev229` vLLM stack with **11.69K tok/s steady uncached C1
prefill**, **158.8 tok/s median per-stream static DSpark-7 decode**, a **1,278
tok/s C64 burst**, and **384K validated context**. Its production scheduler uses
a 4096-token budget with capacity reserved for speculative verification.

Those numbers are a useful reference, not an automatic target/config transplant:
the runtime revision, overlay inventory, host CPU-KV capacity and this project's
500K requirement differ.

## References

- [vLLM](https://github.com/vllm-project/vllm) — serving engine
- [vLLM optimization and tuning](https://docs.vllm.ai/en/latest/configuration/optimization/) — chunked-prefill / scheduler-budget tradeoffs
- [vLLM speculative configuration](https://docs.vllm.ai/en/latest/api/vllm/config/speculative/) — speculative and dynamic-K configuration
- [vLLM automatic prefix caching](https://docs.vllm.ai/en/stable/design/prefix_caching/) — block-hash cache design
- [ryanzhou/deepseek-v4-flash-mi300x](https://github.com/ryanzhou/deepseek-v4-flash-mi300x) — primary gfx942 production reference and source of the historical overlay stack
- [DeepSeek-V4-Flash-0731](https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash-0731) — model checkpoint

## Contributing

Issues and patches are welcome, particularly reproducible improvements to
fresh-prefill throughput, low-concurrency decode, long-context stability, cache
retention, or tool-call correctness on gfx942. See
[`CONTRIBUTING.md`](CONTRIBUTING.md).

## License

The repository code is licensed under Apache-2.0. Model weights and upstream
components retain their own licenses and are not redistributed here.

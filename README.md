# deepseek-v4-flash-mi308x

<div align="center">

**DeepSeek-V4-Flash-0731 on a single AMD Instinct MI300X / MI308X (gfx942), served with native vLLM on ROCm.**

512K configured context · 500K validated · DSpark · prefix caching · native CPU-KV tier · coding-agent benchmarks · no Docker required

[![License](https://img.shields.io/badge/License-Apache--2.0-blue.svg)](LICENSE)
[![ROCm](https://img.shields.io/badge/ROCm-7.2-red)](https://rocm.docs.amd.com/)
[![vLLM](https://img.shields.io/badge/vLLM-dev306-4B32C3)](https://github.com/vllm-project/vllm)
[![GPU](https://img.shields.io/badge/GPU-gfx942%20%7C%20192GB-ED1C24)]()

</div>

This repository is a **reproducible serving recipe and benchmark harness** for
`deepseek-ai/DeepSeek-V4-Flash-0731` on one 192 GB-class gfx942 GPU. The primary
workload is a **long-lived coding agent**: large stable prefix, growing multi-turn
history, streaming tool calls, mixed cold/hot requests, and a required ~500K
context ceiling.

The patch source is pinned by full commit SHA, the runtime is audited after
restart, and performance changes are accepted only after end-to-end agent and
correctness gates — not from a single tokens/s microbenchmark.

## Validated local baseline

Measured on the MI308X profile in [`docs/PERFORMANCE.md`](docs/PERFORMANCE.md):

| Metric | Result |
|---|---:|
| Configured context ceiling | **524,288 tokens** |
| Long-context ladder | **50K / 128K / 256K / 384K / 500K all pass** |
| 512-token generation incl. TTFT | **133.5 tok/s** |
| C1 / C2 / C4 / C8 aggregate | **128 / 197 / 286 / 446 tok/s** |
| Repeated-prefix latency | **8.67s -> 0.49s (17.5x)** |
| Last per-request cached-token trace | **99.4% (856K / 860K prompt tokens)** |
| Hot coding-agent TTFT | **~0.21–0.32s** across trace/warm-state revisions |
| Short request during 200K prefill | **+0.04s TTFT overhead** |
| Last streaming tool-call validation | **10/10 round trips** |

The next GPU session re-establishes these numbers with the updated benchmark
harness before changing the runtime.

## Stable runtime provenance

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

As of **2026-08-16**, that patch-source SHA is also the current upstream
`ryanzhou/deepseek-v4-flash-mi300x` `main` commit. The serving runtimes are still
not byte-identical:

```text
ryanzhou production: vLLM dev229 + AITER 0.1.19
this recipe:         vLLM dev306 + AITER 0.1.19
                     + activation=None compatibility edit for dev306
                     + local sandbox madvise fallback patch
```

Pinning the full source SHA remains important: a future branch move must not
silently change a reproducible install.

## Why gfx942 needs a dedicated recipe

DeepSeek-V4-Flash combines sparse MLA, MXFP4 experts, FP8/FNUZ-sensitive paths,
DeepSeek-specific speculative decoding and long-context hybrid KV behavior. The
validated gfx942 stack includes fixes/tuning for deterministic sparse top-k,
ROCm DSpark verification, expert routing/SwiGLU correctness, sparse prefill,
AITER GEMM shapes, block-table overhead and native CPU-KV synchronization.

The primary upstream reference is
[ryanzhou/deepseek-v4-flash-mi300x](https://github.com/ryanzhou/deepseek-v4-flash-mi300x).
This project ports that source stack to a **native Docker-less dev306 venv** and
adds restart recovery, provenance auditing, 500K-class validation, coding-agent
benchmarks and sandbox-specific compatibility.

## Quick start

### 1. Pin the patch source

```bash
git clone https://github.com/DeliciousBuding/deepseek-v4-flash-mi308x.git
cd deepseek-v4-flash-mi308x
bash scripts/prepare_patch_repo.sh
```

`prepare_patch_repo.sh` fetches the exact full SHA above and checks the files
required by the installer.

### 2. Create the isolated venv

```bash
bash scripts/env_setup.sh
```

The venv uses `--system-site-packages` so the platform ROCm Torch build is reused
rather than replaced. System Python/Torch remain untouched.

### 3. Install the pinned serving runtime

Place the exact vLLM/AITER/flydsl wheels under
`${WHEELS:-/mnt/workspace/wheels}`, then:

```bash
bash scripts/install_vllm_nightly.sh
```

The installer refuses to proceed if the patch checkout is not at the expected
full SHA. It applies the overlays, validates artifacts, and persists venv/AITER
snapshots for restart recovery.

### 4. Download the model

```bash
bash scripts/01_download_model.sh dsflash
```

The serve path checks for all **48 weight shards** before launch.

### 5. Audit and serve

```bash
python3 scripts/audit_runtime.py
bash scripts/02_serve_vllm.sh dsflash
```

`audit_runtime.py` verifies the runtime versions, patch-source revision,
installed overlays, patched C++ extension, sparse-prefill artifact and restart
snapshots.

## CPU-instance preparation

If CPU/GPU instances share persistent storage, finish the cheap work first:

```bash
bash scripts/preflight_cpu.sh
```

It checks shell/Python syntax, pins the upstream source SHA, verifies 48/48 model
shards, validates exact vLLM/AITER/flydsl wheel archives, checks persistent
venv/JIT/AITER snapshots, and reports storage headroom. A CPU-preflight failure
should be fixed **before paying for GPU time**.

## Stable launch profile

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

The 12 GB native CPU-KV tier is a host constraint, not a general recommendation:
the validated sandbox exposes only a 16 GB `/dev/shm`.

### A/B without editing the launcher

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

Examples:

```bash
# Native decoder control
DSPARK_ENABLED=0 bash scripts/02_serve_vllm.sh dsflash

# Scheduler-budget comparison
MAX_BATCHED_TOKENS=2048 bash scripts/02_serve_vllm.sh dsflash
```

Defaults stay unchanged until a candidate wins the full agent/correctness matrix.

## Coding-agent benchmark suite

```text
scripts/bench/
├── bench_full.py                  decode/prefill/cache/context suite
├── bench_latency.py               TTFT / decode latency fixture
├── bench_agent_trace.py           one growing agent; per-request cache accounting
├── bench_session_concurrency.py   N independent long-lived agent histories
├── bench_tool_roundtrip.py        streamed tool call -> tool result -> final answer
├── bench_ttft_isolation.py        short request injected during long prefill
└── collect_shapes.py              runtime/GEMM shape helper
```

Important benchmark semantics:

- cache hit attribution uses each response's
  `usage.prompt_tokens_details.cached_tokens`;
- global Prometheus cache counters are diagnostic only under concurrent traffic;
- hidden `reasoning_content` is **not** replayed into history by default;
  `--include-reasoning-history` provides the explicit comparison;
- performance traces never forge an orphan `role=tool` message;
- actual tool protocol is tested by `bench_tool_roundtrip.py`, including
  `forced` / `required` / `auto`, long prefixes and concurrent streaming.

Examples:

```bash
python3 scripts/bench/bench_agent_trace.py 30 20000
python3 scripts/bench/bench_session_concurrency.py --sessions 4 --rounds 8
python3 scripts/bench/bench_tool_roundtrip.py --mode auto --prefix-tokens 100000 --rounds 10
python3 scripts/bench/bench_tool_roundtrip.py --mode auto --prefix-tokens 100000 --rounds 16 --concurrency 4
```

## 500K context is an upper bound, not a per-request reservation

`--max-model-len 524288` does not make every short request consume 512K worth of
KV blocks. Paged KV/cache blocks are used as requests actually grow. A larger
configured bound can still influence planner/scheduler/admission structures, so
the GPU validation plan explicitly A/Bs 256K / 384K / 512K ceilings on identical
short requests instead of assuming the larger ceiling is completely free.

The product requirement remains: **allow approximately 500K context while
preserving short-request latency and concurrency as much as possible.**

## Prefix caching for coding agents

Cache identity comes from token-prefix blocks, **not conversation IDs**. Keep
stable material first:

```text
system instructions
-> stable tool schemas
-> repository / AGENTS.md / policy context
-> growing conversation
-> newest environment/tool output
```

Avoid timestamps, random request IDs, unstable JSON serialization, or reordered
tool schemas near the front. For untrusted multi-tenant sharing, use a per-trust-
group `cache_salt` to isolate prefix reuse.

## Current upstream comparison

Current upstream `main` is the same pinned source commit. Its production README
reports:

| Metric | Upstream production |
|---|---:|
| Runtime | vLLM dev229 + AITER 0.1.19 |
| Uncached C1 prefill | **11.69K tok/s steady** (11.53K median) |
| Static DSpark-7 C1 | 152.6 aggregate / **158.8 median per stream** |
| Native C1 | 67.3 aggregate |
| C64 K7 burst | **1,278 aggregate** |
| Context | **384K validated** (architecture supports 1M) |
| KV | 16 GB GPU + 96 GiB native CPU tier |
| Scheduler | 4096 budget; 384 reserved for spec, up to 3712 ordinary prefill |

These are reference points, not a direct transplant target. Our runtime base,
CPU-KV capacity and ~500K context requirement differ.

## Known issue-aware validation

The next GPU session explicitly tests failure classes currently reported in vLLM:
long-context DeepSeek-V4 DSML tool-call leakage, parser corruption under concurrent
streaming, DSpark hybrid-prefix-cache interactions, and speculative decoding
becoming throughput-negative at high batch size.

See [`docs/GPU_VALIDATION_PLAN.md`](docs/GPU_VALIDATION_PLAN.md) for the ordered
correctness/performance matrix.

## Rejected experiments on dev306

Two FULL_AND_PIECEWISE graph-capture experiments were slower on this pinned
runtime — including a second pass extended through the 4096-token chunk size —
while consuming additional HBM/startup time. DSpark K=5 also lost to K=7 for the
measured single-stream path. Do not repeat those exact tests unless the runtime
or kernel stack changes.

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

## References

- [vLLM](https://github.com/vllm-project/vllm)
- [vLLM optimization and tuning](https://docs.vllm.ai/en/latest/configuration/optimization/)
- [vLLM speculative configuration](https://docs.vllm.ai/en/latest/api/vllm/config/speculative/)
- [vLLM automatic prefix caching](https://docs.vllm.ai/en/stable/design/prefix_caching/)
- [ryanzhou/deepseek-v4-flash-mi300x](https://github.com/ryanzhou/deepseek-v4-flash-mi300x)
- [DeepSeek-V4-Flash-0731](https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash-0731)

## Contributing

Reproducible improvements to fresh-prefill throughput, low-concurrency decode,
long-context correctness, cache retention or tool-call reliability on gfx942 are
welcome. See [`CONTRIBUTING.md`](CONTRIBUTING.md).

## License

Repository code is Apache-2.0. Model weights and upstream components retain their
own licenses and are not redistributed here.

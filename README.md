# vllm-rocm-dsv4-flash

<div align="center">

**Serving DeepSeek-V4-Flash-0731 on a single AMD Instinct MI300X/MI308X (192 GB) with vLLM on ROCm — no Docker, 512K context, production tuning included.**

[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](LICENSE)
[![ROCm](https://img.shields.io/badge/ROCm-7.2-red)](https://www.amd.com/en/products/software/rocm.html)
[![vLLM](https://img.shields.io/badge/vLLM-nightly%20cb8104839-2b6fb3)](https://github.com/vllm-project/vllm)
[![GPU](https://img.shields.io/badge/GPU-MI300X%20%7C%20MI308X-ED1C24)]()

</div>

This repository is a **production recipe** for running DeepSeek-V4-Flash-0731 — a 284B-parameter MoE model (13B active) — on a **single AMD Instinct accelerator** via native vLLM on ROCm. It packages the environment setup, kernel patch stack, launch configuration, and benchmark harness that we use to serve a long-context coding-agent API with an OpenAI-compatible endpoint.

> **Why it exists**: upstream vLLM does not run DeepSeek-V4-Flash stably on gfx942. Sparse-attention `topk_indices` intermittently returns empty sets (vLLM issue #52109), crashing long generations and long prefills. The patch stack in this repo fixes the crash and unlocks the full performance headroom (DSpark speculative decoding, tuned AITER GEMM tables, fused MXFP4 kernels).

## Highlights

- **Single GPU, 284B MoE.** DeepSeek-V4-Flash-0731 fits a 192 GB MI300X/MI308X with the official FP4/FP8 checkpoint (~156 GB on disk, ~170-175 GB at runtime).
- **512K context, verified.** 50K → 500K prompt ladder passes without crashes or OOM (see [docs/PERFORMANCE.md](docs/PERFORMANCE.md)).
- **~133 tok/s single-stream decode** (incl. TTFT) with DSpark-7 speculative decoding; up to 446 tok/s aggregate at C8.
- **0.32s hot TTFT** on a 30K-token stable agent prefix (34x vs cold) — per-turn agent latency is decode-bound.
- **+0.04s TTFT isolation**: a short request mid-way through a 200K prefill gets its first token in 1.96s vs 1.91s alone.
- **17.5x prefix-cache speedup** on repeated long prefixes (cold 8.7s → hot 0.5s).
- **CPU KV offload**: pinned 16 GB GPU KV pool + native CPU layer (12 GB on this sandbox; upstream uses 96 GB on bare metal).
- **No Docker.** Native `vllm serve` in an isolated venv that reuses the system ROCm torch.
- **One-command install** of nightly vLLM + AITER + an 18-patch kernel stack, idempotent and re-runnable.
- Also serves **Qwen3.8-27B** and **Qwen3.6-35B-A3B** as fallback models with day-0 AMD support.

## Requirements

| Component | Version | Notes |
|---|---|---|
| GPU | AMD Instinct MI300X or MI308X (gfx942) | 192 GB HBM3 |
| ROCm | 7.2.x | system-installed |
| Python | 3.12 | system python untouched |
| torch | 2.11.0+gitd0c8b1f (ROCm build) | reused from system; **never replaced** |
| vLLM | `0.26.1rc1.dev306+gcb8104839.rocm723` | nightly wheel, pinned |
| AITER | 0.1.19 | `manylinux_2_34` wheel required |
| flydsl | >= 0.2.4 | runtime dependency of AITER 0.1.19 (long-prefix path) |

## Quick start

```bash
# 1. Create the isolated environment (reuses system torch)
bash scripts/env_setup.sh

# 2. Install nightly vLLM + AITER 0.1.19 + the 18-patch kernel stack (idempotent)
bash scripts/install_vllm_nightly.sh

# 3. Download weights (idempotent, resumable)
bash scripts/01_download_model.sh dsflash    # ~156 GB official FP4/FP8

# 4. Serve (OpenAI-compatible /v1/chat/completions)
bash scripts/02_serve_vllm.sh dsflash
```

Sanity check:

```bash
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer $VLLM_API_KEY" \
  -d '{"model":"deepseek-v4-flash","messages":[{"role":"user","content":"Hello!"}],"max_tokens":64}'
```

## Models

| Key | Model | Size | Notes |
|---|---|---|---|
| `dsflash` | [deepseek-ai/DeepSeek-V4-Flash-0731](https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash-0731) | 284B / 13B active, ~156 GB | **primary**. Official FP4/FP8, only official checkpoint that fits 192 GB. |
| `qwen38` | [Qwen/Qwen3.8-27B](https://huggingface.co/Qwen/Qwen3.8-27B) | 27B dense, ~56 GB | fallback. Day-0 AMD support, 262K context. |
| `qwen36` | [Qwen/Qwen3.6-35B-A3B](https://huggingface.co/Qwen/Qwen3.6-35B-A3B) | 35B / 3B active, ~70 GB | fallback. Agentic-coding specialist (Gated DeltaNet + MoE). |

## The patch stack

Upstream vLLM 0.26.0 crashes on gfx942 for DeepSeek-V4-Flash (empty `topk_indices` in sparse attention) and runs unoptimized. `install_vllm_nightly.sh` installs nightly `cb8104839` and applies a 17-file patch overlay ported from the community production stack at [ryanzhou/deepseek-v4-flash-mi300x](https://github.com/ryanzhou/deepseek-v4-flash-mi300x), plus our own environment patches in [`patches/`](patches/) (applied first when names collide):

| Category | Patches | Effect |
|---|---|---|
| **Crash fix** | `_C_stable_libtorch.topk-tiebreak-sanitize.abi3.so` | prebuilt C++ extension fixing empty-topk crash (root cause: FLAT-write / SHUFFLE-read mismatch in the indexer K-cache) |
| **Speculative decoding** | `dspark-speculator.*`, `rocm_aiter_mla.dspark-causal`, `spec-decode-utils.*` | DSpark-7: probabilistic drafting + block rejection, ~1.75x vs MTP |
| **MLA sparse attention** | `rocm_aiter_mla_sparse.decode-h32-k16`, `fused_compress_quant_cache.fnuz-shuffle` | gfx942 sparse MLA decode, causal verification |
| **MoE / GEMM** | `mxfp4.fused-silu` (+64% decode), `activation.rocm-exact-swiglu`, `gpt_oss_triton_kernels_moe.row-i8asym-candidate` | fused SiLU, exact BF16 SwiGLU, custom INT8 MoE kernel |
| **KV / cache** | `cache_utils.gather2048`, `block_table.active-width-copy`, `kv_offload_cpu_gpu_worker.load-war` | 2048-token gather, active-width block copy |
| **Tuning tables** | AITER A8W8 GEMM CSVs (decode + blockscale-bpreshuffle) | per-shape tuned GEMM configs |
| **Sandbox fix (ours)** | `shared_offload_region.madvise-tolerant.py` | degrade MADV_POPULATE_WRITE EINVAL (Kata/tmpfs) to demand paging |

Every patch is a byte-for-byte file overlay into the venv site-packages; no source rebuild is required. The kernel source is staged to `/opt/cj-moe` for runtime JIT of AITER/triton kernels.

## Launch configuration (dsflash)

The serve script pins the production configuration we validated on MI308X:

```
--kv-cache-dtype fp8_ds_mla        # MLA-compressed KV
--block-size 256                   # prefix-cache granularity
--enable-prefix-caching            # agent workload: stable prompts hit cache
--max-model-len 524288             # 512K context
--kv-cache-memory-bytes 16G        # pinned GPU KV pool
--kv-offloading-size 12G           # CPU KV layer (native backend)
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
--speculative-config '{"method":"dspark","num_speculative_tokens":7,
  "draft_sample_method":"probabilistic","rejection_sample_method":"block"}'
--gpu-memory-utilization 0.95
```

Notes:

- DSpark is the DeepSeek-specific speculative method — **not MTP** (`method=mtp` raises KeyError on this checkpoint).
- CPU KV offload requires the GPU pool to be **pinned** via `--kv-cache-memory-bytes`; offload size alone shrinks the pool below what a 512K request needs. Our sandbox caps the CPU layer at 12 GB (`/dev/shm` is 16 GB); bare metal can use 96 GB like upstream.
- On short requests TTFT is isolated by the 1024-token long-prefill threshold; cudagraph prefill capture is available via `CUDAGRAPH=1` (see PERFORMANCE.md).

## Performance

Measured on a single MI308X (192 GB, ROCm 7.2.3) — see [docs/PERFORMANCE.md](docs/PERFORMANCE.md) for the full harness output.

| Metric | Result |
|---|---|
| Single-stream decode incl. TTFT (128/512 tok) | 98.2 / 133.5 tok/s |
| Pure decode (agent harness) | ~142 tok/s |
| Concurrency C1/C2/C4/C8 aggregate | 128 / 197 / 286 / 446 tok/s |
| Agent multi-turn hot TTFT (30K stable prefix) | **0.32s** (34x vs cold) |
| Short-request TTFT during 200K prefill | 1.96s vs 1.91s alone (**+0.04s**) |
| Prefix cache (cold → hot) | 8.7s → 0.5s (**17.5x**) |
| Long-context ladder 50K/128K/256K/384K/500K | all pass, 500K = 91.7s |

Community reference (ryanzhou production stack on MI300X) reaches 168 tok/s
single-stream and 11.5K tok/s cold prefill; the remaining gap is documented
in PERFORMANCE.md rather than hidden.

## Repository layout

```
vllm-rocm-dsv4-flash/
├── docs/
│   └── PERFORMANCE.md          # measured baselines + harness methodology
├── patches/                    # our environment-specific patch overlays
└── scripts/
    ├── 00_check_env.sh         # environment probe (ROCm/vLLM/AITER/GPU/disk)
    ├── 01_download_model.sh    # idempotent weight download (qwen38|qwen36|dsflash)
    ├── 02_serve_vllm.sh        # native vllm serve launcher
    ├── 03_benchmark.sh         # concurrency sweep via vllm bench
    ├── 04_bench_decode.py      # decode / prefill / prefix-cache measurements
    ├── env_setup.sh            # venv isolation (reuses system torch)
    ├── install_vllm_nightly.sh # nightly vLLM + AITER + patch stack (idempotent)
    └── bench/                  # benchmark harness (incl. agent multi-turn)
```

## Environment isolation

- **System Python is never modified.** The ROCm torch build shipped by the platform is exactly the one vLLM nightly was compiled against — conda/pip cannot reproduce it, so the install script creates a `--system-site-packages` venv that isolates version-sensitive packages (vLLM, AITER, flydsl) while reusing system torch.
- `VLLM_ROCM_USE_AITER=1` enables the AITER MoE/FP8 kernels.
- The venv lives on local disk for speed; a tarball snapshot is kept on persistent storage for restart recovery (see the private `infra` companion repo).

## Contributing

Bug reports and patches are welcome — especially anything that closes the 133 → 168 tok/s gap. See [CONTRIBUTING.md](CONTRIBUTING.md).

## Acknowledgments

- [vllm-project/vllm](https://github.com/vllm-project/vllm) — the serving engine (Apache-2.0)
- [ryanzhou/deepseek-v4-flash-mi300x](https://github.com/ryanzhou/deepseek-v4-flash-mi300x) — community production stack for DSV4 on MI300X; source of the patch overlay
- [AMD ROCm](https://rocm.docs.amd.com/) — platform
- DeepSeek and Qwen — model weights (their respective licenses apply; weights are not redistributed here)

## License

Apache-2.0. See [LICENSE](LICENSE). Model weights are subject to their original licenses and are downloaded at runtime — they are not part of this repository.

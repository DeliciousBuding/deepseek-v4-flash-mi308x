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
- **~112 tok/s single-stream decode** with DSpark-7 speculative decoding on our MI308X baseline.
- **14x prefix-cache speedup** on repeated long prefixes (cold 18.7s → hot 1.3s at ~45K tokens).
- **No Docker.** Native `vllm serve` in an isolated venv that reuses the system ROCm torch.
- **One-command install** of nightly vLLM + AITER + a 17-patch kernel stack, idempotent and re-runnable.
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

# 2. Install nightly vLLM + AITER 0.1.19 + the 17-patch kernel stack (idempotent)
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

Upstream vLLM 0.26.0 crashes on gfx942 for DeepSeek-V4-Flash (empty `topk_indices` in sparse attention) and runs unoptimized. `install_vllm_nightly.sh` installs nightly `cb8104839` and applies a 17-file patch overlay (ported from the community production stack at [ryanzhou/deepseek-v4-flash-mi300x](https://github.com/ryanzhou/deepseek-v4-flash-mi300x)):

| Category | Patches | Effect |
|---|---|---|
| **Crash fix** | `_C_stable_libtorch.topk-tiebreak-sanitize.abi3.so` | prebuilt C++ extension fixing empty-topk crash (root cause: FLAT-write / SHUFFLE-read mismatch in the indexer K-cache) |
| **Speculative decoding** | `dspark-speculator.*`, `rocm_aiter_mla.dspark-causal`, `spec-decode-utils.*` | DSpark-7: probabilistic drafting + block rejection, ~1.75x vs MTP |
| **MLA sparse attention** | `rocm_aiter_mla_sparse.decode-h32-k16`, `fused_compress_quant_cache.fnuz-shuffle` | gfx942 sparse MLA decode, causal verification |
| **MoE / GEMM** | `mxfp4.fused-silu` (+64% decode), `activation.rocm-exact-swiglu`, `gpt_oss_triton_kernels_moe.row-i8asym-candidate` | fused SiLU, exact BF16 SwiGLU, custom INT8 MoE kernel |
| **KV / cache** | `cache_utils.gather2048`, `block_table.active-width-copy`, `kv_offload_cpu_gpu_worker.load-war` | 2048-token gather, active-width block copy |
| **Tuning tables** | AITER A8W8 GEMM CSVs (decode + blockscale-bpreshuffle) | per-shape tuned GEMM configs |

Every patch is a byte-for-byte file overlay into the venv site-packages; no source rebuild is required. The kernel source is staged to `/opt/cj-moe` for runtime JIT of AITER/triton kernels.

## Launch configuration (dsflash)

The serve script pins the production configuration we validated on MI308X:

```
--kv-cache-dtype fp8_ds_mla        # 20 GB GPU KV pool ≈ 1.93M tokens
--block-size 256                   # prefix-cache granularity
--enable-prefix-caching            # agent workload: stable prompts hit cache
--max-model-len 524288             # 512K context
--max-num-seqs 8
--max-num-batched-tokens 4096
--long-prefill-token-threshold 1024
--moe-backend triton
--enable-expert-parallel
--tokenizer-mode deepseek_v4
--reasoning-parser deepseek_v4
--tool-call-parser deepseek_v4
--enable-auto-tool-choice
--speculative-config '{"method":"dspark","num_speculative_tokens":7,
  "draft_sample_method":"probabilistic","rejection_sample_method":"block"}'
--gpu-memory-utilization 0.95     # 512K single request needs 8.94 GB KV
```

Notes:

- DSpark is the DeepSeek-specific speculative method — **not MTP** (`method=mtp` raises KeyError on this checkpoint).
- CPU KV offload is currently **off**: `--kv-offloading-size 64` shrinks the GPU KV pool to 8.1 GB, which cannot fit a 512K request. Revisit only with `--kv-cache-memory-bytes` pinned.
- On short requests TTFT is isolated by the 2048-token scheduling budget / 1024-token long-prefill threshold.

## Performance

Measured on a single MI308X (192 GB, ROCm 7.2.3) — see [docs/PERFORMANCE.md](docs/PERFORMANCE.md) for the full harness output.

| Metric | Result |
|---|---|
| Single-stream decode (DSpark-7) | **~112 tok/s** |
| Concurrency C1/C2/C4/C8 aggregate | 76 / 141 / 177 / 372 tok/s |
| DSpark mean accepted tokens | 4.5-4.9 / step |
| Prefix cache (100K prefix, cold → hot) | 18.7s → 1.3s (**14x**) |
| Long-context ladder 50K/128K/256K/384K/500K | all pass, 500K = 91.7s, peak VRAM 184 GB |

Community reference (ryanzhou production stack) reaches 168 tok/s single-stream; we are transparent about the gap and treat it as an open tuning target rather than a claim.

## Repository layout

```
vllm-rocm-dsv4-flash/
├── docs/
│   └── PERFORMANCE.md          # measured baselines + harness methodology
├── scripts/
│   ├── 00_check_env.sh         # environment probe (ROCm/vLLM/AITER/GPU/disk)
│   ├── 01_download_model.sh    # idempotent weight download (qwen38|qwen36|dsflash)
│   ├── 02_serve_vllm.sh        # native vllm serve launcher
│   ├── 03_benchmark.sh         # concurrency sweep via vllm bench
│   ├── 04_bench_decode.py      # decode / prefill / prefix-cache measurements
│   ├── install_vllm_nightly.sh # nightly vLLM + AITER + patch stack (idempotent)
│   └── bench/                  # comprehensive benchmark harness
└── scripts/env_setup.sh        # venv isolation (reuses system torch)
```

## Environment isolation

- **System Python is never modified.** The ROCm torch build shipped by the platform is exactly the one vLLM nightly was compiled against — conda/pip cannot reproduce it, so the install script creates a `--system-site-packages` venv that isolates version-sensitive packages (vLLM, AITER, flydsl) while reusing system torch.
- `VLLM_ROCM_USE_AITER=1` enables the AITER MoE/FP8 kernels.
- The venv lives on local disk for speed; a tarball snapshot is kept on persistent storage for restart recovery (see the private `infra` companion repo).

## Contributing

Bug reports and patches are welcome — especially anything that closes the 112 → 168 tok/s gap. See [CONTRIBUTING.md](CONTRIBUTING.md).

## Acknowledgments

- [vllm-project/vllm](https://github.com/vllm-project/vllm) — the serving engine (Apache-2.0)
- [ryanzhou/deepseek-v4-flash-mi300x](https://github.com/ryanzhou/deepseek-v4-flash-mi300x) — community production stack for DSV4 on MI300X; source of the patch overlay
- [AMD ROCm](https://rocm.docs.amd.com/) — platform
- DeepSeek and Qwen — model weights (their respective licenses apply; weights are not redistributed here)

## License

Apache-2.0. See [LICENSE](LICENSE). Model weights are subject to their original licenses and are downloaded at runtime — they are not part of this repository.

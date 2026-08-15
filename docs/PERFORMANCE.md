# Performance — DeepSeek-V4-Flash-0731 on a single MI308X

Measured baselines, methodology, and tuning history. All numbers below are
**measured on our hardware**, not community references.

**Environment**: MI308X (gfx942) 192 GB + nightly vLLM `cb8104839`
(`0.26.1rc1.dev306`) + AITER 0.1.19 + the 17-file patch stack + flydsl 0.2.4.

## Versions

```text
vLLM  : 0.26.1rc1.dev306+gcb8104839
AITER : 0.1.19
flydsl: 0.2.4
torch : 2.11.0+gitd0c8b1f, hip 7.2.53211
ROCm  : 7.2.3
```

## Serve configuration (v2, current)

```text
--max-model-len 524288 (512K)
--kv-cache-dtype fp8_ds_mla --block-size 256 --enable-prefix-caching
--kv-cache-memory-bytes 16G          # GPU KV pool pinned
--kv-offloading-size 12G             # CPU KV layer (sandbox /dev/shm cap)
--kv-offloading-backend native
--max-num-seqs 64 --max-num-batched-tokens 4096
--long-prefill-token-threshold 1024
--enable-expert-parallel --moe-backend triton
--tokenizer-mode deepseek_v4 --reasoning-parser deepseek_v4
--speculative-config {"method":"dspark","num_speculative_tokens":7,
  "draft_sample_method":"probabilistic","rejection_sample_method":"block"}
--gpu-memory-utilization 0.95
```

Environment-specific patch on top of the ryanzhou stack:
`patches/shared_offload_region.madvise-tolerant.py` — the sandbox kernel rejects
`MADV_POPULATE_WRITE` on tmpfs (EINVAL); the patch degrades to demand paging
instead of failing engine startup.

## Long-context ladder (all pass, no crashes)

| Target | Measured prompt tokens | Total time | Notes |
|---|---|---|---|
| 50K | 47,505 | 14.7s | OK (cold) |
| 128K | 121,605 | 27.4s | OK (partial cache hit) |
| 256K | 243,205 | 71.6s | OK |
| 384K | 364,805 | 73.4s | OK |
| 500K | 475,005 | 91.7s | OK |

- 50K–500K all complete; the engine does not crash.
- Ladder requests share a common prefix, so later requests hit the prefix
  cache and total time does not grow linearly.
- Engine-reported prompt throughput peaks at **~12.2K tok/s** during
  long-prefill windows (vs ~3.3K in the v1 config).

## Prefix cache

| Scenario | Time | Notes |
|---|---|---|
| Cold (~18K-token prefix) | 8.67s | |
| Hot (same prefix) | 0.49s | **17.5x speedup** |

- vLLM APC matches on token-prefix block hashes; cumulative hit rate ~63%
  on long-prefix reuse workloads.
- Hitting depends on **content/token prefix**, not conversation ID. Keep
  stable system/tool/repo prefixes first, variable content last.

## Coding-agent multi-turn session (bench_agent_trace.py)

8-turn simulated agent session with a ~30K-token stable context prefix:

| Metric | Result |
|---|---|
| Cold turn-1 TTFT | 12.5s |
| **Avg hot TTFT (turns 2-8)** | **0.32s** |
| Hot vs cold total latency | **34.4x** |
| Avg decode | 142 tok/s |
| Session total (8 turns) | 16.7s |

This is the primary workload target: stable system/tool/repo prefix cached,
per-turn cost dominated by decode.

## Concurrency

| Concurrency | Aggregate tok/s | Per-stream tok/s | Notes |
|---|---|---|---|
| C1 | 128.2 | 128.2 | was 76 in v1 config |
| C2 | 197.3 | 98.7 | was 141 |
| C4 | 286.2 | 71.5 | was 177 |
| C8 | 446.2 | 55.8 | was 372 |

Aggregate throughput scales monotonically; the v2 config (pinned 16 GB GPU
KV pool + 12 GB CPU layer + max-num-seqs 64) raised every level vs v1.

## Single-stream decode

| Output length | Total time (incl. TTFT) | tok/s incl. TTFT | Notes |
|---|---|---|---|
| 128 | 1.30s | 98.2 | was 66 |
| 512 | 3.83s | 133.5 | was 90.6 |

- Pure decode (excluding TTFT) measured ~142 tok/s in the agent harness.
- Target: 168 tok/s (ryanzhou production reference on MI300X). Gap is now
  ~1.2-1.5x instead of the previous ~1.5x.

## Tuning history

### v1 → v2 config (2026-08-16)

Aligned with the ryanzhou production compose (upstream commit `012b994`):

- Pinned GPU KV pool (`--kv-cache-memory-bytes 16G`) + CPU KV offload
  (`--kv-offloading-size 12G --kv-offloading-backend native`) — the earlier
  offload failure was caused by *not* pinning the GPU pool (pool shrank to
  8.1 GB and could not fit 512K). Sandbox `/dev/shm` is 16 GB and cannot be
  remounted, hence 12G instead of ryanzhou's 96G.
- `--max-num-seqs 64` (was 8).
- `/dev/shm` stale-mmap cleanup before launch (dead EngineCore cannot unlink
  its own mmap; copied from ryanzhou's entrypoint).
- `--generation-config vllm`, `--trust-remote-code`,
  `--enable-prompt-tokens-details`.
- madvise-tolerant patch for the sandbox kernel (see above).

Results vs v1 (before/after): decode-512 90.6→133.5, C1 76→128, C4 177→286,
C8 372→446, engine prefill throughput ~3.3K→~12K tok/s, prefix-cache
speedup 14x→17.5x.

### Earlier (v0 → v1)

`--moe-backend triton` + `draft-sample-method=probabilistic` +
`rejection-sample-method=block` + `--long-prefill-token-threshold 1024` +
`--max-num-batched-tokens 4096`: TTFT -40%, cold prefill +36%, decode +15%,
C4 +30%.

**Tested and rejected**: `--compilation-config cudagraph FULL_AND_PIECEWISE`
on dev306 yields no decode gain, adds 32s compile time and 6.24 GB VRAM.
Reverted. (The serve script keeps it behind `CUDAGRAPH=1` for future A/B.)

### v2 A/B round (2026-08-16)

| Experiment | Result | Verdict |
|---|---|---|
| cudagraph FULL_AND_PIECEWISE (ryanzhou capture sizes to 3712) | cold prefill 2509 vs 3222 tok/s (worse); decode 129.9 vs 133.5 (flat); +10 GB VRAM (195.2/205.8 GB); +2 min startup | **rejected** — chunked prefill uses 4096-token chunks above the 3712 graph max, so prefill stays eager; upstream's 11.5K tok/s depends on 3712-quantum alignment we don't have |
| DSpark K=5 (vs K=7) | decode-512 121.7 vs 133.5 tok/s | **rejected** — K=7 is the Pareto point for single-stream agent latency |
| AITER tuning for M=1/2 decode shapes | tuner not shipped in the AITER wheel (`tuned_gemm.py` is the config loader only); upstream tuning harness (`gen_nsplit.py`, `bench_r3_nsplit.py`) lives outside the public clone | **deferred** — requires AITER source build + CK tooling; documented as the path to the last decode mile |

## Remaining bottlenecks

| Problem | Current | Target (ryanzhou ref) | Gap | Root cause |
|---|---|---|---|---|
| Cold prefill (single stream) | ~3.3-5.2K tok/s | 11.5K tok/s | ~2.2-3.5x | 3712-quantum prefill graphs unavailable on this stack (cudagraph A/B rejected, see above) |
| Single-stream decode | ~133-142 tok/s | 168 tok/s | ~1.2x | decode GEMM shapes at M=1/2 partially tuned; wheel version delta; tuning harness not available in-sandbox |

Next steps:

1. AITER source-build tuning harness for the remaining M=1/2 decode shapes
   (requires the AITER 0.1.19 source + CK tooling; upstream's own harness is
   not published in the ryanzhou clone);
2. watch ryanzhou upstream for prefill-graph-compatible stack updates;
3. raise the CPU KV layer if the platform ever allows a larger `/dev/shm`.

## Summary

- **Usability target met**: 50K–500K all work, 17.5x prefix-cache speedup,
  healthy concurrency scaling, DSpark functioning, CPU KV offload live.
- **Agent workload validated**: hot TTFT 0.32s on a 30K stable prefix —
  per-turn latency is decode-bound.
- **Remaining gap**: prefill graphs (cudagraph) and the last mile of decode
  tuning. Tracked in the private infra repo's plan.

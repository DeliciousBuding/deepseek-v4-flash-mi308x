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

## Serve configuration

```text
--max-model-len 524288 (512K)
--gpu-memory-utilization 0.95
--kv-cache-dtype fp8_ds_mla --block-size 256 --enable-prefix-caching
--max-num-seqs 8 --max-num-batched-tokens 8192 --enable-expert-parallel
--tokenizer-mode deepseek_v4 --tool-call-parser deepseek_v4
--speculative-config {"method":"dspark","num_speculative_tokens":7}
no CPU KV offload (GPU-only)
```

## Long-context ladder (all pass, no crashes)

| Target | Measured prompt tokens | Total time | Notes |
|---|---|---|---|
| 50K | 47,505 | 20.6s | OK |
| 128K | 121,605 | 35.6s | OK |
| 256K | 243,205 | 69.3s | OK |
| 384K | 364,805 | 83.4s | OK |
| 500K | 475,005 | 91.7s | OK |

- 50K–500K all complete; the engine does not crash (flydsl 0.2.4 fix).
- Ladder requests share a common prefix, so later requests hit the prefix
  cache and total time does not grow linearly.
- Peak VRAM for the 500K request: ~184 GB / 192 GB, no OOM.

## Prefix cache

| Scenario | Time | Notes |
|---|---|---|
| Cold (random ~45K-token prefix) | 18.72s | prefill ~2400 tok/s |
| Hot (same prefix) | 1.34s | **14.0x speedup** |

- vLLM APC matches on token-prefix block hashes; cumulative hit rate ~64%
  on long-prefix reuse workloads.
- Hitting depends on **content/token prefix**, not conversation ID. Keep
  stable system/tool/repo prefixes first, variable content last.

## Concurrency (healthy after the flydsl fix)

| Concurrency | Aggregate tok/s | Per-stream tok/s | DSpark mean accepted |
|---|---|---|---|
| C1 | 76 | 76 | 4.83 |
| C2 | 141 | 71 | 4.68 |
| C4 | 177 | 44 | 4.53 |
| C8 | 372 | 47 | 4.58 |

- Aggregate throughput scales monotonically (76 → 141 → 177 → 372).
- The earlier C2≈63 / C4≈81 anomaly is gone — root cause was flydsl 0.2.0
  pushing batch>1 MoE/GEMM onto fallback slow paths; 0.2.4 restored them.
- DSpark acceptance holds at 4.5–4.9 tokens/step (50–56% accept rate).

## Single-stream decode

| Output length | Total time (incl. TTFT) | tok/s incl. TTFT | Notes |
|---|---|---|---|
| 128 | 1.9s | 66 | DSpark mean_accepted 4.00 |
| 512 | 6.4s | 79 | DSpark mean_accepted 4.89 |

- Pure decode (excluding TTFT): **~112 tok/s** (historical baseline).
- Target: 168 tok/s (ryanzhou production reference). Current gap ~1.5x.

## Tuning history (2026-08-16, aligned with ryanzhou serve flags)

After aligning `--moe-backend triton` + `draft-sample-method=probabilistic` +
`rejection-sample-method=block` + `--long-prefill-token-threshold 1024` +
`--max-num-batched-tokens 4096` + `--reasoning-parser` +
`--enable-auto-tool-choice`:

| Metric | Before | After | Change |
|---|---|---|---|
| TTFT (short prompt) | 0.10s | 0.06s | **-40%** |
| Cold prefill | 2402 tok/s | 3269 tok/s | **+36%** |
| decode-512 (incl. TTFT) | 79 tok/s | 90.6 tok/s | **+15%** |
| C4 aggregate | 177 tok/s | 230 tok/s | **+30%** |
| DSpark acceptance | 50–56% | 40–60% (probabilistic) | more uniform |

**Tested and rejected**: `--compilation-config cudagraph FULL_AND_PIECEWISE`
on dev306 yields no decode gain, adds 32s compile time and 6.24 GB VRAM.
Reverted.

## Remaining bottlenecks

| Problem | Current | Target (ryanzhou ref) | Gap | Root cause |
|---|---|---|---|---|
| Cold prefill | ~3269 tok/s | 7.9–8.5K tok/s | ~2.4x | `BLOCK_H=64` sparse prefill + 21-shape tuning missing |
| Single-stream decode | ~111 tok/s | 168 tok/s | ~1.5x | AITER GEMM shapes miss the tuning table (M=1/2 decode shapes) |

Root cause analysis: 219 GEMM shapes miss the ryanzhou tuning tables
(collected with `scripts/bench/collect_shapes.py`). Key decode shapes M=1/M=2
are absent from the tuning CSV (CSV starts at M=7). The 168 tok/s reference
combines 21-shape tuning with the `124154a88` (dev229) wheel; we run
`cb8104839` (dev306) with a tuning table that does not cover M=1/2.

Next steps:

1. Run the AITER tuning tool over M=1/2/4/8 decode shapes and append the CSV;
2. or align with the pinned ryanzhou wheel (`124154a88`; re-download + reinstall);
3. `BLOCK_H=64` sparse prefill (ryanzhou `rocm_aiter_mla_sparse.prefill-bh64.py`).

## Summary

- **Usability target met**: 50K–500K all work, 14x prefix-cache speedup,
  healthy concurrency scaling, DSpark functioning.
- **Performance materially improved**: TTFT -40%, prefill +36%, decode +15%,
  C4 +30% after aligning serve flags.
- **Remaining gap**: prefill 2.4x, decode 1.5x — GEMM shape tuning coverage
  and wheel-version differences. Tracked in the private infra repo's plan.

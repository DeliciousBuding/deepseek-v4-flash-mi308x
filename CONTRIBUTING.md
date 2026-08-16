# Contributing

Thanks for your interest in contributing to deepseek-v4-flash-mi308x!

This repository is a production recipe for serving DeepSeek-V4-Flash-0731 on a single AMD Instinct accelerator. Contributions in any of these areas are especially welcome:

- **Performance**: anything that narrows the gap between the fixed decode-512 fixture (~141.8 tok/s) and the average decode of the 30-turn agent trace (~167 tok/s) — kernel flags, GEMM tuning tables, DSpark configurations.
- **Stability**: reproductions of crashes with minimal scripts, fixes for long-context edge cases.
- **Runtime compatibility**: focused ports and validation for newer pinned vLLM, AITER, ROCm, or DeepSeek-V4-Flash revisions.
- **Docs**: corrections, clearer instructions, benchmark methodology.

## Ground rules

- The model weights are **not** redistributed here — scripts download them at runtime.
- The API key and any instance-specific state must stay out of the repository. Use environment variables (`VLLM_API_KEY`, `VLLM_API_KEY_FILE`, `VLLM_BASE_URL`, ...) — the scripts already honor them.
- Keep scripts idempotent: re-running them must be safe.
- Performance claims go into `docs/PERFORMANCE.md` with the commit hash, environment versions, and the harness command that produced them.

## Development flow

1. Fork the repository and create a feature branch.
2. Make your change; keep commits small and descriptive.
3. Validate scripts with `bash -n scripts/*.sh`.
4. Open a pull request describing the problem and the evidence for your change.

## Reporting issues

Include: GPU model, ROCm version, vLLM/AITER versions, the exact serve command, and a minimal reproduction script. Screenshots of benchmark output are fine, but raw numbers are better.

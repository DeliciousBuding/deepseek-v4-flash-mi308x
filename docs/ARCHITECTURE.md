# Architecture

This repository is deliberately narrow: it turns the official
`deepseek-ai/DeepSeek-V4-Flash-0731` checkpoint into a reproducible,
OpenAI-compatible vLLM service on a single 192 GB gfx942-class AMD Instinct GPU.

```text
OpenAI-compatible client
        |
        |  /v1/chat/completions
        v
      vLLM
        |
        |  DeepSeek V4 tokenizer/reasoning/tool parsers
        |  DSpark speculative decode
        |  prefix cache + fp8_ds_mla KV
        |  optional native CPU-KV tier
        v
DeepSeek-V4-Flash-0731
        |
        v
AMD Instinct MI308X / MI300X class (gfx942)
```

## Repository boundary

This project owns only the inference/runtime layer:

- environment and runtime validation;
- exact upstream patch pinning;
- model download and shard verification;
- vLLM launch configuration and MI308X-specific tuning-table selection;
- correctness and performance benchmarks;
- restart/provenance auditing and performance-cache snapshots.

It intentionally does **not** own:

- chat UI or user accounts;
- LiteLLM or another multi-provider gateway;
- Open WebUI deployment;
- public ingress, reverse proxy or tunnels;
- SSH/bootstrap automation;
- platform credentials or API-key persistence;
- ModelScope Studio packaging.

Those concerns belong in a separate application/Studio repository or a private
infrastructure layer. Keeping the boundary strict makes runtime regressions and
benchmark changes attributable to this serving stack.

## Public interface

The stable product boundary is the OpenAI-compatible vLLM HTTP interface. The
launcher publishes the model as `deepseek-v4-flash` by default.

Authentication is optional at this layer. When required, inject
`VLLM_API_KEY` from the deployment platform or secret manager. The public recipe
never generates, stores or commits machine credentials.

## Validated runtime versus deployment policy

The performance control profile remains documented in `PERFORMANCE.md`,
`TUNING_LOG_2026-08-16.md` and `GPU_VALIDATION_PLAN.md`. The launcher's
performance-sensitive defaults stay unchanged unless a candidate wins the full
correctness/agent workload matrix.

Host-local model staging is a performance optimization only: the launcher may
prefer a complete local hot copy, but the persistent checkpoint remains the
source from which that copy is prepared.

## Downstream Studio/application layer

A separate Studio repository may compose this backend with a gateway and UI:

```text
browser -> Open WebUI -> optional LiteLLM -> this vLLM API
```

LiteLLM belongs there when the application needs policy, virtual keys, usage
tracking, rate limits, aliases, routing or multiple upstreams. It is deliberately
not required by the single-model inference recipe.

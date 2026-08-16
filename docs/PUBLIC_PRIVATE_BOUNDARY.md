# Public / private boundary

This repository is public. Treat every committed file, comment, workflow log and
example value as internet-visible.

## Safe to publish

- model IDs and public upstream commit SHAs;
- ROCm/vLLM/AITER/flydsl versions;
- generic filesystem variables and overridable defaults;
- reproducible launch flags and benchmark fixtures;
- aggregate performance/correctness results;
- generic secret interfaces such as `VLLM_API_KEY`;
- public GitHub/ModelScope documentation links.

## Keep outside this repository

- API keys, tokens, cookies and session material;
- account identifiers that are not intentionally public;
- SSH configuration, private keys, `authorized_keys` and known-host material;
- jump hosts, private IPs, internal DNS names and tunnel identifiers;
- machine-specific bootstrap/recovery code that exposes private topology;
- admin passwords and Open WebUI/LiteLLM master keys;
- private infrastructure logs or raw support bundles;
- unpublished endpoints and private service URLs.

## Configuration rule

Public code may define the **name/interface** of a secret, never its value.
Deployment systems inject values at runtime.

```bash
# OK: deployment interface
VLLM_API_KEY="${VLLM_API_KEY:-}"

# Never commit a real credential.
```

Machine-specific paths should be environment overrides whenever they are not
intrinsic to the public recipe.

## Studio and infrastructure

The inference repository remains independent from the end-user Studio
application. A Studio repository may contain Open WebUI/LiteLLM configuration,
but real secrets still come from the deployment platform's secret store.

SSH, tunnels, private ingress and host bootstrap remain in a private
infrastructure project and must not be copied into either public repository.

## Before merging public changes

1. inspect the diff for credentials, hostnames, account IDs and private URLs;
2. run repository CI and syntax checks;
3. confirm examples use placeholders or environment variables;
4. confirm benchmark data contains no raw private prompts/logs;
5. keep model weights, wheels, caches and `.env` files untracked.

`.gitignore` is a safety net, not the security boundary. Review every public
diff before merge.

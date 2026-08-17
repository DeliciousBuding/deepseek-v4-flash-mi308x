#!/usr/bin/env python3
"""Audit the installed vLLM runtime against this repository's declared stack.

Run after a host/bootstrap restart and before performance testing. The audit is
read-only: it verifies runtime versions, the exact upstream patch-source SHA,
installed overlay bytes, the patched C++ extension, sparse-prefill artifact,
and persistent restart snapshots.

As of 2026-08-16 the pinned patch-source SHA is also ryanzhou upstream main.
The meaningful production difference is runtime base: upstream uses dev229,
while this recipe ports the overlays onto dev306 plus two compatibility/local
changes. Matching source provenance therefore does not claim byte-identical
serving runtimes.
"""
from __future__ import annotations

import hashlib
import os
from pathlib import Path
import subprocess

VENV = Path(os.environ.get("VLLM_VENV", "/root/.venvs/vllm"))
PATCH_REPO = Path(os.environ.get("PATCH_REPO", "/mnt/workspace/deepseek-v4-flash-mi300x"))
EXPECTED_PATCH_REPO_REV = os.environ.get(
    "PATCH_REPO_REV", "012b9945c1e61ec7a7c7de12da58e8c7cafd92ab"
)
RECIPE_REPO = Path(__file__).resolve().parent.parent
OWN_PATCHES = RECIPE_REPO / "patches"

PATCHES = [
    ("gpt_oss_triton_kernels_moe.row-i8asym-candidate.py", "vllm/model_executor/layers/fused_moe/experts/gpt_oss_triton_kernels_moe.py"),
    ("mxfp4.fused-silu.py", "vllm/model_executor/layers/fused_moe/oracle/mxfp4.py"),
    ("activation.rocm-exact-swiglu.py", "vllm/model_executor/layers/activation.py"),
    ("block_table.active-width-copy.py", "vllm/v1/worker/block_table.py"),
    ("deepseek_v4_amd_model.router-bf16.py", "vllm/models/deepseek_v4/amd/model.py"),
    ("triton-kernels-matmul-ogs-opt-flags.dsv4-mi300x.py", "vllm/third_party/triton_kernels/matmul_ogs_details/opt_flags.py"),
    ("fused_compress_quant_cache.fnuz-shuffle.py", "vllm/models/deepseek_v4/common/ops/fused_compress_quant_cache.py"),
    ("cache_utils.gather2048.py", "vllm/models/deepseek_v4/common/ops/cache_utils.py"),
    ("aiter_pa_mqa_logits.i64.py", "aiter/ops/triton/gluon/pa_mqa_logits.py"),
    ("rocm_aiter_mla_sparse.decode-h32-k16.py", "vllm/v1/attention/ops/rocm_aiter_mla_sparse.py"),
    ("deepseek_v4_attention.wqb-bpreshuffle.py", "vllm/models/deepseek_v4/attention.py"),
    ("deepseek_v4_rocm.wqb-bpreshuffle.py", "vllm/models/deepseek_v4/amd/rocm.py"),
    ("rocm_aiter_mla.dspark-causal.py", "vllm/v1/attention/backends/mla/rocm_aiter_mla.py"),
    ("dspark-speculator.independent-draft-gumbel.py", "vllm/v1/worker/gpu/spec_decode/dspark/speculator.py"),
    ("spec-decode-utils.independent-draft-gumbel.py", "vllm/v1/worker/gpu/spec_decode/utils.py"),
    ("kv_offload_cpu_gpu_worker.load-war.py", "vllm/v1/kv_offload/cpu/gpu_worker.py"),
    ("scheduler.contention-aware.py", "vllm/v1/core/sched/scheduler.py"),
    ("shared_offload_region.madvise-tolerant.py", "vllm/v1/kv_offload/cpu/shared_offload_region.py"),
]

TOPK_SHA256 = "a2912b897911c75d77611dcd42e4b0e0126bb8535f069045b32efc5f8f105610"


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def run(cmd: list[str]) -> str:
    p = subprocess.run(
        cmd,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )
    return p.stdout.strip()


def git_head(path: Path) -> str:
    if not (path / ".git").exists():
        return "not-a-git-checkout"
    return run(["git", "-C", str(path), "rev-parse", "HEAD"]) or "unknown"


def main() -> int:
    failures: list[str] = []
    warnings: list[str] = []
    py = VENV / "bin" / "python"
    if not py.exists():
        print(f"FAIL: venv python missing: {py}")
        return 2

    version_probe = run([
        str(py),
        "-c",
        "import importlib.metadata as m, torch, vllm, flydsl; "
        "print('python=',__import__('sys').version.split()[0]); "
        "print('vllm=',vllm.__version__); "
        "print('aiter=',m.version('amd-aiter')); "
        "print('flydsl=',flydsl.__version__); "
        "print('torch=',torch.__version__); print('hip=',torch.version.hip)",
    ])
    print("=== runtime versions ===")
    print(version_probe)
    if "flydsl= 0.2.4" not in version_probe:
        failures.append("flydsl is not exactly 0.2.4")
    if "aiter= 0.1.19" not in version_probe:
        failures.append("AITER is not exactly 0.1.19")
    if "0.26.1rc1.dev306" not in version_probe:
        failures.append("vLLM is not the pinned dev306 runtime")

    pyver = run([
        str(py),
        "-c",
        "import sys; print(f'python{sys.version_info.major}.{sys.version_info.minor}')",
    ])
    site = VENV / "lib" / pyver / "site-packages"
    recipe_head = git_head(RECIPE_REPO)
    patch_head = git_head(PATCH_REPO)

    print()
    print("=== revisions ===")
    print(f"recipe_repo={recipe_head}")
    print(f"patch_repo={patch_head}")
    print(f"expected_patch_repo={EXPECTED_PATCH_REPO_REV}")
    print(f"site_packages={site}")
    if patch_head != EXPECTED_PATCH_REPO_REV:
        failures.append(
            "external patch repo revision drift: expected "
            f"{EXPECTED_PATCH_REPO_REV}, got {patch_head}; run "
            "scripts/prepare_patch_repo.sh before reinstall/serve"
        )

    print()
    print("=== overlay byte audit ===")
    matched = 0
    for src_name, dst_rel in PATCHES:
        own = OWN_PATCHES / src_name
        upstream = PATCH_REPO / "patches" / src_name
        src = own if own.exists() else upstream
        dst = site / dst_rel
        origin = "local" if own.exists() else "upstream"
        if not src.exists():
            print(f"MISS source [{origin:8s}] {src_name}")
            failures.append(f"missing source overlay: {src_name}")
            continue
        if not dst.exists():
            print(f"MISS target [{origin:8s}] {dst_rel}")
            failures.append(f"missing installed target: {dst_rel}")
            continue

        s_src = sha256(src)
        s_dst = sha256(dst)
        ok = s_src == s_dst

        # dev306's caller still passes activation=. The upstream mxfp4 overlay
        # omits that argument, so the installed file intentionally receives one
        # signature-only compatibility edit after overlay copy.
        if src_name == "mxfp4.fused-silu.py" and not ok:
            text = dst.read_text(errors="ignore")
            if "activation=None" in text:
                print(f"COMPAT     [{origin:8s}] {src_name} -> {dst_rel}")
                matched += 1
                continue

        print(f"{'OK' if ok else 'DIFF':6s}     [{origin:8s}] {src_name} -> {dst_rel}")
        if ok:
            matched += 1
        else:
            failures.append(f"overlay hash mismatch: {dst_rel}")

    print()
    print("=== binary/artifact audit ===")
    topk = site / "vllm" / "_C_stable_libtorch.abi3.so"
    if topk.exists() and sha256(topk) == TOPK_SHA256:
        print("OK topk-tiebreak _C_stable_libtorch.abi3.so")
    else:
        print("DIFF/MISS topk-tiebreak _C_stable_libtorch.abi3.so")
        failures.append("topk patched binary missing or hash mismatch")

    opus = site / "aiter" / "jit" / "module_pa_sparse_prefill_opus942.so"
    if opus.exists():
        print(f"OK sparse-prefill opus942 module sha256={sha256(opus)[:16]}…")
    else:
        print("MISS sparse-prefill opus942 module")
        failures.append("sparse-prefill opus942 module missing")

    for path, label in [
        (Path(os.environ.get("JIT_KERNEL_SOURCE", "/opt/cj-moe")), "JIT kernel source"),
        (Path(os.environ.get("PERSIST_DIR", "/mnt/workspace/.venvs")) / "vllm.tar.gz", "persistent venv snapshot"),
    ]:
        print(f"{'OK' if path.exists() else 'MISS'} {label}")
        if not path.exists():
            failures.append(f"missing {label}")

    tuning_validator = RECIPE_REPO / "scripts" / "validate_tuning_tables.py"
    tuning_check = subprocess.run(
        [str(py), str(tuning_validator)],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )
    print(tuning_check.stdout.strip())
    if tuning_check.returncode != 0:
        failures.append("MI308X tuning table validation failed")

    # Runtime-generated caches improve restart latency but are not correctness
    # prerequisites. A fresh GPU instance can regenerate them on first warm-up.
    home = Path.home()
    persist_dir = Path(os.environ.get("PERSIST_DIR", "/mnt/workspace/.venvs"))
    for path, label in [
        (home / ".aiter", "AITER runtime cache"),
        (persist_dir / "aiter_cache.tar.gz", "persistent AITER cache snapshot"),
        (home / ".cache" / "torch_extensions", "torch_extensions runtime cache"),
        (persist_dir / "torch_ext_cache.tar.gz", "persistent torch_extensions snapshot"),
        (home / ".cache" / "comgr", "ROCm COMGR runtime cache"),
        (persist_dir / "comgr_cache.tar.gz", "persistent ROCm COMGR snapshot"),
        (home / ".triton", "Triton runtime cache"),
        (persist_dir / "triton_cache.tar.gz", "persistent Triton snapshot"),
    ]:
        print(f"{'OK' if path.exists() else 'WARN'} {label}")
        if not path.exists():
            warnings.append(f"missing warm-start artifact: {label}")

    print()
    print(f"overlay summary: {matched}/{len(PATCHES)} accepted")
    if warnings:
        print("AUDIT WARNINGS (warm-start only):")
        for item in warnings:
            print(f"  - {item}")
    if failures:
        print("AUDIT FAILED:")
        for item in failures:
            print(f"  - {item}")
        return 1

    print("AUDIT PASSED: runtime matches the declared stable dev306 port.")
    print(
        "Patch-source provenance matches the pinned upstream SHA. This does not "
        "claim equivalence to upstream's dev229 production runtime; test any "
        "dev229 comparison in a second venv."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

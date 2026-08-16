#!/usr/bin/env python3
"""bench_full.py — DS0731 综合基准
覆盖: 长上下文 prefill 阶梯 / prefix cache 冷热 / 并发 / DSpark acceptance
用法: python3 bench_full.py [prefill|prefix|concurrency|decode|all]
"""
import json, urllib.request, time, sys, concurrent.futures, re, os

def resolve_api_key():
    configured_api_key = os.environ.get("VLLM_API_KEY")
    if configured_api_key:
        return configured_api_key

    api_key_file = os.environ.get("VLLM_API_KEY_FILE")
    if not api_key_file:
        raise RuntimeError(
            "No API key configured; set VLLM_API_KEY or VLLM_API_KEY_FILE"
        )

    with open(api_key_file, encoding="utf-8") as api_key_stream:
        resolved_api_key = api_key_stream.read().strip()
    if not resolved_api_key:
        raise RuntimeError(f"VLLM_API_KEY_FILE is empty: {api_key_file}")
    return resolved_api_key


KEY = resolve_api_key()
BASE = os.environ.get("VLLM_BASE_URL", "http://127.0.0.1:8000")
MODEL = "deepseek-v4-flash"

def chat(prompt, max_tokens=64, temperature=0.0):
    body = json.dumps({"model": MODEL, "messages": [{"role": "user", "content": prompt}],
                       "max_tokens": max_tokens, "temperature": temperature}).encode()
    req = urllib.request.Request(BASE + "/v1/chat/completions", data=body,
        headers={"Content-Type": "application/json", "Authorization": "Bearer " + KEY})
    t0 = time.time()
    d = json.load(urllib.request.urlopen(req, timeout=1800))
    dt = time.time() - t0
    u = d["usage"]
    return {"total_s": dt, "prompt_tokens": u["prompt_tokens"],
            "completion_tokens": u["completion_tokens"],
            "content": d["choices"][0]["message"]["content"][:80]}

def spec_metrics():
    txt = urllib.request.urlopen(BASE + "/metrics", timeout=10).read().decode()
    def g(name):
        m = re.search(r'^vllm:%s_total\S*\s([0-9.e+]+)$' % name, txt, re.M)
        return float(m.group(1)) if m else 0.0
    return {"drafts": g("spec_decode_num_drafts"),
            "accepted": g("spec_decode_num_accepted_tokens"),
            "draft_tokens": g("spec_decode_num_draft_tokens")}

def report_acceptance(label, before, after):
    dd = after["drafts"] - before["drafts"]
    ac = after["accepted"] - before["accepted"]
    dt = after["draft_tokens"] - before["draft_tokens"]
    if dd > 0:
        print("  [DSpark] %s: mean_accepted_len=%.2f (1+%.0f/%.0f), accept_rate=%.0f%% (%.0f/%.0f)" %
              (label, 1 + ac / dd, ac, dd, 100 * ac / dt if dt else 0, ac, dt), flush=True)
    else:
        print("  [DSpark] %s: 无 draft (可能 DSpark 未触发)" % label, flush=True)

UNIT = ("The software engineering coding standards mandate camelCase for function names, "
        "snake_case for variable names, explicit error handling, single responsibility, "
        "dependency injection, comprehensive unit test coverage, and no magic numbers. ")

def make_prompt(target_tokens):
    # 粗略按 unit ~40 token 估算重复次数，实际长度以返回 prompt_tokens 为准
    return UNIT * max(1, int(target_tokens / 40))

def stage_decode():
    print("=== decode 单流 ===", flush=True)
    for n in (128, 512):
        before = spec_metrics()
        r = chat("用 Go 写一个带过期时间的并发安全 LRU 缓存，详细解释", max_tokens=n)
        report_acceptance("decode-%d" % n, before, spec_metrics())
        print("  decode-%d: %.1fs, %d tok -> %.1f tok/s (含 TTFT)" %
              (n, r["total_s"], r["completion_tokens"], r["completion_tokens"] / r["total_s"]), flush=True)

def stage_prefill():
    print("=== 长上下文 prefill 阶梯 (短 output 64) ===", flush=True)
    for target in (50000, 128000, 256000, 384000, 500000):
        p = make_prompt(target)
        before = spec_metrics()
        try:
            r = chat(p, max_tokens=64)
            report_acceptance("prefill-%dK" % (target // 1000), before, spec_metrics())
            print("  prefill-约%dK: %.1fs, prompt=%d tok, prefill~%.0f tok/s, completion=%d" %
                  (target // 1000, r["total_s"], r["prompt_tokens"],
                   r["prompt_tokens"] / r["total_s"] if r["total_s"] else 0, r["completion_tokens"]), flush=True)
        except Exception as e:
            print("  prefill-约%dK: ❌ FAIL %r" % (target // 1000, e), flush=True)

def stage_prefix():
    print("=== prefix cache 冷热 (约 100K 前缀) ===", flush=True)
    prefix = make_prompt(100000)
    before = spec_metrics()
    try:
        cold = chat(prefix + "\n总结上述规范", max_tokens=32)
        report_acceptance("prefix-cold", before, spec_metrics())
        before2 = spec_metrics()
        hot = chat(prefix + "\n总结上述规范", max_tokens=32)
        report_acceptance("prefix-hot", before2, spec_metrics())
        print("  冷: %.1fs (prompt=%d tok) | 热: %.1fs -> 加速 %.1fx" %
              (cold["total_s"], cold["prompt_tokens"], hot["total_s"],
               cold["total_s"] / hot["total_s"] if hot["total_s"] else 0), flush=True)
    except Exception as e:
        print("  prefix cache: ❌ FAIL %r" % e, flush=True)

def stage_concurrency():
    print("=== 并发 (每流 output 256) ===", flush=True)
    prompt = "写一个快速排序算法并解释其时间复杂度"
    for n in (1, 2, 4, 8):
        before = spec_metrics()
        t0 = time.time()
        try:
            with concurrent.futures.ThreadPoolExecutor(n) as ex:
                rs = list(ex.map(lambda i: chat(prompt, max_tokens=256), range(n)))
            dt = time.time() - t0
            tot = sum(r["completion_tokens"] for r in rs)
            report_acceptance("C%d" % n, before, spec_metrics())
            print("  C%d: %.1fs, 合计 %d tok -> %.1f tok/s 聚合, 每流 %.1f tok/s" %
                  (n, dt, tot, tot / dt, tot / dt / n), flush=True)
        except Exception as e:
            print("  C%d: ❌ FAIL %r" % (n, e), flush=True)

if __name__ == "__main__":
    stage = sys.argv[1] if len(sys.argv) > 1 else "all"
    if stage in ("prefill", "all"):
        stage_prefill()
    if stage in ("prefix", "all"):
        stage_prefix()
    if stage in ("concurrency", "all"):
        stage_concurrency()
    if stage in ("decode", "all"):
        stage_decode()

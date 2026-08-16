#!/usr/bin/env python3
# 04_bench_decode.py — 实测 DSV4 解码/prefill/并发/前缀缓存命中
# 前置: vLLM 已起(02_serve_vllm.sh dsflash), 监听 8000, 带 --api-key
# 用法: python3 04_bench_decode.py
# 注意: 裸 vLLM 0.26.0 在 MI300X 上长输出/长 prefill 会触发 sparse attention
#       topk 空集崩溃(见 README), 长上下文测试放到最后, 崩了也不丢前面结果。
import time, json, urllib.request, concurrent.futures, sys, os

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
URL = os.environ.get("VLLM_BASE_URL", "http://127.0.0.1:8000") + "/v1/chat/completions"

def chat(prompt, max_tokens):
    body = json.dumps({"model": "deepseek-v4-flash",
                       "messages": [{"role": "user", "content": prompt}],
                       "max_tokens": max_tokens}).encode()
    req = urllib.request.Request(URL, data=body,
        headers={"Content-Type": "application/json", "Authorization": "Bearer " + KEY})
    t0 = time.time()
    d = json.load(urllib.request.urlopen(req, timeout=300))
    dt = time.time() - t0
    return dt, d["usage"]["completion_tokens"], d["usage"]["prompt_tokens"]

def run(name, prompt, mt):
    try:
        dt, c, p = chat(prompt, mt)
        print(f"{name}: {c} tok 输出 / {dt:.2f}s = {c/dt:.1f} tok/s  (prompt {p} tok)", flush=True)
        return c / dt
    except Exception as e:
        print(f"{name}: ❌ 崩溃({e})—— 裸栈 sparse attention bug, 见 README", flush=True)
        return None

print("=== 1. 单流解码速度 ===", flush=True)
run("decode-128",  "用一句话介绍你自己", 128)
run("decode-512",  "写一个 Go 并发安全的 LRU 缓存, 带过期时间, 详细解释", 512)
run("decode-1024", "用 Python 实现一个完整的布隆过滤器, 解释每个方法的时间复杂度", 1024)

print("=== 2. 并发解码(4 并发) ===", flush=True)
def worker(i):
    return chat("写一个快速排序并解释", 256)
t0 = time.time()
with concurrent.futures.ThreadPoolExecutor(4) as ex:
    rs = list(ex.map(worker, range(4)))
dt = time.time() - t0
tot = sum(r[1] for r in rs)
print(f"4 并发: {tot} tok / {dt:.2f}s = 合计 {tot/dt:.1f} tok/s, 每流 {tot/dt/4:.1f} tok/s", flush=True)

print("=== 3. 前缀缓存命中(同一大前缀发两次) ===", flush=True)
prefix = "以下是代码规范文档:\n" + "\n".join(f"第{i}条: 函数驼峰, 变量下划线, 禁魔法数字, 显式处理错误" for i in range(1, 200))
t1, *_ = chat(prefix + "\n总结", 32)
t2, *_ = chat(prefix + "\n总结", 32)
print(f"前缀缓存: 冷 prefill {t1:.2f}s / 命中 {t2:.2f}s → 加速 {t1/t2:.1f}x", flush=True)

print("=== 4. 长上下文 prefill(可能崩, 放最后) ===", flush=True)
long_prompt = "请分析以下代码:\n" + "\n".join(f"func example{i}() {{ return {i} }}" for i in range(1, 1500))
run("prefill-~15K", long_prompt, 32)

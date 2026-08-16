#!/usr/bin/env python3
"""bench_latency.py — 精确测量 TTFT / ITL / TPOT / 吞吐（streaming 模式）

用法: python3 bench_latency.py [prompt_tokens] [output_tokens]
默认: 短 prompt, 输出 256 token

指标:
  TTFT  (Time To First Token)   = 首 token 延迟（含 prefill）
  ITL   (Inter-Token Latency)   = 相邻 token 平均间隔（decode 延迟）
  TPOT  (Time Per Output Token) = decode 总时间 / 输出 token
  decode tok/s                   = 输出 token / decode 时间
"""
import json, urllib.request, time, sys, os

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

def chat_stream(prompt, max_tokens=256, temperature=0.0):
    body = json.dumps({
        "model": "deepseek-v4-flash",
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens, "temperature": temperature, "stream": True,
        "stream_options": {"include_usage": True},
    }).encode()
    req = urllib.request.Request(URL, data=body, headers={
        "Content-Type": "application/json", "Authorization": "Bearer " + KEY})
    t0 = time.time()
    first = None
    times = []
    n_tokens = 0
    resp = urllib.request.urlopen(req, timeout=1800)
    for line in resp:
        line = line.decode("utf-8", "ignore").strip()
        if not line.startswith("data:"):
            continue
        data = line[5:].strip()
        if data == "[DONE]":
            break
        try:
            obj = json.loads(data)
        except Exception:
            continue
        # 最后一个 chunk 带 usage，拿准确 completion_tokens
        usage = obj.get("usage")
        if usage and usage.get("completion_tokens"):
            n_tokens = usage["completion_tokens"]
        now = time.time()
        if first is None:
            first = now
        times.append(now)
    t_end = time.time()
    ttft = first - t0 if first else None
    total = t_end - t0
    decode = t_end - first if first else 0
    itl = (times[-1] - times[0]) / (len(times) - 1) if len(times) > 1 else 0
    return {"ttft": ttft, "total": total, "decode": decode,
            "itl": itl, "n_tokens": n_tokens,
            "tpot": decode / n_tokens if n_tokens else 0,
            "decode_tok_s": n_tokens / decode if decode else 0}

if __name__ == "__main__":
    prompt = "用 Go 实现一个并发安全的 LRU 缓存，带过期时间，详细解释每个方法" 
    max_tokens = int(sys.argv[2]) if len(sys.argv) > 2 else 256
    r = chat_stream(prompt, max_tokens=max_tokens)
    print("TTFT=%.2fs | ITL=%.1fms | TPOT=%.1fms | decode=%.1f tok/s | total=%.2fs | %d tok" %
          (r["ttft"], r["itl"] * 1000, r["tpot"] * 1000, r["decode_tok_s"], r["total"], r["n_tokens"]))

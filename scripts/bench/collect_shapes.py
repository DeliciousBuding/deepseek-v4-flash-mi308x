#!/usr/bin/env python3
"""collect_shapes.py — 从 vllm 日志收集未命中 tuning 表的 GEMM shape

统计哪些 MoE/GEMM shape 走了 default config（未命中 ryanzhou tuning CSV），
并对照两张 CSV 看缺哪些 shape。这是 P6 kernel 对齐的第一手数据。

用法: python3 collect_shapes.py [log_file]
"""
import re, os, sys, collections, csv as csvmod

LOG = sys.argv[1] if len(sys.argv) > 1 else "/tmp/vllm_dsflash.log"
pat = re.compile(r"shape is M:(\d+), N:(\d+), K:(\d+), not found tuned config in ([\w./-]+\.csv)")

cnt = collections.Counter()
csv_of = collections.defaultdict(set)

if not os.path.exists(LOG):
    print("日志不存在:", LOG)
    sys.exit(1)

with open(LOG, errors="ignore") as f:
    for line in f:
        m = pat.search(line)
        if m:
            M, N, K = int(m.group(1)), int(m.group(2)), int(m.group(3))
            csv_file = os.path.basename(m.group(4))
            cnt[(M, N, K)] += 1
            csv_of[(M, N, K)].add(csv_file)

print("=== 未命中 tuning 的 GEMM shape（去重 %d 个）===" % len(cnt))
for (M, N, K), c in cnt.most_common():
    print("  M=%-6d N=%-6d K=%-6d x%-3d %s" % (M, N, K, c, ",".join(sorted(csv_of[(M, N, K)]))))

# 对照 ryanzhou tuning CSV，看这些 shape 是否真的缺失
print("\n=== 对照 ryanzhou tuning CSV ===")
tuning_dir = os.environ.get("PATCH_REPO", "/mnt/workspace/deepseek-v4-flash-mi300x") + "/tuning"
if os.path.isdir(tuning_dir):
    for csv_file in sorted(os.listdir(tuning_dir)):
        if not csv_file.endswith(".csv"):
            continue
        path = os.path.join(tuning_dir, csv_file)
        keys = set()
        with open(path, errors="ignore") as f:
            rd = csvmod.reader(f)
            header = next(rd, None)
            # 找 M/N/K 列
            if not header:
                continue
            idx = {h.strip().upper(): i for i, h in enumerate(header)}
            mi, ni, ki = idx.get("M"), idx.get("N"), idx.get("K")
            if mi is None:
                # 尝试常见列名
                for h in header:
                    hu = h.strip().upper()
                    if "M" == hu or hu.startswith("M_"):
                        mi = header.index(h)
                    if "N" == hu or hu.startswith("N_"):
                        ni = header.index(h)
                    if "K" == hu or hu.startswith("K_"):
                        ki = header.index(h)
            if mi is None or ni is None or ki is None:
                print("  %s: 无法解析列 %r" % (csv_file, header[:8]))
                continue
            for row in rd:
                if len(row) <= max(mi, ni, ki):
                    continue
                try:
                    keys.add((int(row[mi]), int(row[ni]), int(row[ki])))
                except (ValueError, IndexError):
                    continue
        missing = [s for s in cnt if s not in keys]
        print("  %s: 表内 %d 个 shape, 我们的 miss 里有 %d 个不在此表" %
              (csv_file, len(keys), len(missing)))
        if missing:
            for s in missing[:10]:
                print("     缺: M=%d N=%d K=%d" % s)
else:
    print("  tuning 目录不存在:", tuning_dir)

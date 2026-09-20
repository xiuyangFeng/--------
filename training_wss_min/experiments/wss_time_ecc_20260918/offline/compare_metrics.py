"""兼容性验收：两份 metrics.json 逐字段比较（数值叶子最大差、缺失/多出的键）。用法：compare_metrics.py A.json B.json [atol]"""
import json, sys, math
from pathlib import Path


def leaves(obj, prefix=""):
    if isinstance(obj, dict):
        for k, v in obj.items():
            yield from leaves(v, f"{prefix}/{k}")
    elif isinstance(obj, list):
        for i, v in enumerate(obj):
            yield from leaves(v, f"{prefix}[{i}]")
    else:
        yield prefix, obj


a = dict(leaves(json.loads(Path(sys.argv[1]).read_text())))
b = dict(leaves(json.loads(Path(sys.argv[2]).read_text())))
atol = float(sys.argv[3]) if len(sys.argv) > 3 else 1e-4
only_a, only_b = sorted(set(a) - set(b)), sorted(set(b) - set(a))
n_num = 0; worst = (0.0, None); mism = []
for k in sorted(set(a) & set(b)):
    if "/efficiency/" in k:
        continue
    x, y = a[k], b[k]
    if isinstance(x, (int, float)) and isinstance(y, (int, float)) and not isinstance(x, bool):
        n_num += 1
        if (isinstance(x, float) and math.isnan(x)) and (isinstance(y, float) and math.isnan(y)):
            continue
        d = abs(float(x) - float(y))
        if d > worst[0]:
            worst = (d, k)
        if d > atol:
            mism.append((k, x, y, d))
    elif x != y:
        mism.append((k, x, y, None))
print(f"numeric leaves compared: {n_num}; max abs diff {worst[0]:.3e} at {worst[1]}")
print(f"keys only in A: {len(only_a)}; only in B: {len(only_b)}")
for k in only_b[:20]:
    print("   +B", k)
print(f"mismatches beyond atol={atol}: {len(mism)}")
for k, x, y, d in mism[:30]:
    print(f"   {k}: {x} vs {y} (diff {d})")
sys.exit(0 if not mism and not only_a else 1)

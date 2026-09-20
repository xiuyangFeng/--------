"""Compare two b5_dump.py outputs bit-for-bit (exact equality; any drift fails)."""
from __future__ import annotations

import sys

import numpy as np

a = np.load(sys.argv[1])
b = np.load(sys.argv[2])
ka, kb = set(a.files), set(b.files)
bad = []
if ka - kb:
    bad.append(f"only in A: {sorted(ka - kb)}")
if kb - ka:
    bad.append(f"only in B: {sorted(kb - ka)}")
worst = 0.0
for k in sorted(ka & kb):
    x, y = a[k], b[k]
    if x.shape != y.shape:
        bad.append(f"{k}: shape {x.shape} vs {y.shape}")
        continue
    if np.array_equal(x, y, equal_nan=True):        # 两边同为 NaN 的元数据标量算一致
        continue
    finite = np.isfinite(x) & np.isfinite(y)
    d = float(np.max(np.abs(x[finite] - y[finite]))) if finite.any() else float("nan")
    worst = max(worst, d if np.isfinite(d) else 0.0)
    bad.append(f"{k}: max abs diff {d:.3e}")
print(f"arrays compared: {len(ka & kb)}; max abs diff {worst:.3e}")
for line in bad:
    print("  MISMATCH", line)
print("RESULT:", "PASS (bit-identical)" if not bad else f"FAIL ({len(bad)} issues)")
sys.exit(1 if bad else 0)

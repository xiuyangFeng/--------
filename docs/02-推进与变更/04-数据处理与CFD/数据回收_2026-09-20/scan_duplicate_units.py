"""重复病例扫描（只读，2026-09-22）：对新单元 + 在库 170 例，取壁面导出 1162 步的节点坐标（四舍五入 1e-7 m）与 wall-shear 列做哈希，
同节点数 + 同坐标哈希 = 同一网格（不同患者/术前术后不可能逐位相同）。09-15 已知 LIU_WEN_QI=ZHU_ZI_HAI、HOU_SHEN_QIAN≈KANG_XI_MING 即此类。
用法：python scan_duplicate_units.py [--out duplicate_units_scan.csv]
"""
from __future__ import annotations
import argparse, csv, glob, hashlib, json, sys
from collections import defaultdict
from pathlib import Path
import numpy as np
ROOT = Path("/public/newhome/cy/Digital_twin/GNN"); HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
def wall_sig(cid: str):
    hits = [p for p in glob.glob(str(ROOT / "data_new" / cid / "ascii" / "*-1162")) if not p.endswith(".log")]
    if len(hits) != 1:
        return None
    X = np.loadtxt(hits[0], delimiter=",", skiprows=1, usecols=(1, 2, 3), ndmin=2)
    w = np.loadtxt(hits[0], delimiter=",", skiprows=1, usecols=(5,), ndmin=1)
    return dict(n=len(X), coord_hash=hashlib.sha256(np.round(X, 7).tobytes()).hexdigest()[:16], wss_hash=hashlib.sha256(np.round(w, 5).tobytes()).hexdigest()[:16],
                bbox_mm=np.round(np.r_[X.min(0), X.max(0)] * 1e3, 1).tolist())
def main() -> None:
    ap = argparse.ArgumentParser(); ap.add_argument("--out", default=str(HERE / "duplicate_units_scan.csv")); a = ap.parse_args()
    new = json.load(open(HERE / "new_units_manifest.json"))["new_units"]
    split = json.load(open(ROOT / "data_wss_v5/views_v5_1/wss_min_view_v1/split_V5_train136_test34.json")); lib = split["train_cases"] + split["test_cases"]
    sig = {}
    for cid in new + lib:
        try:
            s = wall_sig(cid)
        except Exception:
            s = None
        if s: sig[cid] = s
    groups = defaultdict(list)
    for cid, s in sig.items(): groups[(s["n"], s["coord_hash"])].append(cid)
    dup = {k: v for k, v in groups.items() if len(v) > 1}
    with open(a.out, "w", newline="") as fh:
        w = csv.writer(fh); w.writerow(["case", "in_library", "n_wall_rows", "coord_hash", "wss_hash", "bbox_mm", "duplicate_of"])
        for cid, s in sorted(sig.items()):
            same = [c for c in groups[(s["n"], s["coord_hash"])] if c != cid]
            w.writerow([cid, cid in lib, s["n"], s["coord_hash"], s["wss_hash"], s["bbox_mm"], ";".join(same)])
    print(f"{len(sig)} units scanned ({len(new)} new + {len(lib)} library); identical-geometry groups: {len(dup)} → {a.out}")
    for (n, h), v in dup.items():
        print("  ", v, f"n={n}", "identical WSS:", len({sig[c]['wss_hash'] for c in v}) == 1)
if __name__ == "__main__":
    main()

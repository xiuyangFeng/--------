"""壁面导出 1162 步扫描（只读；第二批验收用的同一口径，脚本化 2026-09-22）：每例读 ascii/<prefix>-1162 的 pressure 与 wall-shear 列，
输出 n_nodes、p_mean/p_p99（Pa）、wss_p50/p99/max（Pa）。正常带：壁面压力 p99 13–20 kPa。
用法：python wall_scan_1162.py --cases-file rerun_batch3_cases.txt [--out wall_scan_1162_batch3.csv]
"""
from __future__ import annotations
import argparse, csv, glob, io
from pathlib import Path
import numpy as np
ROOT = Path("/public/newhome/cy/Digital_twin/GNN/data_new")
def scan(cid: str) -> dict:
    hits = [p for p in glob.glob(str(ROOT / cid / "ascii" / "*-1162")) if not p.endswith(".log")]
    if len(hits) != 1:
        return dict(case=cid, error=f"{len(hits)} files for step 1162")
    with open(hits[0], encoding="latin-1") as fh:
        header = [h.strip() for h in fh.readline().split(",")]
        data = np.loadtxt(fh, delimiter=",", ndmin=2)
    cols = {h: i for i, h in enumerate(header)}
    pc = next((cols[k] for k in cols if k == "pressure"), None); wc = next((cols[k] for k in cols if k.startswith("wall-shear")), None)
    if pc is None or wc is None:
        return dict(case=cid, error=f"columns {header}")
    p = data[:, pc]; w = data[:, wc]; wpos = w[w > 0] if (w > 0).sum() >= 0.5 * len(w) else w
    return dict(case=cid, n_rows=len(p), n_wall_rows=int((w > 0).sum()), p_mean_1162=f"{p.mean():.0f}", p_p99_1162=f"{np.percentile(p, 99):.0f}",
                wss_p50_1162=f"{np.percentile(wpos, 50):.2f}", wss_p99_1162=f"{np.percentile(wpos, 99):.1f}", wss_max_1162=f"{wpos.max():.1f}")
def main() -> None:
    ap = argparse.ArgumentParser(); ap.add_argument("--cases-file", required=True); ap.add_argument("--out", default=None)
    a = ap.parse_args(); here = Path(__file__).resolve().parent
    cases = [l.strip() for l in open(a.cases_file) if l.strip() and not l.startswith("#")]
    rows = [scan(c) for c in cases]
    keys = ["case", "n_rows", "n_wall_rows", "p_mean_1162", "p_p99_1162", "wss_p50_1162", "wss_p99_1162", "wss_max_1162", "error"]
    out = Path(a.out) if a.out else here / ("wall_scan_1162_" + Path(a.cases_file).stem + ".csv")
    with open(out, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=keys); w.writeheader(); [w.writerow({k: r.get(k, "") for k in keys}) for r in rows]
    for r in rows:
        print("  ".join(f"{k}={r.get(k, '')}" for k in keys if r.get(k, "") != ""))
    print(f"→ {out}")
if __name__ == "__main__":
    main()

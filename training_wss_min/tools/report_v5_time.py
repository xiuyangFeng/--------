"""Report for the time-conditioned WSS runs (T1 = R1 + t, T4 = R4 + t) on test34.

Prints: peak-frame comparison against the single-frame R1/R4 runs (physical + normalized), the per-frame R² curve,
pooled-over-frames numbers (flagged as inflated), and TAWSS metrics; also saves a PNG of the R² curve over the cycle
with the protocol inflow waveform.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
RUNS = ROOT / "runs" / "v5_rerun_20260906" / "outputs"
PAIRS = {  # time run -> single-frame reference
    "T1 = R1 + t": ("t1_lsa2_h2_logradius_time_s1234", "r1_lsa2_h2_logradius_s1234"),
    "T4 = R4 + t": ("t4_lsa2_h2_logradius_v5feat_time_s1234", "r4_lsa2_h2_logradius_v5feat_s1234"),
}
WAVEFORM = ROOT.parent / "data_wss_v5/views/wss_min_view_v1/protocol_inlet_waveform.json"


def load(run, ck="best"):
    p = RUNS / run / "eval" / f"ckpt_{ck}" / "metrics.json"
    if not p.is_file():
        return None
    m = json.loads(p.read_text(encoding="utf-8"))
    return m.get("test", m)


def f(x, d=4):
    try:
        return f"{float(x):.{d}f}"
    except (TypeError, ValueError):
        return "—"


def row(label, m):
    n = m["normalized"]
    return (f"| {label} | {f(m['field_casebalanced']['r2'])} | {f(m['field']['r2'])} | {f(m['aggregate']['r2_casemean'])} | {f(m['aggregate']['r2_casep10'])} | "
            f"{m['aggregate']['r2_negative_cases']}/{m['aggregate']['n_cases']} | {f(m['field']['mae'],3)} | {f(m['regional_field']['high_wss']['r2'])} | "
            f"{f(m['calibration']['top10_pred_true_ratio'],3)} | {f(m['calibration']['p99_pred_true_ratio'],3)} | {f(m['hotspot']['top10_iou_casemean'],3)} | {f(m['hotspot']['spearman_all_casemean'],3)} | "
            f"{f(n['field_casebalanced']['r2'])} | {f(n['aggregate']['r2_casemean'])} | {f(n['field'].get('nmae_range'))} |")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--png", default=str(ROOT / "experiments/v5_rerun_20260906/time_runs_frame_curve.png"))
    args = ap.parse_args(argv)
    hdr = ("| run·ckpt | 物理 R² cb | 物理 R² pooled | 逐例均值 | 逐例 p10 | 负 R² | MAE (Pa) | high-WSS R² | top10 比 | p99 比 | top10 IoU | Spearman | 归一化 R² cb | 归一化逐例均值 | 归一化 nMAE |")
    sep = "| --- |" + " ---: |" * 14
    print("### 峰值帧 1162（时间模型与单帧模型同口径）")
    print(hdr); print(sep)
    loaded = {}
    for label, (trun, rrun) in PAIRS.items():
        ref = load(rrun)
        if ref is not None:
            print(row(f"参照 {rrun.split('_')[0].upper()}·best（单帧训练）", ref))
        for ck in ("best", "last"):
            m = load(trun, ck)
            if m is None:
                print(f"| {label}·{ck} | (missing) |"); continue
            loaded[(label, ck)] = m
            print(row(f"{label}·{ck}（峰值帧）", m))
    print()
    wf = json.loads(WAVEFORM.read_text(encoding="utf-8")) if WAVEFORM.is_file() else None
    fig, ax = plt.subplots(figsize=(10, 4.8))
    for (label, ck), m in loaded.items():
        if ck != "best" or "frame_curve" not in m:
            continue
        curve = m["frame_curve"]
        steps = [c["step"] for c in curve]
        print(f"### {label}·best：逐帧 R²（每 5 帧一列；物理 cb / 归一化 cb / 逐例均值物理 / MAE Pa）")
        cols = curve[::5] + ([curve[-1]] if (len(curve) - 1) % 5 else [])
        print("| 步 | " + " | ".join(str(c["step"]) for c in cols) + " |"); print("| --- |" + " ---: |" * len(cols))
        print("| 物理 R² cb | " + " | ".join(f(c["r2_casebalanced_physical"],3) for c in cols) + " |")
        print("| 归一化 R² cb | " + " | ".join(f(c["r2_casebalanced_normalized"],3) for c in cols) + " |")
        print("| 逐例均值物理 | " + " | ".join(f(c["r2_casemean_physical"],3) for c in cols) + " |")
        print("| MAE Pa | " + " | ".join(f(c["mae_pa"],3) for c in cols) + " |")
        r2p = np.array([c["r2_casebalanced_physical"] for c in curve]); r2n = np.array([c["r2_casebalanced_normalized"] for c in curve])
        print(f"\n逐帧物理 R² cb：均值 {r2p.mean():.3f}，最小 {r2p.min():.3f}（步 {steps[int(r2p.argmin())]}），最大 {r2p.max():.3f}（步 {steps[int(r2p.argmax())]}）；"
              f"归一化：均值 {r2n.mean():.3f}，最小 {r2n.min():.3f}（步 {steps[int(r2n.argmin())]}）")
        po = m["pooled_frames"]; ta = m["tawss"]
        print(f"81 帧 pooled（虚高口径）：物理 R² cb {po['physical']['field_casebalanced']['r2']:.4f} / pooled {po['physical']['field']['r2']:.4f} / MAE {po['physical']['field']['mae']:.3f} Pa；归一化 R² cb {po['normalized']['field_casebalanced']['r2']:.4f}")
        print(f"TAWSS（81 帧 |WSS| 均值，Pa）：R² cb {ta['field_casebalanced']['r2']:.4f} / pooled {ta['field']['r2']:.4f} / 逐例均值 {ta['aggregate']['r2_casemean']:.4f} / p10 {ta['aggregate']['r2_casep10']:.4f} / MAE {ta['field']['mae']:.3f} / high R² {ta['regional_field']['high_wss']['r2']:.3f} / top10 比 {ta['calibration']['top10_pred_true_ratio']:.3f} / IoU {ta['hotspot']['top10_iou_casemean']:.3f}\n")
        ax.plot(steps, r2p, marker="o", ms=3, label=f"{label} physical R² cb")
        ax.plot(steps, r2n, marker="s", ms=3, ls="--", label=f"{label} normalized R² cb")
    if wf is not None:
        ax2 = ax.twinx(); ax2.plot(wf["steps"], wf["q_norm"], color="0.5", lw=1, alpha=0.7, label="Q(t)/Q_peak"); ax2.set_ylabel("Q(t)/Q_peak"); ax2.set_ylim(0, 1.05)
        ax2.legend(loc="upper right", fontsize=8)
    ax.set_xlabel("Fluent step (1120–1280 = one cardiac cycle)"); ax.set_ylabel("R² (case-balanced, test34)"); ax.grid(alpha=0.3)
    ax.axvline(1162, color="crimson", lw=0.8, ls=":"); ax.legend(loc="lower left", fontsize=8)
    ax.set_title("Per-frame R² over the cycle · time-conditioned models (best ckpt)")
    fig.tight_layout(); Path(args.png).parent.mkdir(parents=True, exist_ok=True); fig.savefig(args.png, dpi=160)
    print("curve png ->", args.png)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

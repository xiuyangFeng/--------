"""导师汇报图表生成（只读）：从已保存的 test34 逐点预测、周期标签视图与离线报告重算并出图，不做任何推理。

用法（仓库根目录）：
  /public/newhome/cy/.conda/envs/GNN/bin/python "docs/02-推进与变更/03-周期量TAWSS_OSI/TAWSS_OSI_导师汇报_2026-09-22/scripts/make_report_figures.py"

数据来源：
  - 周期标签：data_wss_v5/views_v5_1/wss_min_cycle_v1/<case>/cycle.npz（帧 0–79 各权 1/80）
  - 单头 A1/O1/O2 三 seed：training_wss_min/runs/wss_cycle_stage3_20260921/<arm>_s<seed>/eval/ckpt_best/predictions/test
  - 三头 M1 三 seed：training_wss_min/runs/wss_cycle_m1_stage3_20260922/M1_s<seed>/eval/ckpt_best/predictions/test/<channel>
  - 峰值 X5D_v51 同 seed：training_wss_min/runs/wss_v51_wave1_20260916/X5D_v51_s<seed>/eval/ckpt_best/predictions/test
  - 自由基线：复用 experiments/wss_cycle_20260920/offline/c1_baselines.py 的定义（TAWSS-null / TAWSS-ratio / OSI-null）
  - 阶段 1 / 阈值敏感性 / 逐帧指标：对应 offline JSON / CSV
调色板与图形规范：dataviz 参考调色板（分类 8 槽固定顺序、单色渐变、蓝红发散）。
"""
from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib import font_manager  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap, ListedColormap, LogNorm, BoundaryNorm, TwoSlopeNorm  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402
from scipy.stats import spearmanr  # noqa: E402

ROOT = Path("/public/newhome/cy/Digital_twin/GNN")
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "training_wss_min/experiments/wss_time_ecc_20260918/offline"))
sys.path.insert(0, str(ROOT / "training_wss_min/experiments/wss_cycle_20260920/offline"))
import c1_baselines as B  # noqa: E402
from common import TRAIN, TEST, load_frame_stats, load_oof_peak_ln, PEAK_INDEX, EPS, FLOOR, VIEW  # noqa: E402
from training_wss_min import metrics as M  # noqa: E402

HERE = Path(__file__).resolve().parent.parent
FIG = HERE / "figures"
TAB = HERE / "tables"
FIG.mkdir(exist_ok=True)
TAB.mkdir(exist_ok=True)
RUNS = ROOT / "training_wss_min/runs"
EXP = ROOT / "training_wss_min/experiments"
CYC = ROOT / "data_wss_v5/views_v5_1/wss_min_cycle_v1"
SEEDS = (1234, 7, 2025)
SEM = {0: "主动脉干", 1: "左髂总", 2: "右髂总", 3: "左髂外", 4: "左髂内", 5: "右髂外", 6: "右髂内"}
COHORT_ORDER = ("AG", "AAA", "ILO")
COHORT_NAME = {"AG": "AG（瘤体生长）", "AAA": "AAA（破裂/未破裂）", "ILO": "ILO（髂支闭塞）"}
DEPLOY_JOB = ROOT / "outputs/wss_deploy_jobs/20260922_173705_27065942b465"

# ------------------------------------------------------------------ 风格
for f in ("NotoSansCJK-Regular.ttc", "NotoSansCJK-Bold.ttc"):
    p = Path("/usr/share/fonts/opentype/noto") / f
    if p.exists():
        font_manager.fontManager.addfont(str(p))
_names = sorted({f.name for f in font_manager.fontManager.ttflist if "NotoSansCJK" in f.fname})
FONT = next((n for n in _names if "SC" in n), _names[0] if _names else "DejaVu Sans")
C = dict(blue="#2a78d6", orange="#eb6834", aqua="#1baf7a", yellow="#eda100", magenta="#e87ba4",
         green="#008300", violet="#4a3aa7", red="#e34948")
INK = dict(primary="#0b0b0b", secondary="#52514e", muted="#898781", grid="#e1e0d9", axis="#c3c2b7", surface="#fcfcfb")
# 固定角色着色（全篇一致）：直接回归单头=蓝，自由基线=橙，三头 M1=青，ratio 基线=黄，O2(logit)=紫，峰值参照=深灰
ROLE = dict(direct=C["blue"], null=C["orange"], m1=C["aqua"], ratio=C["yellow"], o2=C["violet"], peak=INK["secondary"])
COH = {"AG": C["blue"], "AAA": C["orange"], "ILO": C["aqua"]}
# 壁面场色图（改这里即可整体换风格）：TAWSS/WSS 用 plasma（深蓝 = 低，紫红 = 中，亮黄 = 高，感知均匀，主体不发黑）；
# 想要 ParaView 风格可改 "turbo"（彩虹，非感知均匀）或 "coolwarm"（蓝白红）。
CMAP_TAWSS, CMAP_OSI, CMAP_ERR = "plasma", "viridis", "coolwarm"
CM_TAWSS = plt.get_cmap(CMAP_TAWSS)
CM_OSI = plt.get_cmap(CMAP_OSI)      # 深蓝 = 单向，黄 = 强振荡
CM_DIV = plt.get_cmap(CMAP_ERR)      # 蓝 − 浅灰 − 红，浅灰中点在浅底上仍能看到血管轮廓
CM_COUNT = LinearSegmentedColormap.from_list("cnt", ["#f3f7fd", "#9ec5f4", "#3987e5", "#184f95", "#0d366b"])
plt.rcParams.update({
    "font.family": FONT, "font.size": 10, "axes.titlesize": 11, "axes.labelsize": 10,
    "axes.edgecolor": INK["axis"], "axes.labelcolor": INK["secondary"], "xtick.color": INK["secondary"],
    "ytick.color": INK["secondary"], "axes.spines.top": False, "axes.spines.right": False,
    "grid.color": INK["grid"], "grid.linewidth": 0.8, "axes.grid": True, "axes.axisbelow": True,
    "figure.facecolor": INK["surface"], "axes.facecolor": INK["surface"], "savefig.facecolor": INK["surface"],
    "legend.frameon": False, "axes.unicode_minus": False, "text.color": INK["primary"], "axes.titlecolor": INK["primary"],
    "figure.dpi": 110, "savefig.dpi": 170, "savefig.bbox": "tight",
})
BOLD = dict(fontweight="bold")


def save(fig, name):
    fig.savefig(FIG / f"{name}.png")
    plt.close(fig)
    print("saved", name)


def write_table(name, header, rows, note=None):
    with open(TAB / f"{name}.csv", "w", newline="", encoding="utf-8-sig") as fh:
        w = csv.writer(fh)
        w.writerow(header)
        w.writerows(rows)
    lines = ["| " + " | ".join(header) + " |", "|" + "|".join(["---"] * len(header)) + "|"]
    for r in rows:
        lines.append("| " + " | ".join(str(x) for x in r) + " |")
    if note:
        lines += ["", note]
    (TAB / f"{name}.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("table", name)


def f3(x):
    return "—" if x is None or (isinstance(x, float) and np.isnan(x)) else f"{x:.3f}"


def f4(x):
    return "—" if x is None or (isinstance(x, float) and np.isnan(x)) else f"{x:.4f}"


def pct(x, d=1):
    return "—" if x is None or (isinstance(x, float) and np.isnan(x)) else f"{100 * x:.{d}f}%"


# ------------------------------------------------------------------ 数据读取
def lnf(x):
    return np.log(np.clip(np.asarray(x, dtype=np.float64), FLOOR, None) + EPS)


def logit_osi(x):
    z = np.clip(2.0 * np.asarray(x, dtype=np.float64), 1e-3, 1 - 1e-3)
    return np.log(z / (1 - z))


def load_pred(path, n):
    with np.load(path, allow_pickle=False) as z:
        p = z["pred_pa"].astype(np.float64)
        t = z["true_pa"].astype(np.float64)
        ri = z["row_index"].astype(np.int64)
    if not (len(ri) == n and np.array_equal(ri, np.arange(n))):
        fp = np.full(n, np.nan)
        ft = np.full(n, np.nan)
        fp[ri] = p
        ft[ri] = t
        p, t = fp, ft
    return p, t


def iou(a, b):
    u = np.count_nonzero(a | b)
    return float(np.count_nonzero(a & b) / u) if u else np.nan


WARN = []


def ens(paths, n, check=None, tag=""):
    ps = []
    for pth in paths:
        p, t = load_pred(pth, n)
        ps.append(p)
        if check is not None:
            d = float(np.nanmax(np.abs(t - check)))
            if d > 1e-3 * max(1.0, float(np.nanmax(np.abs(check)))):
                WARN.append(f"{tag}: true 与标签最大差 {d:.3g}")
    return np.mean(ps, 0), ps


print("loading baselines (train136 stats / ratio / isotonic OSI-null) ...", flush=True)
MU, SD, _ = load_frame_stats("train136")
C_RATIO, _ = B.fit_ratio(TRAIN)
ISO, _ = B.fit_osi_null(TRAIN, "train")

CASES = []
for cid in TEST:
    c = B.load_cycle(cid)
    n = len(c["ids"])
    with np.load(VIEW / cid / "bundle.npz", allow_pickle=False) as z:
        assert np.array_equal(z["wall_node_id_cas"], c["ids"]), cid
        pos = z["wall_coords_aligned_mm"].astype(np.float64)
        sem = z["wall_semantic_id"].astype(int)
        peak = z["wall_wss"][PEAK_INDEX].astype(np.float64)
    d = dict(cid=cid, cohort=cid.split("/")[0], n=n, pos=pos, sem=sem, tawss=c["tawss"], osi=c["osi"], peak=peak)
    lp = load_oof_peak_ln(cid, "test", c["ids"])
    zsc = (lp - MU[PEAK_INDEX]) / SD[PEAK_INDEX]
    ln = MU[:80, None] + SD[:80, None] * zsc[None, :]
    d["tawss_null"] = np.clip(np.exp(ln) - EPS, 0, None).mean(0)
    d["tawss_ratio"] = C_RATIO * np.clip(np.exp(lp) - EPS, 0, None)
    d["osi_null"] = np.clip(ISO.predict(lp), 0.0, 0.5)
    d["peak_x5d5"] = np.clip(np.exp(lp) - EPS, 0, None)
    s3 = RUNS / "wss_cycle_stage3_20260921"
    m3 = RUNS / "wss_cycle_m1_stage3_20260922"
    x5 = RUNS / "wss_v51_wave1_20260916"
    d["A1"], d["A1_seeds"] = ens([s3 / f"A1_s{s}/eval/ckpt_best/predictions/test/{cid}/predictions.npz" for s in SEEDS], n, c["tawss"], f"{cid} A1")
    d["O1"], d["O1_seeds"] = ens([s3 / f"O1_s{s}/eval/ckpt_best/predictions/test/{cid}/predictions.npz" for s in SEEDS], n, c["osi"], f"{cid} O1")
    d["O2"], d["O2_seeds"] = ens([s3 / f"O2_s{s}/eval/ckpt_best/predictions/test/{cid}/predictions.npz" for s in SEEDS], n, c["osi"], f"{cid} O2")
    d["M1_tawss"], d["M1_tawss_seeds"] = ens([m3 / f"M1_s{s}/eval/ckpt_best/predictions/test/tawss/{cid}/predictions.npz" for s in SEEDS], n, c["tawss"], f"{cid} M1 tawss")
    d["M1_osi"], d["M1_osi_seeds"] = ens([m3 / f"M1_s{s}/eval/ckpt_best/predictions/test/osi/{cid}/predictions.npz" for s in SEEDS], n, c["osi"], f"{cid} M1 osi")
    d["M1_wss"], d["M1_wss_seeds"] = ens([m3 / f"M1_s{s}/eval/ckpt_best/predictions/test/wss/{cid}/predictions.npz" for s in SEEDS], n, peak, f"{cid} M1 wss")
    d["X5D3"], d["X5D3_seeds"] = ens([x5 / f"X5D_v51_s{s}/eval/ckpt_best/predictions/test/{cid}/predictions.npz" for s in SEEDS], n, peak, f"{cid} X5D")
    CASES.append(d)
    print("  loaded", cid, n, flush=True)
if WARN:
    print("WARN:", *WARN[:10], sep="\n  ")
N_CASES = len(CASES)


# ------------------------------------------------------------------ 指标
def r2cb(tkey, pkey, tf=None, seed_idx=None, mask_fn=None):
    tl, pl = [], []
    for d in CASES:
        t = d[tkey]
        p = d[pkey] if seed_idx is None else d[pkey + "_seeds"][seed_idx]
        if mask_fn is not None:
            m = mask_fn(d)
            t, p = t[m], p[m]
        if tf is not None:
            t, p = tf(t), tf(p)
        tl.append(t)
        pl.append(p)
    return float(M.casebalanced_field_metrics(tl, pl)["r2"])


def per_case(tkey, pkey, kind):
    rows = []
    for d in CASES:
        t, p = d[tkey], d[pkey]
        o = dict(cid=d["cid"], cohort=d["cohort"], r2=M.r2_score(t, p), mae=float(np.abs(p - t).mean()),
                 rel_l2=float(np.linalg.norm(p - t) / np.linalg.norm(t)), nmae_max=float(np.abs(p - t).mean() / np.abs(t).max()),
                 spearman=float(spearmanr(t, p).correlation), mean_true=float(t.mean()), mean_pred=float(p.mean()))
        if kind == "tawss":
            o["r2_ln"] = M.r2_score(lnf(t), lnf(p))
            o["ccc"] = M.lin_ccc(lnf(t), lnf(p))
            tm, pm = t < 0.4, p < 0.4
            o.update(low_iou=iou(tm, pm), low_dice=M.dice_masks(tm, pm), low_true_frac=float(tm.mean()), low_pred_frac=float(pm.mean()))
            o["case_mean_rel"] = float((p.mean() - t.mean()) / t.mean())
            top = t >= np.quantile(t, 0.9)
            o["top10_ratio"] = float(p[top].mean() / t[top].mean())
            segs = {}
            for s in range(7):
                m = d["sem"] == s
                if m.sum() >= 200:
                    segs[s] = float((p[m].mean() - t[m].mean()) / t[m].mean())
            o["segments"] = segs
            o["seg_med"] = float(np.median(np.abs(list(segs.values())))) if segs else np.nan
            o["seg_worst"] = float(np.max(np.abs(list(segs.values())))) if segs else np.nan
        else:
            o["ccc"] = M.lin_ccc(t, p)
            for thr in (0.1, 0.2, 0.3):
                tm, pm = t > thr, p > thr
                o[f"iou_{thr}"] = iou(tm, pm)
                o[f"dice_{thr}"] = M.dice_masks(tm, pm)
                o[f"frac_err_{thr}"] = float(pm.mean() - tm.mean())
                o[f"true_frac_{thr}"] = float(tm.mean())
            o["case_mean_diff"] = float(p.mean() - t.mean())
            segs = {}
            for s in range(7):
                m = d["sem"] == s
                if m.sum() >= 200:
                    segs[s] = float(p[m].mean() - t[m].mean())
            o["segments"] = segs
        rows.append(o)
    return rows


def stag_rows(tk, ok):
    rows = []
    for d in CASES:
        tm = (d["tawss"] < 0.4) & (d["osi"] > 0.1)
        pm = (d[tk] < 0.4) & (d[ok] > 0.1)
        rows.append(dict(cid=d["cid"], cohort=d["cohort"], iou=iou(tm, pm), dice=M.dice_masks(tm, pm),
                         true_frac=float(tm.mean()), pred_frac=float(pm.mean())))
    return rows


def agg(rows, key, how="mean"):
    v = np.array([r[key] for r in rows], dtype=float)
    v = v[~np.isnan(v)]
    if how == "mean":
        return float(v.mean())
    if how == "med":
        return float(np.median(v))
    if how == "p10":
        return float(np.quantile(v, 0.1))
    if how == "min":
        return float(v.min())
    if how == "sd":
        return float(v.std(ddof=1))
    raise ValueError(how)


def ba(rows, key):
    v = np.array([r[key] for r in rows], dtype=float)
    return float(v.mean()), float(v.mean() - 1.96 * v.std(ddof=1)), float(v.mean() + 1.96 * v.std(ddof=1))


print("computing metrics ...", flush=True)
PC = {
    "A1": per_case("tawss", "A1", "tawss"), "M1_tawss": per_case("tawss", "M1_tawss", "tawss"),
    "tawss_null": per_case("tawss", "tawss_null", "tawss"), "tawss_ratio": per_case("tawss", "tawss_ratio", "tawss"),
    "O1": per_case("osi", "O1", "osi"), "O2": per_case("osi", "O2", "osi"), "M1_osi": per_case("osi", "M1_osi", "osi"),
    "osi_null": per_case("osi", "osi_null", "osi"),
    "X5D3": per_case("peak", "X5D3", "tawss"), "M1_wss": per_case("peak", "M1_wss", "tawss"), "X5D5": per_case("peak", "peak_x5d5", "tawss"),
}
STAG = {"A1xO1": stag_rows("A1", "O1"), "A1xO2": stag_rows("A1", "O2"), "M1": stag_rows("M1_tawss", "M1_osi"), "null": stag_rows("tawss_null", "osi_null")}

SUM = {}
for k, tk in (("A1", "tawss"), ("M1_tawss", "tawss"), ("tawss_null", "tawss"), ("tawss_ratio", "tawss")):
    SUM[k] = dict(r2cb_pa=r2cb(tk, k), r2cb_ln=r2cb(tk, k, lnf), r2_casemean=agg(PC[k], "r2"), r2_casep10=agg(PC[k], "r2", "p10"),
                  ccc_med=agg(PC[k], "ccc", "med"), ccc_p10=agg(PC[k], "ccc", "p10"), ccc_min=agg(PC[k], "ccc", "min"),
                  mae=agg(PC[k], "mae"), rel_l2=agg(PC[k], "rel_l2"), nmae_max=agg(PC[k], "nmae_max"), spearman=agg(PC[k], "spearman"),
                  low_iou=agg(PC[k], "low_iou"), low_dice_med=agg(PC[k], "low_dice", "med"), seg_med=agg(PC[k], "seg_med", "med"),
                  seg_worst_med=agg(PC[k], "seg_worst", "med"), top10_ratio=agg(PC[k], "top10_ratio"),
                  ba=ba(PC[k], "case_mean_rel"))
for k in ("O1", "O2", "M1_osi", "osi_null"):
    SUM[k] = dict(r2cb=r2cb("osi", k), r2cb_logit=r2cb("osi", k, logit_osi), r2_casemean=agg(PC[k], "r2"), r2_casep10=agg(PC[k], "r2", "p10"),
                  ccc_med=agg(PC[k], "ccc", "med"), ccc_p10=agg(PC[k], "ccc", "p10"), mae=agg(PC[k], "mae"), rel_l2=agg(PC[k], "rel_l2"),
                  nmae_max=agg(PC[k], "nmae_max"), spearman=agg(PC[k], "spearman"),
                  iou_01=agg(PC[k], "iou_0.1"), iou_02=agg(PC[k], "iou_0.2"), iou_03=agg(PC[k], "iou_0.3"),
                  dice_01_med=agg(PC[k], "dice_0.1", "med"), dice_03_med=agg(PC[k], "dice_0.3", "med"),
                  frac_err_01_med=float(np.median(np.abs([r["frac_err_0.1"] for r in PC[k]]))),
                  ba_mean=ba(PC[k], "case_mean_diff"), ba_frac01=ba(PC[k], "frac_err_0.1"))
for k in ("X5D3", "M1_wss", "X5D5"):
    SUM[k] = dict(r2cb_pa=r2cb("peak", k if k != "X5D5" else "peak_x5d5"), r2cb_ln=r2cb("peak", k if k != "X5D5" else "peak_x5d5", lnf),
                  r2_casemean=agg(PC[k], "r2"), top10_ratio=agg(PC[k], "top10_ratio"), mae=agg(PC[k], "mae"))
for k, rows in STAG.items():
    SUM[f"stag_{k}"] = dict(iou=agg(rows, "iou"), dice_med=agg(rows, "dice", "med"), ba_frac=ba([dict(v=r["pred_frac"] - r["true_frac"]) for r in rows], "v"))
# 逐 seed
SEED = {}
for k, tk, tf in (("A1", "tawss", None), ("M1_tawss", "tawss", None), ("A1_ln", "tawss", lnf), ("M1_tawss_ln", "tawss", lnf),
                  ("O1", "osi", None), ("O2", "osi", None), ("M1_osi", "osi", None), ("O2_logit", "osi", logit_osi), ("M1_osi_logit", "osi", logit_osi),
                  ("X5D3", "peak", None), ("M1_wss", "peak", None), ("X5D3_ln", "peak", lnf), ("M1_wss_ln", "peak", lnf)):
    base = k.replace("_ln", "").replace("_logit", "")
    SEED[k] = [r2cb(tk, base, tf, seed_idx=i) for i in range(3)]
# 尾部分解
TAIL = {}
for q in (1.0, 0.99, 0.95, 0.90):
    def mk(qq):
        return (lambda d: d["tawss"] <= np.quantile(d["tawss"], qq)) if qq < 1 else None

    def mkp(qq):
        return (lambda d: d["peak"] <= np.quantile(d["peak"], qq)) if qq < 1 else None
    TAIL[q] = dict(A1=r2cb("tawss", "A1", mask_fn=mk(q)), M1_tawss=r2cb("tawss", "M1_tawss", mask_fn=mk(q)),
                   null=r2cb("tawss", "tawss_null", mask_fn=mk(q)), ratio=r2cb("tawss", "tawss_ratio", mask_fn=mk(q)),
                   X5D3=r2cb("peak", "X5D3", mask_fn=mkp(q)), M1_wss=r2cb("peak", "M1_wss", mask_fn=mkp(q)))
# OSI 阈值敏感性（test34 集成）
OSI_THR = {}
for thr in (0.1, 0.15, 0.2, 0.25, 0.3):
    row = {}
    for k in ("osi_null", "O1", "O2", "M1_osi"):
        row[k] = float(np.nanmean([iou(d["osi"] > thr, d[k] > thr) for d in CASES]))
    row["true_frac"] = float(np.median([np.mean(d["osi"] > thr) for d in CASES]))
    OSI_THR[thr] = row

print("check vs 文档：A1_ens3 Pa %.4f ln %.4f | M1 tawss Pa %.4f ln %.4f | O1 %.4f O2 %.4f M1_osi %.4f | X5D3 Pa %.4f ln %.4f | M1 wss Pa %.4f ln %.4f | 滞留区 M1 %.3f A1xO2 %.3f null %.3f | null Pa %.4f ratio %.4f OSI-null %.4f" % (
    SUM["A1"]["r2cb_pa"], SUM["A1"]["r2cb_ln"], SUM["M1_tawss"]["r2cb_pa"], SUM["M1_tawss"]["r2cb_ln"], SUM["O1"]["r2cb"], SUM["O2"]["r2cb"], SUM["M1_osi"]["r2cb"],
    SUM["X5D3"]["r2cb_pa"], SUM["X5D3"]["r2cb_ln"], SUM["M1_wss"]["r2cb_pa"], SUM["M1_wss"]["r2cb_ln"], SUM["stag_M1"]["iou"], SUM["stag_A1xO2"]["iou"], SUM["stag_null"]["iou"],
    SUM["tawss_null"]["r2cb_pa"], SUM["tawss_ratio"]["r2cb_pa"], SUM["osi_null"]["r2cb"]), flush=True)

# ------------------------------------------------------------------ 图 1：动机——逐帧 R² 与周期均值
rows = list(csv.DictReader(open(EXP / "wss_time_ecc_20260918/analysis_20260921/frame_metric_suite_best.csv", encoding="utf-8")))
t0 = [r for r in rows if r["arm"] == "T0"]
steps = sorted({int(r["step"]) for r in t0})
fr_r2 = np.array([np.mean([float(r["r2cb"]) for r in t0 if int(r["step"]) == s]) for s in steps])
fr_q = np.array([np.mean([float(r["q_norm"]) for r in t0 if int(r["step"]) == s]) for s in steps])
fr_t = (np.array(steps) - 1120) * 0.005
i_peak = steps.index(1162)
i_min = int(np.argmin(fr_r2))
fig, (a0, a1) = plt.subplots(2, 1, figsize=(7.6, 5.2), sharex=True, gridspec_kw=dict(height_ratios=[1, 2.2], hspace=0.12))
a0.plot(fr_t, fr_q, color=INK["secondary"], lw=2)
a0.set_ylabel("入口流量\n（归一化）")
a0.axvline(fr_t[i_peak], color=INK["axis"], lw=1, ls="--")
a0.set_title("为什么直接回归周期量：逐帧 WSS 只有峰值附近好学，周期均值反而稳定", loc="left", **BOLD)
a1.plot(fr_t, fr_r2, color=ROLE["direct"], lw=2, label="逐帧 WSS 时间条件模型 T0（三折折外 R²_cb）")
a1.scatter([fr_t[i_peak]], [fr_r2[i_peak]], s=70, color=ROLE["direct"], zorder=5)
a1.annotate(f"峰值帧 R² {fr_r2[i_peak]:.2f}", (fr_t[i_peak], fr_r2[i_peak]), xytext=(10, -22), textcoords="offset points", color=INK["primary"])
a1.scatter([fr_t[i_min]], [fr_r2[i_min]], s=70, color=ROLE["null"], zorder=5)
a1.annotate(f"谷底 R² {fr_r2[i_min]:.2f}（惯性滞后/回流，相关本身低）", (fr_t[i_min], fr_r2[i_min]), xytext=(8, -16), textcoords="offset points", color=INK["primary"])
a1.axhline(0.80, color=ROLE["m1"], lw=1.6, ls="--")
a1.text(0.795, 0.83, "周期均值分量 c0（时间基臂，三折 R² 0.79 / 0.81 / 0.80）→ 直接回归 TAWSS 的依据", color=ROLE["m1"], fontsize=9, ha="right")
a1.set_ylim(-0.3, 1.0)
a1.axhline(0, color=INK["axis"], lw=1)
a1.set_ylim(-0.3, 1.0)
a1.set_xlim(0, 0.8)
a1.set_xlabel("心动周期时间 (s)，T = 0.8 s，帧 0–79 各权 1/80")
a1.set_ylabel("R²_cb（病例等权）")
a1.legend(loc="lower right")
save(fig, "fig01_motivation_frame_r2")

# ------------------------------------------------------------------ 图 2：标签结构（170 例真值）
ALL = TRAIN + TEST
xs, ys = [], []
coh_frac = {c: dict(osi01=[], osi03=[], low=[], stag=[]) for c in COHORT_ORDER}
sp_all = []
rng = np.random.default_rng(20260922)
for cid in ALL:
    c = B.load_cycle(cid)
    t, o = c["tawss"], c["osi"]
    pick = rng.choice(len(t), min(3000, len(t)), replace=False)
    xs.append(lnf(t[pick]))
    ys.append(o[pick])
    sp_all.append(spearmanr(lnf(t), o).correlation)
    co = cid.split("/")[0]
    coh_frac[co]["osi01"].append(np.mean(o > 0.1))
    coh_frac[co]["osi03"].append(np.mean(o > 0.3))
    coh_frac[co]["low"].append(np.mean(t < 0.4))
    coh_frac[co]["stag"].append(np.mean((t < 0.4) & (o > 0.1)))
xs = np.concatenate(xs)
ys = np.concatenate(ys)
fig, (a0, a1) = plt.subplots(1, 2, figsize=(11, 4.4), gridspec_kw=dict(width_ratios=[1.15, 1], wspace=0.28))
hb = a0.hexbin(xs, ys, gridsize=60, cmap=CM_COUNT, norm=LogNorm(), mincnt=1, linewidths=0.2)
bins = np.linspace(np.quantile(xs, 0.005), np.quantile(xs, 0.995), 30)
idx = np.digitize(xs, bins)
med = [np.median(ys[idx == i]) if np.any(idx == i) else np.nan for i in range(1, len(bins))]
a0.plot(0.5 * (bins[1:] + bins[:-1]), med, color=ROLE["null"], lw=2.2, label="分箱中位 OSI")
a0.set_xlabel("ln TAWSS（Pa，下限 0.05）")
a0.set_ylabel("OSI")
a0.set_title("(a) OSI 的空间结构主要由「哪里流弱」决定", loc="left", **BOLD)
a0.text(0.02, 0.96, f"170 例真值，逐例 Spearman(OSI, ln TAWSS) 中位 {np.median(sp_all):.2f}\n→ 所以要设「峰值预测单调映射」这个自由基线（OSI-null）",
        transform=a0.transAxes, va="top", fontsize=9, color=INK["secondary"])
a0.set_ylim(0, 0.56)
a0.legend(loc="center right")
cb = fig.colorbar(hb, ax=a0, pad=0.01)
cb.set_label("点数（对数）")
labels = ["OSI > 0.1", "OSI > 0.3", "TAWSS < 0.4 Pa", "滞留区\n(TAWSS<0.4 ∧ OSI>0.1)"]
keys = ["osi01", "osi03", "low", "stag"]
xpos = np.arange(len(keys))
w = 0.26
for j, co in enumerate(COHORT_ORDER):
    meds = [np.median(coh_frac[co][k]) for k in keys]
    q1 = [np.quantile(coh_frac[co][k], 0.25) for k in keys]
    q3 = [np.quantile(coh_frac[co][k], 0.75) for k in keys]
    a1.bar(xpos + (j - 1) * w, meds, width=w - 0.03, color=COH[co], label=f"{COHORT_NAME[co]} n={len(coh_frac[co][keys[0]])}")
    a1.errorbar(xpos + (j - 1) * w, meds, yerr=[np.array(meds) - np.array(q1), np.array(q3) - np.array(meds)], fmt="none", ecolor=INK["secondary"], lw=1, capsize=2)
a1.set_xticks(xpos)
a1.set_xticklabels(labels)
a1.set_ylabel("壁面面积份额（柱 = 逐例中位，线 = 四分位距）")
a1.set_title("(b) 临床关注区域在三队列的面积份额", loc="left", **BOLD)
a1.legend(loc="upper right", fontsize=9)
save(fig, "fig02_label_structure")

# ------------------------------------------------------------------ 图 3：阶段 1 门控（三折折外，单 seed）
G = json.load(open(EXP / "wss_cycle_20260920/offline/gate_report_best.json"))
C1 = json.load(open(EXP / "wss_cycle_20260920/offline/c1_baselines.json"))
folds = [0, 1, 2]
A1f = G["arms"]["A1"]
O1f = G["arms"]["O1"]
O2f = G["arms"]["O2"]
nullPa = [C1[f"fold{k}"]["TAWSS_null"]["r2_cb"] for k in folds]
ratioPa = [C1[f"fold{k}"]["TAWSS_ratio"]["r2_cb"] for k in folds]
nullLog = [C1[f"fold{k}"]["TAWSS_null"]["log_r2_cb"] for k in folds]
osinull = [C1[f"fold{k}"]["OSI_null"]["r2_cb"] for k in folds]
osinull_iou = [C1[f"fold{k}"]["OSI_null"]["threshold_masks"]["above_0.1"]["iou"] for k in folds]
stag_base = G["gates"]["G3.A1xO1"]["baseline"]
stag_o1 = G["gates"]["G3.A1xO1"]["iou"]
stag_o2 = G["gates"]["G3.A1xO2"]["iou"]


def fold_panel(ax, series, title, gate=None, gate_label=None, ylim=None, legend_loc="lower left"):
    """series: list of (label, values[3], color, marker)。x = fold0/1/2/均值。"""
    xs_ = np.arange(4)
    for i, (lab, vals, col, mk) in enumerate(series):
        off = (i - (len(series) - 1) / 2) * 0.16
        v = list(vals) + [float(np.mean(vals))]
        ax.scatter(xs_ + off, v, s=[46] * 3 + [90], color=col, marker=mk, zorder=4, label=lab)
        ax.plot(xs_[:3] + off, v[:3], color=col, lw=0.8, alpha=0.5)
    if gate is not None:
        ax.plot([-0.4, 3.4], [gate, gate], color=INK["primary"], lw=1.2, ls=(0, (4, 3)), label=gate_label or "预注册门")
    ax.set_xticks(xs_)
    ax.set_xticklabels(["折 0", "折 1", "折 2", "三折均值"])
    ax.set_title(title, loc="left", **BOLD)
    if ylim:
        ax.set_ylim(*ylim)
    ax.legend(loc=legend_loc, fontsize=8.5)


fig, axs = plt.subplots(2, 3, figsize=(15, 8.2), gridspec_kw=dict(wspace=0.32, hspace=0.42))
fold_panel(axs[0, 0], [("A1 TAWSS 直接回归", [r["norm_r2cb"] for r in A1f], ROLE["direct"], "o"),
                       ("同折峰值帧模型 X5D_v51", [r["base_norm_r2cb"] for r in A1f], ROLE["peak"], "s")],
           "(a) G1.1 TAWSS 归一化 R²_cb ≥ 峰值 − 0.02 → PASS", gate=float(np.mean([r["base_norm_r2cb"] for r in A1f])) - 0.02, gate_label="门", ylim=(0.80, 0.90))
fold_panel(axs[0, 1], [("A1 TAWSS 直接回归", [r["pa_r2cb"] for r in A1f], ROLE["direct"], "o"),
                       ("TAWSS-null（峰值预测×波形）", nullPa, ROLE["null"], "s"),
                       ("TAWSS-ratio（峰值×常数）", ratioPa, ROLE["ratio"], "^")],
           "(b) G1.2 TAWSS Pa R²_cb ≥ 基线 + 0.05 → FAIL（+0.020）", gate=max(np.mean(nullPa), np.mean(ratioPa)) + 0.05, gate_label="预注册门（基线均值 + 0.05）", ylim=(0.56, 0.82), legend_loc="lower right")
fold_panel(axs[0, 2], [("A1 直接回归（ln 空间）", [r["norm_r2cb"] for r in A1f], ROLE["direct"], "o"),
                       ("TAWSS-null（ln 空间）", nullLog, ROLE["null"], "s")],
           "(c) 同一对比放到 ln 空间：+0.118（未预注册）", ylim=(0.68, 0.90))
fold_panel(axs[1, 0], [("O1 OSI 线性", [r["pa_r2cb"] for r in O1f], ROLE["direct"], "o"),
                       ("O2 OSI logit", [r["pa_r2cb"] for r in O2f], ROLE["o2"], "D"),
                       ("OSI-null（峰值预测单调映射）", osinull, ROLE["null"], "s")],
           "(d) G2.1 OSI R²_cb > 0 → PASS（对基线 +0.11 / +0.09）", ylim=(0.30, 0.60), legend_loc="upper right")
fold_panel(axs[1, 1], [("O1", [r["above_0.1_iou"] for r in O1f], ROLE["direct"], "o"),
                       ("O2", [r["above_0.1_iou"] for r in O2f], ROLE["o2"], "D"),
                       ("OSI-null", osinull_iou, ROLE["null"], "s")],
           "(e) G2.2 OSI > 0.1 掩膜 IoU ≥ 基线 + 0.05 → FAIL（+0.02）", gate=float(np.mean(osinull_iou)) + 0.05, gate_label="门", ylim=(0.60, 0.72))
fold_panel(axs[1, 2], [("A1 × O1", stag_o1, ROLE["direct"], "o"),
                       ("A1 × O2", stag_o2, ROLE["o2"], "D"),
                       ("基线组合 null × null", stag_base, ROLE["null"], "s")],
           "(f) G3 滞留区 IoU ≥ 基线 + 0.05 → PASS（+0.10）", gate=float(np.mean(stag_base)) + 0.05, gate_label="门", ylim=(0.36, 0.58))
fig.suptitle("阶段 1 筛选：A1 / O1 / O2 × 患者分组三折（单 seed 1234，与峰值底座同配方同预算 400 epoch）", x=0.01, ha="left", fontsize=12, **BOLD)
save(fig, "fig03_stage1_gates_3fold")

# ------------------------------------------------------------------ 图 4：为什么 Pa 门不过——尾部分解（test34 集成）
qs = [1.0, 0.99, 0.95, 0.90]
xl = ["全量", "剔除每例最高 1%", "剔除最高 5%", "剔除最高 10%"]
fig, (a0, a1) = plt.subplots(1, 2, figsize=(12, 4.4), gridspec_kw=dict(wspace=0.25))
for k, lab, col, mk in (("A1", "A1 三 seed 集成（直接回归）", ROLE["direct"], "o"), ("M1_tawss", "M1 三头集成 TAWSS 头", ROLE["m1"], "D"),
                        ("null", "TAWSS-null（峰值×波形）", ROLE["null"], "s"), ("ratio", "TAWSS-ratio", ROLE["ratio"], "^")):
    a0.plot(range(4), [TAIL[q][k] for q in qs], color=col, marker=mk, lw=2, ms=7, label=lab)
a0.set_xticks(range(4))
a0.set_xticklabels(xl, fontsize=9)
a0.set_ylabel("TAWSS Pa 空间 R²_cb（test34）")
a0.set_title("(a) Pa 口径「打平」只由每例不到 1% 的极端高值点造成", loc="left", **BOLD)
a0.set_ylim(0.61, 0.84)
a0.legend(loc="lower left", fontsize=9)
a0.text(0.98, 0.97, f"ln 空间：A1 {SUM['A1']['r2cb_ln']:.3f}｜null {SUM['tawss_null']['r2cb_ln']:.3f}｜ratio {SUM['tawss_ratio']['r2cb_ln']:.3f}\n逐例 R²：A1 胜基线 {sum(r['r2'] > max(n_['r2'], q_['r2']) for r, n_, q_ in zip(PC['A1'], PC['tawss_null'], PC['tawss_ratio']))}/34 例",
        transform=a0.transAxes, ha="right", va="top", fontsize=9, color=INK["secondary"])
for k, lab, col, mk in (("X5D3", "已部署峰值模型 X5D_v51 三 seed 集成", ROLE["peak"], "s"), ("M1_wss", "M1 三头集成 峰值通道", ROLE["m1"], "D")):
    a1.plot(range(4), [TAIL[q][k] for q in qs], color=col, marker=mk, lw=2, ms=7, label=lab)
a1.set_xticks(range(4))
a1.set_xticklabels(xl, fontsize=9)
a1.set_ylabel("峰值 WSS Pa 空间 R²_cb（test34）")
a1.set_title("(b) 三头模型峰值通道的 −0.03 全在每例最高 10% 点（q90 pinball 关闭）", loc="left", **BOLD)
a1.set_ylim(0.725, 0.805)
a1.legend(loc="lower right", fontsize=9)
a1.text(0.02, 0.97, f"ln 空间：X5D {SUM['X5D3']['r2cb_ln']:.3f}｜M1 {SUM['M1_wss']['r2cb_ln']:.3f}\ntop10% 幅值比：X5D {SUM['X5D3']['top10_ratio']:.2f}｜M1 {SUM['M1_wss']['top10_ratio']:.2f}",
        transform=a1.transAxes, ha="left", va="top", fontsize=9, color=INK["secondary"])
save(fig, "fig04_tail_decomposition")

# ------------------------------------------------------------------ 图 5：OSI 阈值敏感性
S = json.load(open(EXP / "wss_cycle_20260920/offline/osi_threshold_sensitivity.json"))["fold_mean"]
thr_keys = ["abs_0.1", "abs_0.15", "abs_0.2", "abs_0.25", "abs_0.3"]
thr_val = [0.1, 0.15, 0.2, 0.25, 0.3]
fig, (a0, a1) = plt.subplots(1, 2, figsize=(12, 4.4), gridspec_kw=dict(wspace=0.25))
for k, lab, col, mk in (("O1", "O1 线性", ROLE["direct"], "o"), ("O2", "O2 logit", ROLE["o2"], "D"), ("OSI-null", "OSI-null 单调映射", ROLE["null"], "s")):
    a0.plot(thr_val, [S[t][k]["iou"] for t in thr_keys], color=col, marker=mk, lw=2, ms=7, label=lab)
tf_ = [S[t]["OSI-null"]["true_frac_med"] for t in thr_keys]
for x, y, f_ in zip(thr_val, [S[t]["O1"]["iou"] for t in thr_keys], tf_):
    a0.annotate(f"真值面积 {100 * f_:.0f}%", (x, y), xytext=(0, 9), textcoords="offset points", ha="center", fontsize=8, color=INK["secondary"])
a0.axvspan(0.09, 0.11, color=INK["grid"], zorder=0)
a0.text(0.115, 0.05, "预注册阈值 0.1\n恰在 OSI 分布中位\n（增益最小的一档）", ha="left", fontsize=8.5, color=INK["secondary"])
a0.set_xlabel("OSI 掩膜阈值")
a0.set_ylabel("掩膜 IoU（三折折外均值，单 seed）")
a0.set_title("(a) 阈值越高，直接回归对单调映射的增益越大；≥0.25 时映射到不了", loc="left", **BOLD)
a0.legend(loc="upper right", fontsize=9)
a0.set_ylim(-0.03, 0.75)
for k, lab, col, mk in (("O1", "O1 集成", ROLE["direct"], "o"), ("O2", "O2 集成", ROLE["o2"], "D"), ("M1_osi", "M1 三头集成 OSI 头", ROLE["m1"], "v"), ("osi_null", "OSI-null", ROLE["null"], "s")):
    a1.plot(thr_val, [OSI_THR[t][k] for t in thr_val], color=col, marker=mk, lw=2, ms=7, label=lab)
a1.set_xlabel("OSI 掩膜阈值")
a1.set_ylabel("掩膜 IoU（test34，三 seed 集成）")
a1.set_title("(b) test34 三 seed 集成上同样的形态", loc="left", **BOLD)
a1.legend(loc="upper right", fontsize=9)
a1.set_ylim(-0.03, 0.75)
save(fig, "fig05_osi_threshold_sensitivity")


# ------------------------------------------------------------------ 图 6：test34 最终读数（逐 seed + 集成）
def dot_col(ax, x, seeds, ensv, col, mk="o", label=None, seed_label=False):
    if seeds:
        ax.scatter([x] * len(seeds), seeds, s=34, facecolor="none", edgecolor=col, lw=1.4, zorder=3, label=("单 seed" if seed_label else None))
    ax.scatter([x], [ensv], s=95, color=col, marker=mk, zorder=4, label=label)


fig, axs = plt.subplots(1, 5, figsize=(19, 4.6), gridspec_kw=dict(wspace=0.38))
ax = axs[0]
dot_col(ax, 0, SEED["X5D3_ln"], SUM["X5D3"]["r2cb_ln"], ROLE["peak"], "s")
dot_col(ax, 1, SEED["A1_ln"], SUM["A1"]["r2cb_ln"], ROLE["direct"], "o")
dot_col(ax, 2, SEED["M1_tawss_ln"], SUM["M1_tawss"]["r2cb_ln"], ROLE["m1"], "D")
ax.set_xticks([0, 1, 2])
ax.set_xticklabels(["峰值帧 WSS\n（已部署 X5D）", "TAWSS\nA1 单头", "TAWSS\nM1 三头"])
ax.set_ylim(0.85, 0.90)
ax.set_ylabel("归一化（ln）R²_cb")
ax.set_title("(a) TAWSS 与峰值帧一样好学", loc="left", **BOLD)
ax = axs[1]
dot_col(ax, 0, None, SUM["tawss_null"]["r2cb_pa"], ROLE["null"], "s")
dot_col(ax, 1, None, SUM["tawss_ratio"]["r2cb_pa"], ROLE["ratio"], "^")
dot_col(ax, 2, SEED["A1"], SUM["A1"]["r2cb_pa"], ROLE["direct"], "o")
dot_col(ax, 3, SEED["M1_tawss"], SUM["M1_tawss"]["r2cb_pa"], ROLE["m1"], "D")
ax.set_xticks(range(4))
ax.set_xticklabels(["TAWSS-\nnull", "TAWSS-\nratio", "A1 单头", "M1 三头"])
ax.set_ylim(0.70, 0.78)
ax.set_ylabel("TAWSS Pa 空间 R²_cb")
ax.set_title("(b) Pa 口径被尾部主导，基线已不差", loc="left", **BOLD)
ax = axs[2]
dot_col(ax, 0, None, SUM["osi_null"]["r2cb"], ROLE["null"], "s")
dot_col(ax, 1, SEED["O1"], SUM["O1"]["r2cb"], ROLE["direct"], "o")
dot_col(ax, 2, SEED["O2"], SUM["O2"]["r2cb"], ROLE["o2"], "D")
dot_col(ax, 3, SEED["M1_osi"], SUM["M1_osi"]["r2cb"], ROLE["m1"], "v")
ax.set_xticks(range(4))
ax.set_xticklabels(["OSI-null", "O1 线性", "O2 logit", "M1 三头"])
ax.set_ylim(0.35, 0.62)
ax.set_ylabel("OSI R²_cb（线性）")
ax.set_title("(c) OSI：集成增益大，单模型方差大", loc="left", **BOLD)
ax = axs[3]
for j, (k, col, mk) in enumerate((("osi_null", ROLE["null"], "s"), ("O1", ROLE["direct"], "o"), ("O2", ROLE["o2"], "D"), ("M1_osi", ROLE["m1"], "v"))):
    ax.scatter([j], [SUM[k]["iou_01"]], s=95, color=col, marker=mk, zorder=4)
    ax.scatter([j], [SUM[k]["iou_03"]], s=95, facecolor="none", edgecolor=col, marker=mk, lw=1.8, zorder=4)
ax.set_xticks(range(4))
ax.set_xticklabels(["OSI-null", "O1", "O2", "M1 三头"])
ax.set_ylim(-0.03, 0.78)
ax.set_ylabel("掩膜 IoU（逐例均值）")
ax.set_title("(d) OSI > 0.1（实心）/ > 0.3（空心）", loc="left", **BOLD)
ax.legend(handles=[Line2D([], [], marker="o", color=INK["secondary"], ls="", label="OSI > 0.1"),
                   Line2D([], [], marker="o", markerfacecolor="none", color=INK["secondary"], ls="", label="OSI > 0.3")], loc="center right", fontsize=9)
ax = axs[4]
for j, (k, col, mk) in enumerate((("stag_null", ROLE["null"], "s"), ("stag_A1xO1", ROLE["direct"], "o"), ("stag_A1xO2", ROLE["o2"], "D"), ("stag_M1", ROLE["m1"], "v"))):
    ax.scatter([j], [SUM[k]["iou"]], s=95, color=col, marker=mk, zorder=4)
ax.set_xticks(range(4))
ax.set_xticklabels(["基线组合", "A1 × O1", "A1 × O2", "M1 三头"])
ax.set_ylim(0.38, 0.60)
ax.set_ylabel("滞留区 IoU（TAWSS<0.4 ∧ OSI>0.1）")
ax.set_title("(e) 滞留区：三头集成最高", loc="left", **BOLD)
fig.legend(handles=[Line2D([], [], marker="o", markerfacecolor="none", color=INK["secondary"], ls="", label="空心小点 = 单 seed（1234 / 7 / 2025）"),
                    Line2D([], [], marker="o", color=INK["secondary"], ls="", label="实心大点 = 三 seed Pa 均值集成 / 基线")], loc="lower center", ncol=2, bbox_to_anchor=(0.5, -0.06))
fig.suptitle("阶段 3 确认：train136 训练、锁定 test34 读一次（34 例，不做任何选择）", x=0.01, ha="left", fontsize=12, **BOLD)
save(fig, "fig06_test34_summary")

# ------------------------------------------------------------------ 图 7：逐例散点
fig, (a0, a1) = plt.subplots(1, 2, figsize=(11.5, 5), gridspec_kw=dict(wspace=0.28))
for co in COHORT_ORDER:
    ii = [i for i, d in enumerate(CASES) if d["cohort"] == co]
    a0.scatter([PC["X5D3"][i]["r2"] for i in ii], [PC["A1"][i]["r2"] for i in ii], s=55, color=COH[co], label=COHORT_NAME[co], zorder=3, edgecolor="white", lw=0.6)
    a1.scatter([PC["A1"][i]["r2"] for i in ii], [PC["M1_osi"][i]["r2"] for i in ii], s=55, color=COH[co], label=COHORT_NAME[co], zorder=3, edgecolor="white", lw=0.6)
a0.plot([0, 1], [0, 1], color=INK["axis"], lw=1)
a0.set_xlim(0.3, 1)
a0.set_ylim(0.3, 1)
a0.set_xlabel("峰值帧 WSS 逐例 R²（已部署 X5D 三 seed 集成）")
a0.set_ylabel("TAWSS 逐例 R²（A1 三 seed 集成）")
a0.set_title(f"(a) 逐例看：TAWSS 与峰值帧同难度（中位 {agg(PC['A1'], 'r2', 'med'):.2f} 对 {agg(PC['X5D3'], 'r2', 'med'):.2f}）", loc="left", **BOLD)
a0.legend(loc="lower right", fontsize=9)
a0.set_aspect("equal")
a1.plot([0, 1], [0, 1], color=INK["axis"], lw=1)
a1.set_xlim(0.0, 1)
a1.set_ylim(0.0, 1)
a1.set_xlabel("TAWSS 逐例 R²（A1 集成）")
a1.set_ylabel("OSI 逐例 R²（M1 三头集成）")
a1.set_title(f"(b) OSI 逐例更难（中位 {agg(PC['M1_osi'], 'r2', 'med'):.2f}，p10 {agg(PC['M1_osi'], 'r2', 'p10'):.2f}）", loc="left", **BOLD)
a1.set_aspect("equal")
worst = sorted(range(N_CASES), key=lambda i: PC["M1_osi"][i]["r2"])[:2]
for i in worst:
    a1.annotate(CASES[i]["cid"].split("/")[-1], (PC["A1"][i]["r2"], PC["M1_osi"][i]["r2"]), xytext=(6, -10), textcoords="offset points", fontsize=8, color=INK["secondary"])
save(fig, "fig07_percase_scatter")

# ------------------------------------------------------------------ 图 8：CFD 对预测 汇总散点（hexbin）
rng = np.random.default_rng(7)
sub = [rng.choice(d["n"], min(4000, d["n"]), replace=False) for d in CASES]


def pooled(key):
    return np.concatenate([d[key][s] for d, s in zip(CASES, sub)])


T_ta = pooled("tawss")
T_os = pooled("osi")
fig, axs = plt.subplots(2, 3, figsize=(14, 9), gridspec_kw=dict(wspace=0.22, hspace=0.3))
for j, (k, title) in enumerate((("A1", "A1 单头三 seed 集成"), ("M1_tawss", "M1 三头集成 TAWSS 头"), ("tawss_null", "TAWSS-null 基线（峰值×波形）"))):
    ax = axs[0, j]
    P = pooled(k)
    lo, hi = 0.05, np.quantile(T_ta, 0.999)
    ax.hexbin(np.clip(T_ta, lo, None), np.clip(P, lo, None), gridsize=55, xscale="log", yscale="log", cmap=CM_COUNT, norm=LogNorm(), mincnt=1, linewidths=0.2)
    ax.plot([lo, hi * 2], [lo, hi * 2], color=INK["primary"], lw=1)
    ax.set_xlim(lo, hi * 2)
    ax.set_ylim(lo, hi * 2)
    ax.set_xlabel("CFD TAWSS (Pa)")
    ax.set_ylabel("预测 TAWSS (Pa)")
    ax.set_title(f"TAWSS · {title}", loc="left", **BOLD)
    ax.text(0.03, 0.97, f"R²_cb Pa {SUM[k]['r2cb_pa']:.3f}｜ln {SUM[k]['r2cb_ln']:.3f}\nCCC(ln) 中位 {SUM[k]['ccc_med']:.3f}｜MAE {SUM[k]['mae']:.2f} Pa", transform=ax.transAxes, va="top", fontsize=9)
for j, (k, title) in enumerate((("O2", "O2 单头三 seed 集成（logit）"), ("M1_osi", "M1 三头集成 OSI 头"), ("osi_null", "OSI-null 基线（单调映射）"))):
    ax = axs[1, j]
    P = pooled(k)
    ax.hexbin(T_os, P, gridsize=55, cmap=CM_COUNT, norm=LogNorm(), mincnt=1, linewidths=0.2, extent=(0, 0.5, 0, 0.5))
    ax.plot([0, 0.5], [0, 0.5], color=INK["primary"], lw=1)
    ax.set_xlim(0, 0.5)
    ax.set_ylim(0, 0.5)
    ax.set_xlabel("CFD OSI")
    ax.set_ylabel("预测 OSI")
    ax.set_title(f"OSI · {title}", loc="left", **BOLD)
    ax.text(0.03, 0.97, f"R²_cb {SUM[k]['r2cb']:.3f}｜CCC 中位 {SUM[k]['ccc_med']:.3f}\nMAE {SUM[k]['mae']:.3f}｜>0.3 IoU {SUM[k]['iou_03']:.2f}", transform=ax.transAxes, va="top", fontsize=9)
fig.suptitle("CFD 真值 对 预测：test34 全部 34 例逐点汇总（每例抽 4000 点，颜色 = 点密度）", x=0.01, ha="left", fontsize=12, **BOLD)
save(fig, "fig08_cfd_vs_pred_hexbin")

# ------------------------------------------------------------------ 图 9：病例级 Bland–Altman
fig, axs = plt.subplots(1, 3, figsize=(15, 4.4), gridspec_kw=dict(wspace=0.3))
for ax, k, ttl in ((axs[0], "A1", "A1 单头集成"), (axs[1], "M1_tawss", "M1 三头集成")):
    rows_ = PC[k]
    for co in COHORT_ORDER:
        rr = [r for r in rows_ if r["cohort"] == co]
        ax.scatter([0.5 * (r["mean_true"] + r["mean_pred"]) for r in rr], [100 * r["case_mean_rel"] for r in rr], s=50, color=COH[co], label=COHORT_NAME[co], zorder=3, edgecolor="white", lw=0.6)
    b, lo, hi = SUM[k]["ba"]
    ax.axhline(100 * b, color=INK["primary"], lw=1.4)
    ax.axhline(100 * lo, color=INK["primary"], lw=1, ls="--")
    ax.axhline(100 * hi, color=INK["primary"], lw=1, ls="--")
    ax.text(0.99, 0.97, f"偏差 {100 * b:+.1f}%\n95% 一致性界 [{100 * lo:+.0f}%, {100 * hi:+.0f}%]", transform=ax.transAxes, ha="right", va="top", fontsize=9,
            bbox=dict(facecolor=INK["surface"], edgecolor="none", alpha=0.9, pad=2))
    ax.set_xlabel("病例平均 TAWSS (Pa)，CFD 与预测的均值")
    ax.set_ylabel("病例均值相对差 (预测 − CFD)/CFD (%)")
    ax.set_title(f"TAWSS 病例均值 · {ttl}", loc="left", **BOLD)
    ax.set_xscale("log")
axs[0].legend(loc="lower left", fontsize=9)
ax = axs[2]
rows_ = PC["M1_osi"]
for co in COHORT_ORDER:
    rr = [r for r in rows_ if r["cohort"] == co]
    ax.scatter([100 * r["true_frac_0.1"] for r in rr], [100 * r["frac_err_0.1"] for r in rr], s=50, color=COH[co], label=COHORT_NAME[co], zorder=3, edgecolor="white", lw=0.6)
b, lo, hi = SUM["M1_osi"]["ba_frac01"]
ax.axhline(100 * b, color=INK["primary"], lw=1.4)
ax.axhline(100 * lo, color=INK["primary"], lw=1, ls="--")
ax.axhline(100 * hi, color=INK["primary"], lw=1, ls="--")
ax.text(0.99, 0.97, f"偏差 {100 * b:+.1f} pp\n95% 一致性界 [{100 * lo:+.0f}, {100 * hi:+.0f}] pp", transform=ax.transAxes, ha="right", va="top", fontsize=9,
        bbox=dict(facecolor=INK["surface"], edgecolor="none", alpha=0.9, pad=2))
ax.set_xlabel("CFD 的 OSI > 0.1 面积份额 (%)")
ax.set_ylabel("面积份额差 (预测 − CFD)，百分点")
ax.set_title("OSI > 0.1 面积份额 · M1 三头集成", loc="left", **BOLD)
fig.suptitle("病例级一致性（Bland–Altman，test34）：报告里给「病例均值 / 面积份额」这类汇总数时的误差带", x=0.01, ha="left", fontsize=12, **BOLD)
save(fig, "fig09_bland_altman_case")

# ------------------------------------------------------------------ 图 10：按血管段的误差
fig, (a0, a1) = plt.subplots(1, 2, figsize=(13, 4.6), gridspec_kw=dict(wspace=0.22))
segs = list(range(7))
for ax, key_list, ylabel, ttl, scale in ((a0, (("A1", ROLE["direct"], "A1 单头集成"), ("M1_tawss", ROLE["m1"], "M1 三头集成")), "段均值相对误差 (预测−CFD)/CFD (%)", "(a) TAWSS 段均值相对误差（≥200 点的段）", 100),
                                         (a1, (("O2", ROLE["o2"], "O2 单头集成"), ("M1_osi", ROLE["m1"], "M1 三头集成")), "段均值绝对误差 (预测−CFD)", "(b) OSI 段均值绝对误差", 1)):
    for i, (k, col, lab) in enumerate(key_list):
        data = [[scale * r["segments"][s] for r in PC[k] if s in r["segments"]] for s in segs]
        pos = np.arange(7) + (i - 0.5) * 0.34
        bp = ax.boxplot(data, positions=pos, widths=0.28, patch_artist=True, showfliers=False, medianprops=dict(color=INK["primary"], lw=1.4),
                        boxprops=dict(facecolor=col, alpha=0.55, edgecolor=col), whiskerprops=dict(color=col), capprops=dict(color=col))
        for s_, dd in zip(pos, data):
            ax.scatter(np.full(len(dd), s_) + rng.normal(0, 0.03, len(dd)), dd, s=8, color=col, alpha=0.6, zorder=3)
        ax.plot([], [], color=col, lw=6, alpha=0.55, label=lab)
    ax.axhline(0, color=INK["axis"], lw=1)
    ax.set_xticks(range(7))
    ax.set_xticklabels([SEM[s] for s in segs])
    ax.set_ylabel(ylabel)
    ax.set_title(ttl, loc="left", **BOLD)
    ax.legend(loc="upper left", fontsize=9)
a0.set_ylim(-60, 80)
a1.set_ylim(-0.15, 0.15)
fig.suptitle("报告按血管段给数的粒度：34 例 test34 的分段误差（箱 = 四分位，点 = 病例）", x=0.01, ha="left", fontsize=12, **BOLD)
save(fig, "fig10_segment_error")

# ------------------------------------------------------------------ 图 11–13：病例壁面图（CFD 对 预测）
order = sorted(range(N_CASES), key=lambda i: PC["A1"][i]["ccc"])
rank_pct = {i: 100.0 * order.index(i) / (N_CASES - 1) for i in range(N_CASES)}
pick_idx, pick_lab = [], []
for co in COHORT_ORDER:  # 每个队列取 A1 CCC 的中位病例，展示三种解剖
    ii = [i for i in range(N_CASES) if CASES[i]["cohort"] == co]
    med = float(np.median([PC["A1"][i]["ccc"] for i in ii]))
    i_pick = min(ii, key=lambda i: abs(PC["A1"][i]["ccc"] - med))
    pick_idx.append(i_pick)
    pick_lab.append(f"{co} 队列中位例（34 例中第 {rank_pct[i_pick]:.0f} 百分位）")


def log_ticks(cb, lo, hi):
    """对数色标用 0.2 / 0.5 / 1 / 2 / 5 / 10 这类整数刻度，不用科学计数。"""
    ticks = [t for t in (0.05, 0.1, 0.2, 0.5, 1, 2, 5, 10, 20, 50) if lo <= t <= hi]
    if len(ticks) >= 2:
        cb.set_ticks(ticks)
        cb.set_ticklabels([f"{t:g}" for t in ticks])
        cb.ax.minorticks_off()


def draw_pts(ax, d, val, cmap, norm, title, sub_title=None):
    pos = d["pos"]
    o = np.argsort(pos[:, 1])  # 远的先画
    s = float(np.clip(55000 / d["n"], 0.45, 6.0))
    sc = ax.scatter(pos[o, 0], pos[o, 2], c=val[o], s=s, cmap=cmap, norm=norm, linewidths=0, rasterized=True)
    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_title(title, fontsize=9.5, loc="left", **BOLD)
    if sub_title:
        ax.text(0.0, -0.01, sub_title, transform=ax.transAxes, va="top", fontsize=8.5, color=INK["secondary"])
    return sc


def case_title(d):
    return f"{d['cohort']} · {'/'.join(d['cid'].split('/')[1:]).replace('ruputer/', '').replace('unruputer/', '')}"


fig, axs = plt.subplots(3, 5, figsize=(18, 16.5), gridspec_kw=dict(wspace=0.2, hspace=0.34))
for r, (i, lab) in enumerate(zip(pick_idx, pick_lab)):
    d = CASES[i]
    vlo = float(max(np.quantile(d["tawss"], 0.05), 0.05))
    vhi = float(np.quantile(d["tawss"], 0.95))
    norm = LogNorm(vmin=vlo, vmax=vhi)
    sc = draw_pts(axs[r, 0], d, d["tawss"], CM_TAWSS, norm, f"CFD TAWSS\n{case_title(d)}", f"{lab}\nn = {d['n']:,} 点\n色标 {vlo:.2f}–{vhi:.1f} Pa（本例 p5–p95，对数）")
    for j, (k, ttl) in enumerate((("A1", "A1 单头集成"), ("M1_tawss", "M1 三头集成"), ("tawss_null", "基线 峰值×波形"))):
        m = PC[k][i]
        draw_pts(axs[r, j + 1], d, d[k], CM_TAWSS, norm, ttl, f"逐例 R² {m['r2']:.2f}｜CCC(ln) {m['ccc']:.2f}\n低 TAWSS IoU {m['low_iou']:.2f}｜病例均值差 {100 * m['case_mean_rel']:+.0f}%")
    err = np.log(np.clip(d["A1"], 0.05, None) / np.clip(d["tawss"], 0.05, None))
    sc2 = draw_pts(axs[r, 4], d, err, CM_DIV, TwoSlopeNorm(vmin=-1.0, vcenter=0, vmax=1.0), "误差 ln(A1 / CFD)", "蓝 = 低估，红 = 高估，灰 = 一致\n±1 ≈ ×0.37 / ×2.7")
    cb = fig.colorbar(sc, ax=axs[r, :4].tolist(), fraction=0.012, pad=0.01, extend="both")
    cb.set_label("TAWSS (Pa)")
    log_ticks(cb, vlo, vhi)
    cb2 = fig.colorbar(sc2, ax=[axs[r, 4]], fraction=0.05, pad=0.02)
    cb2.set_label("ln 比值")
fig.suptitle("TAWSS 壁面分布：CFD 真值 对 直接回归预测（test34 每个队列取 A1 CCC 中位病例；正视投影，主动脉朝上）", x=0.01, ha="left", fontsize=12, **BOLD)
save(fig, "fig11_case_maps_tawss")

fig, axs = plt.subplots(3, 5, figsize=(18, 16.5), gridspec_kw=dict(wspace=0.2, hspace=0.34))
for r, (i, lab) in enumerate(zip(pick_idx, pick_lab)):
    d = CASES[i]
    norm = plt.Normalize(0, 0.4)
    sc = draw_pts(axs[r, 0], d, d["osi"], CM_OSI, norm, f"CFD OSI\n{case_title(d)}", f"{lab}\n按 TAWSS 选例\nOSI>0.1 面积 {100 * np.mean(d['osi'] > 0.1):.0f}%，>0.3 面积 {100 * np.mean(d['osi'] > 0.3):.0f}%")
    for j, (k, ttl) in enumerate((("O2", "O2 单头集成（logit）"), ("M1_osi", "M1 三头集成"), ("osi_null", "基线 单调映射"))):
        m = PC[k][i]
        draw_pts(axs[r, j + 1], d, d[k], CM_OSI, norm, ttl, f"逐例 R² {m['r2']:.2f}｜CCC {m['ccc']:.2f}\n>0.1 IoU {m['iou_0.1']:.2f}｜>0.3 IoU {m['iou_0.3']:.2f}")
    err = d["M1_osi"] - d["osi"]
    sc2 = draw_pts(axs[r, 4], d, err, CM_DIV, TwoSlopeNorm(vmin=-0.2, vcenter=0, vmax=0.2), "误差 M1 − CFD", "蓝 = 低估，红 = 高估")
    cb = fig.colorbar(sc, ax=axs[r, :4].tolist(), fraction=0.012, pad=0.01, extend="max")
    cb.set_label("OSI")
    cb2 = fig.colorbar(sc2, ax=[axs[r, 4]], fraction=0.05, pad=0.02)
    cb2.set_label("ΔOSI")
fig.suptitle("OSI 壁面分布：CFD 真值 对 预测（同 3 例；单调映射基线到不了高 OSI 区，直接回归能）", x=0.01, ha="left", fontsize=12, **BOLD)
save(fig, "fig12_case_maps_osi")

MASK_CM = ListedColormap([INK["grid"], C["aqua"], C["orange"], C["blue"]])
MASK_NORM = BoundaryNorm([-0.5, 0.5, 1.5, 2.5, 3.5], 4)
fig, axs = plt.subplots(3, 3, figsize=(11.5, 16), gridspec_kw=dict(wspace=0.12, hspace=0.32))
for r, (i, lab) in enumerate(zip(pick_idx, pick_lab)):
    d = CASES[i]
    for j, (tm, pm, ttl) in enumerate(((d["tawss"] < 0.4, d["M1_tawss"] < 0.4, "低 TAWSS < 0.4 Pa"),
                                      (d["osi"] > 0.3, d["M1_osi"] > 0.3, "高 OSI > 0.3"),
                                      ((d["tawss"] < 0.4) & (d["osi"] > 0.1), (d["M1_tawss"] < 0.4) & (d["M1_osi"] > 0.1), "滞留区 TAWSS<0.4 ∧ OSI>0.1"))):
        code = np.where(tm & pm, 3, np.where(pm & ~tm, 2, np.where(tm & ~pm, 1, 0)))
        draw_pts(axs[r, j], d, code, MASK_CM, MASK_NORM, f"{case_title(d)}\n{ttl}" if j == 0 else ttl,
                 f"IoU {iou(tm, pm):.2f}｜Dice {M.dice_masks(tm, pm):.2f}\nCFD 面积 {100 * tm.mean():.0f}% / 预测 {100 * pm.mean():.0f}%")
fig.legend(handles=[Patch(color=C["blue"], label="CFD 与预测一致命中"), Patch(color=C["orange"], label="仅预测（假阳）"), Patch(color=C["aqua"], label="仅 CFD（漏检）"), Patch(color=INK["grid"], label="两者皆无")],
           loc="lower center", ncol=4, bbox_to_anchor=(0.5, 0.01))
fig.suptitle("临床掩膜的一致性（M1 三头集成，同 3 例）：报告交付的是「区域」而不是逐点数值", x=0.01, ha="left", fontsize=12, **BOLD)
save(fig, "fig13_case_maps_masks")

# ------------------------------------------------------------------ 图 14：三头 M1 对单头（逐 seed Δ 与集成 Δ）
fig, (a0, a1) = plt.subplots(1, 2, figsize=(12, 4.6), gridspec_kw=dict(wspace=0.28))
chan = ["峰值 WSS\n（对同 seed X5D_v51）", "TAWSS\n（对同 seed A1）", "OSI\n（对同 seed O2）"]
d_norm = [np.array(SEED["M1_wss_ln"]) - np.array(SEED["X5D3_ln"]), np.array(SEED["M1_tawss_ln"]) - np.array(SEED["A1_ln"]), np.array(SEED["M1_osi_logit"]) - np.array(SEED["O2_logit"])]
d_pa = [np.array(SEED["M1_wss"]) - np.array(SEED["X5D3"]), np.array(SEED["M1_tawss"]) - np.array(SEED["A1"]), np.array(SEED["M1_osi"]) - np.array(SEED["O2"])]
e_norm = [SUM["M1_wss"]["r2cb_ln"] - SUM["X5D3"]["r2cb_ln"], SUM["M1_tawss"]["r2cb_ln"] - SUM["A1"]["r2cb_ln"], SUM["M1_osi"]["r2cb_logit"] - SUM["O2"]["r2cb_logit"]]
e_pa = [SUM["M1_wss"]["r2cb_pa"] - SUM["X5D3"]["r2cb_pa"], SUM["M1_tawss"]["r2cb_pa"] - SUM["A1"]["r2cb_pa"], SUM["M1_osi"]["r2cb"] - SUM["O2"]["r2cb"]]
for ax, dd, ee, ttl, ylab in ((a0, d_norm, e_norm, "(a) 归一化空间 ΔR²_cb（预注册门：峰值 ≥ −0.02，两头 ≥ −0.01）", "M1 − 单头（归一化 R²_cb）"),
                              (a1, d_pa, e_pa, "(b) 物理空间 ΔR²_cb（Pa / OSI 线性）", "M1 − 单头（物理 R²_cb）")):
    ax.axhspan(-0.02, 0, color=INK["grid"], zorder=0)
    ax.axhline(0, color=INK["primary"], lw=1)
    for j in range(3):
        ax.scatter([j] * 3, dd[j], s=34, facecolor="none", edgecolor=ROLE["m1"], lw=1.4, zorder=3)
        ax.scatter([j], [ee[j]], s=110, color=ROLE["m1"], marker="D", zorder=4)
        ax.annotate(f"{ee[j]:+.3f}", (j, ee[j]), xytext=(10, 0), textcoords="offset points", va="center", fontsize=9)
    ax.set_xticks(range(3))
    ax.set_xticklabels(chan)
    ax.set_ylabel(ylab)
    ax.set_title(ttl, loc="left", **BOLD)
    ax.set_xlim(-0.5, 2.7)
a0.set_ylim(-0.03, 0.03)
a1.set_ylim(-0.06, 0.04)
a0.text(2.65, -0.019, "−0.02 门", ha="right", fontsize=8.5, color=INK["secondary"])
fig.legend(handles=[Line2D([], [], marker="o", markerfacecolor="none", color=ROLE["m1"], ls="", label="单 seed 差（1234 / 7 / 2025）"),
                    Line2D([], [], marker="D", color=ROLE["m1"], ls="", label="三 seed 集成差")], loc="lower center", ncol=2, bbox_to_anchor=(0.5, -0.05))
fig.suptitle("三头单模型 M1（峰值 + TAWSS + OSI 一次前向）对三个单头模型：test34", x=0.01, ha="left", fontsize=12, **BOLD)
save(fig, "fig14_m1_multihead_delta")

# ------------------------------------------------------------------ 图 15：文献对比
LIT = [
    # (标签, 数据性质, TAWSS relL2, OSI relL2, TAWSS MAE Pa, OSI MAE, 说明)
    ("本工作 test34 三 seed 集成（A1 / O2 单头）", "real", SUM["A1"]["rel_l2"], SUM["O2"]["rel_l2"], SUM["A1"]["mae"], SUM["O2"]["mae"], "170 例真实主-髂动脉，锁定 34 例"),
    ("本工作 test34 三 seed 集成（M1 三头）", "real", SUM["M1_tawss"]["rel_l2"], SUM["M1_osi"]["rel_l2"], SUM["M1_tawss"]["mae"], SUM["M1_osi"]["mae"], "同上，单模型三输出"),
    ("本工作 cv3 折外单 seed（A1 / O1）", "real", 0.35, 0.48, 0.39, 0.062, "患者分组三折，136 例折外"),
    ("Rygiel 2025 AAA 十折（LaB-GATr）", "real", 0.39, 0.38, 0.136, 0.064, "100 例真实 AAA × 4 入口流量，入口 Q_max 作输入"),
    ("Rygiel 2025 AAA 外部集", "real", 0.333, 0.298, 0.148, 0.048, "29 人 118 次扫描"),
    ("Sheng 2026 颅内动脉瘤（合成）", "synthetic", 0.135, 0.68, np.nan, np.nan, "14000 稳态 + 808 脉动合成几何"),
]
fig, (a0, a1, a2) = plt.subplots(1, 3, figsize=(16, 4.8), gridspec_kw=dict(width_ratios=[1.2, 1.2, 1], wspace=0.55))
ylab = [r[0] for r in LIT]  # 与 yy 同序：第一行在最上
yy = np.arange(len(LIT))[::-1]
for ax, col_i, ttl, xlab in ((a0, 2, "(a) TAWSS 逐例相对 L2（越小越好）", "相对 L2 = ‖预测 − CFD‖ / ‖CFD‖（L2 范数，逐例均值）"), (a1, 3, "(b) OSI 逐例相对 L2", "相对 L2")):
    for y, r in zip(yy, LIT):
        col = ROLE["direct"] if r[0].startswith("本工作") else INK["secondary"]
        ax.scatter([r[col_i]], [y], s=110, color=col if r[1] == "real" else "none", edgecolor=col, lw=1.8, zorder=4)
        ax.annotate(f"{r[col_i]:.2f}", (r[col_i], y), xytext=(9, 0), textcoords="offset points", va="center", fontsize=9)
    ax.set_yticks(yy)
    ax.set_yticklabels(ylab, fontsize=9)
    ax.set_xlim(0, 0.8)
    ax.set_xlabel(xlab)
    ax.set_title(ttl, loc="left", **BOLD)
a1.set_yticklabels([])
for y, r in zip(yy, LIT):
    if np.isnan(r[4]):
        continue
    col = ROLE["direct"] if r[0].startswith("本工作") else INK["secondary"]
    a2.scatter([r[4]], [y], s=110, color=col, zorder=4)
    a2.annotate(f"{r[4]:.2f} Pa", (r[4], y), xytext=(9, 0), textcoords="offset points", va="center", fontsize=9)
a2.set_yticks(yy)
a2.set_yticklabels([])
a2.set_xlim(0, 0.6)
a2.set_xlabel("TAWSS MAE (Pa)")
a2.set_title("(c) TAWSS MAE：受幅值影响不可直接比\n（我们均值 1.28 Pa，含髂动脉射流）", loc="left", **BOLD)
fig.legend(handles=[Line2D([], [], marker="o", color=ROLE["direct"], ls="", label="本工作"), Line2D([], [], marker="o", color=INK["secondary"], ls="", label="文献（真实队列）"),
                    Line2D([], [], marker="o", markerfacecolor="none", color=INK["secondary"], ls="", label="文献（合成几何）")], loc="lower center", ncol=3, bbox_to_anchor=(0.5, -0.06))
fig.suptitle("与文献的可比口径：真实队列上 TAWSS 持平/略好，OSI 落后（OSI 是全网最难的周期量）", x=0.01, ha="left", fontsize=12, **BOLD)
save(fig, "fig15_literature_comparison")

# ------------------------------------------------------------------ 图 16：部署——计时与真实 STL 三场图
SUMJ = json.load(open(DEPLOY_JOB / "summary.json"))
tim = SUMJ["timing_s"]
with np.load(DEPLOY_JOB / "field.npz", allow_pickle=False) as z:
    pts = z["pts"].astype(float)
    dep = dict(wss=z["wss_pa"].astype(float), tawss=z["tawss_pa"].astype(float), osi=z["osi"].astype(float))
cyc = SUMJ["cycle"]
fig = plt.figure(figsize=(16, 6.2))
gs = fig.add_gridspec(1, 4, width_ratios=[1.1, 1, 1, 1], wspace=0.12)
ax = fig.add_subplot(gs[0])
keys_t = [("centerline", "中心线"), ("smooth_resample", "平滑重采样"), ("features", "27 维特征"), ("inference_5_models", "三模型推理"), ("morphology", "瘤体形态"), ("export", "导出"), ("ingest", "读入")]
vals_t = [tim.get(k, 0) for k, _ in keys_t]
ax.barh([n for _, n in keys_t][::-1], vals_t[::-1], color=ROLE["m1"], height=0.6)
for y, v in enumerate(vals_t[::-1]):
    ax.text(v + 0.3, y, f"{v:.1f} s", va="center", fontsize=9)
ax.set_xlabel("秒")
ax.set_title(f"(a) 端到端 {tim['total']:.1f} s（{SUMJ.get('gpu') or SUMJ.get('device', 'GPU')}）\n三模型推理只占 {tim['inference_5_models']:.1f} s", loc="left", **BOLD)
ax.grid(axis="y", visible=False)
# 投影：PCA 到最大两个主轴
pc_ = pts - pts.mean(0)
u, s_, vt = np.linalg.svd(pc_, full_matrices=False)
proj = pc_ @ vt.T
o = np.argsort(proj[:, 2])
q = lambda a, lo: (float(max(np.quantile(a, 0.05), lo)), float(np.quantile(a, 0.95)))
for j, (k, ttl, cmap, norm, unit, ext) in enumerate((("wss", "峰值 WSS", CM_TAWSS, LogNorm(*q(dep["wss"], 0.05)), "Pa", "both"),
                                                      ("tawss", "TAWSS", CM_TAWSS, LogNorm(*q(dep["tawss"], 0.05)), "Pa", "both"),
                                                      ("osi", "OSI", CM_OSI, plt.Normalize(0, 0.4), "", "max"))):
    ax = fig.add_subplot(gs[j + 1])
    sc = ax.scatter(proj[o, 1], proj[o, 0], c=dep[k][o], s=float(np.clip(55000 / len(pts), 0.45, 6.0)), cmap=cmap, norm=norm, linewidths=0, rasterized=True)
    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_title(f"({'bcd'[j]}) {ttl}", loc="left", **BOLD)
    cb = fig.colorbar(sc, ax=ax, fraction=0.04, pad=0.02, extend=ext)
    cb.set_label(unit)
    if k != "osi":
        log_ticks(cb, norm.vmin, norm.vmax)
st = cyc.get("stagnation", {})
fig.suptitle(f"部署：LV_GUO_YOU 真实壁面 STL → 三头发布包 M1_3head_3seed_20260922 一次前向出三场（{len(pts):,} 点）\n"
             f"TAWSS < 0.4 Pa 面积 {100 * cyc['fields']['tawss']['area_frac'].get('low', np.nan):.0f}%，OSI > 0.3 面积 {100 * cyc['fields']['osi']['area_frac'].get('above_t2', np.nan):.0f}%，滞留区 {100 * st.get('area_frac', np.nan):.0f}%",
             x=0.01, y=1.02, ha="left", fontsize=11.5, **BOLD)
save(fig, "fig16_deployment")

# ------------------------------------------------------------------ 表
write_table("T1_stage1_gates_3fold",
            ["臂", "指标", "折0", "折1", "折2", "三折均值", "参照（三折均值）", "Δ", "门", "判定"],
            [["A1 TAWSS", "归一化 R²_cb", *[f4(r["norm_r2cb"]) for r in A1f], f4(np.mean([r["norm_r2cb"] for r in A1f])), f"同折峰值 {np.mean([r['base_norm_r2cb'] for r in A1f]):.4f}", f"{G['gates']['G1.1']['mean']:+.4f}", "≥ −0.02", "G1.1 PASS"],
             ["A1 TAWSS", "Pa R²_cb", *[f4(r["pa_r2cb"]) for r in A1f], f4(np.mean([r["pa_r2cb"] for r in A1f])), f"max(null {np.mean(nullPa):.4f}, ratio {np.mean(ratioPa):.4f})", f"{G['gates']['G1.2']['mean']:+.4f}", "≥ +0.05", "G1.2 FAIL"],
             ["A1 TAWSS", "ln 空间 R²_cb", *[f4(r["norm_r2cb"]) for r in A1f], f4(np.mean([r["norm_r2cb"] for r in A1f])), f"TAWSS-null ln {np.mean(nullLog):.4f}", f"{np.mean([r['norm_r2cb'] for r in A1f]) - np.mean(nullLog):+.4f}", "（未预注册）", "—"],
             ["A1 TAWSS", "低 TAWSS<0.4 IoU", *[f3(r["below_0.4_iou"]) for r in A1f], f3(np.mean([r["below_0.4_iou"] for r in A1f])), f"基线 {np.mean([C1[f'fold{k}']['TAWSS_null']['threshold_masks']['below_0.4']['iou'] for k in folds]):.3f}", f"{np.mean(G['gates']['G1.2']['low_tawss_iou_delta']):+.3f}", "不劣于基线", "附属 PASS"],
             ["O1 OSI 线性", "OSI R²_cb", *[f4(r["pa_r2cb"]) for r in O1f], f4(np.mean([r["pa_r2cb"] for r in O1f])), f"OSI-null {np.mean(osinull):.4f}", f"{np.mean([r['pa_r2cb'] for r in O1f]) - np.mean(osinull):+.4f}", "> 0", "G2.1 PASS"],
             ["O1 OSI 线性", "OSI>0.1 IoU", *[f3(r["above_0.1_iou"]) for r in O1f], f3(np.mean([r["above_0.1_iou"] for r in O1f])), f"OSI-null {np.mean(osinull_iou):.3f}", f"{np.mean(G['gates']['O1.G2.2']['delta_iou_0p1']):+.3f}", "≥ +0.05", "G2.2 FAIL"],
             ["O1 OSI 线性", "面积份额误差中位", *[f3(x) for x in G["gates"]["O1.G2.3"]["frac_err_med"]], f3(np.mean(G["gates"]["O1.G2.3"]["frac_err_med"])), "—", "—", "≤ 0.05", "G2.3 FAIL"],
             ["O2 OSI logit", "OSI R²_cb", *[f4(r["pa_r2cb"]) for r in O2f], f4(np.mean([r["pa_r2cb"] for r in O2f])), f"OSI-null {np.mean(osinull):.4f}", f"{np.mean([r['pa_r2cb'] for r in O2f]) - np.mean(osinull):+.4f}", "> 0", "G2.1 PASS"],
             ["O2 OSI logit", "OSI>0.1 IoU", *[f3(r["above_0.1_iou"]) for r in O2f], f3(np.mean([r["above_0.1_iou"] for r in O2f])), f"OSI-null {np.mean(osinull_iou):.3f}", f"{np.mean(G['gates']['O2.G2.2']['delta_iou_0p1']):+.3f}", "≥ +0.05", "G2.2 FAIL"],
             ["O2 OSI logit", "面积份额误差中位", *[f3(x) for x in G["gates"]["O2.G2.3"]["frac_err_med"]], f3(np.mean(G["gates"]["O2.G2.3"]["frac_err_med"])), "—", "—", "≤ 0.05", "G2.3 PASS"],
             ["A1 × O1", "滞留区 IoU", *[f3(x) for x in stag_o1], f3(np.mean(stag_o1)), f"基线组合 {np.mean(stag_base):.3f}", f"{np.mean(stag_o1) - np.mean(stag_base):+.3f}", "≥ +0.05", "G3 PASS"],
             ["A1 × O2", "滞留区 IoU", *[f3(x) for x in stag_o2], f3(np.mean(stag_o2)), f"基线组合 {np.mean(stag_base):.3f}", f"{np.mean(stag_o2) - np.mean(stag_base):+.3f}", "≥ +0.05", "G3 PASS"]],
            note="来源：experiments/wss_cycle_20260920/offline/gate_report_best.json 与 c1_baselines.json（作业 15335，单 seed 1234，best 检查点）。")

rows_t2 = []
for k, name in (("tawss_null", "TAWSS-null（峰值预测×训练折波形）"), ("tawss_ratio", "TAWSS-ratio（峰值预测×常数 0.317）"), ("A1", "A1 单头 三 seed 集成"), ("M1_tawss", "M1 三头 三 seed 集成（TAWSS 头）")):
    s = SUM[k]
    seeds_s = f"{np.mean(SEED[k]):.4f} ± {np.std(SEED[k], ddof=1):.4f}" if k in SEED else "—"
    rows_t2.append([name, f4(s["r2cb_ln"]), f4(s["r2cb_pa"]), seeds_s, f"{s['r2_casemean']:.3f} / {s['r2_casep10']:.3f}", f"{s['ccc_med']:.3f} / {s['ccc_p10']:.3f} / {s['ccc_min']:.3f}",
                    f"{s['mae']:.3f}", f"{s['rel_l2']:.3f}", pct(s["nmae_max"]), f"{s['spearman']:.3f}", f"{s['low_iou']:.3f}", pct(s["seg_med"]) + " / " + pct(s["seg_worst_med"]),
                    f"{100 * s['ba'][0]:+.1f}% [{100 * s['ba'][1]:+.0f}%, {100 * s['ba'][2]:+.0f}%]", f"{s['top10_ratio']:.2f}"])
for k, name in (("X5D3", "参照：峰值帧 WSS，已部署 X5D_v51 三 seed 集成"), ("M1_wss", "参照：峰值帧 WSS，M1 三头峰值通道")):
    s = SUM[k]
    rows_t2.append([name, f4(s["r2cb_ln"]), f4(s["r2cb_pa"]), f"{np.mean(SEED[k]):.4f} ± {np.std(SEED[k], ddof=1):.4f}", f"{s['r2_casemean']:.3f} / —", "—", f"{s['mae']:.3f}", "—", "—", "—", "—", "—", "—", f"{s['top10_ratio']:.2f}"])
write_table("T2_test34_tawss",
            ["模型", "归一化(ln) R²_cb", "Pa R²_cb", "单 seed Pa R²_cb 均值±sd", "逐例 R² 均值 / p10", "CCC(ln) 中位 / p10 / 最差", "MAE (Pa)", "相对 L2", "NMAE_max", "Spearman", "低 TAWSS<0.4 IoU", "分段均值相对误差 中位 / 最差段", "病例均值 BA 偏差 [95% 界]", "top10% 幅值比"],
            rows_t2, note="test34（34 例）逐点重算；R²_cb 病例等权；集成 = 三 seed（1234/7/2025）Pa 均值。峰值帧行只作参照。")

rows_t3 = []
for k, name in (("osi_null", "OSI-null（峰值预测等渗单调映射）"), ("O1", "O1 单头线性 三 seed 集成"), ("O2", "O2 单头 logit 三 seed 集成"), ("M1_osi", "M1 三头 三 seed 集成（OSI 头）")):
    s = SUM[k]
    seeds_s = f"{np.mean(SEED[k]):.4f} ± {np.std(SEED[k], ddof=1):.4f}" if k in SEED else "—"
    rows_t3.append([name, f4(s["r2cb"]), seeds_s, f"{s['r2_casemean']:.3f} / {s['r2_casep10']:.3f}", f"{s['ccc_med']:.3f} / {s['ccc_p10']:.3f}", f"{s['mae']:.4f}", f"{s['rel_l2']:.3f}", f"{s['spearman']:.3f}",
                    f"{s['iou_01']:.3f} / {s['iou_02']:.3f} / {s['iou_03']:.3f}", f"{s['dice_01_med']:.3f} / {s['dice_03_med']:.3f}", f"{s['frac_err_01_med']:.3f}",
                    f"{100 * s['ba_frac01'][0]:+.1f} pp [{100 * s['ba_frac01'][1]:+.0f}, {100 * s['ba_frac01'][2]:+.0f}]"])
write_table("T3_test34_osi",
            ["模型", "OSI R²_cb（线性）", "单 seed R²_cb 均值±sd", "逐例 R² 均值 / p10", "CCC 中位 / p10", "MAE", "相对 L2", "Spearman", "IoU >0.1 / >0.2 / >0.3", "Dice 中位 >0.1 / >0.3", ">0.1 面积份额误差中位(绝对)", ">0.1 面积份额 BA 偏差 [95% 界]"],
            rows_t3, note="test34 逐点重算；OSI ∈ [0, 0.5]，预测裁到该区间。")

rows_t4 = []
for k, name in (("stag_null", "基线组合 TAWSS-null × OSI-null"), ("stag_A1xO1", "A1 集成 × O1 集成"), ("stag_A1xO2", "A1 集成 × O2 集成"), ("stag_M1", "M1 三头集成（同一前向的两头）")):
    s = SUM[k]
    rows_t4.append([name, f3(s["iou"]), f3(s["dice_med"]), f"{100 * s['ba_frac'][0]:+.1f} pp [{100 * s['ba_frac'][1]:+.0f}, {100 * s['ba_frac'][2]:+.0f}]"])
write_table("T4_test34_stagnation", ["组合", "滞留区 IoU（逐例均值）", "Dice 中位", "面积份额 BA 偏差 [95% 界]"], rows_t4,
            note="滞留区 = TAWSS < 0.4 Pa ∧ OSI > 0.1；test34 真值面积份额中位见图 2(b)。")

rows_t5 = []
for j, (cn, ref) in enumerate((("峰值 WSS", "同 seed 已部署 X5D_v51"), ("TAWSS", "同 seed 单头 A1"), ("OSI", "同 seed 单头 O2"))):
    rows_t5.append([cn, ref, " / ".join(f"{x:+.4f}" for x in d_norm[j]), f"{e_norm[j]:+.4f}", " / ".join(f"{x:+.4f}" for x in d_pa[j]), f"{e_pa[j]:+.4f}"])
write_table("T5_m1_vs_singlehead", ["通道", "参照", "单 seed Δ 归一化 R²_cb（1234 / 7 / 2025）", "集成 Δ 归一化", "单 seed Δ 物理 R²_cb", "集成 Δ 物理"], rows_t5,
            note=f"test34。M1 = out_dim 3 三通道等权 MSE，q90 pinball 关闭。尾部分解：峰值通道剔除每例最高 10% 点后 M1 {TAIL[0.90]['M1_wss']:.4f} 对 X5D {TAIL[0.90]['X5D3']:.4f}；TAWSS 头剔 5% 后 {TAIL[0.95]['M1_tawss']:.4f} 对 A1 {TAIL[0.95]['A1']:.4f}。")

write_table("T6_tail_decomposition", ["保留 ≤ 真值分位", "A1 集成", "M1 TAWSS 头", "TAWSS-null", "TAWSS-ratio", "峰值 X5D 集成", "M1 峰值通道"],
            [[f"{q:.2f}", f4(TAIL[q]["A1"]), f4(TAIL[q]["M1_tawss"]), f4(TAIL[q]["null"]), f4(TAIL[q]["ratio"]), f4(TAIL[q]["X5D3"]), f4(TAIL[q]["M1_wss"])] for q in qs],
            note="逐例剔除真值最高分位以上的点后重算 Pa 空间 R²_cb（test34）。")

write_table("T7_literature", ["工作", "数据", "TAWSS 相对 L2", "OSI 相对 L2", "TAWSS MAE (Pa)", "OSI MAE", "备注"],
            [[r[0], "真实" if r[1] == "real" else "合成", f3(r[2]), f3(r[3]), f3(r[4]) if not np.isnan(r[4]) else "—", f3(r[5]) if not np.isnan(r[5]) else "—", r[6]] for r in LIT],
            note="文献数字取自 docs/04-论文创新与框架/论文定位_创新点与文献竞争力分析_2026-09-21.md §2.2（Rygiel 2025 arXiv 2507.22817 的 approximation disparity 即逐例相对 L2；Sheng 2026 arXiv 2601.19876 常规 rL2）。本工作 test34 行由本脚本重算。")

rows_pc = []
for i, d in enumerate(CASES):
    rows_pc.append([d["cid"], d["cohort"], d["n"], f3(PC["X5D3"][i]["r2"]), f3(PC["A1"][i]["r2"]), f3(PC["A1"][i]["ccc"]), f3(PC["M1_tawss"][i]["r2"]), f3(PC["tawss_null"][i]["r2"]),
                    f3(PC["O2"][i]["r2"]), f3(PC["M1_osi"][i]["r2"]), f3(PC["M1_osi"][i]["ccc"]), f3(PC["osi_null"][i]["r2"]), f3(PC["M1_osi"][i]["iou_0.3"]), f3(STAG["M1"][i]["iou"]),
                    pct(PC["A1"][i]["case_mean_rel"]), f"{100 * PC['M1_osi'][i]['frac_err_0.1']:+.1f}"])
write_table("T8_percase_test34", ["病例", "队列", "点数", "峰值 R²(X5D)", "TAWSS R²(A1)", "TAWSS CCC(A1)", "TAWSS R²(M1)", "TAWSS R²(null)", "OSI R²(O2)", "OSI R²(M1)", "OSI CCC(M1)", "OSI R²(null)", "OSI>0.3 IoU(M1)", "滞留区 IoU(M1)", "TAWSS 病例均值相对差(A1)", "OSI>0.1 面积份额差 pp(M1)"], rows_pc)

write_table("T9_deploy_timing", ["阶段", "秒"], [[n, f"{tim.get(k, 0):.2f}"] for k, n in keys_t] + [["合计", f"{tim['total']:.2f}"]],
            note=f"来源 {DEPLOY_JOB.name}/summary.json（LV_GUO_YOU，发布包 M1_3head_3seed_20260922，{SUMJ.get('device', '')}）。矩阵 §14.4 首次端到端记录 59.5 s（含 19 s 中心线）。")

# 汇总数字
OUT = dict(summary=SUM, seeds=SEED, tail=TAIL, osi_thr=OSI_THR, picked_cases=[CASES[i]["cid"] for i in pick_idx],
           frame_peak_r2=float(fr_r2[i_peak]), frame_min_r2=float(fr_r2[i_min]), frame_min_t=float(fr_t[i_min]),
           label_spearman_med=float(np.median(sp_all)), cohort_area={c: {k: float(np.median(v)) for k, v in dd.items()} for c, dd in coh_frac.items()},
           warnings=WARN, deploy_timing=tim)
(TAB / "summary_numbers.json").write_text(json.dumps(OUT, indent=1, ensure_ascii=False, default=float), encoding="utf-8")
print("picked cases:", OUT["picked_cases"])
print("done")

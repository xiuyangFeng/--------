"""Pre-registration diagnostics for the wave-1 matrix, computed from C1's saved test predictions only.

No training, no GPU.  For every test34 case the C1 residual r = true - pred (log_z space) is
related to the new flow-reference features so that each arm's *expected* head-room is written
down before its result is read:

1. bend/bifurcation angles (F1/F2): variance of r explained by a per-case linear fit on
   [bend_cos, bend_signed, bend_cos_up2, carina_cos, bif_plane_cos] (upper bound for X2/X3);
2. section-mean vs within-section split of r (segment x 4 mm bins): the share a section-token
   context (X11) could address vs what only a local model (X7-X9) can address;
3. per-branch mean residual vs Murray log flow share (X5) and vs the population prior (X6);
4. residual anisotropy: correlation of r between neighbours at 2-4 mm separation, split into
   axially- and circumferentially-dominant pairs (motivates the tangent-frame patch, X7).

    python -m training_wss_min.tools.prereg_wss_local_wave1
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree

from training_wss_min import config as C

ROOT = C.PROJECT_ROOT
NAME = "wss_local_wave1_20260912"
EXP = ROOT / "training_wss_min/experiments" / NAME
PRED = ROOT / "training_wss_min/runs/wss_direct_recovery_20260912/C1_s1234/eval/ckpt_best/predictions/test"
VIEW = ROOT / "data_wss_v5/views/wss_min_view_v1"
FLOW = ROOT / "data_wss_v5/views/wss_min_flowref_v1"
GEOM = ROOT / "data_wss_v5/views/wss_min_geom_v2"
BIN_MM = 4.0
PAIR_MIN_MM, PAIR_MAX_MM = 2.0, 4.0


def r2_of_fit(y: np.ndarray, X: np.ndarray) -> float:
    """Variance of y explained by an ordinary least-squares fit on X (with intercept)."""
    if len(y) < 20:
        return float("nan")
    A = np.column_stack([X, np.ones(len(y))])
    coef, *_ = np.linalg.lstsq(A, y, rcond=None)
    resid = y - A @ coef
    return float(1.0 - resid.var() / max(y.var(), 1e-12))


def case_diagnostics(unit: str) -> dict:
    with np.load(PRED / unit / "predictions.npz") as z:
        pred, true = z["pred_norm"].astype(np.float64), z["true_norm"].astype(np.float64)
    with np.load(VIEW / unit / "bundle.npz", allow_pickle=True) as b:
        pos_mm = b["wall_coords_aligned_mm"].astype(np.float64)
        normal = b["wall_normal_pca_aligned"].astype(np.float64)
        seg = b["wall_segment_id"].astype(int)
        s_local = b["wall_s_local_mm"].astype(np.float64)
        radius = b["wall_local_radius"].astype(np.float64)
        node = b["wall_node_id_cas"]
    with np.load(FLOW / unit / "features.npz") as f:
        assert np.array_equal(f["wall_node_id_cas"], node)
        flow = {k[5:]: f[k].astype(np.float64) for k in f.files if k.startswith("wall_") and f[k].dtype.kind == "f"}
    with np.load(GEOM / unit / "features.npz") as g:
        assert np.array_equal(g["wall_node_id_cas"], node)
        tn_dot = g["wall_tn_dot"].astype(np.float64)
    r = true - pred
    n = len(r)
    var_r = float(r.var())
    out = {"unit": unit, "n": int(n), "case_r2_norm": float(1 - r.var() / true.var()), "residual_var": var_r}
    # 1) angle features -> residual (per-case OLS)
    X_bend = np.column_stack([flow["bend_cos"], flow["bend_signed"], flow["bend_cos_up2"], flow["bend_cos_up5"]])
    X_bif = np.column_stack([flow["carina_cos"], flow["bif_plane_cos"], flow["bif_angle_cos"]])
    out["r2_bend_linear"] = r2_of_fit(r, X_bend)
    out["r2_bif_linear"] = r2_of_fit(r, X_bif)
    out["r2_bend_bif_linear"] = r2_of_fit(r, np.column_stack([X_bend, X_bif]))
    # interaction with the true field: does the residual grow where |bend_signed| is large?
    strong = np.abs(flow["bend_signed"]) > np.percentile(np.abs(flow["bend_signed"]), 75)
    out["residual_mse_strong_bend_over_rest"] = float((r[strong] ** 2).mean() / max((r[~strong] ** 2).mean(), 1e-12))
    # 2) section-mean vs within-section split (segment x 4 mm bins)
    cell = seg * 100000 + np.floor(s_local / BIN_MM).astype(int)
    _, inverse, counts = np.unique(cell, return_inverse=True, return_counts=True)
    cell_mean = np.bincount(inverse, weights=r) / counts
    between = cell_mean[inverse]
    within = r - between
    out["residual_share_section_mean"] = float(between.var() / max(var_r, 1e-12))
    out["residual_share_within_section"] = float(within.var() / max(var_r, 1e-12))
    out["n_sections"] = int(len(counts))
    # 3) branch-level residual vs Murray share and vs population prior
    branch_mean = {int(s): float(r[seg == s].mean()) for s in np.unique(seg)}
    branch_logq = {int(s): float(flow["log_q_branch_murray"][seg == s][0]) for s in np.unique(seg)}
    out["branch_mean_residual"] = branch_mean
    out["branch_log_q"] = branch_logq
    out["residual_share_branch_mean"] = float(np.var([branch_mean[int(s)] for s in seg]) / max(var_r, 1e-12))
    out["r2_prior_linear"] = r2_of_fit(r, flow["atlas_prior_logwss"][:, None])
    out["r2_logtau0_linear"] = r2_of_fit(r, flow["log_tau0_murray"][:, None])
    # 4) anisotropy of the residual at 2-4 mm separation
    rng = np.random.default_rng(0)
    anchors = rng.choice(n, size=min(n, 3000), replace=False)
    tree = cKDTree(pos_mm)
    groups = tree.query_ball_point(pos_mm[anchors], PAIR_MAX_MM, workers=1)
    a = np.repeat(anchors, [len(g) for g in groups])
    b = np.concatenate(groups).astype(int)
    keep = a != b
    a, b = a[keep], b[keep]
    d = pos_mm[b] - pos_mm[a]
    dist = np.linalg.norm(d, axis=1)
    same_sheet = (normal[a] * normal[b]).sum(1) > 0.5
    keep = (dist >= PAIR_MIN_MM) & same_sheet
    a, b, d = a[keep], b[keep], d[keep]
    # axial direction = centreline tangent projected on the tangent plane; tangent = unknown here, use
    # the local principal direction proxy: project d on the normal-plane and split by |d . t_hat| where
    # t_hat is the normalised local s_local gradient (finite difference over the pair itself)
    ds = np.abs(s_local[b] - s_local[a])
    axial_dom = ds > 0.6 * np.linalg.norm(d, axis=1)
    circ_dom = ds < 0.3 * np.linalg.norm(d, axis=1)
    def corr(mask):
        if mask.sum() < 200:
            return float("nan")
        return float(np.corrcoef(r[a[mask]], r[b[mask]])[0, 1])
    out["residual_corr_axial_pairs"] = corr(axial_dom)
    out["residual_corr_circ_pairs"] = corr(circ_dom)
    out["n_pairs_axial"] = int(axial_dom.sum())
    out["n_pairs_circ"] = int(circ_dom.sum())
    out["abs_tn_dot_median"] = float(np.median(np.abs(tn_dot)))
    return out


def main() -> None:
    started = time.time()
    units = sorted(str(p.parent.relative_to(PRED)) for p in PRED.glob("*/*/*/predictions.npz"))
    if len(units) != 34:
        raise RuntimeError(f"expected 34 saved C1 test predictions, found {len(units)}")
    rows = [case_diagnostics(u) for u in units]
    def med(key):
        values = np.array([row[key] for row in rows], dtype=np.float64)
        values = values[np.isfinite(values)]
        return {"median": float(np.median(values)), "p25": float(np.percentile(values, 25)),
                "p75": float(np.percentile(values, 75)), "mean": float(values.mean())}
    summary = {key: med(key) for key in (
        "case_r2_norm", "r2_bend_linear", "r2_bif_linear", "r2_bend_bif_linear", "residual_mse_strong_bend_over_rest",
        "residual_share_section_mean", "residual_share_within_section", "residual_share_branch_mean",
        "r2_prior_linear", "r2_logtau0_linear", "residual_corr_axial_pairs", "residual_corr_circ_pairs")}
    # pooled branch-level relation: does the per-branch mean residual track the Murray log share?
    xs, ys = [], []
    for row in rows:
        for s, m in row["branch_mean_residual"].items():
            xs.append(row["branch_log_q"][s]); ys.append(m)
    summary["branch_mean_residual_vs_log_q_corr"] = float(np.corrcoef(xs, ys)[0, 1])
    summary["branch_mean_residual_sd_across_case_branches"] = float(np.std(ys))
    text = [f"# 波 1 预登记诊断（C1 best 的 test34 保存预测，log_z 残差；生成 {time.strftime('%Y-%m-%d %H:%M')}）", "",
            "读法：每项是「残差里能被该假设解释的方差份额」的病例中位数（p25–p75），它是对应臂在归一化 R² 上的**理想上限**，不是预测值。", "",
            "| 诊断 | 中位 | p25 | p75 | 对应臂 |", "|---|---:|---:|---:|---|"]
    label = {
        "r2_bend_linear": ("残差 ~ 弯曲参考角 [bend_cos, bend_signed, up2, up5] 线性", "X2"),
        "r2_bif_linear": ("残差 ~ 分叉参考角 [carina_cos, bif_plane_cos, bif_angle_cos] 线性", "X3"),
        "r2_bend_bif_linear": ("残差 ~ 弯曲 + 分叉参考角线性", "X2+X3 / X15"),
        "residual_mse_strong_bend_over_rest": ("强弯曲区（|κR cos| 上四分位）残差 MSE / 其余区", "X2"),
        "residual_share_section_mean": ("残差方差中'截面均值'（分支 × 4 mm bin）份额", "X11（截面 token 上限）"),
        "residual_share_within_section": ("残差方差中'截面内模式'份额", "X7–X9（局部 patch）"),
        "residual_share_branch_mean": ("残差方差中'分支均值'份额", "X5（Murray 先验）"),
        "r2_logtau0_linear": ("残差 ~ log τ0(Murray) 线性", "X5"),
        "r2_prior_linear": ("残差 ~ 人群先验 ln WSS 线性", "X6"),
        "residual_corr_axial_pairs": ("2–4 mm 邻对残差相关（轴向主导对）", "X7 各向异性"),
        "residual_corr_circ_pairs": ("2–4 mm 邻对残差相关（周向主导对）", "X7 各向异性"),
        "case_r2_norm": ("C1 归一化逐例 R²（参考）", "—"),
    }
    for key, (name, arm) in label.items():
        s = summary[key]
        text.append(f"| {name} | {s['median']:.3f} | {s['p25']:.3f} | {s['p75']:.3f} | {arm} |")
    text += ["", f"分支均值残差对 Murray log Q 的相关（34 例 × 7 支 pooled）：{summary['branch_mean_residual_vs_log_q_corr']:+.3f}；"
             f"分支均值残差的跨（例, 支）标准差：{summary['branch_mean_residual_sd_across_case_branches']:.3f}（log_z 单位）。", "",
             "边界：线性拟合是对每例单独做的（含截距），因此「上限」里包含逐例自由度；section 份额用的是 4 mm bin 的经验均值，随 bin 变小份额单调上升。"]
    EXP.mkdir(parents=True, exist_ok=True)
    (EXP / "prereg_diagnostics.json").write_text(json.dumps({"summary": summary, "cases": rows,
                                                             "seconds": time.time() - started}, ensure_ascii=False, indent=1))
    (EXP / "prereg_diagnostics.md").write_text("\n".join(text) + "\n", encoding="utf-8")
    print("\n".join(text))


if __name__ == "__main__":
    main()

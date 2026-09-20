"""Rank cases from the screen outputs and draw two 172-tile contact sheets."""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(__file__).resolve().parent / "screen_out"
df = pd.read_csv(OUT / "screen_per_case.csv", index_col=0)
fl = pd.read_csv(OUT / "screen_section_flags.csv")

# ---- per-case normalised dV/ds anomalies (remove the per-case systematic offset) ----
v = fl[(fl.check == "area_vs_volume_dVds") & (fl.source == "ref")].copy()
med = df["L2_ref_vs_volume_median_relerr"]
v["norm"] = (1 + v.value) / (1 + v.canonical_id.map(med)) - 1
strong_pos = v[v.norm > 0.5].groupby("canonical_id").size()          # section area >1.5x volume estimate
strong_neg = v[v.norm < -0.4].groupby("canonical_id").size()         # section area <0.6x volume estimate
df["L2_dVds_strong_over"] = strong_pos.reindex(df.index).fillna(0).astype(int)
df["L2_dVds_strong_under"] = strong_neg.reindex(df.index).fillna(0).astype(int)
df["L2_dVds_max_over"] = v.groupby("canonical_id").norm.max().reindex(df.index).fillna(0)
df["L1_ref_nonstar"] = df["L1_ref_nonstar_from_center"]
df["L2_Rmis_high"] = fl[(fl.check == "A_over_piRmis2") & (fl.value > 3)].groupby("canonical_id").size().reindex(df.index).fillna(0).astype(int)
df["L2_Rmis_low"] = fl[(fl.check == "A_over_piRmis2") & (fl.value < 0.85)].groupby("canonical_id").size().reindex(df.index).fillna(0).astype(int)

# ---- L3 robust z-scores of anatomy scalars across the cohort ----
scalars = ["trunk_length_mm", "trunk_tortuosity", "trunk_daughter_angle_deg",
           "cia_left_length_mm", "cia_left_tortuosity", "cia_left_parent_angle_deg", "cia_left_daughter_angle_deg",
           "cia_right_length_mm", "cia_right_tortuosity", "cia_right_parent_angle_deg", "cia_right_daughter_angle_deg",
           "inlet_area_mm2", "outlet_area_sum_over_inlet", "ref_valid_frac", "pc_valid_frac", "wall_valid_frac",
           "wall_ambiguous_frac", "L2_ref_vs_volume_median_relerr", "J_seg0_s_mm", "J_seg1_s_mm", "J_seg2_s_mm"]
z = pd.DataFrame(index=df.index)
for c in scalars:
    x = df[c].astype(float)
    m = x.median(); mad = (x - m).abs().median() * 1.4826
    z[c] = (x - m) / (mad if mad > 0 else np.nan)
df["L3_n_outlier_scalars"] = (z.abs() > 3.5).sum(axis=1)
df["L3_outlier_scalars"] = [", ".join(f"{c}={df.loc[i, c]:.3g}(z{z.loc[i, c]:+.1f})" for c in scalars if abs(z.loc[i, c]) > 3.5) for i in df.index]
z.to_csv(OUT / "screen_L3_robust_z.csv")

# hard anatomical bands (not z): CIA length 20-120 mm, tortuosity 1-2, angles 0-90
band = []
for i, r in df.iterrows():
    b = []
    for side in ("left", "right"):
        if not (20 <= r[f"cia_{side}_length_mm"] <= 120): b.append(f"cia_{side}_length={r[f'cia_{side}_length_mm']:.0f}")
        if not (1.0 <= r[f"cia_{side}_tortuosity"] <= 2.0): b.append(f"cia_{side}_tort={r[f'cia_{side}_tortuosity']:.2f}")
        for a in ("parent", "daughter"):
            if not (0 <= r[f"cia_{side}_{a}_angle_deg"] <= 90): b.append(f"cia_{side}_{a}_angle={r[f'cia_{side}_{a}_angle_deg']:.0f}")
    if r["openings_count"] != 5 or r["openings_valid"] != 5: b.append(f"openings={r['openings_valid']}/{r['openings_count']}")
    if not r["topology_valid"] or not r["aortoiliac_pattern"]: b.append("topology")
    band.append("; ".join(b))
df["L3_band_violations"] = band

# ---- composite ranking ----
def short(cid):
    parts = cid.split("/")
    return (parts[1] if parts[0] == "ILO" else parts[-1])[:18]


def reasons(r):
    out = []
    if r["L2_dVds_strong_over"]: out.append(f"截面积>1.5×体网格 {int(r['L2_dVds_strong_over'])} 站(最大 {r['L2_dVds_max_over']:+.1f})")
    if r["L2_dVds_strong_under"]: out.append(f"截面积<0.6×体网格 {int(r['L2_dVds_strong_under'])} 站")
    if r["L2_Rmis_high"]: out.append(f"A/πRmis²>3 {int(r['L2_Rmis_high'])} 站")
    if r["L2_Rmis_low"]: out.append(f"A/πRmis²<0.85 {int(r['L2_Rmis_low'])} 站")
    if r["L2_ref_adjacent_jump_count"]: out.append(f"相邻站跳变>1.5× {int(r['L2_ref_adjacent_jump_count'])} 处")
    if r["L1_ref_nonstar"]: out.append(f"非星形轮廓 {int(r['L1_ref_nonstar'])} 站")
    if abs(r["L2_ref_vs_volume_median_relerr"]) > 0.1: out.append(f"整例系统偏差 {r['L2_ref_vs_volume_median_relerr']:+.2f}")
    if r["L2_pc_vs_ref_n_over20pct"]: out.append(f"点云/面几何分歧 {int(r['L2_pc_vs_ref_n_over20pct'])} 站")
    if r["bif_undetermined"]: out.append(f"分叉未定 {int(r['bif_undetermined'])}")
    if r["L3_band_violations"]: out.append("解剖量出界: " + r["L3_band_violations"])
    if r["L3_n_outlier_scalars"]: out.append("队列离群: " + r["L3_outlier_scalars"])
    return out
df["reasons"] = [reasons(r) for _, r in df.iterrows()]
df["score"] = (3 * df["L2_dVds_strong_over"].clip(upper=5) + 2 * df["L2_Rmis_high"].clip(upper=5) + 2 * df["L2_ref_adjacent_jump_count"].clip(upper=5)
               + 1 * df["L2_dVds_strong_under"].clip(upper=5) + 1 * df["L1_ref_nonstar"].clip(upper=5) + 2 * df["bif_undetermined"]
               + 3 * (df["L3_band_violations"] != "") + 1 * df["L3_n_outlier_scalars"] + 2 * (df["L2_ref_vs_volume_median_relerr"].abs() > 0.1)
               + 1 * df["L2_pc_vs_ref_n_over20pct"].clip(upper=5))
rank = df.sort_values("score", ascending=False)
rank.to_csv(OUT / "screen_ranked.csv")

clean = rank[rank.score == 0]
with open(OUT / "screen_ranked.md", "w") as f:
    f.write(f"# 几何候选自动筛查排名（{len(df)} 例）\n\n")
    f.write(f"零标记（可以只看拼图）：{len(clean)} 例。有标记：{len(rank) - len(clean)} 例。\n\n")
    f.write("| # | 病例 | 分数 | 原因 |\n|---|---|---|---|\n")
    for n, (cid, r) in enumerate(rank[rank.score > 0].iterrows(), 1):
        f.write(f"| {n} | {cid} | {int(r.score)} | {'；'.join(r.reasons)} |\n")
    f.write("\n## 零标记病例\n\n" + ", ".join(clean.index) + "\n")
print(rank[["score", "L2_dVds_strong_over", "L2_Rmis_high", "L2_ref_adjacent_jump_count", "L1_ref_nonstar", "bif_undetermined", "L3_n_outlier_scalars"]].head(40).to_string())
print("zero-score cases:", len(clean))

# ---- contact sheets ----
keys = rank["key"].tolist(); ids = rank.index.tolist(); scores = rank["score"].tolist()
ncol = 12; nrow = int(np.ceil(len(keys) / ncol))
cmap = plt.get_cmap("tab10")

fig, axes = plt.subplots(nrow, ncol, figsize=(ncol * 2.2, nrow * 2.2))
for ax in axes.flat: ax.axis("off")
for ax, key, cid, sc in zip(axes.flat, keys, ids, scores):
    t = np.load(OUT / "thumbs" / f"{key}.npz")
    W = t["wall"]; cl = t["cl"]; seg = t["clseg"]
    # project onto the two principal axes of the wall cloud
    c = W.mean(0); U, S, Vt = np.linalg.svd(W - c, full_matrices=False)
    P = (W - c) @ Vt[:2].T; C = (cl - c) @ Vt[:2].T; SC = (t["sc"] - c) @ Vt[:2].T; OP = (t["open"] - c) @ Vt[:2].T
    ax.scatter(P[:, 0], P[:, 1], s=0.3, c="0.8", rasterized=True)
    for k in np.unique(seg):
        m = seg == k; ax.plot(C[m, 0], C[m, 1], "-", lw=0.8, color=cmap(int(k) % 10))
    ax.scatter(SC[~t["sv"], 0], SC[~t["sv"], 1], s=2, c="red", marker="x", lw=0.4)
    ax.scatter(SC[t["sv"], 0], SC[t["sv"], 1], s=1, c="green")
    ax.scatter(OP[:, 0], OP[:, 1], s=18, facecolors="none", edgecolors="k", lw=0.6)
    ax.set_aspect("equal")
    ax.set_title(f"{short(cid)}  [{int(sc)}]", fontsize=6, color="red" if sc >= 8 else ("darkorange" if sc > 0 else "black"))
fig.suptitle("Geometry contact sheet: wall (grey) · centerline by segment · valid section centroids (green) / invalid (red x) · openings (circles); sorted by screen score", fontsize=9)
fig.tight_layout(rect=(0, 0, 1, 0.985)); fig.savefig(OUT / "contact_sheet_geometry.png", dpi=110); plt.close(fig)

fig, axes = plt.subplots(nrow, ncol, figsize=(ncol * 2.4, nrow * 1.6))
for ax in axes.flat: ax.axis("off")
for ax, key, cid, sc in zip(axes.flat, keys, ids, scores):
    t = np.load(OUT / "thumbs" / f"{key}.npz")
    ss, sA, sv, sseg, pcA, pcv, Av = t["ss"], t["sA"], t["sv"], t["sseg"], t["pcA"], t["pcv"], t["Avol"]
    ax.axis("on"); ax.tick_params(labelsize=4, length=1.5, pad=1)
    offset = 0.0
    for k in np.unique(sseg):
        m = sseg == k; s = ss[m] + offset
        ax.plot(s, Av[m], ".", ms=1.2, color="0.6")
        ax.plot(s[sv[m]], sA[m][sv[m]], "-", lw=0.8, color=cmap(int(k) % 10))
        ax.plot(s[pcv[m]], pcA[m][pcv[m]], ":", lw=0.8, color=cmap(int(k) % 10))
        offset += ss[m].max() + 4
    ax.set_yscale("log")
    ax.set_title(f"{short(cid)}  [{int(sc)}]", fontsize=6, color="red" if sc >= 8 else ("darkorange" if sc > 0 else "black"))
fig.suptitle("Area-profile contact sheet: reference section area (solid) · point-cloud (dotted) · Fluent volume dV/ds oracle (grey dots); segments concatenated along x, log y", fontsize=9)
fig.tight_layout(rect=(0, 0, 1, 0.98)); fig.savefig(OUT / "contact_sheet_area.png", dpi=110); plt.close(fig)
print("wrote", OUT / "contact_sheet_geometry.png", OUT / "contact_sheet_area.png")

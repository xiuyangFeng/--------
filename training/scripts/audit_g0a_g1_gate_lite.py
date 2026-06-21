"""G0-a 轻量审计（登录节点 · 纯 CPU · 不训练 · 少 I/O）。"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]


def _denorm(z, mean, std):
    return z * std + mean


def audit_one_graph(graph_path: Path, norm: dict, max_wall: int = 500) -> dict:
    from training.scripts.run_v3_f0_decision import _denorm_zscore, _load_graph
    from training.scripts.probe_linear_wss import _project_local

    stats = norm["statistics"]
    wl = norm["wss_local"]
    data = _load_graph(graph_path)
    x = data.x.numpy()
    wall = x[:, 9] > 0.5
    idx = np.where(wall)[0]
    if idx.size > max_wall:
        rng = np.random.default_rng(0)
        idx = rng.choice(idx, max_wall, replace=False)

    y_wss = data.y_wss.numpy()[idx]
    yl = data.y_wss_local.numpy()[idx]
    gx = _denorm_zscore(y_wss[:, 1], stats["wss_x"])
    gy = _denorm_zscore(y_wss[:, 2], stats["wss_y"])
    gz = _denorm_zscore(y_wss[:, 3], stats["wss_z"])
    wmag = np.linalg.norm(np.stack([gx, gy, gz], axis=1), axis=1)

    la_s, lc_s, lr_s = yl[:, 1], yl[:, 2], yl[:, 3]
    lc_p = _denorm(lc_s, wl["circ"]["mean"], wl["circ"]["std"])
    lr_p = _denorm(lr_s, wl["rad"]["mean"], wl["rad"]["std"])

    wall_xyz = x[idx, :3]
    int_xyz = x[~wall, :3]
    tan = x[idx, 6:9]
    ax_k, ci_k, ra_k, bv = _project_local(wall_xyz, int_xyz, tan, y_wss[:, 1:4])

    m = bv & np.isfinite(lc_s)
    recon = np.sqrt(yl[:, 1] ** 2 + np.nan_to_num(yl[:, 2]) ** 2 + np.nan_to_num(yl[:, 3]) ** 2)
    # 用 z-score 空间重建 vs wss dim0
    recon_err = np.abs(recon - y_wss[:, 0]) / np.maximum(np.abs(y_wss[:, 0]), 1e-6)

    return {
        "graph": graph_path.name,
        "case": graph_path.parent.parent.parent.name,
        "n_wall": int(idx.size),
        "basis_valid_frac": float(m.mean()),
        "stored_is_zscore_std": float(np.std(np.r_[la_s, lc_s, lr_s])),
        "corr_circ_stored_vs_knn_z": float(np.corrcoef(lc_s[m], ci_k[m])[0, 1]) if m.sum() > 30 else None,
        "recon_zscore_err_median": float(np.nanmedian(recon_err[m])) if m.any() else None,
        "frac_p95_phys": {
            "xy_over_mag": float(np.percentile(np.sqrt(gx[m] ** 2 + gy[m] ** 2) / np.maximum(wmag[m], 1e-12), 95)) if m.any() else None,
            "circ_over_mag": float(np.percentile(np.abs(lc_p[m]) / np.maximum(wmag[m], 1e-12), 95)) if m.any() else None,
            "rad_over_mag": float(np.percentile(np.abs(lr_p[m]) / np.maximum(wmag[m], 1e-12), 95)) if m.any() else None,
        },
    }


def mini_probe(max_train: int = 15, max_test: int = 8) -> dict:
    from training.scripts.probe_linear_wss import run_probe
    from training.scripts.run_v3_f0_decision import _load_norm_stats

    norm = _load_norm_stats(REPO / "data_new/normalization_params_global.json")
    feats = [
        "Abscissa", "NormRadius", "Curvature", "Tangent_X", "Tangent_Y", "Tangent_Z",
        "dist_to_bifurcation", "branch_id", "dR_ds", "torsion", "d_tangent_ds", "dist_to_wall",
    ]
    split = REPO / "training/splits/split_AG_v1.json"
    out = {}
    for frame, src in [("global", "precomputed"), ("local", "precomputed"), ("local", "knn")]:
        key = f"{frame}_{src}" if frame == "local" else "global"
        r = run_probe(
            split_path=split, data_root=REPO / "data_new/AG",
            graphs_subdir_global="graphs", graphs_subdir_local="graphs_local_v1",
            feature_names=feats, norm_stats=norm, target_frame=frame,
            models=["ridge"], max_graphs_per_case=2, max_wall_per_graph=400,
            local_source=src, max_train_cases=max_train, max_test_cases=max_test,
        )
        comps = r["models"]["ridge"]["components"]
        if frame == "global":
            best = max(comps["wss_x"]["test_r2"], comps["wss_y"]["test_r2"])
            out[key] = {"best_xy": best, "detail": {k: comps[k]["test_r2"] for k in comps}}
        else:
            best = max(comps["wss_circ"]["test_r2"], comps["wss_rad"]["test_r2"])
            out[key] = {"best_circ_rad": best, "detail": {k: comps[k]["test_r2"] for k in comps}}

    g = out["global"]["best_xy"]
    lp = out["local_precomputed"]["best_circ_rad"]
    lk = out["local_knn"]["best_circ_rad"]
    return {
        "cases": {"train": max_train, "test": max_test},
        "global_xy_best": g,
        "local_precomputed_best": lp,
        "local_knn_best": lk,
        "delta_precomputed": lp - g,
        "delta_knn": lk - g,
        "detail": out,
    }


def main():
    norm_path = REPO / "data_new/normalization_params_global.json"
    with norm_path.open() as f:
        norm = json.load(f)

    ag = REPO / "data_new/AG"
    graphs = []
    for rel in ["fast/LI_HUAN_GE", "fast/ZHANG_CHUN", "slow/HOU_SHEN_QIAN"]:
        d = ag / rel / "processed/graphs_local_v1"
        if d.is_dir():
            graphs.append(next(iter(sorted(d.glob("*.pt")))))

    consistency = [audit_one_graph(p, norm) for p in graphs]
    probe = mini_probe()

    # 读 S0 正式结论对照
    s0_path = REPO / "outputs/field/f0_decision/v3p_g0_oracle_20260606_v2.json"
    s0_gate = None
    if s0_path.is_file():
        with s0_path.open() as f:
            s0 = json.load(f)
        s0_gate = s0.get("gates", {}).get("G0a_coordinate_burden")

    report = {
        "mode": "lite_cpu_no_train",
        "graph_consistency": consistency,
        "mini_probe_post_denylist_split": probe,
        "s0_official_gate": s0_gate,
        "risks": [],
    }

    # 自动风险标注
    fracs = [c["frac_p95_phys"]["circ_over_mag"] for c in consistency if c["frac_p95_phys"]["circ_over_mag"]]
    if fracs and np.median(fracs) < 0.05:
        report["risks"].append("circ 物理占比 p95 中位 <5%：横向分量信号极弱，探针 R²≈0 可能是真实现象而非 bug。")

    if probe["delta_precomputed"] < 0 and probe["delta_knn"] < 0:
        report["risks"].append("precomputed 与 knn 两种 local 源均不优于 global：不太像单一实现 bug。")

    if s0_gate and s0_gate.get("pass") is False:
        report["risks"].append(
            f"S0 正式跑（60/17 pre-denylist）Δ={s0_gate.get('delta_local_minus_global'):.4f}；"
            "mini_probe 用 post-denylist split 子集复核。"
        )

    out = REPO / "outputs/field/f0_decision/g0a_g1_audit_lite.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()

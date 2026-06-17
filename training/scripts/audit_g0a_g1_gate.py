"""G0-a / G1 门控审计：数据一致性 + 探针敏感性复核。

检查项：
  1. graphs_local_v1 中 y_wss_local 是否为 z-score（与 norm json 对照）
  2. 由 global wss + 切向/法向重投影 local，与存储值一致性
  3. circ/rad 物理方差占比（是否「本来就没信号」）
  4. precomputed vs knn local_source 探针对比
  5. 子采样 vs 全量探针对比（post-denylist split）
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[2]


def _load_norm(path: Path) -> Dict:
    with path.open() as f:
        return json.load(f)


def _denorm(z: np.ndarray, mean: float, std: float) -> np.ndarray:
    return z * std + mean


def audit_graph_consistency(
    graph_path: Path,
    norm: Dict,
    *,
    max_wall: int = 2000,
) -> Dict[str, object]:
    from training.scripts.run_v3_f0_decision import _denorm_zscore, _load_graph

    stats = norm["statistics"]
    wl = norm.get("wss_local", {})
    data = _load_graph(graph_path)
    x = data.x.numpy()
    wall = x[:, 9] > 0.5
    n_wall = int(wall.sum())
    if n_wall < 10:
        return {"error": "too few wall points", "path": str(graph_path)}

    idx = np.where(wall)[0]
    if idx.size > max_wall:
        rng = np.random.default_rng(0)
        idx = rng.choice(idx, max_wall, replace=False)

    y_wss = data.y_wss.numpy()[idx]
    yl = data.y_wss_local.numpy()[idx] if hasattr(data, "y_wss_local") else None
    if yl is None:
        return {"error": "no y_wss_local", "path": str(graph_path)}

    # global denorm
    gx = _denorm_zscore(y_wss[:, 1], stats["wss_x"])
    gy = _denorm_zscore(y_wss[:, 2], stats["wss_y"])
    gz = _denorm_zscore(y_wss[:, 3], stats["wss_z"])
    gmag = _denorm_zscore(y_wss[:, 0], stats["wss"])
    wvec = np.stack([gx, gy, gz], axis=1)

    # local stored: z-score or physical?
    la_s, lc_s, lr_s = yl[:, 1], yl[:, 2], yl[:, 3]
    la_p = _denorm(la_s, wl["axial"]["mean"], wl["axial"]["std"])
    lc_p = _denorm(lc_s, wl["circ"]["mean"], wl["circ"]["std"])
    lr_p = _denorm(lr_s, wl["rad"]["mean"], wl["rad"]["std"])

    # heuristic: if stored values look like z-score (small std range), flag
    stored_std = float(np.std(np.concatenate([la_s, lc_s, lr_s])))
    phys_std = float(np.std(np.concatenate([la_p, lc_p, lr_p])))

    # reproject from global using graph tangent + kNN normal (same as probe knn)
    from training.scripts.probe_linear_wss import _project_local

    wall_xyz = x[idx, :3]
    int_xyz = x[~wall, :3]
    tan = x[idx, 6:9]
    wvec_z = y_wss[:, 1:4]  # z-scored components (graph 空间)
    ax_k, ci_k, ra_k, bv = _project_local(wall_xyz, int_xyz, tan, wvec_z)

    # compare z-scored stored vs knn on z-scored wss (apples-to-apples in graph space)
    m = bv & np.isfinite(lc_s) & np.isfinite(la_s)
    corr_ax = float(np.corrcoef(la_s[m], ax_k[m])[0, 1]) if m.sum() > 20 else float("nan")
    corr_ci = float(np.corrcoef(lc_s[m], ci_k[m])[0, 1]) if m.sum() > 20 else float("nan")

    # physical magnitude fractions
    wmag = np.linalg.norm(wvec, axis=1)
    circ_frac = np.abs(lc_p[m]) / np.maximum(wmag[m], 1e-12)
    rad_frac = np.abs(lr_p[m]) / np.maximum(wmag[m], 1e-12)
    xy_frac = np.sqrt(gx[m] ** 2 + gy[m] ** 2) / np.maximum(wmag[m], 1e-12)

    return {
        "path": str(graph_path),
        "n_wall_sampled": int(idx.size),
        "basis_valid_frac": float(m.mean()),
        "stored_looks_zscore_std": stored_std,
        "phys_std_after_denorm": phys_std,
        "corr_stored_vs_knn_reproj": {"axial": corr_ax, "circ": corr_ci},
        "frac_p50": {
            "global_xy_over_mag": float(np.median(xy_frac)),
            "local_circ_over_mag": float(np.median(circ_frac)),
            "local_rad_over_mag": float(np.median(rad_frac)),
        },
        "frac_p95": {
            "global_xy_over_mag": float(np.percentile(xy_frac, 95)),
            "local_circ_over_mag": float(np.percentile(circ_frac, 95)),
            "local_rad_over_mag": float(np.percentile(rad_frac, 95)),
        },
        "wss_mag_phys_p50": float(np.median(wmag)),
    }


def run_probe_summary(
    split_path: Path,
    *,
    max_graphs: int,
    max_wall: int,
    local_source: str,
) -> Dict[str, object]:
    from training.scripts.probe_linear_wss import run_probe
    from training.scripts.run_v3_f0_decision import _load_norm_stats

    norm_stats = _load_norm_stats(REPO / "data_new/normalization_params_global.json")
    out = {}
    for frame in ("global", "local"):
        out[frame] = run_probe(
            split_path=split_path,
            data_root=REPO / "data_new/AG",
            graphs_subdir_global="graphs",
            graphs_subdir_local="graphs_local_v1",
            feature_names=[
                "Abscissa", "NormRadius", "Curvature",
                "Tangent_X", "Tangent_Y", "Tangent_Z",
                "dist_to_bifurcation", "branch_id",
                "dR_ds", "torsion", "d_tangent_ds", "dist_to_wall",
            ],
            norm_stats=norm_stats,
            target_frame=frame,
            models=["ridge"],
            max_graphs_per_case=max_graphs,
            max_wall_per_graph=max_wall,
            local_source=local_source,
        )
    g = out["global"]["models"]["ridge"]["components"]
    l = out["local"]["models"]["ridge"]["components"]
    gxy = max(g["wss_x"]["test_r2"], g["wss_y"]["test_r2"])
    lcr = max(l["wss_circ"]["test_r2"], l["wss_rad"]["test_r2"])
    return {
        "config": {"max_graphs": max_graphs, "max_wall": max_wall, "local_source": local_source},
        "global_xy_best": gxy,
        "local_circ_rad_best": lcr,
        "delta": lcr - gxy,
        "detail": {
            "wss_x": g["wss_x"]["test_r2"],
            "wss_y": g["wss_y"]["test_r2"],
            "wss_z": g["wss_z"]["test_r2"],
            "wss_axial": l["wss_axial"]["test_r2"],
            "wss_circ": l["wss_circ"]["test_r2"],
            "wss_rad": l["wss_rad"]["test_r2"],
        },
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", type=Path, default=REPO / "training/splits/split_AG_v1.json")
    ap.add_argument("--n-graphs-audit", type=int, default=5)
    ap.add_argument("--output", type=Path, default=REPO / "outputs/field/f0_decision/g0a_g1_audit.json")
    args = ap.parse_args()

    norm = _load_norm(REPO / "data_new/normalization_params_global.json")
    # pick diverse graphs from train+test
    ag = REPO / "data_new/AG"
    graph_paths: List[Path] = []
    for case in ["fast/LI_HUAN_GE", "slow/HOU_SHEN_QIAN", "slow/LIN_SHU_TIAN"]:
        d = ag / case / "processed/graphs_local_v1"
        if d.is_dir():
            graph_paths.extend(sorted(d.glob("*.pt"))[:2])

    consistency = [audit_graph_consistency(p, norm) for p in graph_paths[: args.n_graphs_audit]]

    probes = [
        run_probe_summary(args.split, max_graphs=3, max_wall=800, local_source="precomputed"),
        run_probe_summary(args.split, max_graphs=10, max_wall=2000, local_source="precomputed"),
        run_probe_summary(args.split, max_graphs=3, max_wall=800, local_source="knn"),
    ]

    report = {
        "split": str(args.split),
        "graph_consistency_samples": consistency,
        "probe_sensitivity": probes,
        "gate_threshold_delta": 0.10,
        "interpretation_notes": [
            "G0-a 比较的是 local circ/rad vs global x/y 的 Ridge test R²，不是 G1 VNHead 本身。",
            "若 consistency 中 circ/rad 物理占比极低，No-Go 可能反映真实物理而非代码 bug。",
            "若 precomputed 与 knn 探针结论相反，优先怀疑 local 图资产或法向 snap。",
        ],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()

"""单病例 local frame / STL 法向 / 切向对齐诊断（V3P 数据质检）。

用法::

    python -m training.scripts.diagnose_local_frame_case \\
        --case slow/HOU_SHEN_QIAN \\
        --ref-case slow/LI_HUAN_GE
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from pipeline.local_wss import (
    BASIS_PARALLEL_COS,
    StlNormalLookup,
    compute_qa_stats,
    find_case_stl,
    project_wss_to_local_frame,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
BASIS_PARALLEL_COS_VAL = BASIS_PARALLEL_COS


def _load_features_csv(case_dir: Path, timestep: str = "1120") -> pd.DataFrame:
    feat_dir = case_dir / "processed" / "features"
    matches = sorted(feat_dir.glob(f"result_features_merged-{timestep}.csv"))
    if not matches:
        matches = sorted(feat_dir.glob("result_features_merged-*.csv"))
    if not matches:
        raise FileNotFoundError(f"无 features CSV: {feat_dir}")
    return pd.read_csv(matches[0])


def _wall_arrays(df: pd.DataFrame) -> Dict[str, np.ndarray]:
    wall = df["is_wall"].astype(bool).values
    tan = df.loc[wall, ["Tangent_X", "Tangent_Y", "Tangent_Z"]].values.astype(np.float64)
    tnorm = np.linalg.norm(tan, axis=1)
    t_hat = tan / np.maximum(tnorm[:, None], 1e-12)
    wss = df.loc[wall, ["wss_x", "wss_y", "wss_z"]].values.astype(np.float64)
    wmag = np.linalg.norm(wss, axis=1)
    return {
        "wall": wall,
        "tan": tan,
        "t_hat": t_hat,
        "tnorm": tnorm,
        "wss": wss,
        "wmag": wmag,
        "abscissa": df.loc[wall, "Abscissa"].values.astype(np.float64),
    }


def diagnose_case(
    case_rel: str,
    *,
    data_root: Path,
    timestep: str = "1120",
) -> Dict[str, object]:
    case_dir = data_root / case_rel
    df = _load_features_csv(case_dir, timestep=timestep)
    stl_path = find_case_stl(case_dir)
    if stl_path is None:
        raise FileNotFoundError(f"未找到 STL: {case_dir}")

    stl_lookup = StlNormalLookup(stl_path)
    proj = project_wss_to_local_frame(df, normal_source="stl", stl_lookup=stl_lookup)
    qa = compute_qa_stats(proj, case_name=case_rel, normal_source="stl")

    wall = proj["is_wall"].astype(bool).values
    coords = proj.loc[wall, ["x", "y", "z"]].values.astype(np.float64)
    stl_n, nv = stl_lookup.normals_at(proj[["x", "y", "z"]].values.astype(np.float64), wall)
    tan = proj.loc[wall, ["Tangent_X", "Tangent_Y", "Tangent_Z"]].values.astype(np.float64)
    tnorm = np.linalg.norm(tan, axis=1)
    t_hat = tan / np.maximum(tnorm[:, None], 1e-12)

    cos_nt = np.sum(stl_n[wall] * t_hat, axis=1)
    abs_cos_nt = np.abs(cos_nt)
    b_raw = np.cross(stl_n[wall], t_hat)
    b_norm = np.linalg.norm(b_raw, axis=1)

    wss = proj.loc[wall, ["wss_x", "wss_y", "wss_z"]].values.astype(np.float64)
    wmag = np.linalg.norm(wss, axis=1)
    axial = proj.loc[wall, "wss_axial"].values.astype(np.float64)
    circ = proj.loc[wall, "wss_circ"].values.astype(np.float64)
    rad = proj.loc[wall, "wss_rad"].values.astype(np.float64)
    bv = proj.loc[wall, "wss_basis_valid"].values.astype(np.float64)
    absc = proj.loc[wall, "Abscissa"].values.astype(np.float64)

    valid_basis = bv > 0.5
    recon = np.sqrt(axial**2 + np.nan_to_num(circ) ** 2 + np.nan_to_num(rad) ** 2)
    recon_err = np.abs(recon - wmag) / np.maximum(wmag, 1e-12)

    # 高 rad 区域
    rad_abs = np.abs(rad)
    rad_p95 = float(np.nanpercentile(rad_abs[valid_basis], 95)) if valid_basis.any() else float("nan")
    high_rad_thr = max(rad_p95 * 0.5, np.nanpercentile(rad_abs[valid_basis], 90)) if valid_basis.any() else 0.0
    high_rad_mask = valid_basis & (rad_abs >= high_rad_thr)

    stats = {
        "case": case_rel,
        "stl": str(stl_path),
        "timestep": timestep,
        "n_wall": int(wall.sum()),
        "qa": qa,
        "tangent": {
            "unit_frac": float(np.mean(np.abs(tnorm - 1.0) < 0.05)),
            "norm_p50": float(np.percentile(tnorm, 50)),
            "norm_p05": float(np.percentile(tnorm, 5)),
        },
        "stl_normal": {
            "valid_rate": float(nv[wall].mean()),
        },
        "n_t_alignment": {
            "abs_cos_median": float(np.median(abs_cos_nt)),
            "abs_cos_p95": float(np.percentile(abs_cos_nt, 95)),
            "abs_cos_p99": float(np.percentile(abs_cos_nt, 99)),
            "fraction_parallel_ge_0.95": float(np.mean(abs_cos_nt >= BASIS_PARALLEL_COS_VAL)),
            "fraction_parallel_ge_0.80": float(np.mean(abs_cos_nt >= 0.80)),
        },
        "basis": {
            "valid_rate": float(bv.mean()),
            "b_norm_p50": float(np.percentile(b_norm, 50)),
            "b_norm_p05": float(np.percentile(b_norm, 5)),
        },
        "wss_decomp": {
            "wmag_p50": float(np.percentile(wmag, 50)),
            "wmag_p95": float(np.percentile(wmag, 95)),
            "axial_frac_p95": float(np.percentile(np.abs(axial) / np.maximum(wmag, 1e-12), 95)),
            "circ_frac_p95": float(np.percentile(np.abs(circ[valid_basis]) / np.maximum(wmag[valid_basis], 1e-12), 95))
            if valid_basis.any() else None,
            "rad_frac_p95": float(np.percentile(rad_abs[valid_basis] / np.maximum(wmag[valid_basis], 1e-12), 95))
            if valid_basis.any() else None,
            "recon_rel_err_median": float(np.nanmedian(recon_err[valid_basis])) if valid_basis.any() else None,
        },
        "high_rad": {
            "n_points": int(high_rad_mask.sum()),
            "abscissa_range": [
                float(absc[high_rad_mask].min()) if high_rad_mask.any() else None,
                float(absc[high_rad_mask].max()) if high_rad_mask.any() else None,
            ],
            "abs_cos_nt_median": float(np.median(abs_cos_nt[high_rad_mask])) if high_rad_mask.any() else None,
        },
    }
    arrays = {
        "abscissa": absc,
        "abs_cos_nt": abs_cos_nt,
        "b_norm": b_norm,
        "wmag": wmag,
        "axial": axial,
        "circ": circ,
        "rad": rad,
        "rad_abs": rad_abs,
        "basis_valid": bv,
        "recon_err": recon_err,
        "high_rad_mask": high_rad_mask,
    }
    return {"stats": stats, "arrays": arrays}


def _plot_comparison(
    target: Dict[str, object],
    ref: Optional[Dict[str, object]],
    out_dir: Path,
) -> List[str]:
    fig_paths: List[str] = []
    ta = target["arrays"]
    case = target["stats"]["case"]

    # 1. |n·t| 沿 Abscissa
    fig, ax = plt.subplots(figsize=(10, 4))
    sc = ax.scatter(
        ta["abscissa"], ta["abs_cos_nt"], c=ta["rad_abs"], s=3, cmap="viridis",
        alpha=0.5, vmin=0, vmax=np.nanpercentile(ta["rad_abs"], 99),
    )
    ax.axhline(BASIS_PARALLEL_COS_VAL, color="r", ls="--", label=f"parallel thresh={BASIS_PARALLEL_COS_VAL}")
    ax.set_xlabel("Abscissa")
    ax.set_ylabel("|n̂·t̂|")
    ax.set_title(f"{case}: STL normal vs centerline tangent alignment")
    plt.colorbar(sc, ax=ax, label="|wss_rad| (Pa)")
    ax.legend()
    fig.tight_layout()
    p1 = out_dir / "01_abs_cos_nt_vs_abscissa.png"
    fig.savefig(p1, dpi=150)
    plt.close(fig)
    fig_paths.append(str(p1))

    # 2. WSS 分量占比
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    valid = ta["basis_valid"] > 0.5
    w = ta["wmag"][valid]
    for ax, comp, data, title in zip(
        axes,
        ["axial", "circ", "rad"],
        [ta["axial"][valid], ta["circ"][valid], ta["rad"][valid]],
        ["|axial|/|WSS|", "|circ|/|WSS|", "|rad|/|WSS|"],
    ):
        frac = np.abs(data) / np.maximum(w, 1e-12)
        ax.hist(frac, bins=60, color="steelblue", alpha=0.85)
        ax.axvline(0.3, color="r", ls="--", label="QA thresh 0.3")
        ax.set_title(title)
        ax.set_xlabel("fraction")
    fig.suptitle(f"{case}: local WSS component fractions (basis-valid wall)")
    fig.tight_layout()
    p2 = out_dir / "02_component_fraction_hist.png"
    fig.savefig(p2, dpi=150)
    plt.close(fig)
    fig_paths.append(str(p2))

    # 3. 对照病例 |n·t| CDF
    if ref is not None:
        fig, ax = plt.subplots(figsize=(7, 5))
        for label, pack, color in [
            (case, ta["abs_cos_nt"], "C0"),
            (ref["stats"]["case"], ref["arrays"]["abs_cos_nt"], "C1"),
        ]:
            xs = np.sort(pack)
            ys = np.arange(1, xs.size + 1) / xs.size
            ax.plot(xs, ys, label=label, color=color)
        ax.axvline(BASIS_PARALLEL_COS_VAL, color="r", ls="--", label="parallel thresh")
        ax.set_xlabel("|n̂·t̂|")
        ax.set_ylabel("CDF")
        ax.set_title("STL normal – tangent alignment CDF")
        ax.legend()
        fig.tight_layout()
        p3 = out_dir / "03_abs_cos_nt_cdf_vs_ref.png"
        fig.savefig(p3, dpi=150)
        plt.close(fig)
        fig_paths.append(str(p3))

    # 4. 重建误差 vs rad
    fig, ax = plt.subplots(figsize=(8, 5))
    m = valid
    ax.scatter(ta["rad_abs"][m], ta["recon_err"][m], s=2, alpha=0.3, c=ta["abs_cos_nt"][m], cmap="coolwarm")
    ax.set_xlabel("|wss_rad| (Pa)")
    ax.set_ylabel("|recon - |WSS|| / |WSS|")
    ax.set_title(f"{case}: decomposition consistency (color=|n·t|)")
    fig.tight_layout()
    p4 = out_dir / "04_recon_err_vs_rad.png"
    fig.savefig(p4, dpi=150)
    plt.close(fig)
    fig_paths.append(str(p4))

    return fig_paths


def _verdict(stats: Dict[str, object], ref_stats: Optional[Dict[str, object]]) -> Dict[str, object]:
    qa = stats["qa"]
    nt = stats["n_t_alignment"]
    wss = stats["wss_decomp"]
    lines: List[str] = []
    severity = "warn"

    if qa.get("wss_rad_p95_over_wss_p95", 0) > 0.3:
        lines.append(
            f"物理层 rad_ratio p95={qa['wss_rad_p95_over_wss_p95']:.3f} > 0.3："
            "径向 WSS 分量相对模长偏高，local 标签可信度下降。"
        )
        severity = "fail"

    if nt["fraction_parallel_ge_0.95"] > 0.05:
        lines.append(
            f"{nt['fraction_parallel_ge_0.95']*100:.1f}% 壁面点 |n̂·t̂|≥0.95："
            "法向与切向近平行，circ/rad 基底退化（basis_invalid）。"
        )
        if severity != "fail":
            severity = "warn"

    if wss.get("rad_frac_p95") and wss["rad_frac_p95"] > 0.35:
        lines.append(
            f"径向分量占 |WSS| 的 p95 份额={wss['rad_frac_p95']:.3f}："
            "WSS 矢量并非 primarily 切向，需核对 CFD 矢量或法向 snap。"
        )

    if ref_stats is not None:
        ref_nt = ref_stats["n_t_alignment"]
        if nt["abs_cos_p95"] > ref_nt["abs_cos_p95"] * 1.5:
            lines.append(
                f"|n̂·t̂| p95={nt['abs_cos_p95']:.3f} vs 对照 {ref_nt['abs_cos_p95']:.3f}："
                "该病例法向-切向错位明显重于正常病例。"
            )

    if not lines:
        lines.append("未发现显著 STL/切向/局部投影异常。")
        severity = "ok"

    return {"severity": severity, "summary_lines": lines}


def main() -> None:
    ap = argparse.ArgumentParser(description="单病例 local frame / STL 对齐诊断")
    ap.add_argument("--case", required=True, help="相对 data_new/AG，如 slow/HOU_SHEN_QIAN")
    ap.add_argument("--ref-case", default="slow/LI_HUAN_GE")
    ap.add_argument("--data-root", type=Path, default=REPO_ROOT / "data_new" / "AG")
    ap.add_argument("--timestep", default="1120")
    ap.add_argument("--output-dir", type=Path, default=None)
    args = ap.parse_args()

    slug = args.case.replace("/", "_")
    out_dir = args.output_dir or (
        REPO_ROOT / "outputs" / "field" / "diagnostics" / f"v3p_local_frame_{slug}"
    )
    out_dir.mkdir(parents=True, exist_ok=True)

    target = diagnose_case(args.case, data_root=args.data_root, timestep=args.timestep)
    ref = None
    ref_stats = None
    if args.ref_case:
        ref = diagnose_case(args.ref_case, data_root=args.data_root, timestep=args.timestep)
        ref_stats = ref["stats"]

    figs = _plot_comparison(target, ref, out_dir)
    verdict = _verdict(target["stats"], ref_stats)

    report = {
        "target": target["stats"],
        "reference": ref_stats,
        "verdict": verdict,
        "figures": figs,
        "thresholds": {
            "basis_parallel_cos": BASIS_PARALLEL_COS_VAL,
            "qa_rad_ratio_max": 0.3,
        },
    }
    json_path = out_dir / "diagnosis.json"
    with json_path.open("w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    md_path = out_dir / "README.md"
    md_lines = [
        f"# Local frame 诊断：{args.case}",
        "",
        f"对照病例：`{args.ref_case}` · 时间步 `merged-{args.timestep}`",
        "",
        "## 结论",
        "",
    ]
    md_lines.extend(f"- {line}" for line in verdict["summary_lines"])
    md_lines.extend(["", "## 关键指标", ""])
    t = target["stats"]
    md_lines.append(f"| 指标 | 值 |")
    md_lines.append(f"| --- | --- |")
    md_lines.append(f"| QA rad_ratio (物理) | {t['qa'].get('wss_rad_p95_over_wss_p95', float('nan')):.4f} |")
    md_lines.append(f"| basis_valid 率 | {t['basis']['valid_rate']:.4f} |")
    md_lines.append(f"| \|n·t\| median | {t['n_t_alignment']['abs_cos_median']:.4f} |")
    md_lines.append(f"| \|n·t\| p95 | {t['n_t_alignment']['abs_cos_p95']:.4f} |")
    md_lines.append(f"| \|n·t\|≥0.95 占比 | {t['n_t_alignment']['fraction_parallel_ge_0.95']*100:.2f}% |")
    md_lines.append(f"| \|rad\|/\|WSS\| p95 | {t['wss_decomp'].get('rad_frac_p95', float('nan')):.4f} |")
    if ref_stats:
        md_lines.append(f"| 对照 \|n·t\| p95 | {ref_stats['n_t_alignment']['abs_cos_p95']:.4f} |")
    md_lines.extend(["", "## 图", ""])
    for fp in figs:
        md_lines.append(f"![{Path(fp).name}]({Path(fp).name})")
    md_path.write_text("\n".join(md_lines), encoding="utf-8")

    print(json.dumps({"out_dir": str(out_dir), "verdict": verdict}, ensure_ascii=False, indent=2))
    print(json_path)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""V3P M-E · E-J 压力梯度桥接 oracle（0 重训 · CPU）。

命题：壁面 |∇p| 模长（主信号）或 ∂p/∂s（对照）能否在 GBDT probe 中增量解释 GT WSS？
对照工程方案 §5 E-J / O4；规避 5276 direct vel_diff 失败模式。

产出 ``outputs/field/f0_decision/v3p_me_ej_oracle_<date>.json``。

用法::

    python -m training.scripts.run_v3_me_ej_oracle
    python -m training.scripts.run_v3_me_ej_oracle --max-cases 6 --max-graphs-per-case 2
"""
from __future__ import annotations

import argparse
import json
import math
import re
import time
from datetime import date
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import LinearRegression

from ..core.denylist import filter_case_names
from ..core.splits import SplitSpec
from .run_v3_f0_decision import (
    DEFAULT_NORM_PARAMS,
    REPO_ROOT,
    _load_norm_stats,
    _r2_score,
    _safe_json_float,
    _safe_pearson,
    _safe_spearman,
    oracle_pressure_gradient_wss,
)
from .run_v3_g0_oracle import _collect_pressure_geom

# O4 / §5+.x 阈值
CLOSURE_SPEARMAN_MIN = 0.30
CLOSURE_RESIDUAL_MAX = 0.70  # 1 - R²(linear WSS ~ -dp/ds) ≤ 0.70 ⇒ R² ≥ 0.30
DELTA_R2_GO_MIN = 0.05
WEAK_GO_GRADP_R2_MIN = 0.45
GEOM_COLS = slice(2, 8)  # Abscissa..torsion
GRADP_COLS = [0, 2, 3, 4, 5, 6, 7]  # |∇p| + geom
DPDS_COLS = [1, 2, 3, 4, 5, 6, 7]  # -dp/ds + geom
FEATURE_NAMES = [
    "grad_p_mag", "neg_dp_ds", "Abscissa", "Curvature", "NormRadius",
    "dist_to_bif", "dR_ds", "torsion",
]
DEFAULT_I6DIAG_RUN = (
    "outputs/field/field_v3_pointnext_i6diag_localpool_main01_geom_pw_asymw_a_wall13000_near2000"
    "_split_AG_v1_seed1_20260619_174001"
)


def _patient_family(case_rel: str) -> Optional[str]:
    name = case_rel.split("/")[-1]
    m = re.match(r"^(GUO|ZHANG|CHEN)", name, re.IGNORECASE)
    return m.group(1).upper() if m else None


def _round4(v: Optional[float]) -> Optional[float]:
    if v is None or not math.isfinite(float(v)):
        return None
    return round(float(v), 4)


def _fit_gbdt_probe(
    Xtr: np.ndarray,
    ytr: np.ndarray,
    Xte: np.ndarray,
    yte: np.ndarray,
    feat_cols: Sequence[int],
    *,
    seed: int = 0,
) -> Tuple[Optional[float], np.ndarray]:
    if Xtr.shape[0] < 50 or Xte.shape[0] < 20:
        return None, np.empty(0)
    reg = HistGradientBoostingRegressor(
        max_iter=200, learning_rate=0.05, l2_regularization=1.0, random_state=seed,
    ).fit(Xtr[:, feat_cols], ytr)
    pred = reg.predict(Xte[:, feat_cols])
    return _r2_score(yte, pred), pred


def _per_case_r2(
    cases: Sequence[str],
    yte: np.ndarray,
    pred: np.ndarray,
    case_ids: Sequence[str],
) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for case_rel in cases:
        mask = np.array([c == case_rel for c in case_ids], dtype=bool)
        if int(np.sum(mask)) < 10:
            continue
        r2 = _r2_score(yte[mask], pred[mask])
        rows.append({
            "case": case_rel,
            "family": _patient_family(case_rel),
            "n": int(np.sum(mask)),
            "test_r2": _round4(r2),
        })
    return rows


def _family_breakdown(case_rows: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for fam in ("GUO", "ZHANG", "CHEN"):
        rows = [r for r in case_rows if r.get("family") == fam and r.get("test_r2") is not None]
        if not rows:
            out[fam] = {"n_cases": 0, "mean_r2": None}
            continue
        out[fam] = {
            "n_cases": len(rows),
            "mean_r2": _round4(float(np.mean([float(r["test_r2"]) for r in rows]))),
            "cases": [r["case"] for r in rows],
        }
    return out


def _collect_with_case_ids(
    cases: Sequence[str],
    *,
    data_root: Path,
    graphs_subdir: str,
    norm_stats: Mapping[str, Dict[str, float]],
    max_graphs_per_case: int,
    max_wall_per_graph: int,
    seed: int,
    max_cases: Optional[int] = None,
    verbose: bool = False,
) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """逐点收集特征与 WSS，并记录每点所属 case。"""
    from sklearn.neighbors import NearestNeighbors

    from .run_v3_f0_decision import (
        NODE_IDX,
        X_TAN,
        Y_P,
        _denorm_wss_mag,
        _denorm_zscore,
        _load_graph,
    )

    rng = np.random.default_rng(seed)
    X_rows: List[np.ndarray] = []
    y_rows: List[np.ndarray] = []
    case_ids: List[str] = []
    case_list = list(cases) if max_cases is None else list(cases)[:max_cases]

    for case_rel in case_list:
        case_dir = data_root / case_rel / "processed" / graphs_subdir
        if not case_dir.is_dir():
            continue
        for gp in sorted(case_dir.glob("*.pt"))[:max_graphs_per_case]:
            data = _load_graph(gp)
            x = data.x.numpy()
            wall = x[:, NODE_IDX["is_wall"]] > 0.5
            if int(np.sum(wall)) < 20:
                continue
            y = data.y.numpy()
            y_wss = data.y_wss.numpy()
            wall_xyz = x[wall, :3].astype(np.float64)
            p_wall = _denorm_zscore(y[wall, Y_P], norm_stats.get("p"))
            tan = x[wall, X_TAN].astype(np.float64)
            wss_mag = _denorm_wss_mag(y_wss[wall, 0], norm_stats)
            geom = np.stack([
                x[wall, NODE_IDX["Abscissa"]], x[wall, NODE_IDX["Curvature"]],
                x[wall, 4],
                x[wall, 10], x[wall, 12], x[wall, 13],
            ], axis=1).astype(np.float64)

            n_wall = wall_xyz.shape[0]
            if n_wall > max_wall_per_graph:
                q_idx = rng.choice(n_wall, size=max_wall_per_graph, replace=False)
            else:
                q_idx = np.arange(n_wall)
            k_eff = int(min(12, n_wall - 1))
            nn = NearestNeighbors(n_neighbors=k_eff + 1).fit(wall_xyz)
            _, idx_nbr = nn.kneighbors(wall_xyz[q_idx])
            dpds = np.full(q_idx.size, np.nan)
            gmag = np.full(q_idx.size, np.nan)
            for r, wi in enumerate(q_idx):
                nb = idx_nbr[r, 1:]
                dx = wall_xyz[nb] - wall_xyz[wi]
                dp = p_wall[nb] - p_wall[wi]
                g, *_ = np.linalg.lstsq(dx, dp, rcond=None)
                t = tan[wi]
                tn = np.linalg.norm(t)
                if tn > 1e-9:
                    dpds[r] = float(g @ (t / tn))
                gmag[r] = float(np.linalg.norm(g))
            feat = np.column_stack([gmag, -dpds, geom[q_idx]])
            wss_q = wss_mag[q_idx]
            ok = np.all(np.isfinite(feat), axis=1) & np.isfinite(wss_q)
            if int(np.sum(ok)) < 10:
                continue
            n_ok = int(np.sum(ok))
            X_rows.append(feat[ok])
            y_rows.append(wss_q[ok])
            case_ids.extend([case_rel] * n_ok)
            if verbose:
                print(f"  {case_rel}/{gp.name}: ok={n_ok}", flush=True)

    if not X_rows:
        return np.empty((0, 8)), np.empty(0), []
    return np.concatenate(X_rows, axis=0), np.concatenate(y_rows, axis=0), case_ids


def _closure_audit(
    Xte: np.ndarray,
    yte: np.ndarray,
    pg_corr: Mapping[str, Any],
) -> Dict[str, Any]:
    """GT 闭合审计：∂p/∂s 与 WSS 局部物理关系。"""
    neg_dpds = Xte[:, 1]  # already -dp/ds
    mask = np.isfinite(neg_dpds) & np.isfinite(yte)
    closure: Dict[str, Any] = {"n": int(np.sum(mask))}
    if int(np.sum(mask)) < 20:
        closure["error"] = "insufficient_samples"
        return closure

    x_lin = neg_dpds[mask]
    y_lin = yte[mask]
    lr = LinearRegression().fit(x_lin.reshape(-1, 1), y_lin)
    pred = lr.predict(x_lin.reshape(-1, 1))
    lin_r2 = _r2_score(y_lin, pred)
    resid_var = float(np.var(y_lin - pred))
    tot_var = float(np.var(y_lin))
    resid_frac = resid_var / max(tot_var, 1e-20)

    spearman_dpds = pg_corr.get("mean_spearman_neg_dpds")
    spearman_gmag = pg_corr.get("mean_spearman_gradmag")
    closure.update({
        "linear_wss_vs_neg_dpds_r2": _round4(lin_r2),
        "closure_residual_fraction": _round4(resid_frac),
        "pearson_neg_dpds_pooled": _round4(_safe_pearson(y_lin, x_lin)),
        "spearman_neg_dpds_pooled": _round4(_safe_spearman(y_lin, x_lin)),
        "mean_spearman_neg_dpds_per_graph": _round4(
            spearman_dpds if isinstance(spearman_dpds, (int, float)) else None
        ),
        "mean_spearman_gradmag_per_graph": _round4(
            spearman_gmag if isinstance(spearman_gmag, (int, float)) else None
        ),
        "closure_ok_spearman": (
            spearman_dpds is not None
            and math.isfinite(float(spearman_dpds))
            and abs(float(spearman_dpds)) >= CLOSURE_SPEARMAN_MIN
        ),
        "closure_ok_residual": resid_frac <= CLOSURE_RESIDUAL_MAX,
        "closure_ok": (
            (spearman_dpds is not None
             and math.isfinite(float(spearman_dpds))
             and abs(float(spearman_dpds)) >= CLOSURE_SPEARMAN_MIN)
            or resid_frac <= CLOSURE_RESIDUAL_MAX
        ),
        "interpretation": (
            "顺压梯度 -dp/ds 与 WSS 的 Spearman/线性 R²；E-J 主信号为 |∇p| 模长而非 dp/ds"
        ),
    })
    return closure


def _run_probe_suite(
    Xtr: np.ndarray,
    ytr: np.ndarray,
    Xte: np.ndarray,
    yte: np.ndarray,
    test_cases: Sequence[str],
    case_ids_te: Sequence[str],
    *,
    seed: int,
    verbose: bool,
) -> Dict[str, Any]:
    probes: Dict[str, Any] = {}

    for name, cols, feat_names in (
        ("geom_only", list(range(2, 8)), FEATURE_NAMES[2:]),
        ("gradp_mag_geom", GRADP_COLS, [FEATURE_NAMES[0]] + FEATURE_NAMES[2:]),
        ("dpds_geom", DPDS_COLS, [FEATURE_NAMES[1]] + FEATURE_NAMES[2:]),
    ):
        if verbose:
            print(f"  probe {name} ...", flush=True)
        r2_test, pred = _fit_gbdt_probe(Xtr, ytr, Xte, yte, cols, seed=seed)
        case_rows = _per_case_r2(test_cases, yte, pred, case_ids_te) if pred.size else []
        # pooled train+test 上界（oracle 口径）
        X_pool = np.concatenate([Xtr, Xte], axis=0)
        y_pool = np.concatenate([ytr, yte], axis=0)
        r2_pooled, _ = _fit_gbdt_probe(Xtr, ytr, X_pool, y_pool, cols, seed=seed)
        probes[name] = {
            "features": feat_names,
            "n_train": int(Xtr.shape[0]),
            "n_test": int(Xte.shape[0]),
            "test_r2": _round4(r2_test),
            "pooled_r2": _round4(r2_pooled),
            "test_spearman": _round4(_safe_spearman(yte, pred)) if pred.size else None,
            "per_case_test": case_rows,
            "by_family_test": _family_breakdown(case_rows),
        }

    geom_r2 = probes["geom_only"]["test_r2"]
    gradp_r2 = probes["gradp_mag_geom"]["test_r2"]
    dpds_r2 = probes["dpds_geom"]["test_r2"]
    delta_gradp = None
    delta_dpds = None
    if geom_r2 is not None and gradp_r2 is not None:
        delta_gradp = _round4(float(gradp_r2) - float(geom_r2))
    if geom_r2 is not None and dpds_r2 is not None:
        delta_dpds = _round4(float(dpds_r2) - float(geom_r2))

    return {
        "probes": probes,
        "delta_r2_gradp_vs_geom": delta_gradp,
        "delta_r2_dpds_vs_geom": delta_dpds,
        "contrast": (
            "|∇p|+geom 应显著优于 dp/ds+geom；后者仅作 O4 对照弱信号文档"
        ),
    }


def _verdict(
    closure: Mapping[str, Any],
    probe_suite: Mapping[str, Any],
) -> Dict[str, Any]:
    closure_ok = bool(closure.get("closure_ok"))
    delta = probe_suite.get("delta_r2_gradp_vs_geom")
    gradp_test = probe_suite["probes"]["gradp_mag_geom"]["test_r2"]
    gradp_pooled = probe_suite["probes"]["gradp_mag_geom"]["pooled_r2"]

    full_go = (
        closure_ok
        and delta is not None
        and float(delta) >= DELTA_R2_GO_MIN
    )
    weak_go = (
        gradp_pooled is not None
        and float(gradp_pooled) >= WEAK_GO_GRADP_R2_MIN
    )

    if full_go:
        verdict = "go"
    elif weak_go:
        verdict = "weak_go"
    else:
        verdict = "no_go"

    return {
        "verdict": verdict,
        "full_go": full_go,
        "weak_go": weak_go,
        "closure_ok": closure_ok,
        "delta_r2_gradp_vs_geom": delta,
        "gradp_pooled_r2": gradp_pooled,
        "gradp_test_r2": gradp_test,
        "go_thresholds": {
            "closure_spearman_min": CLOSURE_SPEARMAN_MIN,
            "closure_residual_max": CLOSURE_RESIDUAL_MAX,
            "delta_r2_min": DELTA_R2_GO_MIN,
            "weak_go_gradp_pooled_r2_min": WEAK_GO_GRADP_R2_MIN,
        },
        "recommendation": (
            "Go → E-J |∇p| rich 特征进 wss_head（wss_pgrad_context_mode=rich）；"
            "weak_go → 可开 GPU probe 但 dp/ds 闭合弱；"
            "no_go → 封存 E-J GPU"
        ),
        "gpu_probe_submit": verdict in ("go", "weak_go"),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="V3P M-E E-J 压力梯度桥接 oracle")
    ap.add_argument("--split", type=Path, default=REPO_ROOT / "training" / "splits" / "split_AG_v1.json")
    ap.add_argument("--data-root", type=Path, default=REPO_ROOT / "data_new" / "AG")
    ap.add_argument("--graphs-subdir", type=str, default="graphs")
    ap.add_argument("--norm-params", type=Path, default=DEFAULT_NORM_PARAMS)
    ap.add_argument("--max-graphs-per-case", type=int, default=3)
    ap.add_argument("--max-wall-per-graph", type=int, default=1200)
    ap.add_argument("--max-cases", type=int, default=0, help="0=train+test 全例")
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--reference-run-dir", type=Path, default=None,
                    help="可选 I6-diag run 目录（仅记录对照路径，GT oracle 为主）")
    ap.add_argument("--output", type=Path, default=None)
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args()

    split_path = args.split.resolve()
    data_root = args.data_root.resolve()
    stats = _load_norm_stats(args.norm_params.resolve())
    split = SplitSpec.from_json(split_path)
    train_cases = filter_case_names(split.train_cases, data_root)
    test_cases = filter_case_names(split.test_cases, data_root)

    max_train = args.max_cases if args.max_cases > 0 else None
    max_test = args.max_cases if args.max_cases > 0 else None
    ref_run = args.reference_run_dir
    if ref_run is None:
        ref_candidate = REPO_ROOT / DEFAULT_I6DIAG_RUN
        if ref_candidate.is_dir():
            ref_run = ref_candidate

    print(
        f"E-J pressure-gradient oracle: train={len(train_cases)} test={len(test_cases)} "
        f"max_graphs={args.max_graphs_per_case}",
        flush=True,
    )

    t0 = time.perf_counter()

    # --- 压力梯度相关（test 逐图）---
    n_test_for_pg = max_test if max_test is not None else len(test_cases)
    print("closure corr (oracle_pressure_gradient_wss) ...", flush=True)
    pg_corr = oracle_pressure_gradient_wss(
        split_path, stats,
        max_cases=n_test_for_pg,
        max_graphs_per_case=args.max_graphs_per_case,
        max_wall_per_graph=args.max_wall_per_graph,
        verbose=args.verbose,
    )

    # --- 特征收集（带 case id 供 per-case breakdown）---
    print("collect train features ...", flush=True)
    Xtr, ytr, _ = _collect_with_case_ids(
        train_cases, data_root=data_root, graphs_subdir=args.graphs_subdir,
        norm_stats=stats, max_graphs_per_case=args.max_graphs_per_case,
        max_wall_per_graph=args.max_wall_per_graph, seed=args.seed,
        max_cases=max_train, verbose=args.verbose,
    )
    print(f"collect test features ... train n={Xtr.shape[0]}", flush=True)
    Xte, yte, case_ids_te = _collect_with_case_ids(
        test_cases, data_root=data_root, graphs_subdir=args.graphs_subdir,
        norm_stats=stats, max_graphs_per_case=args.max_graphs_per_case,
        max_wall_per_graph=args.max_wall_per_graph, seed=args.seed + 1,
        max_cases=max_test, verbose=args.verbose,
    )

    closure = _closure_audit(Xte, yte, pg_corr)
    print("GBDT probes ...", flush=True)
    probe_suite = _run_probe_suite(
        Xtr, ytr, Xte, yte, test_cases, case_ids_te,
        seed=args.seed, verbose=args.verbose,
    )
    verdict_block = _verdict(closure, probe_suite)

    report: Dict[str, Any] = {
        "label": "V3P-ME-E-J-pressure-gradient-oracle",
        "date": date.today().isoformat(),
        "split": str(split_path),
        "data_root": str(data_root),
        "context": "V3P post5463 band · split_AG_v1 · GT oracle · E-J |∇p| 主信号",
        "reference": {
            "engineering_plan": "V3P_工程化精度优化方案 §5 E-J / §5+.x O4",
            "prior_o4_wave": "v3p_me_oracle_wave_20260626.json (|∇p| GBDT ~0.698)",
            "eb_nearwall": "v3p_me_eb_nearwall_oracle_20260629.json (No-Go → 改 E-J)",
            "i6diag_run": str(ref_run) if ref_run else None,
            "avoid_mode": "5276 direct vel_diff / tang_normal long train",
        },
        "pressure_gradient_corr": pg_corr,
        "closure_audit": closure,
        "probe_suite": probe_suite,
        "verdict": verdict_block["verdict"],
        "full_go": verdict_block["full_go"],
        "weak_go": verdict_block["weak_go"],
        "gpu_probe_submit": verdict_block["gpu_probe_submit"],
        "go_thresholds": verdict_block["go_thresholds"],
        "recommendation": verdict_block["recommendation"],
        "key_metrics": {
            "mean_spearman_gradmag": _round4(pg_corr.get("mean_spearman_gradmag")),
            "mean_spearman_neg_dpds": _round4(pg_corr.get("mean_spearman_neg_dpds")),
            "closure_residual_fraction": closure.get("closure_residual_fraction"),
            "geom_only_test_r2": probe_suite["probes"]["geom_only"]["test_r2"],
            "gradp_mag_geom_test_r2": probe_suite["probes"]["gradp_mag_geom"]["test_r2"],
            "gradp_mag_geom_pooled_r2": probe_suite["probes"]["gradp_mag_geom"]["pooled_r2"],
            "dpds_geom_test_r2": probe_suite["probes"]["dpds_geom"]["test_r2"],
            "delta_r2_gradp_vs_geom": probe_suite["delta_r2_gradp_vs_geom"],
            "delta_r2_dpds_vs_geom": probe_suite["delta_r2_dpds_vs_geom"],
        },
        "elapsed_s": round(time.perf_counter() - t0, 2),
        "interpretation": (
            f"E-J {verdict_block['verdict'].upper()}: "
            f"|∇p|+geom test R²={probe_suite['probes']['gradp_mag_geom']['test_r2']} "
            f"pooled={probe_suite['probes']['gradp_mag_geom']['pooled_r2']} "
            f"Δvs geom={probe_suite['delta_r2_gradp_vs_geom']}; "
            f"dp/ds Spearman={pg_corr.get('mean_spearman_neg_dpds')}"
        ),
    }

    out = args.output or (
        REPO_ROOT / "outputs" / "field" / "f0_decision" / f"v3p_me_ej_oracle_{date.today():%Y%m%d}.json"
    )
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")

    slim = {k: v for k, v in report.items() if k not in ("pressure_gradient_corr",)}
    if "pressure_gradient_corr" in report:
        slim["pressure_gradient_corr"] = {
            k: report["pressure_gradient_corr"].get(k)
            for k in (
                "n_rows", "mean_spearman_neg_dpds", "mean_spearman_gradmag",
                "mean_pearson_neg_dpds", "mean_pearson_gradmag",
            )
        }
    print(json.dumps(slim, indent=2, ensure_ascii=False))
    print(out, flush=True)


if __name__ == "__main__":
    main()

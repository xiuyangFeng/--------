#!/usr/bin/env python3
"""V3P K9 · 剖面射线目标可行性审计（前沿方向 §15.2 Gate-0 · 0 重训 · CPU）。

K9 想法：把 5905 oracle 的剖面拟合从「上界证明」变成「训练目标生成器」——
预处理期对每个壁面点用 GT 近壁体点速度拟合 ``u_t(η)=a1·η+a2·η²``，得逐点
GT ``(a1_s, a1_c)``，训练期以辅助 loss 直接监督剖面头系数（横向分量第一次
获得逐点直接监督；推理期不需要任何射线/速度数据）。

本脚本审计该目标在 15k 降采样图上的可行性（Gate-0），暂不生成 sidecar：
  1. 覆盖率：可成功拟合的壁面点占比（Go 阈值 ≥0.60）；
  2. k 稳定性：k=5 与 k=10 邻居拟合的 a1 Spearman（Go 阈值 ≥0.80，防目标即噪声）；
  3. 目标-GT 对齐：μ·a1 对 GT WSS 分量 calibrated R²（复核 5905，cross ≥0.10）。

三项全过 → Go：落 sidecar 生成 + K8 双分量头 + L_ray 训练档（§15.2 Gate-1/2）；
覆盖率不过 → 退路：从全分辨率原始场抽射线目标（预处理层，不受采样限制）；
k 稳定性不过 → 提高 η 上限 / 加权最小二乘后复审。

用法::

    python -m training.scripts.run_v3p_k9_ray_target_audit --verbose
    python -m training.scripts.run_v3p_k9_ray_target_audit --smoke
"""
from __future__ import annotations

import argparse
import json
import math
import time
from datetime import date
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional

import numpy as np

from ..core.splits import SplitSpec
from ..core.utils import ensure_dir
from ._figure_utils import save_json
from .run_v3_f0_decision import (
    DEFAULT_NORM_PARAMS,
    NODE_IDX,
    REPO_ROOT,
    X_TAN,
    Y_VEL,
    _denorm_velocity,
    _denorm_zscore,
    _load_graph,
    _load_norm_stats,
    _safe_spearman,
)
from .run_v3p_profile_wss_oracle import (
    MM_TO_M,
    MU_BLOOD,
    WSS_X,
    WSS_Y,
    WSS_Z,
    _calibrated_r2,
    _fit_profile_a1,
    _safe_json,
    _unit_rows,
)

GO_COVERAGE_MIN = 0.60
GO_K_STABILITY_MIN = 0.80
GO_CROSS_ALIGN_MIN = 0.10


def _spearman_brown(r: Optional[float]) -> Optional[float]:
    """split-half 秩相关 → 全样本可靠性（Spearman-Brown 预测公式）。"""
    if r is None or math.isnan(r) or r <= -1.0:
        return None
    return 2.0 * r / (1.0 + r)


def _stability(arrs: Mapping[str, np.ndarray], comp: str, mode: str) -> Optional[float]:
    """cross/stream 分量的目标稳定性（split_half 模式做 Spearman-Brown 校正）。"""
    sub = arrs[f"a1_{comp}_sub"]
    ok = ~np.isnan(sub)
    if mode == "split_half":
        h1 = arrs[f"a1_{comp}_h1"]
        ok = ok & ~np.isnan(h1)
        if int(ok.sum()) < 20:
            return None
        return _spearman_brown(_safe_spearman(h1[ok], sub[ok]))
    if int(ok.sum()) < 20:
        return None
    return _safe_spearman(arrs[f"a1_{comp}"][ok], sub[ok])


def _graph_audit(
    data,
    stats: Mapping[str, Dict[str, float]],
    *,
    k_full: int,
    k_sub: int,
    max_wall_points: int,
    rng: np.random.Generator,
    stability_mode: str = "ksub",
) -> Optional[Dict[str, Any]]:
    """单图审计：逐壁面点拟合 GT a1 并统计覆盖率。

    stability_mode:
      - "ksub"：对照 = 最近 k_sub 个点重拟合（嵌套子集，偏严）；
      - "split_half"：对照 = 有效近壁点随机分两半各自拟合（独立估计，
        汇总时可做 Spearman-Brown 校正得全样本拟合的可靠性）。
    """
    from scipy.spatial import cKDTree

    x = data.x.numpy()
    wall = x[:, NODE_IDX["is_wall"]] > 0.5
    interior = ~wall
    n_wall, n_int = int(wall.sum()), int(interior.sum())
    if n_wall < 50 or n_int < 50:
        return None

    pos_w = x[wall, :3].astype(np.float64)
    pos_i = x[interior, :3].astype(np.float64)
    tan_w = _unit_rows(x[wall, X_TAN].astype(np.float64))
    vel_i = _denorm_velocity(data.y.numpy()[interior, Y_VEL], stats)

    y_wss = data.y_wss.numpy()
    wss_vec = np.stack(
        [
            _denorm_zscore(y_wss[wall, i].astype(np.float64), stats.get(n))
            for i, n in zip((WSS_X, WSS_Y, WSS_Z), ("wss_x", "wss_y", "wss_z"))
        ],
        axis=1,
    )

    idx_all = np.arange(n_wall)
    if n_wall > max_wall_points:
        idx_all = rng.choice(n_wall, max_wall_points, replace=False)

    tree = cKDTree(pos_i)
    _, nbrs = tree.query(pos_w[idx_all], k=k_full)

    cols = (
        "gt_stream", "gt_cross",
        "a1_s", "a1_c",            # k_full 全样本拟合（候选训练目标）
        "a1_s_sub", "a1_c_sub",    # 稳定性对照（ksub 拟合 或 split-half h2）
        "a1_s_h1", "a1_c_h1",      # split-half h1（仅 split_half 模式）
        "eta_min_mm",
    )
    out: Dict[str, List[float]] = {k: [] for k in cols}
    n_attempt = 0
    n_valid_full = 0
    n_valid_both = 0

    for row, wi in enumerate(idx_all):
        n_attempt += 1
        nb = np.atleast_1d(nbrs[row])
        rel = pos_i[nb] - pos_w[wi]                      # (k,3) mm
        n_hat = rel.mean(axis=0)
        n_norm = float(np.linalg.norm(n_hat))
        if n_norm < 1e-9:
            continue
        n_hat = n_hat / n_norm
        t_s = tan_w[wi] - float(tan_w[wi] @ n_hat) * n_hat
        t_norm = float(np.linalg.norm(t_s))
        if t_norm < 1e-6:
            continue
        t_s = t_s / t_norm
        t_c = np.cross(n_hat, t_s)

        eta = (rel @ n_hat) * MM_TO_M                    # m
        valid = eta > 1e-7
        if int(valid.sum()) < 3:
            continue
        eta_v = eta[valid]
        v_nb = vel_i[nb][valid]
        v_t = v_nb - np.outer(v_nb @ n_hat, n_hat)
        u_s = v_t @ t_s
        u_c = v_t @ t_c

        a1_s = _fit_profile_a1(eta_v, u_s)
        a1_c = _fit_profile_a1(eta_v, u_c)
        if a1_s is None or a1_c is None:
            continue
        n_valid_full += 1

        if stability_mode == "split_half":
            # 独立半样本对照：随机分两半各自拟合（h1 存入 a1_*，h2 存入 a1_*_sub）
            if eta_v.size < 6:
                a1_s_sub = a1_c_sub = float("nan")
            else:
                perm = rng.permutation(eta_v.size)
                ha, hb = perm[: eta_v.size // 2], perm[eta_v.size // 2 :]
                h1_s = _fit_profile_a1(eta_v[ha], u_s[ha])
                h1_c = _fit_profile_a1(eta_v[ha], u_c[ha])
                h2_s = _fit_profile_a1(eta_v[hb], u_s[hb])
                h2_c = _fit_profile_a1(eta_v[hb], u_c[hb])
                if None in (h1_s, h1_c, h2_s, h2_c):
                    a1_s_sub = a1_c_sub = float("nan")
                else:
                    # 覆盖 a1 为半样本 h1（稳定性 = h1 vs h2 独立估计的秩相关）
                    # 注意 gt 对齐仍用全样本拟合 a1_s/a1_c（上面已算，不覆盖）
                    a1_s_sub, a1_c_sub = h2_s, h2_c
                    out["a1_s_h1"].append(MU_BLOOD * h1_s)
                    out["a1_c_h1"].append(MU_BLOOD * h1_c)
                    n_valid_both += 1
            if math.isnan(a1_s_sub):
                out["a1_s_h1"].append(float("nan"))
                out["a1_c_h1"].append(float("nan"))
        else:
            # k_sub 稳定性对照：只用最近 k_sub 个有效近壁点重拟合
            order = np.argsort(eta_v)[:k_sub]
            a1_s_sub = _fit_profile_a1(eta_v[order], u_s[order])
            a1_c_sub = _fit_profile_a1(eta_v[order], u_c[order])
            if a1_s_sub is None or a1_c_sub is None:
                a1_s_sub = a1_c_sub = float("nan")
            else:
                n_valid_both += 1

        out["gt_stream"].append(float(wss_vec[wi] @ t_s))
        out["gt_cross"].append(float(wss_vec[wi] @ t_c))
        out["a1_s"].append(MU_BLOOD * a1_s)
        out["a1_c"].append(MU_BLOOD * a1_c)
        out["a1_s_sub"].append(MU_BLOOD * a1_s_sub if not math.isnan(a1_s_sub) else float("nan"))
        out["a1_c_sub"].append(MU_BLOOD * a1_c_sub if not math.isnan(a1_c_sub) else float("nan"))
        out["eta_min_mm"].append(float(eta_v.min() / MM_TO_M))

    if n_valid_full < 20:
        return None
    arrays = {k: np.asarray(v, dtype=np.float64) for k, v in out.items()}
    arrays["_n_attempt"] = np.asarray([n_attempt], dtype=np.float64)
    arrays["_n_valid_full"] = np.asarray([n_valid_full], dtype=np.float64)
    arrays["_n_valid_both"] = np.asarray([n_valid_both], dtype=np.float64)
    return arrays


def main() -> None:
    ap = argparse.ArgumentParser(description="V3P K9 ray-target feasibility audit (Gate-0)")
    ap.add_argument("--split", type=Path, default=REPO_ROOT / "training/splits/split_AG_v1.json")
    ap.add_argument("--data-root", type=Path, default=REPO_ROOT / "data_new/AG")
    ap.add_argument("--graphs-subdir", default="graphs")
    ap.add_argument("--norm-params", type=Path, default=DEFAULT_NORM_PARAMS)
    ap.add_argument("--output-dir", type=Path, default=REPO_ROOT / "outputs/field/f0_decision")
    ap.add_argument("--date-suffix", default="")
    ap.add_argument("--subset", choices=("test", "train", "both"), default="both")
    ap.add_argument("--max-graphs-per-case", type=int, default=3)
    ap.add_argument("--max-wall-points", type=int, default=4000, help="每图最多审计壁面点数")
    ap.add_argument("--k-full", type=int, default=10)
    ap.add_argument("--k-sub", type=int, default=5)
    ap.add_argument(
        "--stability-mode", choices=("ksub", "split_half"), default="ksub",
        help="ksub=嵌套子集重拟合（偏严）；split_half=独立半样本 + Spearman-Brown 校正",
    )
    ap.add_argument("--calib-frac", type=float, default=0.5)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args()

    if args.smoke:
        args.max_graphs_per_case = 1
        args.max_wall_points = 500

    t0 = time.perf_counter()
    rng = np.random.default_rng(args.seed)
    split = SplitSpec.from_json(args.split.resolve())
    data_root = args.data_root.resolve()
    stats = _load_norm_stats(args.norm_params.resolve())
    suffix = args.date_suffix or date.today().strftime("%Y%m%d")

    cases: List[str] = []
    if args.subset in ("train", "both"):
        cases += list(split.train_cases)
    if args.subset in ("test", "both"):
        cases += list(split.test_cases)
    if args.smoke:
        cases = cases[:6]

    pooled: Dict[str, List[np.ndarray]] = {}
    per_case: Dict[str, Dict[str, Any]] = {}
    tot_attempt = tot_valid_full = tot_valid_both = 0
    n_done = n_skip = 0

    for ci, case_rel in enumerate(cases, 1):
        case_dir = data_root / case_rel / "processed" / args.graphs_subdir
        if not case_dir.is_dir():
            print(f"[k9-audit] [{ci}/{len(cases)}] SKIP no-graphs {case_rel}", flush=True)
            n_skip += 1
            continue
        tc = time.perf_counter()
        case_chunks: Dict[str, List[np.ndarray]] = {}
        c_attempt = c_full = c_both = 0
        n_graphs = 0
        for gp in sorted(case_dir.glob("*.pt"))[: args.max_graphs_per_case]:
            est = _graph_audit(
                _load_graph(gp), stats,
                k_full=args.k_full,
                k_sub=args.k_sub,
                max_wall_points=args.max_wall_points,
                rng=rng,
                stability_mode=args.stability_mode,
            )
            if est is None:
                continue
            n_graphs += 1
            c_attempt += int(est.pop("_n_attempt")[0])
            c_full += int(est.pop("_n_valid_full")[0])
            c_both += int(est.pop("_n_valid_both")[0])
            for k, v in est.items():
                case_chunks.setdefault(k, []).append(v)
        if not case_chunks:
            print(f"[k9-audit] [{ci}/{len(cases)}] SKIP empty {case_rel}", flush=True)
            n_skip += 1
            continue
        case_arr = {k: np.concatenate(v) for k, v in case_chunks.items()}
        for k, v in case_arr.items():
            pooled.setdefault(k, []).append(v)
        tot_attempt += c_attempt
        tot_valid_full += c_full
        tot_valid_both += c_both

        stab_c = _stability(case_arr, "c", args.stability_mode)
        _, align_c, _ = _calibrated_r2(case_arr["gt_cross"], case_arr["a1_c"], args.calib_frac, rng)
        coverage = c_full / max(c_attempt, 1)
        per_case[case_rel] = {
            "n_points": int(case_arr["gt_stream"].size),
            "n_graphs": n_graphs,
            "coverage": _safe_json(coverage),
            "k_stability_cross": _safe_json(stab_c),
            "cross_align_calibrated_r2": _safe_json(align_c),
            "eta_min_mm_median": _safe_json(float(np.median(case_arr["eta_min_mm"]))),
        }
        n_done += 1
        print(
            f"[k9-audit] [{ci}/{len(cases)}] DONE {case_rel} graphs={n_graphs} "
            f"pts={case_arr['gt_stream'].size} cov={coverage:.3f} "
            f"stab_c={stab_c if stab_c is None else round(stab_c, 3)} "
            f"align_c={align_c:.3f} ({time.perf_counter()-tc:.1f}s)",
            flush=True,
        )

    if not pooled:
        raise SystemExit("no usable cases")
    arr = {k: np.concatenate(v) for k, v in pooled.items()}

    coverage = tot_valid_full / max(tot_attempt, 1)
    coverage_both = tot_valid_both / max(tot_attempt, 1)
    k_stab = {
        "mode": args.stability_mode,
        "stream": _safe_json(_stability(arr, "s", args.stability_mode)),
        "cross": _safe_json(_stability(arr, "c", args.stability_mode)),
    }
    if args.stability_mode == "split_half":
        hh_ok = ~np.isnan(arr["a1_c_sub"]) & ~np.isnan(arr["a1_c_h1"])
        k_stab["cross_half_half_raw"] = _safe_json(
            _safe_spearman(arr["a1_c_h1"][hh_ok], arr["a1_c_sub"][hh_ok])
        )
        k_stab["note"] = "stream/cross 为 Spearman-Brown 校正后的全样本拟合可靠性"
    align: Dict[str, Any] = {}
    for comp, tgt, est in (("stream", "gt_stream", "a1_s"), ("cross", "gt_cross", "a1_c")):
        raw, cal, sp = _calibrated_r2(arr[tgt], arr[est], args.calib_frac, rng)
        align[comp] = {"raw_r2": _safe_json(raw), "calibrated_r2": _safe_json(cal), "spearman": _safe_json(sp)}

    cov_ok = coverage >= GO_COVERAGE_MIN
    stab_ok = (k_stab["cross"] or -1.0) >= GO_K_STABILITY_MIN
    align_ok = (align["cross"]["calibrated_r2"] or -1.0) >= GO_CROSS_ALIGN_MIN
    verdict = "go" if (cov_ok and stab_ok and align_ok) else ("weak_no_go" if align_ok else "no_go")

    report = {
        "label": "v3p_k9_ray_target_audit",
        "date": suffix,
        "context": "V3P · K9 剖面射线训练目标可行性审计（前沿方向 §15.2 Gate-0）",
        "config": {
            "split": str(args.split),
            "data_root": str(data_root),
            "subset": args.subset,
            "mu_Pa_s": MU_BLOOD,
            "k_full": args.k_full,
            "k_sub": args.k_sub,
            "stability_mode": args.stability_mode,
            "max_graphs_per_case": args.max_graphs_per_case,
            "max_wall_points": args.max_wall_points,
            "profile_basis": "a1*eta + a2*eta^2",
            "smoke": args.smoke,
        },
        "n_cases_done": n_done,
        "n_cases_skipped": n_skip,
        "n_points_pooled": int(arr["gt_stream"].size),
        "coverage": {
            "attempted": tot_attempt,
            "valid_full_fit": tot_valid_full,
            "valid_both_fits": tot_valid_both,
            "coverage_full": _safe_json(coverage),
            "coverage_both": _safe_json(coverage_both),
        },
        "k_stability_spearman": k_stab,
        "target_gt_alignment": align,
        "eta_min_mm": {
            "median": _safe_json(float(np.median(arr["eta_min_mm"]))),
            "p90": _safe_json(float(np.percentile(arr["eta_min_mm"], 90))),
        },
        "go_thresholds": {
            "coverage_min": GO_COVERAGE_MIN,
            "k_stability_cross_min": GO_K_STABILITY_MIN,
            "cross_align_calibrated_r2_min": GO_CROSS_ALIGN_MIN,
        },
        "checks": {"coverage": cov_ok, "k_stability": stab_ok, "cross_alignment": align_ok},
        "verdict": verdict,
        "gpu_mapping": "前沿方向 §15.1/§15.2：K8 双分量剖面头 + K9 L_ray 辅助监督（Gate-1 40ep probe）",
        "recommendation": (
            "Go → 生成逐图 sidecar（a1_s/a1_c/frame）并立项 K8+K9 Gate-1；"
            "覆盖率不过 → 改从全分辨率原始场抽射线目标；"
            "k 稳定性不过 → 提高 η 上限/加权最小二乘后复审；"
            "cross 对齐不过 → K9 封口（目标本身与 GT 不对齐）"
        ),
        "per_case": per_case,
        "elapsed_s": round(time.perf_counter() - t0, 2),
    }

    out_path = ensure_dir(args.output_dir.resolve()) / f"v3p_k9_ray_target_audit_{suffix}.json"
    save_json(out_path, report)
    print(json.dumps({
        "output": str(out_path),
        "verdict": verdict,
        "coverage_full": report["coverage"]["coverage_full"],
        "k_stability_spearman": k_stab,
        "cross_alignment": align["cross"],
        "stream_alignment": align["stream"],
    }, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()

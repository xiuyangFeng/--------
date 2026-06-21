#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
WSS 局部坐标系投影（local_v1）。

在 coord_normalize 之前，将 WSS 矢量投影到 axial / circ / rad 标量分量，并写入 fallback 掩码。

默认法向来源：病例根目录 STL 表面网格点法向（snap 最近点）。
可选 kNN 内部点几何估计（调试用，QA 易未过 §6.5 阈值）。
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Dict, List, Literal, Optional, Tuple

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree
from tqdm import tqdm

try:
    from .config import (
        DATA_ROOT,
        FEATURES_DIR,
        WSS_LOCAL_MASK_NAMES,
        get_case_dirs,
    )
    from .export_gap_preprocess_queue import PREPROCESS_DENYLIST
    from .utils.progress import batch_progress_logging, case_progress_logging
    from .case_match import case_dir_matches_query
except ImportError:
    from config import (
        DATA_ROOT,
        FEATURES_DIR,
        WSS_LOCAL_MASK_NAMES,
        get_case_dirs,
    )
    from export_gap_preprocess_queue import PREPROCESS_DENYLIST
    from pipeline.utils.progress import batch_progress_logging, case_progress_logging
    from pipeline.case_match import case_dir_matches_query

K_INTERNAL_DEFAULT = 5
BASIS_PARALLEL_COS = 0.95
NormalSource = Literal["stl", "knn"]


def _normalize_rows(vectors: np.ndarray, eps: float = 1e-10) -> Tuple[np.ndarray, np.ndarray]:
    norms = np.linalg.norm(vectors, axis=1)
    safe = np.where(norms > eps, norms, 1.0)
    unit = vectors / safe[:, np.newaxis]
    valid = norms > eps
    return unit, valid


def find_case_stl(case_dir: Path) -> Optional[Path]:
    """查找病例 STL：优先 `{case_name}.stl`，否则唯一 *.stl。"""
    named = case_dir / f"{case_dir.name}.stl"
    if named.is_file():
        return named
    stls = sorted(case_dir.glob("*.stl"))
    if not stls:
        return None
    for p in stls:
        if p.stem == case_dir.name:
            return p
    return stls[0] if len(stls) == 1 else None


class StlNormalLookup:
    """缓存 STL 表面点法向 + 最近点定位器。"""

    def __init__(self, stl_path: Path):
        import vtk
        from vtk.util.numpy_support import vtk_to_numpy

        reader = vtk.vtkSTLReader()
        reader.SetFileName(str(stl_path))
        reader.Update()
        poly = reader.GetOutput()
        if poly is None or poly.GetNumberOfPoints() == 0:
            raise ValueError(f"STL 无有效网格: {stl_path}")

        norm_filter = vtk.vtkPolyDataNormals()
        norm_filter.SetInputData(poly)
        norm_filter.ComputePointNormalsOn()
        norm_filter.SplittingOff()
        norm_filter.Update()
        surf = norm_filter.GetOutput()

        self.stl_path = stl_path
        self.normals = vtk_to_numpy(surf.GetPointData().GetNormals()).astype(np.float64)
        nn = np.linalg.norm(self.normals, axis=1, keepdims=True)
        self.normals = np.divide(
            self.normals,
            np.where(nn > 1e-12, nn, 1.0),
        )

        self._locator = vtk.vtkKdTreePointLocator()
        self._locator.SetDataSet(surf)
        self._locator.BuildLocator()

    def normals_at(self, coords: np.ndarray, wall_mask: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        n = coords.shape[0]
        normals = np.zeros((n, 3), dtype=np.float64)
        normal_valid = np.zeros(n, dtype=np.float32)
        wall_indices = np.where(wall_mask)[0]

        for wi in wall_indices:
            cid = int(self._locator.FindClosestPoint(coords[wi]))
            n_hat = self.normals[cid]
            nn = float(np.linalg.norm(n_hat))
            if nn > 1e-10:
                normals[wi] = n_hat / nn
                normal_valid[wi] = 1.0

        return normals, normal_valid


def estimate_wall_normals_kdtree(
    coords: np.ndarray,
    is_wall: np.ndarray,
    k_internal: int = K_INTERNAL_DEFAULT,
) -> Tuple[np.ndarray, np.ndarray]:
    """对每个壁面点，在内部点 KDTree 中找 k 近邻估计外法向。"""
    n = coords.shape[0]
    normals = np.zeros((n, 3), dtype=np.float64)
    normal_valid = np.zeros(n, dtype=np.float32)

    internal_mask = is_wall.astype(bool) == False  # noqa: E712
    wall_mask = is_wall.astype(bool) == True  # noqa: E712
    internal_coords = coords[internal_mask]
    if internal_coords.shape[0] == 0:
        return normals, normal_valid

    tree = cKDTree(internal_coords)
    wall_indices = np.where(wall_mask)[0]
    k_query = min(k_internal, internal_coords.shape[0])

    for wi in wall_indices:
        _, idxs = tree.query(coords[wi], k=k_query)
        if np.isscalar(idxs):
            idxs = np.array([idxs], dtype=int)
        neighbors = internal_coords[idxs]
        direction = coords[wi] - neighbors.mean(axis=0)
        norm = float(np.linalg.norm(direction))
        if norm > 1e-10:
            normals[wi] = direction / norm
            normal_valid[wi] = 1.0

    return normals, normal_valid


def estimate_wall_normals(
    coords: np.ndarray,
    is_wall: np.ndarray,
    normal_source: NormalSource = "stl",
    stl_lookup: Optional[StlNormalLookup] = None,
    k_internal: int = K_INTERNAL_DEFAULT,
) -> Tuple[np.ndarray, np.ndarray]:
    wall_mask = is_wall.astype(bool)
    if normal_source == "stl":
        if stl_lookup is None:
            raise ValueError("normal_source=stl 需要提供 StlNormalLookup")
        return stl_lookup.normals_at(coords, wall_mask)
    return estimate_wall_normals_kdtree(coords, is_wall, k_internal=k_internal)


def project_wss_to_local_frame(
    df: pd.DataFrame,
    normal_source: NormalSource = "stl",
    stl_lookup: Optional[StlNormalLookup] = None,
    k_internal: int = K_INTERNAL_DEFAULT,
) -> pd.DataFrame:
    """对单个 DataFrame 计算 local WSS 分量与掩码。"""
    required = ["x", "y", "z", "is_wall", "Tangent_X", "Tangent_Y", "Tangent_Z", "wss_x", "wss_y", "wss_z", "wss"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"缺少必要列: {missing}")

    out = df.copy()
    coords = out[["x", "y", "z"]].values.astype(np.float64)
    is_wall = out["is_wall"].values.astype(np.int32)
    wss_vec = out[["wss_x", "wss_y", "wss_z"]].values.astype(np.float64)
    tangent = out[["Tangent_X", "Tangent_Y", "Tangent_Z"]].values.astype(np.float64)

    normals, normal_valid = estimate_wall_normals(
        coords,
        is_wall,
        normal_source=normal_source,
        stl_lookup=stl_lookup,
        k_internal=k_internal,
    )
    t_hat, t_valid = _normalize_rows(tangent)

    n_pts = coords.shape[0]
    basis_valid = np.zeros(n_pts, dtype=np.float32)
    wss_axial = np.full(n_pts, np.nan, dtype=np.float64)
    wss_circ = np.full(n_pts, np.nan, dtype=np.float64)
    wss_rad = np.full(n_pts, np.nan, dtype=np.float64)

    for i in range(n_pts):
        if not (is_wall[i] == 1 and normal_valid[i] > 0.5 and t_valid[i]):
            continue
        n_hat = normals[i]
        t_i = t_hat[i]
        b_raw = np.cross(n_hat, t_i)
        b_norm = float(np.linalg.norm(b_raw))
        cos_parallel = abs(float(np.dot(n_hat, t_i)))
        if b_norm <= 1e-10 or cos_parallel >= BASIS_PARALLEL_COS:
            basis_valid[i] = 0.0
            wss_axial[i] = float(np.dot(wss_vec[i], t_i))
            continue
        b_hat = b_raw / b_norm
        basis_valid[i] = 1.0
        wss_axial[i] = float(np.dot(wss_vec[i], t_i))
        wss_circ[i] = float(np.dot(wss_vec[i], b_hat))
        wss_rad[i] = float(np.dot(wss_vec[i], n_hat))

    out["wss_axial"] = wss_axial
    out["wss_circ"] = wss_circ
    out["wss_rad"] = wss_rad
    out[WSS_LOCAL_MASK_NAMES[0]] = normal_valid
    out[WSS_LOCAL_MASK_NAMES[1]] = basis_valid
    return out


def _percentiles(arr: np.ndarray, ps=(25, 50, 75, 95)) -> Dict[str, float]:
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return {f"p{p}": float("nan") for p in ps}
    qs = np.percentile(arr, ps)
    return {f"p{p}": float(v) for p, v in zip(ps, qs)}


def compute_qa_stats(
    df: pd.DataFrame,
    case_name: str = "",
    normal_source: NormalSource = "stl",
) -> Dict:
    """单 case / 单文件 QA 统计（壁面点）。"""
    wall = df["is_wall"].astype(bool).values
    nv = df[WSS_LOCAL_MASK_NAMES[0]].values[wall]
    bv = df[WSS_LOCAL_MASK_NAMES[1]].values[wall]
    wss = df["wss"].values[wall].astype(np.float64)
    wss_rad = df["wss_rad"].values[wall].astype(np.float64)

    n_wall = int(wall.sum())
    normal_invalid_rate = float(1.0 - nv.mean()) if n_wall else float("nan")
    basis_invalid_rate = float(1.0 - bv.mean()) if n_wall else float("nan")

    wss_p95 = float(np.nanpercentile(wss, 95)) if n_wall else float("nan")
    rad_p95 = float(np.nanpercentile(np.abs(wss_rad), 95)) if n_wall else float("nan")
    rad_ratio = rad_p95 / wss_p95 if wss_p95 > 1e-12 else float("nan")

    stats = {
        "case_name": case_name,
        "normal_source": normal_source,
        "n_wall": n_wall,
        "normal_invalid_rate": normal_invalid_rate,
        "basis_invalid_rate": basis_invalid_rate,
        "wss_rad_p95_over_wss_p95": rad_ratio,
        "wss_rad_abs": _percentiles(np.abs(wss_rad)),
        "wss_magnitude": _percentiles(wss),
        "wss_axial": _percentiles(df["wss_axial"].values[wall]),
        "wss_circ": _percentiles(df["wss_circ"].values[wall]),
        "mask_histogram": {
            "normal_valid_1": int((nv > 0.5).sum()),
            "normal_valid_0": int((nv <= 0.5).sum()),
            "basis_valid_1": int((bv > 0.5).sum()),
            "basis_valid_0": int((bv <= 0.5).sum()),
        },
        "thresholds": {
            "normal_invalid_rate_max": 0.05,
            "basis_invalid_rate_max": 0.10,
            "wss_rad_p95_over_wss_p95_max": 0.30,
        },
        "pass": bool(
            normal_invalid_rate <= 0.05
            and basis_invalid_rate <= 0.10
            and (not np.isfinite(rad_ratio) or rad_ratio <= 0.30)
        ),
    }
    return stats


def save_qa_json(stats: Dict, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2, ensure_ascii=False)


def _case_rel_path(case_dir: Path, data_root: Path) -> str:
    return case_dir.resolve().relative_to(data_root.resolve()).as_posix()


def _denylist_hit(rel: str, data_root: Path) -> bool:
    """匹配 PREPROCESS_DENYLIST（相对 data_new 根）。"""
    if rel in PREPROCESS_DENYLIST:
        return True
    # data_root=data_new/AG 时 rel=fast/XXX，denylist 键为 AG/fast/XXX
    prefixed = f"{data_root.name}/{rel}"
    return prefixed in PREPROCESS_DENYLIST


def filter_case_dirs(
    case_dirs: List[Path],
    data_root: Path,
    skip_denylist: bool,
    require_features: bool,
) -> List[Path]:
    kept: List[Path] = []
    for case_dir in case_dirs:
        rel = _case_rel_path(case_dir, data_root)
        if skip_denylist and _denylist_hit(rel, data_root):
            print(f"  ⏭ 跳过 denylist: {rel}")
            continue
        if require_features:
            feat_dir = case_dir / FEATURES_DIR
            if not feat_dir.is_dir() or not list(feat_dir.glob("result_features_*.csv")):
                print(f"  ⏭ 跳过无 features: {rel}")
                continue
        kept.append(case_dir)
    return kept


def process_dataframe(
    df: pd.DataFrame,
    normal_source: NormalSource = "stl",
    stl_lookup: Optional[StlNormalLookup] = None,
    k_internal: int = K_INTERNAL_DEFAULT,
) -> pd.DataFrame:
    return project_wss_to_local_frame(
        df,
        normal_source=normal_source,
        stl_lookup=stl_lookup,
        k_internal=k_internal,
    )


def process_case(
    case_dir: Path,
    input_subdir: str,
    output_subdir: str,
    normal_source: NormalSource = "stl",
    k_internal: int = K_INTERNAL_DEFAULT,
    write_qa: bool = True,
) -> bool:
    with case_progress_logging(case_dir, "step_local_wss") as log_path:
        print(f"📝 进度日志: {log_path}")
        input_dir = case_dir / input_subdir
        output_dir = case_dir / output_subdir
        if not input_dir.exists():
            print(f"  ❌ 输入目录不存在: {input_subdir}")
            return False

        csv_files = sorted(input_dir.glob("result_features_*.csv"))
        if not csv_files:
            print(f"  ❌ 未找到特征文件")
            return False

        stl_lookup: Optional[StlNormalLookup] = None
        if normal_source == "stl":
            stl_path = find_case_stl(case_dir)
            if stl_path is None:
                print(f"  ❌ 未找到 STL: {case_dir.name}")
                return False
            print(f"  📐 STL 法向: {stl_path.name}")
            stl_lookup = StlNormalLookup(stl_path)

        output_dir.mkdir(parents=True, exist_ok=True)
        qa_rows: List[Dict] = []
        ok = 0
        for csv_file in tqdm(csv_files, desc=f"local_wss {case_dir.name}", leave=False):
            try:
                df = pd.read_csv(csv_file)
                df_out = project_wss_to_local_frame(
                    df,
                    normal_source=normal_source,
                    stl_lookup=stl_lookup,
                    k_internal=k_internal,
                )
                out_path = output_dir / csv_file.name
                df_out.to_csv(out_path, index=False)
                qa_rows.append(
                    compute_qa_stats(df_out, case_name=case_dir.name, normal_source=normal_source)
                )
                ok += 1
            except Exception as exc:
                print(f"  ❌ {csv_file.name}: {exc}")

        if write_qa and qa_rows:
            agg = _aggregate_qa(qa_rows, case_dir.name, normal_source=normal_source)
            save_qa_json(agg, case_dir / f"local_wss_qa_{case_dir.name}.json")

        print(f"  ✅ 完成: {ok}/{len(csv_files)}")
        return ok > 0


def _aggregate_qa(rows: List[Dict], case_name: str, normal_source: NormalSource = "stl") -> Dict:
    if len(rows) == 1:
        return rows[0]
    n_wall = sum(int(r.get("n_wall", 0)) for r in rows)
    nv0 = sum(int(r.get("mask_histogram", {}).get("normal_valid_0", 0)) for r in rows)
    bv0 = sum(int(r.get("mask_histogram", {}).get("basis_valid_0", 0)) for r in rows)
    normal_invalid = nv0 / n_wall if n_wall else float("nan")
    basis_invalid = bv0 / n_wall if n_wall else float("nan")
    rad_ratios = [
        r.get("wss_rad_p95_over_wss_p95")
        for r in rows
        if np.isfinite(r.get("wss_rad_p95_over_wss_p95", float("nan")))
    ]
    rad_ratio = float(np.mean(rad_ratios)) if rad_ratios else float("nan")
    return {
        "case_name": case_name,
        "normal_source": normal_source,
        "n_files": len(rows),
        "n_wall": n_wall,
        "normal_invalid_rate": normal_invalid,
        "basis_invalid_rate": basis_invalid,
        "wss_rad_p95_over_wss_p95": rad_ratio,
        "per_file": rows,
        "pass": bool(
            normal_invalid <= 0.05
            and basis_invalid <= 0.10
            and (not np.isfinite(rad_ratio) or rad_ratio <= 0.30)
        ),
    }


def run_prevalidation(
    case_dirs: List[Path],
    input_subdir: str = FEATURES_DIR,
    normal_source: NormalSource = "stl",
    k_internal: int = K_INTERNAL_DEFAULT,
    qa_output_dir: Optional[Path] = None,
) -> Tuple[List[Dict], bool]:
    """§6.5 预验证：不落盘 CSV，仅 QA JSON。"""
    qa_output_dir = qa_output_dir or Path("pipeline/qa/local_wss_prevalidation")
    qa_output_dir.mkdir(parents=True, exist_ok=True)
    all_stats: List[Dict] = []
    all_pass = True

    for case_dir in case_dirs:
        input_dir = case_dir / input_subdir
        csv_files = sorted(input_dir.glob("result_features_*.csv"))
        if not csv_files:
            print(f"⚠️  {case_dir.name}: 无 CSV，跳过")
            continue

        stl_lookup: Optional[StlNormalLookup] = None
        if normal_source == "stl":
            stl_path = find_case_stl(case_dir)
            if stl_path is None:
                print(f"⚠️  {case_dir.name}: 无 STL，跳过")
                all_pass = False
                continue
            stl_lookup = StlNormalLookup(stl_path)

        csv_file = csv_files[len(csv_files) // 2]
        df = pd.read_csv(csv_file)
        df_out = project_wss_to_local_frame(
            df,
            normal_source=normal_source,
            stl_lookup=stl_lookup,
            k_internal=k_internal,
        )
        stats = compute_qa_stats(df_out, case_name=str(case_dir.name), normal_source=normal_source)
        stats["sample_file"] = csv_file.name
        if normal_source == "stl" and stl_lookup is not None:
            stats["stl_path"] = str(stl_lookup.stl_path)
        all_stats.append(stats)
        qa_path = qa_output_dir / f"local_wss_qa_{case_dir.name}.json"
        save_qa_json(stats, qa_path)
        print(
            f"  {case_dir.name}: pass={stats['pass']} "
            f"normal_inv={stats['normal_invalid_rate']:.4f} "
            f"basis_inv={stats['basis_invalid_rate']:.4f} "
            f"rad_ratio={stats['wss_rad_p95_over_wss_p95']:.4f}"
        )
        all_pass = all_pass and stats["pass"]

    summary = {
        "n_cases": len(all_stats),
        "all_pass": all_pass,
        "normal_source": normal_source,
        "cases": all_stats,
    }
    save_qa_json(summary, qa_output_dir / "local_wss_prevalidation_summary.json")
    return all_stats, all_pass


def process_all_cases(
    data_root: Path = None,
    target_case: Optional[str] = None,
    input_subdir: str = None,
    output_subdir: str = None,
    normal_source: NormalSource = "stl",
    k_internal: int = K_INTERNAL_DEFAULT,
    sources: Optional[List[str]] = None,
    skip_denylist: bool = True,
    require_features: bool = True,
) -> None:
    if data_root is None:
        data_root = DATA_ROOT
    else:
        data_root = Path(data_root)
    if input_subdir is None:
        input_subdir = FEATURES_DIR
    if output_subdir is None:
        output_subdir = FEATURES_DIR

    case_dirs = get_case_dirs(data_root, sources=sources)
    if target_case:
        case_dirs = [d for d in case_dirs if case_dir_matches_query(d, data_root, target_case)]
    case_dirs = filter_case_dirs(case_dirs, data_root, skip_denylist, require_features)
    if not case_dirs:
        print("❌ 未找到病例")
        return

    with batch_progress_logging(data_root, "step_local_wss_batch.log", "step_local_wss_batch") as log_path:
        print(f"📝 批量日志: {log_path}")
        print(f"📐 法向来源: {normal_source} | 病例数: {len(case_dirs)}")
        t0 = time.time()
        ok = 0
        for case_dir in case_dirs:
            if process_case(
                case_dir,
                input_subdir,
                output_subdir,
                normal_source=normal_source,
                k_internal=k_internal,
            ):
                ok += 1
        print(f"🎉 local_wss 完成: {ok}/{len(case_dirs)} ({time.time()-t0:.1f}s)")


def main() -> None:
    parser = argparse.ArgumentParser(description="WSS 局部坐标系投影 (local_v1)")
    parser.add_argument("--data-root", type=str, default=None)
    parser.add_argument("--case", type=str, default=None)
    parser.add_argument("--input-subdir", type=str, default=FEATURES_DIR)
    parser.add_argument("--output-subdir", type=str, default=FEATURES_DIR)
    parser.add_argument(
        "--normal-source",
        type=str,
        choices=("stl", "knn"),
        default="stl",
        help="壁面法向来源：stl（默认，CFD 表面网格）或 knn（内部点几何估计）",
    )
    parser.add_argument("--k-internal", type=int, default=K_INTERNAL_DEFAULT)
    parser.add_argument("--prevalidate", action="store_true", help="§6.5 预验证模式（不写 CSV）")
    parser.add_argument("--sources", nargs="+", default=None)
    parser.add_argument(
        "--no-skip-denylist",
        action="store_true",
        help="不跳过 PREPROCESS_DENYLIST 中的病例",
    )
    parser.add_argument(
        "--allow-missing-features",
        action="store_true",
        help="允许无 processed/features 的病例入队（默认跳过）",
    )
    args = parser.parse_args()

    data_root = Path(args.data_root) if args.data_root else DATA_ROOT
    normal_source: NormalSource = args.normal_source  # type: ignore[assignment]
    skip_denylist = not args.no_skip_denylist
    require_features = not args.allow_missing_features

    if args.prevalidate:
        case_dirs = get_case_dirs(data_root, sources=args.sources)
        if args.case:
            case_dirs = [d for d in case_dirs if case_dir_matches_query(d, data_root, args.case)]
        case_dirs = filter_case_dirs(case_dirs, data_root, skip_denylist, require_features)
        if not case_dirs:
            print("❌ 预验证：未找到病例")
            sys.exit(1)
        case_dirs = case_dirs[:3] if len(case_dirs) > 3 and not args.case else case_dirs
        _, all_pass = run_prevalidation(
            case_dirs,
            input_subdir=args.input_subdir,
            normal_source=normal_source,
            k_internal=args.k_internal,
        )
        sys.exit(0 if all_pass else 2)

    process_all_cases(
        data_root=data_root,
        target_case=args.case,
        input_subdir=args.input_subdir,
        output_subdir=args.output_subdir,
        normal_source=normal_source,
        k_internal=args.k_internal,
        sources=args.sources,
        skip_denylist=skip_denylist,
        require_features=require_features,
    )


if __name__ == "__main__":
    main()

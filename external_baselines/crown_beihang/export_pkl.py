from __future__ import annotations

import argparse
import json
import pickle
import re
import statistics
import time
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence

import numpy as np
import pandas as pd
from tqdm import tqdm

from pipeline.config import GLOBAL_COND_NAMES, NODE_FEATURE_NAMES, TARGET_NAMES
from training.core.splits import SplitSpec

from .raw_io import list_raw_frames, load_raw_frame
from .utils import append_jsonl, default_private_preprocessed_root, dump_json, load_config, project_root
from .voxelize import coord_span_mm, parse_time_index, voxelize_and_average

GEOM_COLUMNS = [c for c in NODE_FEATURE_NAMES if c not in ("x", "y", "z")]
WALL_VEL_THRESHOLD = 0.01


def _normalize_point_filter(point_filter: str) -> str:
    if point_filter == "interior":
        return "volume"
    return point_filter


def _safe_case_id(case_name: str) -> str:
    return re.sub(r"[^\w.-]+", "_", case_name.replace("/", "__"))


def _load_bc_metadata(features_dir: Path) -> Dict[str, List[float]]:
    path = features_dir / "bc_metadata.json"
    if not path.exists():
        return {}
    with open(path, "r", encoding="utf-8") as f:
        payload = json.load(f)
    if isinstance(payload, dict) and "data" in payload:
        return payload["data"]
    return payload


def _global_cond_from_bc(bc_map: Mapping[str, Sequence[float]], stem: str) -> np.ndarray:
    if stem not in bc_map:
        return np.zeros(len(GLOBAL_COND_NAMES), dtype=np.float32)
    raw = list(bc_map[stem])
    if len(raw) == 5:
        values = [0.0] + [float(v) for v in raw]
    elif len(raw) == 6:
        values = [float(v) for v in raw]
    else:
        return np.zeros(len(GLOBAL_COND_NAMES), dtype=np.float32)
    return np.asarray(values, dtype=np.float32)


def _velocity_wall_count(targets: np.ndarray) -> int:
    u, v, w = targets[:, 0], targets[:, 1], targets[:, 2]
    return int(np.sum(u * u + v * v + w * w <= WALL_VEL_THRESHOLD))


def _voxelize_record(
    points: np.ndarray,
    features: np.ndarray,
    targets: np.ndarray,
    voxel_size_mm: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    dummy_wall = np.zeros(len(points), dtype=np.float32)
    centers, voxel_features, _ = voxelize_and_average(points, features, dummy_wall, voxel_size_mm)
    _, voxel_targets, _ = voxelize_and_average(points, targets, dummy_wall, voxel_size_mm)
    return centers, voxel_features, voxel_targets


def process_raw_frame(
    case_dir: Path,
    frame_key: str,
    case_name: str,
    split: str,
    point_filter: str,
    voxel_size_mm: float,
    max_raw_points: int | None,
    seed: int,
    bc_map: Mapping[str, List[float]],
    log_path: Path | None,
) -> Dict[str, Any]:
    pf = _normalize_point_filter(point_filter)
    start = time.time()
    log_lines: List[str] = []

    def log(msg: str) -> None:
        line = f"[{case_name}] {frame_key} | {msg}"
        print(line, flush=True)
        log_lines.append(line)

    try:
        points, targets = load_raw_frame(case_dir, frame_key, pf)
        n_raw = int(points.shape[0])
        span = coord_span_mm(points)
        log(f"source=raw_ascii point_filter={pf} n_raw={n_raw} coord_span_mm={span}")

        if max_raw_points is not None and n_raw > max_raw_points:
            rng = np.random.default_rng(seed)
            idx = rng.choice(n_raw, size=max_raw_points, replace=False)
            points = points[idx]
            targets = targets[idx]
            n_raw = int(points.shape[0])
            log(f"pre_voxel_subsample n_raw={n_raw}")

        features = points.copy()
        feature_names = ["x", "y", "z"]
        centers, voxel_features, voxel_targets = _voxelize_record(
            points, features, targets, voxel_size_mm
        )
        n_voxel = int(centers.shape[0])
        n_wall_voxel = _velocity_wall_count(voxel_targets)
        n_interior_voxel = n_voxel - n_wall_voxel
        log(
            f"voxel_size_mm={voxel_size_mm} n_voxel={n_voxel} "
            f"n_wall_vel={n_wall_voxel} n_interior_vel={n_interior_voxel}"
        )

        sample_id = f"{case_name}/{frame_key}"
        record = {
            "sample_id": sample_id,
            "case_name": case_name,
            "time_index": parse_time_index(frame_key),
            "split": split,
            "point_filter": pf,
            "export_source": "raw_ascii",
            "feature_names": feature_names,
            "target_names": list(TARGET_NAMES),
            "features": voxel_features.T.astype(np.float32),
            "targets": voxel_targets.T.astype(np.float32),
            "global_cond": _global_cond_from_bc(bc_map, frame_key),
        }
        status = "ok"
        error = ""
    except Exception as exc:
        record = None
        status = "error"
        error = str(exc)
        n_raw = 0
        n_voxel = 0
        n_wall_voxel = 0
        n_interior_voxel = 0
        span = [0.0, 0.0, 0.0]
        log(f"ERROR: {exc}")

    duration = time.time() - start
    if log_path is not None:
        log_path.parent.mkdir(parents=True, exist_ok=True)
        with open(log_path, "a", encoding="utf-8") as lf:
            lf.write("\n".join(log_lines) + "\n")

    return {
        "case_name": case_name,
        "frame_key": frame_key,
        "split": split,
        "csv_path": str(case_dir / "ascii_in"),
        "point_filter": pf,
        "export_source": "raw_ascii",
        "n_raw": n_raw,
        "n_voxel": n_voxel,
        "n_wall_voxel": n_wall_voxel,
        "n_interior_voxel": n_interior_voxel if status == "ok" else 0,
        "voxel_size_mm": voxel_size_mm,
        "coord_span_mm": span,
        "duration_sec": round(duration, 3),
        "status": status,
        "error": error,
        "record": record,
    }


def process_features_csv(
    csv_path: Path,
    case_name: str,
    split: str,
    point_filter: str,
    voxel_size_mm: float,
    max_raw_points: int | None,
    seed: int,
    bc_map: Mapping[str, List[float]],
    log_path: Path | None,
) -> Dict[str, Any]:
    """保留：从 pipeline features 导出（含几何列），仅用于后续几何消融，非论文默认口径。"""
    pf = _normalize_point_filter(point_filter)
    stem = csv_path.stem.replace("result_features_", "")
    start = time.time()
    log_lines: List[str] = []

    def log(msg: str) -> None:
        line = f"[{case_name}] {stem} | {msg}"
        print(line, flush=True)
        log_lines.append(line)

    try:
        df = pd.read_csv(csv_path)
        required = ["x", "y", "z", "u", "v", "w", "p"]
        missing = [c for c in required if c not in df.columns]
        if missing:
            raise KeyError(f"缺少列 {missing}")

        geom_cols = [c for c in GEOM_COLUMNS if c in df.columns]
        feature_names = ["x", "y", "z"] + geom_cols
        points = df[["x", "y", "z"]].to_numpy(dtype=np.float64)
        targets = df[["u", "v", "w", "p"]].to_numpy(dtype=np.float64)
        if geom_cols:
            features = np.concatenate([points, df[geom_cols].to_numpy(dtype=np.float64)], axis=1)
        else:
            features = points.copy()

        if pf == "volume" and "is_wall" in df.columns:
            interior = ~df["is_wall"].astype(bool).to_numpy()
            points, features, targets = points[interior], features[interior], targets[interior]

        n_raw = int(points.shape[0])
        if max_raw_points is not None and n_raw > max_raw_points:
            rng = np.random.default_rng(seed)
            idx = rng.choice(n_raw, size=max_raw_points, replace=False)
            points, features, targets = points[idx], features[idx], targets[idx]
            n_raw = int(points.shape[0])

        span = coord_span_mm(points)
        log(f"source=features point_filter={pf} n_raw={n_raw} (pipeline 已降采样)")
        _, voxel_features, voxel_targets = _voxelize_record(points, features, targets, voxel_size_mm)
        n_voxel = int(voxel_features.shape[0])
        n_wall_voxel = _velocity_wall_count(voxel_targets)

        record = {
            "sample_id": f"{case_name}/{stem}",
            "case_name": case_name,
            "time_index": parse_time_index(stem),
            "split": split,
            "point_filter": pf,
            "export_source": "features",
            "feature_names": feature_names,
            "target_names": list(TARGET_NAMES),
            "features": voxel_features.T.astype(np.float32),
            "targets": voxel_targets.T.astype(np.float32),
            "global_cond": _global_cond_from_bc(bc_map, stem),
        }
        status = "ok"
        error = ""
    except Exception as exc:
        record = None
        status = "error"
        error = str(exc)
        n_raw = n_voxel = n_wall_voxel = 0
        span = [0.0, 0.0, 0.0]
        log(f"ERROR: {exc}")

    duration = time.time() - start
    if log_path is not None:
        log_path.parent.mkdir(parents=True, exist_ok=True)
        with open(log_path, "a", encoding="utf-8") as lf:
            lf.write("\n".join(log_lines) + "\n")

    return {
        "case_name": case_name,
        "frame_key": stem,
        "split": split,
        "csv_path": str(csv_path),
        "point_filter": pf,
        "export_source": "features",
        "n_raw": n_raw,
        "n_voxel": n_voxel,
        "n_wall_voxel": n_wall_voxel,
        "voxel_size_mm": voxel_size_mm,
        "coord_span_mm": span,
        "duration_sec": round(duration, 3),
        "status": status,
        "error": error,
        "record": record,
    }


def export_split_pkl(records: List[Dict[str, Any]], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "wb") as f:
        pickle.dump(records, f)


def compute_p_stats(train_records: List[Dict[str, Any]]) -> Dict[str, float]:
    p_vals = [rec["targets"][3] for rec in train_records]
    p_all = np.concatenate(p_vals)
    return {"p_min": float(p_all.min()), "p_max": float(p_all.max())}


def build_case_split_map(split: SplitSpec) -> Dict[str, str]:
    mapping: Dict[str, str] = {}
    for split_name, cases in (
        ("train", split.train_cases),
        ("val", split.val_cases),
        ("test", split.test_cases),
    ):
        for case_name in cases:
            mapping[case_name] = split_name
    return mapping


def load_cases_file(path: Path) -> List[str]:
    lines = path.read_text(encoding="utf-8").splitlines()
    return [ln.strip() for ln in lines if ln.strip() and not ln.strip().startswith("#")]


def write_all_cases_file(split: SplitSpec, path: Path) -> List[str]:
    cases: List[str] = []
    for cases_list in (split.train_cases, split.val_cases, split.test_cases):
        cases.extend(cases_list)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(cases) + "\n", encoding="utf-8")
    return cases


def _timing_summary(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    ok_rows = [r for r in rows if r.get("status") == "ok"]
    durations = [float(r["duration_sec"]) for r in ok_rows]
    if not durations:
        return {"n_frames": len(rows), "n_ok": 0}
    sorted_d = sorted(durations)
    n = len(sorted_d)
    return {
        "n_frames": len(rows),
        "n_ok": len(ok_rows),
        "total_sec": round(sum(durations), 3),
        "mean_sec": round(statistics.mean(durations), 3),
        "median_sec": round(statistics.median(durations), 3),
        "p90_sec": round(sorted_d[int(0.9 * (n - 1))], 3),
        "max_sec": round(max(durations), 3),
        "mean_n_raw": round(statistics.mean(r["n_raw"] for r in ok_rows), 1),
        "mean_n_voxel": round(statistics.mean(r["n_voxel"] for r in ok_rows), 1),
    }


def export_single_case(
    case_name: str,
    split_name: str,
    config: Dict[str, Any],
    *,
    write_partial: bool = False,
    jsonl_path: Path | None = None,
) -> Dict[str, Any]:
    data_cfg = config["data"]
    export_cfg = config.get("export", {})
    root = project_root()
    data_root = Path(data_cfg["data_root"])
    if not data_root.is_absolute():
        data_root = root / data_root

    export_source = export_cfg.get("source", "raw_ascii")
    features_subdir = data_cfg.get("features_subdir", "processed/features")
    output_root = Path(data_cfg.get("output_root", str(default_private_preprocessed_root())))
    if not output_root.is_absolute():
        output_root = root / output_root

    point_filters = [_normalize_point_filter(pf) for pf in export_cfg.get("point_filters", ["volume", "all"])]
    point_filters = list(dict.fromkeys(point_filters))
    voxel_size_mm = float(export_cfg.get("voxel_size_mm", 0.05))
    max_raw_points = export_cfg.get("max_raw_points")
    seed = int(config.get("system", {}).get("seed", 1))

    audit_dir = output_root / "audit"
    partial_dir = output_root / "pkl" / "partial"
    case_dir = data_root / case_name
    bc_map = _load_bc_metadata(case_dir / features_subdir)
    safe_id = _safe_case_id(case_name)

    case_result: Dict[str, Any] = {
        "case_name": case_name,
        "split": split_name,
        "point_filters": {},
        "failures": [],
    }

    for point_filter in point_filters:
        case_log = audit_dir / "logs" / f"{safe_id}_{point_filter}.log"
        if case_log.exists():
            case_log.unlink()

        per_case_jsonl = audit_dir / "jsonl" / f"{safe_id}_{point_filter}.jsonl"
        if per_case_jsonl.exists():
            per_case_jsonl.unlink()

        records: List[Dict[str, Any]] = []
        frame_rows: List[Dict[str, Any]] = []

        if export_source == "raw_ascii":
            frames = list_raw_frames(case_dir)
            if not frames:
                row = {
                    "case_name": case_name,
                    "point_filter": point_filter,
                    "status": "missing",
                    "error": f"无 ascii 原始帧: {case_dir}",
                }
                append_jsonl(per_case_jsonl, row)
                if jsonl_path is not None:
                    append_jsonl(jsonl_path, row)
                case_result["failures"].append(row)
                continue
            for frame_key in sorted(frames):
                result = process_raw_frame(
                    case_dir, frame_key, case_name, split_name, point_filter,
                    voxel_size_mm, max_raw_points, seed, bc_map, case_log,
                )
                row = {k: v for k, v in result.items() if k != "record"}
                append_jsonl(per_case_jsonl, row)
                if jsonl_path is not None:
                    append_jsonl(jsonl_path, row)
                frame_rows.append(row)
                if result["status"] != "ok":
                    case_result["failures"].append(row)
                elif result["record"] is not None:
                    records.append(result["record"])
        else:
            csvs = sorted((case_dir / features_subdir).glob("result_features_*.csv"))
            if not csvs:
                row = {"case_name": case_name, "point_filter": point_filter, "status": "missing"}
                append_jsonl(per_case_jsonl, row)
                case_result["failures"].append(row)
                continue
            for csv_path in csvs:
                result = process_features_csv(
                    csv_path, case_name, split_name, point_filter,
                    voxel_size_mm, max_raw_points, seed, bc_map, case_log,
                )
                row = {k: v for k, v in result.items() if k != "record"}
                append_jsonl(per_case_jsonl, row)
                if jsonl_path is not None:
                    append_jsonl(jsonl_path, row)
                frame_rows.append(row)
                if result["status"] != "ok":
                    case_result["failures"].append(row)
                elif result["record"] is not None:
                    records.append(result["record"])

        if write_partial:
            partial_path = partial_dir / f"{safe_id}_{point_filter}.pkl"
            export_split_pkl(records, partial_path)
            print(f"Saved partial {partial_path} ({len(records)} samples)", flush=True)

        case_result["point_filters"][point_filter] = {
            "n_samples": len(records),
            "timing": _timing_summary(frame_rows),
        }

    return case_result


def merge_partials(config: Dict[str, Any]) -> Dict[str, Any]:
    data_cfg = config["data"]
    export_cfg = config.get("export", {})
    root = project_root()
    split = SplitSpec.from_json(data_cfg["split_file"])
    export_source = export_cfg.get("source", "raw_ascii")
    features_subdir = data_cfg.get("features_subdir", "processed/features")
    output_root = Path(data_cfg.get("output_root", str(default_private_preprocessed_root())))
    if not output_root.is_absolute():
        output_root = root / output_root

    point_filters = [_normalize_point_filter(pf) for pf in export_cfg.get("point_filters", ["volume", "all"])]
    point_filters = list(dict.fromkeys(point_filters))
    split_names = ("train", "val", "test")

    partial_dir = output_root / "pkl" / "partial"
    audit_dir = output_root / "audit"
    jsonl_dir = audit_dir / "jsonl"
    jsonl_path = audit_dir / "preprocess_cases.jsonl"

    all_records_by_filter: Dict[str, Dict[str, List[Dict[str, Any]]]] = {
        pf: {sn: [] for sn in split_names} for pf in point_filters
    }
    failures: List[Dict[str, Any]] = []

    if not partial_dir.is_dir():
        raise FileNotFoundError(f"partial 目录不存在: {partial_dir}")

    for partial_path in sorted(partial_dir.glob("*.pkl")):
        with open(partial_path, "rb") as f:
            records = pickle.load(f)
        if not records:
            continue
        pf = _normalize_point_filter(str(records[0].get("point_filter", "volume")))
        for rec in records:
            sn = rec.get("split", "train")
            if pf in all_records_by_filter and sn in all_records_by_filter[pf]:
                all_records_by_filter[pf][sn].append(rec)

    if jsonl_dir.is_dir():
        merged_lines: List[str] = []
        for jl in sorted(jsonl_dir.glob("*.jsonl")):
            merged_lines.extend(jl.read_text(encoding="utf-8").splitlines())
        jsonl_path.parent.mkdir(parents=True, exist_ok=True)
        jsonl_path.write_text(
            "\n".join(ln for ln in merged_lines if ln.strip()) + ("\n" if merged_lines else ""),
            encoding="utf-8",
        )
        for jl in jsonl_dir.glob("*.jsonl"):
            for line in jl.read_text(encoding="utf-8").splitlines():
                if not line.strip():
                    continue
                row = json.loads(line)
                if row.get("status") in ("error", "missing"):
                    failures.append(row)

    for point_filter in point_filters:
        for split_name in split_names:
            recs = all_records_by_filter[point_filter][split_name]
            recs.sort(key=lambda r: (r["case_name"], r.get("time_index", 0)))
            pkl_path = output_root / "pkl" / f"crown_{point_filter}_{split_name}.pkl"
            export_split_pkl(recs, pkl_path)
            print(f"Merged {pkl_path} ({len(recs)} samples)", flush=True)

    train_stats: Dict[str, Any] = {}
    for point_filter in point_filters:
        train_recs = all_records_by_filter[point_filter]["train"]
        if train_recs:
            train_stats[point_filter] = compute_p_stats(train_recs)

    data_root = Path(data_cfg["data_root"])
    if not data_root.is_absolute():
        data_root = root / data_root

    manifest: Dict[str, Any] = {
        "split_version": split.split_version,
        "data_root": str(data_root),
        "export_source": export_source,
        "features_subdir": features_subdir,
        "point_filters": point_filters,
        "voxel_size_mm": float(export_cfg.get("voxel_size_mm", 0.05)),
        "max_raw_points": export_cfg.get("max_raw_points"),
        "wall_detection": "velocity_threshold_0.01_at_train",
        "target_columns": list(TARGET_NAMES),
        "global_cond_names": list(GLOBAL_COND_NAMES),
    }

    dump_json(output_root / "stats" / "train_stats.json", train_stats)
    audit = {
        "split_version": split.split_version,
        "export_source": export_source,
        "point_filters": point_filters,
        "samples_per_filter_split": {
            pf: {sn: len(all_records_by_filter[pf][sn]) for sn in split_names}
            for pf in point_filters
        },
        "failure_count": len(failures),
        "failures": failures[:50],
        "train_p_stats": train_stats,
    }
    dump_json(audit_dir / "preprocess_audit.json", audit)
    dump_json(output_root / "manifests" / "export_manifest.json", manifest)
    return audit


def run_export(
    config: Dict[str, Any],
    *,
    dry_run: bool = False,
    cases: Sequence[str] | None = None,
    write_partial: bool = False,
    merge_only: bool = False,
    write_pilot_timing: bool = False,
) -> Dict[str, Any]:
    if merge_only:
        return merge_partials(config)

    data_cfg = config["data"]
    export_cfg = config.get("export", {})
    root = project_root()
    data_root = Path(data_cfg["data_root"])
    if not data_root.is_absolute():
        data_root = root / data_root

    split = SplitSpec.from_json(data_cfg["split_file"])
    case_split_map = build_case_split_map(split)
    export_source = export_cfg.get("source", "raw_ascii")
    features_subdir = data_cfg.get("features_subdir", "processed/features")
    output_root = Path(data_cfg.get("output_root", str(default_private_preprocessed_root())))
    if not output_root.is_absolute():
        output_root = root / output_root

    point_filters = [_normalize_point_filter(pf) for pf in export_cfg.get("point_filters", ["volume", "all"])]
    point_filters = list(dict.fromkeys(point_filters))
    voxel_size_mm = float(export_cfg.get("voxel_size_mm", 0.05))
    max_raw_points = export_cfg.get("max_raw_points")

    split_map = {
        "train": split.train_cases,
        "val": split.val_cases,
        "test": split.test_cases,
    }

    audit_dir = output_root / "audit"
    jsonl_path = audit_dir / "preprocess_cases.jsonl"

    if dry_run:
        print(f"CROWN export dry-run source={export_source}")
        for split_name, split_cases in split_map.items():
            print(f"\n[{split_name}] {len(split_cases)} cases")
            for case_name in split_cases[:3]:
                case_dir = data_root / case_name
                if export_source == "raw_ascii":
                    frames = list_raw_frames(case_dir)
                    print(f"  {case_name}: {len(frames)} raw frames")
                else:
                    csvs = sorted((case_dir / features_subdir).glob("result_features_*.csv"))
                    print(f"  {case_name}: {len(csvs)} feature csv")
        return {"dry_run": True}

    if write_partial:
        if cases is None:
            cases = []
            for split_cases in split_map.values():
                cases.extend(split_cases)
        pilot_results: List[Dict[str, Any]] = []
        t0 = time.time()
        for case_name in cases:
            if case_name not in case_split_map:
                print(f"SKIP unknown case (not in split): {case_name}", flush=True)
                continue
            split_name = case_split_map[case_name]
            result = export_single_case(
                case_name, split_name, config, write_partial=True, jsonl_path=None
            )
            pilot_results.append(result)
        elapsed = time.time() - t0
        if write_pilot_timing or (cases and len(cases) == 1):
            timing_doc = {
                "cases": cases,
                "elapsed_sec": round(elapsed, 3),
                "point_filters": point_filters,
                "per_case": pilot_results,
            }
            dump_json(audit_dir / "pilot_timing.json", timing_doc)
            print(f"Wrote {audit_dir / 'pilot_timing.json'}", flush=True)
        return {"partial_cases": len(pilot_results), "elapsed_sec": round(elapsed, 3)}

    # 串行全量 fallback：逐病例 partial 后 merge
    case_list: List[str] = []
    if cases:
        case_list = [c for c in cases if c in case_split_map]
    else:
        for split_cases in split_map.values():
            case_list.extend(split_cases)

    t0 = time.time()
    for case_name in tqdm(case_list, desc="cases"):
        split_name = case_split_map[case_name]
        export_single_case(case_name, split_name, config, write_partial=True, jsonl_path=None)
    elapsed = time.time() - t0
    audit = merge_partials(config)
    audit["serial_elapsed_sec"] = round(elapsed, 3)
    return audit


def main() -> None:
    parser = argparse.ArgumentParser(description="Export CROWN pkl (raw ascii or features)")
    parser.add_argument("--config", required=True)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--cases", nargs="*", default=None, help="限定病例，如 slow/LI_HUAN_GE")
    parser.add_argument("--cases-file", default=None, help="每行一个病例名")
    parser.add_argument("--case-index", type=int, default=None, help="配合 --cases-file 与 array task id")
    parser.add_argument("--write-partial", action="store_true", help="写 pkl/partial 按病例 shard")
    parser.add_argument("--merge-only", action="store_true", help="合并 partial → crown_*_{split}.pkl")
    parser.add_argument("--write-pilot-timing", action="store_true", help="写 audit/pilot_timing.json")
    parser.add_argument("--write-all-cases-file", default=None, help="写出 split 全病例列表到文件")
    args = parser.parse_args()
    config = load_config(args.config)

    if args.write_all_cases_file:
        split = SplitSpec.from_json(config["data"]["split_file"])
        cases = write_all_cases_file(split, Path(args.write_all_cases_file))
        print(f"Wrote {len(cases)} cases to {args.write_all_cases_file}")
        return

    cases: List[str] | None = None
    if args.cases_file:
        cases = load_cases_file(Path(args.cases_file))
        if args.case_index is not None:
            if args.case_index < 0 or args.case_index >= len(cases):
                raise IndexError(f"case-index {args.case_index} out of range [0, {len(cases)})")
            cases = [cases[args.case_index]]
    elif args.cases:
        cases = list(args.cases)

    audit = run_export(
        config,
        dry_run=args.dry_run,
        cases=cases,
        write_partial=args.write_partial or (cases is not None and not args.merge_only and not args.dry_run),
        merge_only=args.merge_only,
        write_pilot_timing=args.write_pilot_timing or (cases is not None and len(cases) == 1),
    )
    if not args.dry_run:
        print(json.dumps(audit, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

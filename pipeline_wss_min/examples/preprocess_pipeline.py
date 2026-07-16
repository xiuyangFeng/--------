#!/usr/bin/env python3
"""250 行内的自包含 WSS 预处理：不调用项目内 pipeline 模块。"""
from __future__ import annotations

import argparse, json, re, time
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.neighbors import NearestNeighbors

ROOT = Path(__file__).resolve().parents[2]
RAW_ROOT, OUT_ROOT = ROOT / "data_new", ROOT / "outputs/wss_min/teacher_preprocess"
DEFAULT_SPLIT = ROOT / "training/splits/split_AG_wss_min_v1.json"
DEFAULT_STATS = ROOT / "data_wss_min/wss_global_stats.json"
WALL_COL = {"id": "nodenumber", "xyz": ["x-coordinate", "y-coordinate", "z-coordinate"],
            "wss": "wall-shear", "vec": ["x-wall-shear", "y-wall-shear", "z-wall-shear"], "p": "pressure"}
INT_COL = {"xyz": ["x-coordinate", "y-coordinate", "z-coordinate"]}
CL_COL = {"xyz": ["x", "y", "z"], "s": "Abscissas",
          "r": "MaximumInscribedSphereRadius", "k": "Curvature"}


def path_of(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def require_included_case(cohort: str, case: str, split_path: str | Path) -> str:
    """只接收 split 中正式纳入的 AG 病例。"""
    if cohort not in ("AG/fast", "AG/slow"):
        raise ValueError("cohort 必须是 AG/fast 或 AG/slow")
    split = json.loads(path_of(split_path).read_text()); label = f"{cohort[3:]}/{case}"
    included = sum((split.get(f"{part}_cases", []) for part in ("train", "val", "test")), [])
    if label not in included:
        raise ValueError(f"{label} 未被当前 split 正式纳入")
    return label


def step_file(case_dir: Path, folder: str, case: str, step: int) -> Path:
    exact = case_dir / folder / f"{case}-{step}"
    matches = sorted((case_dir / folder).glob(f"*-{step}"))
    path = exact if exact.is_file() else (matches[0] if matches else exact)
    if not path.is_file(): raise FileNotFoundError(path)
    return path


def read_table(path: Path) -> pd.DataFrame:
    """兼容逗号或空白分隔的 Fluent ASCII。"""
    header = path.open().readline()
    frame = pd.read_csv(path, skipinitialspace=True) if "," in header else pd.read_csv(path, sep=r"\s+", engine="python")
    frame.columns = [name.strip() for name in frame.columns]
    return frame


def list_steps(case_dir: Path) -> list[int]:
    steps = []
    for path in (case_dir / "ascii").iterdir():
        match = re.search(r"-(\d+)$", path.name)
        if path.is_file() and match: steps.append(int(match.group(1)))
    if not steps: raise ValueError("壁面 ascii 目录中没有时间步")
    return sorted(set(steps))


def peak_step(case_dir: Path, steps: list[int]) -> int:
    """入口流量最大值对应峰值步，缺波形时取中间步。"""
    values = {}
    path = case_dir / "Global_conditions/vf-in-rfile.out"
    if path.is_file():
        for line in path.read_text().splitlines():
            try: key, value = line.split()[:2]; values[int(key)] = float(value)
            except (ValueError, IndexError): pass
    valid = {step: values[step] for step in steps if step in values}
    return max(valid, key=valid.get) if valid else steps[len(steps) // 2]


def read_wall(case_dir: Path, case: str, step: int) -> dict:
    frame = read_table(step_file(case_dir, "ascii", case, step))
    id_name = WALL_COL["id"] if WALL_COL["id"] in frame else "cellnumber"
    required = WALL_COL["xyz"] + [id_name, WALL_COL["wss"], WALL_COL["p"]] + WALL_COL["vec"]
    missing = [name for name in required if name not in frame]
    if missing: raise KeyError(f"壁面文件缺列：{missing}")
    return {"id": frame[id_name].to_numpy(np.int64), "xyz": frame[WALL_COL["xyz"]].to_numpy(float),
            "wss": frame[WALL_COL["wss"]].to_numpy(float), "p": frame[WALL_COL["p"]].to_numpy(float),
            "vec": frame[WALL_COL["vec"]].to_numpy(float)}


def read_centerline(case_dir: Path) -> dict:
    frame = pd.read_csv(case_dir / "centerline/centerline_points.csv")
    frame.columns = [name.strip() for name in frame.columns]
    return {"xyz": frame[CL_COL["xyz"]].to_numpy(float), "s": frame[CL_COL["s"]].to_numpy(float),
            "r": frame[CL_COL["r"]].to_numpy(float), "k": frame[CL_COL["k"]].to_numpy(float)}


def align_wall(fields: dict, reference_ids: np.ndarray) -> dict:
    """按 nodenumber 恢复稳定顺序，禁止节点缺失或重复。"""
    ids = fields["id"]
    if len(np.unique(ids)) != len(ids): raise ValueError("nodenumber 存在重复")
    order = np.argsort(ids); position = np.searchsorted(ids[order], reference_ids)
    if np.any(position == len(ids)) or not np.array_equal(ids[order][position], reference_ids):
        raise ValueError("时间步缺少稳定壁面节点")
    take = order[position]
    return {key: (value[take] if len(value) == len(ids) else value) for key, value in fields.items()}


def unit_factor(wall: np.ndarray, centerline: np.ndarray) -> tuple[float, bool]:
    """用中心线/壁面范围反推米到毫米；覆盖不一致时采用整十次幂。"""
    wall_d = np.linalg.norm(np.ptp(wall, axis=0)); cl_d = np.linalg.norm(np.ptp(centerline, axis=0))
    ratio = cl_d / max(wall_d, 1e-12); snap = 10.0 ** round(np.log10(250 / max(wall_d, 1e-12)))
    mismatch = not 1 / 1.5 <= ratio / snap <= 1.5
    return float(snap if mismatch else ratio), mismatch


def repair_centerline(centerline: dict, wall: np.ndarray) -> bool:
    """仅在明显纯平移错位且修复后贴近壁面时接受平移。"""
    diag = max(np.linalg.norm(np.ptp(wall, axis=0)), 1e-9)
    shift = (wall.min(0) + wall.max(0) - centerline["xyz"].min(0) - centerline["xyz"].max(0)) / 2
    if np.linalg.norm(shift) <= .8 * diag: return False
    candidate = centerline["xyz"] + shift
    p90 = np.percentile(NearestNeighbors(n_neighbors=1).fit(wall).kneighbors(candidate)[0], 90)
    if p90 > .16 * diag: return False
    centerline["xyz"] = candidate; return True


def read_stl_points(case_dir: Path, wall_mm: np.ndarray, factor: float) -> tuple[np.ndarray, str]:
    """读取并选择包围盒最接近 CFD 壁面的原始 STL。"""
    try:
        import vtk
        from vtk.util.numpy_support import vtk_to_numpy
    except ImportError as exc: raise RuntimeError("读取 STL 需要 vtk") from exc
    best = None; wall_center = (wall_mm.min(0) + wall_mm.max(0)) / 2; wall_span = np.ptp(wall_mm, axis=0)
    for path in sorted(case_dir.glob("*.stl")):
        reader = vtk.vtkSTLReader(); reader.SetFileName(str(path)); reader.Update()
        raw = vtk_to_numpy(reader.GetOutput().GetPoints().GetData()).astype(float)
        for scale in (1.0, factor, np.linalg.norm(wall_span) / max(np.linalg.norm(np.ptp(raw, axis=0)), 1e-9)):
            points = raw * scale; center = (points.min(0) + points.max(0)) / 2
            score = np.linalg.norm(center - wall_center) / max(np.linalg.norm(wall_span), 1e-9)
            score += .35 * np.linalg.norm(np.log(np.maximum(np.ptp(points, axis=0), 1e-9) / np.maximum(wall_span, 1e-9)))
            if best is None or score < best[0]: best = (score, points, str(path))
    if best is None: raise FileNotFoundError("病例目录中没有可读 STL")
    points = best[1]; return points[::max(1, len(points) // 30000)], best[2]


def anatomical_frame(points: np.ndarray) -> tuple[np.ndarray, np.ndarray, dict]:
    """由 STL 主轴两端展开度识别主干/双髂支，建立右手解剖坐标系。"""
    center = np.median(points, axis=0); shifted = points - center
    _, _, vt = np.linalg.svd(shifted[::max(1, len(shifted) // 12000)], full_matrices=False); axis = vt[0]
    axial = shifted @ axis; low, high = np.quantile(axial, [.12, .88])
    a, b = shifted[axial <= low], shifted[axial >= high]
    spread = lambda cloud: np.mean(np.linalg.norm(cloud - (cloud @ axis)[:, None] * axis, axis=1))
    fork, trunk = (a, b) if spread(a) > spread(b) else (b, a)
    z_axis = trunk.mean(0) - fork.mean(0); z_axis /= np.linalg.norm(z_axis)
    labels = KMeans(2, random_state=0, n_init=10).fit_predict(fork)
    branches = np.stack([fork[labels == i].mean(0) for i in range(2)])
    x_axis = branches[1] - branches[0]; x_axis -= (x_axis @ z_axis) * z_axis; x_axis /= np.linalg.norm(x_axis)
    if x_axis[0] < 0: x_axis = -x_axis
    y_axis = np.cross(z_axis, x_axis); y_axis /= np.linalg.norm(y_axis); x_axis = np.cross(y_axis, z_axis)
    origin = (branches.mean(0) + center); rotation = np.column_stack((x_axis, y_axis, z_axis))
    return origin, rotation, {"rotation_det": float(np.linalg.det(rotation)), "branch_separation_mm": float(np.linalg.norm(np.diff(branches, axis=0)))}


def centerline_features(points: np.ndarray, centerline: dict) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    index = NearestNeighbors(n_neighbors=1).fit(centerline["xyz"]).kneighbors(points, return_distance=False)[:, 0]
    s = centerline["s"]; s_norm = (s[index] - s.min()) / max(np.ptp(s), 1e-9)
    return s_norm.astype(np.float32), centerline["r"][index].astype(np.float32), centerline["k"][index].astype(np.float32)


def farthest_point_sample(points: np.ndarray, count: int, seed: int) -> np.ndarray:
    if count <= 0 or count >= len(points): return np.arange(len(points))
    rng = np.random.default_rng(seed); selected = np.empty(count, np.int64); selected[0] = rng.integers(len(points))
    distance = np.linalg.norm(points - points[selected[0]], axis=1)
    for index in range(1, count):
        selected[index] = np.argmax(distance); distance = np.minimum(distance, np.linalg.norm(points - points[selected[index]], axis=1))
    return selected


def normalize_wss(wss: np.ndarray, stats: dict) -> np.ndarray:
    if stats.get("method") != "log_z": raise ValueError("统计文件必须采用 log_z")
    return ((np.log(np.clip(wss, 0, None) + stats["eps"]) - stats["log"]["mean"]) / stats["log"]["std"]).astype(np.float32)


def preprocess(case_dir: Path, case: str, output: Path, wall_n: int, seed: int, stats: dict) -> dict:
    """完成单病例原始数据到 bundle、QA 和峰值训练样本的全过程。"""
    started = time.time(); steps = list_steps(case_dir); peak = peak_step(case_dir, steps)
    reference = read_wall(case_dir, case, peak); ref_ids = reference["id"]
    int_frame = read_table(step_file(case_dir, "ascii_in", case, peak)); int_native = int_frame[INT_COL["xyz"]].to_numpy(float)
    centerline = read_centerline(case_dir); factor, mismatch = unit_factor(reference["xyz"], centerline["xyz"])
    wall_mm, int_mm = reference["xyz"] * factor, int_native * factor
    repaired = repair_centerline(centerline, wall_mm)
    anatomy, stl_path = read_stl_points(case_dir, wall_mm, factor); origin, rotation, report = anatomical_frame(anatomy)
    wall_aligned, int_aligned = (wall_mm - origin) @ rotation, (int_mm - origin) @ rotation
    cl_aligned = (centerline["xyz"] - origin) @ rotation; cl_span = max(np.ptp(cl_aligned[:, 2]), 1e-9)
    cutoff = cl_aligned[:, 2].max() + .12 * cl_span; wall_keep = wall_aligned[:, 2] <= cutoff
    crop = wall_aligned[:, 2].max() - cl_aligned[:, 2].max() > .2 * cl_span and 1 - wall_keep.mean() > .08
    if crop:
        int_keep = int_aligned[:, 2] <= cutoff; wall_mm, wall_aligned = wall_mm[wall_keep], wall_aligned[wall_keep]
        int_mm, int_aligned = int_mm[int_keep], int_aligned[int_keep]; ref_ids = ref_ids[wall_keep]
        reference["xyz"] = reference["xyz"][wall_keep]
    scale = max(np.abs(wall_aligned).max(), 1e-9); wall_norm, int_norm = wall_aligned / scale, int_aligned / scale
    distance = NearestNeighbors(n_neighbors=1).fit(wall_mm).kneighbors(int_mm)[0][:, 0]
    point_type = np.where(distance <= 1.5, 1, 0).astype(np.int8)
    abscissa, radius, curvature = centerline_features(wall_mm, centerline)
    wss, pressure, vectors = [], [], []
    for step in steps:
        fields = align_wall(read_wall(case_dir, case, step), ref_ids)
        if np.max(np.abs(fields["xyz"] - reference["xyz"])) > 1e-8: raise ValueError(f"step {step} 壁面坐标漂移")
        wss.append(fields["wss"]); pressure.append(fields["p"]); vectors.append(fields["vec"] @ rotation)
    wss = np.asarray(wss, np.float32); output.mkdir(parents=True, exist_ok=True)
    bundle = output / "bundle.npz"
    np.savez_compressed(bundle, case=case, steps=np.asarray(steps), peak_step=peak, unit_factor=factor,
        transform_origin=origin, transform_rotation=rotation, coord_scale=scale,
        wall_nodenumber=ref_ids, wall_coords_norm=wall_norm.astype(np.float32), wall_wss=wss,
        wall_pressure=np.asarray(pressure, np.float32), wall_wss_vec=np.asarray(vectors, np.float32),
        wall_dist_to_wall=np.zeros(len(wall_mm), np.float32), wall_abscissa_norm=abscissa,
        wall_local_radius=radius, wall_curvature=curvature, int_coords_norm=int_norm.astype(np.float32),
        int_type=point_type, int_dist_to_wall=distance.astype(np.float32))
    peak_wss = wss[steps.index(peak)]; sample_idx = farthest_point_sample(wall_norm, wall_n, seed)
    np.savez_compressed(output / "peak_sample.npz", coords=wall_norm[sample_idx].astype(np.float32),
        y=normalize_wss(peak_wss[sample_idx], stats), y_raw=peak_wss[sample_idx], sample_indices=sample_idx)
    report.update({"case": case, "status": "ok", "n_wall": len(wall_mm), "n_interior": len(int_mm),
        "n_near_wall": int((point_type == 1).sum()), "n_steps": len(steps), "peak_step": peak,
        "unit_factor": factor, "unit_extent_mismatch": mismatch, "coord_scale_mm": scale,
        "centerline_translation_repaired": repaired, "wall_crop_applied": bool(crop), "stl_path": stl_path,
        "wss_nonfinite": int((~np.isfinite(wss)).sum()),
        "peak_zero_frac": float((peak_wss <= 0).mean()), "elapsed_s": time.time() - started})
    failures = [name for name, bad in {"rotation": abs(report["rotation_det"] - 1) > 1e-5,
        "nonfinite": report["wss_nonfinite"] > 0, "zero_wss": report["peak_zero_frac"] > .98,
        "coordinate_range": np.abs(wall_norm).max() > 1.0001}.items() if bad]
    report["qa_passed"], report["qa_failures"] = not failures, failures
    (output / "report.json").write_text(json.dumps(report, indent=2, ensure_ascii=False))
    if failures: raise RuntimeError(f"QA 未通过：{failures}")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="自包含 WSS 单病例完整预处理")
    parser.add_argument("--cohort", required=True, choices=("AG/fast", "AG/slow")); parser.add_argument("--case", required=True)
    parser.add_argument("--split", default=str(DEFAULT_SPLIT)); parser.add_argument("--stats", default=str(DEFAULT_STATS))
    parser.add_argument("--output-dir"); parser.add_argument("--wall-n", type=int, default=2000); parser.add_argument("--seed", type=int, default=1234)
    args = parser.parse_args(); label = require_included_case(args.cohort, args.case, args.split)
    stats = json.loads(path_of(args.stats).read_text())
    if stats.get("partitions") != ["train"] or stats.get("timesteps_scope") != "peak":
        raise ValueError("WSS 统计必须只来自 train 的峰值步")
    output = path_of(args.output_dir) if args.output_dir else OUT_ROOT / args.cohort / args.case
    report = preprocess(RAW_ROOT / args.cohort / args.case, args.case, output, args.wall_n, args.seed, stats)
    print(json.dumps({"case": label, "qa_passed": report["qa_passed"], "output": str(output)}, ensure_ascii=False, indent=2))


if __name__ == "__main__": main()

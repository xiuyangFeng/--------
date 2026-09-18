"""Explicit units and conservative geometry checks before centreline extraction."""
from __future__ import annotations

import struct
from pathlib import Path

import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
from training_wss_min.surface import load_stl

MAX_BYTES = 128 * 1024 * 1024
MAX_FACES = 2_000_000
MAX_FRAGMENT_FRACTION = 0.01
UNIT_FACTORS = {"mm": 1.0, "cm": 10.0, "m": 1000.0}
MIN_BBOX_DIAG_MM, MAX_BBOX_DIAG_MM = 50.0, 1500.0


def _unit_confidence(raw_diag: float, selected: str, suggested: str) -> tuple[float, list[str]]:
    """Estimate whether an STL's unit can be selected without a pause.

    STL has no unit field, so this is deliberately conservative.  A unit is
    considered high confidence only when exactly one of mm/cm/m puts the
    vessel in the supported physical size range and it is not right on a
    range boundary.  A user supplied unit is an explicit decision and has
    confidence 1.0.  The confidence is a routing aid, not a measurement of
    model uncertainty.
    """
    if selected != "auto":
        return 1.0, []
    converted = {u: raw_diag * factor for u, factor in UNIT_FACTORS.items()}
    candidates = [u for u, diag in converted.items()
                  if MIN_BBOX_DIAG_MM <= diag <= MAX_BBOX_DIAG_MM]
    if suggested not in candidates or len(candidates) != 1:
        if len(candidates) > 1:
            return 0.60, ["自动单位有多个合理候选，请确认 STL 原始单位。"]
        return 0.0, ["自动单位没有唯一的合理候选，请明确选择 mm、cm 或 m。"]
    diag = converted[suggested]
    # Distance from the nearest physical-size boundary in log space.  Values
    # well inside the range get 0.999; borderline values stay below 0.95.
    margin = min(diag / MIN_BBOX_DIAG_MM, MAX_BBOX_DIAG_MM / diag)
    confidence = 0.90 + 0.099 * min(max(np.log(margin) / np.log(3.0), 0.0), 1.0)
    if confidence < 0.95:
        return float(confidence), ["自动单位接近尺寸判定边界，请确认 STL 原始单位。"]
    return float(confidence), []


def write_binary_stl(path: Path, vertices: np.ndarray, faces: np.ndarray, header: str = "wss_deploy") -> None:
    tri = vertices[faces].astype(np.float32)
    normal = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
    normal /= np.maximum(np.linalg.norm(normal, axis=1, keepdims=True), 1e-12)
    rec = np.zeros(len(faces), dtype=[("n", "<f4", 3), ("v", "<f4", (3, 3)), ("attr", "<u2")])
    rec["n"], rec["v"] = normal, tri
    with Path(path).open("wb") as stream:
        stream.write(header.encode("ascii")[:80].ljust(80, b"\0"))
        stream.write(struct.pack("<I", len(faces)))
        stream.write(rec.tobytes())


def _read_checked(path: Path):
    if not path.is_file() or not 84 <= path.stat().st_size <= MAX_BYTES:
        raise ValueError("STL 文件为空、不可读或超过 128 MiB 上限。")
    with path.open("rb") as stream:
        header = stream.read(84)
        count = int.from_bytes(header[80:84], "little")
        if 84 + 50 * count == path.stat().st_size:
            if not 0 < count <= MAX_FACES:
                raise ValueError("STL 面片数超过 2,000,000 上限或没有面片。")
        else:
            stream.seek(0)
            count = 0
            for line in stream:
                if len(line) > 16384 or b"\0" in line:
                    raise ValueError("STL 二进制长度不匹配，或 ASCII 格式不正确。")
                if line.lstrip().startswith(b"vertex "):
                    count += 1
                    if count > MAX_FACES * 3:
                        raise ValueError("STL 面片数超过 2,000,000 上限。")
    try:
        vertices, faces = load_stl(path)
    except Exception as exc:
        raise ValueError("无法读取 STL，请导出有效的 ASCII 或二进制 STL。") from exc
    vertices, faces = np.asarray(vertices, np.float64), np.asarray(faces, np.int64)
    if vertices.ndim != 2 or vertices.shape[1] != 3 or not np.isfinite(vertices).all():
        raise ValueError("STL 坐标含 NaN/Inf 或维数不正确。")
    if len(faces) > MAX_FACES:
        raise ValueError("STL 面片数超过 2,000,000 上限。")
    return vertices, faces


def _areas(vertices, faces):
    tri = vertices[faces]
    return np.linalg.norm(np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0]), axis=1) * 0.5


def _components(vertices, faces):
    i = np.concatenate([faces[:, 0], faces[:, 1], faces[:, 2]])
    j = np.concatenate([faces[:, 1], faces[:, 2], faces[:, 0]])
    graph = coo_matrix((np.ones(len(i)), (i, j)), shape=(len(vertices), len(vertices)))
    count, labels = connected_components(graph, directed=False)
    area = _areas(vertices, faces)
    totals = np.bincount(labels[faces[:, 0]], weights=area, minlength=count)
    keep = int(np.argmax(totals))
    mask = labels == keep
    remap = np.full(len(vertices), -1, np.int64)
    remap[mask] = np.arange(mask.sum())
    fmask = mask[faces].all(axis=1)
    removed = float(1 - totals[keep] / totals.sum()) if totals.sum() else 0.0
    return vertices[mask], remap[faces[fmask]], int(count), removed


def largest_component(vertices, faces):
    """Compatibility helper; interactive ingest requires confirmation before removal."""
    vertices, faces, count, _ = _components(vertices, faces)
    return vertices, faces, count


def ingest(stl_path: Path, out_dir: Path, *, units: str = "mm", remove_fragments: bool = False) -> dict:
    if units not in {*UNIT_FACTORS, "auto"}:
        raise ValueError("单位必须选择 mm、cm、m，或 auto（仅建议、需要确认）。")
    vertices, faces = _read_checked(Path(stl_path))
    raw_size = np.ptp(vertices, axis=0)
    raw_diag = float(np.linalg.norm(raw_size))
    suggested = "m" if raw_diag < 2 else ("cm" if raw_diag < 100 else "mm")
    factor = UNIT_FACTORS[suggested if units == "auto" else units]
    vertices *= factor
    errors, flags, confirmations = [], [], []
    unit_confidence, unit_reasons = _unit_confidence(raw_diag, units, suggested)
    confirmations.extend(unit_reasons)
    if not MIN_BBOX_DIAG_MM <= raw_diag * factor <= MAX_BBOX_DIAG_MM:
        errors.append("换算后的整体尺寸超出工具预检范围（包围盒对角 50–1500 mm）；请重新核对单位或血管范围。")
    area = _areas(vertices, faces)
    if not np.isfinite(area).all() or np.any(area <= 1e-12):
        errors.append("表面含退化面片或无效面积，请修复后重新导出。")
    if len(np.unique(np.sort(faces, axis=1), axis=0)) != len(faces):
        errors.append("表面含重复三角面，请修复后重新导出。")
    kept_v, kept_f, count, removed = _components(vertices, faces)
    if count > 1:
        if removed > MAX_FRAGMENT_FRACTION:
            errors.append(f"有 {count} 个连通片；较小部分占总面积 {removed:.2%}，可能含断裂分支，不能自动删除。")
        elif not remove_fragments:
            confirmations.append(f"有 {count} 个连通片；删除小片将移除 {removed:.3%} 的面积，请核对后确认。")
        else:
            flags.append(f"经确认删除 {count - 1} 个小片（原总面积的 {removed:.3%}）。")
    edges = np.sort(np.concatenate([kept_f[:, [0, 1]], kept_f[:, [1, 2]], kept_f[:, [2, 0]]]), axis=1)
    unique, counts = np.unique(edges, axis=0, return_counts=True)
    boundary = unique[counts == 1]
    nonmanifold = int(np.sum(counts > 2))
    openings, irregular = 0, 0
    if len(boundary):
        touched, degree = np.unique(boundary.ravel(), return_counts=True)
        irregular = int(np.sum(degree != 2))
        graph = coo_matrix((np.ones(len(boundary)), (boundary[:, 0], boundary[:, 1])), shape=(len(kept_v), len(kept_v)))
        _, labels = connected_components(graph, directed=False)
        openings = len(np.unique(labels[touched]))
    if nonmanifold:
        errors.append(f"存在 {nonmanifold} 条非流形边，请修复表面。")
    if irregular:
        errors.append(f"边界含 {irregular} 个非环状连接点，不能视为有效开口。")
    if openings != 5:
        errors.append(f"开口数 {openings} ≠ 5；需要主动脉入口与四个髂动脉出口，且切口保持开放。")
    status = "fail" if errors else ("needs_confirmation" if confirmations else "pass")
    clean = Path(out_dir) / "input_clean_mm.stl"
    if status == "pass":
        Path(out_dir).mkdir(parents=True, exist_ok=True)
        write_binary_stl(clean, kept_v, kept_f)
    unit_note = "mm" if factor == 1 else f"{suggested if units == 'auto' else units}→mm (×{factor:g})"
    return {"source_stl": str(stl_path), "clean_stl": str(clean) if status == "pass" else None,
            "unit": unit_note, "selected_units": units,
            "resolved_units": suggested if units == "auto" else units,
            "suggested_units": suggested,
            "unit_confidence": round(float(unit_confidence), 4),
            "unit_confirmation_required": bool(unit_confidence < 0.95),
            "scale_factor": factor, "raw_bbox_size": raw_size.tolist(), "raw_bbox_diag": raw_diag,
            "bbox_size_mm": (raw_size * factor).tolist(), "bbox_diag_mm": raw_diag * factor,
            "vertices": len(kept_v), "faces": len(kept_f), "area_mm2": float(_areas(kept_v, kept_f).sum()),
            "original_area_mm2": float(area.sum()), "components": count,
            "removed_area_fraction": removed, "fragments_removed": count > 1 and remove_fragments and status == "pass",
            "openings": int(openings), "nonmanifold_edges": nonmanifold, "irregular_boundary_vertices": irregular,
            "flags": flags + confirmations, "errors": errors, "status": status, "ok": status == "pass",
            "orientation_source": "unknown_stl", "direction_note": "STL 未提供患者方向；世界 XYZ 不能自动解释为患者左右，请参考原始影像核对。"}

"""Explicit units and conservative geometry checks before centreline extraction."""
from __future__ import annotations

from pathlib import Path

import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
from wss_features.stl import load_stl, write_binary_stl

from .io_utils import portable_job_path

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


def _opening_geometry(vertices, boundary, labels) -> list[dict]:
    """Measure simple boundary loops without interpreting them as anatomy."""
    out = []
    for label in np.unique(labels[boundary.ravel()]):
        loop_edges = boundary[labels[boundary[:, 0]] == label]
        touched, degree = np.unique(loop_edges, return_counts=True)
        if len(touched) < 3 or np.any(degree != 2):
            continue
        neighbors = {int(index): [] for index in touched}
        for i, j in loop_edges:
            neighbors[int(i)].append(int(j))
            neighbors[int(j)].append(int(i))
        ordered = [int(touched[0])]
        previous, current = -1, ordered[0]
        for _ in range(len(touched)):
            following = next(i for i in neighbors[current] if i != previous)
            if following == ordered[0]:
                break
            ordered.append(following)
            previous, current = current, following
        xyz = vertices[ordered]
        centered = xyz - xyz.mean(axis=0)
        _, _, vh = np.linalg.svd(centered, full_matrices=False)
        residual = centered @ vh[-1]
        projected_area = float(np.linalg.norm(np.cross(centered, np.roll(centered, -1, axis=0)).sum(axis=0)) * .5)
        radius = float(np.sqrt(projected_area / np.pi))
        rms = float(np.sqrt(np.mean(residual ** 2)))
        out.append({"opening_index": len(out), "boundary_vertices": len(ordered),
                    "perimeter_mm": float(np.linalg.norm(xyz - np.roll(xyz, -1, axis=0), axis=1).sum()),
                    "projected_area_mm2": projected_area, "equivalent_radius_mm": radius,
                    "planarity_rms_mm": rms,
                    "planarity_rms_over_radius": rms / radius if radius > 0 else None,
                    "center_mm": xyz.mean(axis=0).tolist()})
    return out


def _quality_card(*, status, units, unit_confidence, diag, components, removed,
                  fragments_removed, openings, nonmanifold, irregular,
                  degenerate, duplicate, inconsistent, slivers, opening_geometry) -> dict:
    """Transparent check results; an unmeasured property is never a pass."""
    checks = []

    def add(key, label, result, value, note):
        checks.append({"key": key, "label": label, "status": result, "value": value, "note": note})

    add("units", "输入单位", "pass" if unit_confidence >= .95 else "review", units,
        "用户指定单位" if units != "auto" else "基于物理尺寸的规则估计，不是校准概率")
    add("physical_size", "物理尺寸", "pass" if MIN_BBOX_DIAG_MM <= diag <= MAX_BBOX_DIAG_MM else "fail", diag,
        "包围盒对角线 50–1500 mm 仅为输入范围，不等于模型训练分布")
    add("components", "连通片与碎片", "pass" if components == 1 or fragments_removed else ("fail" if removed > MAX_FRAGMENT_FRACTION else "review"),
        components, f"较小片面积占比 {removed:.4%}；删除状态：{'已确认删除' if fragments_removed else '未删除'}")
    add("degenerate_faces", "退化三角面", "fail" if degenerate else "pass", degenerate, "面片面积必须为有限正数")
    add("duplicate_faces", "重复三角面", "fail" if duplicate else "pass", duplicate, "按顶点索引检查重复三角面")
    add("nonmanifold_edges", "非流形边", "fail" if nonmanifold else "pass", nonmanifold, "每条内部边应恰好连接两个面片")
    add("boundary_loops", "切口与孔洞", "pass" if openings == 5 and irregular == 0 else "fail", openings,
        f"当前模型需要 5 个开放环；非环状连接点 {irregular} 个。开口计数不能证明解剖位置正确。")
    add("winding", "相邻面法向一致性", "review" if inconsistent else "pass", inconsistent,
        "检查共享边的面片绕序，不判定法向朝内或朝外")
    add("sliver_faces", "狭长三角面", "review" if slivers else "pass", slivers,
        "形状质量 4√3×面积/边长平方和 < 0.01 的面片数；提示阈值，不自动修复")
    tiny = [o["opening_index"] for o in opening_geometry if o["equivalent_radius_mm"] < .5]
    nonplanar = [o["opening_index"] for o in opening_geometry if (o["planarity_rms_over_radius"] or 0) > .05]
    add("opening_radius", "切口细小半径", "review" if tiny else ("pass" if opening_geometry else "not_checked"), tiny,
        "投影等效半径 < 0.5 mm 的切口需要复核；不替代沿中心线的最小半径")
    add("cut_planarity", "切口平面性", "review" if nonplanar else ("pass" if opening_geometry else "not_checked"), nonplanar,
        "最佳拟合平面 RMS 距离/切口等效半径 > 5% 时提示复核")
    add("self_intersection", "表面自交", "not_checked", None, "本版未运行可靠的三角面自交检测；不能将其视为已通过")
    add("patient_orientation", "患者方向", "unknown", None, "STL 没有患者坐标信息，世界轴不等于患者左右")
    add("training_domain", "模型训练范围", "not_checked", None, "由当前模型发布包的适用范围另行评估；尺寸预检不能替代训练域判断")
    review = any(c["status"] == "review" for c in checks)
    return {"schema_version": "wss-deploy.input-quality/v1",
            "grade": "fail" if status == "fail" else ("review" if status != "pass" or review else "pass_with_limits"),
            "grade_label": "输入不通过" if status == "fail" else ("需要关注" if status != "pass" or review else "已检查项通过（存在未评估项）"),
            "checks": checks, "opening_geometry": opening_geometry,
            "not_evaluated": [c["key"] for c in checks if c["status"] in {"not_checked", "unknown"}],
            "note": "质量等级只汇总实际输入检查，不表示预测准确率或自动命名置信度。"}


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
    degenerate = int(np.sum(~np.isfinite(area) | (area <= 1e-12)))
    if degenerate:
        errors.append("表面含退化面片或无效面积，请修复后重新导出。")
    duplicate = len(faces) - len(np.unique(np.sort(faces, axis=1), axis=0))
    if duplicate:
        errors.append("表面含重复三角面，请修复后重新导出。")
    kept_v, kept_f, count, removed = _components(vertices, faces)
    if count > 1:
        if removed > MAX_FRAGMENT_FRACTION:
            errors.append(f"有 {count} 个连通片；较小部分占总面积 {removed:.2%}，可能含断裂分支，不能自动删除。")
        elif not remove_fragments:
            confirmations.append(f"有 {count} 个连通片；删除小片将移除 {removed:.3%} 的面积，请核对后确认。")
        else:
            flags.append(f"经确认删除 {count - 1} 个小片（原总面积的 {removed:.3%}）。")
    directed = np.concatenate([kept_f[:, [0, 1]], kept_f[:, [1, 2]], kept_f[:, [2, 0]]])
    edges = np.sort(directed, axis=1)
    unique, inverse, counts = np.unique(edges, axis=0, return_inverse=True, return_counts=True)
    signs = np.where(directed[:, 0] < directed[:, 1], 1, -1)
    winding = np.bincount(inverse, weights=signs, minlength=len(unique))
    inconsistent = int(np.sum((counts == 2) & (np.abs(winding) == 2)))
    tri = kept_v[kept_f]
    edge_square = np.sum((tri - np.roll(tri, -1, axis=1)) ** 2, axis=(1, 2))
    shape_quality = 4 * np.sqrt(3.) * _areas(kept_v, kept_f) / np.maximum(edge_square, 1e-30)
    slivers = int(np.sum(shape_quality < .01))
    boundary = unique[counts == 1]
    nonmanifold = int(np.sum(counts > 2))
    openings, irregular, opening_geometry = 0, 0, []
    if len(boundary):
        touched, degree = np.unique(boundary.ravel(), return_counts=True)
        irregular = int(np.sum(degree != 2))
        graph = coo_matrix((np.ones(len(boundary)), (boundary[:, 0], boundary[:, 1])), shape=(len(kept_v), len(kept_v)))
        _, labels = connected_components(graph, directed=False)
        openings = len(np.unique(labels[touched]))
        opening_geometry = _opening_geometry(kept_v, boundary, labels)
    if inconsistent:
        flags.append(f"{inconsistent} 条内部边的相邻面片绕序不一致，建议复核法向。")
    if slivers:
        flags.append(f"{slivers} 个面片极度狭长，建议复核网格质量。")
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
    fragments_removed = count > 1 and remove_fragments and status == "pass"
    quality = _quality_card(status=status, units=units, unit_confidence=unit_confidence,
                            diag=raw_diag * factor, components=count, removed=removed,
                            fragments_removed=fragments_removed, openings=int(openings),
                            nonmanifold=nonmanifold, irregular=irregular, degenerate=degenerate,
                            duplicate=duplicate, inconsistent=inconsistent, slivers=slivers,
                            opening_geometry=opening_geometry)
    # Paths inside the job directory are stored relative to it so a job can be moved, copied or
    # re-run from another root; readers resolve them with ``io_utils.resolve_job_path``.
    return {"source_stl": portable_job_path(out_dir, stl_path),
            "clean_stl": portable_job_path(out_dir, clean) if status == "pass" else None,
            "unit": unit_note, "selected_units": units,
            "resolved_units": suggested if units == "auto" else units,
            "suggested_units": suggested,
            "unit_confidence": round(float(unit_confidence), 4),
            "unit_confirmation_required": bool(unit_confidence < 0.95),
            "scale_factor": factor, "raw_bbox_size": raw_size.tolist(), "raw_bbox_diag": raw_diag,
            "bbox_size_mm": (raw_size * factor).tolist(), "bbox_diag_mm": raw_diag * factor,
            "vertices": len(kept_v), "faces": len(kept_f), "area_mm2": float(_areas(kept_v, kept_f).sum()),
            "original_area_mm2": float(area.sum()), "components": count,
            "removed_area_fraction": removed, "fragments_removed": fragments_removed,
            "quality": quality, "degenerate_faces": degenerate, "duplicate_faces": duplicate,
            "inconsistent_winding_edges": inconsistent, "sliver_faces": slivers,
            "openings": int(openings), "nonmanifold_edges": nonmanifold, "irregular_boundary_vertices": irregular,
            "flags": flags + confirmations, "errors": errors, "status": status, "ok": status == "pass",
            "orientation_source": "unknown_stl", "direction_note": "STL 未提供患者方向；世界 XYZ 不能自动解释为患者左右，请参考原始影像核对。"}

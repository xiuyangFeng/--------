"""Stage 1: vessel_geom (VMTK, separate conda env) -> atlas; automatic outlet naming + manual confirmation hook.

Naming rules (validated on the 170 corrected training atlases, 2026-09-17):
  * left / right common iliac: the subtree with the smaller mean world x is LEFT (patient left = -x in our CT frame; 166/169 correct).
  * internal / external iliac inside each side: weighted score of four anatomical-frame cues (internal is more posterior, more medial,
    slightly smaller, and its endpoint sits higher); 325/340 correct.  |score| < 0.3 is flagged for the human check.
"""
from __future__ import annotations
import json, subprocess, time
from dataclasses import replace
from pathlib import Path
import numpy as np
from wss_v5.centerline_features import Atlas, load_atlas, _semantics
from wss_v5.views.wss_min_view import anatomical_frame
from .paths import VESSEL_GEOM_DIR, VMTK_PYTHON, OUTLET_NAMES, OUTLET_CN

IE_SCALE = np.array([16.2, 12.5, 0.77, 10.56])   # median |Δ| of (y, lateral, radius, z) on the training atlases
IE_WEIGHT = np.array([1.0, 1.0, 0.3, 0.3])


def _confidence_from_score(score: float) -> float:
    """Conservative probability proxy for the internal/external score margin.

    The historical review threshold was a score magnitude of 0.3.  The
    calibrated map gives that boundary 50% confidence and reaches 95% at a
    margin of about 2.0, avoiding the false precision of treating the raw
    score itself as a probability.
    """
    value = float(score)
    if not np.isfinite(value):
        return 0.0
    slope = np.log(19.0) / 1.7  # p(.3)=.50, p(2.0)=.95
    return float(1.0 / (1.0 + np.exp(-slope * (abs(value) - 0.3))))


def _confidence_from_x_gap(gap_mm: float) -> float:
    """Probability proxy for separating the two common iliac sides."""
    value = float(gap_mm)
    if not np.isfinite(value):
        return 0.0
    return float(1.0 / (1.0 + np.exp(-0.8 * (value - 10.0))))


def run_vessel_geom(stl_path: Path, out_dir: Path, *, inlet: int | None = None, smooth_iterations: int = 0, timeout: int = 900) -> dict:
    stl_path = Path(stl_path).resolve(); out_dir = Path(out_dir).resolve(); out_dir.mkdir(parents=True, exist_ok=True)
    cmd = [str(VMTK_PYTHON), "-m", "vessel_geom.cli", "--surface", str(stl_path), "--out", str(out_dir), "--preset", "frozen-aortoiliac",
           "--no-surface-features", "--smooth-iterations", str(smooth_iterations)]
    if inlet is not None:
        cmd += ["--inlet", str(inlet)]
    t = time.perf_counter()
    proc = subprocess.run(cmd, cwd=str(VESSEL_GEOM_DIR), capture_output=True, text=True, timeout=timeout)
    (out_dir / "vessel_geom.stdout.txt").write_text(proc.stdout + "\n--- stderr ---\n" + proc.stderr, encoding="utf-8")
    if proc.returncode != 0:
        raise RuntimeError(f"vessel_geom failed (rc={proc.returncode}): {proc.stderr[-2000:]}")
    run = json.loads((out_dir / "run.json").read_text(encoding="utf-8"))
    result = json.loads((out_dir / "centerline" / "result.json").read_text(encoding="utf-8"))
    return {"seconds": time.perf_counter() - t, "hard_pass": bool(run["extraction"]["hard_pass"]), "attempt": run["extraction"].get("selected_attempt"),
            "topology": run["graph_topology"], "openings": result["surface_report"]["openings"], "surface_report": {k: v for k, v in result["surface_report"].items() if k != "openings"}}


def load_vessel_geom_atlas(out_dir: Path) -> Atlas:
    return load_atlas(out_dir / "atlas.npz", out_dir / "atlas_summary.json")


def _tree(atlas: Atlas):
    by = {int(s["segment_id"]): s for s in atlas.segments}; kids: dict[int, list[int]] = {}
    for s in atlas.segments:
        kids.setdefault(int(s["parent_id"]), []).append(int(s["segment_id"]))
    root = [int(s["segment_id"]) for s in atlas.segments if s.get("starts_at_root")]
    return by, kids, (root[0] if root else None)


def _subtree(kids, sid):
    out = [sid]
    for k in kids.get(sid, []):
        out += _subtree(kids, k)
    return out


def propose_outlets(atlas: Atlas) -> dict:
    """Automatic naming proposal + everything the confirmation page needs (endpoints, radii, 2-D preview coordinates)."""
    by, kids, root = _tree(atlas)
    seg = atlas.col("segment_id").astype(int); idx = atlas.col("sample_index").astype(int); xyz = atlas.xyz
    cias = kids.get(root, []) if root is not None else []
    leaves_of = {c: [k for k in _subtree(kids, c) if by[k].get("ends_at_leaf")] for c in cias}
    proposal = {"root_segment": root, "cia_segments": cias, "auto_ok": False,
                "confirmation_required": True, "confidence_threshold": 0.95,
                "confidence": 0.0, "confidence_reasons": [], "flags": [],
                "mapping": {}, "scores": {}, "side_confidence": {}, "sides": {}}
    if len(cias) != 2 or any(len(leaves_of[c]) != 2 for c in cias):
        proposal["flags"].append(f"拓扑不是 1 入口 → 2 髂总 → 各 2 出口（髂总 {len(cias)} 条，出口 {[len(v) for v in leaves_of.values()]}），无法自动命名")
    else:
        mx = {c: float(xyz[np.isin(seg, _subtree(kids, c)), 0].mean()) for c in cias}
        left = min(mx, key=mx.get); right = max(mx, key=mx.get)
        x_gap = abs(mx[left] - mx[right])
        x_confidence = _confidence_from_x_gap(x_gap)
        proposal["side_confidence"]["left_right"] = round(x_confidence, 4)
        if x_confidence < 0.95:
            proposal["flags"].append(f"左右髂总在 x 轴上只差 {x_gap:.1f} mm，左右判定置信度 {x_confidence:.1%}，请人工核对")
        sem = {int(s["segment_id"]): -1 for s in atlas.segments}; sem[root] = 0
        for k in _subtree(kids, left): sem[k] = 1
        for k in _subtree(kids, right): sem[k] = 2
        fr = anatomical_frame(atlas.table, list(atlas.columns), {str(k): v for k, v in sem.items()}); R = fr["rotation"]
        for c, side, names in ((left, 1, ("out-li", "out-le")), (right, -1, ("out-ri", "out-re"))):
            crow = np.flatnonzero(seg == c); crow = crow[np.argsort(idx[crow])]; jpt = xyz[crow[-1]]
            feats = {}
            for k in leaves_of[c]:
                r = np.flatnonzero(seg == k); r = r[np.argsort(idx[r])]; d = (xyz[r[-1]] - jpt) @ R.T
                feats[k] = np.array([d[1], d[0] * side, float(np.median(atlas.col("radius_mm")[seg == k])), d[2]])
            a, b = leaves_of[c]
            delta = np.array([feats[b][0] - feats[a][0], feats[b][1] - feats[a][1], feats[b][2] - feats[a][2], feats[a][3] - feats[b][3]])
            score = float((delta / IE_SCALE) @ IE_WEIGHT)
            internal, external = (a, b) if score > 0 else (b, a)
            proposal["mapping"][str(internal)] = names[0]; proposal["mapping"][str(external)] = names[1]
            proposal["scores"][str(c)] = score; proposal["sides"][str(c)] = "left" if side == 1 else "right"
            confidence = _confidence_from_score(score)
            proposal["side_confidence"][str(c)] = round(confidence, 4)
            if confidence < 0.95:
                proposal["flags"].append(f"{'左' if side==1 else '右'}侧髂内/髂外区分置信度 {confidence:.1%}（score {score:.2f}），请人工核对")
        # Keep the unrounded value for the strict 0.95 routing gate; the UI
        # formats it for display.
        proposal["confidence"] = float(min(
            _confidence_from_x_gap(x_gap),
            *(_confidence_from_score(proposal["scores"][str(c)]) for c in cias),
        ))
        proposal["confidence_reasons"] = list(proposal["flags"])
        proposal["confirmation_required"] = bool(proposal["flags"] or proposal["confidence"] < 0.95)
        proposal["auto_ok"] = True
    # endpoints + 2-D preview (PCA plane of the centreline)
    c0 = xyz.mean(0); _, _, vt = np.linalg.svd(xyz - c0, full_matrices=False); P = vt[:2]
    ends = []
    for e in atlas.endpoints():
        sid = int(e["segment_id"]); label = "inlet" if e["label"] == "inlet" else proposal["mapping"].get(str(sid), e["label"])
        p2 = ((np.asarray(e["center_mm"]) - c0) @ P.T).tolist()
        ends.append({"segment_id": sid, "kind": "inlet" if e["label"] == "inlet" else "outlet", "auto_name": label, "name_cn": OUTLET_CN.get(label, label),
                     "radius_mm": float(e["radius_mm"]), "center_mm": np.asarray(e["center_mm"]).round(2).tolist(), "xy": p2})
    poly = []
    for s in atlas.segments:
        sid = int(s["segment_id"]); rows = np.flatnonzero(seg == sid); rows = rows[np.argsort(idx[rows])]
        poly.append({"segment_id": sid, "parent_id": int(s["parent_id"]), "xyz": xyz[rows].round(2).tolist(), "xy": ((xyz[rows] - c0) @ P.T).round(2).tolist(), "radius_mm": atlas.col("radius_mm")[rows].round(2).tolist()})
    proposal["endpoints"] = ends; proposal["preview_polylines"] = poly
    return proposal


def validate_mapping(atlas: Atlas, mapping: dict[str, str]) -> list[str]:
    by, kids, root = _tree(atlas); errors = []
    leaves = [int(s["segment_id"]) for s in atlas.segments if s.get("ends_at_leaf")]
    names = [mapping.get(str(k)) for k in leaves]
    if sorted(n for n in names if n) != sorted(OUTLET_NAMES):
        errors.append(f"四个出口必须各命名一次 {OUTLET_NAMES}，当前 {names}")
        return errors
    for c in kids.get(root, []):
        ln = {mapping[str(k)] for k in _subtree(kids, c) if by[k].get("ends_at_leaf")}
        if ln not in ({"out-le", "out-li"}, {"out-re", "out-ri"}):
            errors.append(f"髂总 {c} 下面的两个出口 {sorted(ln)} 不在同一侧；同一侧必须是 左内+左外 或 右内+右外")
    return errors


def apply_mapping(atlas: Atlas, mapping: dict[str, str]) -> Atlas:
    errors = validate_mapping(atlas, mapping)
    if errors:
        raise ValueError("; ".join(errors))
    old_to_new = {}
    segments = []
    for s in atlas.segments:
        s = dict(s); sid = str(int(s["segment_id"]))
        if s.get("ends_at_leaf"):
            old_to_new[s.get("outlet_name", "")] = mapping[sid]; s["outlet_name"] = mapping[sid]
        segments.append(s)
    for s in segments:
        s["descendant_outlets"] = [old_to_new.get(n, n) for n in s.get("descendant_outlets", [])]
    return replace(atlas, segments=segments, semantic_of_segment=_semantics(segments))

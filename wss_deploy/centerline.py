"""Stage 1: vessel_geom (VMTK, separate conda env) -> atlas; automatic outlet naming + manual confirmation hook.

Naming rules (validated on the 170 corrected training atlases, 2026-09-17):
  * left / right common iliac: the subtree with the smaller mean world x is LEFT (patient left = -x in our CT frame; 166/169 correct).
  * internal / external iliac inside each side: weighted score of four anatomical-frame cues (internal is more posterior, more medial,
    slightly smaller, and its endpoint sits higher); 325/340 correct.  |score| < 0.3 is flagged for the human check.
"""
from __future__ import annotations
import json, os, signal, subprocess, time
from dataclasses import replace
from pathlib import Path
import numpy as np
from wss_features.atlas import Atlas, load_atlas, semantics as _semantics
from wss_features.frame import anatomical_frame
from .paths import VESSEL_GEOM_DIR, VMTK_PYTHON, OUTLET_NAMES, OUTLET_CN
from .errors import ToolchainError

IE_SCALE = np.array([16.2, 12.5, 0.77, 10.56])   # median |Δ| of (y, lateral, radius, z) on the training atlases
IE_WEIGHT = np.array([1.0, 1.0, 0.3, 0.3])
CONFIDENCE_THRESHOLD = 0.95
CONFIDENCE_METHOD = "geometry_margin_proxy_v1"
# 2026-10-01: convention-free left/right check.  The anatomical frame's +Y is +Z × (assigned left), i.e. posterior when
# the left/right assignment is right.  The internal iliac runs posterior into the pelvis, so in the internal/external
# score the anterior-posterior term agrees with the mirror-invariant terms (lateral, radius, height); with the sides
# swapped by a mirrored STL it disagrees on both sides.  Library: 240 / 252 correct units agree (the 12 that do not
# all score below 0.94), the one mirrored STL (ZHANG_WEI_XIAN) disagrees.
HANDEDNESS_METHOD = "internal_iliac_posterior_v1"
VESSEL_GEOM_PRESET = "frozen-aortoiliac"         # the centreline preset the training atlases were built with


def _release_name(release) -> str | None:
    """Return a stable release id without serialising the release object."""
    if release is None:
        return None
    name = getattr(release, "name", None)
    if callable(name):
        name = name()
    if name:
        return str(name)
    if isinstance(release, dict):
        return str(release.get("release") or release.get("name") or "") or None
    return None


NAMING_PROFILE_DIR = Path(__file__).resolve().parent / "naming_profiles"
_NAMING_FINGERPRINT: str | None = None


def naming_fingerprint() -> str:
    """Identity of everything that decides an outlet name (2026-10-01): the proposal rules, both confidence maps,
    the anatomical frame, the constants and the frozen centreline toolkit.  Outlet naming runs in stage A and never
    reads model weights, so a naming profile holds for any release — as long as this fingerprint is unchanged; any
    edit of that code changes it and closes the gate until the library calibration is rerun."""
    global _NAMING_FINGERPRINT
    if _NAMING_FINGERPRINT is None:
        import hashlib
        import inspect
        parts = [inspect.getsource(fn) for fn in (propose_outlets, _confidence_from_score, _confidence_from_x_gap,
                                                   _tree, _subtree, anatomical_frame)]
        parts += [repr(IE_SCALE.tolist()), repr(IE_WEIGHT.tolist()), repr(CONFIDENCE_THRESHOLD), CONFIDENCE_METHOD,
                  HANDEDNESS_METHOD, Path(VESSEL_GEOM_DIR).resolve().name, VESSEL_GEOM_PRESET]
        _NAMING_FINGERPRINT = hashlib.sha256("\n".join(parts).encode("utf-8")).hexdigest()[:16]
    return _NAMING_FINGERPRINT


def _sidecar_profile(release_id: str | None) -> dict | None:
    """A validated profile under ``naming_profiles/`` for ``release_id`` (2026-10-01).

    The release folders are fingerprinted and never edited, so a profile made after a release was frozen lives
    here.  ``"releases": "*"`` binds it to the naming code instead of to release ids (naming never reads weights):
    it then holds for every release, including later ones, while ``naming_fingerprint`` matches; a list binds
    the named releases only.
    """
    if not release_id or not NAMING_PROFILE_DIR.is_dir():
        return None
    for path in sorted(NAMING_PROFILE_DIR.glob("*.json")):
        try:
            profile = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        if not isinstance(profile, dict):
            continue
        releases = profile.get("releases")
        if releases == "*" or (isinstance(releases, list) and str(release_id) in [str(r) for r in releases]):
            return {**profile, "release": str(release_id), "source_file": path.name}
    return None


def release_confidence_profile(release) -> dict | None:
    """Read the explicitly validated naming profile of a model release.

    A geometry margin is useful for ranking suggestions, but is not a
    probability.  The profile is deliberately opt-in: in the release metadata, or (2026-10-01) a sidecar under
    ``naming_profiles/`` that names the release id, so replacing weights cannot silently reuse a calibration
    belonging to another release.
    """
    if release is None:
        return None
    info = getattr(release, "info", release if isinstance(release, dict) else None)
    if isinstance(info, dict):
        profile = info.get("confidence_profile") or info.get("outlet_naming_confidence_profile")
        if isinstance(profile, dict):
            return dict(profile)
    return _sidecar_profile(_release_name(release))


def evaluate_confidence_gate(proposal: dict | None, *, orientation_source: str = "unknown_stl",
                            release=None) -> dict:
    """Evaluate whether a proposal may bypass a human confirmation.

    ``proposal.confidence`` is an uncalibrated geometry-margin proxy.  It is
    never interpreted as a 95% correctness probability by itself.  Bypass is
    permitted only when an independently validated profile is explicitly
    bound to the active model release and the patient's orientation is known.
    The returned record is persisted in the job for auditability.
    """
    proposal = proposal or {}
    try:
        requested_threshold = float(proposal.get("confidence_threshold", CONFIDENCE_THRESHOLD))
        threshold = max(CONFIDENCE_THRESHOLD, requested_threshold) if np.isfinite(requested_threshold) else CONFIDENCE_THRESHOLD
    except (TypeError, ValueError):
        threshold = CONFIDENCE_THRESHOLD
    profile = release_confidence_profile(release)
    try:
        threshold = max(threshold, float((profile or {}).get("proxy_threshold", threshold)))
    except (TypeError, ValueError):
        pass
    raw = proposal.get("confidence")
    reasons: list[str] = []
    if proposal.get("auto_ok") is not True:
        reasons.append("拓扑或中心线未形成完整的自动命名建议")
    if not isinstance(raw, (int, float)) or not np.isfinite(raw):
        reasons.append("缺少有限的几何置信度代理值")
        raw_value = None
    else:
        raw_value = float(raw)
        if raw_value < threshold:
            reasons.append(f"几何置信度代理 {raw_value:.1%} 低于门槛 {threshold:.1%}")
    if proposal.get("confirmation_required") is True:
        reasons.append("命名建议自身标记为需要确认")

    release_id = _release_name(release)
    profile_id = profile.get("id") if profile else None
    profile_version = profile.get("version") if profile else None
    profile_status = profile.get("status") if profile else None
    joint_lower_bound = ((profile or {}).get("joint_lower_bound")
                         if profile else None)
    if joint_lower_bound is None and profile:
        joint_lower_bound = profile.get("joint_correctness_lower_bound")
    if profile_status != "validated":
        reasons.append("当前发布包没有经过独立标注集验证的命名置信度 profile")
    if profile and not profile.get("validation_set"):
        reasons.append("命名 profile 缺少独立验证集标识")
    sample_count = (profile or {}).get("sample_count")
    if profile and (not isinstance(sample_count, (int, float)) or sample_count < 1):
        reasons.append("命名 profile 缺少有效的独立验证样本数")
    if not isinstance(joint_lower_bound, (int, float)) or float(joint_lower_bound) < threshold:
        reasons.append("命名 profile 没有达到联合正确率下界 95%")
    profile_release = (profile or {}).get("release") or (profile or {}).get("release_id")
    if not release_id or not profile_release:
        reasons.append("命名 profile 未绑定具体模型发布版本")
    elif str(profile_release) != str(release_id):
        reasons.append(f"命名 profile 属于 {profile_release}，当前发布包为 {release_id}")

    # The profile holds for the naming code it was measured with (2026-10-01): another proxy or handedness method
    # is not covered by it; a profile that relies on the handedness check needs a consistent one.
    if profile and profile.get("proxy_method") and proposal.get("confidence_method", CONFIDENCE_METHOD) != profile["proxy_method"]:
        reasons.append("命名置信度的计算方法与 profile 校准时不同")
    if profile and profile.get("naming_fingerprint") and profile["naming_fingerprint"] != naming_fingerprint():
        reasons.append("出口命名代码在 profile 校准之后改过，需要重新校准")
    if profile and profile.get("require_handedness"):
        hand = proposal.get("handedness") if isinstance(proposal.get("handedness"), dict) else {}
        if hand.get("method") != profile.get("handedness_method", HANDEDNESS_METHOD):
            reasons.append("缺少与 profile 一致的左右解剖朝向检查")
        elif hand.get("consistent") is not True:
            reasons.append("左右解剖朝向检查不一致（可能是镜像或另一种坐标约定的 STL）")
    allowed_orientation = (profile or {}).get("allowed_orientation_sources")
    if allowed_orientation is None and profile:
        source = profile.get("orientation_source")
        allowed_orientation = [source] if source else []
    if not isinstance(allowed_orientation, (list, tuple, set)) or orientation_source not in allowed_orientation:
        reasons.append("STL 未提供已验证的患者方向，不能自动解释左右")

    passed = not reasons
    return {
        "passed": bool(passed), "threshold": threshold,
        "proxy_confidence": raw_value, "proxy_method": CONFIDENCE_METHOD,
        "calibration_status": profile_status or "missing",
        "profile_id": profile_id, "profile_version": profile_version,
        "profile_release": profile_release, "release": release_id,
        "orientation_source": orientation_source,
        "joint_lower_bound": float(joint_lower_bound) if isinstance(joint_lower_bound, (int, float)) else None,
        "reasons": reasons,
    }


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


def _read_text(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return ""


def _write_atomic(path: Path, text: str) -> None:
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(text, encoding="utf-8")
    tmp.replace(path)


POLL_SECONDS = 0.5          # how often a running vessel_geom is checked for cancellation
TERMINATE_GRACE_SECONDS = 2.0   # SIGTERM -> SIGKILL grace for the whole process group


def vessel_geom_command(stl_path: Path, out_dir: Path, *, inlet: int | None = None, smooth_iterations: int = 0) -> list[str]:
    """The vessel_geom CLI invocation.  Factored out so tests can substitute a harmless child."""
    cmd = [str(VMTK_PYTHON), "-m", "vessel_geom.cli", "--surface", str(stl_path), "--out", str(out_dir), "--preset", VESSEL_GEOM_PRESET,
           "--no-surface-features", "--smooth-iterations", str(smooth_iterations)]
    if inlet is not None:
        cmd += ["--inlet", str(inlet)]
    return cmd


def _terminate_group(proc: "subprocess.Popen") -> None:
    """Stop the child *and everything it spawned*: vmtk shells out, so a bare terminate() leaks workers."""
    for signal_number, grace in ((signal.SIGTERM, TERMINATE_GRACE_SECONDS), (signal.SIGKILL, TERMINATE_GRACE_SECONDS)):
        if proc.poll() is not None:
            break
        try:
            os.killpg(os.getpgid(proc.pid), signal_number)
        except (ProcessLookupError, PermissionError, OSError):
            try:
                proc.kill()
            except OSError:
                pass
        try:
            proc.wait(timeout=grace)
        except subprocess.TimeoutExpired:
            continue


def run_vessel_geom(stl_path: Path, out_dir: Path, *, inlet: int | None = None, smooth_iterations: int = 0, timeout: int = 900,
                    cancelled=None) -> dict:
    """Run vessel_geom in its own session so a cancelled job kills the whole process group at once.

    ``cancelled`` is polled every :data:`POLL_SECONDS`; output goes to files (no pipes), so a chatty
    child can never deadlock the poll loop.
    """
    stl_path = Path(stl_path).resolve(); out_dir = Path(out_dir).resolve(); out_dir.mkdir(parents=True, exist_ok=True)
    cmd = vessel_geom_command(stl_path, out_dir, inlet=inlet, smooth_iterations=smooth_iterations)
    out_file, err_file = out_dir / ".vessel_geom.stdout.part", out_dir / ".vessel_geom.stderr.part"
    t = time.perf_counter()
    with open(out_file, "w", encoding="utf-8") as out_stream, open(err_file, "w", encoding="utf-8") as err_stream:
        proc = subprocess.Popen(cmd, cwd=str(VESSEL_GEOM_DIR), stdout=out_stream, stderr=err_stream,
                                text=True, start_new_session=True)
        stopped = None
        while proc.poll() is None:
            if cancelled is not None and cancelled():
                stopped = InterruptedError("任务已取消")
                break
            if time.perf_counter() - t > timeout:
                stopped = subprocess.TimeoutExpired(cmd, timeout)
                break
            time.sleep(POLL_SECONDS)
        if stopped is not None:
            _terminate_group(proc)
    stdout = _read_text(out_file); stderr = _read_text(err_file)
    _write_atomic(out_dir / "vessel_geom.stdout.txt", stdout + "\n--- stderr ---\n" + stderr)
    for part in (out_file, err_file):
        part.unlink(missing_ok=True)
    # v0.14 typed errors (errors.py): the user sees a short Chinese message, admins the tool's stderr tail.
    if isinstance(stopped, subprocess.TimeoutExpired):
        raise ToolchainError(f"中心线提取超时（超过 {int(timeout)} 秒）。请检查网格是否异常大或含大量碎片。",
                             admin_detail=f"vessel_geom timeout after {timeout} s; stderr tail:\n{stderr[-2000:]}")
    if stopped is not None:
        raise stopped
    if proc.returncode != 0:
        raise ToolchainError("中心线提取失败（VMTK / vessel_geom 出错）。请检查表面质量后重试，或联系维护者。",
                             admin_detail=f"vessel_geom failed (rc={proc.returncode}); stderr tail:\n{stderr[-2000:]}")
    try:
        run = json.loads((out_dir / "run.json").read_text(encoding="utf-8"))
        result = json.loads((out_dir / "centerline" / "result.json").read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise ToolchainError("中心线提取没有生成完整结果，请重试或联系维护者。",
                             admin_detail=f"vessel_geom outputs unreadable: {type(exc).__name__}: {exc}; stderr tail:\n{stderr[-1500:]}") from exc
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


def propose_outlets(atlas: Atlas, *, orientation_source: str = "unknown_stl") -> dict:
    """Automatic naming proposal + everything the confirmation page needs (endpoints, radii, 2-D preview coordinates)."""
    by, kids, root = _tree(atlas)
    seg = atlas.col("segment_id").astype(int); idx = atlas.col("sample_index").astype(int); xyz = atlas.xyz
    cias = kids.get(root, []) if root is not None else []
    leaves_of = {c: [k for k in _subtree(kids, c) if by[k].get("ends_at_leaf")] for c in cias}
    proposal = {"root_segment": root, "cia_segments": cias, "auto_ok": False,
                "confirmation_required": True, "confidence_threshold": CONFIDENCE_THRESHOLD,
                "confidence": 0.0, "confidence_method": CONFIDENCE_METHOD,
                "confidence_is_calibrated": False, "confidence_profile": None,
                "orientation_source": str(orientation_source or "unknown_stl"),
                "confidence_reasons": [], "flags": [], "mapping": {}, "scores": {},
                "side_confidence": {}, "sides": {}}
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
        handedness = 0.0
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
            terms = delta / IE_SCALE * IE_WEIGHT
            score = float(terms.sum())
            ap, rest = float(terms[0]), float(terms[1:].sum())
            handedness += abs(ap + rest) - abs(rest - ap)        # > 0: the AP term agrees with the mirror-invariant ones
            internal, external = (a, b) if score > 0 else (b, a)
            proposal["mapping"][str(internal)] = names[0]; proposal["mapping"][str(external)] = names[1]
            proposal["scores"][str(c)] = score; proposal["sides"][str(c)] = "left" if side == 1 else "right"
            confidence = _confidence_from_score(score)
            proposal["side_confidence"][str(c)] = round(confidence, 4)
            if confidence < 0.95:
                proposal["flags"].append(f"{'左' if side==1 else '右'}侧髂内/髂外区分置信度 {confidence:.1%}（score {score:.2f}），请人工核对")
        proposal["handedness"] = {"evidence": round(handedness, 4), "consistent": bool(handedness > 0),
                                  "method": HANDEDNESS_METHOD}
        if handedness <= 0:
            proposal["flags"].append("按当前左右判定，髂内动脉朝向腹侧，与解剖不符：STL 可能是镜像坐标，请对照原始影像核对左右")
        # Keep the unrounded value for the strict 0.95 routing gate; the UI
        # formats it for display.
        proposal["confidence"] = float(min(
            _confidence_from_x_gap(x_gap),
            *(_confidence_from_score(proposal["scores"][str(c)]) for c in cias),
        ))
        proposal["confidence_reasons"] = list(proposal["flags"])
        proposal["confidence_reasons"].append("几何间隔置信度只是未校准的代理值")
        if proposal["orientation_source"] == "unknown_stl":
            proposal["confidence_reasons"].append("STL 未提供患者方向：左右由坐标 x 方向与解剖朝向（髂内动脉向后）交叉判定")
        # This flag is intentionally conservative.  A validated release
        # profile is checked later by evaluate_confidence_gate; the centreline
        # stage alone must never claim that its proxy is a true probability.
        # The orientation source is judged by the release's validated profile (evaluate_confidence_gate), not here.
        proposal["confirmation_required"] = bool(proposal["flags"] or proposal["confidence"] < CONFIDENCE_THRESHOLD)
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

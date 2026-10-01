"""Outlet-naming calibration on the CFD library (2026-10-01).

Every real unit of the anatomy library goes through the deployment stage A unchanged (ingest → vessel_geom
centreline → ``centerline.propose_outlets``); the proposal is compared with the outlet names of that unit's
library atlas (``case.h5`` geometry, re-corresponded to the CFD outlet faces).  Each detected outlet is matched
to the nearest library outlet end; a unit is *joint correct* when all four names agree.

Run (patient-derived outputs stay under ``outputs/``, never in git)::

    python -m wss_deploy.naming_calibration run --out outputs/naming_calibration_20261001 --processes 8
    python -m wss_deploy.naming_calibration report --out outputs/naming_calibration_20261001
"""
from __future__ import annotations

import argparse
import json
import math
import multiprocessing as mp
import time
import traceback
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
LIBRARY = ROOT / "data_wss_v5/anatomy_pointcloud_v5_2d_20261001"
# IND split of v5.2: train170 = the v5.1 units the naming constants were fitted on, test91 = later recovered units
FIT_SPLIT = ROOT / "data_wss_v5/views_v5_2d_20261001/wss_min_view_v1/split_v52_ind_train170_test91.json"


def library_units(library: Path = LIBRARY) -> list[str]:
    """Real units of the library (synthetic ``~`` children excluded), as ``cohort/subset/case`` ids."""
    return sorted(p.name.replace("__", "/") for p in (library / "cases").iterdir() if "~" not in p.name)


def fit_units() -> set[str]:
    """Units the naming rule's constants were fitted on (the v5.1 atlases = the IND split's train170)."""
    try:
        split = json.loads(FIT_SPLIT.read_text())
    except OSError:
        return set()
    return set(split.get("train_cases", []))


def patient_of(uid: str) -> str:
    """Patient key: ILO before/after (and ``-0`` / ``-1`` suffixes) of one person count once."""
    cohort, subset, case = uid.split("/")
    name = subset if cohort == "ILO" else case
    return cohort + "/" + name.split("-")[0]


def truth_outlets(library: Path, uid: str) -> dict[str, np.ndarray]:
    """Library inlet + outlet names → the end sample of that segment (mm; the inlet is the trunk's first sample).

    The library atlas and the STL of ``data_new`` are not always in the same coordinates (a few units are
    translated by 0.3–2 m): :func:`match_names` aligns them before matching."""
    import h5py
    with h5py.File(library / "cases" / uid.replace("/", "__") / "case.h5", "r") as h:
        g = h["geometry"]
        table = np.asarray(g["atlas_table"][()])
        columns = json.loads(g.attrs["atlas_columns"])
        segments = json.loads(g.attrs["atlas_segments"])
    col = lambda name: table[:, columns.index(name)]
    seg, idx = col("segment_id").astype(int), col("sample_index").astype(int)
    xyz = np.stack([col("x_mm"), col("y_mm"), col("z_mm")], 1)
    out = {}
    for s in segments:
        rows = np.flatnonzero(seg == int(s["segment_id"]))
        rows = rows[np.argsort(idx[rows])]
        if s.get("ends_at_leaf"):
            out[str(s["outlet_name"])] = xyz[rows[-1]]
        if s.get("starts_at_root") or int(s.get("parent_id", 0)) < 0:
            out["inlet"] = xyz[rows[0]]
    return out


def match_names(endpoints: list[dict], truth: dict[str, np.ndarray]) -> tuple[dict[str, str], dict[str, float], float, float]:
    """Detected outlet segment id → library name, by a one-to-one assignment after aligning the five end points
    (inlet + outlets) by their centroids.  Returns (names, per-outlet distance mm, the translation applied mm,
    the RMS residual mm).  A pure translation leaves a residual of a few mm; anything else shows as a large one."""
    from scipy.optimize import linear_sum_assignment
    det = {str(e["segment_id"]): np.asarray(e["center_mm"], float) for e in endpoints if e["kind"] == "outlet"}
    inlet = next((np.asarray(e["center_mm"], float) for e in endpoints if e["kind"] == "inlet"), None)
    names = [k for k in truth if k != "inlet"]
    a = np.array([det[k] for k in det] + ([inlet] if inlet is not None else []))
    b = np.array([truth[k] for k in names] + ([truth["inlet"]] if "inlet" in truth and inlet is not None else []))
    shift = b.mean(0) - a.mean(0) if len(a) == len(b) else np.zeros(3)
    keys = list(det)
    cost = np.array([[np.linalg.norm(det[k] + shift - truth[n]) for n in names] for k in keys])
    r, c = linear_sum_assignment(cost)
    out = {keys[i]: names[j] for i, j in zip(r, c)}
    dist = {keys[i]: round(float(cost[i, j]), 2) for i, j in zip(r, c)}
    return out, dist, round(float(np.linalg.norm(shift)), 1), round(float(np.sqrt(np.mean(cost[r, c] ** 2))), 2)


def compare(row: dict, a: dict, uid: str, library: Path = LIBRARY, *, transform: str | None = None) -> dict:
    """The proposal of stage-A record ``a`` against the library names (fills ``row``); ``transform``: the STL was
    transformed (``TRANSFORMS``), so are the library coordinates — the anatomy, and so the names, are unchanged."""
    p = a["proposal"]
    row.update(auto_ok=bool(p.get("auto_ok")), confidence=p.get("confidence"), side_confidence=p.get("side_confidence"),
               scores=p.get("scores"), sides=p.get("sides"), flags=p.get("flags"), mapping=p.get("mapping"),
               handedness=p.get("handedness"), confirmation_required=p.get("confirmation_required"))
    if not p.get("auto_ok"):
        return row
    truth = truth_outlets(library, uid)
    if transform:
        truth = {k: v * TRANSFORMS[transform] for k, v in truth.items()}
    names, dists, shift, rms = match_names(p["endpoints"], truth)
    row.update(truth=names, match_mm=dists, align_shift_mm=shift, align_rms_mm=rms, unique_match=len(set(names.values())) == 4)
    proposed = {k: v for k, v in p["mapping"].items()}
    side = lambda n: n[4]          # out-le → 'l'
    kind = lambda n: n[5]          # out-le → 'e'
    row["lr_correct"] = all(side(proposed[k]) == side(names[k]) for k in proposed)
    row["lr_swapped"] = all(side(proposed[k]) != side(names[k]) for k in proposed)
    for s, label in (("l", "left"), ("r", "right")):
        keys = [k for k in proposed if side(names[k]) == s]
        row[f"ie_correct_{label}"] = all(kind(proposed[k]) == kind(names[k]) for k in keys) if keys else None
    row["joint_correct"] = all(proposed[k] == names[k] for k in proposed)
    return row


# Coordinate transforms of the STL that a different export convention produces: ``mirror_x`` (left-handed, det −1)
# and ``rot_z180`` (LPS ↔ RAS: x and y negated, a proper rotation).  The anatomy and so the names do not change.
TRANSFORMS = {"mirror_x": np.array([-1.0, 1.0, 1.0]), "rot_z180": np.array([-1.0, -1.0, 1.0])}


def transformed_stl(stl: Path, out: Path, transform: str) -> Path:
    """``stl`` under ``transform`` (triangle order reversed for a reflection, so normals stay outward), binary STL."""
    import pyvista as pv
    sign = TRANSFORMS[transform]
    mesh = pv.read(str(stl))
    points = np.asarray(mesh.points, dtype=np.float64) * sign
    faces = np.asarray(mesh.faces).reshape(-1, 4)
    if np.prod(sign) < 0:
        faces = faces[:, [0, 1, 3, 2]]
    out.parent.mkdir(parents=True, exist_ok=True)
    pv.PolyData(points, faces.ravel()).save(str(out), binary=True)
    return out


def evaluate_unit(uid: str, out_root: Path, library: Path = LIBRARY, *, transform: str | None = None) -> dict:
    """Stage A on the unit's STL (or its mirror image) + comparison with the library names (never raises)."""
    from training_wss_min.tools.deployment_stl_simulation import find_stl
    from .pipeline import stage_a
    row = {"unit_id": uid, "patient": patient_of(uid), "cohort": uid.split("/")[0]}
    started = time.perf_counter()
    try:
        stl = find_stl(uid)
        if stl is None:
            return {**row, "error": "no STL"}
        job_dir = out_root / "jobs" / uid.replace("/", "__")
        if transform:
            stl = transformed_stl(Path(stl), out_root / "transformed_stl" / (uid.replace("/", "__") + ".stl"), transform)
        a = stage_a(Path(stl), job_dir, units="auto")
        row["seconds"] = round(time.perf_counter() - started, 2)
        if a.get("stage") != "A":
            return {**row, "error": "stage A did not pass: " + "; ".join((a.get("input_check") or {}).get("errors") or [])[:300]}
        return compare(row, a, uid, library, transform=transform)
    except Exception as exc:  # noqa: BLE001
        return {**row, "error": f"{type(exc).__name__}: {exc}"[:300], "traceback": traceback.format_exc()[-1200:]}


def _worker(args):
    uid, out_root, library, transform = args
    return evaluate_unit(uid, Path(out_root), Path(library), transform=transform)


def run(out_root: Path, *, processes: int = 8, units: list[str] | None = None, library: Path = LIBRARY,
        transform: str | None = None) -> list[dict]:
    out_root.mkdir(parents=True, exist_ok=True)
    units = units or library_units(library)
    fitted = fit_units()
    rows, path = [], out_root / "rows.jsonl"
    done = {}
    if path.exists():
        for line in path.read_text().splitlines():
            r = json.loads(line); done[r["unit_id"]] = r
    todo = [u for u in units if u not in done]
    with path.open("a") as fh, mp.get_context("spawn").Pool(processes) as pool:
        for row in pool.imap_unordered(_worker, [(u, str(out_root), str(library), transform) for u in todo]):
            row["fit_set"] = row["unit_id"] in fitted
            fh.write(json.dumps(row, ensure_ascii=False) + "\n"); fh.flush()
            done[row["unit_id"]] = row
            print(json.dumps({k: row.get(k) for k in ("unit_id", "joint_correct", "confidence", "error")}, ensure_ascii=False), flush=True)
    return [done[u] for u in units if u in done]


def rescore(out_root: Path, library: Path = LIBRARY, *, transform: str | None = None) -> list[dict]:
    """Recompare every finished unit (no stage A rerun): the proposal is rebuilt by the current
    ``centerline.propose_outlets`` from the saved centreline; rewrites rows.jsonl."""
    from . import centerline as CL
    rows = load_rows(out_root)
    fitted = fit_units()
    new = []
    for r in rows:
        path = out_root / "jobs" / r["unit_id"].replace("/", "__") / "stage_a.json"
        keep = {k: r[k] for k in ("unit_id", "patient", "cohort", "seconds", "error") if k in r}
        if path.is_file() and not r.get("error"):
            a = json.loads(path.read_text())
            if a.get("stage") == "A":
                try:
                    atlas = CL.load_vessel_geom_atlas(path.parent / "centerline")
                    a["proposal"] = CL.propose_outlets(atlas, orientation_source=(a.get("input_check") or {}).get(
                        "orientation_source", "unknown_stl"))
                    keep = compare(keep, a, r["unit_id"], library, transform=transform)
                except Exception as exc:  # noqa: BLE001
                    keep["error"] = f"{type(exc).__name__}: {exc}"[:300]
        keep["fit_set"] = keep["unit_id"] in fitted
        new.append(keep)
    (out_root / "rows.jsonl").write_text("".join(json.dumps(x, ensure_ascii=False) + "\n" for x in new))
    return new


def clopper_pearson_lower(k: int, n: int, alpha: float = 0.05) -> float:
    """One-sided (1 − alpha) lower confidence bound of a binomial proportion k / n."""
    if n == 0:
        return 0.0
    if k == 0:
        return 0.0
    if k == n:
        return alpha ** (1.0 / n)
    from scipy.stats import beta
    return float(beta.ppf(alpha, k, n - k + 1))


def passes(row: dict) -> bool:
    """The proposal would be released without a person under a profile that accepts its orientation source:
    complete topology, no flag (low margins or the handedness check) and a proxy of at least the threshold."""
    return bool(row.get("auto_ok")) and row.get("confirmation_required") is False


def summarise(rows: list[dict], *, max_rms_mm: float = 5.0) -> dict:
    """Pass rate and correctness of the passing proposals, per unit and per patient (one-sided 95 % bounds).

    Units whose STL differs from the library geometry beyond a translation (residual > ``max_rms_mm``) have an
    uncertain reference and are left out."""
    import collections
    scored = [r for r in rows if "joint_correct" in r]
    certain = [r for r in scored if (r.get("align_rms_mm") or 0.0) <= max_rms_mm]
    out = {"units": len(rows), "scored": len(scored), "reference_certain": len(certain),
           "errors_total": sum(not r["joint_correct"] for r in certain),
           "errors_passed": sum(1 for r in certain if passes(r) and not r["joint_correct"])}
    for name, sub in (("all", certain), ("independent", [r for r in certain if not r.get("fit_set")]),
                      ("fit_set", [r for r in certain if r.get("fit_set")])):
        passed = [r for r in sub if passes(r)]
        k = sum(r["joint_correct"] for r in passed)
        by_patient = collections.defaultdict(list)
        for r in passed:
            by_patient[r["patient"]].append(r["joint_correct"])
        kp = sum(all(v) for v in by_patient.values())
        out[name] = {"units": len(sub), "passed": len(passed), "correct": k,
                     "lower_bound": round(clopper_pearson_lower(k, len(passed)), 4),
                     "patients": len(by_patient), "patients_correct": kp,
                     "patient_lower_bound": round(clopper_pearson_lower(kp, len(by_patient)), 4)}
    return out


def transform_summary(rows: list[dict]) -> dict:
    """A coordinate-transform run: how many proposals would pass and how many of those name wrongly."""
    scored = [r for r in rows if "joint_correct" in r and (r.get("align_rms_mm") or 0.0) <= 5.0]
    passed = [r for r in scored if passes(r)]
    return {"scored": len(scored), "left_right_wrong": sum(not r["lr_correct"] for r in scored),
            "passed": len(passed), "passed_wrong": sum(not r["joint_correct"] for r in passed),
            "handedness_inconsistent": sum(1 for r in scored if not (r.get("handedness") or {}).get("consistent"))}


def build_profile(out_root: Path, *, releases: list[str], transforms: dict[str, Path] | None = None) -> dict:
    """The sidecar profile (aggregate numbers only, no unit names) for ``centerline.release_confidence_profile``."""
    from . import centerline as CL
    stats = summarise(load_rows(out_root))
    tests = {name: transform_summary(load_rows(path)) for name, path in (transforms or {}).items() if (path / "rows.jsonl").exists()}
    bound = stats["all"]["patient_lower_bound"]
    return {
        "id": "outlet_naming_library_v52d_20261001", "version": "1",
        "status": "validated" if bound >= CL.CONFIDENCE_THRESHOLD and stats["errors_passed"] == 0 else "not_validated",
        # naming runs in stage A and never reads weights: bound to the naming code, valid for every release
        "releases": "*" if list(releases) == ["*"] else list(releases),
        "naming_fingerprint": CL.naming_fingerprint(),
        "proxy_method": CL.CONFIDENCE_METHOD, "proxy_threshold": CL.CONFIDENCE_THRESHOLD,
        "handedness_method": CL.HANDEDNESS_METHOD, "require_handedness": True,
        "allowed_orientation_sources": ["unknown_stl"],
        "validation_set": "CFD library v5.2d real units (anatomy_pointcloud_v5_2d_20261001); reference names = the "
                          "library atlas outlet names (re-corresponded to the CFD outlet faces); deployment stage A on "
                          "the unit's STL, outlets matched after aligning the five end points",
        "population": "STL from the AG / AAA / ILO segmentation pipeline of the library cohorts; other sources are not covered",
        "sample_unit": "patient", "sample_count": stats["all"]["patients"],
        "joint_correct": stats["all"]["patients_correct"], "joint_lower_bound": bound,
        "lower_bound_method": "one-sided 95 % Clopper-Pearson",
        "statistics": stats, "coordinate_tests": tests,
        "notes": ["internal/external errors change only report labels; left/right errors change the anatomical frame",
                  "a left-handed (mirrored) export defeats both left/right rules (x sign and internal-iliac-posterior); "
                  "none was seen in the library",
                  "the four naming constants (IE_SCALE) were fitted on the fit_set units; the independent subset is reported apart"],
        "created_at": time.strftime("%Y-%m-%d"), "tool": "wss_deploy.naming_calibration",
    }


def load_rows(out_root: Path) -> list[dict]:
    return [json.loads(line) for line in (out_root / "rows.jsonl").read_text().splitlines() if line.strip()]


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run"); r.add_argument("--out", type=Path, required=True); r.add_argument("--processes", type=int, default=8)
    r.add_argument("--units", nargs="*"); r.add_argument("--transform", choices=sorted(TRANSFORMS))
    rs = sub.add_parser("rescore"); rs.add_argument("--out", type=Path, required=True); rs.add_argument("--transform", choices=sorted(TRANSFORMS))
    pr = sub.add_parser("profile"); pr.add_argument("--out", type=Path, required=True)
    pr.add_argument("--transform-run", nargs=2, action="append", metavar=("NAME", "DIR"), default=[])
    pr.add_argument("--releases", nargs="+", required=True); pr.add_argument("--write", type=Path)
    args = ap.parse_args(argv)
    if args.cmd == "profile":
        profile = build_profile(args.out, releases=args.releases, transforms={n: Path(d) for n, d in args.transform_run})
        text = json.dumps(profile, ensure_ascii=False, indent=1) + "\n"
        if args.write:
            args.write.parent.mkdir(parents=True, exist_ok=True); args.write.write_text(text)
        print(text)
        return 0
    if args.cmd == "rescore":
        rows = rescore(args.out, transform=args.transform)
        ok = [x for x in rows if "joint_correct" in x]
        print(json.dumps({"units": len(rows), "scored": len(ok), "joint_correct": sum(x["joint_correct"] for x in ok)}))
        return 0
    if args.cmd == "run":
        rows = run(args.out, processes=args.processes, units=args.units, transform=args.transform)
        ok = [x for x in rows if "joint_correct" in x]
        print(json.dumps({"units": len(rows), "scored": len(ok), "joint_correct": sum(x["joint_correct"] for x in ok)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

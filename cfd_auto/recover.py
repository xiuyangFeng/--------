"""Recover library units that are not in v5.2 with a fresh, protocol-exact CFD (2026-09-29 batch).

    python -m cfd_auto.recover <unit> --plan outputs/cfd_auto_trial_20260927/_recover/plan.json \
        --work-root outputs/cfd_auto_trial_20260927/_recover/units [--cores 92] [--submit]

Two modes; both end in the regression pipeline's setup / smoke / run stages with a settings template that passed the
cfd_auto regression (or a library unit of the same naming family):

  own-mesh  the unit's own library mesh (a path-rewritten copy), zones renamed to the template's names
  stl       the unit's STL: surface -> (P family) calibrated curvature remesh / (T family) STL facets -> mesh -> finalize

Boundary conditions are always the library protocol, whatever the unit's own UDF says (several were mis-entered):
template UDF (waveform, Carreau, RCR recursion) with the final case's outlet thread ids, inlet divisor = final inlet
boundary area, and R1/R2/C = ``udf.protocol_rcr`` on the final anatomy cut-face areas (A1 = 0.033 kg/s for AAA/ILO;
reproduces the library UDFs to 0.01-0.03 %). With the protocol the solution does not depend on which outlet is called
left/right or internal/external (the RCR of an outlet is a function of its own area, Murray within a side, 50/50 across
sides); only the side pairing matters, and it comes from the vessel tree. Names still matter for the data set, so STL
openings are named by (a) a same-frame partner unit's opening centres, (b) rigid registration to a partner in another
frame, or (c) the deployment naming (vessel_geom centreline -> ``wss_deploy`` proposal); every method that applies is
run and they must agree.
"""
from __future__ import annotations

import argparse
import itertools
import json
import shutil
import subprocess
from pathlib import Path

import numpy as np

from cfd_auto import guard, journals, meshcheck, pipeline, rebuild, refcase, slurm, surface, udf

KEYS = udf.OUTLETS
DEPLOY_KEY = {"out-le": "outle", "out-li": "outli", "out-ri": "outri", "out-re": "outre", "inlet": "inlet"}
DENSITY_TOL = 0.10          # calibrated dual wall-face density vs target


def _dump(path: Path, obj) -> None:
    pipeline._dump(path, obj)


def cohort(unit: str) -> str:
    return unit.split("/")[0]


# ----------------------------------------------------------------------------------------------------------- naming
def stl_openings(stl: Path) -> tuple[np.ndarray, list[dict]]:
    pts, tri = surface.read_stl(stl)
    out = []
    for i, l in enumerate(surface.boundary_loops(tri)):
        P = pts[l]; c = P.mean(0); _, _, vt = np.linalg.svd(P - c)
        uv = np.c_[(P - c) @ vt[0], (P - c) @ vt[1]]
        out.append({"loop": i, "centre_mm": c, "area_mm2": 0.5 * abs(np.dot(uv[:, 0], np.roll(uv[:, 1], -1)) - np.dot(uv[:, 1], np.roll(uv[:, 0], -1)))})
    return pts, out


def _assign(centres: np.ndarray, ref: list[tuple[str, np.ndarray]]) -> tuple[dict[int, str], list[float]]:
    """One-to-one loop -> key by minimum total centre distance (5 openings: 120 permutations)."""
    best, cost = None, np.inf
    for perm in itertools.permutations(range(len(ref)), len(centres)):
        c = sum(np.linalg.norm(centres[i] - ref[j][1]) for i, j in enumerate(perm))
        if c < cost:
            best, cost = perm, c
    keys = {i: ref[j][0] for i, j in enumerate(best)}
    return keys, [round(float(np.linalg.norm(centres[i] - ref[j][1])), 2) for i, j in enumerate(best)]


def partner_keys(partner_prof: dict) -> list[tuple[str, np.ndarray]]:
    zone_key = {z: k for k, z in partner_prof["udf_thread_zones"].items()}
    ref = []
    for o in partner_prof["openings"]:
        k = "inlet" if o["bc_type"] == "velocity-inlet" else zone_key.get(o["name"])
        if k is None:
            raise KeyError(f"partner opening {o['name']} has no UDF key ({partner_prof['udf_thread_zones']})")
        ref.append((k, np.asarray(o["centre_mm"], float)))
    return ref


def name_by_partner(openings: list[dict], partner_prof: dict, transform=None) -> dict:
    c = np.array([o["centre_mm"] for o in openings])
    if transform is not None:
        c = transform(c)
    keys, dist = _assign(c, partner_keys(partner_prof))
    return {"keys": keys, "distance_mm": dist}


def _pca_frame(P: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    c = P.mean(0); _, _, vt = np.linalg.svd(P - c, full_matrices=False)
    return c, vt


def register_rigid(src: np.ndarray, dst: np.ndarray, iters: int = 60) -> tuple[np.ndarray, np.ndarray, float]:
    """Rigid (rotation + translation, no reflection) ICP from src to dst, started from the four proper PCA alignments;
    returns R, t and the median closest-point distance (mm)."""
    from scipy.spatial import cKDTree
    tree = cKDTree(dst)
    rng = np.random.default_rng(0)
    s = src[rng.choice(len(src), min(len(src), 20000), replace=False)]
    cs, vs = _pca_frame(src); cd, vd = _pca_frame(dst)
    best = (None, None, np.inf)
    for signs in itertools.product((1, -1), repeat=3):
        S = np.diag(signs)
        R = vd.T @ S @ vs
        if np.linalg.det(R) < 0:
            continue
        t = cd - R @ cs
        for _ in range(iters):
            x = s @ R.T + t
            d, j = tree.query(x)
            keep = d < np.percentile(d, 90)
            a, b = s[keep], dst[j[keep]]
            ma, mb = a.mean(0), b.mean(0)
            U, _, Vt = np.linalg.svd((a - ma).T @ (b - mb))
            D = np.diag([1, 1, np.sign(np.linalg.det(Vt.T @ U.T))])
            R = Vt.T @ D @ U.T; t = mb - R @ ma
        med = float(np.median(tree.query(s @ R.T + t)[0]))
        if med < best[2]:
            best = (R, t, med)
    return best


def name_by_deploy(stl: Path, work: Path, openings: list[dict]) -> dict:
    """The deployment stage A (clean STL -> vessel_geom centreline -> automatic proposal), then endpoints -> STL loops."""
    from wss_deploy import pipeline as dp
    job = work / "naming_deploy"
    a = json.loads((job / "stage_a.json").read_text()) if (job / "stage_a.json").exists() else dp.stage_a(stl, job)
    prop = a.get("proposal") or {}
    ends = prop.get("endpoints", [])
    if len(ends) != 5 or not prop.get("auto_ok"):
        return {"ok": False, "reason": f"proposal not usable: {len(ends)} endpoints, flags {prop.get('flags')}"}
    ref = [(DEPLOY_KEY.get(e["auto_name"], e["auto_name"]), np.asarray(e["center_mm"], float)) for e in ends]
    keys, dist = _assign(np.array([o["centre_mm"] for o in openings]), ref)
    return {"ok": True, "keys": keys, "distance_mm": dist, "confidence": prop.get("confidence"), "side_confidence": prop.get("side_confidence"),
            "scores": prop.get("scores"), "flags": prop.get("flags"), "confirmation_required": prop.get("confirmation_required")}


# ----------------------------------------------------------------------------------------------------------- protocol UDF
def cut_areas(case: Path, key_zone: dict[str, str]) -> dict[str, float]:
    from wss_pinn.v4.fluent_topology import anatomy_topology, read_fluent_mesh
    m = read_fluent_mesh(case, keep_interior=True)
    by = {it.distal_bc_name: it.area_m2 for it in anatomy_topology(m)["interfaces"]}
    return {k: float(by[z]) for k, z in key_zone.items()}


def protocol_udf_text(tprof: dict, case: Path, unit: str) -> tuple[str, dict]:
    areas = cut_areas(case, tprof["udf_thread_zones"])
    rcr = udf.protocol_rcr(areas, cohort(unit))
    text = udf.render(udf.read_udf(tprof["udf"]), tprof["udf_constants"]["threads"], rcr=rcr)
    return text, {"cut_area_mm2": {k: round(v * 1e6, 4) for k, v in areas.items()}, "rcr": rcr, "cohort": cohort(unit), "A1_kg_s": udf.A1_KG_S[cohort(unit)]}


def check_final_udf(work: Path, tprof: dict, final: Path, unit: str) -> dict:
    """The UDF next to the final case: thread ids = final zone ids, inlet divisor = final inlet area, RCR = protocol on
    the final cut faces."""
    from wss_pinn.v4.fluent_topology import read_fluent_mesh
    c = refcase.parse_udf(udf.read_udf(work / Path(tprof["udf"]).name))
    m = read_fluent_mesh(final, keep_interior=False)
    ids = {m.zone_name(z): int(z) for z in m.zone_names}
    want_ids = {k: ids[z] for k, z in tprof["udf_thread_zones"].items()}
    rcr = udf.protocol_rcr(cut_areas(final, tprof["udf_thread_zones"]), cohort(unit))
    worst = max(abs(c["rcr"][k][n] / rcr[k][n] - 1) for k in KEYS for n in ("R1", "R2", "C"))
    a_in = pipeline.inlet_bc_area(m, tprof)
    out = {"threads_ok": c["threads"] == want_ids, "threads": c["threads"], "inlet_divisor_rel": c["inlet_area_m2"] / a_in - 1,
           "rcr_max_rel_vs_protocol": worst, "rcr": c["rcr"]}
    out["ok"] = bool(out["threads_ok"] and abs(out["inlet_divisor_rel"]) < 1e-6 and worst < 1e-3)
    return out


# ----------------------------------------------------------------------------------------------------------- own mesh
def own_mesh_renames(own: dict, tprof: dict) -> dict[str, str]:
    """Own zone -> template zone, per opening key (anatomy fluid/wall, each opening's BC, interface, extension fluid/wall)."""
    ren = {own["anatomy_zone"]: tprof["anatomy_zone"], own["anatomy_wall_zones"][0]: tprof["anatomy_wall_zones"][0]}
    def by_key(p):
        zk = {z: k for k, z in p["udf_thread_zones"].items()}
        out = {}
        for o, e in zip(p["openings"], p["extensions"]):
            assert o["name"] == e["opening"]
            out["inlet" if o["bc_type"] == "velocity-inlet" else zk[o["name"]]] = e
        return out
    a, b = by_key(own), by_key(tprof)
    if set(a) != set(b) or len(a) != 5:
        raise RuntimeError(f"opening keys differ: own {sorted(a)} template {sorted(b)}")
    for k in a:
        ren[a[k]["opening"]] = b[k]["opening"]; ren[a[k]["fluid_zone"]] = b[k]["fluid_zone"]; ren[a[k]["wall_zone"]] = b[k]["wall_zone"]
        if len(a[k]["interface_zone"]) != 1 or len(b[k]["interface_zone"]) != 1:
            raise RuntimeError(f"{k}: interface zones {a[k]['interface_zone']} / {b[k]['interface_zone']}")
        ren[a[k]["interface_zone"][0]] = b[k]["interface_zone"][0]
    return {o: n for o, n in ren.items() if o != n}


def own_key_zones(own: dict, lib_names: dict[str, str]) -> dict[str, str]:
    """Outlet key -> own zone from the zone names (the unit's UDF thread ids are wrong for some units)."""
    out = {}
    for o in own["openings"]:
        if o["bc_type"] == "velocity-inlet":
            continue
        k = lib_names.get(o["name"])
        if k is None:
            raise KeyError(f"no semantic key for own outlet zone {o['name']}")
        out[k] = o["name"]
    if sorted(out) != sorted(KEYS):
        raise RuntimeError(f"own outlet keys {out}")
    return out


def stage_own_mesh(case_dir: Path, work: Path, own: dict, tprof: dict) -> tuple[Path, dict]:
    (work / "mesh").mkdir(exist_ok=True)
    # the copy sits next to the UDF source: auto-compilation looks for it in the case file's directory
    copy = refcase.copy_for_fluent(case_dir, work, dest_name="own_source.cas.gz")
    renames = own_mesh_renames(own, tprof)
    # boundary types follow the template (LI_JIE-1 outlets were left as outflow, type 36)
    ttype = {o["name"]: o["bc_type"] for o in tprof["openings"]}
    types = {renames.get(o["name"], o["name"]): ttype[renames.get(o["name"], o["name"])] for o in own["openings"]
             if o["bc_type"] != ttype[renames.get(o["name"], o["name"])]}
    rep = {"source": copy["source"], "renames": renames, "zone_types": types}
    if not renames and not types:
        _dump(work / "own_mesh_report.json", rep)
        return Path(copy["copy"]), rep
    # reading the copy auto-compiles the unit's UDF from the work directory: give it a compilable source under that name
    udf.write_udf(work / Path(own["udf"]).name, udf.read_udf(tprof["udf"]))
    out = work / "mesh" / "own_renamed.cas.gz"
    jou = work / "rename.jou"
    jou.write_text(journals.rename_zones(work, Path(copy["copy"]), renames, out, zone_types=types))
    slurm.run(work, jou, ntasks=4)
    shutil.rmtree(work / "libudf", ignore_errors=True); shutil.rmtree(work / "mesh" / "libudf", ignore_errors=True)
    (work / Path(own["udf"]).name).unlink(missing_ok=True)
    _dump(work / "own_mesh_report.json", rep)
    return out, rep


# ----------------------------------------------------------------------------------------------------------- calibration
def surface_density(msh: Path, wall_zone: str) -> dict:
    P, zones = meshcheck.read_boundary_msh(msh)
    f = zones[wall_zone]
    v = P[f] * 1e3
    area = float(0.5 * np.linalg.norm(np.cross(v[:, 1] - v[:, 0], v[:, 2] - v[:, 0]), axis=1).sum())
    nodes = int(len(np.unique(f)))
    return {"wall_nodes": nodes, "wall_area_mm2": round(area, 1), "per_cm2": nodes / area * 100}


def calibrate(work: Path, surf: dict, prof: dict, target_per_cm2: float, base: dict, rounds: int = 3) -> dict:
    """Parallel surface-only remeshes; pick the size field whose dual wall-face density is closest to the target."""
    cal = work / "calib"; cal.mkdir(exist_ok=True)
    wall = prof["anatomy_wall_zones"][0]
    sets = [dict(min_size=base["min_size"], max_size=base["max_size"], curvature_angle=18),
            dict(min_size=base["min_size"] * 1.25, max_size=base["max_size"], curvature_angle=22),
            dict(min_size=base["min_size"] * 1.25, max_size=base["max_size"], curvature_angle=26),
            dict(min_size=base["min_size"] * 1.5, max_size=base["max_size"] * 1.1, curvature_angle=30)]
    history = []
    for r in range(rounds):
        jobs = []
        for i, s in enumerate(sets):
            tag = f"r{r}_{i}"
            out = cal / f"surf_{tag}.msh.gz"
            jou = work / f"calib_{tag}.jou"
            text = journals.surface_remesh(work, work / "surface_extended.msh.gz", surf["zones"], out, s["min_size"], s["max_size"], curvature_angle=s["curvature_angle"])
            if out.exists() and jou.exists() and jou.read_text() == text:      # same journal already written this surface
                jobs.append((s, out, ("reused", None))); continue
            jou.write_text(text)
            jobs.append((s, out, slurm.submit(work, jou, mode="meshing", ntasks=4, time_limit="01:00:00")))
        for s, out, (job, log) in jobs:
            state = "COMPLETED" if job == "reused" else slurm.wait(job)
            row = {"round": r, **{k: round(v, 6) if isinstance(v, float) else v for k, v in s.items()}, "job": job, "state": state}
            if state == "COMPLETED" and out.exists():
                row.update(surface_density(out, wall)); row["rel"] = row["per_cm2"] / target_per_cm2 - 1
            history.append(row)
        ok = [h for h in history if "rel" in h]
        if not ok:
            raise RuntimeError(f"calibration: no surface written ({history})")
        best = min(ok, key=lambda h: abs(h["rel"]))
        if abs(best["rel"]) <= DENSITY_TOL:
            break
        # density ~ 1 / size^2: rescale the best set's sizes, bracket with +-8 %
        f = np.sqrt(1 + best["rel"])
        sets = [dict(min_size=best["min_size"] * f * g, max_size=best["max_size"] * f * g, curvature_angle=best["curvature_angle"]) for g in (0.92, 0.97, 1.03, 1.08)]
    rep = {"target_per_cm2": target_per_cm2, "history": history, "chosen": best, "within_tol": abs(best["rel"]) <= DENSITY_TOL}
    _dump(work / "calibration.json", rep)
    if not rep["within_tol"]:
        raise RuntimeError(f"calibration: best density {best['rel']:+.1%} off target after {rounds} rounds")
    return {"min_size": best["min_size"], "max_size": best["max_size"], "curvature_angle": best["curvature_angle"]}


# ----------------------------------------------------------------------------------------------------------- driver
def load_profile(unit: str, prof_dir: Path, library: Path) -> dict:
    p = prof_dir / (unit.replace("/", "__") + ".json")
    if p.exists():
        d = json.loads(p.read_text())
        if "error" not in d:
            return d
    return refcase.profile(library / unit)


def run_unit(unit: str, spec: dict, work_root: Path, prof_dir: Path, library: Path, cores: int, submit: bool) -> Path:
    case_dir = library / unit
    work = (work_root / unit).resolve(); work.mkdir(parents=True, exist_ok=True); guard.assert_inside(work, work)
    tmpl_dir = library / spec["template"]
    tprof = load_profile(spec["template"], prof_dir, library)
    name = spec["case_name"]
    rep = {"unit": unit, "spec": spec, "template_profile": {k: tprof[k] for k in ("case_dir", "udf", "family", "udf_thread_zones", "anatomy_zone", "anatomy_wall_zones")}}
    if spec["mode"] == "own-mesh":
        own = load_profile(unit, prof_dir, library)
        own["udf_thread_zones"] = own_key_zones(own, spec["own_outlet_keys"])
        mesh_case, rep["own_mesh"] = stage_own_mesh(case_dir, work, own, tprof)
    elif spec["mode"] == "stl":
        stl = case_dir / spec["stl"]
        if spec.get("repair_max_move_mm"):
            surface.REPAIR_MAX_MOVE_MM = float(spec["repair_max_move_mm"])
        pts, openings = stl_openings(stl)
        if len(openings) != 5:
            raise RuntimeError(f"{stl.name}: {len(openings)} openings")
        naming = {}
        for method in spec["naming"]:
            if method["method"] == "partner":
                naming["partner"] = name_by_partner(openings, load_profile(method["unit"], prof_dir, library))
            elif method["method"] == "registration":
                pprof = load_profile(method["unit"], prof_dir, library)
                pstl, _ = surface.read_stl(library / method["unit"] / method["stl"])
                R, t, med = register_rigid(pts, pstl)
                n = name_by_partner(openings, pprof, transform=lambda c: c @ R.T + t)
                naming["registration"] = {**n, "icp_median_mm": round(med, 3), "rotation": R.round(5).tolist(), "translation_mm": t.round(3).tolist()}
            elif method["method"] == "deploy":
                naming["deploy"] = name_by_deploy(stl, work, openings)
        usable = {k: v["keys"] for k, v in naming.items() if v.get("keys")}
        agree = len({json.dumps({str(i): usable[m][i] for i in sorted(usable[m])}) for m in usable}) == 1
        primary = spec["naming"][0]["method"]
        rep["naming"] = {"openings": [{**o, "centre_mm": np.round(o["centre_mm"], 3).tolist(), "area_mm2": round(o["area_mm2"], 3)} for o in openings],
                         "methods": {k: {kk: (vv if kk != "keys" else {str(i): x for i, x in vv.items()}) for kk, vv in v.items()} for k, v in naming.items()},
                         "primary": primary, "all_agree": agree}
        _dump(work / "naming.json", rep["naming"])
        if not agree and not spec.get("naming_override_ok"):
            raise RuntimeError(f"opening naming methods disagree: {rep['naming']['methods']}")
        keys = usable[primary]
        tzone = {"inlet": next(o["name"] for o in tprof["openings"] if o["bc_type"] == "velocity-inlet"), **tprof["udf_thread_zones"]}
        names = {i: (tzone[k], "velocity-inlet" if k == "inlet" else "pressure-outlet") for i, k in keys.items()}
        prof = rebuild.lost_case_profile(case_dir, tprof, names, Path(tprof["udf"]), stl=stl)
        prof["naming"] = rep["naming"]
        if spec.get("inlet_length_mm"):
            for e in prof["extensions"]:
                if e["opening"] == tzone["inlet"]:
                    e["length_mm"] = float(spec["inlet_length_mm"])
        _dump(work / "reference_profile.json", prof)
        surf = pipeline.stage_surface(case_dir, work, prof, stl_path=stl)
        if tprof["family"] == "poly":
            ref = spec["density_ref"]
            target = float(np.median([load_profile(u, prof_dir, library)["dual_wall_faces_per_cm2"] for u in ref]))
            base = spec.get("calib_base") or {"min_size": tprof["poly_mesh_estimate"]["min_size_mm"] * 1e-3, "max_size": tprof["poly_mesh_estimate"]["max_size_mm"] * 1e-3}
            params = json.loads((work / "calibration.json").read_text())["chosen"] if (work / "calibration.json").exists() else None
            params = {k: params[k] for k in ("min_size", "max_size", "curvature_angle")} if params else calibrate(work, surf, prof, target, base)
        else:
            params = None
        if spec.get("mesh_improve_skew"):
            params = {**(params or {}), "improve_skew": float(spec["mesh_improve_skew"])}
        mesh_file = pipeline.stage_mesh(case_dir, work, surf, prof, mesh_params=params, reference_mesh=False)
        # boundary-layer gate: Fluent Meshing deletes a prism layer that fails its quality check and stops growing,
        # silently leaving a tet-only mesh (LI_FA_XIANG-1/before, 2026-09-29)
        bl = json.loads((work / "mesh_gate.json").read_text())["new"].get("bl_layers_mode")
        if bl != 10:
            raise RuntimeError(f"boundary-layer gate: {bl} prism layers at the wall (expected 10); see logs/mesh_*.out")
        mesh_case = pipeline.stage_finalize(work, mesh_file, prof)
        rep["mesh_params"] = params
    else:
        raise ValueError(spec["mode"])
    text, rep["protocol"] = protocol_udf_text(tprof, mesh_case, unit) if spec["mode"] == "stl" else _own_protocol(tprof, mesh_case, unit)
    _dump(work / "protocol_bc.json", rep["protocol"])
    final = pipeline.stage_setup(tmpl_dir, work, tprof, mesh_case, name, udf_text=text, udf_zones=tprof["udf_thread_zones"])
    rep["final_udf_check"] = check_final_udf(work, tprof, final, unit)
    if not rep["final_udf_check"]["ok"]:
        _dump(work / "recover_report.json", rep)
        raise RuntimeError(f"final UDF check failed: {rep['final_udf_check']}")
    pipeline.stage_smoke(work, final)
    pipeline.stage_run_files(case_dir, work, final, cores, slurm_from=tmpl_dir)
    _dump(work / "recover_report.json", rep)
    print(f"ready: {work}")
    if submit:
        job = subprocess.run(["sbatch", "--parsable", "fluent.slurm"], cwd=work, capture_output=True, text=True, env=slurm._env(), check=True).stdout.strip()
        (work / "logs").mkdir(exist_ok=True); (work / "logs" / "cfd_job_id").write_text(job + "\n")
        print(f"submitted CFD job {job}")
    return work


def _own_protocol(tprof: dict, mesh_case: Path, unit: str) -> tuple[str, dict]:
    """Own meshes are renamed to the template's zone names before this point, so the template key -> zone map holds."""
    return protocol_udf_text(tprof, mesh_case, unit)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("unit"); ap.add_argument("--plan", type=Path, required=True); ap.add_argument("--work-root", type=Path, required=True)
    ap.add_argument("--profiles", type=Path, default=None, help="directory of pre-computed refcase profiles (read-only scan)")
    ap.add_argument("--library", type=Path, default=pipeline.LIBRARY); ap.add_argument("--cores", type=int, default=92)
    ap.add_argument("--submit", action="store_true")
    a = ap.parse_args()
    plan = json.loads(a.plan.read_text())
    run_unit(a.unit, plan["units"][a.unit], a.work_root, a.profiles or a.plan.parent / "profiles", a.library, a.cores, a.submit)


if __name__ == "__main__":
    main()

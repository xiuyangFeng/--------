"""Rebuild a library case whose own .cas is lost (e.g. AAA/ruputer/YANG_BAO_KUI, overwritten 2026-09-04) from its STL,
with the solver settings of a same-template library case and the lost case's own UDF constants.

    python -m cfd_auto.rebuild AAA/ruputer/YANG_BAO_KUI --template AAA/ruputer/YU_TIAN_HAI \
        --original-log Global_conditions/Fluent_14165.out --work-root outputs/cfd_auto_trial_20260927 [--submit]

Openings are named without the lost mesh: the original run's UDF printed every outlet's boundary area
(``sum_area_out*``); STL outlet loops are assigned to the UDF keys by area under a left/right constraint taken from the
template (left outlets on the template's left of the inlet centre), and the key -> zone-name map is the template's.
Extension lengths follow the library rule of the template's family (inlet length of the template; outlets
``OUTLET_L_OVER_D`` x equivalent diameter, fitted on the 2026-01 AAA poly batch: 9.9-10.8, median ~10.2).
"""
from __future__ import annotations

import argparse
import itertools
import json
import re
import subprocess
from pathlib import Path

import numpy as np

from cfd_auto import guard, pipeline, refcase, slurm, surface, udf

OUTLET_L_OVER_D = 10.2


def original_outlet_areas(log: Path) -> dict[str, float]:
    areas = {}
    for line in open(log, errors="replace"):
        for k, v in re.findall(r"sum_area_(out\w\w)=([0-9.eE+-]+)", line):
            areas[k] = float(v) * 1e6
    if len(areas) != 4:
        raise ValueError(f"{log}: outlet areas not printed ({areas})")
    return areas


def name_openings(stl: Path, tprof: dict, printed_mm2: dict[str, float]) -> tuple[dict[int, tuple[str, str]], dict]:
    pts, tri = surface.read_stl(stl)
    loops = surface.boundary_loops(tri)
    info = []
    for l in loops:
        P = pts[l]; c = P.mean(0); _, _, vt = np.linalg.svd(P - c)
        uv = np.c_[(P - c) @ vt[0], (P - c) @ vt[1]]
        info.append({"centre_mm": c, "area_mm2": 0.5 * abs(np.dot(uv[:, 0], np.roll(uv[:, 1], -1)) - np.dot(uv[:, 1], np.roll(uv[:, 0], -1)))})
    inlet = int(np.argmax([i["area_mm2"] for i in info]))
    t_open = {o["name"]: o for o in tprof["openings"]}
    t_inlet = next(o for o in tprof["openings"] if o["bc_type"] == "velocity-inlet")
    key_zone = tprof["udf_thread_zones"]                                   # e.g. outle -> outlet-lw
    side = lambda x, x_in: "L" if x < x_in else "R"
    key_side = {k: side(t_open[z]["centre_mm"][0], t_inlet["centre_mm"][0]) for k, z in key_zone.items()}
    outlets = [i for i in range(len(loops)) if i != inlet]
    keys = list(key_zone)
    best, best_cost = None, np.inf
    for perm in itertools.permutations(outlets):
        if any(key_side[k] != side(info[i]["centre_mm"][0], info[inlet]["centre_mm"][0]) for k, i in zip(keys, perm)):
            continue
        cost = sum(abs(np.log(info[i]["area_mm2"] / printed_mm2[k])) for k, i in zip(keys, perm))
        if cost < best_cost:
            best, best_cost = perm, cost
    if best is None:
        raise RuntimeError("no outlet assignment satisfies the template's left/right layout")
    names = {inlet: (t_inlet["name"], "velocity-inlet")}
    rows = []
    for k, i in zip(keys, best):
        names[i] = (key_zone[k], "pressure-outlet")
        rows.append({"udf_key": k, "zone": key_zone[k], "loop": i, "stl_area_mm2": round(info[i]["area_mm2"], 2), "printed_area_mm2": printed_mm2[k],
                     "ratio": round(info[i]["area_mm2"] / printed_mm2[k], 3), "side": key_side[k]})
    return names, {"inlet_loop": inlet, "inlet_area_mm2": round(info[inlet]["area_mm2"], 2), "assignment": rows, "cost": round(float(best_cost), 4)}


def lost_case_profile(case_dir: Path, tprof: dict, names: dict[int, tuple[str, str]], udf_path: Path) -> dict:
    """A reference_profile-like dict for the lost case: openings from the STL, extension naming from the template."""
    pts, tri = surface.read_stl(sorted(case_dir.glob("*.stl"))[0])
    surf, _ = surface.close_surface(sorted(case_dir.glob("*.stl"))[0], names)
    t_ext = {e["opening"]: e for e in tprof["extensions"]}
    openings, extensions = [], []
    for c in surf.caps:
        openings.append({"name": c.name, "bc_type": c.bc_type, "centre_mm": c.centre_mm.round(4).tolist(), "outward_normal": c.normal.round(6).tolist(), "area_mm2": round(c.area_mm2, 4)})
        te = t_ext[c.name]
        L = te["length_mm"] if c.bc_type == "velocity-inlet" else OUTLET_L_OVER_D * 2 * np.sqrt(c.area_mm2 / np.pi)
        extensions.append({**{k: te[k] for k in ("opening", "fluid_zone", "wall_zone", "interface_zone")}, "length_mm": round(float(L), 3)})
    consts = refcase.parse_udf(udf.read_udf(udf_path))
    return {"case_dir": str(case_dir), "udf": str(udf_path), "udf_constants": consts, "udf_thread_zones": tprof["udf_thread_zones"],
            "anatomy_zone": tprof["anatomy_zone"], "anatomy_wall_zones": tprof["anatomy_wall_zones"], "family": tprof["family"],
            "poly_mesh_estimate": tprof.get("poly_mesh_estimate"), "openings": openings, "extensions": extensions, "template": tprof["case_dir"],
            "note": "lost-case rebuild: no library mesh; openings named from the original run's UDF-printed outlet areas"}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("case_id"); ap.add_argument("--template", required=True); ap.add_argument("--original-log", required=True)
    ap.add_argument("--work-root", type=Path, required=True); ap.add_argument("--library", type=Path, default=pipeline.LIBRARY)
    ap.add_argument("--mesh-params", type=json.loads, default=None, help='JSON overrides, e.g. {"min_size": 0.0004, "max_size": 0.002}')
    ap.add_argument("--cores", type=int, default=64); ap.add_argument("--submit", action="store_true")
    a = ap.parse_args()
    case_dir, tmpl_dir = a.library / a.case_id, a.library / a.template
    work = (a.work_root / a.case_id).resolve(); work.mkdir(parents=True, exist_ok=True); guard.assert_inside(work, work)
    tprof_path = (a.work_root / a.template).resolve() / "reference_profile.json"
    tprof = json.loads(tprof_path.read_text()) if tprof_path.exists() else refcase.profile(tmpl_dir)
    udf_path = refcase.find_udf(case_dir)
    printed = original_outlet_areas(case_dir / a.original_log)
    names, naming = name_openings(sorted(case_dir.glob("*.stl"))[0], tprof, printed)
    prof = lost_case_profile(case_dir, tprof, names, udf_path)
    prof["naming"] = naming
    pipeline._dump(work / "reference_profile.json", prof)
    surf = pipeline.stage_surface(case_dir, work, prof)
    mesh_file = pipeline.stage_mesh(case_dir, work, surf, prof, mesh_params=a.mesh_params, reference_mesh=False)
    ext = pipeline.stage_finalize(work, mesh_file, prof)
    name = case_dir.name
    final = pipeline.stage_setup(tmpl_dir, work, tprof, ext, name, udf_text=udf.read_udf(udf_path), udf_zones=tprof["udf_thread_zones"])
    pipeline.stage_smoke(work, final)
    pipeline.stage_run_files(case_dir, work, final, a.cores)
    print(f"ready: {work}")
    if a.submit:
        job = subprocess.run(["sbatch", "--parsable", "fluent.slurm"], cwd=work, capture_output=True, text=True, env=slurm._env(), check=True).stdout.strip()
        (work / "logs" / "cfd_job_id").write_text(job + "\n")
        print(f"submitted CFD job {job}")


if __name__ == "__main__":
    main()

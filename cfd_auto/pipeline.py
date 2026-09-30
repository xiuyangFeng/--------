"""One library case -> rebuilt Fluent case ready to run (and optionally submitted), with a gate after every stage.

    python -m cfd_auto.pipeline AG/slow/QIN_SI_FU --work-root outputs/cfd_auto_trial_20260927 [--submit]

Stages (each writes a JSON report in the work directory; any failed gate raises):
  1 profile   library case -> reference_profile.json (openings, extension lengths, zone names, UDF constants)
  2 surface   STL + planar caps + straight extensions -> surface_extended.msh.gz          gate: regions closed, no collisions
  3 mesh      Fluent Meshing (10 prism layers AR 20 x1.2, tet geometric 1.2)             gate: anatomy mesh stats vs reference
  4 finalize  solver: region cell zones renamed blood / bloodN by adjacency              gate: mesh check
  5 setup     reference settings copy + replace-mesh + UDF with the final thread ids    gate: settings_diff unexpected == 0
  6 smoke     read the final case (libudf compiled from the regenerated UDF)            gate: 7 UDF hooks, no Fluent error
  7 run files 2.jou + fluent.slurm (library script: 64 cores, same cleanup), empty ascii/; --submit sends it
"""
from __future__ import annotations

import argparse
import json
import re
import shutil
import subprocess
from pathlib import Path

import numpy as np

from cfd_auto import guard, journals, meshcheck, refcase, settings_diff, slurm, surface, udf

ROOT = Path("/public/newhome/cy/Digital_twin/GNN")
LIBRARY = ROOT / "data_new"
UDF_HOOKS = ("my_inlet", "cell_viscosity", "execute_at_end", "pressure_outle", "pressure_outli", "pressure_outri", "pressure_outre")
# anatomy mesh vs library mesh (same STL, same parameters): relative tolerances
MESH_GATE = {"anatomy_cells": 0.05, "tets": 0.10, "volume": 0.001, "first_height_over_edge": 0.05, "bl_total_median": 0.05, "tet_L_median": 0.10}


def _dump(path: Path, obj) -> None:
    path.write_text(json.dumps(obj, indent=1, default=lambda o: o.tolist() if isinstance(o, np.ndarray) else str(o)))


def stage_surface(case_dir: Path, work: Path, prof: dict, stl_path: Path | None = None) -> dict:
    stl = [Path(stl_path)] if stl_path is not None else sorted(case_dir.glob("*.stl"))
    if len(stl) != 1:
        raise FileNotFoundError(f"{case_dir}: expected one STL, found {[p.name for p in stl]}")
    pts, wall = surface.read_stl(stl[0])
    loops = surface.boundary_loops(wall)
    names = surface.name_caps_by_reference(np.array([pts[l].mean(0) for l in loops]), prof["openings"])
    surf, cap_stats = surface.close_surface(stl[0], names)
    # an optional per-extension "direction" (managed builds of oblique cuts, surface.extension_directions); absent = cap normal
    exts = [surface.Extension(e["opening"], e["length_mm"], e["wall_zone"], e["interface_zone"][0], np.asarray(e["direction"], float) if e.get("direction") is not None else None)
            for e in prof["extensions"]]
    P, zones, ext_stats = surface.extend_surface(surf, exts)
    check = surface.check_regions(P, zones, exts)
    info = surface.write_fluent_zones(P, zones, work / "surface_extended.msh.gz")
    rep = {"stl": str(stl[0]), "wall_nodes_equal_stl": bool(np.array_equal(surf.points_mm[: surf.n_wall_nodes], pts)), "self_intersection_repair": surf.repair, "caps": cap_stats,
           "extensions": ext_stats, "regions": check, "boundary_mesh": info}
    if prof.get("extension_directions"):
        rep["extension_directions"] = prof["extension_directions"]
    _dump(work / "surface_report.json", rep)
    if not check["ok"]:
        raise RuntimeError(f"surface gate failed: {check}")
    return {"zones": [z["name"] for z in info["zones"]], "walls": [z["name"] for z in info["zones"] if z["type"] == "wall"]}


def stage_mesh(case_dir: Path, work: Path, surf: dict, prof: dict, mesh_params: dict | None = None, reference_mesh: bool = True) -> Path:
    from wss_pinn.v4.fluent_topology import read_fluent_mesh
    (work / "mesh").mkdir(exist_ok=True)
    out = work / "mesh" / "regions.msh.gz"
    jou = work / "mesh.jou"
    wall = prof["anatomy_wall_zones"][0]
    if prof["family"] == "poly":
        est = prof["poly_mesh_estimate"]
        params = {"min_size": est["min_size_mm"] * 1e-3, "max_size": est["max_size_mm"] * 1e-3, **(mesh_params or {})}
        # volume_fill "poly-hexcore" (managed builds that ask for it): the operator's cell type; default tet (09-28 regression)
        fill = params.pop("volume_fill", "tet")
        build = journals.meshing_polyhexcore if fill == "poly-hexcore" else journals.meshing_poly
        jou.write_text(build(work, work / "surface_extended.msh.gz", surf["walls"], surf["zones"], out, **params))
        params["volume_fill"] = fill
    else:
        # tet-prism: the STL facets are the wall; only prism/tet controls and the managed repair option apply
        params = {k: v for k, v in (mesh_params or {}).items() if k in ("n_layers", "first_aspect_ratio", "growth", "tet_volume_growth", "improve_skew")}
        jou.write_text(journals.meshing(work, work / "surface_extended.msh.gz", surf["walls"], surf["zones"], out, **params))
    slurm.run(work, jou, mode="meshing", ntasks=4, time_limit="02:00:00")
    new = read_fluent_mesh(out, keep_interior=True)
    new_mask = new.cell_zone_array() == _zone_adjacent_to(new, wall)
    ns = meshcheck.gate_summary(new, new_mask, (wall,))
    rep = {"family": prof["family"], "params": params, "new": ns}
    if reference_mesh:
        ref = read_fluent_mesh(refcase.find_case_file(case_dir), keep_interior=True)
        rs = meshcheck.gate_summary(ref, meshcheck.anatomy_mask(ref), (wall,))
        rel, fails = meshcheck.gate_compare(rs, ns)
        rep.update(reference=rs, relative=rel, failed=fails)
        _dump(work / "mesh_gate.json", rep)
        if fails:
            raise RuntimeError(f"mesh gate failed: {fails}")
    else:
        _dump(work / "mesh_gate.json", rep)
    return out


def _zone_adjacent_to(mesh, wall_name: str) -> int:
    cz = mesh.cell_zone_array()
    for sec in mesh.face_sections_of_type(3):
        if meshcheck.zname(mesh, sec.zone_id) == wall_name:
            zs = np.unique(cz[sec.c0])
            if len(zs) != 1:
                raise RuntimeError(f"wall {wall_name} touches {len(zs)} cell zones")
            return int(zs[0])
    raise KeyError(wall_name)


def stage_finalize(work: Path, mesh_file: Path, prof: dict) -> Path:
    from wss_pinn.v4.fluent_topology import read_fluent_mesh
    m = read_fluent_mesh(mesh_file, keep_interior=True)
    renames = {meshcheck.zname(m, _zone_adjacent_to(m, prof["anatomy_wall_zones"][0])): prof["anatomy_zone"]}
    for e in prof["extensions"]:
        renames[meshcheck.zname(m, _zone_adjacent_to(m, e["wall_zone"]))] = e["fluid_zone"]
    if len(set(renames)) != len(prof["extensions"]) + 1:
        raise RuntimeError(f"region/zone mapping is not one-to-one: {renames}")
    out = work / "mesh" / "extended.cas.gz"
    jou = work / "finalize.jou"
    # poly family: v231's solver convert-domain splits the wall faces (99k faces instead of the library's 27k duals), so
    # the rebuilt P-family mesh stays tet/prism at matched dual resolution (see meshcheck.gate_summary)
    jou.write_text(journals.finalize(work, mesh_file, renames, out, to_poly=False))
    slurm.run(work, jou, ntasks=4)
    _dump(work / "finalize_report.json", {"renames": renames})
    return out


def stage_setup(case_dir: Path, work: Path, prof: dict, mesh_case: Path, case_name: str, udf_text: str | None = None, udf_zones: dict | None = None) -> Path:
    """``case_dir`` supplies the solver settings (normally the case itself; a same-template case when the case's own .cas is
    lost). ``udf_text``/``udf_zones`` override the UDF source and its outlet-key -> zone-name map (default: the template's)."""
    from wss_pinn.v4.fluent_topology import read_fluent_mesh
    template = refcase.find_case_file(case_dir).name.split(".cas")[0]
    copy = refcase.copy_for_fluent(case_dir, work, rename_to=case_name if case_name != template else None)
    export_fix = refcase.fix_export_layout(copy["copy"], work)      # no-op unless the template's exports collide / miss ascii/
    ref_udf = udf_text if udf_text is not None else udf.read_udf(prof["udf"])
    udf_name = Path(prof["udf"]).name            # the case's compile list names this file; auto-compile looks for it next to the case
    zones_of = udf_zones or prof["udf_thread_zones"]
    # replace-mesh re-assigns the template's zone ids by name (observed on v231); the UDF is rendered with those ids first and
    # re-rendered below from the written case
    tmpl_ids = {k: prof["udf_constants"]["threads"][k] for k in zones_of}
    udf.write_udf(work / udf_name, udf.render(ref_udf, tmpl_ids))
    shutil.rmtree(work / "libudf", ignore_errors=True)
    final = work / f"{case_name}.cas.gz"
    jou = work / "setup.jou"
    jou.write_text(journals.setup(work, Path(copy["copy"]), mesh_case, final))
    slurm.run(work, jou, ntasks=4)
    m = read_fluent_mesh(final, keep_interior=False)
    ids = {m.zone_name(z): int(z) for z in m.zone_names}
    tid = {k: ids[z] for k, z in zones_of.items()}
    # library protocol: the UDF inlet-area divisor equals the mesh's inlet boundary area (every library case checked has
    # outflow / protocol = 1.103 exactly, the waveform-truncation factor). Keeping the library divisor on a rebuilt inlet
    # scales the inflow by A_new / A_udf (2026-09-28: YANG +2.1 % -> pressure +3 %).
    inlet_area = inlet_bc_area(m, prof)
    udf.write_udf(work / udf_name, udf.render(ref_udf, tid, inlet_area_m2=inlet_area))
    shutil.rmtree(work / "libudf", ignore_errors=True)     # recompiled from the final UDF by the smoke read
    diff = settings_diff.compare(Path(copy["copy"]), final, work, case_dir)
    _dump(work / "settings_diff.json", {**diff, "thread_ids": tid, "thread_zones": zones_of, "udf_file": udf_name,
                                         "inlet_area_m2": {"udf_source": refcase.parse_udf(ref_udf)["inlet_area_m2"], "mesh": inlet_area},
                                         "udf_identical_to_source": udf.read_udf(work / udf_name) == ref_udf, "paths_rewritten": copy["paths_rewritten"],
                                         "export_layout_fix": export_fix})
    if diff["unexpected"]:
        raise RuntimeError(f"settings gate failed: {diff['unexpected'][:3]}")
    return final


def inlet_bc_area(mesh, prof: dict) -> float:
    """Area (m^2) of the velocity-inlet boundary zone named in the profile."""
    name = next(o["name"] for o in prof["openings"] if o["bc_type"] == "velocity-inlet")
    area = [float(np.linalg.norm(mesh.face_geometry(s)[1], axis=1).sum()) for s in mesh.face_sections if mesh.zone_name(s.zone_id) == name]
    if len(area) != 1:
        raise RuntimeError(f"inlet zone {name!r}: {len(area)} face sections")
    return area[0]


def stage_smoke(work: Path, final: Path, quality: bool = False) -> Path:
    jou = work / "smoke.jou"
    jou.write_text(journals.smoke(work, final, quality=quality))
    log = slurm.run(work, jou, ntasks=4)
    text = log.read_text(errors="replace")
    hooks = [h for h in UDF_HOOKS if re.search(rf"^\s+{h}\s*$", text, re.M)]
    if len(hooks) != len(UDF_HOOKS):
        raise RuntimeError(f"smoke: UDF hooks loaded {hooks}")
    return log


def stage_run_files(case_dir: Path, work: Path, final: Path, cores: int, slurm_from: Path | None = None) -> None:
    """``slurm_from``: library unit whose fluent.slurm is adapted (default ``case_dir``)."""
    (work / "2.jou").write_text(journals.run(work, final))
    lib = ((slurm_from or case_dir) / "fluent.slurm").read_text()
    s = re.sub(r"^#SBATCH -w .*\n", "", lib, flags=re.M)
    s = re.sub(r"^#SBATCH --ntasks-per-node=.*$", f"#SBATCH --ntasks-per-node={cores}\n#SBATCH --exclude={slurm.EXCLUDE}", s, flags=re.M)
    s = re.sub(r"^#SBATCH --job-name=.*$", "#SBATCH --job-name=Fluent_cfdauto", s, flags=re.M)
    s = re.sub(r"fluent 3ddp -g -t\d+", f"fluent 3ddp -g -t{cores}", s)
    (work / "fluent.slurm").write_text(s)
    (work / "ascii").mkdir(exist_ok=True)
    if any((work / "ascii").iterdir()):
        raise RuntimeError("ascii/ is not empty: Fluent would silently skip existing export files")
    # some library templates autosave into a relative directory (e.g. "export/NAME.gz"); it must exist
    rp = settings_diff.sections(settings_diff.case_text(final))["rp"]
    auto = re.findall(r'"([^"]*)"', rp.get("autosave/filename", ""))
    if auto and "/" in auto[0] and not auto[0].startswith("/"):
        (work / Path(auto[0]).parent).mkdir(parents=True, exist_ok=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("case_id"); ap.add_argument("--work-root", type=Path, required=True)
    ap.add_argument("--library", type=Path, default=LIBRARY); ap.add_argument("--cores", type=int, default=64)
    ap.add_argument("--mesh-params", type=json.loads, default=None, help='poly family overrides (m / deg), e.g. {"min_size": 0.00075, "max_size": 0.0016, "curvature_angle": 26}')
    ap.add_argument("--submit", action="store_true"); ap.add_argument("--from-stage", default="profile",
                    choices=["profile", "surface", "mesh", "finalize", "setup", "smoke", "run"])
    a = ap.parse_args()
    case_dir = a.library / a.case_id
    work = (a.work_root / a.case_id).resolve()
    work.mkdir(parents=True, exist_ok=True)
    guard.assert_inside(work, work)
    order = ["profile", "surface", "mesh", "finalize", "setup", "smoke", "run"]
    start = order.index(a.from_stage)
    name = refcase.find_case_file(case_dir).name.split(".cas")[0]
    prof = refcase.profile(case_dir) if start == 0 else json.loads((work / "reference_profile.json").read_text())
    if start == 0:
        _dump(work / "reference_profile.json", prof)
    surf = stage_surface(case_dir, work, prof) if start <= 1 else json.loads((work / "surface_report.json").read_text())
    if start > 1:
        zs = surf["boundary_mesh"]["zones"]; surf = {"zones": [z["name"] for z in zs], "walls": [z["name"] for z in zs if z["type"] == "wall"]}
    mesh_file = stage_mesh(case_dir, work, surf, prof, mesh_params=a.mesh_params) if start <= 2 else work / "mesh" / "regions.msh.gz"
    ext = stage_finalize(work, mesh_file, prof) if start <= 3 else work / "mesh" / "extended.cas.gz"
    final = stage_setup(case_dir, work, prof, ext, name) if start <= 4 else work / f"{name}.cas.gz"
    if start <= 5:
        stage_smoke(work, final)
    stage_run_files(case_dir, work, final, a.cores)
    print(f"ready: {work}")
    if a.submit:
        job = subprocess.run(["sbatch", "--parsable", "fluent.slurm"], cwd=work, capture_output=True, text=True, env=slurm._env(), check=True).stdout.strip()
        (work / "logs" / "cfd_job_id").write_text(job + "\n")
        print(f"submitted CFD job {job}")


if __name__ == "__main__":
    main()

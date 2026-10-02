"""Managed preparation of one unit: geometry -> named openings -> surface + extensions -> mesh (with repair ladder and
quality gate) -> protocol boundary conditions -> set-up checks -> case + UDF -> smoke -> managed run files. Runs inside a
Slurm driver job started by ``cfd_auto.orchestrate`` (it submits and waits for its own small Fluent jobs).

Unit specs (batch file ``units`` entries; the 2026-09-29 recover plan entries are valid specs):
  mode "own-mesh"  a library unit's own mesh renamed to the template's zone names (``own_outlet_keys``)
  mode "stl"       an STL — in the library (``stl`` relative to the unit dir) or anywhere (absolute path: a NEW case)
Defaults for a new case come from the protocol (``templates[cohort]``: settings template, mesh family, density reference;
naming by the deployment proposal; extension direction ``direction_for_new_cases``; inlet length by cohort).

Differences from ``recover.run_unit`` (which stays as it was, for the 2026-09-29 batch): the naming gate (a person must
confirm names below the protocol confidence, ``naming_confirmed.json``), oblique-cut extensions, the mesh repair ladder
with the orthogonal-quality / aspect-ratio / boundary-layer gate, ``preflight.json`` (protocol applicability; a failure
blocks the unit), exports limited to the needed frames and a requeue-safe Slurm script (``cfd_auto.schedule``), and a
provenance manifest.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

from cfd_auto import guard, meshcheck, pipeline, preflight, rebuild, recover, refcase, schedule, settings_diff, surface, udf

LABEL = {"inlet": "inlet", "outle": "L ext. iliac (outle)", "outli": "L int. iliac (outli)", "outri": "R int. iliac (outri)", "outre": "R ext. iliac (outre)"}


class Blocked(RuntimeError):
    """The unit needs a person (naming confirmation, protocol not applicable, mesh gate failed after every repair)."""

    def __init__(self, kind: str, message: str, detail=None):
        super().__init__(message)
        self.kind, self.detail = kind, detail


def _dump(path: Path, obj) -> None:
    pipeline._dump(path, obj)


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def resolve_spec(unit: str, spec: dict, proto: dict, library: Path) -> dict:
    s = dict(spec)
    s.setdefault("cohort", unit.split("/")[0])
    if s["mode"] == "stl":
        stl = Path(s["stl"])
        s["stl_path"] = str(stl if stl.is_absolute() else library / unit / stl)
        s["fresh"] = not Path(s["stl_path"]).resolve().is_relative_to(library.resolve())
    else:
        s["fresh"] = False
    t = proto["templates"].get(s["cohort"], {})
    s.setdefault("template", t.get("template"))
    s.setdefault("density_ref", t.get("density_ref", []))
    s.setdefault("naming", [{"method": "deploy"}])
    s.setdefault("case_name", default_case_name(unit))
    s.setdefault("extension_direction", proto["extension"]["direction_for_new_cases"] if s["fresh"] else proto["extension"]["direction"])
    if not s["template"]:
        raise ValueError(f"{unit}: no settings template (spec or protocol templates[{s['cohort']}])")
    return s


def default_case_name(unit: str) -> str:
    parts = [p for p in unit.split("/") if p not in ("before", "after")]
    return parts[-1].split("-")[0] if unit.endswith(("/before", "/after")) else parts[-1]


def naming_figure(stl: Path, openings: list[dict], keys: dict, naming: dict, out: Path, title: str) -> None:
    """Frontal / lateral projections with every opening labelled (for the person who confirms the names)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt       # ASCII labels only: compute nodes have no CJK font
    pts, _ = surface.read_stl(stl)
    sub = pts[np.random.default_rng(0).choice(len(pts), min(len(pts), 40000), replace=False)]
    fig, axes = plt.subplots(1, 2, figsize=(11, 7))
    for ax, (a, b, lab) in zip(axes, ((0, 2, "x-z (front)"), (1, 2, "y-z (side)"))):
        ax.scatter(sub[:, a], sub[:, b], s=0.2, c="0.6")
        for o in openings:
            k = keys[int(o["loop"])]; c = np.asarray(o["centre_mm"])
            ax.scatter(c[a], c[b], s=60, c="tab:red" if k == "inlet" else ("tab:blue" if k in ("outle", "outli") else "tab:green"), zorder=3)
            ax.annotate(f"{LABEL.get(k, k)}\n{o['area_mm2']:.1f} mm2", (c[a], c[b]), fontsize=9, xytext=(6, 6), textcoords="offset points")
        ax.set_aspect("equal"); ax.set_title(lab); ax.set_xlabel(["x", "y"][a] + " (mm)"); ax.set_ylabel("z (mm)")
    dep = naming.get("methods", {}).get("deploy", {})
    fig.suptitle(f"{title}\nmethods {', '.join(naming.get('methods', {}))}; agree {naming.get('all_agree')}; deploy confidence {dep.get('confidence')} "
                 f"side {dep.get('side_confidence')}", fontsize=10)
    fig.tight_layout(); fig.savefig(out, dpi=110); plt.close(fig)


def name_openings(unit: str, s: dict, stl: Path, work: Path, openings: list[dict], prof_dir: Path, library: Path) -> tuple[dict[int, str], dict, bool]:
    """Loop -> UDF key by every naming method of the spec (they must agree); a person's ``naming_confirmed.json``
    ({"keys": {"<loop>": "<key>"}}) overrides them. Returns (keys, naming report, confirmed)."""
    naming = {}
    pts = None
    for m in s["naming"]:
        if m["method"] == "partner":
            naming["partner"] = recover.name_by_partner(openings, recover.load_profile(m["unit"], prof_dir, library))
        elif m["method"] == "registration":
            pts = pts if pts is not None else surface.read_stl(stl)[0]
            pprof = recover.load_profile(m["unit"], prof_dir, library)
            pstl, _ = surface.read_stl(library / m["unit"] / m["stl"])
            R, t, med = recover.register_rigid(pts, pstl)
            n = recover.name_by_partner(openings, pprof, transform=lambda c: c @ R.T + t)
            naming["registration"] = {**n, "icp_median_mm": round(med, 3)}
        elif m["method"] == "deploy":
            naming["deploy"] = recover.name_by_deploy(stl, work, openings)
    usable = {k: v["keys"] for k, v in naming.items() if v.get("keys")}
    agree = len({json.dumps({str(i): usable[m][i] for i in sorted(usable[m])}) for m in usable}) == 1 if usable else False
    rep = {"openings": [{**o, "centre_mm": np.round(o["centre_mm"], 3).tolist(), "area_mm2": round(float(o["area_mm2"]), 3)} for o in openings],
           "methods": {k: {kk: (vv if kk != "keys" else {str(i): x for i, x in vv.items()}) for kk, vv in v.items()} for k, v in naming.items()},
           "primary": s["naming"][0]["method"], "all_agree": agree}
    conf_file = work / "naming_confirmed.json"
    if conf_file.exists():
        c = json.loads(conf_file.read_text())
        keys = {int(i): k for i, k in c["keys"].items()}
        rep.update(confirmed=c)
        return keys, rep, True
    primary = rep["primary"]
    if primary not in usable:
        raise Blocked("naming", f"primary naming method {primary} gave no names: {naming.get(primary)}", rep)
    return dict(usable[primary]), rep, False


def _directions(stl: Path, names: dict[int, tuple[str, str]], mode: str, proto: dict, repair_max: float | None) -> dict:
    if repair_max:
        surface.REPAIR_MAX_MOVE_MM = float(repair_max)
    surf, _ = surface.close_surface(stl, names)
    return surface.extension_directions(surf, mode, proto["extension"]["oblique_deg"])


def dual_density(unit: str, prof_dir: Path, library: Path) -> float:
    """Dual wall-face density (per cm²) of a library unit's anatomy wall, the calibration target of P-family remeshes;
    computed once from the library mesh (Python read only) and cached in ``prof_dir``."""
    p = recover.load_profile(unit, prof_dir, library)
    if "dual_wall_faces_per_cm2" in p:
        return float(p["dual_wall_faces_per_cm2"])
    from wss_pinn.v4.fluent_topology import anatomy_topology, anatomy_wall_faces, read_fluent_mesh
    m = read_fluent_mesh(refcase.find_case_file(library / unit), keep_interior=True)
    g = meshcheck.gate_summary(m, meshcheck.anatomy_mask(m), tuple(p["anatomy_wall_zones"]))
    w = anatomy_wall_faces(m, anatomy_topology(m))
    area = float(np.linalg.norm(w["area_vectors_m2"], axis=1).sum() * 1e6)
    p.update(gate_summary=g, anatomy_wall_area_mm2=area, dual_wall_faces_per_cm2=g["dual_wall_faces"] / area * 100)
    prof_dir.mkdir(parents=True, exist_ok=True)
    _dump(prof_dir / (unit.replace("/", "__") + ".json"), p)
    return float(p["dual_wall_faces_per_cm2"])


def _latest_log(work: Path, tag: str) -> Path:
    logs = sorted((work / "logs").glob(f"{tag}_*.out"), key=lambda p: p.stat().st_mtime)
    if not logs:
        raise FileNotFoundError(f"no {tag} log in {work / 'logs'}")
    return logs[-1]


def mesh_with_repair(case_dir: Path, work: Path, surf: dict, prof: dict, base: dict | None, proto: dict, start_skew: float | None = None) -> tuple[Path, dict]:
    """Mesh + finalize up the protocol's repair ladder until the boundary-layer / worst-cell gate passes."""
    mp = proto["mesh"]
    bl = {"n_layers": mp["prism_layers"], "first_aspect_ratio": mp["first_aspect_ratio"], "growth": mp["prism_growth"], "tet_volume_growth": mp["tet_volume_growth"]}
    ladder = [dict(x) for x in mp["repair_ladder"]]
    if start_skew:
        ladder = [x for x in ladder if x.get("improve_skew") and x["improve_skew"] <= start_skew] or [{"improve_skew": start_skew}]
    attempts = []
    for i, extra in enumerate(ladder):
        params = {**(base or {}), **bl, **extra}
        try:
            mesh_file = pipeline.stage_mesh(case_dir, work, surf, prof, mesh_params=params, reference_mesh=False)
            gs = json.loads((work / "mesh_gate.json").read_text())["new"]
            mesh_case = pipeline.stage_finalize(work, mesh_file, prof)
            gate = meshcheck.quality_gate(meshcheck.fluent_quality(_latest_log(work, "finalize").read_text(errors="replace")), gs, mp)
        except RuntimeError as exc:
            gate = {"ok": False, "failures": [f"{type(exc).__name__}: {exc}"], "warnings": []}
        attempts.append({"attempt": i, "params": params, **{k: gate.get(k) for k in ("ok", "failures", "warnings", "min_orthogonal_quality", "max_aspect_ratio")}})
        _dump(work / "mesh_attempts.json", attempts)
        if gate["ok"]:
            return mesh_case, {"attempts": attempts, "chosen": i, "gate": gate}
    raise Blocked("mesh", f"mesh gate failed after {len(ladder)} attempts: {attempts[-1]['failures']}", attempts)


def outlet_key_of_atlas(work: Path, openings: list[dict], keys: dict[int, str]) -> tuple[Path | None, dict]:
    """vessel_geom atlas of the deployment naming run and its outlet names -> UDF keys (nearest STL opening)."""
    nd = work / "naming_deploy"
    atlas, sa = nd / "centerline" / "atlas.npz", nd / "stage_a.json"
    if not atlas.exists() or not sa.exists():
        return None, {}
    a = json.loads(sa.read_text())
    out = {}
    for o in a["centerline"]["openings"]:
        if o.get("role") != "outlet":
            continue
        c = np.asarray(o["center_mm"])
        j = int(np.argmin([np.linalg.norm(np.asarray(x["centre_mm"]) - c) for x in openings]))
        out[o["outlet_name"]] = keys[int(openings[j]["loop"])]
    return atlas, out


def prepare_unit(unit: str, spec: dict, proto: dict, work_root: Path, library: Path, prof_dir: Path, cores: int, job_name: str,
                 fingerprint: str = "", exclude: str | None = "node05") -> dict:
    """``exclude``: nodes the managed fluent.slurm avoids (batch ``resources.exclude``; None / "" = every CPU node)."""
    s = resolve_spec(unit, spec, proto, library)
    work = (work_root / unit).resolve(); work.mkdir(parents=True, exist_ok=True); guard.assert_inside(work, work)
    case_dir = library / unit
    tmpl_dir = library / s["template"]
    tprof = recover.load_profile(s["template"], prof_dir, library)
    coh = s["cohort"]
    rep: dict = {"unit": unit, "spec": s, "protocol": proto["name"], "fingerprint": fingerprint,
                 "template_profile": {k: tprof[k] for k in ("case_dir", "udf", "family", "udf_thread_zones", "anatomy_zone", "anatomy_wall_zones")}}
    naming = surface_rep = split = None
    confirmed = False
    if s["mode"] == "own-mesh":
        own = recover.load_profile(unit, prof_dir, library)
        own["udf_thread_zones"] = recover.own_key_zones(own, s["own_outlet_keys"])
        mesh_case, rep["own_mesh"] = recover.stage_own_mesh(case_dir, work, own, tprof)
        openings, keys = [], {}
    elif s["mode"] == "stl":
        stl = Path(s["stl_path"])
        rep["stl_sha256"] = sha256(stl)
        if s.get("repair_max_move_mm"):
            surface.REPAIR_MAX_MOVE_MM = float(s["repair_max_move_mm"])
        _, openings = recover.stl_openings(stl)
        if len(openings) != proto["openings"]["count"]:
            raise Blocked("geometry", f"{stl.name}: {len(openings)} openings (protocol expects {proto['openings']['count']})")
        keys, naming, confirmed = name_openings(unit, s, stl, work, openings, prof_dir, library)
        _dump(work / "naming.json", naming)
        try:
            naming_figure(stl, naming["openings"], keys, naming, work / "naming.png", unit)
        except Exception as exc:          # the figure is a convenience; never fail the build on it
            rep["naming_figure_error"] = f"{type(exc).__name__}: {exc}"
        nm_gate = preflight.naming_item(proto, naming, confirmed)
        if nm_gate["level"] == "fail":
            raise Blocked("naming", "opening naming needs confirmation (see naming.png / naming.json)", nm_gate)
        tzone = {"inlet": next(o["name"] for o in tprof["openings"] if o["bc_type"] == "velocity-inlet"), **tprof["udf_thread_zones"]}
        names = {i: (tzone[k], "velocity-inlet" if k == "inlet" else "pressure-outlet") for i, k in keys.items()}
        prof = rebuild.lost_case_profile(case_dir, tprof, names, Path(tprof["udf"]), stl=stl)
        prof["naming"] = naming
        inlet_len = float(s.get("inlet_length_mm") or proto["extension"]["inlet_length_mm"][coh])
        area = {o["name"]: o["area_mm2"] for o in prof["openings"]}
        for e in prof["extensions"]:       # protocol extension rule: inlet length by cohort, outlets L/D x equivalent diameter
            e["length_mm"] = inlet_len if e["opening"] == tzone["inlet"] else round(float(proto["extension"]["outlet_L_over_D"] * 2 * np.sqrt(area[e["opening"]] / np.pi)), 3)
        if s["extension_direction"] != "normal":
            dirs = _directions(stl, names, s["extension_direction"], proto, s.get("repair_max_move_mm"))
            prof["extension_directions"] = dirs
            for e in prof["extensions"]:
                if dirs.get(e["opening"], {}).get("direction") is not None:
                    e["direction"] = dirs[e["opening"]]["direction"]
        _dump(work / "reference_profile.json", prof)
        pipeline.stage_surface(case_dir, work, prof, stl_path=stl)
        surface_rep = json.loads((work / "surface_report.json").read_text())
        zs = surface_rep["boundary_mesh"]["zones"]
        surf = {"zones": [z["name"] for z in zs], "walls": [z["name"] for z in zs if z["type"] == "wall"]}
        params = None
        if tprof["family"] == "poly" and s.get("mesh_params"):          # sizes given (e.g. a library regression): no calibration
            params = {k: float(v) for k, v in s["mesh_params"].items()}
        elif tprof["family"] == "poly":
            if (work / "calibration.json").exists():
                c = json.loads((work / "calibration.json").read_text())["chosen"]
                params = {k: c[k] for k in ("min_size", "max_size", "curvature_angle")}
            else:
                ref = s["density_ref"]
                if not ref:
                    raise ValueError(f"{unit}: poly family needs density_ref units")
                target = float(np.median([dual_density(u, prof_dir, library) for u in ref]))
                base = s.get("calib_base") or {"min_size": tprof["poly_mesh_estimate"]["min_size_mm"] * 1e-3, "max_size": tprof["poly_mesh_estimate"]["max_size_mm"] * 1e-3}
                params = recover.calibrate(work, surf, prof, target, base)
        fill = s.get("mesh_volume_fill") or proto["mesh"].get("volume_fill", {}).get(tprof["family"], "tet")
        if tprof["family"] == "poly":
            params = {**(params or {}), "volume_fill": fill}
        mesh_case, rep["mesh"] = mesh_with_repair(case_dir, work, surf, prof, params, proto, start_skew=s.get("mesh_improve_skew"))
    else:
        raise ValueError(s["mode"])
    bc = proto["bc"]
    area_src, a1 = bc.get("area_source", {}).get(coh, "cut"), float(bc["A1_kg_s"][coh])
    text, prot = recover.protocol_udf_text(tprof, mesh_case, unit, cohort_name=coh, area_source=area_src, a1_kg_s=a1)
    _dump(work / "protocol_bc.json", prot)
    if s["mode"] == "stl":
        atlas, okey = outlet_key_of_atlas(work, openings, keys)
        if atlas is not None and len(okey) == len(proto["openings"]["outlet_keys"]):
            zone_key = {z: k for k, z in tprof["udf_thread_zones"].items()}
            ext_len = {zone_key[e["opening"]]: e["length_mm"] for e in prof["extensions"] if e["opening"] in zone_key}
            try:
                rk = proto["risk"]
                split = preflight.anatomy_split(atlas, okey, prot["rcr"], prot["cut_area_mm2"], ext_len, proto["openings"]["side_pairs"], rk["path_resistance_ratio_warn"],
                                                rk.get("split_dev_slope", 9.415), rk.get("split_dev_warn", 0.015))
            except (KeyError, ValueError) as exc:
                rep["anatomy_split_error"] = f"{type(exc).__name__}: {exc}"
    pf = preflight.check(proto, prot, naming=naming, surface_report=surface_rep, split=split, naming_confirmed=confirmed)
    pf["anatomy_split"] = split
    preflight.dump(work / "preflight.json", pf)
    rep["preflight"] = {"fail": pf["fail"], "warn": pf["warn"]}
    if not pf["ok"]:
        _dump(work / "prepare_report.json", rep)
        raise Blocked("protocol", f"set-up checks failed: {pf['fail']}", pf)
    final = pipeline.stage_setup(tmpl_dir, work, tprof, mesh_case, s["case_name"], udf_text=text, udf_zones=tprof["udf_thread_zones"])
    rep["final_udf_check"] = recover.check_final_udf(work, tprof, final, unit, cohort_name=coh, area_source=area_src, a1_kg_s=a1)
    if not rep["final_udf_check"]["ok"]:
        _dump(work / "prepare_report.json", rep)
        raise RuntimeError(f"final UDF check failed: {rep['final_udf_check']}")
    smoke_log = pipeline.stage_smoke(work, final, quality=True)
    q = meshcheck.quality_gate(meshcheck.fluent_quality(smoke_log.read_text(errors="replace")), None, proto["mesh"])
    rep["final_quality"] = q
    if not q["ok"] and s["mode"] == "stl":
        raise Blocked("mesh", f"final case quality: {q['failures']}", q)
    rp = settings_diff.sections(settings_diff.case_text(final))["rp"]
    sch = schedule.make(proto["schedules"]["ladder"][0], proto)
    rep["run_files"] = schedule.write_run_files(work, final, sch, cores, job_name, proto["solver"]["fluent_module"],
                                                make_dirs=tuple(sorted({"export", *schedule.autosave_dirs(rp.get("autosave/filename", ""))})),
                                                exclude=exclude or None)
    rep["case"] = str(final)
    _dump(work / "prepare_report.json", rep)
    return rep


def rewrite_schedule(work: Path, proto: dict, name: str, cores: int, job_name: str, exclude: str | None = "node05") -> dict:
    """Same prepared case, another schedule of the divergence ladder (new 2.jou / fluent.slurm / kept steps)."""
    fr = json.loads((work / schedule.FRAMES_FILE).read_text())
    case = Path(fr["case"])
    rp = settings_diff.sections(settings_diff.case_text(case))["rp"]
    return schedule.write_run_files(work, case, schedule.make(name, proto), cores, job_name, proto["solver"]["fluent_module"],
                                    make_dirs=tuple(sorted({"export", *schedule.autosave_dirs(rp.get("autosave/filename", ""))})),
                                    exclude=exclude or None)

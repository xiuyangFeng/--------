"""End-to-end volume acceptance on recover8: canonical STL -> deploy stage A (vessel_geom centreline + automatic naming)
-> stage B with the old (PF6_VF6_peak_3seed_20260920, v5.0) and new (PF6_VF6_v52d_3seed_20261003) volume releases on the
same stage-A output -> score against the v5.2d CFD volume labels at the nearest CFD point (pressure: wall nodes + anatomy
cells, relative to the volume mean; velocity: anatomy cells, world axes). Both releases see identical geometry and query
points, so old vs new is a paired comparison; the nearest-point pairing is not the CFD-cell protocol of training."""
import json, shutil, sys, time, traceback
from pathlib import Path
import numpy as np
from scipy.spatial import cKDTree
ROOT = Path("/public/newhome/cy/Digital_twin/GNN"); sys.path.insert(0, str(ROOT))
from wss_deploy import pipeline as P, centerline as CL
from wss_deploy.registry import ReleaseRegistry
from training_wss_min.tools.deployment_stl_simulation import find_stl
from training_wss_min.tools import report_wss_v52c_retrain as R
NEW_ROOT, OUT = Path(sys.argv[1]), Path(sys.argv[2])
OLD_ROOT = ROOT / "outputs/wss_deploy_release"
VIEW = ROOT / "data_wss_v5/views_v5_2d_20261001/wss_min_view_v1"; VOL = ROOT / "data_wss_v5/views_v5_2d_20261001/wss_min_volview_v1"
split = json.load(open(VIEW / "split_v52p4_full265_train265_test8.json"))
RELEASES = [("PF6_VF6_peak_3seed_20260920", OLD_ROOT), ("PF6_VF6_v52d_3seed_20261003", NEW_ROOT)]
rels = {rid: ReleaseRegistry(root, device="cuda").load(rid) for rid, root in RELEASES}
peak = ReleaseRegistry(OLD_ROOT, device="cuda").load("X5Dcap_asym2_v52d_3seed_20261002")   # only for the naming gate
rows = []; t0 = time.time()
for uid in split["test_cases"]:
    tag = uid.replace("/", "__"); row = {"unit": uid}
    try:
        stl = find_stl(uid); row["stl"] = str(stl)
        base = OUT / "jobs" / tag / "_stage_a"
        if base.exists(): shutil.rmtree(base)
        ta = time.perf_counter(); a = P.stage_a(stl, base); row["stage_a_s"] = round(time.perf_counter() - ta, 1)
        if a.get("stage") != "A":
            raise RuntimeError("input check failed: " + "; ".join(a["input_check"].get("errors", []) + a["input_check"].get("flags", [])))
        gate = CL.evaluate_confidence_gate(a["proposal"], orientation_source=a["input_check"].get("orientation_source", "unknown_stl"), release=peak)
        row["gate_passed"] = bool(gate.get("passed"))
        mapping = a["proposal"]["mapping"]
        with np.load(VIEW / uid / "bundle.npz", allow_pickle=True) as b:
            rot = np.asarray(b["transform_rotation"], float); wall_xyz = np.asarray(b["wall_coords_raw"], float)
        with np.load(VOL / uid / "volume.npz", allow_pickle=True) as v:
            cell_xyz = np.asarray(v["vol_coords_raw"], float); cell_p = np.asarray(v["vol_pressure_rel_peak"], float)
            cell_u = np.asarray(v["vol_velocity_aligned_peak"], float) @ rot      # aligned = world @ rot.T
            wall_p = np.asarray(v["wall_pressure_rel_peak"], float)
        p_tree = cKDTree(np.r_[wall_xyz, cell_xyz]); p_truth = np.r_[wall_p, cell_p]; u_tree = cKDTree(cell_xyz)
        for rid, _ in RELEASES:
            d = OUT / "jobs" / tag / rid
            if d.exists(): shutil.rmtree(d)
            shutil.copytree(base, d)
            sa = json.loads((d / "stage_a.json").read_text()); sa.setdefault("input_check", {})["clean_stl"] = str(d / "input_clean_mm.stl")
            (d / "stage_a.json").write_text(json.dumps(sa, ensure_ascii=False, indent=1))
            tb = time.perf_counter(); P.stage_b(d, mapping, rels[rid], confirmed=True, case_id=tag); sb = time.perf_counter() - tb
            z = np.load(d / "field.npz")
            pts, ipts = z["pts"].astype(float), z["internal_pts"].astype(float)
            dp, ip = p_tree.query(pts); du, iu = u_tree.query(ipts)
            pred_u = z["velocity_m_s"].astype(float); true_u = cell_u[iu]
            row[rid] = {"stage_b_s": round(sb, 1), "n_points": int(len(pts)), "n_interior": int(len(ipts)),
                        "nn_median_mm_pressure": round(float(np.median(dp)), 3), "nn_median_mm_velocity": round(float(np.median(du)), 3),
                        "_p": (p_truth[ip], z["pressure_pa"].astype(float)),
                        "_s": (np.linalg.norm(true_u, axis=1), np.linalg.norm(pred_u, axis=1)),
                        "vector_rmse_m_s": float(np.sqrt(np.mean(np.sum((true_u - pred_u) ** 2, axis=1))))}
            for k, name in (("_p", "pressure"), ("_s", "speed")):
                y, p = row[rid][k]; row[rid][name] = {"r2": R.r2(y, p), "mae": float(np.mean(np.abs(y - p)))}
    except Exception as exc:
        row["error"] = f"{type(exc).__name__}: {exc}"; row["tb"] = traceback.format_exc()[-1500:]
    rows.append(row)
    print(json.dumps({k: (v if not isinstance(v, dict) else {f: (round(v[f]['r2'], 4) if isinstance(v.get(f), dict) else v.get(f))
                      for f in ('pressure', 'speed', 'vector_rmse_m_s', 'stage_b_s', 'nn_median_mm_velocity') if f in v})
                      for k, v in row.items() if k in ('unit', 'error', 'gate_passed', 'stage_a_s') or k in dict(RELEASES)}, ensure_ascii=False), flush=True)
summary = {}
for rid, _ in RELEASES:
    for k, name in (("_p", "pressure"), ("_s", "speed")):
        units = {r["unit"]: r[rid][k] for r in rows if rid in r}
        if units:
            per = [R.r2(y, p) for y, p in units.values()]
            summary.setdefault(rid, {})[name] = {"n": len(units), "r2cb": R.r2cb(units), "r2_casemedian": float(np.median(per)),
                                                "r2_casemin": float(np.min(per))}
    better = {name: sum(1 for r in rows if all(x in r for x, _ in RELEASES) and r[RELEASES[1][0]][name]["r2"] > r[RELEASES[0][0]][name]["r2"])
              for name in ("pressure", "speed")}
for r in rows:
    for rid, _ in RELEASES:
        if rid in r:
            r[rid].pop("_p", None); r[rid].pop("_s", None)
out = {"rows": rows, "summary": summary, "new_better_units": better, "elapsed_s": round(time.time() - t0, 1)}
(OUT / "e2e_recover8_volume.json").write_text(json.dumps(out, ensure_ascii=False, indent=1, default=str))
print(json.dumps({"summary": summary, "new_better_units": better, "elapsed_s": out["elapsed_s"]}, ensure_ascii=False, indent=1))

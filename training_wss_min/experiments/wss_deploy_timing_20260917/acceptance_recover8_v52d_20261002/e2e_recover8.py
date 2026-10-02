"""End-to-end acceptance on recover8: canonical STL -> deploy stage A (vessel_geom centreline + automatic naming)
-> stage B with four releases (old X5D_v51 / new X5Dcap_asym2 peak; old M1 / new M1cap three-head) -> score against
the v5.2d CFD labels at the nearest CFD wall node (peak WSS; TAWSS / OSI for the three-head releases)."""
import json, shutil, sys, time, traceback
from pathlib import Path
import numpy as np
from scipy.spatial import cKDTree
ROOT = Path("/public/newhome/cy/Digital_twin/GNN"); sys.path.insert(0, str(ROOT))
from wss_deploy import pipeline as P, centerline as CL, naming_calibration as NC
from wss_deploy.registry import ReleaseRegistry
from training_wss_min.tools.deployment_stl_simulation import find_stl
from training_wss_min.tools import report_wss_v52c_retrain as R
NEW_ROOT, OUT = Path(sys.argv[1]), Path(sys.argv[2])
OLD_ROOT = ROOT / "outputs/wss_deploy_release"
VIEW = ROOT / "data_wss_v5/views_v5_2d_20261001/wss_min_view_v1"; CYCLE = ROOT / "data_wss_v5/views_v5_2d_20261001/wss_min_cycle_v1"
split = json.load(open(VIEW / "split_v52p4_full265_train265_test8.json"))
RELEASES = [("X5D_v51_5seed_20260916", OLD_ROOT), ("X5Dcap_asym2_v52d_3seed_20261002", NEW_ROOT),
            ("M1_3head_3seed_20260922", OLD_ROOT), ("M1cap_v52d_3seed_20261002", NEW_ROOT)]
regs = {OLD_ROOT: ReleaseRegistry(OLD_ROOT, device="cuda"), NEW_ROOT: ReleaseRegistry(NEW_ROOT, device="cuda")}
rels = {rid: regs[root].load(rid) for rid, root in RELEASES}
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
        gate = CL.evaluate_confidence_gate(a["proposal"], orientation_source=a["input_check"].get("orientation_source", "unknown_stl"),
                                           release=rels["X5Dcap_asym2_v52d_3seed_20261002"])
        row["gate_passed"] = bool(gate.get("passed")); row["gate_reasons"] = gate.get("reasons")
        NC.compare(row, a, uid)
        mapping = a["proposal"]["mapping"]
        b = np.load(VIEW / uid / "bundle.npz", allow_pickle=True); steps = b["steps"].tolist()
        cfd_xyz = np.asarray(b["wall_coords_raw"], float); cfd_wss = b["wall_wss"][steps.index(int(b["peak_step"]))].astype(float)
        with np.load(CYCLE / uid / "cycle.npz") as z:
            assert np.array_equal(z["wall_node_id_cas"], b["wall_node_id_cas"])
            cyc = {"wall_tawss": z["wall_tawss"].astype(float), "wall_osi": z["wall_osi"].astype(float)}
        tawss_key, osi_key = "wall_tawss", "wall_osi"
        tree = cKDTree(cfd_xyz)
        for rid, _ in RELEASES:
            d = OUT / "jobs" / tag / rid
            if d.exists(): shutil.rmtree(d)
            shutil.copytree(base, d)
            sa = json.loads((d / "stage_a.json").read_text()); sa.setdefault("input_check", {})["clean_stl"] = str(d / "input_clean_mm.stl")
            (d / "stage_a.json").write_text(json.dumps(sa, ensure_ascii=False, indent=1))
            tb = time.perf_counter(); P.stage_b(d, mapping, rels[rid], confirmed=True, case_id=tag); sb = time.perf_counter() - tb
            z = np.load(d / "field.npz"); pts = z["pts"].astype(float)
            shift = np.zeros(3)
            dist, idx = tree.query(pts)
            if np.median(dist) > 2.0 and row.get("align_shift_mm"):
                raise RuntimeError(f"prediction points do not overlay the CFD wall (median NN {np.median(dist):.1f} mm)")
            res = {"stage_b_s": round(sb, 1), "nn_median_mm": round(float(np.median(dist)), 3), "n_points": int(len(pts))}
            fields = {"wss": (z["wss_pa"], cfd_wss)}
            if "tawss_pa" in z.files:
                fields["tawss"] = (z["tawss_pa"], cyc[tawss_key]); fields["osi"] = (z["osi"], cyc[osi_key])
            for name, (pred, truth) in fields.items():
                y = truth[idx]; p = np.asarray(pred, float)
                res[name] = {"r2": R.r2(y, p), "mae": float(np.mean(np.abs(y - p))), "y": None}
                res[name]["_pairs"] = (y, p)
            row[rid] = res
    except Exception as exc:
        row["error"] = f"{type(exc).__name__}: {exc}"; row["tb"] = traceback.format_exc()[-1500:]
    rows.append(row)
    print(json.dumps({k: (v if not isinstance(v, dict) else {f: (round(v[f]['r2'], 4) if isinstance(v.get(f), dict) else v.get(f)) for f in ('wss', 'tawss', 'osi', 'stage_b_s', 'nn_median_mm') if f in v})
                      for k, v in row.items() if k in ('unit', 'error', 'joint_correct', 'gate_passed', 'stage_a_s') or k in dict(RELEASES)}, ensure_ascii=False), flush=True)
# pooled unit-balanced R2 per release / field
summary = {}
for rid, _ in RELEASES:
    for name in ("wss", "tawss", "osi"):
        units = {r["unit"]: r[rid][name]["_pairs"] for r in rows if rid in r and name in r[rid]}
        if units:
            per = [R.r2(y, p) for y, p in units.values()]
            summary.setdefault(rid, {})[name] = {"n": len(units), "pa_r2cb": R.r2cb(units), "r2_casemean": float(np.mean(per)),
                                                "r2_casemedian": float(np.median(per)), "r2_casemin": float(np.min(per))}
for r in rows:
    for rid, _ in RELEASES:
        for name in ("wss", "tawss", "osi"):
            if rid in r and name in r[rid]:
                r[rid][name].pop("_pairs", None); r[rid][name].pop("y", None)
out = {"rows": rows, "summary": summary, "elapsed_s": round(time.time() - t0, 1),
       "naming": {"joint_correct": sum(bool(r.get("joint_correct")) for r in rows), "n": len(rows),
                  "gate_passed": sum(bool(r.get("gate_passed")) for r in rows)}}
(OUT / "e2e_recover8.json").write_text(json.dumps(out, ensure_ascii=False, indent=1, default=str))
print(json.dumps({"summary": summary, "naming": out["naming"], "elapsed_s": out["elapsed_s"]}, ensure_ascii=False, indent=1))

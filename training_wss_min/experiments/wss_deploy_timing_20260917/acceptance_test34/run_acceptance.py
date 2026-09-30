"""P1 acceptance: run the deployment pipeline (vessel_geom centreline + auto outlet naming + 5-seed ensemble) on all test34 STLs,
score against nearest CFD wall node (peak frame) and check the automatic names against the CFD-derived atlas names."""
import json, sys, time, traceback
from pathlib import Path
import numpy as np, h5py
from scipy.spatial import cKDTree
from wss_deploy.pipeline import run_all
from wss_deploy.infer import Release
from training_wss_min.tools.deployment_stl_simulation import find_stl
ROOT = Path("/public/newhome/cy/Digital_twin/GNN"); OUT = Path(__file__).parent; JOBS = OUT / "jobs"
view = ROOT / "data_wss_v5/views_v5_1/wss_min_view_v1"; split = json.load(open(view / "split_V5_train136_test34.json"))
rel = Release(); rows = []; t0 = time.time()
for uid in split["test_cases"]:
    stl = find_stl(uid); tag = uid.replace("/", "__"); row = {"unit_id": uid, "stl": str(stl)}
    try:
        meta = run_all(stl, JOBS / tag, rel, case_id=tag)
        z = np.load(JOBS / tag / "field.npz"); pts = z["pts"]; w = z["wss_pa"].astype(float)
        b = np.load(view / uid / "bundle.npz", allow_pickle=True); steps = b["steps"].tolist(); si = steps.index(int(b["peak_step"])); cfd_xyz = b["wall_coords_raw"]; cfd_w = b["wall_wss"][si].astype(float)
        d, i = cKDTree(cfd_xyz).query(pts); y = cfd_w[i]
        row.update(r2=float(1 - np.sum((y - w) ** 2) / np.sum((y - y.mean()) ** 2)), mae=float(np.mean(np.abs(y - w))), nn_mm=float(np.median(d)), n=int(len(pts)),
                   true_p99=float(np.quantile(y, .99)), pred_p99=float(np.quantile(w, .99)), true_max=float(y.max()), pred_max=float(w.max()), timing=meta["timing_s"])
        # naming check
        with h5py.File(ROOT / "data_wss_v5/anatomy_pointcloud_v5_1_20260916/cases" / tag / "case.h5", "r") as h:
            g = h["geometry"]; tab = np.asarray(g["atlas_table"][()]); cols = json.loads(g.attrs["atlas_columns"]); segs = json.loads(g.attrs["atlas_segments"])
        col = lambda c: tab[:, cols.index(c)]; seg = col("segment_id").astype(int); idx = col("sample_index").astype(int); xyz = np.stack([col("x_mm"), col("y_mm"), col("z_mm")], 1)
        truth = {}
        for s in segs:
            if s.get("ends_at_leaf"):
                r = np.flatnonzero(seg == int(s["segment_id"])); r = r[np.argsort(idx[r])]; truth[s["outlet_name"]] = xyz[r[-1]]
        ok = 0; det = []
        for e in meta["endpoints"]:
            if e["kind"] == "outlet":
                c = np.array(e["center_mm"]); best = min(truth, key=lambda k: np.linalg.norm(truth[k] - c)); ok += int(best == e["name"]); det.append((e["name"], best))
        row.update(names_ok=ok, names=det, flags=meta.get("flags", []))
    except Exception as e:
        row.update(error=f"{type(e).__name__}: {e}", tb=traceback.format_exc()[-1500:])
    rows.append(row); print(json.dumps({k: v for k, v in row.items() if k in ("unit_id", "r2", "names_ok", "error", "pred_p99", "true_p99")}, ensure_ascii=False), flush=True)
    json.dump({"rows": rows, "elapsed_s": time.time() - t0}, open(OUT / "acceptance.json", "w"), indent=1, ensure_ascii=False)
okr = [r for r in rows if "r2" in r]
print("cases ok", len(okr), "/", len(rows), "mean per-case R2", np.mean([r["r2"] for r in okr]).round(4), "names correct", sum(r["names_ok"] for r in okr), "/", 4 * len(okr))

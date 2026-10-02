"""Model-level equivalence for the v5.2d volume release: deploy Release.predict on the training-side recover8 cases
(v5.2d volume root, wall rows then interior rows) vs the three-seed mean of the saved full265 eval predictions.
frame_rotation = identity, so velocity stays in the atlas frame in which training saved it."""
import json, sys, time
from pathlib import Path
import numpy as np
ROOT = Path("/public/newhome/cy/Digital_twin/GNN"); sys.path.insert(0, str(ROOT))
from training_wss_min import config as C, dataset as D
from wss_deploy.registry import ReleaseRegistry
REL_ROOT, OUT = Path(sys.argv[1]), Path(sys.argv[2])
RID = "PF6_VF6_v52d_3seed_20261003"
RUNS = ROOT / "training_wss_min/runs/pf6vf6_v52d_retrain_20261002"
rel = ReleaseRegistry(REL_ROOT, device="cuda").load(RID)
cfg = next(m["cfg"] for m in rel.models if m["field"] == "pressure")
stats = D.load_wss_stats(cfg.data.wss_stats_path)
cases = D.load_partition(cfg.data.split_path, "test", stats, target=cfg.data.target, target_normalization=cfg.data.target_normalization,
                         data_root=cfg.data.data_root, required_frame_version=cfg.data.required_frame_version,
                         extra_point_features=C.v6_point_features(cfg), point_features_root=cfg.data.point_features_root)
def saved(prefix, uid):
    out = []
    for s in (1234, 7, 2025):
        base = RUNS / f"{prefix}_full265_s{s}/eval/ckpt_best/predictions/test"
        f = next(c["file"] for c in json.loads((base / "manifest.json").read_text())["cases"] if c["unit_id"] == uid)
        with np.load(base / f) as z:
            out.append((z["pred_raw"].astype(np.float64), z["query_idx"].copy()))
    assert all(np.array_equal(out[0][1], q) for _, q in out)
    return np.mean([p for p, _ in out], axis=0), out[0][1]
rows = []
for case in cases:
    uid = case["unit_id"]; n_wall = int(case["n_wall"]); n = len(case["pos"])
    view = dict(case, support_pool=np.arange(n_wall), frame_rotation=np.eye(3))
    t = time.perf_counter(); out = rel.predict(view); dt = time.perf_counter() - t
    row = {"unit": uid, "n_wall": n_wall, "n_interior": n - n_wall, "seconds": round(dt, 2)}
    for prefix, key in (("PF6", "pressure_pa"), ("VF6", "velocity_m_s")):
        ref, q = saved(prefix, uid)
        mine = np.asarray(out[key], np.float64)
        mine = mine[q] if prefix == "PF6" else mine[q - n_wall]
        d = np.abs(mine - ref)
        row[prefix] = {"n_rows": int(len(q)), "max_abs": float(d.max()),
                       "max_rel": float((d / np.maximum(np.abs(ref), 1e-3)).max()),
                       "r2_vs_saved": float(1 - np.sum((mine - ref) ** 2) / np.sum((ref - ref.mean(axis=0)) ** 2))}
    rows.append(row); print(json.dumps(row), flush=True)
OUT.write_text(json.dumps({"release": RID, "rows": rows}, indent=1))

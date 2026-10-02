"""Model-level equivalence: deploy Release.predict on the training-side recover8 cases vs the saved eval predictions."""
import json, sys, time
from pathlib import Path
import numpy as np
ROOT = Path("/public/newhome/cy/Digital_twin/GNN"); sys.path.insert(0, str(ROOT))
from training_wss_min import config as C, dataset as D
from wss_deploy.registry import ReleaseRegistry
REL_ROOT = Path(sys.argv[1]); OUT = Path(sys.argv[2])
RUNS = ROOT / "training_wss_min/runs/wss_v52d_retrain_20261001"
reg = ReleaseRegistry(REL_ROOT, device="cuda")
report = {}
for rid, prefix, channels in (("X5Dcap_asym2_v52d_3seed_20261002", "X5Dcap_asym2_full265", (None,)),
                              ("M1cap_v52d_3seed_20261002", "M1cap_full265", ("wss", "tawss", "osi"))):
    rel = reg.load(rid)
    cfg = rel.models[0]["cfg"]
    stats = D.load_wss_stats(cfg.data.wss_stats_path)
    cases = D.load_partition(cfg.data.split_path, "test", stats, target=cfg.data.target,
                             target_normalization=cfg.data.target_normalization, data_root=cfg.data.data_root,
                             required_frame_version=cfg.data.required_frame_version, case_features_path=cfg.data.case_features_path,
                             timesteps=getattr(cfg.data, "timesteps", "peak"), extra_point_features=C.v6_point_features(cfg),
                             point_features_root=getattr(cfg.data, "point_features_root", None),
                             cycle_view_root=getattr(cfg.data, "cycle_view_root", None),
                             multi_aux_channels=tuple(getattr(cfg.data, "multi_aux_channels", ()) or ()))
    rows = []
    for case in cases:
        uid = case["unit_id"]
        t = time.perf_counter(); out = rel.predict(case); dt = time.perf_counter() - t
        row = {"unit": uid, "n": int(len(case["pos"])), "seconds": round(dt, 2)}
        for ch in channels:
            saved = []
            for s in (1234, 7, 2025):
                base = RUNS / f"{prefix}_s{s}/eval/ckpt_best/predictions/test" / (ch or "")
                man = json.loads((base / "manifest.json").read_text())
                f = next(c["file"] for c in man["cases"] if c["unit_id"] == uid)
                with np.load(base / f) as z:
                    saved.append((z["pred_pa"].astype(np.float64), z["row_index"].copy()))
            ri = saved[0][1]
            assert all(np.array_equal(ri, r) for _, r in saved)
            ref = np.mean([p for p, _ in saved], axis=0)
            key = {"wss": "wss_pa", "tawss": "tawss_pa", "osi": "osi", None: "wss_pa"}[ch]
            mine = np.asarray(out[key], dtype=np.float64)[ri]
            d = np.abs(mine - ref)
            row[ch or "wss"] = {"max_abs": float(d.max()), "max_rel": float((d / np.maximum(np.abs(ref), 1e-3)).max()),
                                "r2_vs_saved": float(1 - np.sum((mine - ref) ** 2) / np.sum((ref - ref.mean()) ** 2)),
                                "n_rows": int(len(ri))}
        rows.append(row); print(rid, json.dumps(row), flush=True)
    report[rid] = rows
OUT.write_text(json.dumps(report, indent=1))

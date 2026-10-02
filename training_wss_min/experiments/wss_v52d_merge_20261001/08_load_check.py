"""Step 8 of the 2026-10-01 merge: strict loading of the v5.2d mainline configs (train + test), every density level of the
units changed by the merge, and the feature z-scores the configs point at re-derived from the loaded training partition."""
import json, sys
from pathlib import Path
G = Path("/public/newhome/cy/Digital_twin/GNN"); sys.path.insert(0, str(G))
from training_wss_min import config as C
from training_wss_min import dataset as D
HERE = Path(__file__).resolve().parent
CD = G / "training_wss_min/configs/wss_v52d_retrain_20261001"
U = json.loads((HERE / "units.json").read_text()); CH = set(U["recomputed_same_mesh"]) | set(U["rebuilt_from_stl"]) | set(U["renamed_outlets"])
out = {}
for name in ["X5Dcap_asym2_full265_s1234"] + [f"X5Dcap_asym2_v52cv_f{k}_s1234" for k in range(5)]:
    cfg = json.loads((CD / f"{name}.json").read_text()); d = cfg["data"]
    feats = tuple(d["input_features"]); extra = tuple(f for f in feats if f in C.SIDECAR_FEATURE_KEYS)
    st = D.load_wss_stats(d["wss_stats_path"]); res = {}
    for part in ("train", "test"):
        cases = D.load_partition(d["split_path"], part, st, strict=True, target="wss", data_root=d["data_root"], required_frame_version=d["required_frame_version"],
                                 extra_point_features=extra, point_features_root=d["point_features_root"])
        ch = [c for c in cases if c["unit_id"] in CH]; nd = 0
        if part == "train" and d.get("density_aug_root"):
            for c in ch:
                for lv in d.get("density_aug_levels", [70, 50, 35, 25]):
                    D.density_view(c, D.load_density_level(c, d["density_aug_root"], lv), lv); nd += 1
        res[part] = {"n": len(cases), "changed_units": len(ch), "density_views_checked": nd}
        if part == "train":
            fs = D.compute_feature_stats(cases, feats, d["curvature_transform"])
            ref = {k: v for k, v in json.loads(Path(d["feature_stats_path"]).read_text()).items() if not k.startswith("_")}
            res["feature_stats_max_abs_diff_vs_file"] = max(abs(fs[k][m] - ref[k][m]) for k in ref for m in ("mean", "std"))
    out[name] = res; print(name, res, flush=True)
out["passed"] = all(v["feature_stats_max_abs_diff_vs_file"] <= 1e-9 for v in out.values() if isinstance(v, dict))
(HERE / "load_check.json").write_text(json.dumps(out, indent=1)); print("passed", out["passed"])

import json, sys
from pathlib import Path
import numpy as np
from training_wss_min import config as C
from training_wss_min import dataset as D
G = Path("/public/newhome/cy/Digital_twin/GNN")
CD = G / "training_wss_min/configs/wss_v52c_labelfix_20260930"
UNITS9 = list(json.loads((G / "outputs/cfd_auto_trial_20260927/_label_fix/batch.json").read_text())["units"])
out = {}
for name in ["X5Dcap_asym2_s1234", "X5Dcap_asym2_v52cv_f2_s1234", "X5Dcap_asym2_full265_s1234"]:
    cfg = json.loads((CD / f"{name}.json").read_text())
    d = cfg["data"]
    feats = tuple(d["input_features"]); extra = tuple(f for f in feats if f in C.SIDECAR_FEATURE_KEYS)
    st = D.load_wss_stats(d["wss_stats_path"])
    res = {}
    for part in ("train", "test"):
        cases = D.load_partition(d["split_path"], part, st, strict=True, target="wss", data_root=d["data_root"], required_frame_version=d["required_frame_version"],
                                 extra_point_features=extra, point_features_root=d["point_features_root"])
        fixed = [c for c in cases if c["unit_id"] in UNITS9]
        for c in fixed:
            for lv in cfg["data"].get("density_aug_levels", [70, 50, 35, 25]):
                D.density_view(c, D.load_density_level(c, d["density_aug_root"], lv), lv)
        res[part] = {"n": len(cases), "label_fix_units": [c["unit_id"] for c in fixed]}
    out[name] = res
    print(name, res, flush=True)
print("ok")

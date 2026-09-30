"""Unified v5.2c root: strict loading of the mainline configs (train + test, all density levels of every unit that has them)
and X5Dcap feature z-scores recomputed on the unified root vs the files the configs use (must be identical)."""
import json
import sys
from pathlib import Path

G = Path("/public/newhome/cy/Digital_twin/GNN")
sys.path.insert(0, str(G))
from training_wss_min import config as C  # noqa: E402
from training_wss_min import dataset as D  # noqa: E402

HERE = Path(__file__).resolve().parent
CD = G / "training_wss_min/configs/wss_v52c_labelfix_20260930"


def cmp(a, b, path=""):
    worst = [0.0, ""]
    def walk(x, y, p):
        if isinstance(x, dict):
            for k in x:
                if k.startswith("_") or k not in y:
                    continue
                walk(x[k], y[k], f"{p}.{k}")
        elif isinstance(x, list) and isinstance(y, list) and len(x) == len(y):
            for i, (u, v) in enumerate(zip(x, y)):
                walk(u, v, f"{p}[{i}]")
        elif isinstance(x, (int, float)) and isinstance(y, (int, float)) and not isinstance(x, bool):
            d = abs(float(x) - float(y))
            if d > worst[0]:
                worst[:] = [d, p]
    walk(a, b, path)
    return {"max_abs_diff": worst[0], "field": worst[1]}


out = {}
for name in ["X5Dcap_asym2_s1234"] + [f"X5Dcap_asym2_v52cv_f{k}_s1234" for k in range(5)] + ["X5Dcap_asym2_full265_s1234"]:
    cfg = json.loads((CD / f"{name}.json").read_text()); d = cfg["data"]
    feats = tuple(d["input_features"]); extra = tuple(f for f in feats if f in C.SIDECAR_FEATURE_KEYS)
    st = D.load_wss_stats(d["wss_stats_path"])
    kw = dict(strict=True, target="wss", data_root=d["data_root"], required_frame_version=d["required_frame_version"],
              extra_point_features=extra, point_features_root=d["point_features_root"])
    train = D.load_partition(d["split_path"], "train", st, **kw)
    test = D.load_partition(d["split_path"], "test", st, **kw)
    ndens = 0
    for c in train:
        for lv in d.get("density_aug_levels", [70, 50, 35, 25]):
            D.density_view(c, D.load_density_level(c, d["density_aug_root"], lv), lv); ndens += 1
    fs = D.compute_feature_stats(train, feats, d["curvature_transform"])
    ref = {k: v for k, v in json.loads(Path(d["feature_stats_path"]).read_text()).items() if not k.startswith("_")}
    out[name] = {"train": len(train), "test": len(test), "density_views": ndens, "feature_stats_vs_config_file": cmp(ref, fs), "feature_stats_path": d["feature_stats_path"]}
    print(name, out[name], flush=True)
out["passed"] = all(v["feature_stats_vs_config_file"]["max_abs_diff"] <= 1e-9 for v in out.values() if isinstance(v, dict))
(HERE / "verify_unified.json").write_text(json.dumps(out, indent=1))
print("passed", out["passed"])

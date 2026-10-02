"""Step 6 of the 2026-10-01 merge: every statistics file of the v5.2d view root recomputed with the code paths that wrote
the v5.2c ones (WSS log statistics, cycle TAWSS / OSI statistics, X5Dcap input-feature z-scores). The v5.2c files are kept
under <view root>/_stats_v52c/ for the comparison report (stats_report.json) and removed at the final clean-up.

    python 06_stats.py
"""
import json, shutil, sys, time
from pathlib import Path

G = Path("/public/newhome/cy/Digital_twin/GNN"); sys.path.insert(0, str(G))
C = G / "data_wss_v5/views_v5_2d_20261001"; V = C / "wss_min_view_v1"; BK = C / "_stats_v52c"
HERE = Path(__file__).resolve().parent
FOLDS = range(5)
SKIP = {"generated_at", "created_at", "split_path", "split_sha256", "statistics_scope", "scope", "_provenance", "label_fix_20260930"}


def compare(a, b):
    worst = [0.0, ""]
    def walk(x, y, p):
        if isinstance(x, dict):
            for k in x:
                if k in SKIP or k not in y: continue
                walk(x[k], y[k], f"{p}.{k}")
        elif isinstance(x, list) and isinstance(y, list):
            if len(x) != len(y): worst[:] = [float("inf"), p]; return
            for i, (xi, yi) in enumerate(zip(x, y)): walk(xi, yi, f"{p}[{i}]")
        elif isinstance(x, (int, float)) and not isinstance(x, bool) and isinstance(y, (int, float)):
            d = abs(float(x) - float(y))
            if d > worst[0]: worst[:] = [d, p]
    walk(a, b, "")
    return {"max_abs_diff": worst[0], "field": worst[1]}


def backup(p: Path) -> Path:
    dst = BK / p.relative_to(C); dst.parent.mkdir(parents=True, exist_ok=True)
    if not dst.exists(): shutil.move(str(p), str(dst))
    elif p.exists(): p.unlink()
    return dst


def train(split: Path) -> list[str]:
    return json.loads(split.read_text())["train_cases"]


def main():
    from wss_v5.views import wall_cycle_v1 as CY
    from wss_v5.views.wss_min_view import write_wss_stats
    from training_wss_min import config as TC
    from training_wss_min import dataset as D
    rep = {"date": time.strftime("%Y-%m-%dT%H:%M:%S%z")}
    jobs = [(V / "split_v52_ind_train170_test91.json", "wss_global_stats_ind_train170.json", "v5.2d IND train170"),
            (V / "split_v52p4_full265_train265_test8.json", "wss_global_stats_full265_train265.json", "v5.2d full265 train265")]
    jobs += [(V / "cv5_v52" / f"fold{k}.json", f"cv5_v52/wss_global_stats_fold{k}_train.json", f"v5.2d CV5 fold{k} train") for k in FOLDS]
    jobs += [(V / "syn_v52/ind_syn.json", "syn_v52/wss_global_stats_ind_syn_train.json", "v5.2d IND + synthetic train")]
    jobs += [(V / "syn_v52" / f"cv5_fold{k}_syn.json", f"syn_v52/wss_global_stats_cv5_fold{k}_syn_train.json", f"v5.2d CV5 fold{k} + synthetic train") for k in FOLDS]
    for split, name, scope in jobs:
        old = json.loads(backup(V / name).read_text())
        p = write_wss_stats(train(split), V, split, filename=name, scope=f"{scope} partition only (v5.2d, 2026-10-01 merge)")
        new = json.loads(p.read_text())
        rep[f"wss:{name}"] = {"n_cases": new["n_cases"], "log_mean": [old["log"]["mean"], new["log"]["mean"]], "log_std": [old["log"]["std"], new["log"]["std"]], **compare(old, new)}
    cyc = C / "wss_min_cycle_v1"
    scopes = {"ind_train170": V / "split_v52_ind_train170_test91.json", "full265_train265": V / "split_v52p4_full265_train265_test8.json"}
    scopes.update({f"cv5_fold{k}": V / "cv5_v52" / f"fold{k}.json" for k in FOLDS})
    for scope, split in scopes.items():
        olds = {t: json.loads(backup(cyc / "stats" / f"cycle_stats_{t}_{scope}.json").read_text()) for t in ("tawss", "osi_linear", "osi_logit")}
        CY.build_stats(cyc, scope, split)
        for t, old in olds.items():
            rep[f"cycle:{t}_{scope}"] = compare(old, json.loads((cyc / "stats" / f"cycle_stats_{t}_{scope}.json").read_text()))
    cfg = json.loads((G / "training_wss_min/configs/wss_v52c_labelfix_20260930/X5Dcap_asym2_s1234.json").read_text())
    feats = tuple(cfg["data"]["input_features"]); extra = tuple(f for f in feats if f in TC.SIDECAR_FEATURE_KEYS)
    fjobs = {"ind": ("split_v52_ind_train170_test91.json", "wss_global_stats_ind_train170.json"), "full265": ("split_v52p4_full265_train265_test8.json", "wss_global_stats_full265_train265.json")}
    fjobs.update({f"cv5_fold{k}": (f"cv5_v52/fold{k}.json", f"cv5_v52/wss_global_stats_fold{k}_train.json") for k in FOLDS})
    for name, (split, stats) in fjobs.items():
        f = C / "feature_stats" / f"{name}_X5Dcap_train_feature_stats.json"
        old = json.loads(backup(f).read_text())
        st = D.load_wss_stats(V / stats)
        cases = D.load_partition(str(V / split), "train", st, strict=True, target="wss", data_root=V, required_frame_version="v5_atlas_frame_v1",
                                 extra_point_features=extra, point_features_root=[str(C / "wss_min_geom_v2"), str(C / "wss_min_flowref_v1")])
        fs = D.compute_feature_stats(cases, feats, cfg["data"]["curvature_transform"])
        fs["_provenance"] = {"train_split": str(V / split), "n_cases": len(cases), "wss_stats": str(V / stats), "input_features": list(feats),
                             "point_features_root": [str(C / "wss_min_geom_v2"), str(C / "wss_min_flowref_v1")], "data_version": "v5.2d (2026-10-01 merge)",
                             "predecessor": old.get("_provenance")}
        f.write_text(json.dumps(fs, indent=2))
        d = compare({k: v for k, v in old.items() if not k.startswith("_")}, fs)
        rel = max((abs(fs[k]["mean"] - old[k]["mean"]) / (abs(old[k]["std"]) + 1e-12) for k in old if not k.startswith("_") and k in fs), default=0.0)
        rep[f"feature:{name}"] = {"n_cases": len(cases), **d, "max_mean_shift_in_std": rel}
    (HERE / "stats_report.json").write_text(json.dumps(rep, indent=1, default=float))
    print(json.dumps(rep, indent=1, default=float)[:6000])


if __name__ == "__main__":
    main()

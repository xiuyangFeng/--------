"""Step 5a of the 2026-10-01 merge: refit the cap-area iliac flow-split rule on the v5.2d data.

Same protocol and code as the 2026-09-30 refit (training_wss_min.tools.fit_flow_split_rule.side_rows, OLS on the train136
sides, test34 for reference). First the rule in use (v5.2c) is re-derived bit-exactly from the pre-merge case.h5 of every
unit (the replaced ones are in data_wss_v5/_replaced_v52c_20261001), then the new rule is fitted on the v5.2d snapshot.
The outlet renaming changes which iliac branch is 'external' on a side for 14 real units, the rebuilt WANG_TIAN_QING-1/after
is a new geometry (test34? no: it is not in train136/test34 -> unused here).

    python 05_refit_rule.py
"""
import json, sys, time
from pathlib import Path
import numpy as np

G = Path("/public/newhome/cy/Digital_twin/GNN"); sys.path.insert(0, str(G))
from training_wss_min.tools import fit_flow_split_rule as F  # noqa: E402

HERE = Path(__file__).resolve().parent
C = G / "data_wss_v5/views_v5_2d_20261001"; S = G / "data_wss_v5/anatomy_pointcloud_v5_2d_20261001/cases"
REPL = G / "data_wss_v5/_replaced_v52c_20261001/snapshot"
OLD_RULE = C / "wss_min_flowref_v1/flow_split_rule_v52c_train136.json"
SPLIT136 = C / "wss_min_view_v1/legacy_splits/views_v5_1/split_V5_train136_test34.json"


def fit(case_dirs, train, test):
    rows = []
    for cid, d in case_dirs.items():
        for r in F.side_rows(d):
            r["cid"] = cid; r["part"] = "train" if cid in train else "test" if cid in test else None
            rows.append(r)
    y = np.array([r["y"] for r in rows]); x = np.array([r["x"] for r in rows])
    tr = np.array([r["part"] == "train" for r in rows]); te = np.array([r["part"] == "test" for r in rows])
    a, b = np.polyfit(x[tr], y[tr], 1)
    yhat = a * x + b; murray = 1.5 * x
    per = {}
    for cohort in sorted({r["cohort"] for r in rows}):
        m = np.array([r["cohort"] == cohort for r in rows])
        per[cohort] = {"n_sides": int(m.sum()), "r2_rule": F.r2(y[m], yhat[m]), "r2_murray_raw": F.r2(y[m], murray[m])}
    return {"a": float(a), "b": float(b), "n_train_sides": int(tr.sum()), "n_test_sides": int(te.sum()),
            "train_r2": F.r2(y[tr], yhat[tr]), "test_r2": F.r2(y[te], yhat[te]),
            "murray_raw_r2_train": F.r2(y[tr], murray[tr]), "murray_raw_r2_test": F.r2(y[te], murray[te]),
            "residual_sd_train": float((y[tr] - yhat[tr]).std()), "residual_sd_test": float((y[te] - yhat[te]).std()),
            "target_sd": float(y[tr | te].std()), "per_cohort": per}


def main():
    split = json.loads(SPLIT136.read_text()); train, test = set(split["train_cases"]), set(split["test_cases"])
    old = json.loads(OLD_RULE.read_text())
    pre = {c: (REPL / c.replace("/", "__") if (REPL / c.replace("/", "__")).is_dir() else S / c.replace("/", "__")) for c in sorted(train | test)}
    orig = fit(pre, train, test)
    assert orig["a"] == old["a"] and orig["b"] == old["b"] and orig["n_train_sides"] == old["n_train_sides"], (orig["a"], old["a"], orig["b"], old["b"])
    src = {c: S / c.replace("/", "__") for c in sorted(train | test)}
    changed = sorted(c for c in src if (REPL / c.replace("/", "__")).is_dir())
    new = fit(src, train, test)
    rule = {"rule": old["rule"], "a": new["a"], "b": new["b"], "feature": old["feature"], "target": old["target"], "fit_on": old["fit_on"],
            **{k: v for k, v in new.items() if k not in ("a", "b")},
            "protocol_shares": old["protocol_shares"], "cases_with_missing_sides": old.get("cases_with_missing_sides", []),
            "deployment_inputs_only_geometry": True, "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
            "sources": "each unit's v5.2d case.h5 (data_wss_v5/anatomy_pointcloud_v5_2d_20261001)",
            "units_changed_since_v52c": changed,
            "supersedes": {"path": str(OLD_RULE), "a": old["a"], "b": old["b"], "reproduced_bit_exactly_from_pre_merge_sources": True},
            "script": str(Path(__file__).resolve())}
    out = C / "wss_min_flowref_v1/flow_split_rule_v52d_train136.json"
    out.write_text(json.dumps(rule, indent=1, ensure_ascii=False))
    (HERE / "flow_split_rule_train136_v52d.json").write_text(json.dumps(rule, indent=1, ensure_ascii=False))
    xs = np.linspace(-1.5, 1.5, 7)
    print(json.dumps({"old": [old["a"], old["b"]], "new": [new["a"], new["b"]], "units_changed_in_train136_test34": changed,
                      "train_r2": [old["train_r2"], new["train_r2"]], "test_r2": [old["test_r2"], new["test_r2"]],
                      "delta_logratio_over_x": np.round((new["a"] - old["a"]) * xs + (new["b"] - old["b"]), 4).tolist()}, indent=1))
    print("->", out)


if __name__ == "__main__":
    main()

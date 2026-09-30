"""Refit the cap-area iliac flow-split rule (X5Dcap input features) on the corrected data (2026-09-30).

The rule in use (data_wss_v5/views_v5_1/wss_min_flowref_v1/flow_split_rule_train136.json, a = 1.1504, b = 0.1473) was fitted
on the CFD branch flows of the v5.1 train136 sides; 5 of those units carried wrong boundary conditions (3 with swapped iliac
flows) and 11 had the pre-end-radius-fix cap radii. Same protocol and code (training_wss_min.tools.fit_flow_split_rule.side_rows,
OLS on the train136 sides, test34 for reference), each unit read from the case.h5 its current v5.2c view was built from.
Step 1 re-derives the original rule from the original sources bit-exactly. Writes a NEW file; nothing existing is changed.

    python refit_flow_split_rule.py
"""
import json
import sys
import time
from pathlib import Path

import h5py
import numpy as np

G = Path("/public/newhome/cy/Digital_twin/GNN")
sys.path.insert(0, str(G))
from training_wss_min.tools import fit_flow_split_rule as F  # noqa: E402

HERE = Path(__file__).resolve().parent
OLD_RULE = G / "data_wss_v5/views_v5_1/wss_min_flowref_v1/flow_split_rule_train136.json"
SPLIT136 = G / "data_wss_v5/views_v5_1/wss_min_view_v1/split_V5_train136_test34.json"
V51 = G / "data_wss_v5/anatomy_pointcloud_v5_1_20260916/cases"
VIEWS = [G / "data_wss_v5/views_v5_2c_20260930/wss_min_view_v1", G / "data_wss_v5/views_v5_2p5_full265_20260930/wss_min_view_v1"]


def canonical_h5(cid: str) -> Path:
    for v in VIEWS:
        b = v / cid / "bundle.npz"
        if b.exists():
            return Path(json.loads((b.resolve().parent / "view_report.json").read_text())["source"]["case_h5"])
    raise FileNotFoundError(cid)


def fit(case_dirs: dict, train: set, test: set) -> dict:
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


def main() -> None:
    split = json.loads(SPLIT136.read_text()); train, test = set(split["train_cases"]), set(split["test_cases"])
    old = json.loads(OLD_RULE.read_text())
    orig = fit({c: V51 / c.replace("/", "__") for c in sorted(train | test)}, train, test)
    assert orig["a"] == old["a"] and orig["b"] == old["b"] and orig["n_train_sides"] == old["n_train_sides"], (orig, old)
    src = {c: canonical_h5(c).parent for c in sorted(train | test)}
    changed = sorted(c for c, d in src.items() if d.resolve() != (V51 / c.replace("/", "__")).resolve())
    new = fit(src, train, test)
    rule = {"rule": old["rule"], "a": new["a"], "b": new["b"], "feature": old["feature"], "target": old["target"],
            "fit_on": "train136 sides only (v5.1 train136 = IND train170 minus nothing new; same protocol as the 09-15 rule)",
            **{k: v for k, v in new.items() if k not in ("a", "b")},
            "protocol_shares": old["protocol_shares"], "cases_with_missing_sides": old.get("cases_with_missing_sides", []),
            "deployment_inputs_only_geometry": True, "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
            "sources": "each unit's current v5.2c case.h5 (view_report source); label-fixed and end-radius-rebuilt units differ from v5.1",
            "units_with_new_source": changed,
            "supersedes": {"path": str(OLD_RULE), "a": old["a"], "b": old["b"], "reproduced_bit_exactly": True},
            "script": str(Path(__file__).resolve())}
    out = HERE / "flow_split_rule_train136_v52c.json"
    out.write_text(json.dumps(rule, indent=1, ensure_ascii=False))
    xs = np.linspace(-1.5, 1.5, 7)
    print(json.dumps({"old": [old["a"], old["b"]], "new": [new["a"], new["b"]], "units_with_new_source": len(changed),
                      "train_r2": [old["train_r2"], new["train_r2"]], "test_r2": [old["test_r2"], new["test_r2"]],
                      "delta_logratio_over_x": np.round((new["a"] - old["a"]) * xs + (new["b"] - old["b"]), 4).tolist()}, indent=1))
    print("->", out)


if __name__ == "__main__":
    main()

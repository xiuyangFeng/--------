"""Fit the deployable iliac flow-split rule on train138 and write it next to the flowref sidecar.

Target   : log(Q_external / Q_internal) at the peak step (1162) per common-iliac side, from the CFD interface
           fluxes stored in case.h5 (labels only; never a model input).
Feature  : log((r_cap_ext / r_cap_int)^2) — the virtual-cap radii of the point-cloud geometry program, i.e. the
           outlet cross-section at the cut, available for a new patient without CFD.
Rule     : log(Q_ext/Q_int) = a * log((r_ext/r_int)^2) + b, ordinary least squares on the train138 sides only.
The fitted JSON is consumed by wss_v5.views.wall_flowref_v1 (pack v1.2) to build ``log_q_branch_capfit`` /
``log_tau0_capfit``.  Root and common-iliac shares stay at the protocol values (1, 0.5, 0.5).

    python -m training_wss_min.tools.fit_flow_split_rule
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import h5py
import numpy as np

from wss_v5 import contract as VC
from wss_v5.views import wall_flowref_v1 as FR

EXTERNAL = {"out-le", "out-re"}
INTERNAL = {"out-li", "out-ri"}
PEAK_STEP = 1162


def side_rows(case_dir: Path) -> list[dict]:
    with h5py.File(case_dir / "case.h5", "r") as h5:
        step = h5["wall_temporal/step"][()]
        k = int(np.where(step == PEAK_STEP)[0][0]) if PEAK_STEP in step else int(np.argmax(np.abs(h5["interfaces/inlet/flux_outward_m3s"][()])))
        flux = {o: float(h5[f"interfaces/{o}/flux_outward_m3s"][()][k]) for o in VC.OUTLET_ORDER}
        g = h5["geometry"]
        cap_labels = json.loads(g.attrs["cap_labels"])
        cap_radius = g["cap_radius_mm"][()]
        segments = json.loads(g.attrs["atlas_segments"])
        role, cohort = str(h5.attrs["role"]), str(h5.attrs["cohort"])
    leaves = {s["outlet_name"]: s for s in segments if s.get("ends_at_leaf")}
    rows = []
    for parent in {s["parent_id"] for s in leaves.values()}:
        kids = [s for s in leaves.values() if s["parent_id"] == parent]
        ext = [s for s in kids if s["outlet_name"] in EXTERNAL]
        inte = [s for s in kids if s["outlet_name"] in INTERNAL]
        if len(kids) != 2 or len(ext) != 1 or len(inte) != 1:
            continue
        e, i = ext[0]["outlet_name"], inte[0]["outlet_name"]
        if e not in cap_labels or i not in cap_labels:
            continue
        re, ri = float(cap_radius[cap_labels.index(e)]), float(cap_radius[cap_labels.index(i)])
        if not (np.isfinite(re) and np.isfinite(ri) and re > 0 and ri > 0 and flux[e] > 0 and flux[i] > 0):
            continue
        rows.append(dict(case=case_dir.name, role=role, cohort=cohort, side=e, y=float(np.log(flux[e] / flux[i])),
                         x=float(np.log((re / ri) ** 2))))
    return rows


def r2(y: np.ndarray, yhat: np.ndarray) -> float:
    return float(1.0 - ((y - yhat) ** 2).sum() / ((y - y.mean()) ** 2).sum())


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(FR.CAPFIT_RULE_PATH))
    ap.add_argument("--expect-train", type=int, default=138, help="2026-09-15: v5.1 snapshot has train136")
    args = ap.parse_args(argv)
    rows, skipped, roles = [], [], {}
    for case_dir in sorted((VC.SNAPSHOT_ROOT / "cases").glob("*")):
        if (case_dir / "case.h5").is_file():
            with h5py.File(case_dir / "case.h5", "r") as h5:
                roles[case_dir.name] = str(h5.attrs["role"])
            case_rows = side_rows(case_dir)
            rows.extend(case_rows)
            if len(case_rows) < 2:
                skipped.append((case_dir.name, roles[case_dir.name], len(case_rows)))
    if sum(v == "train" for v in roles.values()) != args.expect_train:
        raise RuntimeError(f"expected {args.expect_train} train cases in the snapshot, found {sum(v == 'train' for v in roles.values())}")
    y = np.array([r["y"] for r in rows]); x = np.array([r["x"] for r in rows])
    train = np.array([r["role"] == "train" for r in rows]); test = ~train
    print("cases with <2 usable sides (skipped sides fall back to Murray-on-caps in the sidecar):", skipped)
    a, b = np.polyfit(x[train], y[train], 1)
    yhat = a * x + b
    murray = 3.0 * x / 2.0   # Murray R^3 on the same cap radii, for reference
    per_cohort = {}
    for cohort in sorted({r["cohort"] for r in rows}):
        m = np.array([r["cohort"] == cohort for r in rows])
        per_cohort[cohort] = {"n_sides": int(m.sum()), "r2_rule": r2(y[m], yhat[m]), "r2_murray_raw": r2(y[m], murray[m])}
    rule = {
        "rule": "log(Q_ext/Q_int) = a * log((r_cap_ext / r_cap_int)^2) + b", "a": float(a), "b": float(b),
        "feature": "virtual-cap radii of the geometry program (geometry/cap_radius_mm), external over internal iliac",
        "target": f"log(Q_ext/Q_int) at peak step {PEAK_STEP} from interfaces/*/flux_outward_m3s (labels only)",
        "fit_on": f"train{args.expect_train} sides only", "n_train_sides": int(train.sum()), "n_test_sides": int(test.sum()),
        "train_r2": r2(y[train], yhat[train]), "test_r2": r2(y[test], yhat[test]),
        "murray_raw_r2_train": r2(y[train], murray[train]), "murray_raw_r2_test": r2(y[test], murray[test]),
        "residual_sd_train": float((y[train] - yhat[train]).std()), "residual_sd_test": float((y[test] - yhat[test]).std()),
        "target_sd": float(y.std()), "per_cohort": per_cohort,
        "protocol_shares": {"root": 1.0, "common_iliac": 0.5}, "cases_with_missing_sides": skipped,
        "deployment_inputs_only_geometry": True, "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "snapshot_root": str(VC.SNAPSHOT_ROOT),
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(rule, indent=1), encoding="utf-8")
    print(json.dumps({k: v for k, v in rule.items() if k not in ("per_cohort",)}, indent=1))
    print(json.dumps(per_cohort, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

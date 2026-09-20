"""Parser regression: compare V5 wall labels with the legacy WSS-min bundle of the same case (same raw text -> bit-identical)."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import h5py
import numpy as np

from wss_pinn.utils import ROOT

from . import contract as C


def compare(canonical_id: str, root: Path) -> dict:
    h5_path = C.case_dir(canonical_id, root) / "case.h5"
    old_path = ROOT / "data_wss_min" / canonical_id / "bundle.npz"
    out = {"canonical_id": canonical_id, "old_bundle": str(old_path), "available": old_path.is_file()}
    if not old_path.is_file() or not h5_path.is_file():
        return out
    old = np.load(old_path, allow_pickle=True)
    with h5py.File(h5_path, "r") as h5:
        src = h5["wall_static/source_row"][()]
        valid = h5["wall_static/valid"][()]
        new_wss = h5["wall_temporal/wss_scalar_pa"][()]
        new_vec = h5["wall_temporal/wss_vector_pa"][()]
        new_p = h5["wall_temporal/pressure_pa"][()]
        steps = h5["wall_temporal/step"][()]
    if not np.array_equal(steps, old["steps"]):
        out["steps_equal"] = False
        return out
    rows = src[valid]
    old_wss = old["wall_wss"][:, rows]
    old_vec = old["wall_wss_vec"][:, rows]
    old_p = old["wall_pressure"][:, rows]
    out.update({
        "steps_equal": True, "compared_nodes": int(valid.sum()), "old_rows": int(old["wall_wss"].shape[1]),
        "wss_max_abs_diff": float(np.max(np.abs(new_wss[:, valid] - old_wss))),
        "wss_exact_fraction": float(np.mean(new_wss[:, valid] == old_wss)),
        "vector_max_abs_diff": float(np.max(np.abs(new_vec[:, valid] - old_vec))),
        "pressure_max_abs_diff": float(np.max(np.abs(new_p[:, valid] - old_p))),
        "old_bundle_mtime": old_path.stat().st_mtime,
    })
    return out


def main(argv=None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=C.SNAPSHOT_ROOT)
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--cases", nargs="*")
    args = parser.parse_args(argv)
    if args.cases:
        cases = args.cases
    else:
        cases = [json.loads(p.read_text())["canonical_id"] for p in sorted((args.root / "cases").glob("*/manifest.json"))]
    results = [compare(c, args.root) for c in cases]
    audit = args.root / "audits" / args.run_name
    audit.mkdir(parents=True, exist_ok=True)
    (audit / "regression_vs_wss_min.json").write_text(json.dumps(results, indent=1, ensure_ascii=False), encoding="utf-8")
    for r in results:
        print(r["canonical_id"], "n/a" if not r.get("available") else f"exact={r.get('wss_exact_fraction')} maxdiff={r.get('wss_max_abs_diff')} p={r.get('pressure_max_abs_diff')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Build ``reference.json`` (geometry ranges + CV3 out-of-fold p99 population) for a WSS release.

    PYTHONPATH=. python -m wss_deploy.build_reference_profiles --release X5D_v51_5seed_20260916 [--write]

Sources (read-only, training side):
* population: ``data_wss_v5/views_v5_1/wss_min_cascade_v1/<case>/features.npz::wall_log_wss_base`` =
  natural-log peak-frame WSS predicted for every train136 case by the CV3 fold model that did *not*
  train on it (X5D_v51 fold runs, seed 1234).  The per-case spatial p99 of exp(.) is the reference value.
* geometry: per-branch centreline radius/length of the same 136 training atlases (case.h5).

The file is a sidecar next to ``release.json``: it does not enter the release fingerprint, so jobs
bound to the release keep validating, and every run records its SHA256.  It is *not* copied into
``release.json`` on purpose.
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
from pathlib import Path

import numpy as np

from .metrics import LOW_PA, HIGH_PA, VERY_HIGH_PA
from .paths import PROJECT_ROOT, RELEASE_DIR

SEMANTIC_CN = {0: "主动脉", 1: "左髂总", 2: "右髂总", 3: "左髂外", 4: "左髂内", 5: "右髂外", 6: "右髂内"}
GEOMETRY_FIELDS = ("length_mm", "radius_min_mm", "radius_median_mm", "radius_max_mm")
MARGIN = 0.10  # bounds = [min × (1 − MARGIN), max × (1 + MARGIN)] of the training atlases


def population_protocol(spacing_placeholder: float = 1.0) -> dict:
    """The exact ``statistics_protocol`` block written by ``metrics.compute`` (volatile keys excluded)."""
    return {
        "field": "fixed_peak_systolic_frame", "support": "prediction_point_cloud", "quantile_method": "linear",
        "p99_definition": "固定收缩期帧预测点云的空间第 99 百分位，不是时间最大值",
        "position_definition": "位置和黄色标记对应全场最大值预测点，不对应 p99",
        "area_method": "point_fraction_times_input_surface_area", "area_label": "估计面积",
        "area_note": "点数占比乘输入壁面总面积，未做逐面片面积加权",
        "hotspot_definition": "WSS ≥ 空间 p99 的预测点；同值并列时可超过 1%",
    }


def collect(project_root: Path = PROJECT_ROOT) -> dict:
    import h5py
    from scipy.stats import spearmanr
    from wss_features.atlas import semantics
    root = Path(project_root)
    split_path = root / "data_wss_v5/views_v5_1/wss_min_view_v1/split_V5_train136_test34.json"
    split = json.loads(split_path.read_text(encoding="utf-8"))
    train = list(split["train_cases"])
    cascade = root / "data_wss_v5/views_v5_1/wss_min_cascade_v1"
    view = root / "data_wss_v5/views_v5_1/wss_min_view_v1"
    snapshot = root / "data_wss_v5/anatomy_pointcloud_v5_1_20260916/cases"
    manifest = json.loads((cascade / "cascade_manifest.json").read_text(encoding="utf-8"))
    oof, truth, geometry = {}, {}, {}
    for cid in train:
        z = np.load(cascade / cid / "features.npz")
        pred = np.exp(z["wall_log_wss_base"].astype(np.float64))
        bundle = np.load(view / cid / "bundle.npz", allow_pickle=True)
        if not np.array_equal(bundle["wall_node_id_cas"], z["wall_node_id_cas"]):
            raise ValueError(f"{cid}: cascade rows do not match the frozen view rows")
        steps = bundle["steps"].tolist()
        peak = bundle["wall_wss"][steps.index(int(bundle["peak_step"]))].astype(np.float64)
        oof[cid] = float(np.quantile(pred, .99)); truth[cid] = float(np.quantile(peak, .99))
        with h5py.File(snapshot / cid.replace("/", "__") / "case.h5", "r") as h5:
            g = h5["geometry"]; table = g["atlas_table"][()]
            cols = json.loads(g.attrs["atlas_columns"]); segs = json.loads(g.attrs["atlas_segments"])
            col = lambda n: table[:, cols.index(n)]
            sem = semantics(segs)
            for s in segs:
                sid = int(s["segment_id"]); name = SEMANTIC_CN.get(sem.get(sid, -1))
                if name is None:
                    continue
                mask = col("segment_id").astype(int) == sid
                if mask.sum() < 3:
                    continue
                r = col("radius_mm")[mask]
                d = geometry.setdefault(name, {k: [] for k in GEOMETRY_FIELDS})
                d["length_mm"].append(float(s.get("length_mm", 0.0))); d["radius_min_mm"].append(float(r.min()))
                d["radius_median_mm"].append(float(np.median(r))); d["radius_max_mm"].append(float(r.max()))
    p = np.array([oof[c] for c in train]); t = np.array([truth[c] for c in train])
    validation = {
        "n_cases": int(len(train)), "spearman_oof_vs_cfd_p99": float(spearmanr(p, t).correlation),
        "median_ratio_oof_over_cfd_p99": float(np.median(p / t)),
        "log_ratio_sd": float(np.std(np.log(p / t))),
        "oof_p99_pa_p10_p50_p90": [float(x) for x in np.percentile(p, [10, 50, 90])],
        "cfd_p99_pa_p10_p50_p90": [float(x) for x in np.percentile(t, [10, 50, 90])],
        "fold_runs": manifest.get("fold_runs"), "cascade_pack": f"{manifest.get('pack')} {manifest.get('version')}",
        "seed": manifest.get("seed"),
    }
    return {"train_cases": train, "oof_p99_pa": oof, "cfd_p99_pa": truth, "geometry": geometry, "validation": validation,
            "split_version": split.get("split_version"), "split_sha256": split.get("source_split_sha256")}


def build(release_id: str, collected: dict, *, today: str | None = None, geometry_only: bool = False,
          geometry_note: str | None = None) -> dict:
    """Assemble the sidecar.  ``geometry_only`` omits the WSS population (e.g. for the PF6/VF6 volume release)."""
    today = today or dt.date.today().isoformat()
    v = collected["validation"]
    bounds = [{"path": "cloud.spacing_mm", "units": "mm", "min": 0.4, "max": 0.6,
               "note": "部署合同：0.5 mm 重采样；粗于 0.8 mm 精度明显下降"}]
    for branch, fields in collected["geometry"].items():
        for field in GEOMETRY_FIELDS:
            values = np.asarray(fields[field], dtype=np.float64)
            if not len(values):
                continue
            bounds.append({"path": f"geometry.{branch}.{field}", "units": "mm",
                           "min": round(float(values.min()) * (1 - MARGIN), 3), "max": round(float(values.max()) * (1 + MARGIN), 3),
                           "train_min": round(float(values.min()), 3), "train_max": round(float(values.max()), 3), "n": int(len(values))})
    geometry_reference = {
        "schema_version": "wss-deploy.geometry-reference/v1", "id": f"geometry-train136-{today}", "status": "validated",
        "release": release_id, "reference_set": "train136 centreline atlases (v5.1 snapshot)",
        "rule": f"bounds = training min × {1 - MARGIN:.2f} to training max × {1 + MARGIN:.2f}; outside → review",
        "bounds": bounds, "built_on": today, "split_version": collected.get("split_version"),
        **({"note": geometry_note} if geometry_note else {}),
    }
    if geometry_only:
        return {"schema_version": "wss-deploy.reference-sidecar/v1", "release": release_id, "built_on": today,
                "geometry_reference": geometry_reference}
    values = [collected["oof_p99_pa"][c] for c in collected["train_cases"]]
    population_reference = {
        "schema_version": "wss-deploy.population-reference/v1", "id": f"population-cv3-oof-p99-{today}", "status": "validated",
        "release": release_id, "reference_set": "train136 CV3 out-of-fold predictions (X5D_v51 fold models, seed 1234)",
        "source": "cv3_oof_predictions", "fold_count": 3, "metric": "p99_pa",
        "field": {"units": "Pa", "location": "wall", "kind": "scalar", "components": 1},
        "time_axis": [{"index": 0, "step": 1162, "time_s": 0.21, "label": "peak_systole"}],
        "statistics_protocol": population_protocol(),
        "values_pa": [round(x, 4) for x in values], "case_ids": list(collected["train_cases"]),
        "reference_support": "折外预测取自 CFD 壁面节点（中位间距约 0.41 mm），部署预测取自 STL 0.5 mm 重采样点云；两者都是模型预测的空间 p99",
        "caveats": [
            "参照为单 seed 折模型的折外预测，部署为五 seed 集成，集成会略压低极值",
            "参照病例来自训练集三队列（AG / AAA / ILO），与临床人群分布不同",
            "分位只表示在该参照分布中的位置，不是风险概率或临床阈值",
        ],
        "validation": {
            "n_cases": v["n_cases"], "spearman_oof_vs_cfd_p99": round(v["spearman_oof_vs_cfd_p99"], 4),
            "median_ratio_oof_over_cfd_p99": round(v["median_ratio_oof_over_cfd_p99"], 4),
            "log_ratio_sd": round(v["log_ratio_sd"], 4),
            "oof_p99_pa_p10_p50_p90": [round(x, 3) for x in v["oof_p99_pa_p10_p50_p90"]],
            "cfd_p99_pa_p10_p50_p90": [round(x, 3) for x in v["cfd_p99_pa_p10_p50_p90"]],
            "seed": v["seed"], "cascade_pack": v["cascade_pack"], "fold_runs": v["fold_runs"],
        },
        "thresholds_pa": [LOW_PA, HIGH_PA, VERY_HIGH_PA], "built_on": today,
        "split_version": collected.get("split_version"), "split_sha256": collected.get("split_sha256"),
    }
    return {"schema_version": "wss-deploy.reference-sidecar/v1", "release": release_id, "built_on": today,
            "geometry_reference": geometry_reference, "population_reference": population_reference}


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--release", default=RELEASE_DIR.name)
    parser.add_argument("--release-root", default=str(RELEASE_DIR.parent))
    parser.add_argument("--write", action="store_true", help="write <release>/reference.json (refuses to overwrite)")
    parser.add_argument("--out", default=None, help="write the JSON here instead (for inspection)")
    parser.add_argument("--geometry-only", action="store_true", help="only the geometry ranges (no WSS population)")
    parser.add_argument("--geometry-note", default=None, help="provenance note stored with a geometry-only profile")
    args = parser.parse_args(argv)
    collected = collect()
    profile = build(args.release, collected, geometry_only=args.geometry_only, geometry_note=args.geometry_note)
    text = json.dumps(profile, ensure_ascii=False, indent=1) + "\n"
    if args.out:
        Path(args.out).write_text(text, encoding="utf-8"); print(f"wrote {args.out}")
    if args.write:
        target = Path(args.release_root) / args.release / "reference.json"
        if target.exists():
            raise SystemExit(f"refusing to overwrite {target}; delete it first if a rebuild is intended")
        target.write_text(text, encoding="utf-8"); print(f"wrote {target}")
    summary = {"release": args.release, "n_geometry_bounds": len(profile["geometry_reference"]["bounds"])}
    if "population_reference" in profile:
        v = profile["population_reference"]["validation"]
        summary.update(n_population=len(profile["population_reference"]["values_pa"]),
                       spearman_oof_vs_cfd=v["spearman_oof_vs_cfd_p99"], median_ratio=v["median_ratio_oof_over_cfd_p99"])
    print(json.dumps(summary, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

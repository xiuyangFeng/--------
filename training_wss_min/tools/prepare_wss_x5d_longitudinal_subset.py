"""Prepare a three-seed X5D + *pruned* longitudinal geometry experiment on CPU.

``--channels`` selects which longitudinal value channels to append (their mask
channels come along automatically).  Every audit of the full-32 preparation is
kept; only the appended channel set and the experiment name change.

Run this on an allocated CPU node, not a login node.  The production loader
fits all feature statistics on train136 and checks all 170 full wall clouds
and their four density subsets.  No training, evaluation, or submission occurs.

    python -u -m training_wss_min.tools.prepare_wss_x5d_longitudinal_subset \
        --experiment-name wss_x5d_long_ref8_20260917 --channels ref8

``--execution-root`` redirects the fresh-initialization reference to an already
copied code/config snapshot. ``--sidecar-root`` can select a frozen copy of the
coverage export. Neither option copies files or changes the data split.
"""
from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path

import numpy as np
import torch

from training_wss_min import config as C
from training_wss_min import dataset as D
from training_wss_min import longitudinal_geometry as L
from training_wss_min import paired_initialization as P

ROOT = C.PROJECT_ROOT
NAME = "wss_x5d_longitudinal_20260917"   # overridden by --experiment-name
CHANNELS = L.VALUE_KEYS                   # overridden by --channels
PRESETS = {
    "all32": L.VALUE_KEYS,
    "ref8": tuple(k for k in L.VALUE_KEYS if k.startswith("geom_ref_")),
    "ref8pcround": tuple(k for k in L.VALUE_KEYS if k.startswith("geom_ref_")) + ("geom_pc_roundness",),
    "ref5": ("geom_ref_roundness", "geom_ref_upstream_min_distance", "geom_ref_upstream_min_area_ratio",
             "geom_ref_area_slope", "geom_ref_downstream_min_area_ratio"),
}


def feature_keys():
    """Interleaved value/mask columns, in the frozen VALUE_KEYS order."""
    return tuple(k for name in L.VALUE_KEYS if name in CHANNELS for k in (name, f"{name}_valid"))


def n_features():
    return 27 + 2 * len(CHANNELS)
BASE_NAME = "wss_v51_wave1_20260916"
SEEDS = (1234, 7, 2025)
CONFIGS = ROOT / "training_wss_min/configs" / NAME
EXP = ROOT / "training_wss_min/experiments" / NAME
SIDECAR = ROOT / "outputs/wss_v5_longitudinal_features_20260916/coverage_features/features"


def save(path: Path, value) -> None:
    """Write deterministic artifacts without replacing different evidence."""
    payload = json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if path.read_text() != payload:
            raise FileExistsError(f"refusing to overwrite different prepared artifact: {path}")
        return
    with path.open("x") as stream:
        stream.write(payload)


def reference_path(seed: int, root: Path = ROOT) -> Path:
    return root / "training_wss_min/configs" / BASE_NAME / f"X5D_v51_s{seed}.json"


def changes(before: dict, after: dict, prefix: str = "") -> dict:
    result = {}
    for key in sorted(before.keys() | after.keys()):
        path = f"{prefix}.{key}" if prefix else key
        a, b = before.get(key), after.get(key)
        if isinstance(a, dict) and isinstance(b, dict):
            result.update(changes(a, b, path))
        elif a != b:
            result[path] = {"before": a, "after": b}
    return result


def prepare_config(base: dict, seed: int, args) -> dict:
    cfg = copy.deepcopy(base)
    aid = f"X5D_long_s{seed}"
    cfg["name"] = f"{NAME}/{aid}"
    cfg["notes"] = (
        f"X5D v5.1 + {2 * len(CHANNELS)} longitudinal value/mask channels "
        f"({','.join(CHANNELS)}), seed {seed}; "
        "27 original channels retained in order; full X5D recipe unchanged; "
        "train136 valid-only geometry statistics; fresh paired initialization "
        "from same-seed X5D_v51 with appended columns initialized to zero. "
        "Density augmentation subsets fixed full-cloud longitudinal context."
    )
    cfg["data"]["input_features"] = list(base["data"]["input_features"]) + list(feature_keys())
    cfg["data"]["point_features_root"] = list(base["data"]["point_features_root"]) + [str(args.sidecar_root)]
    cfg["data"]["feature_stats_path"] = str(args.experiment_dir / "feature_stats/all_train136.json")
    cfg["train"]["init_reference_config"] = str(reference_path(seed, args.execution_root))
    allowed = {"name", "notes", "data.input_features", "data.point_features_root",
               "data.feature_stats_path", "train.init_reference_config"}
    if not set(changes(base, cfg)) <= allowed:
        raise ValueError("an original X5D setting changed outside the declared feature addition")
    if cfg["train"]["seed"] != seed or len(cfg["data"]["input_features"]) != n_features():
        raise ValueError(f"expected seed-matched 27 + {2 * len(CHANNELS)} input channels")
    return cfg


def loader_kwargs(cfg: dict) -> dict:
    data = cfg["data"]
    return dict(target=data["target"], target_normalization=data["target_normalization"],
                data_root=data["data_root"], required_frame_version=data["required_frame_version"],
                timesteps=data["timesteps"],
                extra_point_features=tuple(k for k in data["input_features"] if k in C.SIDECAR_FEATURE_KEYS),
                point_features_root=data["point_features_root"])


def validate_masks(case: dict, features: np.ndarray, names: tuple[str, ...]) -> dict:
    counts = {}
    for name in CHANNELS:
        _, valid = L.checked_pair(case[name], case[name + "_valid"], name, len(features))
        if (not np.array_equal(features[:, names.index(name + "_valid")], valid.astype(np.float32))
                or np.any(features[~valid, names.index(name)] != 0.0)):
            raise ValueError(f"{case['unit_id']}: invalid value/mask encoding for {name}")
        counts[name] = int(valid.sum())
    return counts


def audit_case(case: dict, partition: str, cfg: dict, stats: dict, sidecar_root: Path) -> dict:
    cid = case["unit_id"]
    names = tuple(cfg["data"]["input_features"])
    n = len(case["pos"])
    features = D.build_features(case, np.arange(n), names, stats)
    if features.shape != (n, n_features()) or not np.isfinite(features).all():
        raise ValueError(f"{cid}: full wall features are not finite Nx{n_features()}")
    for target in ("y_raw", "y_norm"):
        if len(case[target]) != n or not np.isfinite(case[target]).all():
            raise ValueError(f"{cid}: missing or nonfinite supervision rows in {target}")
    counts = validate_masks(case, features, names)
    path = sidecar_root / cid / "features.npz"
    sidecar_hash = L.sha256_file(path)
    source_hashes = {v["sha256"] for v in case["_longitudinal_sources"].values()}
    if source_hashes != {sidecar_hash}:
        raise ValueError(f"longitudinal file changed after production load: {path}")
    density = {}
    for level in cfg["data"]["density_aug_levels"]:
        level_data = D.load_density_level(case, cfg["data"]["density_aug_root"], level)
        rows = level_data["rows"]
        if len(np.unique(rows)) != len(rows):
            raise ValueError(f"{cid}: duplicate density rows at L{level}")
        sub = D.density_view(case, level_data, level)
        x = D.build_features(sub, np.arange(len(rows)), names, stats)
        if x.shape != (len(rows), n_features()) or not np.isfinite(x).all():
            raise ValueError(f"{cid}: invalid density features at L{level}")
        for key in (*feature_keys(), "pos", "y_raw", "y_norm"):
            if not np.array_equal(sub[key], case[key][rows], equal_nan=True):
                raise ValueError(f"{cid}: density L{level} row identity differs for {key}")
        if not np.array_equal(x[:, 27:], features[rows, 27:]):
            raise ValueError(f"{cid}: density L{level} standardized new channels differ")
        validate_masks(sub, x, names)
        density_path = Path(cfg["data"]["density_aug_root"]) / f"L{level}" / cid / "features.npz"
        density[str(level)] = {"rows": len(rows), "finite_59_channels": True,
                               "longitudinal_subset_exact": True, "supervision_subset_exact": True,
                               "sidecar_sha256": L.sha256_file(density_path)}
    # Do not retain every density cache while auditing the already-loaded train set.
    case.pop("_density", None)
    print(f"audit {partition} {cid}: {n} full rows, four density levels passed", flush=True)
    return {"canonical_id": cid, "partition": partition, "wall_points": n,
            "feature_count": n_features(), "all_finite": True, "all_supervision_rows_retained": True,
            "longitudinal_schema": L.SCHEMA, "sidecar": str(path), "sidecar_sha256": sidecar_hash,
            "valid_counts": counts, "density": density}


def verify_paired(cfg: dict) -> dict:
    candidate, evidence = P.build_paired_model(C.ExpConfig.from_dict(cfg))
    reference, _ = P.build_paired_model(C.ExpConfig.from_json(cfg["train"]["init_reference_config"]))
    before, after = reference.state_dict(), candidate.state_dict()
    if evidence["reference_state_sha256"] != P._state_hash(before):
        raise ValueError("reference construction was not reproducible")
    if set(before) != set(after) or evidence["new_keys"] or evidence["replaced_module_roots"]:
        raise ValueError("longitudinal addition unexpectedly changed model modules")
    layouts = P._expandable_keys(before, after, 27, n_features())
    for key, old in before.items():
        new = after[key]
        if key not in layouts:
            if not torch.equal(old, new):
                raise ValueError(f"shared tensor differs: {key}")
            continue
        src = dst = 0
        for old_width, new_width in layouts[key]:
            if not torch.equal(old[:, src:src + old_width], new[:, dst:dst + old_width]):
                raise ValueError(f"reference input columns changed: {key}")
            if torch.count_nonzero(new[:, dst + old_width:dst + new_width]).item():
                raise ValueError(f"appended input columns are not zero: {key}")
            src += old_width
            dst += new_width
    if not layouts or sorted(layouts) != sorted(evidence["zero_extended_input_keys"]):
        raise ValueError("missing paired input expansion evidence")
    return {"passed": True, "seed": cfg["train"]["seed"],
            "reference_config": evidence["reference_config"],
            "reference_config_sha256": evidence["reference_config_sha256"],
            "reference_state_sha256": evidence["reference_state_sha256"],
            "initial_state_sha256": evidence["initial_state_sha256"],
            "unchanged_tensors": len(before) - len(layouts),
            "zero_extended_input_keys": sorted(layouts), "appended_channels": 2 * len(CHANNELS),
            "shared_tensors_exact": True, "trained_checkpoint_loaded": False}


def main(argv=None) -> None:
    global NAME, CHANNELS, CONFIGS, EXP
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment-name", default=NAME)
    parser.add_argument("--channels", default="all32",
                        help="preset name (" + ", ".join(PRESETS) + ") or a comma separated value-channel list")
    parser.add_argument("--execution-root", type=Path, default=ROOT)
    parser.add_argument("--sidecar-root", type=Path, default=SIDECAR)
    parser.add_argument("--config-dir", type=Path, default=CONFIGS)
    parser.add_argument("--experiment-dir", type=Path, default=EXP)
    args = parser.parse_args(argv)
    NAME = args.experiment_name
    CHANNELS = PRESETS.get(args.channels) or tuple(a.strip() for a in args.channels.split(",") if a.strip())
    if not set(CHANNELS) <= set(L.VALUE_KEYS) or len(set(CHANNELS)) != len(CHANNELS):
        raise ValueError(f"unknown or duplicated longitudinal channels: {CHANNELS}")
    CHANNELS = tuple(k for k in L.VALUE_KEYS if k in CHANNELS)   # keep the frozen order
    CONFIGS = ROOT / "training_wss_min/configs" / NAME
    EXP = ROOT / "training_wss_min/experiments" / NAME
    if args.config_dir == Path(__file__).parent:   # never reached; keeps defaults explicit
        pass
    for key in ("execution_root", "sidecar_root", "config_dir", "experiment_dir"):
        setattr(args, key, getattr(args, key).resolve())
    bases = {seed: json.loads(reference_path(seed).read_text()) for seed in SEEDS}
    configs = {seed: prepare_config(base, seed, args) for seed, base in bases.items()}
    cfg = configs[SEEDS[0]]
    for seed in SEEDS:
        # The snapshot must already contain matching seed-specific donor configs.
        donor = json.loads(reference_path(seed, args.execution_root).read_text())
        if donor["data"]["input_features"] != bases[seed]["data"]["input_features"] or donor["model"] != bases[seed]["model"]:
            raise ValueError(f"snapshot X5D donor differs for seed {seed}")
        if configs[seed]["data"] != cfg["data"]:
            raise ValueError("three seeds must share the exact same input data and frozen statistics")
        run = ROOT / "training_wss_min/runs" / configs[seed]["name"]
        if run.exists():
            raise FileExistsError(f"refusing to prepare an existing training run: {run}")
    split_path = Path(cfg["data"]["split_path"])
    split = json.loads(split_path.read_text())
    train, test = split["train_cases"], split["test_cases"]
    if len(train) != 136 or len(test) != 34 or len(set(train + test)) != 170:
        raise ValueError("expected disjoint explicit train136/test34")
    manifest_path = args.sidecar_root / "longitudinal_manifest.json"
    manifest = json.loads(manifest_path.read_text())
    if manifest["schema"] != L.SCHEMA or tuple(manifest["feature_names"]) != L.FEATURE_KEYS:
        raise ValueError("longitudinal export schema or channel order differs")
    if not set(feature_keys()) <= set(manifest["feature_names"]):
        raise ValueError("selected channels are not all present in the frozen export")
    if manifest["case_count"] != 170 or {row["canonical_id"] for row in manifest["cases"]} != set(train + test):
        raise ValueError("longitudinal export does not cover this exact 170-case cohort")
    geometry_split = json.loads(Path(manifest["split"]).read_text())
    if any(set(geometry_split[key]) != set(split[key]) for key in ("train_cases", "test_cases")):
        raise ValueError("geometry export partition memberships differ from X5D")
    manifest_hash = L.sha256_file(manifest_path)
    kwargs = loader_kwargs(cfg)
    wss_stats = D.load_wss_stats(cfg["data"]["wss_stats_path"])
    print("loading explicit train136 with production loader for 59-channel statistics", flush=True)
    cases = D.load_partition(str(split_path), "train", wss_stats, strict=True, **kwargs)
    if [case["unit_id"] for case in cases] != train:
        raise ValueError("production training order differs from the frozen X5D split")
    names = tuple(cfg["data"]["input_features"])
    stats = D.compute_feature_stats(cases, names, cfg["data"]["curvature_transform"])
    L.validate_frozen_stats(stats, names, train)
    for seed, base in bases.items():
        old = json.loads(Path(base["data"]["feature_stats_path"]).read_text())
        changed = [key for key in base["data"]["input_features"] if key not in ("x", "y", "z") and old[key] != stats[key]]
        if changed:
            raise ValueError(f"original X5D feature normalization changed for seed {seed}: {changed}")
    stats["_provenance"] = {
        "train_split": str(split_path), "split_sha256": L.sha256_file(split_path),
        "n_cases": 136, "feature_count": n_features(), "data_root": cfg["data"]["data_root"],
        "sidecar_roots": cfg["data"]["point_features_root"],
        "original_longitudinal_export": str(SIDECAR),
        "longitudinal_manifest_sha256": manifest_hash,
        "curvature_transform": cfg["data"]["curvature_transform"],
        "statistics_scope": L.STATS_SCOPE, "original_27_feature_statistics_unchanged": True,
        "coordinate_channels": "x/y/z use normalized coordinates without fitted z-score",
    }
    save(Path(cfg["data"]["feature_stats_path"]), stats)
    rows = [audit_case(case, "train", cfg, stats, args.sidecar_root) for case in cases]
    del cases
    for cid in test:
        cohort, case_name = cid.rsplit("/", 1)
        case = D.load_case(cohort, case_name, wss_stats, **kwargs)
        rows.append(audit_case(case, "test", cfg, stats, args.sidecar_root))
    if L.sha256_file(manifest_path) != manifest_hash:
        raise ValueError("longitudinal manifest changed during preparation")
    paired = {}
    arms = []
    for seed, candidate in configs.items():
        aid = f"X5D_long_s{seed}"
        print(f"verifying 27 -> 59 paired initialization: {aid}", flush=True)
        paired[aid] = verify_paired(candidate)
        C.ExpConfig.from_dict(candidate)
        arms.append({"id": aid, "title": f"X5D v5.1 + {2 * len(CHANNELS)} longitudinal geometry/mask channels, seed {seed}",
                     "config": f"{aid}.json", "phase": 0, "modules": [f"longitudinal_geometry_{2 * len(CHANNELS)}"],
                     "seed": seed, "fold": None, "parent": f"X5D_v51_s{seed}", "depends_on": [],
                     "evaluate": ["best", "last"], "single_change": True,
                     "config_diff_vs_anchor": changes(bases[seed], candidate)})
    audit = {"status": "passed", "passed": True, "experiment": NAME, "training_started": False,
             "case_count": len(rows), "train_cases": 136, "test_cases": 34,
             "feature_count": n_features(), "appended_feature_count": 2 * len(CHANNELS),
             "appended_channels": list(CHANNELS), "statistics_train_only": True,
             "original_27_feature_statistics_unchanged": True, "all_cases_same_schema": True,
             "all_wall_supervision_retained": True, "density_levels": cfg["data"]["density_aug_levels"],
             "density_longitudinal_semantics": "subset fixed full-cloud ref/pc context; PC longitudinal geometry is not recomputed after thinning",
             "source_longitudinal_root": str(SIDECAR), "used_longitudinal_root": str(args.sidecar_root),
             "manifest_sha256": manifest_hash, "split": str(split_path),
             "split_sha256": L.sha256_file(split_path), "execution_root": str(args.execution_root),
             "paired_initialization": paired, "cases": rows}
    save(args.experiment_dir / "prepared_audit.json", audit)
    for seed, candidate in configs.items():
        save(args.config_dir / f"X5D_long_s{seed}.json", candidate)
    save(args.config_dir / "matrix.json", {
        "schema_version": 1, "experiment": NAME, "anchor": str(reference_path(1234)),
        "anchor_run": str(ROOT / "training_wss_min/runs" / BASE_NAME / "X5D_v51_s1234"),
        "control_id": None,
        "external_reference_runs": {f"X5D_v51_s{seed}": str(ROOT / "training_wss_min/runs" / BASE_NAME / f"X5D_v51_s{seed}" / "eval/ckpt_best/metrics.json") for seed in SEEDS},
        "expected_training_runs": 3, "expected_evaluations": 6, "arms": arms,
        "prepared_audit": str(args.experiment_dir / "prepared_audit.json"),
        "reading_rule": "Compare each seed against its same-seed X5D_v51 parent on the unchanged train136/test34 protocol; 400 epochs, train_loss selection, best/last evaluation; geometry masks affect inputs only. The existing test34 is exposed and is not used for model selection.",
    })
    print(json.dumps({"status": "prepared", "matrix": str(args.config_dir / "matrix.json"),
                      "audit": str(args.experiment_dir / "prepared_audit.json"), "seeds": list(SEEDS)}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()

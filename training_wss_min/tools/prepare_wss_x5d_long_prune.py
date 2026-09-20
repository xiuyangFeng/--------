"""Prepare the single-seed longitudinal pruning ladder (one experiment, four arms).

Every arm is a nested subset of the already fully audited ref-8 preparation
(`wss_x5d_long_ref8_20260917`), so its per-case / per-density audit and its
train136 valid-only statistics carry over verbatim.  This script re-verifies
that nesting on real cases, re-verifies the 27 -> N paired zero initialization
for every arm, and writes the configs plus matrix.  No training happens here.
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
NAME = "wss_x5d_long_prune_20260917"
REF8_NAME = "wss_x5d_long_ref8_20260917"
SEED = 1234
REF8 = tuple(k for k in L.VALUE_KEYS if k.startswith("geom_ref_"))
# ladder, ordered by the out-of-fold marginal-increment selection (offline/select_channels_oof)
ARMS = {
    "L8": (REF8, "ref 8 通道（折外解释力 0.00435 = 全 32 通道天花板的 92%）"),
    "L5": (("geom_ref_area_slope", "geom_ref_roundness", "geom_ref_upstream_min_area_ratio",
            "geom_ref_downstream_min_area_ratio", "geom_ref_upstream_min_distance"),
           "前向选出的 5 个（0.00404 = 85%）"),
    "L2": (("geom_ref_roundness", "geom_ref_upstream_min_distance"), "前 2 个（0.00291 = 62%）"),
    "L1": (("geom_ref_roundness",), "只留 roundness（0.00250 = 53%，覆盖率最高）"),
}


def keys_of(channels):
    return tuple(k for name in L.VALUE_KEYS if name in channels for k in (name, f"{name}_valid"))


def save(path: Path, value) -> None:
    payload = json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if path.read_text() != payload:
            raise FileExistsError(f"refusing to overwrite different prepared artifact: {path}")
        return
    with path.open("x") as stream:
        stream.write(payload)


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


def verify_paired(cfg: dict, n_features: int) -> dict:
    candidate, evidence = P.build_paired_model(C.ExpConfig.from_dict(cfg))
    reference, _ = P.build_paired_model(C.ExpConfig.from_json(cfg["train"]["init_reference_config"]))
    before, after = reference.state_dict(), candidate.state_dict()
    if evidence["reference_state_sha256"] != P._state_hash(before):
        raise ValueError("reference construction was not reproducible")
    if set(before) != set(after) or evidence["new_keys"] or evidence["replaced_module_roots"]:
        raise ValueError("pruned addition unexpectedly changed model modules")
    layouts = P._expandable_keys(before, after, 27, n_features)
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
    return {"passed": True, "appended_channels": n_features - 27,
            "reference_config": evidence["reference_config"],
            "reference_state_sha256": evidence["reference_state_sha256"],
            "initial_state_sha256": evidence["initial_state_sha256"],
            "unchanged_tensors": len(before) - len(layouts),
            "zero_extended_input_keys": sorted(layouts), "shared_tensors_exact": True}


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--execution-root", type=Path, default=ROOT)
    parser.add_argument("--config-dir", type=Path, default=ROOT / "training_wss_min/configs" / NAME)
    parser.add_argument("--experiment-dir", type=Path, default=ROOT / "training_wss_min/experiments" / NAME)
    parser.add_argument("--audit-cases", type=int, default=6)
    args = parser.parse_args(argv)
    args.config_dir, args.experiment_dir = args.config_dir.resolve(), args.experiment_dir.resolve()

    # the ref-8 donor lives next to this ladder in the source tree; its own
    # init_reference_config already points into the frozen execution snapshot.
    ref8_cfg_path = args.config_dir.parent / REF8_NAME / f"X5D_long_s{SEED}.json"
    base = json.loads(ref8_cfg_path.read_text())
    ref8_audit = ROOT / "training_wss_min/experiments" / REF8_NAME / "prepared_audit.json"
    parent = json.loads(ref8_audit.read_text())
    if parent.get("passed") is not True or parent["feature_count"] != 43:
        raise ValueError("the ref-8 preparation this ladder inherits from did not pass")
    if tuple(base["data"]["input_features"][27:]) != keys_of(REF8):
        raise ValueError("ref-8 donor config does not carry exactly the eight reference channels")

    configs, paired = {}, {}
    for aid, (channels, note) in ARMS.items():
        if not set(channels) <= set(REF8):
            raise ValueError(f"{aid}: ladder arms must be nested inside the audited ref-8 set")
        cfg = copy.deepcopy(base)
        cfg["name"] = f"{NAME}/{aid}_s{SEED}"
        cfg["notes"] = (f"沿程通道剪枝阶梯 {aid}：{note}；X5D_v51 底座其余配方逐位不变，"
                        f"统计量沿用 {REF8_NAME} 的 train136 valid-only 冻结文件（本臂通道是其子集），"
                        f"27 -> {27 + 2 * len(channels)} 配对零初始化，seed {SEED}。")
        cfg["data"]["input_features"] = list(base["data"]["input_features"][:27]) + list(keys_of(channels))
        allowed = {"name", "notes", "data.input_features"}
        if not set(changes(base, cfg)) <= allowed:
            raise ValueError(f"{aid}: something other than the channel set changed")
        C.ExpConfig.from_dict(cfg)
        run = ROOT / "training_wss_min/runs" / cfg["name"]
        if run.exists():
            raise FileExistsError(f"refusing to prepare an existing training run: {run}")
        configs[aid] = cfg

    # statistics reuse: every selected channel must already carry the frozen train136 contract
    split = json.loads(Path(base["data"]["split_path"]).read_text())
    stats = json.loads(Path(base["data"]["feature_stats_path"]).read_text())
    for aid, cfg in configs.items():
        L.validate_frozen_stats(stats, tuple(cfg["data"]["input_features"]), split["train_cases"])

    # nesting check on real cases: each arm's columns are exactly the ref-8 columns it selects
    kw = dict(target=base["data"]["target"], target_normalization=base["data"]["target_normalization"],
              data_root=base["data"]["data_root"], required_frame_version=base["data"]["required_frame_version"],
              timesteps=base["data"]["timesteps"],
              extra_point_features=tuple(k for k in base["data"]["input_features"] if k in C.SIDECAR_FEATURE_KEYS),
              point_features_root=base["data"]["point_features_root"])
    wss_stats = D.load_wss_stats(base["data"]["wss_stats_path"])
    ref_names = tuple(base["data"]["input_features"])
    checked = []
    picks = split["train_cases"][:args.audit_cases // 2] + split["test_cases"][:args.audit_cases - args.audit_cases // 2]
    for cid in picks:
        cohort, case_name = cid.rsplit("/", 1)
        case = D.load_case(cohort, case_name, wss_stats, **kw)
        n = len(case["pos"])
        full = D.build_features(case, np.arange(n), ref_names, stats)
        if full.shape != (n, 43) or not np.isfinite(full).all():
            raise ValueError(f"{cid}: ref-8 reference features are not finite Nx43")
        for aid, cfg in configs.items():
            names = tuple(cfg["data"]["input_features"])
            sub = D.build_features(case, np.arange(n), names, stats)
            cols = [ref_names.index(nm) for nm in names]
            if sub.shape != (n, len(names)) or not np.isfinite(sub).all():
                raise ValueError(f"{cid}/{aid}: pruned features are not finite")
            if not np.array_equal(sub, full[:, cols]):
                raise ValueError(f"{cid}/{aid}: pruned columns differ from the audited ref-8 columns")
        checked.append({"canonical_id": cid, "wall_points": n, "nested_columns_exact": True})
        print(f"nesting check passed: {cid} ({n} rows)", flush=True)

    arms = []
    for aid, cfg in configs.items():
        n_features = len(cfg["data"]["input_features"])
        print(f"verifying 27 -> {n_features} paired initialization: {aid}", flush=True)
        paired[aid] = verify_paired(cfg, n_features)
        arms.append({"id": f"{aid}_s{SEED}", "title": f"{aid}: {ARMS[aid][1]}", "config": f"{aid}_s{SEED}.json",
                     "phase": 0, "modules": [f"longitudinal_pruned_{n_features - 27}"], "seed": SEED,
                     "fold": None, "parent": f"X5D_v51_s{SEED}", "depends_on": [],
                     "evaluate": ["best", "last"], "single_change": True,
                     "input_dim": n_features, "channels": list(ARMS[aid][0]),
                     "config_diff_vs_anchor": changes(base, cfg)})

    save(args.experiment_dir / "prepared_audit.json", {
        "status": "passed", "passed": True, "experiment": NAME, "training_started": False,
        "seed": SEED, "inherits_full_audit_from": str(ref8_audit),
        "inherited_audit_cases": parent["case_count"], "inherited_density_levels": parent["density_levels"],
        "nesting_verified_cases": checked, "paired_initialization": paired,
        "feature_stats": base["data"]["feature_stats_path"], "statistics_train_only": True,
        "original_27_feature_statistics_unchanged": True,
        "arms": {aid: {"channels": list(ch), "input_dim": len(configs[aid]["data"]["input_features"])}
                 for aid, (ch, _) in ARMS.items()},
    })
    for aid, cfg in configs.items():
        save(args.config_dir / f"{aid}_s{SEED}.json", cfg)
    save(args.config_dir / "matrix.json", {
        "schema_version": 1, "experiment": NAME, "anchor": str(ref8_cfg_path),
        "anchor_run": str(ROOT / "training_wss_min/runs/wss_v51_wave1_20260916" / f"X5D_v51_s{SEED}"),
        "control_id": None,
        "external_reference_runs": {
            f"X5D_v51_s{SEED}": str(ROOT / f"training_wss_min/runs/wss_v51_wave1_20260916/X5D_v51_s{SEED}/eval/ckpt_best/metrics.json"),
            f"X5D_long_s{SEED}": str(ROOT / f"training_wss_min/runs/wss_x5d_longitudinal_20260917/X5D_long_s{SEED}/eval/ckpt_best/metrics.json")},
        "expected_training_runs": len(ARMS), "expected_evaluations": 2 * len(ARMS), "arms": arms,
        "prepared_audit": str(args.experiment_dir / "prepared_audit.json"),
        "reading_rule": ("单 seed 剪枝阶梯：四臂只差沿程通道集合，全部与同 seed 的 X5D_v51（0.7461）和全 32 通道臂"
                         "（X5D_long_s1234 0.7705）在同一 test34 协议下比较。单 seed 读数不能分辨 <0.02 的差别，"
                         "用途是确认剪枝没有损失，通道排序依据仍是折外边际增量。"),
    })
    print(json.dumps({"status": "prepared", "arms": list(ARMS), "seed": SEED,
                      "config_dir": str(args.config_dir)}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()

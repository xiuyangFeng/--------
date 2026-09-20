"""Prepare the query-patch receptive-field ladder (single seed, config-only arms).

Phase 0 (D6) measured the K16 query patch at ~2.1 mm radius = **8.3 % of the local
circumference**, while the within-bin residual is carried by m=1/m=2/m=3 circumferential
modes (net 59 % of the within-bin variance after the permutation null).  These arms widen
the refinement receptive field with the existing, already validated fixed-millimetre ball
branch; nothing else in the X5D_v51 recipe changes and no new feature or statistic appears.
"""
from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path

import torch

from training_wss_min import config as C
from training_wss_min import paired_initialization as P

ROOT = C.PROJECT_ROOT
NAME = "wss_local_morph_radius_20260917"
BASE_NAME = "wss_v51_wave1_20260916"
SEED = 1234
ARMS = {"W04": (4.0, 16), "W06": (6.0, 16), "W08": (8.0, 16)}   # overridden by --arms
RADIUS_BRANCH = ("query_patch.patch_r", "query_patch.to_gamma_r", "query_patch.to_beta_r",
                 "query_patch.out_r", "query_patch.attn_r")


def save(path: Path, value) -> None:
    payload = json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if path.read_text() != payload:
            raise FileExistsError(f"refusing to overwrite a different prepared artifact: {path}")
        return
    with path.open("x") as stream:
        stream.write(payload)


def changes(before: dict, after: dict, prefix: str = "") -> dict:
    out = {}
    for key in sorted(before.keys() | after.keys()):
        path = f"{prefix}.{key}" if prefix else key
        a, b = before.get(key), after.get(key)
        if isinstance(a, dict) and isinstance(b, dict):
            out.update(changes(a, b, path))
        elif a != b:
            out[path] = {"before": a, "after": b}
    return out


def verify(cfg: dict, base: dict) -> dict:
    """Shared tensors bit-identical to the reference; the new branch must start as a no-op."""
    candidate, evidence = P.build_paired_model(C.ExpConfig.from_dict(cfg))
    reference, _ = P.build_paired_model(C.ExpConfig.from_dict(base))
    before, after = reference.state_dict(), candidate.state_dict()
    added = sorted(set(after) - set(before))
    if set(before) - set(after):
        raise ValueError("the radius branch removed existing tensors")
    if any(not key.startswith(RADIUS_BRANCH) for key in added):
        raise ValueError(f"unexpected new tensors outside the radius branch: {added}")
    for key, old in before.items():
        if not torch.equal(old, after[key]):
            raise ValueError(f"shared tensor differs from the reference: {key}")
    zero = [k for k in added if k.startswith("query_patch.out_r") or k.startswith("query_patch.attn_r")]
    if not zero:
        raise ValueError("expected a zero-initialised radius output head")
    for key in zero:
        if torch.count_nonzero(after[key]).item():
            raise ValueError(f"radius branch output is not zero-initialised: {key}")
    return {"passed": True, "seed": SEED, "new_tensors": added, "zero_initialised": zero,
            "shared_tensors_exact": True, "unchanged_tensors": len(before),
            "reference_config": evidence["reference_config"],
            "reference_config_sha256": evidence["reference_config_sha256"],
            "initial_state_sha256": evidence["initial_state_sha256"],
            "trained_checkpoint_loaded": False}


def main(argv=None) -> None:
    global NAME, ARMS
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config-dir", type=Path, default=ROOT / "training_wss_min/configs" / NAME)
    parser.add_argument("--experiment-dir", type=Path, default=ROOT / "training_wss_min/experiments" / NAME)
    parser.add_argument("--experiment-name", default=NAME)
    parser.add_argument("--arms", default="", help="ID:radius_mm:k[,...] overriding the default ladder")
    args = parser.parse_args(argv)
    NAME = args.experiment_name
    if args.arms:
        ARMS = {}
        for spec in args.arms.split(","):
            aid, radius, k = spec.split(":")
            ARMS[aid] = (float(radius), int(k))
    args.config_dir, args.experiment_dir = args.config_dir.resolve(), args.experiment_dir.resolve()
    donor = args.config_dir.parent / BASE_NAME / f"X5D_v51_s{SEED}.json"
    base = json.loads(donor.read_text())
    if base["data"].get("query_patch_radius_mm") or base["model"].get("query_patch_radius_k"):
        raise ValueError("the X5D_v51 donor already carries a radius branch")

    paired, arms = {}, []
    for aid, (radius_mm, radius_k) in ARMS.items():
        cfg = copy.deepcopy(base)
        cfg["name"] = f"{NAME}/{aid}_s{SEED}"
        cfg["notes"] = (
            f"局部形态波 A：query patch 在 K16 之外增加固定 {radius_mm:g} mm 球的 {radius_k} 点覆盖采样"
            f"（零初始化第二分支），其余 X5D_v51 配方与 offset_norm='mm' 逐位不变，seed {SEED}。"
            f"依据 Phase 0：K16 只覆盖约 8.3% 圆周，而区间内残差的 m=1..3 周向模态净份额 0.589。")
        cfg["data"].update(query_patch_radius_mm=radius_mm, query_patch_radius_k=radius_k)
        cfg["model"]["query_patch_radius_k"] = radius_k
        allowed = {"name", "notes", "data.query_patch_radius_mm", "data.query_patch_radius_k",
                   "model.query_patch_radius_k"}
        diff = changes(base, cfg)
        if not set(diff) <= allowed:
            raise ValueError(f"{aid}: something beyond the radius branch changed: {sorted(diff)}")
        C.ExpConfig.from_dict(cfg)
        run = ROOT / "training_wss_min/runs" / cfg["name"]
        if run.exists():
            raise FileExistsError(f"refusing to prepare an existing training run: {run}")
        print(f"verifying zero-initialised radius branch: {aid} ({radius_mm:g} mm, k={radius_k})", flush=True)
        paired[aid] = verify(cfg, base)
        save(args.config_dir / f"{aid}_s{SEED}.json", cfg)
        arms.append({"id": f"{aid}_s{SEED}", "title": f"{aid}: query patch + {radius_mm:g} mm 球 × {radius_k} 点",
                     "config": f"{aid}_s{SEED}.json", "phase": 0, "modules": [f"query_patch_radius_{radius_mm:g}mm"],
                     "seed": SEED, "fold": None, "parent": f"X5D_v51_s{SEED}", "depends_on": [],
                     "evaluate": ["best", "last"], "single_change": True,
                     "radius_mm": radius_mm, "radius_k": radius_k,
                     "config_diff_vs_anchor": diff})

    save(args.experiment_dir / "prepared_audit.json", {
        "status": "passed", "passed": True, "experiment": NAME, "training_started": False, "seed": SEED,
        "donor": str(donor), "feature_set_unchanged": True, "feature_stats_unchanged": True,
        "split": base["data"]["split_path"], "paired_initialization": paired,
        "phase0_evidence": str(ROOT / "training_wss_min/experiments/wss_local_morph_20260917/offline"),
        "arms": {aid: {"radius_mm": r, "radius_k": k} for aid, (r, k) in ARMS.items()}})
    save(args.config_dir / "matrix.json", {
        "schema_version": 1, "experiment": NAME, "anchor": str(donor),
        "anchor_run": str(ROOT / "training_wss_min/runs" / BASE_NAME / f"X5D_v51_s{SEED}"),
        "control_id": None,
        "external_reference_runs": {
            f"X5D_v51_s{SEED}": str(ROOT / f"training_wss_min/runs/{BASE_NAME}/X5D_v51_s{SEED}/eval/ckpt_best/metrics.json"),
            f"X5Ddual_s{SEED}": str(ROOT / f"training_wss_min/runs/{BASE_NAME}/X5Ddual_s{SEED}/eval/ckpt_best/metrics.json")},
        "expected_training_runs": len(ARMS), "expected_evaluations": 2 * len(ARMS), "arms": arms,
        "prepared_audit": str(args.experiment_dir / "prepared_audit.json"),
        "reading_rule": ("感受野阶梯，单 seed 1234，全部与同 seed X5D_v51（0.7461）配对；X5Ddual（4 mm + spacing 归一）"
                         "是已知的近中性参照。单 seed 噪声带 ±0.014 物理 / ±0.004 归一化，本波只用于淘汰与定方向。")})
    print(json.dumps({"status": "prepared", "arms": list(ARMS), "seed": SEED,
                      "config_dir": str(args.config_dir)}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()

"""Prepare the sector-token arms (B2, single seed, config-only on top of W06K32).

Wave A showed that widening the query-patch ball from 8.3 % to 18 % of the local
circumference left the within-bin m=1..3 residual share untouched (+0.5706 -> +0.5625~+0.5728).
The located cause is the ball branch's **masked mean pooling**, which erases the angular
arrangement of the neighbours.  These arms keep the same ball (6 mm x 32 coverage points =
the already trained W06K32 control) and only change the pooling: neighbours are partitioned
by their angle in the query's own (axial, circumferential) surface plane into `sectors`
wedges x `rings` distance rings, each cell becomes one token, the tokens interact once
through a small self-attention, and the result is added through a zero-initialised head.
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
NAME = "wss_local_morph_sectors_20260918"
DONOR_NAME = "wss_local_morph_radius_k_20260917"
DONOR_ARM = "W06K32_s1234"
SEED = 1234
ARMS = {"S4": (4, 1), "S6": (6, 1), "S8": (8, 1), "S6R2": (6, 2)}
SECTOR_PREFIX = "query_patch.sector"


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


def verify(cfg: dict, base: dict, n_tokens: int) -> dict:
    """Shared tensors bit-identical to the no-sector control; the new branch is an exact no-op."""
    candidate, evidence = P.build_paired_model(C.ExpConfig.from_dict(cfg))
    reference, _ = P.build_paired_model(C.ExpConfig.from_dict(base))
    before, after = reference.state_dict(), candidate.state_dict()
    added = sorted(set(after) - set(before))
    if set(before) - set(after):
        raise ValueError("the sector branch removed existing tensors")
    if any(not key.startswith(SECTOR_PREFIX) for key in added):
        raise ValueError(f"unexpected new tensors outside the sector branch: {added}")
    for key, old in before.items():
        if not torch.equal(old, after[key]):
            raise ValueError(f"shared tensor differs from the no-sector control: {key}")
    head = [k for k in added if k.startswith("query_patch.sector_out")]
    if not head:
        raise ValueError("expected a zero-initialised sector output head")
    for key in head:
        if torch.count_nonzero(after[key]).item():
            raise ValueError(f"sector output head is not zero-initialised: {key}")
    if after["query_patch.sector_pos.weight"].shape[0] != n_tokens:
        raise ValueError("sector positional table does not match sectors x rings")
    return {"passed": True, "seed": SEED, "tokens": n_tokens, "new_tensors": added,
            "zero_initialised": head, "shared_tensors_exact": True,
            "unchanged_tensors": len(before),
            "new_parameters": int(sum(after[k].numel() for k in added)),
            "reference_config": evidence["reference_config"],
            "initial_state_sha256": evidence["initial_state_sha256"],
            "trained_checkpoint_loaded": False}


def main(argv=None) -> None:
    global NAME, ARMS
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config-dir", type=Path, default=ROOT / "training_wss_min/configs" / NAME)
    parser.add_argument("--experiment-dir", type=Path, default=ROOT / "training_wss_min/experiments" / NAME)
    parser.add_argument("--experiment-name", default=NAME)
    parser.add_argument("--arms", default="", help="ID:sectors:rings[,...] overriding the default set")
    args = parser.parse_args(argv)
    NAME = args.experiment_name
    if args.arms:
        ARMS = {}
        for spec in args.arms.split(","):
            aid, sec, ring = spec.split(":")
            ARMS[aid] = (int(sec), int(ring))
    args.config_dir, args.experiment_dir = args.config_dir.resolve(), args.experiment_dir.resolve()
    donor = args.config_dir.parent / DONOR_NAME / f"{DONOR_ARM}.json"
    base = json.loads(donor.read_text())
    if int(base["model"].get("query_patch_radius_k", 0)) < 1:
        raise ValueError("the donor must already carry the fixed-mm ball branch")
    if int(base["model"].get("query_patch_sectors", 0)):
        raise ValueError("the donor must be the no-sector control")

    paired, arms = {}, []
    for aid, (sectors, rings) in ARMS.items():
        cfg = copy.deepcopy(base)
        cfg["name"] = f"{NAME}/{aid}_s{SEED}"
        cfg["notes"] = (
            f"局部形态 Wave B（B2 扇区 token）：球不变（{base['data']['query_patch_radius_mm']:g} mm × "
            f"{base['data']['query_patch_radius_k']} 点，= 已训练的 W06K32 对照），只把 masked mean 池化换成"
            f" {sectors} 个周向扇形 × {rings} 个距离环 = {sectors * rings} 个 token，token 间一层自注意力后汇总，"
            f"零初始化接入；seed {SEED}。依据 Wave A：扩大球半径不动周向模态份额，病因是 mean 池化抹平角度。")
        cfg["model"].update(query_patch_sectors=sectors, query_patch_rings=rings, query_patch_sector_heads=4)
        allowed = {"name", "notes", "model.query_patch_sectors", "model.query_patch_rings",
                   "model.query_patch_sector_heads"}
        diff = changes(base, cfg)
        if not set(diff) <= allowed:
            raise ValueError(f"{aid}: something beyond the sector pooling changed: {sorted(diff)}")
        C.ExpConfig.from_dict(cfg)
        run = ROOT / "training_wss_min/runs" / cfg["name"]
        if run.exists():
            raise FileExistsError(f"refusing to prepare an existing training run: {run}")
        print(f"verifying zero-initialised sector branch: {aid} ({sectors} sectors x {rings} rings)", flush=True)
        paired[aid] = verify(cfg, base, sectors * rings)
        save(args.config_dir / f"{aid}_s{SEED}.json", cfg)
        arms.append({"id": f"{aid}_s{SEED}", "title": f"{aid}: {sectors} 扇形 × {rings} 环 = {sectors * rings} token",
                     "config": f"{aid}_s{SEED}.json", "phase": 0, "modules": [f"query_patch_sectors_{sectors}x{rings}"],
                     "seed": SEED, "fold": None, "parent": DONOR_ARM, "depends_on": [],
                     "evaluate": ["best", "last"], "single_change": True,
                     "sectors": sectors, "rings": rings,
                     "new_parameters": paired[aid]["new_parameters"],
                     "config_diff_vs_anchor": diff})

    save(args.experiment_dir / "prepared_audit.json", {
        "status": "passed", "passed": True, "experiment": NAME, "training_started": False, "seed": SEED,
        "donor": str(donor), "control_run": str(ROOT / "training_wss_min/runs" / DONOR_NAME / DONOR_ARM),
        "feature_set_unchanged": True, "feature_stats_unchanged": True, "ball_unchanged": True,
        "split": base["data"]["split_path"], "paired_initialization": paired,
        "wave_a_evidence": str(ROOT / "training_wss_min/experiments/wss_local_morph_radius_20260917/offline"),
        "arms": {aid: {"sectors": s, "rings": r, "tokens": s * r} for aid, (s, r) in ARMS.items()}})
    save(args.config_dir / "matrix.json", {
        "schema_version": 1, "experiment": NAME, "anchor": str(donor),
        "anchor_run": str(ROOT / "training_wss_min/runs" / DONOR_NAME / DONOR_ARM),
        "control_id": None,
        "external_reference_runs": {
            "W06K32_s1234": str(ROOT / f"training_wss_min/runs/{DONOR_NAME}/{DONOR_ARM}/eval/ckpt_best/metrics.json"),
            "X5D_v51_s1234": str(ROOT / "training_wss_min/runs/wss_v51_wave1_20260916/X5D_v51_s1234/eval/ckpt_best/metrics.json")},
        "expected_training_runs": len(ARMS), "expected_evaluations": 2 * len(ARMS), "arms": arms,
        "prepared_audit": str(args.experiment_dir / "prepared_audit.json"),
        "reading_rule": ("扇区 token，单 seed 1234。**主判据是机制列**：区间内残差 m=1..3 的净份额能否从底座的 +0.5706 / "
                         "W06K32 的 +0.5691 被压下去；Pa R²_cb 的单 seed 噪声带 ±0.014，不用于排名。"
                         "匹配对照 = 同球无扇区的 W06K32（0.7560 / 净份额 +0.5691）。")})
    print(json.dumps({"status": "prepared", "arms": list(ARMS), "seed": SEED,
                      "config_dir": str(args.config_dir)}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()

"""Prepare the wss_cycle stage-3 confirmation matrix (train136, seeds 1234/7/2025, test34 read once): A1 / O1 / O2.

Matrix: docs/02-推进与变更/03-周期量TAWSS_OSI/WSS_V5_周期积分量TAWSS_OSI直接回归实验矩阵_2026-09-20.md §6 (stage 3), opened by the user on
2026-09-21 ("GPU 空闲就做实验确定想法") although no arm passed every stage-1 gate: the question is whether the direct TAWSS / OSI
regressors are deployable outputs at peak-frame quality on the locked test34, in the same form as the deployed X5D_v51 ensemble.

Base = deployed peak-frame recipes configs/wss_v51_wave1_20260916/X5D_v51_s{seed}.json (train136 → test34; paired initialisation
against the same-seed C1 reference, i.e. identical initial weights to the deployed peak model of that seed).
Only the label changes: target = tawss (log_z, train136 statistics) / osi (linear | logit_z), labels from wss_min_cycle_v1.

    python -u -m training_wss_min.tools.prepare_wss_cycle_stage3
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

from training_wss_min import config as C
from training_wss_min.tools import prepare_wss_local_wave1 as W1

ROOT = C.PROJECT_ROOT
NAME = "wss_cycle_stage3_20260921"
BASE_NAME = "wss_v51_wave1_20260916"
STAGE1 = "wss_cycle_20260920"
CONFIGS = ROOT / "training_wss_min/configs" / NAME
EXP = ROOT / "training_wss_min/experiments" / NAME
BASE_CONFIGS = ROOT / "training_wss_min/configs" / BASE_NAME
CYC = ROOT / "data_wss_v5/views_v5_1/wss_min_cycle_v1"
SEEDS = (1234, 7, 2025)
KINDS = {"A1": ("tawss", "tawss"), "O1": ("osi", "osi_linear"), "O2": ("osi", "osi_logit")}
ARMS = {f"{kind}_s{seed}": (kind, seed) for kind in KINDS for seed in SEEDS}


def base_path(seed: int) -> Path:
    return BASE_CONFIGS / f"X5D_v51_s{seed}.json"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def configure(aid: str, kind: str, seed: int) -> dict:
    base = json.loads(base_path(seed).read_text())
    cfg = json.loads(json.dumps(base))
    target, stats_tag = KINDS[kind]
    cfg["name"] = f"{NAME}/{aid}"
    cfg["data"]["target"] = target
    cfg["data"]["cycle_view_root"] = str(CYC)
    cfg["data"]["wss_stats_path"] = str(CYC / "stats" / f"cycle_stats_{stats_tag}_train136.json")
    cfg["eval"]["threshold_masks_above"] = [0.1, 0.3] if target == "osi" else []
    cfg["eval"]["threshold_masks_below"] = [0.4] if target == "tawss" else []
    space = {"tawss": "ln(max(TAWSS, 0.05 Pa)+eps) z-scored (log_z, train136)",
             "osi_linear": "OSI z-scored linearly (train136), evaluation clipped to [0, 0.5]",
             "osi_logit": "logit(clip(2·OSI, 1e-3, 1−1e-3)) z-scored (train136), inverse = 0.5·sigmoid"}[stats_tag]
    cfg["notes"] = (f"cycle stage-3 {aid}: deployed X5D_v51 seed-{seed} recipe (train136) with target={target} from wss_min_cycle_v1 (frames 0-79); "
                    f"target space {space}; inputs/sampling/density augmentation/losses/budget/paired initialisation unchanged; test34 read once "
                    f"(locked confirmation, no selection on it); cycle_agreement block on.")
    classes = {"data": C.DataConfig, "model": C.ModelConfig, "train": C.TrainConfig, "eval": C.EvalConfig}
    for section, cls in classes.items():
        unknown = set(cfg[section]) - set(cls.__dataclass_fields__)
        if unknown:
            raise ValueError(f"unknown {section} fields: {unknown}")
    parsed = C.ExpConfig.from_dict(cfg)
    C.validate_features(parsed)
    for key in ("wss_stats_path", "feature_stats_path", "split_path"):
        if not Path(cfg["data"][key]).is_file():
            raise FileNotFoundError(f"{aid}: {key} missing: {cfg['data'][key]}")
    if cfg["train"].get("init_reference_config") and not Path(cfg["train"]["init_reference_config"]).is_file():
        raise FileNotFoundError(f"{aid}: init_reference_config missing")
    stats = json.loads(Path(cfg["data"]["wss_stats_path"]).read_text())
    if stats.get("target") != target or stats.get("scope") != "train136":
        raise ValueError(f"{aid}: statistics file target/scope mismatch: {stats.get('target')} {stats.get('scope')}")
    return cfg


def main() -> None:
    manifest = json.loads((CYC / "cycle_manifest.json").read_text())
    if manifest["n_cases"] != 170 or manifest["n_failed"]:
        raise RuntimeError("C0 view incomplete")
    arms = []
    for aid, (kind, seed) in ARMS.items():
        cfg = configure(aid, kind, seed)
        W1.save(CONFIGS / f"{aid}.json", cfg)
        base = json.loads(base_path(seed).read_text())
        target, stats_tag = KINDS[kind]
        arms.append(dict(id=aid, title=cfg["notes"].split(":")[1].split(";")[0].strip(), config=f"{aid}.json", phase=3,
                         modules=["F6", "density_aug", f"target_{target}", f"stats_{stats_tag}", "threshold_masks", "cycle_agreement"],
                         seed=seed, parent=f"X5D_v51_s{seed} (deployed peak-frame model, same seed)", depends_on=[], evaluate=["best", "last"],
                         single_change=kind in ("A1", "O1"), config_diff_vs_anchor=W1.declared_diff(base, cfg)))
    stage1 = ROOT / "training_wss_min/experiments" / STAGE1 / "offline"
    c1 = json.loads((stage1 / "c1_baselines.json").read_text())
    gate = json.loads((stage1 / "gate_report_best.json").read_text())
    W1.save(CONFIGS / "matrix.json", dict(
        schema_version=1, experiment=NAME, anchor=str(W1.ANCHOR), anchor_run=str(W1.ANCHOR_RUN), control_id=None,
        external_reference_runs={f"X5D_v51_s{s} (deployed peak-frame, same seed)": str(ROOT / "training_wss_min/runs" / BASE_NAME / f"X5D_v51_s{s}/eval/ckpt_best/metrics.json") for s in SEEDS},
        cycle_view={"root": str(CYC), "manifest_sha256": sha256(CYC / "cycle_manifest.json"), "n_cases": manifest["n_cases"], "definition": manifest["definition"]},
        stage1_gates={k: v for k, v in gate.get("gates", {}).items()}, test34_free_baselines=c1.get("test34"),
        expected_training_runs=len(ARMS), expected_evaluations=2 * len(ARMS), arms=arms,
        reading_rule="train136 → test34 read once (locked); per-seed rows plus the three-seed Pa-mean ensemble per target, compared with the "
                     "same-seed deployed X5D_v51 peak model (normalized R2_cb) and the C1 test34 free baselines (TAWSS-null / ratio / OSI-null); "
                     "no selection on test34; stage-1 gate verdicts are carried unchanged"))
    print(CONFIGS / "matrix.json", len(arms), "arms")


if __name__ == "__main__":
    main()

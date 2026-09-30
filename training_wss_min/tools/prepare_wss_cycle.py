"""Prepare the wss_cycle_20260920 stage-1 screening matrix (cv3_v51 folds, seed 1234): A1 / O1 / O2.

Matrix: docs/02-推进与变更/03-周期量TAWSS_OSI/WSS_V5_周期积分量TAWSS_OSI直接回归实验矩阵_2026-09-20.md §4.
Base = wave-2a fold recipes (X5D_v51 on cv3_v51 fold k; train = other two folds, held-out fold = 'test'; test34 unused).
Labels = wss_min_cycle_v1 (C0, wss_v5.views.wall_cycle_v1): TAWSS / OSI from the 81-frame wall shear vector, frames 0..79.

  A1_f{k}  target='tawss'  log_z statistics (cycle_stats_tawss_fold{k})       — single change vs the fold base: the label
  O1_f{k}  target='osi'    linear statistics (cycle_stats_osi_linear_fold{k}) — label + bounded linear target space
  O2_f{k}  target='osi'    logit_z statistics (cycle_stats_osi_logit_fold{k}) — only the target space differs from O1

All 27 inputs, sampling, density augmentation, losses and budget are the fold base's; paired initialisation against the fold
base (identical architecture → identical initial weights). Evaluation adds physical threshold masks (TAWSS < 0.4 Pa;
OSI > 0.1 / > 0.3) and saves same-point predictions for the offline stagnation-region analysis (A1 × O*).

    python -u -m training_wss_min.tools.prepare_wss_cycle
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

from training_wss_min import config as C
from training_wss_min.tools import prepare_wss_local_wave1 as W1

ROOT = C.PROJECT_ROOT
NAME = "wss_cycle_20260920"
BASE_NAME = "wss_v51_wave2a_20260916"
CONFIGS = ROOT / "training_wss_min/configs" / NAME
EXP = ROOT / "training_wss_min/experiments" / NAME
OFF = EXP / "offline"
BASE_CONFIGS = ROOT / "training_wss_min/configs" / BASE_NAME
CYC = ROOT / "data_wss_v5/views_v5_1/wss_min_cycle_v1"
SEED = 1234
KINDS = {"A1": ("tawss", "tawss"), "O1": ("osi", "osi_linear"), "O2": ("osi", "osi_logit")}
ARMS = {f"{kind}_f{k}_s{SEED}": (kind, k) for kind in KINDS for k in range(3)}


def base_path(fold: int) -> Path:
    return BASE_CONFIGS / f"X5D_v51_f{fold}_s{SEED}.json"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def configure(aid: str, kind: str, fold: int) -> dict:
    base = json.loads(base_path(fold).read_text())
    cfg = json.loads(json.dumps(base))
    target, stats_tag = KINDS[kind]
    cfg["name"] = f"{NAME}/{aid}"
    cfg["data"]["target"] = target
    cfg["data"]["cycle_view_root"] = str(CYC)
    cfg["data"]["wss_stats_path"] = str(CYC / "stats" / f"cycle_stats_{stats_tag}_fold{fold}.json")
    cfg["eval"]["threshold_masks_above"] = [0.1, 0.3] if target == "osi" else []
    cfg["eval"]["threshold_masks_below"] = [0.4] if target == "tawss" else []
    cfg["train"]["init_reference_config"] = str(base_path(fold))   # 同 seed 折底座：架构相同 → 初始权重逐位相同
    space = {"tawss": "ln(max(TAWSS, 0.05 Pa)+eps) z-scored (log_z, train fold)",
             "osi_linear": "OSI z-scored linearly (train fold), evaluation clipped to [0, 0.5]",
             "osi_logit": "logit(clip(2·OSI, 1e-3, 1−1e-3)) z-scored (train fold), inverse = 0.5·sigmoid"}[stats_tag]
    cfg["notes"] = (f"cycle stage-1 {aid}: X5D_v51 fold-{fold} recipe with target={target} from wss_min_cycle_v1 (frames 0-79); "
                    f"target space {space}; inputs/sampling/density augmentation/losses/budget unchanged; paired initialisation "
                    f"against the fold base; held-out fold read once; test34 unused.")
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
    stats = json.loads(Path(cfg["data"]["wss_stats_path"]).read_text())
    if stats.get("target") != target or stats.get("scope") != f"fold{fold}":
        raise ValueError(f"{aid}: statistics file target/scope mismatch: {stats.get('target')} {stats.get('scope')}")
    if not (CYC / "cycle_manifest.json").is_file():
        raise FileNotFoundError("C0 cycle view manifest missing")
    return cfg


def main() -> None:
    manifest = json.loads((CYC / "cycle_manifest.json").read_text())
    if manifest["n_cases"] != 170 or manifest["n_failed"]:
        raise RuntimeError(f"C0 view incomplete: {manifest['n_cases']} cases, {manifest['n_failed']} failed")
    arms = []
    for aid, (kind, fold) in ARMS.items():
        cfg = configure(aid, kind, fold)
        W1.save(CONFIGS / f"{aid}.json", cfg)
        base = json.loads(base_path(fold).read_text())
        target, stats_tag = KINDS[kind]
        arms.append(dict(id=aid, title=cfg["notes"].split(":")[1].split(";")[0].strip(), config=f"{aid}.json", phase=1,
                         modules=["F6", "density_aug", f"target_{target}", f"stats_{stats_tag}", "threshold_masks"],
                         seed=SEED, fold=fold, parent=f"X5D_v51_f{fold}_s{SEED} (wave 2a)", depends_on=[], evaluate=["best", "last"],
                         single_change=kind in ("A1", "O1"), config_diff_vs_anchor=W1.declared_diff(base, cfg)))
    cv3 = json.loads((Path(json.loads(base_path(0).read_text())["data"]["split_path"]).parent / "cv3_summary.json").read_text())
    c1 = json.loads((OFF / "c1_baselines.json").read_text()) if (OFF / "c1_baselines.json").is_file() else None
    W1.save(CONFIGS / "matrix.json", dict(
        schema_version=1, experiment=NAME, anchor=str(W1.ANCHOR), anchor_run=str(W1.ANCHOR_RUN), control_id=None,
        external_reference_runs={f"X5D_v51_f{k}_s{SEED} (wave 2a, peak-frame base)": str(ROOT / "training_wss_min/runs" / BASE_NAME / f"X5D_v51_f{k}_s{SEED}/eval/ckpt_best/metrics.json") for k in range(3)},
        cv3=cv3, cycle_view={"root": str(CYC), "manifest_sha256": sha256(CYC / "cycle_manifest.json"), "n_cases": manifest["n_cases"],
                             "period_wrap_flagged": len(manifest["period_wrap_flagged"]), "definition": manifest["definition"]},
        stage0_c1_baselines=(c1["fold_mean"] | {"gates": c1["gates"]}) if c1 else "c1_baselines.json not yet available",
        expected_training_runs=len(ARMS), expected_evaluations=2 * len(ARMS), arms=arms,
        reading_rule="fold arms are read only on their held-out fold; gates G1.1-G3 (matrix §4) use the three-fold mean and fold sign agreement; "
                     "G1.1 compares normalized R2_cb with the same-fold X5D_v51 peak-frame base; G1.2/G2.2/G3 compare with the C1 free baselines; "
                     "test34 is read once only after the winner is locked"))
    print(CONFIGS / "matrix.json", len(arms), "arms")


if __name__ == "__main__":
    main()

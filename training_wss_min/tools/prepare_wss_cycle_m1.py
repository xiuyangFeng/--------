"""Prepare the wss_cycle stage-2 M1 three-head matrix (cv3_v51 folds, seed 1234).

Matrix: docs/02-推进与变更/03-周期量TAWSS_OSI/WSS_V5_周期积分量TAWSS_OSI直接回归实验矩阵_2026-09-20.md §5 (M1), opened on 2026-09-21 under the
user's "use the idle GPU to confirm the ideas" mandate although G1.2 / G2.2 did not pass at stage 1.

  M1_f{k}  target='wss_cycle_multi' (out_dim=3 = [peak-frame WSS log_z, TAWSS log_z, OSI logit_z]); equal-weight channel MSE;
           q90 pinball off (undefined on the channel vector — same known difference as the time-basis arms); everything else is the
           fold base X5D_v51_f{k} recipe; paired initialisation (backbone/decoder shared, output layers rebuilt with 3 channels).

Gates (matrix §5): peak-frame normalized R2_cb drop vs the same-fold X5D_v51 base <= 0.02 (3/3); TAWSS / OSI heads not worse than
the single-target arms A1 / O2 by more than 0.01 (normalized R2_cb, fold-wise).

    python -u -m training_wss_min.tools.prepare_wss_cycle_m1
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

from training_wss_min import config as C
from training_wss_min.tools import prepare_wss_local_wave1 as W1

ROOT = C.PROJECT_ROOT
NAME = "wss_cycle_m1_20260921"
BASE_NAME = "wss_v51_wave2a_20260916"
STAGE1 = "wss_cycle_20260920"
CONFIGS = ROOT / "training_wss_min/configs" / NAME
EXP = ROOT / "training_wss_min/experiments" / NAME
BASE_CONFIGS = ROOT / "training_wss_min/configs" / BASE_NAME
CYC = ROOT / "data_wss_v5/views_v5_1/wss_min_cycle_v1"
SEED = 1234
ARMS = {f"M1_f{k}_s{SEED}": k for k in range(3)}


def base_path(fold: int) -> Path:
    return BASE_CONFIGS / f"X5D_v51_f{fold}_s{SEED}.json"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def multi_stats(fold: int, base: dict) -> Path:
    """Combined method='multi' statistics: channel 0 = the fold base's peak-frame log_z file, 1 = TAWSS log_z, 2 = OSI logit_z (all train fold)."""
    wss = json.loads(Path(base["data"]["wss_stats_path"]).read_text())
    tawss = json.loads((CYC / "stats" / f"cycle_stats_tawss_fold{fold}.json").read_text())
    osi = json.loads((CYC / "stats" / f"cycle_stats_osi_logit_fold{fold}.json").read_text())
    for name, st, expect in (("wss", wss, "log_z"), ("tawss", tawss, "log_z"), ("osi", osi, "logit_z")):
        if st.get("method") != expect:
            raise ValueError(f"{name} statistics must be {expect}")
    if tawss.get("scope") != f"fold{fold}" or osi.get("scope") != f"fold{fold}":
        raise ValueError("cycle statistics scope mismatch")
    out = {"schema_version": 1, "method": "multi", "channels": list(C.MULTI_CHANNELS), "eps": wss["eps"], "floor": tawss["floor"],
           "scope": f"fold{fold}", "statistics_scope": f"train fold{fold} only (peak-frame log_z from {Path(base['data']['wss_stats_path']).name}; cycle files from wss_min_cycle_v1)",
           "sources": {"wss": base["data"]["wss_stats_path"], "tawss": str(CYC / "stats" / f"cycle_stats_tawss_fold{fold}.json"),
                       "osi": str(CYC / "stats" / f"cycle_stats_osi_logit_fold{fold}.json")},
           "wss": wss, "tawss": tawss, "osi": osi}
    path = EXP / "stats" / f"multi_stats_fold{fold}.json"
    W1.save(path, out)
    return path


def configure(aid: str, fold: int) -> dict:
    base = json.loads(base_path(fold).read_text())
    cfg = json.loads(json.dumps(base))
    cfg["name"] = f"{NAME}/{aid}"
    cfg["data"]["target"] = C.MULTI_TARGET
    cfg["data"]["cycle_view_root"] = str(CYC)
    cfg["data"]["wss_stats_path"] = str(multi_stats(fold, base))
    cfg["model"]["out_dim"] = len(C.MULTI_CHANNELS)
    cfg["train"]["loss_pinball_lambda"] = 0.0
    cfg["eval"]["threshold_masks_above"] = [0.1, 0.3]
    cfg["eval"]["threshold_masks_below"] = [0.4]
    cfg["train"]["init_reference_config"] = str(base_path(fold))
    cfg["notes"] = (f"cycle stage-2 {aid}: X5D_v51 fold-{fold} recipe with a three-channel head (out_dim 3 = peak-frame WSS log_z, TAWSS log_z, "
                    f"OSI logit_z; per-channel train-fold statistics in a method='multi' file); equal-weight channel MSE; q90 pinball off "
                    f"(undefined on channel vectors); inputs/sampling/density augmentation/budget unchanged; backbone/decoder paired with the "
                    f"fold base, output layers rebuilt with 3 channels; held-out fold read once; test34 unused.")
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
    return cfg


def main() -> None:
    arms = []
    for aid, fold in ARMS.items():
        cfg = configure(aid, fold)
        W1.save(CONFIGS / f"{aid}.json", cfg)
        base = json.loads(base_path(fold).read_text())
        arms.append(dict(id=aid, title=cfg["notes"].split(":")[1].split(";")[0].strip(), config=f"{aid}.json", phase=2,
                         modules=["F6", "density_aug", "multi_head_3", "pinball_off", "threshold_masks", "cycle_agreement"],
                         seed=SEED, fold=fold, parent=f"X5D_v51_f{fold}_s{SEED} (wave 2a)", depends_on=[], evaluate=["best", "last"],
                         single_change=False, config_diff_vs_anchor=W1.declared_diff(base, cfg)))
    stage1_runs = ROOT / "training_wss_min/runs" / STAGE1
    W1.save(CONFIGS / "matrix.json", dict(
        schema_version=1, experiment=NAME, anchor=str(W1.ANCHOR), anchor_run=str(W1.ANCHOR_RUN), control_id=None,
        external_reference_runs={
            **{f"X5D_v51_f{k}_s{SEED} (wave 2a, peak-frame base)": str(ROOT / "training_wss_min/runs" / BASE_NAME / f"X5D_v51_f{k}_s{SEED}/eval/ckpt_best/metrics.json") for k in range(3)},
            **{f"A1_f{k}_s{SEED} (stage 1 TAWSS single head)": str(stage1_runs / f"A1_f{k}_s{SEED}/eval/ckpt_best/metrics.json") for k in range(3)},
            **{f"O2_f{k}_s{SEED} (stage 1 OSI logit single head)": str(stage1_runs / f"O2_f{k}_s{SEED}/eval/ckpt_best/metrics.json") for k in range(3)}},
        cycle_view={"root": str(CYC), "manifest_sha256": sha256(CYC / "cycle_manifest.json")},
        expected_training_runs=len(ARMS), expected_evaluations=2 * len(ARMS), arms=arms,
        reading_rule="fold arms read only on their held-out fold; gate = peak-frame normalized R2_cb drop vs same-fold X5D_v51 <= 0.02 (3/3) and "
                     "heads within 0.01 of A1 / O2 (normalized R2_cb); test34 unused"))
    print(CONFIGS / "matrix.json", len(arms), "arms")


if __name__ == "__main__":
    main()

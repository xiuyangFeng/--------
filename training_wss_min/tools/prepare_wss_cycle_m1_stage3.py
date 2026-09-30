"""Prepare the M1 three-head confirmation matrix (train136, seeds 1234/7/2025, test34 read once).

Matrix: docs/02-推进与变更/03-周期量TAWSS_OSI/WSS_V5_周期积分量TAWSS_OSI直接回归实验矩阵_2026-09-20.md §14 (M1 passed all fold gates on
2026-09-22); the user then asked for the three-seed deployment-form confirmation ("用M1跑一次三seed").

  M1_s{seed}  target='wss_cycle_multi' (out_dim 3 = peak-frame WSS log_z, TAWSS log_z, OSI logit_z; equal-weight channel MSE;
              q90 pinball off) on the deployed X5D_v51_s{seed} recipe (train136 → test34); statistics = method='multi' file built
              from the train136 peak log_z file + train136 cycle files; paired initialisation against the same-seed C1 reference
              (backbone/decoder identical to the deployed peak model of that seed; output layers rebuilt with 3 channels).

Reading: per-seed rows + three-seed Pa-mean ensemble per channel, compared with the deployed peak ensemble (channel 0), the
stage-3 single-head ensembles A1_ens3 / O2_ens3 (channels 1/2) and the C1 test34 free baselines. No selection on test34.

    python -u -m training_wss_min.tools.prepare_wss_cycle_m1_stage3
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

from training_wss_min import config as C
from training_wss_min.tools import prepare_wss_local_wave1 as W1

ROOT = C.PROJECT_ROOT
NAME = "wss_cycle_m1_stage3_20260922"
BASE_NAME = "wss_v51_wave1_20260916"
CONFIGS = ROOT / "training_wss_min/configs" / NAME
EXP = ROOT / "training_wss_min/experiments" / NAME
BASE_CONFIGS = ROOT / "training_wss_min/configs" / BASE_NAME
CYC = ROOT / "data_wss_v5/views_v5_1/wss_min_cycle_v1"
SEEDS = (1234, 7, 2025)
ARMS = {f"M1_s{seed}": seed for seed in SEEDS}


def base_path(seed: int) -> Path:
    return BASE_CONFIGS / f"X5D_v51_s{seed}.json"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def multi_stats(base: dict) -> Path:
    """method='multi' statistics for train136: channel 0 = deployed peak log_z file, 1 = TAWSS log_z, 2 = OSI logit_z."""
    wss = json.loads(Path(base["data"]["wss_stats_path"]).read_text())
    tawss = json.loads((CYC / "stats/cycle_stats_tawss_train136.json").read_text())
    osi = json.loads((CYC / "stats/cycle_stats_osi_logit_train136.json").read_text())
    for name, st, expect in (("wss", wss, "log_z"), ("tawss", tawss, "log_z"), ("osi", osi, "logit_z")):
        if st.get("method") != expect:
            raise ValueError(f"{name} statistics must be {expect}")
    if tawss.get("scope") != "train136" or osi.get("scope") != "train136" or "train136" not in wss.get("statistics_scope", "train136"):
        raise ValueError("statistics scope must be train136")
    out = {"schema_version": 1, "method": "multi", "channels": list(C.MULTI_CHANNELS), "eps": wss["eps"], "floor": tawss["floor"],
           "scope": "train136", "statistics_scope": "train136 only (peak-frame log_z from the deployed X5D_v51 statistics file; cycle files from wss_min_cycle_v1)",
           "sources": {"wss": base["data"]["wss_stats_path"], "tawss": str(CYC / "stats/cycle_stats_tawss_train136.json"),
                       "osi": str(CYC / "stats/cycle_stats_osi_logit_train136.json")},
           "wss": wss, "tawss": tawss, "osi": osi}
    path = EXP / "stats" / "multi_stats_train136.json"
    W1.save(path, out)
    return path


def configure(aid: str, seed: int) -> dict:
    base = json.loads(base_path(seed).read_text())
    cfg = json.loads(json.dumps(base))
    cfg["name"] = f"{NAME}/{aid}"
    cfg["data"]["target"] = C.MULTI_TARGET
    cfg["data"]["cycle_view_root"] = str(CYC)
    cfg["data"]["wss_stats_path"] = str(multi_stats(base))
    cfg["model"]["out_dim"] = len(C.MULTI_CHANNELS)
    cfg["train"]["loss_pinball_lambda"] = 0.0
    cfg["eval"]["threshold_masks_above"] = [0.1, 0.3]
    cfg["eval"]["threshold_masks_below"] = [0.4]
    cfg["notes"] = (f"M1 three-head confirmation {aid}: deployed X5D_v51 seed-{seed} recipe (train136) with out_dim 3 = [peak WSS log_z, "
                    f"TAWSS log_z, OSI logit_z] (train136 multi statistics); equal-weight channel MSE; q90 pinball off; inputs/sampling/"
                    f"density augmentation/budget unchanged; paired initialisation against the same-seed C1 reference (output layers rebuilt "
                    f"with 3 channels); test34 read once (locked confirmation, no selection).")
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
    return cfg


def main() -> None:
    arms = []
    for aid, seed in ARMS.items():
        cfg = configure(aid, seed)
        W1.save(CONFIGS / f"{aid}.json", cfg)
        base = json.loads(base_path(seed).read_text())
        arms.append(dict(id=aid, title=cfg["notes"].split(":")[1].split(";")[0].strip(), config=f"{aid}.json", phase=3,
                         modules=["F6", "density_aug", "multi_head_3", "pinball_off", "threshold_masks", "cycle_agreement"],
                         seed=seed, parent=f"X5D_v51_s{seed} (deployed peak-frame model, same seed)", depends_on=[], evaluate=["best", "last"],
                         single_change=False, config_diff_vs_anchor=W1.declared_diff(base, cfg)))
    s3 = ROOT / "training_wss_min/experiments/wss_cycle_stage3_20260921/offline"
    W1.save(CONFIGS / "matrix.json", dict(
        schema_version=1, experiment=NAME, anchor=str(W1.ANCHOR), anchor_run=str(W1.ANCHOR_RUN), control_id=None,
        external_reference_runs={
            **{f"X5D_v51_s{s} (deployed peak-frame, same seed)": str(ROOT / "training_wss_min/runs" / BASE_NAME / f"X5D_v51_s{s}/eval/ckpt_best/metrics.json") for s in SEEDS},
            **{f"A1_s{s} (stage 3 TAWSS single head)": str(ROOT / "training_wss_min/runs/wss_cycle_stage3_20260921" / f"A1_s{s}/eval/ckpt_best/metrics.json") for s in SEEDS},
            **{f"O2_s{s} (stage 3 OSI logit single head)": str(ROOT / "training_wss_min/runs/wss_cycle_stage3_20260921" / f"O2_s{s}/eval/ckpt_best/metrics.json") for s in SEEDS}},
        cycle_view={"root": str(CYC), "manifest_sha256": sha256(CYC / "cycle_manifest.json")},
        stage3_single_head_report=str(s3 / "stage3_report_best.json"), peak_ensemble_reference=str(s3 / "peak_ensemble_reference.json"),
        expected_training_runs=len(ARMS), expected_evaluations=2 * len(ARMS), arms=arms,
        reading_rule="train136 → test34 read once; per-seed rows and the three-seed Pa-mean ensemble per channel, compared with the deployed "
                     "peak three-seed ensemble (channel 0), A1_ens3 / O2_ens3 (channels 1 / 2) and the C1 test34 free baselines; no selection on test34"))
    print(CONFIGS / "matrix.json", len(arms), "arms")


if __name__ == "__main__":
    main()

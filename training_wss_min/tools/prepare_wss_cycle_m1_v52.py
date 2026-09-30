"""Prepare M1 three-head on v5.2 (261 units) with the X5Dcap recipe, CV5 × seeds 1234 / 7 / 2025 (2026-09-25, "task 1").

Origin: OSI tail matrix (tracking §34.12) closed the training-side loss / hand-crafted-feature route; data is the strongest known lever
(peak WSS 136 → 261 units: cv3 OOF 0.710 → CV5 OOF 0.754 / 0.761).  User decision 2026-09-24: v5.2 base recipe = X5Dcap.

Steps (idempotent, refuse to change files already written):
  1. cycle view for v5.2: ``wss_v5.views.wall_cycle_v1.build_all`` on ``views_v5_2_full_20260923/wss_min_view_v1`` →
     ``views_v5_2_full_20260923/wss_min_cycle_v1`` (same frozen definitions; builder unchanged, only the paths differ);
  2. cycle statistics per CV5 fold (train partition only) + IND train170;
  3. method='multi' statistics per fold: channel 0 = the fold's X5Dcap peak-frame log_z file, 1 = TAWSS log_z, 2 = OSI logit_z;
  4. configs M1cap_v52cv_f{k}_s{seed} = X5Dcap_v52cv_f{k}_s{seed} with target='wss_cycle_multi', out_dim 3, q90 pinball off
     (undefined on channel vectors, as stage-2 M1), threshold masks; paired init reference = the same X5Dcap fold/seed config.

    python -u -m training_wss_min.tools.prepare_wss_cycle_m1_v52 [--workers 16]
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from training_wss_min import config as C
from training_wss_min.tools import prepare_wss_local_wave1 as W1
from wss_v5.views import wall_cycle_v1 as CY

ROOT = C.PROJECT_ROOT
NAME = "wss_cycle_m1_v52_20260925"
BASE_NAME = "wss_v52_20260923_cv5"
CONFIGS = ROOT / "training_wss_min/configs" / NAME
EXP = ROOT / "training_wss_min/experiments" / NAME
BASE_CONFIGS = ROOT / "training_wss_min/configs" / BASE_NAME
V52 = ROOT / "data_wss_v5/views_v5_2_full_20260923"
VIEW = V52 / "wss_min_view_v1"
CYC = V52 / "wss_min_cycle_v1"
SEEDS = (1234, 7, 2025)
FOLDS = range(5)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def build_cycle_view(workers: int) -> dict:
    manifest_path = CYC / "cycle_manifest.json"
    if manifest_path.is_file():
        manifest = json.loads(manifest_path.read_text())
    else:
        manifest = CY.build_all(CYC, VIEW, workers, None)
    if manifest["n_failed"] or manifest["n_cases"] != 261:
        raise RuntimeError(f"v5.2 cycle view: {manifest['n_cases']} cases, {manifest['n_failed']} failed")
    scopes = {f"cv5_fold{k}": VIEW / "cv5_v52" / f"fold{k}.json" for k in FOLDS}
    scopes["ind_train170"] = VIEW / "split_v52_ind_train170_test91.json"
    for scope, split in scopes.items():
        if not (CYC / "stats" / f"cycle_stats_osi_logit_{scope}.json").is_file():
            CY.build_stats(CYC, scope, split)
    return manifest


def multi_stats(fold: int, base: dict) -> Path:
    path = EXP / "stats" / f"multi_stats_cv5_fold{fold}.json"
    wss = json.loads(Path(base["data"]["wss_stats_path"]).read_text())
    tawss = json.loads((CYC / "stats" / f"cycle_stats_tawss_cv5_fold{fold}.json").read_text())
    osi = json.loads((CYC / "stats" / f"cycle_stats_osi_logit_cv5_fold{fold}.json").read_text())
    for name, st, expect in (("wss", wss, "log_z"), ("tawss", tawss, "log_z"), ("osi", osi, "logit_z")):
        if st.get("method") != expect:
            raise ValueError(f"{name} statistics must be {expect}")
    if tawss.get("split_sha256") != sha256(Path(base["data"]["split_path"])):
        raise ValueError(f"fold{fold}: cycle statistics were not built on the base split")
    out = {"schema_version": 1, "method": "multi", "channels": list(C.MULTI_CHANNELS), "eps": wss["eps"], "floor": tawss["floor"],
           "scope": f"cv5_fold{fold}", "statistics_scope": f"v5.2 CV5 fold{fold} train partition only (peak-frame log_z = X5Dcap fold file; cycle = wss_min_cycle_v1 v5.2)",
           "sources": {"wss": base["data"]["wss_stats_path"], "tawss": str(CYC / "stats" / f"cycle_stats_tawss_cv5_fold{fold}.json"),
                       "osi": str(CYC / "stats" / f"cycle_stats_osi_logit_cv5_fold{fold}.json")},
           "wss": wss, "tawss": tawss, "osi": osi}
    W1.save(path, out)
    return path


def configure(fold: int, seed: int) -> tuple[str, dict, dict]:
    base_path = BASE_CONFIGS / f"X5Dcap_v52cv_f{fold}_s{seed}.json"
    base = json.loads(base_path.read_text())
    cfg = json.loads(json.dumps(base))
    aid = f"M1cap_v52cv_f{fold}_s{seed}"
    cfg["name"] = f"{NAME}/{aid}"
    cfg["data"]["target"] = C.MULTI_TARGET
    cfg["data"]["cycle_view_root"] = str(CYC)
    cfg["data"]["wss_stats_path"] = str(multi_stats(fold, base))
    cfg["model"]["out_dim"] = len(C.MULTI_CHANNELS)
    cfg["train"]["loss_pinball_lambda"] = 0.0
    cfg["train"]["init_reference_config"] = str(base_path)
    cfg["eval"]["threshold_masks_above"] = [0.1, 0.3]
    cfg["eval"]["threshold_masks_below"] = [0.4]
    cfg["notes"] = (f"M1 three-head on v5.2 {aid}: X5Dcap_v52cv_f{fold}_s{seed} recipe (v5.2 base decided 2026-09-24) with out_dim 3 = "
                    f"[peak-frame WSS log_z, TAWSS log_z, OSI logit_z], per-channel CV5 fold train statistics (method='multi'), equal-weight "
                    f"channel MSE, q90 pinball off; inputs/sampling/density augmentation/budget unchanged; backbone paired with the same "
                    f"X5Dcap fold/seed config, output layers rebuilt; held-out fold read once.")
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
    return aid, cfg, base


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=16)
    args = ap.parse_args()
    manifest = build_cycle_view(args.workers)
    print(f"v5.2 cycle view: {manifest['n_cases']} cases, period-wrap flagged {len(manifest['period_wrap_flagged'])}", flush=True)
    arms = []
    for seed in SEEDS:
        for fold in FOLDS:
            aid, cfg, base = configure(fold, seed)
            W1.save(CONFIGS / f"{aid}.json", cfg)
            arms.append(dict(id=aid, arm="M1cap", fold=fold, seed=seed, config=f"{aid}.json", run_name=cfg["name"],
                             title="M1 three-head on the v5.2 X5Dcap recipe", parent=f"X5Dcap_v52cv_f{fold}_s{seed} ({BASE_NAME})",
                             single_change=False, config_diff_vs_parent=W1.declared_diff(base, cfg),
                             paired_reference_run=str(ROOT / "training_wss_min/runs/wss_v52_20260923" / f"X5Dcap_v52cv_f{fold}_s{seed}")))
    W1.save(CONFIGS / "matrix.json", dict(
        schema_version=1, experiment=NAME, parent_experiment=BASE_NAME,
        cycle_view={"root": str(CYC), "manifest_sha256": sha256(CYC / "cycle_manifest.json"), "n_cases": manifest["n_cases"],
                    "period_wrap_flagged": len(manifest["period_wrap_flagged"])},
        execution="master 4x RTX 4090 (Slurm) via tools/run_local_train_queue.py; train -> eval best -> eval last (--save-predictions)",
        expected_training_runs=len(arms), expected_evaluations=2 * len(arms), arms=arms,
        reading_rule="CV5 pooled out-of-fold over 261 units per seed (mean ± sd over 3 seeds) and three-seed OOF ensemble; like-for-like "
                     "subset = the 136 v5.1 train units vs v5.1 cv3 M1 OOF; guard = peak channel normalized R2_cb vs the paired X5Dcap "
                     "fold/seed run (drop <= 0.02, as the stage-2 M1 gate); see experiments/<exp>/PREREG.md"))
    print(CONFIGS / "matrix.json", len(arms), "arms")


if __name__ == "__main__":
    main()

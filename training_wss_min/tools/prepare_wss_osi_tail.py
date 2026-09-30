"""Prepare the OSI tail matrix on the deployed M1 three-head recipe (cv3_v51 folds, seed 1234; node04, outside Slurm).

Origin: tracking §34.11 (2026-09-24 zero-training probes P0a–d): OSI error concentrates in the high-OSI tail (43 % of SSE on the
12.5 % of points with OSI >= 0.3) and is an amplitude compression (true >= 0.3 → predicted mean 0.24); resolution is not binding and
volume-field features add nothing.  Every arm = the stage-2 M1_f{k}_s1234 config with exactly one change:

  M1r_f{k}  none (same-hardware control; also measures node04-vs-master jitter against the stored M1 run)
  K1_f{k}   train.loss_osi_mask_lambda = 0.1 (soft-mask BCE on the OSI channel at OSI > 0.2 / 0.3, temperature 0.25 z)
  K2_f{k}   train.loss_osi_tail_alpha = 2.0 (OSI-channel squared error × (1 + 2·1[OSI_true > 0.2]), batch-mean normalised)
  K5_f{k}   8 appended flow-separation geometry inputs (F3 upstream history s_over_d / up_min_r_ratio_2d / up_min_r_ratio_5d /
            up_max_r_ratio_5d / up_kappa_5d, F2 bif_angle_cos / carina_cos, log_r_over_rdistal); fold-train statistics for the
            appended keys, the 27 reference keys keep the fold base statistics; paired init zero-initialises the new columns

Preregistration: training_wss_min/experiments/wss_osi_tail_20260924/PREREG.md.

    python -u -m training_wss_min.tools.prepare_wss_osi_tail
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

from training_wss_min import config as C
from training_wss_min import dataset as D
from training_wss_min.tools import prepare_wss_local_wave1 as W1

ROOT = C.PROJECT_ROOT
NAME = "wss_osi_tail_20260924"
M1_NAME = "wss_cycle_m1_20260921"
CONFIGS = ROOT / "training_wss_min/configs" / NAME
EXP = ROOT / "training_wss_min/experiments" / NAME
M1_CONFIGS = ROOT / "training_wss_min/configs" / M1_NAME
M1_RUNS = ROOT / "training_wss_min/runs" / M1_NAME
SEED = 1234
SEP_KEYS = ["s_over_d", "up_min_r_ratio_2d", "up_min_r_ratio_5d", "up_max_r_ratio_5d", "up_kappa_5d",
            "bif_angle_cos", "carina_cos", "log_r_over_rdistal"]
ARMS = {
    "M1r": ("same-hardware control: M1 unchanged", {}),
    "K1": ("OSI soft-mask BCE (lambda 0.1, OSI > 0.2 / 0.3, temperature 0.25)",
           {"train.loss_osi_mask_lambda": 0.1, "train.osi_mask_thresholds": [0.2, 0.3], "train.osi_mask_temperature": 0.25}),
    "K2": ("OSI tail-weighted channel MSE (alpha 2 above OSI 0.2)", {"train.loss_osi_tail_alpha": 2.0, "train.osi_tail_threshold": 0.2}),
    "K5": ("8 appended flow-separation geometry inputs", {"data.input_features": "+SEP_KEYS", "data.feature_stats_path": "fold stats + SEP_KEYS"}),
}


def m1_path(fold: int) -> Path:
    return M1_CONFIGS / f"M1_f{fold}_s{SEED}.json"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def sep_feature_stats(fold: int, base: dict) -> Path:
    """Fold base 27-key statistics + fold-train statistics of the appended separation keys (same helper as wave 1)."""
    path = EXP / "feature_stats" / f"K5_fold{fold}_train.json"
    if path.is_file():
        return path
    data = base["data"]
    stats = D.load_wss_stats(data["wss_stats_path"])
    cases = D.load_partition(data["split_path"], "train", stats, strict=True, target=data["target"],
                             data_root=data["data_root"], required_frame_version=data["required_frame_version"],
                             extra_point_features=tuple(SEP_KEYS), point_features_root=data["point_features_root"],
                             cycle_view_root=data["cycle_view_root"])
    expected = json.loads(Path(data["split_path"]).read_text())["counts"]["train"]
    if len(cases) != expected:
        raise RuntimeError(f"fold{fold}: expected {expected} train cases, loaded {len(cases)}")
    new = D.compute_feature_stats(cases, tuple(SEP_KEYS), data["curvature_transform"])
    out = json.loads(Path(data["feature_stats_path"]).read_text())
    for key in SEP_KEYS:
        out[key] = new[key]
    out["_provenance_sep_keys"] = {"train_split": data["split_path"], "n_cases": len(cases), "keys": SEP_KEYS,
                                   "base_stats": data["feature_stats_path"], "point_features_root": data["point_features_root"]}
    W1.save(path, out)
    return path


def configure(arm: str, fold: int) -> dict:
    base = json.loads(m1_path(fold).read_text())
    cfg = json.loads(json.dumps(base))
    aid = f"{arm}_f{fold}_s{SEED}"
    cfg["name"] = f"{NAME}/{aid}"
    if arm == "K1":
        cfg["train"].update(loss_osi_mask_lambda=0.1, osi_mask_thresholds=[0.2, 0.3], osi_mask_temperature=0.25)
    elif arm == "K2":
        cfg["train"].update(loss_osi_tail_alpha=2.0, osi_tail_threshold=0.2)
    elif arm == "K5":
        missing = [k for k in SEP_KEYS if k in cfg["data"]["input_features"]]
        if missing:
            raise ValueError(f"separation keys already present: {missing}")
        cfg["data"]["input_features"] = list(cfg["data"]["input_features"]) + SEP_KEYS
        cfg["data"]["feature_stats_path"] = str(sep_feature_stats(fold, base))
    cfg["notes"] = (f"OSI tail matrix {aid}: M1_f{fold}_s{SEED} (deployed three-head recipe) with one change — {ARMS[arm][0]}; "
                    f"paired initialisation identical to M1 (reference = fold X5D_v51 config); held-out fold read once; test34 unused; "
                    f"executed on node04 outside Slurm.")
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
    for arm in ARMS:
        for fold in range(3):
            aid = f"{arm}_f{fold}_s{SEED}"
            cfg = configure(arm, fold)
            W1.save(CONFIGS / f"{aid}.json", cfg)
            base = json.loads(m1_path(fold).read_text())
            arms.append(dict(id=aid, arm=arm, fold=fold, seed=SEED, config=f"{aid}.json", run_name=cfg["name"],
                             title=ARMS[arm][0], parent=f"M1_f{fold}_s{SEED} ({M1_NAME})", single_change=True,
                             config_diff_vs_parent=W1.declared_diff(base, cfg),
                             done_marker=str(ROOT / "training_wss_min/runs" / cfg["name"] / "eval/ckpt_last/metrics.json")))
    W1.save(CONFIGS / "matrix.json", dict(
        schema_version=1, experiment=NAME, parent_experiment=M1_NAME,
        parent_runs={f"M1_f{k}_s{SEED}": str(M1_RUNS / f"M1_f{k}_s{SEED}") for k in range(3)},
        parent_configs_sha256={f"M1_f{k}_s{SEED}": sha256(m1_path(k)) for k in range(3)},
        execution="node04 (2x A100-40GB, outside Slurm) via tools/run_local_train_queue.py; train -> eval best -> eval last (--save-predictions)",
        expected_training_runs=len(arms), expected_evaluations=2 * len(arms), arms=arms,
        reading_rule="fold arms read only on their held-out fold; primary = OSI IoU@0.3 and OSI R2_cb vs same-fold M1r (same hardware); "
                     "guards = peak / TAWSS channel normalized R2_cb within 0.01 of M1r; test34 unused (see PREREG.md)"))
    print(CONFIGS / "matrix.json", len(arms), "arms")


if __name__ == "__main__":
    main()

"""Prepare the wss_time_ecc_20260918 stage-2 screening matrix (cv3_v51 folds, seed 1234): T0 / TB8 / TB16.

Matrix: docs/02-推进与变更/02-时间建模/WSS_V5_偏心与全周期时间实验矩阵_2026-09-18.md §5.
Base = wave-2a fold recipes (X5D_v51 on cv3_v51 fold k; train = other two folds, held-out fold = 'test'; test34 unused).
Frozen stage-0 artifacts (experiments/wss_time_ecc_20260918/offline, job 15062): protocol_inlet_waveform_v51.json,
wss_frame_stats_fold{k}.json (per-frame log statistics, train fold only), time_basis_fold{k}.npz (train-fold PCA).

  T0_f{k}   direct phase query: 27 inputs + q_norm/dq_norm/t_sin/t_cos, timesteps=random_frame, frame_stats normalisation,
            uniform frame sampling with peak guarantee 1/9, checkpoint by EMA(train_loss).
  TB8_f{k}  time-basis head K=8 (D1 rule K*): out_dim 9, coefficient MSE, no time inputs, pinball off (unsupported on coefficients).
  TB16_f{k} time-basis head K=16 (2K*).

    python -u -m training_wss_min.tools.prepare_wss_time_ecc      (CPU node; loads fold-train cases for T0 feature statistics)
"""
from __future__ import annotations

import json
from pathlib import Path

from training_wss_min import config as C
from training_wss_min import dataset as D
from training_wss_min.tools import prepare_wss_local_wave1 as W1

ROOT = C.PROJECT_ROOT
NAME = "wss_time_ecc_20260918"
BASE_NAME = "wss_v51_wave2a_20260916"
CONFIGS = ROOT / "training_wss_min/configs" / NAME
EXP = ROOT / "training_wss_min/experiments" / NAME
OFF = EXP / "offline"
BASE_CONFIGS = ROOT / "training_wss_min/configs" / BASE_NAME
TIME_KEYS = ["q_norm", "dq_norm", "t_sin", "t_cos"]
SEED = 1234
KINDS = {"T0": None, "TB8": 8, "TB16": 16}
ARMS = {f"{kind}_f{k}_s{SEED}": (kind, K, k) for kind, K in KINDS.items() for k in range(3)}


def base_path(fold: int) -> Path:
    return BASE_CONFIGS / f"X5D_v51_f{fold}_s{SEED}.json"


def fold_time_feature_stats(fold: int, base: dict) -> Path:
    """Fold-train z-scores for the 27 X5D inputs plus identity statistics for the bounded phase features."""
    path = EXP / "feature_stats" / f"fold{fold}_train_feature_stats_time.json"
    if path.is_file():
        return path
    features = list(base["data"]["input_features"]) + TIME_KEYS
    stats = D.load_wss_stats(base["data"]["wss_stats_path"])
    extra = tuple(f for f in features if f in C.SIDECAR_FEATURE_KEYS)
    cases = D.load_partition(base["data"]["split_path"], "train", stats, strict=True, target="wss",
                             data_root=base["data"]["data_root"], required_frame_version=base["data"]["required_frame_version"],
                             extra_point_features=extra, point_features_root=base["data"]["point_features_root"])
    expected = json.loads(Path(base["data"]["split_path"]).read_text())["counts"]["train"]
    if len(cases) != expected:
        raise RuntimeError(f"fold{fold}: expected {expected} train cases, loaded {len(cases)}")
    out = D.compute_feature_stats(cases, tuple(features), base["data"]["curvature_transform"])
    reference = json.loads(Path(base["data"]["feature_stats_path"]).read_text())
    for key in base["data"]["input_features"]:
        if key in reference and key in out and any(abs(float(out[key][f]) - float(reference[key][f])) > 1e-9 for f in ("mean", "std")):
            raise RuntimeError(f"fold{fold}: feature statistics for {key} differ from the wave-2a fold file")
    out["_provenance"] = {"train_split": base["data"]["split_path"], "n_cases": len(cases), "wss_stats": base["data"]["wss_stats_path"],
                          "wave2a_reference": base["data"]["feature_stats_path"], "time_features": TIME_KEYS,
                          "note": "27 X5D statistics identical to wave-2a fold file; phase features are bounded protocol constants (identity)"}
    W1.save(path, out)
    return path


def configure(aid: str, kind: str, K: int | None, fold: int) -> dict:
    base = json.loads(base_path(fold).read_text())
    cfg = json.loads(json.dumps(base))
    cfg["name"] = f"{NAME}/{aid}"
    cfg["data"]["wss_stats_path"] = str(OFF / f"wss_frame_stats_fold{fold}.json")
    cfg["data"]["waveform_path"] = str(OFF / "protocol_inlet_waveform_v51.json")
    cfg["data"]["target_normalization"] = "frame_stats"
    cfg["eval"]["eval_frames"] = "all"
    cfg["eval"]["time_metrics"] = True
    cfg["train"]["init_reference_config"] = str(base_path(fold))   # 同 seed 折底座：骨干/解码器逐位配对初始化
    if kind == "T0":
        cfg["data"]["timesteps"] = "random_frame"
        cfg["data"]["input_features"] = list(base["data"]["input_features"]) + TIME_KEYS
        cfg["data"]["feature_stats_path"] = str(fold_time_feature_stats(fold, base))
        cfg["train"]["selection_rule"] = "train_loss_ema"
        cfg["train"]["selection_ema_alpha"] = 0.2
        cfg["notes"] = (f"time-ecc stage-2 {aid}: X5D_v51 fold-{fold} recipe + phase condition {TIME_KEYS} (31 inputs, appended columns "
                        f"zero-initialised), timesteps=random_frame (uniform frames, peak guarantee 1/9), frame_stats normalisation, "
                        f"checkpoint by EMA(train_loss, alpha 0.2); held-out fold evaluated on all 81 frames; test34 unused.")
    else:
        cfg["data"]["timesteps"] = "time_basis"
        cfg["data"]["time_basis_path"] = str(OFF / f"time_basis_fold{fold}.npz")
        cfg["model"]["time_basis_k"] = int(K)
        cfg["model"]["out_dim"] = int(K) + 1
        cfg["train"]["loss_pinball_lambda"] = 0.0
        cfg["notes"] = (f"time-ecc stage-2 {aid}: X5D_v51 fold-{fold} recipe with a time-basis output head K={K} (out_dim {K + 1}; "
                        f"per-point targets [b0, a_1..a_K] on the train-fold PCA basis; coefficient MSE; q90 pinball off because it is "
                        f"undefined on coefficient vectors); no time inputs; one inference reconstructs all 81 frames; "
                        f"backbone/decoder paired with the fold base, head fresh; held-out fold on all 81 frames; test34 unused.")
    classes = {"data": C.DataConfig, "model": C.ModelConfig, "train": C.TrainConfig, "eval": C.EvalConfig}
    for section, cls in classes.items():
        unknown = set(cfg[section]) - set(cls.__dataclass_fields__)
        if unknown:
            raise ValueError(f"unknown {section} fields: {unknown}")
    parsed = C.ExpConfig.from_dict(cfg)
    C.validate_features(parsed)
    for key in ("wss_stats_path", "waveform_path", "feature_stats_path", "split_path"):
        if not Path(cfg["data"][key]).is_file():
            raise FileNotFoundError(f"{aid}: {key} missing: {cfg['data'][key]}")
    if kind != "T0" and not Path(cfg["data"]["time_basis_path"]).is_file():
        raise FileNotFoundError(f"{aid}: time basis missing: {cfg['data']['time_basis_path']}")
    return cfg


def main() -> None:
    d1 = json.loads((OFF / "d1_ceiling.json").read_text())
    if int(d1["k_star"]) != 8:
        raise RuntimeError(f"D1 K* = {d1['k_star']}, matrix was frozen for K* = 8")
    arms = []
    for aid, (kind, K, fold) in ARMS.items():
        cfg = configure(aid, kind, K, fold)
        W1.save(CONFIGS / f"{aid}.json", cfg)
        base = json.loads(base_path(fold).read_text())
        arms.append(dict(id=aid, title=cfg["notes"].split(":")[1].split(";")[0].strip(), config=f"{aid}.json", phase=2,
                         modules=["F6", "density_aug"] + (["phase_condition", "frame_stats", "train_loss_ema"] if kind == "T0"
                                                          else [f"time_basis_K{K}", "frame_stats", "pinball_off"]),
                         seed=SEED, fold=fold, parent=f"X5D_v51_f{fold}_s{SEED} (wave 2a)", depends_on=[], evaluate=["best", "last"],
                         single_change=kind == "T0", config_diff_vs_anchor=W1.declared_diff(base, cfg)))
    cv3 = json.loads((Path(json.loads(base_path(0).read_text())["data"]["split_path"]).parent / "cv3_summary.json").read_text())
    W1.save(CONFIGS / "matrix.json", dict(
        schema_version=1, experiment=NAME, anchor=str(W1.ANCHOR), anchor_run=str(W1.ANCHOR_RUN), control_id=None,
        external_reference_runs={f"X5D_v51_f{k}_s{SEED} (wave 2a, peak-frame base)": str(ROOT / "training_wss_min/runs" / BASE_NAME / f"X5D_v51_f{k}_s{SEED}/eval/ckpt_best/metrics.json") for k in range(3)},
        cv3=cv3, stage0_job=15062, k_star=int(d1["k_star"]), d1_fold_mean_ceiling={k: d1["fold_mean"][k] for k in ("8", "16")},
        t_null=json.loads((OFF / "d2_tnull.json").read_text())["train"]["B_scale"] | {"definition": "T-null B_scale (sigma-scaled peak prediction), train136 out-of-fold"},
        expected_training_runs=len(ARMS), expected_evaluations=2 * len(ARMS), arms=arms,
        reading_rule="fold arms are read only on their held-out fold (all 81 frames); gates G2.1-G2.6 use the three-fold mean and fold sign agreement; "
                     "T-null = D2 B_scale; test34 is read once only after the winner is locked"))
    print(CONFIGS / "matrix.json", len(arms), "arms")


if __name__ == "__main__":
    main()

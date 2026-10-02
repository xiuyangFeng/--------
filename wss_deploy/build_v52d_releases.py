"""Freeze the v5.2d full265 releases (2026-10-02): peak WSS (X5Dcap_asym2 × 3 seeds) and the three-head M1cap × 3 seeds.

Source: the v5.2d retrain (``training_wss_min/runs/wss_v52d_retrain_20261001``, tracking doc 01 block §41):
265 training units (v5.2d data, no validation split, selection = train_loss), evaluated on recover8.
Copies immutable inference inputs only (checkpoint, config, feature statistics, target statistics,
normalisation record), the v5.2d cap-area split rule and the read-outs; never trains, evaluates or changes a
source checkpoint.  The validation numbers are recomputed here from the saved predictions (read-only).

    PYTHONPATH=. python -m wss_deploy.build_v52d_releases [--output-root outputs/wss_deploy_release] [--only peak|cycle]

Each package is assembled in a staging folder, verified by the registry and only then renamed into place;
an existing destination is never overwritten.  The reference sidecar is built afterwards with
``python -m wss_deploy.build_reference_profiles --data v52d …`` (it is not part of the fingerprint).
"""
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import tempfile
from pathlib import Path

import numpy as np

from .cycle_fields import CYCLE_DEFINITION, FIELD_SPECS, STAGNATION
from .families import CYCLE_FIELDS, CYCLE_MULTI_FAMILY, CYCLE_MULTI_TARGET, PEAK_FRAME
from .io_utils import file_sha256
from .paths import PROJECT_ROOT
from .registry import ReleaseRegistry, _verify_package

FROZEN_ON = "2026-10-02"
EXPERIMENT = "wss_v52d_retrain_20261001"
PEAK_ID = "X5Dcap_asym2_v52d_3seed_20261002"
CYCLE_ID = "M1cap_v52d_3seed_20261002"
SEEDS = (1234, 7, 2025)
FOLDS = range(5)
FILES = ("ckpt_best.pt", "config.json", "feature_stats.json", "wss_global_stats.json", "target_normalization.json")
RULE = "data_wss_v5/views_v5_2d_20261001/wss_min_flowref_v1/flow_split_rule_v52d_train136.json"
SPLIT = "data_wss_v5/views_v5_2d_20261001/wss_min_view_v1/split_v52p4_full265_train265_test8.json"
TRACKING = "docs/02-推进与变更/01-X5D主线与新数据/X5D主线_实验跟踪.md §41"
CYCLE_TRACKING = "docs/02-推进与变更/03-周期量TAWSS_OSI/TAWSS_OSI_实验跟踪.md §34.17"
DATA = ("v5.2d (2026-10-01): 332 library units = 273 real + 59 synthetic; this release trains on full265 = 265 real units "
        "(AG 85 / AAA 61 / ILO 119; IND train170 + test91 + YANG_BAO_KUI + 3 recover units) with no validation split, "
        "and is evaluated on recover8 (8 cfd_auto units never trained on); 150 units whose per-step iterations exited early "
        "keep their labels (about 4 % peak-WSS relative L2 on one measured case)")
INPUT_CONTRACT = {
    "surface": "wall STL in mm, single connected manifold, 5 open boundaries (inlet + 4 iliac outlets), iliac cut level within ±5 mm of the CFD protocol",
    "preprocess": "1 mm Taubin smoothing before resampling; area-weighted resampling to median NN spacing 0.5 mm (never coarser than ~0.8 mm)",
    "centerline": "VMTK centreline atlas (vessel_geom, preset frozen-aortoiliac) with outlet names out-le/out-li/out-re/out-ri and inlet",
    "features": "27 inputs, see models/*/config.json data.input_features; z-scored with models/*/feature_stats.json (full265)",
    "flow_prior": "Murray r^3 shares on virtual-cap radii (log_q_branch_murray_cap, log_tau0_murray_cap) = the CFD outlet protocol; "
                  "rules/flow_split_rule_train136.json (the v5.2d refit, a 1.2624 / b 0.1002) is only read for the capfit keys, which these models do not use",
}


def _git_commit(project_root: Path) -> str | None:
    try:
        return subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=project_root, check=True, capture_output=True, text=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def _feature_contract() -> dict:
    from wss_features import contract
    value = contract()
    if not value.get("matches_record"):
        raise RuntimeError("wss_features 源码与 CONTRACT.json 记录不一致；先核对特征程序再冻结发布包。")
    return {"version": value["version"], "source_hash": value["source_hash"]}


# ----------------------------------------------------------------------------- read-only metric helpers
def _readouts(project_root: Path) -> tuple[dict, dict]:
    exp = Path(project_root) / "training_wss_min/experiments" / EXPERIMENT
    return (json.loads((exp / "readout_ckpt_best.json").read_text(encoding="utf-8")),
            json.loads((exp / "extra_readout.json").read_text(encoding="utf-8")))


def _units(pred_root: Path, channel: str | None = None) -> dict:
    from training_wss_min.tools import report_wss_v52c_retrain as R
    units = R.load_channel(pred_root, channel)
    if not units:
        raise FileNotFoundError(f"没有保存的预测：{pred_root}")
    return units


def _case_stats(units: dict) -> dict:
    """Per-unit R² (mean / median / p10) and the pooled top-10 % amplitude ratio of an ensemble's ``{unit: (y, p)}``."""
    from training_wss_min.metrics import calibration_metrics
    from training_wss_min.tools import report_wss_v52c_retrain as R
    per = [R.r2(y, p) for y, p in units.values()]
    y = np.concatenate([u[0] for u in units.values()]); p = np.concatenate([u[1] for u in units.values()])
    return {"n_units": len(units), "pa_r2cb": R.r2cb(units), "r2_casemean": float(np.mean(per)),
            "r2_casemedian": float(np.median(per)), "r2_casep10": float(np.percentile(per, 10)),
            "top10_pred_true_ratio": float(calibration_metrics(y, p)["top10_pred_true_ratio"])}


def _cohort(uid: str) -> str:
    return uid.split("/", 1)[0]


def peak_metrics(project_root: Path) -> dict:
    from training_wss_min.tools import report_wss_v52c_retrain as R
    runs = Path(project_root) / "training_wss_min/runs" / EXPERIMENT
    readout, extra = _readouts(project_root)
    full = [_units(R.pred_dir(runs / f"X5Dcap_asym2_full265_s{s}", "best")) for s in SEEDS]
    recover = _case_stats(R.ensemble(full))
    oof_members = []
    for seed in SEEDS:
        merged = {}
        for fold in FOLDS:
            merged.update(_units(R.pred_dir(runs / f"X5Dcap_asym2_v52cv_f{fold}_s{seed}", "best")))
        oof_members.append(merged)
    oof = R.ensemble(oof_members)
    by_cohort = {c: _case_stats({u: v for u, v in oof.items() if _cohort(u) == c}) for c in ("AG", "AAA", "ILO")}
    a, b = readout["A"], readout["B"]
    return {
        "protocol": "full265 (265 v5.2d units, no validation, selection = train_loss) -> recover8 (8 held-out cfd_auto units); "
                    "recipe estimate = CV5 patient-grouped out-of-fold on 261 units (same recipe, each fold model trained on ~206 units)",
        "recover8_ensemble": {**recover, "unit_median_readout": b["ensemble3"]["unit_median"],
                              "per_unit_r2": b["ensemble3"]["per_unit"]},
        "recover8_single_seed_pa_r2cb": [b[f"s{s}"]["r2cb"] for s in SEEDS],
        "recover8_previous_release_data_v52p4": {"ensemble_pa_r2cb": b["ensemble3"]["r2cb_pre_fix"],
                                                 "note": "same recipe and seeds trained on v5.2p4 (before the v5.2c/v5.2d label fixes); recover8 labels unchanged"},
        "cv5_oof": {"n_units": a["ensemble3"]["n_units"], "single_seed_pa_r2cb": [a[f"s{s}"]["r2cb"] for s in SEEDS],
                    "single_seed_mean": a["seed_mean"], "single_seed_sd": float(np.std([a[f"s{s}"]["r2cb"] for s in SEEDS], ddof=1)),
                    "ensemble": {**_case_stats(oof), "readout_pa_r2cb": a["ensemble3"]["r2cb"]},
                    "ensemble_by_cohort": by_cohort},
        "e3_cv5_fold_models_on_recover8": readout.get("E3"),
        "expected_new_case": "about 0.79 (CV5 out-of-fold three-seed ensemble on 261 units; the full265 models saw ~30 % more units per model); "
                             "recover8 0.84 is 8 units only",
        "readout": f"training_wss_min/experiments/{EXPERIMENT}/readout_ckpt_best.json (A / B / E3) and extra_readout.json",
    }


def cycle_metrics(project_root: Path) -> dict:
    from training_wss_min.tools import report_wss_v52c_retrain as R
    runs = Path(project_root) / "training_wss_min/runs" / EXPERIMENT
    readout, _ = _readouts(project_root)
    c, e5, b = readout["C"], readout["E5"], readout["B"]
    ens = c["ensemble3"]
    channels = {}
    for channel in ("wss", "tawss", "osi"):
        members = [_units(R.pred_dir(runs / f"M1cap_full265_s{s}", "best"), channel) for s in SEEDS]
        channels[channel] = _case_stats(R.ensemble(members))
    channels["wss"].update(readout_pa_r2cb=ens["peak_pa_r2cb"])
    channels["tawss"].update(readout_pa_r2cb=ens["tawss_pa_r2cb"], ccc_casemedian=ens["tawss_ccc_casemed"], low_iou_0p4=ens["tawss_low_iou"])
    channels["osi"].update(readout_pa_r2cb=ens["osi_r2cb"], ccc_casemedian=ens["osi_ccc_casemed"],
                           iou_gt_0p1=ens["osi_iou_0.1"], iou_gt_0p3=ens["osi_iou_0.3"])
    return {
        "protocol": "full265 (265 v5.2d units, no validation, selection = train_loss) -> recover8 (8 held-out cfd_auto units); "
                    "three-seed Pa-mean ensemble per channel; recover8 is descriptive only (n = 8)",
        "recover8_ensemble": channels,
        "recover8_stagnation": {"iou_casemean": ens["stagnation_iou"], "definition": "TAWSS < 0.4 Pa and OSI > 0.1"},
        "recover8_single_seed": {f"s{s}": c[f"s{s}"] for s in SEEDS},
        "peak_channel_guard": c.get("guard"),
        "previous_release_M1_3head_3seed_20260922_on_recover8": e5["ensemble3"],
        "peak_release_on_recover8": {"release": PEAK_ID, "pa_r2cb": b["ensemble3"]["r2cb"]},
        "known_cost": "peak-channel Pa R2_cb on recover8 0.823 vs 0.839 for the peak release (X5Dcap_asym2): the three-channel loss "
                      "carries no under-prediction weight (asym 1.0) and no q90 pinball; use the peak release for peak WSS",
        "readout": f"training_wss_min/experiments/{EXPERIMENT}/readout_ckpt_best.json (C / E5 / B)",
    }


# ----------------------------------------------------------------------------- package assembly
def _copy_models(staging: Path, runs: Path, prefix: str, short: str) -> tuple[list, list]:
    models, sources = [], []
    for seed in SEEDS:
        source = runs / f"{prefix}_s{seed}"
        relative = Path("models") / f"{short}_s{seed}"
        (staging / relative).mkdir(parents=True)
        for name in FILES:
            shutil.copy2(source / name, staging / relative / name)
        cfg = json.loads((source / "config.json").read_text(encoding="utf-8"))
        if cfg.get("train", {}).get("seed") != seed:
            raise ValueError(f"{source}: config seed {cfg.get('train', {}).get('seed')} != {seed}")
        models.append({"seed": seed, "path": relative.as_posix()})
        sources.append({"seed": seed, "path": source.relative_to(PROJECT_ROOT).as_posix(),
                        "checkpoint_sha256": file_sha256(source / "ckpt_best.pt"),
                        "recover8_metrics": (source / "eval/ckpt_best/metrics.json").relative_to(PROJECT_ROOT).as_posix()})
    return models, sources


def _finish(staging: Path, destination: Path, info: dict, readme: str) -> Path:
    (staging / "release.json").write_text(json.dumps(info, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (staging / "README.md").write_text(readme, encoding="utf-8")
    records = [f"{file_sha256(path)}  {path.relative_to(staging).as_posix()}  {path.stat().st_size}"
               for directory in (staging / "models", staging / "rules", staging / "metrics")
               for path in sorted(directory.rglob("*")) if path.is_file()]
    (staging / "MANIFEST.sha256").write_text("\n".join(records) + "\n", encoding="utf-8")
    registry = ReleaseRegistry(staging)
    _verify_package(registry.default)
    staging.rename(destination)
    return destination


def _stage(output_root: Path, release_id: str) -> tuple[Path, Path]:
    output_root = Path(output_root).resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    destination = output_root / release_id
    if destination.exists():
        raise FileExistsError(f"发布包已存在；不会覆盖冻结权重：{destination}")
    return destination, Path(tempfile.mkdtemp(prefix=".v52d_release_", dir=output_root))


def _common_files(staging: Path, project_root: Path) -> None:
    (staging / "rules").mkdir()
    shutil.copy2(project_root / RULE, staging / "rules/flow_split_rule_train136.json")
    (staging / "metrics").mkdir()
    exp = project_root / "training_wss_min/experiments" / EXPERIMENT
    shutil.copy2(exp / "readout_ckpt_best.json", staging / "metrics/v52d_retrain_readout_ckpt_best.json")
    shutil.copy2(exp / "extra_readout.json", staging / "metrics/v52d_retrain_extra_readout.json")


def build_peak(output_root: Path, project_root: Path = PROJECT_ROOT) -> Path:
    project_root = Path(project_root)
    destination, staging = _stage(output_root, PEAK_ID)
    try:
        runs = project_root / "training_wss_min/runs" / EXPERIMENT
        models, sources = _copy_models(staging, runs, "X5Dcap_asym2_full265", "X5Dcap_asym2")
        _common_files(staging, project_root)
        metrics = peak_metrics(project_root)
        (staging / "metrics/peak_validation_summary.json").write_text(json.dumps(metrics, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
        shutil.copy2(runs / "X5Dcap_asym2_full265_s1234/config.json", staging / "metrics/source_config_X5Dcap_asym2_full265_s1234.json")
        n_params = json.loads((runs / "X5Dcap_asym2_full265_s1234/training_diagnostics.json").read_text(encoding="utf-8")).get("parameter_count")
        info = {
            "release": PEAK_ID, "frozen_on": FROZEN_ON, "git_commit": _git_commit(project_root),
            "model_family": "X5Dcap_asym2",
            "target": "wall shear stress magnitude (Pa) at the peak-systole frame (step 1162 = 0.21 s into the 0.8 s cycle), per wall point",
            "fields": {"wss": {"units": "Pa", "location": "wall", "kind": "scalar", "components": 1, "frame": "peak_systole"}},
            "time_axis": [dict(PEAK_FRAME)],
            "models": models, "source_runs": sources,
            "checkpoint": "ckpt_best.pt (selection_rule=train_loss, 400 epochs)",
            "n_params_per_model": n_params,
            "ensemble_protocol": "predict each seed in log_z, denormalize_wss -> Pa, arithmetic mean over 3 seeds in Pa space",
            "training_note": "X5Dcap_asym2 = X5D (density augmentation, 27 geometry inputs) with the Murray flow prior on virtual-cap radii "
                             "and the T2 under-prediction weight 2 on the loss (§36.3 / §36.5); full265 training statistics",
            "input_contract": INPUT_CONTRACT,
            "feature_contract": _feature_contract(),
            "metrics": metrics,
            "data": DATA,
            "replaces": "X5D_v51_5seed_20260916 (v5.1 train136, retired 2026-10-02)",
            "provenance": {"tracking": TRACKING, "split": SPLIT, "flow_split_rule": RULE,
                           "configs": f"training_wss_min/configs/{EXPERIMENT}/X5Dcap_asym2_full265_s*.json",
                           "frozen_code": "GNN_v52d_frozen_20261001 (training_wss_min/*.py identical to the live tree at freeze time)"},
            "note": "config.json files keep the absolute training data paths for provenance only; a deployment loader must not read those roots.",
        }
        return _finish(staging, destination, info, _peak_readme(info))
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise


def build_cycle(output_root: Path, project_root: Path = PROJECT_ROOT) -> Path:
    project_root = Path(project_root)
    destination, staging = _stage(output_root, CYCLE_ID)
    try:
        runs = project_root / "training_wss_min/runs" / EXPERIMENT
        models, sources = _copy_models(staging, runs, "M1cap_full265", "M1cap")
        _common_files(staging, project_root)
        metrics = cycle_metrics(project_root)
        (staging / "metrics/cycle_validation_summary.json").write_text(json.dumps(metrics, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
        shutil.copy2(runs / "M1cap_full265_s1234/config.json", staging / "metrics/source_config_M1cap_full265_s1234.json")
        n_params = json.loads((runs / "M1cap_full265_s1234/training_diagnostics.json").read_text(encoding="utf-8")).get("parameter_count")
        info = {
            "release": CYCLE_ID, "frozen_on": FROZEN_ON, "git_commit": _git_commit(project_root),
            "model_family": CYCLE_MULTI_FAMILY, "target": CYCLE_MULTI_TARGET,
            "models": models, "source_runs": sources,
            "fields": CYCLE_FIELDS,
            "time_axis": [dict(PEAK_FRAME)],
            "cycle": {**CYCLE_DEFINITION, "thresholds": {"tawss_pa": list(FIELD_SPECS["tawss"]["thresholds"]),
                                                          "osi": list(FIELD_SPECS["osi"]["thresholds"]), "stagnation": dict(STAGNATION)}},
            "checkpoint": "ckpt_best.pt (selection_rule=train_loss, 400 epochs); M1cap recipe = the X5Dcap recipe per seed with a three-channel head",
            "n_params_per_model": n_params,
            "ensemble_protocol": "one forward pass per seed -> (N, 3) channels [peak WSS log_z, TAWSS log_z, OSI logit_z] -> "
                                 "per-channel physical units (exp / 0.5*sigmoid) -> arithmetic mean over 3 seeds per channel; OSI clipped to [0, 0.5]",
            "training_note": "equal-weight channel MSE; no under-prediction weight (asym 1.0); q90 pinball off (undefined on channel vectors); "
                             "paired initialisation with the same-seed X5Dcap_asym2 full265 config; full265 multi-channel statistics",
            "input_contract": INPUT_CONTRACT,
            "feature_contract": _feature_contract(),
            "metrics": metrics,
            "data": DATA,
            "replaces": "M1_3head_3seed_20260922 (v5.1 train136, retired 2026-10-02)",
            "provenance": {"tracking": f"{TRACKING}; {CYCLE_TRACKING}", "split": SPLIT, "flow_split_rule": RULE,
                           "configs": f"training_wss_min/configs/{EXPERIMENT}/M1cap_full265_s*.json",
                           "statistics": f"training_wss_min/experiments/{EXPERIMENT}/stats/multi_stats_full265_train265.json",
                           "labels": "wss_v5/views/wall_cycle_v1.py (frames 0-79 of the 81-frame CFD wall shear vector)"},
            "note": "config.json files keep the absolute training data paths for provenance only; a deployment loader must not read those roots.",
        }
        return _finish(staging, destination, info, _cycle_readme(info))
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise


def _peak_readme(info: dict) -> str:
    m = info["metrics"]; r8, cv = m["recover8_ensemble"], m["cv5_oof"]
    return f"""# {info['release']}：峰值 WSS 部署发布包（X5Dcap_asym2 三 seed，v5.2d full265，冻结于 {info['frozen_on']}）

收缩期峰值帧（0.21 s）壁面 WSS；三 seed 在 Pa 空间取均值。替换 `X5D_v51_5seed_20260916`（v5.1 train136，同日下线）。
来源 run、口径、输入合同、指标全部写在 `release.json`；`MANIFEST.sha256` 是逐文件哈希（含大小），改动任何文件都要重新生成。

| 目录 | 内容 |
|---|---|
| `models/X5Dcap_asym2_s{{1234,7,2025}}/` | `ckpt_best.pt`、`config.json`（27 维输入清单）、`feature_stats.json`（full265 z-score）、`wss_global_stats.json`（log_z 目标统计）、`target_normalization.json` |
| `rules/flow_split_rule_train136.json` | v5.2d 重拟合的 capfit 分流规则（a 1.2624 / b 0.1002）；本包模型只用 Murray 虚拟盖面两列，规则只在计算 capfit 键时被读 |
| `metrics/` | v5.2d 重训读数（A / B / C / E3 / E5）、本包验证汇总 `peak_validation_summary.json`、一份源配置副本 |

验证（只读重算自保存的预测）：
- recover8（8 例，从未参与训练，只作描述）：三 seed 集成 Pa R²_cb {r8['pa_r2cb']:.4f}，逐例中位 {r8['r2_casemedian']:.3f}；单 seed {', '.join(f'{v:.4f}' for v in m['recover8_single_seed_pa_r2cb'])}。
- 同配方 CV5 折外（261 例，按患者分组）：单 seed 均值 {cv['single_seed_mean']:.4f} ± {cv['single_seed_sd']:.4f}，三 seed 折外集成 {cv['ensemble']['pa_r2cb']:.4f}（逐例均值 {cv['ensemble']['r2_casemean']:.3f}、p10 {cv['ensemble']['r2_casep10']:.3f}，最高 10% 幅值比 {cv['ensemble']['top10_pred_true_ratio']:.3f}）。
- 新病例预期按 CV5 折外集成 0.79 说；recover8 的 0.84 只有 8 例。

部署：`wss_deploy` 的 `wall_wss_v1` 模型族；`release.json` 声明了 `feature_contract`（wss_features 源码哈希），特征程序变化时拒绝加载。
"""


def _cycle_readme(info: dict) -> str:
    r8 = info["metrics"]["recover8_ensemble"]; old = info["metrics"]["previous_release_M1_3head_3seed_20260922_on_recover8"]
    return f"""# {info['release']}：三头发布包（峰值 WSS + TAWSS + OSI，M1cap 三 seed，v5.2d full265，冻结于 {info['frozen_on']}）

一次前向同时输出 **峰值帧 WSS、周期平均 TAWSS、振荡剪切指数 OSI**；三 seed 在物理空间取均值。替换 `M1_3head_3seed_20260922`（v5.1 train136，同日下线）。
来源 run、口径、输入合同、指标全部写在 `release.json`；`MANIFEST.sha256` 是逐文件哈希（含大小），改动任何文件都要重新生成。

| 目录 | 内容 |
|---|---|
| `models/M1cap_s{{1234,7,2025}}/` | `ckpt_best.pt`、`config.json`（out_dim 3 + 27 维输入清单）、`feature_stats.json`（full265 z-score）、`wss_global_stats.json`（method=multi：wss/tawss log_z、osi logit_z）、`target_normalization.json` |
| `rules/flow_split_rule_train136.json` | 与峰值包相同的 v5.2d capfit 分流规则（模型不用 capfit 键） |
| `metrics/` | v5.2d 重训读数、本包验证汇总 `cycle_validation_summary.json`、一份源配置副本 |

recover8 三 seed 集成（8 例，只作描述；括号内为旧包 M1_3head_3seed_20260922 在同 8 例上的值）：
峰值 WSS Pa R²_cb {r8['wss']['readout_pa_r2cb']:.3f}（{old['peak_pa_r2cb']:.3f}）；TAWSS {r8['tawss']['readout_pa_r2cb']:.3f}（{old['tawss_pa_r2cb']:.3f}），CCC 中位 {r8['tawss']['ccc_casemedian']:.3f}；
OSI {r8['osi']['readout_pa_r2cb']:.3f}（{old['osi_r2cb']:.3f}），OSI > 0.1 IoU {r8['osi']['iou_gt_0p1']:.3f}；滞留区 IoU {info['metrics']['recover8_stagnation']['iou_casemean']:.3f}（{old['stagnation_iou']:.3f}）。
已知代价：峰值通道低于峰值包（recover8 0.823 对 0.839），三通道损失没有低估加罚；看峰值 WSS 用峰值包。

部署：`wss_deploy` 的 `wall_cycle_multi_v1` 模型族；报告页可在 WSS / TAWSS / OSI 之间切换。
"""


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--output-root", type=Path, default=PROJECT_ROOT / "outputs/wss_deploy_release")
    parser.add_argument("--only", choices=("peak", "cycle"), default=None)
    args = parser.parse_args(argv)
    if args.only in (None, "peak"):
        print(build_peak(args.output_root))
    if args.only in (None, "cycle"):
        print(build_cycle(args.output_root))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

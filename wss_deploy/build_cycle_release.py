"""Freeze the three M1 three-head models (peak WSS + TAWSS + OSI) for deployment.

Copies immutable inference inputs only (checkpoint, config, feature statistics, multi-channel target
statistics, normalisation record) from the train136 → test34 confirmation runs; never opens training/CFD
data, trains, evaluates or changes a source checkpoint.

    python -m wss_deploy.build_cycle_release [--output-root outputs/wss_deploy_release]
"""
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import tempfile
from pathlib import Path

from .cycle_fields import CYCLE_DEFINITION, FIELD_SPECS, STAGNATION
from .families import CYCLE_FIELDS, CYCLE_MULTI_FAMILY, CYCLE_MULTI_TARGET
from .io_utils import file_sha256
from .paths import PROJECT_ROOT, RELEASE_DIR
from .registry import ReleaseRegistry, _verify_package

RELEASE_ID = "M1_3head_3seed_20260922"
SOURCE_EXPERIMENT = "wss_cycle_m1_stage3_20260922"
SEEDS = (1234, 7, 2025)
FILES = ("ckpt_best.pt", "config.json", "feature_stats.json", "wss_global_stats.json", "target_normalization.json")


def _git_commit(project_root: Path) -> str | None:
    try:
        return subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=project_root, check=True, capture_output=True, text=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def _metrics(report: dict) -> dict:
    ens = report.get("ensembles", {})
    refs = report.get("references", {})
    pick = lambda block, *keys: {k: block.get(k) for k in keys if block.get(k) is not None}
    return {
        "protocol": "train136 -> test34 (34 held-out cases, read once); three-seed Pa-mean ensemble per channel",
        "test34_ensemble": {ch: pick(ens.get(ch, {}), "norm_r2cb", "pa_r2cb", "r2_casemean", "r2_casep10", "spearman", "ccc", "seg", "top10_iou")
                            for ch in ("wss", "tawss", "osi")},
        "test34_single_seed_pa_r2cb": {ch: [report["per_seed"][ch][str(s)]["pa_r2cb"] for s in SEEDS] for ch in ("wss", "tawss", "osi")
                                       if all(str(s) in report.get("per_seed", {}).get(ch, {}) for s in SEEDS)},
        "test34_stagnation": pick(ens.get("stagnation", {}), "iou_casemean", "dice_med", "baseline_iou", "single_head_A1xO2"),
        "references": {"peak_wss_ensemble_X5D_v51_3seed": pick(refs.get("peak_ens3", {}), "norm_r2cb", "pa_r2cb", "r2_casemean"),
                       "single_head_tawss_A1_ens3": pick(refs.get("A1_ens3", {}), "norm_r2cb", "pa_r2cb"),
                       "single_head_osi_O2_ens3": pick(refs.get("O2_ens3", {}), "norm_r2cb", "pa_r2cb")},
        "known_cost": "peak-channel Pa R2_cb -0.034 vs the deployed X5D_v51 peak ensemble, entirely in the top-10% tail "
                      "(q90-trimmed gap 0.000; top10 amplitude ratio 0.79 vs 0.84) because q90 pinball is undefined on the channel vector",
    }


def build_release(output_root: Path = RELEASE_DIR.parent, *, project_root: Path = PROJECT_ROOT) -> Path:
    output_root = Path(output_root).resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    destination = output_root / RELEASE_ID
    if destination.exists():
        raise FileExistsError(f"发布包已存在；不会覆盖冻结权重：{destination}")
    legacy = json.loads((RELEASE_DIR / "release.json").read_text(encoding="utf-8"))
    report_path = Path(project_root) / "training_wss_min/experiments" / SOURCE_EXPERIMENT / "offline/m1_stage3_report_best.json"
    report = json.loads(report_path.read_text(encoding="utf-8"))
    staging = Path(tempfile.mkdtemp(prefix=".cycle_release_", dir=output_root))
    try:
        models, sources = [], []
        for seed in SEEDS:
            source = Path(project_root) / "training_wss_min/runs" / SOURCE_EXPERIMENT / f"M1_s{seed}"
            relative = Path("models") / source.name
            (staging / relative).mkdir(parents=True)
            for name in FILES:
                shutil.copy2(source / name, staging / relative / name)
            models.append({"seed": seed, "path": relative.as_posix()})
            sources.append({"seed": seed, "path": str(source), "checkpoint_sha256": file_sha256(source / "ckpt_best.pt"),
                            "test34_metrics": str(source / "eval/ckpt_best/metrics.json")})
        (staging / "rules").mkdir()
        shutil.copy2(RELEASE_DIR / "rules/flow_split_rule_train136.json", staging / "rules/flow_split_rule_train136.json")
        (staging / "metrics").mkdir()
        shutil.copy2(report_path, staging / "metrics/m1_3head_test34.json")
        shutil.copy2(Path(project_root) / "training_wss_min/runs" / SOURCE_EXPERIMENT / "M1_s1234/config.json",
                     staging / "metrics/source_config_M1_s1234.json")
        n_params = None
        diag = Path(project_root) / "training_wss_min/runs" / SOURCE_EXPERIMENT / "M1_s1234/training_diagnostics.json"
        if diag.is_file():
            n_params = json.loads(diag.read_text(encoding="utf-8")).get("parameter_count")
        info = {
            "release": RELEASE_ID, "frozen_on": "2026-09-22", "git_commit": _git_commit(Path(project_root)),
            "model_family": CYCLE_MULTI_FAMILY, "target": CYCLE_MULTI_TARGET,
            "models": models, "source_runs": sources,
            "fields": CYCLE_FIELDS,
            "time_axis": [{"index": 0, "step": 1162, "time_s": 0.21, "label": "peak_systole"}],
            "cycle": {**CYCLE_DEFINITION, "thresholds": {"tawss_pa": list(FIELD_SPECS["tawss"]["thresholds"]),
                                                          "osi": list(FIELD_SPECS["osi"]["thresholds"]), "stagnation": dict(STAGNATION)}},
            "checkpoint": "ckpt_best.pt (selection_rule=train_loss, 400 epochs); deployed X5D_v51 recipe per seed with a three-channel head",
            "n_params_per_model": n_params,
            "ensemble_protocol": "one forward pass per seed -> (N, 3) channels [peak WSS log_z, TAWSS log_z, OSI logit_z] -> "
                                 "per-channel physical units (exp / 0.5*sigmoid) -> arithmetic mean over 3 seeds per channel; OSI clipped to [0, 0.5]",
            "training_note": "equal-weight channel MSE; q90 pinball off (undefined on channel vectors); paired initialisation with the same-seed C1 reference; "
                             "train136 multi-channel statistics (peak log_z file of the deployed X5D_v51 + wss_min_cycle_v1 cycle statistics)",
            "input_contract": legacy.get("input_contract"),
            "metrics": _metrics(report),
            "data": legacy.get("data"),
            "provenance": {"matrix": "docs/02-推进与变更/WSS_PINN/WSS_V5_周期积分量TAWSS_OSI直接回归实验矩阵_2026-09-20.md §14",
                           "tracking": "docs/02-推进与变更/WSS_PINN/WSS_V5_训练实验跟踪.md §34.6",
                           "labels": "wss_v5/views/wall_cycle_v1.py (frames 0-79 of the 81-frame CFD wall shear vector)"},
            "note": "config.json files keep the absolute training data paths for provenance only; a deployment loader must not read those roots.",
        }
        (staging / "release.json").write_text(json.dumps(info, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        (staging / "README.md").write_text(_readme(info), encoding="utf-8")
        records = [f"{file_sha256(path)}  {path.relative_to(staging).as_posix()}  {path.stat().st_size}"
                   for directory in (staging / "models", staging / "rules", staging / "metrics")
                   for path in sorted(directory.rglob("*")) if path.is_file()]
        (staging / "MANIFEST.sha256").write_text("\n".join(records) + "\n", encoding="utf-8")
        registry = ReleaseRegistry(staging)
        _verify_package(registry.default)
        staging.rename(destination)
        return destination
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise


def _readme(info: dict) -> str:
    m = info["metrics"]["test34_ensemble"]
    return f"""# {info['release']}：M1 三头单模型三 seed 部署发布包（冻结于 {info['frozen_on']}）

一次前向同时输出 **峰值帧 WSS、周期平均 TAWSS、振荡剪切指数 OSI** 三个壁面标量场；三 seed 在物理空间取均值。
来源 run、口径、输入合同、指标全部写在 `release.json`；`MANIFEST.sha256` 是逐文件哈希（含大小），改动任何文件都要重新生成。

| 目录 | 内容 |
|---|---|
| `models/M1_s{{1234,7,2025}}/` | `ckpt_best.pt`、`config.json`（out_dim 3 + 27 维输入清单）、`feature_stats.json`（train136 z-score）、`wss_global_stats.json`（method=multi：wss/tawss log_z、osi logit_z）、`target_normalization.json` |
| `rules/flow_split_rule_train136.json` | capfit 分流规则（与 X5D_v51 发布包相同，只在计算 capfit 键时被读） |
| `metrics/` | test34 三 seed 读数（`m1_3head_test34.json`）与一份源配置副本 |

test34 三 seed 集成（归一化 R²_cb / 物理 R²_cb）：峰值 WSS {m['wss'].get('norm_r2cb', 0):.4f} / {m['wss'].get('pa_r2cb', 0):.4f}；
TAWSS {m['tawss'].get('norm_r2cb', 0):.4f} / {m['tawss'].get('pa_r2cb', 0):.4f}；OSI {m['osi'].get('norm_r2cb', 0):.4f} / {m['osi'].get('pa_r2cb', 0):.4f}。
已知代价：峰值通道 Pa 相对已部署 X5D_v51 峰值集成低 0.034，全部在最高 10% 尾部（pinball 关）。

部署：`wss_deploy` 的 `wall_cycle_multi_v1` 模型族；服务里在发布包列表选择本包即可，报告页可在 WSS / TAWSS / OSI 之间切换壁面着色，
`field.npz` / `points_wss.csv` / `wall_wss.vtp` 带 `tawss_pa` 与 `osi` 列，`summary.json` 的 `cycle` 块含阈值面积份额与滞留区（TAWSS < 0.4 Pa ∧ OSI > 0.1）。
"""


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, default=RELEASE_DIR.parent)
    arguments = parser.parse_args()
    print(build_release(arguments.output_root))

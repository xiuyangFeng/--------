#!/usr/bin/env python3
"""V3P 路径 I7 · checkpoint 特征衰减探针（0 重训 · CPU，可搭 GPU 评测）。

对 post5463 band（5466/5468/5478 等价 AsymW-a run）：
  - 解析 history.csv：val_wss_r2_wss vs val_r2_p 早峰曲线
  - 在 best_wss_model vs last_model 上提取 backbone 特征（shared_decoder 输出）
  - 冻结特征 + Ridge probe → 区分「特征覆写」vs「head 过拟合」

产物：``outputs/field/f0_decision/v3p_i7_ckpt_probe_<date>.json``
"""
from __future__ import annotations

import argparse
import csv
import json
from datetime import date
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence

import numpy as np
import torch
from sklearn.linear_model import Ridge

from ..core.config import ExperimentConfig, resolve_wss_effective_dim
from ..core.data import FieldGraphDataset, build_dataloader, build_feature_mask, build_required_data_keys
from ..core.denylist import resolve_split_subset
from ..core.io import load_checkpoint
from ..core.models import build_field_model_from_config
from ..core.splits import SplitSpec
from ..core.utils import resolve_device, set_seed
from .run_v3_f0_decision import DEFAULT_ASYMW_GLOB, REPO_ROOT, _discover_asymw_runs, _r2_score, _safe_json_float


def _read_history(run_dir: Path) -> List[Dict[str, str]]:
    path = run_dir / "history.csv"
    if not path.is_file():
        return []
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def _history_summary(rows: Sequence[Mapping[str, str]]) -> Dict[str, Any]:
    if not rows:
        return {"error": "no history.csv"}
    epochs: List[int] = []
    val_wss: List[float] = []
    val_p: List[float] = []
    for r in rows:
        try:
            epochs.append(int(float(r.get("epoch", 0))))
            val_wss.append(float(r.get("val_wss_r2_wss", "nan")))
            val_p.append(float(r.get("val_r2_p", "nan")))
        except ValueError:
            continue
    if not epochs:
        return {"error": "empty history metrics"}
    best_i = int(np.nanargmax(val_wss))
    last_i = len(epochs) - 1
    return {
        "n_epochs": len(epochs),
        "best_wss_epoch": epochs[best_i],
        "best_val_wss_r2_wss": _safe_json_float(val_wss[best_i]),
        "last_val_wss_r2_wss": _safe_json_float(val_wss[last_i]),
        "val_wss_drop_after_peak": _safe_json_float(val_wss[best_i] - val_wss[last_i]),
        "val_r2_p_at_best_wss": _safe_json_float(val_p[best_i]),
        "val_r2_p_at_last": _safe_json_float(val_p[last_i]),
        "early_peak": bool(epochs[best_i] <= 20),
    }


@torch.no_grad()
def _extract_wall_features(
    model: torch.nn.Module,
    loader,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    feats: List[np.ndarray] = []
    targets: List[np.ndarray] = []
    is_wall_idx = 9
    for batch in loader:
        batch = batch.to(device)
        if not hasattr(model, "_encode"):
            raise TypeError("model has no _encode(); expected FieldPointNeXt")
        h = model._encode(batch)
        wall = batch.x[:, is_wall_idx] > 0.5
        if not bool(wall.any()):
            continue
        wss = batch.y_wss[wall, 0:1].cpu().numpy()
        feats.append(h[wall].cpu().numpy())
        targets.append(wss)
    if not feats:
        return np.empty((0, 0)), np.empty((0, 1))
    return np.concatenate(feats, axis=0), np.concatenate(targets, axis=0)


def _ridge_probe_r2(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_test: np.ndarray,
    y_test: np.ndarray,
) -> float:
    if x_train.shape[0] < 50 or x_test.shape[0] < 20:
        return float("nan")
    reg = Ridge(alpha=1.0)
    reg.fit(x_train, y_train.ravel())
    pred = reg.predict(x_test)
    return _r2_score(y_test.ravel(), pred)


def _probe_run(run_dir: Path) -> Dict[str, Any]:
    cfg_path = run_dir / "config.snapshot.json"
    if not cfg_path.is_file():
        return {"run_dir": str(run_dir), "error": "missing config.snapshot.json"}

    config = ExperimentConfig.from_json(cfg_path)
    config.validate()
    split_path = run_dir / "split.snapshot.json"
    split = SplitSpec.from_json(split_path if split_path.is_file() else config.data.split_file)

    set_seed(config.system.seed, deterministic=config.system.deterministic)
    device = resolve_device(config.system.device)
    model = build_field_model_from_config(config).to(device)

    feature_mask = build_feature_mask(
        enabled_node_features=config.data.enabled_node_features,
        enabled_global_features=config.data.enabled_global_features,
    )
    eff_wss_dim = resolve_wss_effective_dim(
        config.model.wss_dim,
        config.model.wss_output_mode,
        config.model.wss_metric_dim,
    )
    required_data_keys = build_required_data_keys(
        config.model.name,
        wss_dim=eff_wss_dim,
        wss_target_frame=config.data.wss_target_frame,
    )

    def _loader(subset: str):
        ds = FieldGraphDataset(
            root=config.data.data_root,
            case_names=resolve_split_subset(split, subset, config.data.data_root),
            graphs_subdir=config.data.graphs_subdir,
            augment=False,
            preload=False,
            feature_mask=feature_mask,
            required_keys=required_data_keys,
            wss_target_frame=config.data.wss_target_frame,
            wss_domain_norm=config.data.wss_domain_norm,
            wss_domain_norm_stats=config.data.wss_domain_norm_stats,
        )
        return build_dataloader(
            ds,
            batch_size=config.data.batch_size,
            shuffle=False,
            num_workers=0,
            pin_memory=False,
        )

    train_loader = _loader("train")
    test_loader = _loader("test")

    hist = _history_summary(_read_history(run_dir))
    summary_path = run_dir / "summary.json"
    seed = int(json.loads(summary_path.read_text()).get("seed", config.system.seed)) if summary_path.is_file() else config.system.seed
    row: Dict[str, Any] = {
        "run_dir": str(run_dir),
        "seed": seed,
        "history": hist,
    }

    for ckpt_name in ("best_wss_model.pt", "last_model.pt"):
        ckpt = run_dir / ckpt_name
        if not ckpt.is_file():
            row[f"{ckpt_name}_probe"] = {"error": "missing checkpoint"}
            continue
        load_checkpoint(model, ckpt, device)
        x_tr, y_tr = _extract_wall_features(model, train_loader, device)
        x_te, y_te = _extract_wall_features(model, test_loader, device)
        row[f"{ckpt_name}_probe"] = {
            "train_n_wall": int(x_tr.shape[0]),
            "test_n_wall": int(x_te.shape[0]),
            "test_r2_wss_mag_probe": _safe_json_float(_ridge_probe_r2(x_tr, y_tr, x_te, y_te)),
        }

    bw = row.get("best_wss_model.pt_probe", {})
    last = row.get("last_model.pt_probe", {})
    bw_r2 = bw.get("test_r2_wss_mag_probe")
    last_r2 = last.get("test_r2_wss_mag_probe")
    val_drop = hist.get("val_wss_drop_after_peak")
    if isinstance(bw_r2, (int, float)) and isinstance(last_r2, (int, float)) and isinstance(val_drop, (int, float)):
        if val_drop > 0.05 and last_r2 >= bw_r2 - 0.02:
            quadrant = "feature_overwrite_or_stable_probe"
        elif val_drop > 0.05 and last_r2 < bw_r2 - 0.05:
            quadrant = "feature_decay"
        elif val_drop <= 0.05:
            quadrant = "no_clear_early_peak"
        else:
            quadrant = "head_overfit_suspect"
        row["i7_quadrant"] = quadrant
    return row


def main() -> None:
    ap = argparse.ArgumentParser(description="V3P I7 ckpt feature probe")
    ap.add_argument("--field-root", type=Path, default=REPO_ROOT / "outputs/field")
    ap.add_argument("--run-glob", type=str, default=DEFAULT_ASYMW_GLOB)
    ap.add_argument("--output", type=Path, default=None)
    args = ap.parse_args()

    runs = _discover_asymw_runs(args.field_root.resolve(), args.run_glob)
    rows = [_probe_run(r) for r in runs]

    report: Dict[str, Any] = {
        "label": "V3P-I7-ckpt-probe",
        "date": date.today().isoformat(),
        "motivation": "Q0 · post5463 early peak：特征衰减 vs head 过拟合",
        "n_runs": len(rows),
        "runs": rows,
        "note": "无 multi-epoch ckpt 时仅 best_wss vs last；完整 I7 需后续训练每 10ep 存 backbone",
    }
    quadrants = [r.get("i7_quadrant") for r in rows if r.get("i7_quadrant")]
    if quadrants:
        report["summary"] = {
            "dominant_quadrant": max(set(quadrants), key=quadrants.count),
            "recommendation": (
                "I6-a two-stage freeze"
                if "feature_decay" in quadrants or "feature_overwrite_or_stable_probe" in quadrants
                else "TODO-20 EMA/head regularization"
            ),
        }

    out = args.output or (
        REPO_ROOT / "outputs/field/f0_decision" / f"v3p_i7_ckpt_probe_{date.today().strftime('%Y%m%d')}.json"
    )
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(report.get("summary", {"n_runs": len(rows)}), indent=2, ensure_ascii=False))
    print(out)


if __name__ == "__main__":
    main()

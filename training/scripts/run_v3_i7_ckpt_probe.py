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
from collections import Counter
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
from .run_v3_f0_decision import REPO_ROOT, _r2_score, _safe_json_float

DEFAULT_POST5463_GLOB = (
    "field_v3_pointnext_localpool_main01_geom_pw_asymw_a_wall13000_near2000_"
    "split_AG_v1_seed*_202606*"
)
DEFAULT_EXP_ID_FILTER = "V3P-G-Baseline-AsymW-a-post5463"


def _read_history(run_dir: Path) -> List[Dict[str, str]]:
    path = run_dir / "history.csv"
    if not path.is_file():
        return []
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def _matches_exp_id(run_dir: Path, exp_id_filter: str) -> bool:
    if not exp_id_filter:
        return True
    summary = run_dir / "summary.json"
    if summary.is_file():
        try:
            return json.loads(summary.read_text(encoding="utf-8")).get("exp_id") == exp_id_filter
        except json.JSONDecodeError:
            return False
    cfg = run_dir / "config.snapshot.json"
    if cfg.is_file():
        try:
            return json.loads(cfg.read_text(encoding="utf-8")).get("meta", {}).get("exp_id") == exp_id_filter
        except json.JSONDecodeError:
            return False
    return False


def _discover_runs(field_root: Path, pattern: str, exp_id_filter: str) -> List[Path]:
    runs = sorted(p for p in field_root.glob(pattern) if p.is_dir() and _matches_exp_id(p, exp_id_filter))
    by_seed: Dict[int, Path] = {}
    for run_dir in runs:
        summary = run_dir / "summary.json"
        if not summary.is_file():
            continue
        try:
            seed = int(json.loads(summary.read_text(encoding="utf-8")).get("seed", -1))
        except (json.JSONDecodeError, ValueError, TypeError):
            continue
        if seed < 0:
            continue
        prev = by_seed.get(seed)
        if prev is None or run_dir.name > prev.name:
            by_seed[seed] = run_dir
    return [by_seed[k] for k in sorted(by_seed)]


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
    *,
    max_wall_points: int,
    max_wall_per_graph: int,
) -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    feats: List[np.ndarray] = []
    targets: List[np.ndarray] = []
    is_wall_idx = 9
    seen = 0
    for batch in loader:
        batch = batch.to(device)
        if not hasattr(model, "_encode"):
            raise TypeError("model has no _encode(); expected FieldPointNeXt")
        h = model._encode(batch)
        wall_idx = torch.nonzero(batch.x[:, is_wall_idx] > 0.5, as_tuple=False).view(-1)
        if wall_idx.numel() == 0:
            continue
        if max_wall_per_graph > 0 and wall_idx.numel() > max_wall_per_graph:
            perm = torch.randperm(wall_idx.numel(), device=wall_idx.device)[:max_wall_per_graph]
            wall_idx = wall_idx[perm]
        if max_wall_points > 0:
            remain = max_wall_points - seen
            if remain <= 0:
                break
            if wall_idx.numel() > remain:
                wall_idx = wall_idx[:remain]
        wss = batch.y_wss[wall_idx, 0:1].cpu().numpy()
        feats.append(h[wall_idx].cpu().numpy())
        targets.append(wss)
        seen += int(wall_idx.numel())
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


def _build_eval_context(run_dir: Path):
    """从 run_dir 构造 (config, model, device, loader_factory)；供 best/last 与逐 epoch probe 复用。"""
    cfg_path = run_dir / "config.snapshot.json"
    if not cfg_path.is_file():
        raise FileNotFoundError("missing config.snapshot.json")

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

    return config, model, device, _loader


def _probe_run(
    run_dir: Path,
    *,
    max_train_wall: int,
    max_test_wall: int,
    max_wall_per_graph: int,
) -> Dict[str, Any]:
    cfg_path = run_dir / "config.snapshot.json"
    if not cfg_path.is_file():
        return {"run_dir": str(run_dir), "error": "missing config.snapshot.json"}

    config, model, device, _loader = _build_eval_context(run_dir)

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
        x_tr, y_tr = _extract_wall_features(
            model,
            train_loader,
            device,
            max_wall_points=max_train_wall,
            max_wall_per_graph=max_wall_per_graph,
        )
        x_te, y_te = _extract_wall_features(
            model,
            test_loader,
            device,
            max_wall_points=max_test_wall,
            max_wall_per_graph=max_wall_per_graph,
        )
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


def _epoch_val_wss_map(rows: Sequence[Mapping[str, str]]) -> Dict[int, float]:
    out: Dict[int, float] = {}
    for r in rows:
        try:
            out[int(float(r.get("epoch", 0)))] = float(r.get("val_wss_r2_wss", "nan"))
        except ValueError:
            continue
    return out


def _discover_periodic_ckpts(run_dir: Path) -> List[tuple[int, Path]]:
    """收集 checkpoint_epoch_*.pt（save_best_only=false 的周期产物），按 epoch 升序。"""
    out: List[tuple[int, Path]] = []
    for p in run_dir.glob("checkpoint_epoch_*.pt"):
        stem = p.stem.replace("checkpoint_epoch_", "")
        try:
            out.append((int(stem), p))
        except ValueError:
            continue
    return sorted(out, key=lambda t: t[0])


def _plot_probe_curve(curve: Sequence[Dict[str, Any]], out_png: Path) -> Optional[str]:
    """画 probe R² vs epoch，并叠加 val_wss_r2_wss；失败返回 None。"""
    pts = [c for c in curve if isinstance(c.get("probe_r2"), (int, float))]
    if len(pts) < 2:
        return None
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return None
    epochs = [c["epoch"] for c in pts]
    probe = [c["probe_r2"] for c in pts]
    val_wss = [c.get("val_wss_r2_wss") for c in pts]
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(epochs, probe, "o-", label="frozen-feature probe R² (test WSS|·|)")
    if any(isinstance(v, (int, float)) for v in val_wss):
        ax.plot(epochs, val_wss, "s--", color="tab:red", label="val_wss_r2_wss (history)")
    ax.set_xlabel("epoch")
    ax.set_ylabel("R²")
    ax.set_title("I7 full probe: feature-decay vs head-overfit")
    ax.grid(True, alpha=0.3)
    ax.legend()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_png, dpi=130)
    plt.close(fig)
    return str(out_png)


def _judge_full_quadrant(curve: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    """基于 probe R² vs val_wss 的逐 epoch 趋势做 §8 四象限判读。"""
    pts = [
        c
        for c in curve
        if isinstance(c.get("probe_r2"), (int, float))
        and isinstance(c.get("val_wss_r2_wss"), (int, float))
    ]
    if len(pts) < 3:
        return {"quadrant": "insufficient_periodic_ckpts", "n_points": len(pts)}
    peak_i = int(np.nanargmax([c["val_wss_r2_wss"] for c in pts]))
    last_i = len(pts) - 1
    val_drop = pts[peak_i]["val_wss_r2_wss"] - pts[last_i]["val_wss_r2_wss"]
    probe_at_peak = pts[peak_i]["probe_r2"]
    probe_at_last = pts[last_i]["probe_r2"]
    probe_drop = probe_at_peak - probe_at_last
    probe_max = max(c["probe_r2"] for c in pts)
    if probe_max < 0.05:
        quadrant = "feature_never_encodes_wss"
        recommendation = "支持换轨/收口叙事（G5）；backbone 特征几乎不含 WSS 信息"
    elif val_drop <= 0.03:
        quadrant = "no_clear_early_peak"
        recommendation = "早峰不显著；不改 backbone，选优空间有限"
    elif probe_drop > 0.05:
        quadrant = "feature_decay"
        recommendation = "特征被覆写（支持 I6 冲突 / I6-a 两阶段冻结）"
    else:
        quadrant = "head_overfit"
        recommendation = "head 过拟合（转 TODO-20 EMA/多ckpt选优 + head 正则，勿动 backbone）"
    return {
        "quadrant": quadrant,
        "recommendation": recommendation,
        "val_peak_epoch": pts[peak_i]["epoch"],
        "val_wss_drop_after_peak": _safe_json_float(val_drop),
        "probe_r2_at_val_peak": _safe_json_float(probe_at_peak),
        "probe_r2_at_last": _safe_json_float(probe_at_last),
        "probe_r2_drop_after_peak": _safe_json_float(probe_drop),
        "probe_r2_max": _safe_json_float(probe_max),
        "n_points": len(pts),
    }


def _probe_run_periodic(
    run_dir: Path,
    *,
    max_train_wall: int,
    max_test_wall: int,
    max_wall_per_graph: int,
) -> Dict[str, Any]:
    """完整 I7：对 checkpoint_epoch_*.pt（+best_wss/last）逐个 probe，输出 R² vs epoch 曲线。"""
    config, model, device, _loader = _build_eval_context(run_dir)
    train_loader = _loader("train")
    test_loader = _loader("test")

    rows = _read_history(run_dir)
    hist = _history_summary(rows)
    val_map = _epoch_val_wss_map(rows)

    ckpts: List[tuple[int, Path]] = _discover_periodic_ckpts(run_dir)
    # 把 best_wss / last 也并入曲线端点（epoch 取 history 中对应值）。
    extra: List[tuple[int, Path]] = []
    bw = run_dir / "best_wss_model.pt"
    if bw.is_file() and isinstance(hist.get("best_wss_epoch"), int):
        extra.append((int(hist["best_wss_epoch"]), bw))
    last = run_dir / "last_model.pt"
    if last.is_file() and isinstance(hist.get("n_epochs"), int):
        extra.append((int(hist["n_epochs"]), last))

    seen_epochs = {e for e, _ in ckpts}
    for e, p in extra:
        if e not in seen_epochs:
            ckpts.append((e, p))
            seen_epochs.add(e)
    ckpts = sorted(ckpts, key=lambda t: t[0])

    curve: List[Dict[str, Any]] = []
    for epoch, ckpt in ckpts:
        load_checkpoint(model, ckpt, device)
        x_tr, y_tr = _extract_wall_features(
            model, train_loader, device,
            max_wall_points=max_train_wall, max_wall_per_graph=max_wall_per_graph,
        )
        x_te, y_te = _extract_wall_features(
            model, test_loader, device,
            max_wall_points=max_test_wall, max_wall_per_graph=max_wall_per_graph,
        )
        curve.append(
            {
                "epoch": epoch,
                "ckpt": ckpt.name,
                "probe_r2": _safe_json_float(_ridge_probe_r2(x_tr, y_tr, x_te, y_te)),
                "val_wss_r2_wss": _safe_json_float(val_map.get(epoch, float("nan"))),
                "train_n_wall": int(x_tr.shape[0]),
                "test_n_wall": int(x_te.shape[0]),
            }
        )

    judge = _judge_full_quadrant(curve)
    return {
        "run_dir": str(run_dir),
        "history": hist,
        "n_periodic_ckpts": len([c for c in curve if c["ckpt"].startswith("checkpoint_epoch_")]),
        "curve": curve,
        "judge": judge,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="V3P I7 ckpt feature probe")
    ap.add_argument("--run-dir", type=Path, default=None,
                    help="给定单个 run 目录时启用完整 I7（逐 epoch periodic probe）模式")
    ap.add_argument("--field-root", type=Path, default=REPO_ROOT / "outputs/field")
    ap.add_argument("--run-glob", type=str, default=DEFAULT_POST5463_GLOB)
    ap.add_argument("--exp-id-filter", type=str, default=DEFAULT_EXP_ID_FILTER)
    ap.add_argument("--output", type=Path, default=None)
    ap.add_argument("--max-train-wall", type=int, default=120_000)
    ap.add_argument("--max-test-wall", type=int, default=60_000)
    ap.add_argument("--max-wall-per-graph", type=int, default=2048)
    args = ap.parse_args()

    # 完整 I7 模式：对单个 run 的逐 epoch checkpoint 做 probe R² vs epoch 曲线。
    if args.run_dir is not None:
        run_dir = args.run_dir.resolve()
        result = _probe_run_periodic(
            run_dir,
            max_train_wall=args.max_train_wall,
            max_test_wall=args.max_test_wall,
            max_wall_per_graph=args.max_wall_per_graph,
        )
        out = args.output or (
            REPO_ROOT / "outputs/field/f0_decision"
            / f"v3p_i7_ckpt_probe_full_{date.today().strftime('%Y%m%d')}.json"
        )
        out.parent.mkdir(parents=True, exist_ok=True)
        png = _plot_probe_curve(
            result["curve"], out.with_suffix(".png")
        )
        result.update(
            {
                "label": "V3P-I7-ckpt-probe-full",
                "date": date.today().isoformat(),
                "motivation": "Q0 完整 I7：逐 epoch 冻结特征 probe，区分特征衰减 vs head 过拟合",
                "sampling": {
                    "max_train_wall": int(args.max_train_wall),
                    "max_test_wall": int(args.max_test_wall),
                    "max_wall_per_graph": int(args.max_wall_per_graph),
                },
                "plot": png,
            }
        )
        out.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
        print(json.dumps(result.get("judge", {}), indent=2, ensure_ascii=False))
        print(out)
        return

    runs = _discover_runs(args.field_root.resolve(), args.run_glob, args.exp_id_filter)
    rows = [
        _probe_run(
            r,
            max_train_wall=args.max_train_wall,
            max_test_wall=args.max_test_wall,
            max_wall_per_graph=args.max_wall_per_graph,
        )
        for r in runs
    ]

    report: Dict[str, Any] = {
        "label": "V3P-I7-ckpt-probe",
        "date": date.today().isoformat(),
        "motivation": "Q0 · post5463 early peak：特征衰减 vs head 过拟合",
        "n_runs": len(rows),
        "run_glob": args.run_glob,
        "exp_id_filter": args.exp_id_filter,
        "sampling": {
            "max_train_wall": int(args.max_train_wall),
            "max_test_wall": int(args.max_test_wall),
            "max_wall_per_graph": int(args.max_wall_per_graph),
        },
        "runs": rows,
        "note": "无 multi-epoch ckpt 时仅 best_wss vs last；完整 I7 需后续训练每 10ep 存 backbone",
    }
    quadrants = [r.get("i7_quadrant") for r in rows if r.get("i7_quadrant")]
    if quadrants:
        quadrant_counts = Counter(quadrants)
        dominant_quadrant, _ = quadrant_counts.most_common(1)[0]
        if dominant_quadrant in {"feature_decay", "feature_overwrite_or_stable_probe"}:
            recommendation = "I6-a two-stage freeze"
        elif dominant_quadrant == "head_overfit_suspect":
            recommendation = "TODO-20 EMA/head regularization"
        else:
            recommendation = "no I7 override; prioritize Q0 oracle next hop"
        report["summary"] = {
            "quadrant_counts": dict(quadrant_counts),
            "dominant_quadrant": dominant_quadrant,
            "recommendation": recommendation,
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

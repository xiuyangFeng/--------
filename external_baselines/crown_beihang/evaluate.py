from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any, Dict, List, Mapping, MutableMapping

import torch
from torch.utils.data import DataLoader

from .data import CrownDataset, CrownLazyDataset, build_datasets, collate_crown, load_train_p_stats
from .metrics import (
    PooledMetricAccumulator,
    global_ranges_from_minmax,
    grouped_metrics,
    metric_ranges,
    paper_nmae_summary,
    regression_metrics,
    update_global_ranges_from_targets,
    write_metric_rows,
)
from .model import CrownPointNet
from .utils import default_private_preprocessed_root, dump_json, load_config, project_root, resolve_device

WALL_VEL_THRESHOLD = 0.01


def _log(message: str) -> None:
    print(message, flush=True)


def _fmt_eta(elapsed: float, done: int, total: int) -> str:
    if done <= 0 or done >= total:
        return "—"
    rate = elapsed / done
    return f"{rate * (total - done):.0f}s"


def _log_phase(phase: str, detail: str = "", started_at: float | None = None) -> float:
    stamp = time.strftime("%H:%M:%S")
    if started_at is None:
        _log(f"[crown_eval {stamp}] ▶ {phase}" + (f" · {detail}" if detail else ""))
        return time.time()
    elapsed = time.time() - started_at
    _log(
        f"[crown_eval {stamp}] ✓ {phase} · {elapsed:.1f}s"
        + (f" · {detail}" if detail else "")
    )
    return time.time()


def _load_p_stats(config: Mapping[str, Any]) -> tuple[float, float]:
    data_cfg = config["data"]
    root = project_root()
    output_root = Path(data_cfg.get("preprocessed_root", str(default_private_preprocessed_root())))
    if not output_root.is_absolute():
        output_root = root / output_root
    stats = load_train_p_stats(
        output_root / "stats" / "train_stats.json",
        data_cfg.get("point_filter", "volume"),
    )
    return float(stats["p_min"]), float(stats["p_max"])


def _item_to_tensors(
    item: Dict[str, Any], device: torch.device
) -> tuple[torch.Tensor, torch.Tensor, List[int]]:
    feat = item["features"].to(device, non_blocking=True)
    targ = item["targets"].to(device, non_blocking=True)
    return feat, targ, list(item["input_indices"])


def _forward_full_sample(
    model: CrownPointNet,
    feat: torch.Tensor,
    input_indices: List[int],
    device: torch.device,
    chunk_size: int,
    verbose_chunks: bool = False,
) -> torch.Tensor:
    """对齐 model_test.py：单帧全点 forward；GPU 上分块避免 OOM。"""
    n = feat.shape[1]
    if n <= chunk_size:
        with torch.no_grad():
            pred = model(feat[input_indices].unsqueeze(0))
        return pred[0].transpose(0, 1).cpu()

    chunks: List[torch.Tensor] = []
    n_chunks = (n + chunk_size - 1) // chunk_size
    for chunk_idx, start in enumerate(range(0, n, chunk_size), start=1):
        end = min(start + chunk_size, n)
        with torch.no_grad():
            pred = model(feat[input_indices][:, start:end].unsqueeze(0))
        chunks.append(pred[0].transpose(0, 1).cpu())
        if verbose_chunks and (
            chunk_idx == 1 or chunk_idx == n_chunks or chunk_idx % max(1, n_chunks // 5) == 0
        ):
            _log(
                f"[crown_eval]     chunk {chunk_idx}/{n_chunks} · "
                f"points={end:,}/{n:,}"
            )
    return torch.cat(chunks, dim=0)


def scan_test_nmae_ranges(
    dataset: CrownDataset | CrownLazyDataset,
    output_names: List[str],
    p_min: float,
    p_max: float,
    log_every: int,
) -> Dict[str, float]:
    mins: Dict[str, float] = {}
    maxs: Dict[str, float] = {}
    total = len(dataset)
    t0 = time.time()
    for idx in range(total):
        item = dataset[idx]
        target = item["targets"].transpose(0, 1).cpu()
        update_global_ranges_from_targets(
            mins, maxs, target, output_names, p_min=p_min, p_max=p_max
        )
        if idx == 0 or idx + 1 == total or (idx + 1) % log_every == 0:
            _log(
                f"[crown_eval] GT 扫描 {idx + 1}/{total} · "
                f"sample={item['sample_id']} · n={target.shape[0]:,} · "
                f"elapsed={time.time() - t0:.0f}s · eta={_fmt_eta(time.time() - t0, idx + 1, total)}"
            )
    return global_ranges_from_minmax(mins, maxs)


def evaluate_model_paper(
    model: CrownPointNet,
    dataset: CrownDataset | CrownLazyDataset,
    device: torch.device,
    output_names: List[str],
    p_min: float,
    p_max: float,
    nmae_ranges: Mapping[str, float],
    forward_chunk_size: int,
    log_every: int,
) -> tuple[Dict[str, float], List[Dict[str, object]], List[Dict[str, object]]]:
    """对齐原文：batch_size=1 · 全点推理 · test 全局 NMAE 分母 · 逐样本日志。"""
    model.eval()
    total = len(dataset)
    pooled = PooledMetricAccumulator(output_names, p_min, p_max, nmae_ranges)
    wall_acc = PooledMetricAccumulator(output_names, p_min, p_max, nmae_ranges, prefix="wall_")
    interior_acc = PooledMetricAccumulator(output_names, p_min, p_max, nmae_ranges, prefix="interior_")
    by_sample: List[Dict[str, object]] = []
    by_case_acc: Dict[str, PooledMetricAccumulator] = {}
    t0 = time.time()

    _log(
        f"[crown_eval] 论文口径推理开始 · samples={total} · "
        f"forward_chunk_size={forward_chunk_size:,} · log_every={log_every}"
    )

    for idx in range(total):
        t_sample = time.time()
        item = dataset[idx]
        case_name = item["case_name"]
        sample_id = item["sample_id"]
        feat, targ, input_indices = _item_to_tensors(item, device)
        n_points = feat.shape[1]

        pred = _forward_full_sample(
            model,
            feat,
            input_indices,
            device,
            forward_chunk_size,
            verbose_chunks=(idx == 0 or idx + 1 == total or (idx + 1) % log_every == 0),
        )
        target = targ.transpose(0, 1).cpu()

        sample_metrics = regression_metrics(
            pred, target, output_names, p_min=p_min, p_max=p_max, nmae_ranges=nmae_ranges
        )
        row: Dict[str, object] = {
            "group": sample_id,
            "case_name": case_name,
            "sample_id": sample_id,
            "n_points": n_points,
        }
        row.update(sample_metrics)
        by_sample.append(row)

        pooled.update(pred, target)
        vel_sq = target[:, 0].square() + target[:, 1].square() + target[:, 2].square()
        wall_mask = vel_sq <= WALL_VEL_THRESHOLD
        if wall_mask.any():
            wall_acc.update(pred[wall_mask], target[wall_mask])
        interior_mask = ~wall_mask
        if interior_mask.any():
            interior_acc.update(pred[interior_mask], target[interior_mask])

        if case_name not in by_case_acc:
            by_case_acc[case_name] = PooledMetricAccumulator(
                output_names, p_min, p_max, nmae_ranges
            )
        by_case_acc[case_name].update(pred, target)

        if idx == 0 or idx + 1 == total or (idx + 1) % log_every == 0:
            _log(
                f"[crown_eval] 样本 {idx + 1}/{total} · {sample_id} · "
                f"n={n_points:,} · sample={time.time() - t_sample:.1f}s · "
                f"p_nmae={sample_metrics.get('p_nmae', float('nan')):.4f} · "
                f"p_r2={sample_metrics.get('p_r2', float('nan')):.4f} · "
                f"elapsed={time.time() - t0:.0f}s · eta={_fmt_eta(time.time() - t0, idx + 1, total)}"
            )

    metrics = pooled.finalize()
    metrics.update(wall_acc.finalize())
    metrics.update(interior_acc.finalize())
    metrics.update(paper_nmae_summary(by_sample))

    by_case_rows: List[Dict[str, object]] = []
    for case_name in sorted(by_case_acc):
        case_metrics = by_case_acc[case_name].finalize()
        by_case_rows.append({"group": case_name, **case_metrics})

    _log(
        f"[crown_eval] 论文口径推理完成 · samples={total} · "
        f"points={int(metrics.get('n_points', 0)):,} · total={time.time() - t0:.1f}s"
    )
    return metrics, by_sample, by_case_rows


def _prepare_eval_minibatch(
    batch: Dict[str, Any],
    sample_points: int,
    device: torch.device,
    eval_seed: int,
    sample_offset: int,
) -> tuple[torch.Tensor, torch.Tensor, List[tuple[str, str, int]]]:
    input_indices = batch["input_indices"]
    pv_batch: List[torch.Tensor] = []
    label_batch: List[torch.Tensor] = []
    meta: List[tuple[str, str, int]] = []

    for local_idx, (feat, targ, case_name, sample_id) in enumerate(
        zip(batch["features"], batch["targets"], batch["case_names"], batch["sample_ids"])
    ):
        feat = feat.to(device, non_blocking=True)
        targ = targ.to(device, non_blocking=True)
        n = feat.shape[1]
        choice = min(sample_points, n)
        g = torch.Generator(device=device)
        g.manual_seed(eval_seed + sample_offset + local_idx)
        idx = torch.randperm(n, generator=g, device=device)[:choice]
        pv_batch.append(feat[input_indices][:, idx])
        label_batch.append(targ[:, idx])
        meta.append((case_name, sample_id, choice))

    return torch.stack(pv_batch, dim=0), torch.stack(label_batch, dim=0), meta


def evaluate_model_subsample(
    model: CrownPointNet,
    loader: DataLoader,
    device: torch.device,
    output_names: List[str],
    sample_points: int,
    eval_seed: int,
    log_every: int = 1,
) -> tuple[torch.Tensor, torch.Tensor, List[str], List[str]]:
    model.eval()
    preds: List[torch.Tensor] = []
    targets: List[torch.Tensor] = []
    case_names: List[str] = []
    sample_ids: List[str] = []
    sample_offset = 0
    total_batches = len(loader)
    total_samples = len(loader.dataset)
    n_points = 0
    t0 = time.time()
    _log(
        f"[crown_eval] 子采样推理 · split_samples={total_samples} · "
        f"batches={total_batches} · batch_size={loader.batch_size} · "
        f"sample_points={sample_points} · log_every={log_every}"
    )

    for batch_idx, batch in enumerate(loader, start=1):
        t_batch = time.time()
        case_hint = batch["case_names"][0]
        if len(batch["case_names"]) > 1:
            case_hint = f"{case_hint}…{batch['case_names'][-1]}"

        pv_data, label_data, meta = _prepare_eval_minibatch(
            batch, sample_points, device, eval_seed, sample_offset
        )
        with torch.no_grad():
            pred = model(pv_data)

        batch_points = 0
        for b, (case_name, sample_id, choice) in enumerate(meta):
            preds.append(pred[b].transpose(0, 1).cpu())
            targets.append(label_data[b].transpose(0, 1).cpu())
            case_names.extend([case_name] * choice)
            sample_ids.extend([sample_id] * choice)
            batch_points += choice

        sample_offset += len(meta)
        n_points += batch_points
        if batch_idx == 1 or batch_idx == total_batches or batch_idx % log_every == 0:
            _log(
                f"[crown_eval] 子采样 {batch_idx}/{total_batches} · "
                f"samples={sample_offset}/{total_samples} · points={n_points:,} · "
                f"batch={time.time() - t_batch:.1f}s · "
                f"elapsed={time.time() - t0:.0f}s · eta={_fmt_eta(time.time() - t0, batch_idx, total_batches)} · "
                f"cases={case_hint}"
            )

    _log(
        f"[crown_eval] 子采样推理完成 · samples={sample_offset} · points={n_points:,} · "
        f"total={time.time() - t0:.1f}s · 正在拼接张量…"
    )
    return torch.cat(preds, dim=0), torch.cat(targets, dim=0), case_names, sample_ids


def evaluate_checkpoint(
    checkpoint_path: str | Path,
    split_name: str = "test",
    output_dir: str | Path | None = None,
    p_min: float | None = None,
    p_max: float | None = None,
    lazy_load: bool | None = None,
    log_every: int = 1,
    eval_mode: str = "paper",
    forward_chunk_size: int = 65536,
    sample_points: int | None = None,
) -> Dict[str, float]:
    t_total = time.time()
    checkpoint_path = Path(checkpoint_path)
    _log_phase(
        "启动 evaluate",
        f"checkpoint={checkpoint_path.name} split={split_name} mode={eval_mode}",
    )

    t_step = _log_phase("加载 checkpoint")
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    config = checkpoint.get("config") or load_config(checkpoint_path.parent / "config.json")
    best_epoch = checkpoint.get("epoch", checkpoint.get("best_epoch", "?"))
    _log_phase("加载 checkpoint", f"best_epoch={best_epoch}", t_step)

    device = resolve_device(config["system"].get("device", "auto"))
    data_cfg = config["data"]
    lazy_mode = lazy_load
    if lazy_mode is None:
        lazy_mode = bool(data_cfg.get("lazy_load", True))
    t_step = _log_phase(
        "构建数据集",
        f"lazy_load={lazy_mode} · partial_cache_cases={data_cfg.get('partial_cache_cases', 2)}",
    )
    datasets = build_datasets(config, splits=(split_name,), lazy_load=lazy_load)
    dataset = datasets[split_name]
    _log_phase("构建数据集", f"n_samples={len(dataset)} · device={device}", t_step)

    if p_min is None or p_max is None:
        t_step = _log_phase("读取压力归一化统计")
        p_min, p_max = _load_p_stats(config)
        _log_phase("读取压力归一化统计", f"p_min={p_min:.2f} · p_max={p_max:.2f}", t_step)

    t_step = _log_phase("加载模型权重")
    model = CrownPointNet(input_dim=dataset.input_dim, output_dim=4).to(device)
    model.load_state_dict(checkpoint["model"])
    output_names = list(dataset.target_names)
    subsample_points = int(
        sample_points if sample_points is not None else config["data"].get("sample_points", 10000)
    )
    eval_seed = int(config["system"].get("seed", 1)) + 17
    _log_phase(
        "加载模型权重",
        f"input_dim={dataset.input_dim} · outputs={','.join(output_names)}",
        t_step,
    )

    out_dir = Path(output_dir) if output_dir else checkpoint_path.parent

    if eval_mode == "paper":
        t_step = _log_phase(
            "扫描 test GT 全局 range（NMAE 分母）",
            f"log_every={log_every}",
        )
        nmae_ranges = scan_test_nmae_ranges(
            dataset, output_names, p_min, p_max, log_every=log_every
        )
        range_detail = " · ".join(f"{k}={v:.4g}" for k, v in sorted(nmae_ranges.items()))
        _log_phase("扫描 test GT 全局 range（NMAE 分母）", range_detail, t_step)

        metrics, by_sample_rows, by_case_rows = evaluate_model_paper(
            model,
            dataset,
            device,
            output_names,
            p_min,
            p_max,
            nmae_ranges,
            forward_chunk_size=forward_chunk_size,
            log_every=log_every,
        )
        metrics["eval_mode"] = "paper_full"
        metrics["forward_chunk_size"] = forward_chunk_size
        metrics["nmae_range_source"] = "test_global_gt"
        for k, v in nmae_ranges.items():
            metrics[f"nmae_range_{k}"] = v

        t_step = _log_phase(
            "计算整体指标",
            f"n_points={int(metrics.get('n_points', 0)):,} · "
            f"p_r2={metrics.get('p_r2', float('nan')):.4f} · "
            f"p_nmae={metrics.get('p_nmae', float('nan')):.4f} · "
            f"paper_p_nmae={metrics.get('paper_pressure_nmae_mean', float('nan')):.4f}",
            t_step,
        )
    else:
        eval_batch_size = int(data_cfg.get("eval_batch_size", data_cfg.get("batch_size", 16)))
        loader = DataLoader(
            dataset,
            batch_size=eval_batch_size,
            shuffle=False,
            num_workers=int(data_cfg.get("eval_num_workers", 0)),
            pin_memory=bool(data_cfg.get("pin_memory", False)),
            collate_fn=collate_crown,
        )
        pred_all, target_all, case_names, sample_ids = evaluate_model_subsample(
            model,
            loader,
            device,
            output_names,
            subsample_points,
            eval_seed,
            log_every=log_every,
        )

        t_step = _log_phase("计算整体指标", f"n_points={target_all.shape[0]:,}")
        nmae_ranges = metric_ranges(target_all, output_names, p_min=p_min, p_max=p_max)
        metrics = regression_metrics(
            pred_all,
            target_all,
            output_names,
            p_min=p_min,
            p_max=p_max,
            nmae_ranges=nmae_ranges,
        )
        metrics["n_points"] = int(target_all.shape[0])
        metrics["eval_mode"] = "subsample"
        metrics["sample_points"] = subsample_points
        metrics.update(paper_nmae_summary([]))

        vel_wall = (
            target_all[:, 0].square() + target_all[:, 1].square() + target_all[:, 2].square()
        ) <= WALL_VEL_THRESHOLD
        if vel_wall.any():
            wm = regression_metrics(
                pred_all[vel_wall],
                target_all[vel_wall],
                output_names,
                p_min=p_min,
                p_max=p_max,
                nmae_ranges=nmae_ranges,
            )
            for k, v in wm.items():
                metrics[f"wall_{k}"] = v
        interior = ~vel_wall
        if interior.any():
            im = regression_metrics(
                pred_all[interior],
                target_all[interior],
                output_names,
                p_min=p_min,
                p_max=p_max,
                nmae_ranges=nmae_ranges,
            )
            for k, v in im.items():
                metrics[f"interior_{k}"] = v

        by_case_rows = grouped_metrics(
            pred_all,
            target_all,
            case_names,
            output_names,
            p_min=p_min,
            p_max=p_max,
            nmae_ranges=nmae_ranges,
        )
        by_sample_rows = grouped_metrics(
            pred_all,
            target_all,
            sample_ids,
            output_names,
            p_min=p_min,
            p_max=p_max,
            nmae_ranges=nmae_ranges,
        )
        _log_phase(
            "计算整体指标",
            f"p_r2={metrics.get('p_r2', float('nan')):.4f} · nmae={metrics.get('nmae', float('nan')):.4f}",
            t_step,
        )

    t_step = _log_phase("写入 metrics JSON", str(out_dir / f"metrics_{split_name}.json"))
    dump_json(out_dir / f"metrics_{split_name}.json", metrics)
    _log_phase("写入 metrics JSON", "", t_step)

    t_step = _log_phase("写入 by_case CSV", str(out_dir / f"metrics_{split_name}_by_case.csv"))
    write_metric_rows(out_dir / f"metrics_{split_name}_by_case.csv", by_case_rows)
    _log_phase("写入 by_case CSV", f"groups={len(by_case_rows)}", t_step)

    t_step = _log_phase("写入 by_sample CSV", str(out_dir / f"metrics_{split_name}_by_sample.csv"))
    write_metric_rows(out_dir / f"metrics_{split_name}_by_sample.csv", by_sample_rows)
    _log_phase("写入 by_sample CSV", f"groups={len(by_sample_rows)}", t_step)

    _log(f"[crown_eval] 全部完成 · total={time.time() - t_total:.1f}s · out={out_dir}")
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate CROWN checkpoint")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--split", default="test", choices=["train", "val", "test"])
    parser.add_argument("--output-dir", default=None)
    parser.add_argument(
        "--eager-load",
        action="store_true",
        help="强制读取 merged pkl（默认 lazy partial）",
    )
    parser.add_argument(
        "--eval-mode",
        choices=["paper", "subsample"],
        default="paper",
        help="paper=全点推理+test 全局 NMAE 分母（默认，对齐原文）；subsample=随机 10000 点",
    )
    parser.add_argument(
        "--forward-chunk-size",
        type=int,
        default=65536,
        help="paper 模式下 GPU 分块 forward 点数（默认 65536）",
    )
    parser.add_argument(
        "--sample-points",
        type=int,
        default=None,
        help="subsample 模式每帧采样点数（默认读 config）",
    )
    parser.add_argument(
        "--log-every",
        type=int,
        default=1,
        help="进度日志间隔：paper=每 N 个样本；subsample=每 N 个 batch",
    )
    args = parser.parse_args()
    lazy_load = False if args.eager_load else None
    metrics = evaluate_checkpoint(
        args.checkpoint,
        args.split,
        args.output_dir,
        lazy_load=lazy_load,
        log_every=max(1, int(args.log_every)),
        eval_mode=args.eval_mode,
        forward_chunk_size=max(1024, int(args.forward_chunk_size)),
        sample_points=args.sample_points,
    )
    _log("[crown_eval] 指标摘要：")
    print(json.dumps(metrics, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()

"""Single-seed controlled u/p'/scalar-WSS cycle experiment runner.

New, isolated entry point: never changes established training/evaluation paths.
All selection is fixed-last; held-out development data are evaluated only after
training. Geometry/targets/statistics and sampling are supplied by the audited
joint_cycle_data cache. No true CFD field is supplied to the model as input.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import random
import socket
import subprocess
import time
import traceback

import numpy as np
import torch
from torch.utils.data import DataLoader

from .joint_cycle_model import build_joint_cycle_model


SCHEMA = "joint_cycle_v52_v1"
TASKS = ("velocity", "pressure", "wss")
SEGMENTS = {
    "cycle": list(range(80)), "peak": list(range(17, 27)),
    "accel": list(range(10, 17)), "decel": list(range(27, 43)),
    "plateau": list(range(5)) + list(range(58, 80)),
    "trough": list(range(5, 10)) + list(range(43, 58)),
}


def log(event, **fields):
    print(json.dumps({"time": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
                      "event": event, **fields}, ensure_ascii=False, allow_nan=False), flush=True)


def atomic_json(path, obj):
    path = Path(path)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    tmp.replace(path)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def move(value, device):
    if isinstance(value, torch.Tensor):
        return value.to(device, non_blocking=True)
    if isinstance(value, dict):
        return {k: move(v, device) for k, v in value.items()}
    return value


def case_weighted_loss(pred, target, weights, batch):
    """Equal case/phase/task weighting; vector components form ONE task."""
    error = (pred.float() - target.float()).square().mean(dim=(1, 2))
    n_cases = int(batch.max().item()) + 1
    values = []
    for b in range(n_cases):
        mask = batch == b
        w = weights[mask].float()
        if not bool(torch.isfinite(w).all()) or bool((w < 0).any()) or float(w.sum()) <= 0:
            raise ValueError("invalid volume/area quadrature weights")
        values.append((w * error[mask]).sum() / w.sum())
    return torch.stack(values).mean()


def physical_prediction(task, pred, stats):
    spec = stats["targets"][task]
    mu = np.asarray(spec["mean"], dtype=np.float64)
    sd = np.asarray(spec["std"], dtype=np.float64)
    raw = np.asarray(pred, dtype=np.float64) * sd[None] + mu[None]
    if spec["transform"] == "log":
        # Very wide numerical bound, with explicit count; not fitted to heldout.
        clipped = int(np.count_nonzero((raw < -40) | (raw > 40)))
        raw = np.exp(np.clip(raw, -40, 40)) - float(spec.get("eps", 1e-6))
        raw = np.maximum(raw, 0.0)
    else:
        clipped = 0
    return raw, clipped


def weighted_fields(y, p, w):
    """N,F,C fields. Finite physical errors, independently of z scaling."""
    w = np.asarray(w, dtype=np.float64)
    w = w / w.sum()
    err = p - y
    mse = float(np.einsum("n,nfc->", w, err * err) / (y.shape[1] * y.shape[2]))
    mae = float(np.einsum("n,nfc->", w, np.abs(err)) / (y.shape[1] * y.shape[2]))
    energy = float(np.einsum("n,nfc->", w, y * y) / (y.shape[1] * y.shape[2]))
    mean_vector = np.einsum("n,nfc->c", w, y) / y.shape[1]
    # Centre each component separately: vector R2 is rotation invariant.
    variance = max(energy - float(np.mean(mean_vector * mean_vector)), 0.0)
    return {"rmse": math.sqrt(mse), "mae": mae,
            "relative_l2": math.sqrt(mse / energy) if energy > 1e-20 else None,
            "r2": 1 - mse / variance if variance > 1e-20 else None,
            "mse": mse, "truth_energy": energy}


def case_metrics(task, y, p, weights, angle_floor):
    y, p = np.asarray(y, np.float64), np.asarray(p, np.float64)
    if y.shape != p.shape or y.ndim != 3 or y.shape[1] != 80:
        raise ValueError(f"invalid field shapes: {y.shape}, {p.shape}")
    if not np.isfinite(y).all() or not np.isfinite(p).all():
        raise ValueError("non-finite physical prediction or truth")
    out = {name: weighted_fields(y[:, ids], p[:, ids], weights)
           for name, ids in SEGMENTS.items()}
    if task == "velocity":
        for name in SEGMENTS:
            out[name]["component_rmse_m_s"] = out[name]["rmse"]
            out[name]["vector_rmse_m_s"] = math.sqrt(3) * out[name]["rmse"]
        speed_y, speed_p = np.linalg.norm(y, axis=-1), np.linalg.norm(p, axis=-1)
        out["speed"] = {name: weighted_fields(speed_y[:, ids, None], speed_p[:, ids, None], weights)
                        for name, ids in SEGMENTS.items()}
        w = weights[:, None] / weights.sum()
        ok = speed_y >= angle_floor
        cosine = (y * p).sum(-1) / np.maximum(speed_y * speed_p, 1e-20)
        # A zero predicted vector receives cos=0; it is not silently removed.
        cosine = np.clip(cosine, -1, 1)
        for name, ids in SEGMENTS.items():
            mask = ok[:, ids]
            den = float((w * mask).sum())
            out[name]["direction_cosine"] = float((w * cosine[:, ids] * mask).sum() / den) if den else None
            out[name]["direction_truth_speed_floor_m_s"] = float(angle_floor)
            out[name]["direction_weight_coverage"] = den / len(ids)
    if task == "wss":
        from scipy.stats import spearmanr
        correlations = []
        for frame in range(80):
            a, b = y[:, frame, 0], p[:, frame, 0]
            correlations.append(float(spearmanr(a, b).statistic) if np.ptp(a) > 0 and np.ptp(b) > 0 else None)
        for name, ids in SEGMENTS.items():
            vals = [correlations[i] for i in ids if correlations[i] is not None]
            out[name]["spatial_spearman_unweighted"] = float(np.mean(vals)) if vals else None
        out["tawss_80"] = weighted_fields(y.mean(1, keepdims=True), p.mean(1, keepdims=True), weights)
    return out


def mean_records(records):
    """Equal data-unit means; no claim of independent patient confidence bounds."""
    if isinstance(records[0], dict):
        return {k: mean_records([r[k] for r in records]) for k in records[0]}
    vals = [r for r in records if r is not None and np.isfinite(r)]
    return float(np.mean(vals)) if vals else None


def encode_batch(model, b, epoch, seed, evaluation):
    support = b["support"]
    return model.encode(support["pos"], support["x"], support["batch"],
                        unit_ids=b["unit_ids"], epoch=epoch, global_seed=seed,
                        evaluation=evaluation)


def build(cfg, stats):
    return build_joint_cycle_model(
        variant=cfg["variant"], support_in_dim=stats["dimensions"]["support"],
        query_dims={t: stats["dimensions"][t] for t in TASKS},
        seed=cfg["seed"], active_tasks=cfg["tasks"],
        model_options={**cfg["model"], "bc_dim": stats.get("bc_dims", 0),
                       "phase_dim": stats.get("phase_dims", 4)})


def checkpoint(out, model, optimizer, scheduler, scaler, epoch, step, cfg_hash):
    payload = {"schema": SCHEMA, "model": model.state_dict(), "optimizer": optimizer.state_dict(),
               "scheduler": scheduler.state_dict(), "scaler": scaler.state_dict(), "epoch": epoch,
               "step": step, "config_sha256": cfg_hash, "torch_rng": torch.get_rng_state(),
               "cuda_rng": torch.cuda.get_rng_state_all(), "numpy_rng": np.random.get_state(),
               "python_rng": random.getstate()}
    temp = out / "ckpt_last.pt.tmp"
    torch.save(payload, temp)
    temp.replace(out / "ckpt_last.pt")


def validate_config(cfg):
    if cfg.get("schema") != SCHEMA or cfg.get("seed") != 1234:
        raise ValueError("this registered matrix requires joint_cycle_v52_v1 / seed1234")
    if cfg["variant"] not in ("I", "J0", "J1c", "J1"):
        raise ValueError("unknown variant")
    if cfg["variant"] == "I" and len(cfg["tasks"]) != 1:
        raise ValueError("independent baselines are three separately scheduled models")
    if cfg["variant"] != "I" and cfg["tasks"] != list(TASKS):
        raise ValueError("joint variants must supervise all three tasks")
    if cfg["train"]["selection"] != "last" or cfg["train"]["phases"] != 80:
        raise ValueError("fixed-last / all80 phases contract required")


def run(config_path, *, smoke=False, resume=False):
    from .joint_cycle_data import JointCycleDataset, collate_joint_cycle, SPLIT_PATH
    config_path = Path(config_path).resolve()
    cfg = json.loads(config_path.read_text())
    validate_config(cfg)
    if Path(cfg["data"]["split_path"]).resolve() != SPLIT_PATH.resolve() or sha(SPLIT_PATH) != cfg["data"]["split_sha256"]:
        raise ValueError("configured split and data-module split differ")
    out = Path(cfg["out_dir"])
    if smoke:
        out = Path(cfg["experiment_dir"]) / "smoke" / f"{os.environ.get('SLURM_JOB_ID', 'local')}_{cfg['arm']}"
    out.mkdir(parents=True, exist_ok=True)
    if (out / "metrics.json").exists() or ((out / "ckpt_last.pt").exists() and not resume):
        raise FileExistsError(f"refusing to overwrite existing run {out}")
    if not torch.cuda.is_available():
        raise RuntimeError("training must run on the allocated GPU")
    device = torch.device("cuda")
    stats_path = Path(cfg["data"]["stats_path"])
    stats = json.loads(stats_path.read_text())
    audit = json.loads((Path(cfg["data"]["cache_root"]) / "audit.json").read_text())
    manifest = json.loads((Path(cfg["data"]["cache_root"]) / "manifest.json").read_text())
    if audit["errors"] or not manifest["complete"] or stats["split_sha256"] != cfg["data"]["split_sha256"]:
        raise ValueError("joint cache/audit/statistics contract is incomplete")
    seed_everything(cfg["seed"])
    torch.set_num_threads(int(cfg["train"].get("torch_threads", 4)))
    torch.backends.cuda.matmul.allow_tf32 = bool(cfg["train"].get("allow_tf32", True))
    torch.backends.cudnn.allow_tf32 = bool(cfg["train"].get("allow_tf32", True))
    model = build(cfg, stats).to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    gpu_query = subprocess.run(["nvidia-smi", "--query-gpu=index,uuid,name,memory.used", "--format=csv"], capture_output=True, text=True)
    provenance = {"schema": SCHEMA, "config_path": str(config_path), "config_sha256": sha(config_path),
                  "stats_sha256": sha(stats_path), "host": socket.gethostname(), "pid": os.getpid(),
                  "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
                  "slurm_array_task_id": os.environ.get("SLURM_ARRAY_TASK_ID"),
                  "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
                  "gpu_inventory": gpu_query.stdout, "parameter_count": n_params,
                  "torch": torch.__version__, "smoke": smoke,
                  "amp_init_scale": cfg["train"].get("amp_init_scale", 1024.),
                  "code": {str(p): sha(p) for p in Path(__file__).parent.glob("joint_cycle*.py")}}
    atomic_json(out / "provenance.json", provenance)
    atomic_json(out / "config.json", cfg)
    log("run_start", arm=cfg["arm"], tasks=cfg["tasks"], parameters=n_params, output=str(out), smoke=smoke)
    tc = cfg["train"]
    ds_kwargs = dict(cache_root=cfg["data"]["cache_root"], stats_path=str(stats_path),
                     support_n=cfg["data"]["support_n"], seed=cfg["seed"])
    train_ds = JointCycleDataset(partition="train", training=True,
                                query_n=cfg["data"]["train_query_n"], **ds_kwargs)
    optimizer = torch.optim.AdamW(model.parameters(), lr=tc["lr"], weight_decay=tc["weight_decay"])
    epochs = 1 if smoke else tc["epochs"]
    steps_per_epoch = math.ceil(len(train_ds) / tc["batch_cases"])
    total_steps = tc["epochs"] * steps_per_epoch
    warm_steps = max(tc["warmup_epochs"] * steps_per_epoch, 1)
    def lr_factor(step):
        if step < warm_steps:
            return (step + 1) / warm_steps
        progress = min(max((step - warm_steps) / max(total_steps - warm_steps, 1), 0), 1)
        return tc["min_lr_ratio"] + (1 - tc["min_lr_ratio"]) * (1 + math.cos(math.pi * progress)) / 2
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_factor)
    scaler = torch.amp.GradScaler("cuda", enabled=bool(tc["amp"]), init_scale=tc.get("amp_init_scale", 1024.))
    start_epoch, step = 0, 0
    if resume:
        saved = torch.load(out / "ckpt_last.pt", map_location="cpu", weights_only=False)
        if saved["config_sha256"] != sha(config_path):
            raise ValueError("resume configuration differs from checkpoint")
        model.load_state_dict(saved["model"])
        optimizer.load_state_dict(saved["optimizer"])
        scheduler.load_state_dict(saved["scheduler"])
        scaler.load_state_dict(saved["scaler"])
        start_epoch, step = saved["epoch"] + 1, saved["step"]
        torch.set_rng_state(saved["torch_rng"])
        torch.cuda.set_rng_state_all(saved["cuda_rng"])
        np.random.set_state(saved["numpy_rng"])
        random.setstate(saved["python_rng"])
    started = time.perf_counter()
    histories = []
    for epoch in range(start_epoch, epochs):
        train_ds.set_epoch(epoch)
        generator = torch.Generator().manual_seed(cfg["seed"] + epoch)
        loader = DataLoader(train_ds, batch_size=tc["batch_cases"], shuffle=True,
                            collate_fn=collate_joint_cycle, num_workers=tc["num_workers"],
                            pin_memory=True, generator=generator, persistent_workers=False)
        model.train()
        losses = {t: [] for t in cfg["tasks"]}
        epoch_start = time.perf_counter()
        for batch_index, batch_cpu in enumerate(loader):
            batch_start = time.perf_counter()
            b = move(batch_cpu, device)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast("cuda", dtype=torch.float16, enabled=bool(tc["amp"])):
                encoded = encode_batch(model, b, epoch, cfg["seed"], False)
                state = model.prepare_phase(encoded, b["phase"], bc=b.get("bc"), phase_chunk_size=tc["phase_chunk"])
                task_losses = {}
                for task in cfg["tasks"]:
                    q = b["queries"][task]
                    pred = model.decode(encoded, task, q["pos"], q["x"], q["batch"],
                                        phase_state=state, phase_chunk_size=tc["phase_chunk"])
                    task_losses[task] = case_weighted_loss(pred, q["y"], q["weight"], q["batch"])
                total = torch.stack(list(task_losses.values())).mean()
            if not bool(torch.isfinite(total)):
                raise FloatingPointError(f"nonfinite loss {cfg['arm']} epoch={epoch} batch={batch_index}")
            scaler.scale(total).backward()
            scaler.unscale_(optimizer)
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), tc["grad_clip"], error_if_nonfinite=True)
            scale_before = scaler.get_scale()
            scaler.step(optimizer)
            scaler.update()
            if scaler.get_scale() < scale_before:
                raise FloatingPointError("AMP skipped an optimizer update; stop rather than change effective exposure")
            scheduler.step()
            step += 1
            values = {t: float(v.detach()) for t, v in task_losses.items()}
            for t, v in values.items():
                losses[t].append((v, len(b["unit_ids"])))
            log("train_batch", arm=cfg["arm"], epoch=epoch + 1, step=step,
                batch=batch_index + 1, cases=b["unit_ids"], losses=values,
                grad_norm=float(grad_norm), lr=optimizer.param_groups[0]["lr"],
                amp_scale=scaler.get_scale(),
                seconds=round(time.perf_counter() - batch_start, 3))
            if smoke and batch_index >= 2:
                break
        row = {"epoch": epoch + 1, "step": step,
               "losses": {t: sum(v * n for v, n in xs) / sum(n for _, n in xs) for t, xs in losses.items()},
               "seconds": time.perf_counter() - epoch_start,
               "peak_cuda_memory_bytes": torch.cuda.max_memory_allocated()}
        with (out / "history.jsonl").open("a") as f:
            f.write(json.dumps(row, allow_nan=False) + "\n")
        histories.append(row)
        log("epoch_complete", **row)
        if smoke or (epoch + 1) % tc["checkpoint_every"] == 0 or epoch + 1 == epochs:
            checkpoint(out, model, optimizer, scheduler, scaler, epoch, step, sha(config_path))
    train_seconds = time.perf_counter() - started
    atomic_json(out / "training_complete.json", {"epochs": epochs, "steps": step,
                "seconds": train_seconds, "checkpoint": str(out / "ckpt_last.pt"), "smoke": smoke})
    # Smoke evaluation stays within TRAIN patients; never selects a model on IND.
    ev_partition = "train" if smoke else cfg["eval"]["partition"]
    eval_ds = JointCycleDataset(partition=ev_partition, training=False,
                               query_n=cfg["data"]["eval_query_n"], **ds_kwargs)
    model.eval()
    torch.backends.cuda.matmul.allow_tf32 = False
    metrics_rows = {t: [] for t in cfg["tasks"]}
    timing = []
    prediction_dir = out / "predictions"
    prediction_dir.mkdir(exist_ok=True)
    count = min(2, len(eval_ds)) if smoke else len(eval_ds)
    for index in range(count):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        record = eval_ds[index]
        batch = collate_joint_cycle([record])
        unit = batch["unit_ids"][0]
        log("eval_case_start", arm=cfg["arm"], index=index + 1, total=count, unit=unit, partition=ev_partition)
        b = move(batch, device)
        with torch.inference_mode():
            encoded = encode_batch(model, b, epochs, cfg["seed"], True)
            state = model.prepare_phase(encoded, b["phase"], bc=b.get("bc"), phase_chunk_size=tc["phase_chunk"])
            outputs = {}
            for task in cfg["tasks"]:
                q = b["queries"][task]
                parts = []
                chunk = cfg["eval"]["query_chunk"]
                for left in range(0, len(q["pos"]), chunk):
                    sl = slice(left, left + chunk)
                    pred = model.decode(encoded, task, q["pos"][sl], q["x"][sl], q["batch"][sl],
                                        phase_state=state, phase_chunk_size=tc["phase_chunk"])
                    parts.append(pred.float().cpu().numpy())
                outputs[task] = np.concatenate(parts)
        torch.cuda.synchronize()
        inference_seconds = time.perf_counter() - t0
        per_case = {"unit_id": unit, "index": index, "partition": ev_partition,
                    "cache_load_encode_decode_seconds": inference_seconds, "tasks": {}}
        arrays = {}
        for task, pred_z in outputs.items():
            q = batch["queries"][task]
            raw, clip_count = physical_prediction(task, pred_z, stats)
            true = q["y_raw"].numpy()
            weight = q["weight"].numpy().astype(np.float64)
            met = case_metrics(task, true, raw, weight, cfg["eval"]["angle_speed_floor_m_s"])
            met["numeric_exp_clipped_count"] = clip_count
            per_case["tasks"][task] = met
            metrics_rows[task].append(met)
            if cfg["eval"]["save_predictions"] or smoke:
                arrays[f"{task}_prediction"] = raw.astype(np.float32)
                arrays[f"{task}_rows"] = q["rows"].numpy()
        if arrays:
            np.savez_compressed(prediction_dir / f"{index:03d}.npz", unit_id=unit, **arrays)
        timing.append(inference_seconds)
        with (out / "per_case.jsonl").open("a") as f:
            f.write(json.dumps(per_case, allow_nan=False) + "\n")
        log("eval_case_complete", arm=cfg["arm"], unit=unit,
            seconds=round(inference_seconds, 3),
            relative_l2={t: m["cycle"]["relative_l2"] for t, m in per_case["tasks"].items()})
    metrics = {"schema": SCHEMA, "arm": cfg["arm"], "seed": cfg["seed"], "variant": cfg["variant"],
               "status": "smoke_passed" if smoke else "complete", "partition": ev_partition,
               "n_units": count, "phase_count": 80, "selection": "last", "parameter_count": n_params,
               "aggregation": "equal data-unit mean of area/volume weighted metrics; internal development only",
               "tasks": {t: mean_records(rs) for t, rs in metrics_rows.items()},
               "train_seconds": train_seconds, "peak_cuda_memory_bytes": torch.cuda.max_memory_allocated(),
               "inference": {"definition": "audited cache load + tensor transfer + geometry encode + all task/80-phase decode; excludes original geometry preprocessing and file export",
                             "median_s": float(np.median(timing)), "p95_s": float(np.percentile(timing, 95)),
                             "cold_first_case_s": timing[0], "n_units": count}}
    atomic_json(out / "metrics.json", metrics)
    log("run_complete", arm=cfg["arm"], status=metrics["status"], output=str(out))
    return metrics


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    try:
        run(args.config, smoke=args.smoke, resume=args.resume)
    except BaseException as error:
        cfg = json.loads(Path(args.config).read_text())
        out = Path(cfg["out_dir"])
        if args.smoke:
            out = Path(cfg["experiment_dir"]) / "smoke" / f"{os.environ.get('SLURM_JOB_ID', 'local')}_{cfg['arm']}"
        if out.exists() and not (out / "metrics.json").exists():
            atomic_json(out / "failed.json", {"status": "failed", "type": type(error).__name__,
                        "error": str(error), "traceback": traceback.format_exc(),
                        "slurm_job_id": os.environ.get("SLURM_JOB_ID"), "smoke": args.smoke})
        raise


if __name__ == "__main__":
    main()

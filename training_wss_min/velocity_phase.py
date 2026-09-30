"""Isolated velocity phase study; immutable fold0/206+55 and last150 contract.

This runner only uses new entrypoints and leaves every historical runner intact.
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

from .velocity_phase_model import build_velocity_phase_model
from .velocity_phase_objectives import (case_phase_mean, direction_loss, direction_coverage, gradient_l2,
                                         physical_velocity, velocity_objective)


SCHEMA = "velocity_phase_v52_v1"
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


from .velocity_phase_metrics import case_metrics


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
                        evaluation=evaluation, anchors=b.get("anchors"))


def build(cfg, stats):
    return build_velocity_phase_model(cfg, stats)


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
    arms = ("D1", "D2", "A0", "A1", "G00", "G01", "G10", "G11")
    if cfg.get("schema") != SCHEMA or cfg.get("seed") != 1234:
        raise ValueError("velocity phase study requires seed1234 / its isolated schema")
    if cfg.get("arm") not in arms or cfg.get("tasks") != ["velocity"] or cfg.get("variant") != "I":
        raise ValueError("exactly the eight registered velocity-only arms are allowed")
    if cfg["model"].get("velocity_arm") != cfg["arm"] or cfg["model"].get("round2_module") != "query_mean":
        raise ValueError("U0 parent / registered velocity_arm required")
    tc, dc = cfg["train"], cfg["data"]
    if (tc["selection"], tc["phases"], tc["epochs"], tc["batch_cases"]) != ("last", 80, 150, 4):
        raise ValueError("fixed last150 / all80 / batch4 required")
    if dc.get("fold") != 0 or cfg["eval"]["partition"] != "test":
        raise ValueError("grouped fold0 development partition required")
    if dc["support_n"] != 5000 or dc["train_query_n"]["velocity"] != 1024 or dc["eval_query_n"]["velocity"] != 16384:
        raise ValueError("original support/query budget must remain unchanged")
    if tc.get("gradient_method", "mean") != "mean":
        raise ValueError("no joint-task gradient optimizer in velocity-only study")
    expected = {"direction_speed_floor_m_s": .01, "direction_epsilon_m_s": .01,
                "direction_probe_batches": 32, "direction_gradient_ratio": .1,
                "tail_quantile": .8, "tail_multiplier": 2}
    for key, value in expected.items():
        if key in tc and tc[key] != value:
            raise ValueError(f"unregistered objective/probe setting {key}")


def read_tail_stats(cfg):
    path = Path(cfg["data"]["tail_stats_path"])
    payload = json.loads(path.read_text())
    split = json.loads(Path(cfg["data"]["split_path"]).read_text())
    if (payload.get("status") != "passed" or len(payload.get("q80_m_s", [])) != 80
            or set(payload.get("train_ids", [])) != set(split["train_cases"])
            or payload.get("split_sha256") != cfg["data"]["split_sha256"]):
        raise ValueError("train-only tail threshold contract mismatch")
    q80 = np.asarray(payload["q80_m_s"], dtype=np.float64)
    if not np.isfinite(q80).all() or (q80 <= 0).any():
        raise ValueError("tail thresholds must be finite and positive")
    return payload


def load_direction_calibration(cfg, config_path):
    if cfg["arm"] != "D1":
        return 0., None
    path = Path(cfg["train"]["direction_lambda_path"])
    spec = json.loads(path.read_text())
    if (spec.get("status") != "passed" or spec.get("partition") != "train"
            or spec.get("seed") != 1234 or spec.get("n_batches") != 32
            or spec.get("stats_sha256") != cfg["data"]["stats_sha256"]
            or spec.get("config_sha256") != sha(config_path)):
        raise ValueError("D1 requires calibrated lambda for the exact frozen config")
    value = float(spec["direction_lambda"])
    if not 1e-4 <= value <= 1:
        raise ValueError("D1 lambda outside its registered numerical bounds")
    return value, {"path": str(path), "sha256": sha(path), **spec}


def probe_direction_lambda(config_path):
    """Exactly 32 train batches; no optimizer, state mutation, or heldout query."""
    from .velocity_phase_data import (VelocityPhaseDataset, collate_velocity_phase,
                                      validate_geometry_manifest)
    config_path = Path(config_path).resolve()
    cfg = json.loads(config_path.read_text())
    validate_config(cfg)
    if cfg["arm"] != "D1":
        raise ValueError("direction calibration is a D1 training-only preparation")
    target = Path(cfg["train"]["direction_lambda_path"])
    if target.exists():
        load_direction_calibration(cfg, config_path)
        log("calibration_reused", path=str(target), sha256=sha(target))
        return
    if not torch.cuda.is_available():
        raise RuntimeError("direction probe requires an allocated GPU")
    device = torch.device("cuda")
    validate_geometry_manifest(cfg["data"]["geometry_root"], cfg["data"]["cache_root"],
                               require_complete=True, verify_files=True)
    if sha(cfg["data"]["stats_path"]) != cfg["data"]["stats_sha256"]:
        raise ValueError("normalization changed before calibration")
    stats = json.loads(Path(cfg["data"]["stats_path"]).read_text())
    seed_everything(cfg["seed"])
    torch.set_num_threads(cfg["train"].get("torch_threads", 4))
    torch.backends.cuda.matmul.allow_tf32 = False
    model = build(cfg, stats).to(device).eval()
    ds = VelocityPhaseDataset(cache_root=cfg["data"]["cache_root"], stats_path=cfg["data"]["stats_path"],
        geometry_root=cfg["data"]["geometry_root"], include_anchors=False,
        partition="train", training=True, query_n=cfg["data"]["train_query_n"],
        support_n=cfg["data"]["support_n"], seed=cfg["seed"])
    ds.set_epoch(0)
    gen = torch.Generator().manual_seed(cfg["seed"])
    loader = DataLoader(ds, batch_size=cfg["train"]["batch_cases"], shuffle=True,
        collate_fn=collate_velocity_phase, num_workers=cfg["train"]["num_workers"],
        pin_memory=True, generator=gen, persistent_workers=False)
    parameters = list(model.decoders["velocity"].parameters())
    records = []
    for index, batch_cpu in enumerate(loader):
        if index >= 32:
            break
        b = move(batch_cpu, device)
        q = b["queries"]["velocity"]
        encoded = encode_batch(model, b, 0, cfg["seed"], False)
        state = model.prepare_phase(encoded, b["phase"], bc=b["bc"],
            phase_chunk_size=cfg["train"]["phase_chunk"], phase_indices=torch.arange(80, device=device))
        pred = model.decode(encoded, "velocity", q["pos"], q["x"], q["batch"], phase_state=state,
            phase_chunk_size=cfg["train"]["phase_chunk"], query_frame=q["frame"], query_geom=q["geom_extra"])
        err = (pred.float() - q["y"].float()).square().mean(-1)
        base = case_weighted_loss(pred, q["y"], q["weight"], q["batch"])
        directional = direction_loss(physical_velocity(pred, stats), q["y_raw"], q["weight"], q["batch"])
        base_norm = float(gradient_l2(base, parameters, retain_graph=True))
        # A bounded diagnostic on the first batch distinguishes phase loss/gradient scale.
        phase_probe = {}
        if index == 0:
            for name, ids in SEGMENTS.items():
                pl = case_phase_mean(err[:, ids], q["weight"], q["batch"])
                phase_probe[name] = {"z_mse": float(pl.detach()),
                    "decoder_gradient_norm": float(gradient_l2(pl, parameters, retain_graph=True))}
        direction_norm = float(gradient_l2(directional, parameters))
        if not np.isfinite([base_norm, direction_norm]).all() or min(base_norm, direction_norm) <= 0:
            raise FloatingPointError("invalid direction calibration gradient")
        rec = {"batch": index, "unit_ids": b["unit_ids"], "z_mse": float(base.detach()),
            "direction_loss": float(directional.detach()), "base_gradient_norm": base_norm,
            "direction_gradient_norm": direction_norm, "phase_probe": phase_probe}
        records.append(rec)
        log("direction_probe_batch", **rec)
        del pred, encoded, state, base, directional, err
    if len(records) != 32:
        raise ValueError("calibration must cover exactly 32 prescribed training batches")
    base_median = float(np.median([r["base_gradient_norm"] for r in records]))
    direction_median = float(np.median([r["direction_gradient_norm"] for r in records]))
    value = float(np.clip(.1 * base_median / direction_median, 1e-4, 1.))
    record = {"status": "passed", "partition": "train", "seed": 1234, "n_batches": 32,
        "direction_lambda": value, "base_gradient_norm_median": base_median,
        "direction_gradient_norm_median": direction_median, "config_sha256": sha(config_path),
        "stats_sha256": cfg["data"]["stats_sha256"], "split_sha256": cfg["data"]["split_sha256"],
        "optimizer_updates": 0, "model_mode": "eval_with_gradients", "epoch_sampling": 0,
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"), "records": records}
    target.parent.mkdir(parents=True, exist_ok=True)
    atomic_json(target, record)
    log("direction_calibration_complete", direction_lambda=value, path=str(target))


def run(config_path, *, smoke=False, resume=False):
    if resume:
        raise ValueError("velocity phase forbids in-place resume: preserve the failed run and use a separately audited recovery directory")
    from .joint_cycle_data import SPLIT_PATH
    from .velocity_phase_data import (VelocityPhaseDataset, collate_velocity_phase,
                                      validate_geometry_manifest)
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
    if sha(stats_path) != cfg["data"]["stats_sha256"]:
        raise ValueError("training statistics fingerprint differs from registered velocity phase config")
    stats = json.loads(stats_path.read_text())
    audit = json.loads((Path(cfg["data"]["cache_root"]) / "audit.json").read_text())
    manifest = json.loads((Path(cfg["data"]["cache_root"]) / "manifest.json").read_text())
    if audit["errors"] or not manifest["complete"] or stats["split_sha256"] != cfg["data"]["split_sha256"]:
        raise ValueError("joint cache/audit/statistics contract is incomplete")
    split = json.loads(Path(cfg["data"]["split_path"]).read_text())
    if len(split["train_cases"]) != 206 or len(split["test_cases"]) != 55:
        raise ValueError("the full265 dataset is not this experiment's 206/55 split")
    geometry_audit = validate_geometry_manifest(cfg["data"]["geometry_root"],
        cfg["data"]["cache_root"], require_complete=True, verify_files=True)
    tail_spec = read_tail_stats(cfg)
    direction_lambda, calibration = load_direction_calibration(cfg, config_path)
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
                  "round2_contract": getattr(model, "round2_contract", None),
                  "velocity_phase_contract": getattr(model, "velocity_phase_contract", None),
                  "geometry_audit": geometry_audit, "direction_calibration": calibration,
                  "tail_stats_sha256": sha(cfg["data"]["tail_stats_path"]),
                  "gradient_method": cfg["train"].get("gradient_method", "mean"),
                "code": {str(p): sha(p) for p in list(Path(__file__).parent.glob("joint_cycle*.py")) + list(Path(__file__).parent.glob("velocity_phase*.py"))}}
    atomic_json(out / "provenance.json", provenance)
    atomic_json(out / "config.json", cfg)
    log("run_start", arm=cfg["arm"], tasks=cfg["tasks"], parameters=n_params, output=str(out), smoke=smoke)
    tc = cfg["train"]
    ds_kwargs = dict(cache_root=cfg["data"]["cache_root"], stats_path=str(stats_path),
                     support_n=cfg["data"]["support_n"], seed=cfg["seed"],
                     geometry_root=cfg["data"]["geometry_root"], include_anchors=cfg["arm"].startswith("G"))
    train_ds = VelocityPhaseDataset(partition="train", training=True,
                                query_n=cfg["data"]["train_query_n"], **ds_kwargs)
    tail_q80 = torch.tensor(tail_spec["q80_m_s"], dtype=torch.float32, device=device)
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
    started = time.perf_counter()
    histories = []
    for epoch in range(start_epoch, epochs):
        train_ds.set_epoch(epoch)
        generator = torch.Generator().manual_seed(cfg["seed"] + epoch)
        loader = DataLoader(train_ds, batch_size=tc["batch_cases"], shuffle=True,
                            collate_fn=collate_velocity_phase, num_workers=tc["num_workers"],
                            pin_memory=True, generator=generator, persistent_workers=False)
        model.train()
        losses = {t: [] for t in cfg["tasks"]}
        gradient_rows, clipped_steps = [], 0
        coverage_rows = []
        objective_sums, stage_sums = {}, {}
        epoch_start = time.perf_counter()
        for batch_index, batch_cpu in enumerate(loader):
            batch_start = time.perf_counter()
            b = move(batch_cpu, device)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast("cuda", dtype=torch.float16, enabled=bool(tc["amp"])):
                encoded = encode_batch(model, b, epoch, cfg["seed"], False)
                state = model.prepare_phase(encoded, b["phase"], bc=b.get("bc"), phase_chunk_size=tc["phase_chunk"], phase_indices=torch.arange(80, device=device))
                task_losses = {}
                for task in cfg["tasks"]:
                    q = b["queries"][task]
                    pred = model.decode(encoded, task, q["pos"], q["x"], q["batch"],
                                        phase_state=state, phase_chunk_size=tc["phase_chunk"],
                                        query_frame=q["frame"], query_geom=q["geom_extra"])
                    task_losses[task], objective_parts = velocity_objective(pred, q, stats, cfg["arm"],
                        direction_lambda=direction_lambda, q80=tail_q80,
                        direction_floor=tc.get("direction_speed_floor_m_s", .01),
                        direction_epsilon=tc.get("direction_epsilon_m_s", .01))
                total = torch.stack(list(task_losses.values())).mean()
            if not bool(torch.isfinite(total)):
                raise FloatingPointError(f"nonfinite loss {cfg['arm']} epoch={epoch} batch={batch_index}")
            gradient_diagnostics = None
            coverage = None
            if cfg["arm"] == "D1":
                coverage = direction_coverage(q["y_raw"], q["weight"], q["batch"],
                                              tc.get("direction_speed_floor_m_s", .01))
                coverage_rows.append(coverage)
                if batch_index == 0:
                    parameters = tuple(model.decoders["velocity"].parameters())
                    base_norm = float(gradient_l2(objective_parts["z_mse"], parameters, retain_graph=True))
                    direction_norm = float(gradient_l2(objective_parts["direction"], parameters, retain_graph=True))
                    gradient_diagnostics = {
                        "sample": "first_batch_each_epoch", "scope": "original_velocity_decoder",
                        "base_l2": base_norm, "direction_l2": direction_norm,
                        "weighted_direction_l2": direction_lambda * direction_norm,
                        "weighted_direction_to_base": direction_lambda * direction_norm / max(base_norm, 1e-30)}
            scaler.scale(total).backward()
            scaler.unscale_(optimizer)
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), tc["grad_clip"], error_if_nonfinite=True)
            clipped_steps += int(float(grad_norm) > tc["grad_clip"])
            if gradient_diagnostics is not None:
                gradient_rows.append(gradient_diagnostics)
            scale_before = scaler.get_scale()
            scaler.step(optimizer)
            scaler.update()
            if scaler.get_scale() < scale_before:
                raise FloatingPointError("AMP skipped an optimizer update; stop rather than change effective exposure")
            scheduler.step()
            step += 1
            objective_values = {k: float(v.detach()) for k, v in objective_parts.items()}
            with torch.no_grad():
                stage_errors = (pred.detach().float() - q["y"].float()).square().mean(-1)
                stage_values = {name: float(case_phase_mean(stage_errors[:, ids], q["weight"], q["batch"]))
                                for name, ids in SEGMENTS.items()}
            for name, value in objective_values.items():
                objective_sums.setdefault(name, []).append((value, len(b["unit_ids"])))
            for name, value in stage_values.items():
                stage_sums.setdefault(name, []).append((value, len(b["unit_ids"])))
            values = {t: float(v.detach()) for t, v in task_losses.items()}
            for t, v in values.items():
                losses[t].append((v, len(b["unit_ids"])))
            log("train_batch", arm=cfg["arm"], epoch=epoch + 1, step=step,
                batch=batch_index + 1, cases=b["unit_ids"], losses=values, objective=objective_values, phase_z_mse=stage_values,
                grad_norm=float(grad_norm), lr=optimizer.param_groups[0]["lr"],
                amp_scale=scaler.get_scale(), gradient_diagnostics=gradient_diagnostics,
                direction_coverage=coverage,
                clipped=bool(float(grad_norm) > tc["grad_clip"]),
                seconds=round(time.perf_counter() - batch_start, 3))
            if smoke and batch_index >= 2:
                break
        row = {"epoch": epoch + 1, "step": step,
               "losses": {t: sum(v * n for v, n in xs) / sum(n for _, n in xs) for t, xs in losses.items()},
               "objective": {k: sum(v*n for v,n in xs)/sum(n for _,n in xs) for k,xs in objective_sums.items()},
               "phase_z_mse": {k: sum(v*n for v,n in xs)/sum(n for _,n in xs) for k,xs in stage_sums.items()},
               "seconds": time.perf_counter() - epoch_start,
               "gradient_clip_fraction": clipped_steps / (batch_index + 1),
               "peak_cuda_memory_bytes": torch.cuda.max_memory_allocated()}
        if gradient_rows:
            row["gradient_diagnostics"] = gradient_rows[0]
        if coverage_rows:
            count = sum(r["total_case_phases"] for r in coverage_rows)
            row["direction_coverage"] = {
                "eligible_volume_fraction": sum(r["eligible_volume_fraction"] * r["total_case_phases"]
                                                for r in coverage_rows) / count,
                "empty_case_phases": sum(r["empty_case_phases"] for r in coverage_rows),
                "total_case_phases": count}
        with (out / "history.jsonl").open("a") as f:
            f.write(json.dumps(row, allow_nan=False) + "\n")
        histories.append(row)
        log("epoch_complete", **row)
        if smoke or (epoch + 1) % tc["checkpoint_every"] == 0 or epoch + 1 == epochs:
            checkpoint(out, model, optimizer, scheduler, scaler, epoch, step, sha(config_path))
    train_seconds = time.perf_counter() - started
    atomic_json(out / "training_complete.json", {"epochs": epochs, "steps": step,
                "seconds": train_seconds, "checkpoint": str(out / "ckpt_last.pt"), "smoke": smoke})
    # Smoke stays within TRAIN patients; held-out development units are final-only.
    ev_partition = "train" if smoke else cfg["eval"]["partition"]
    eval_ds = VelocityPhaseDataset(partition=ev_partition, training=False,
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
        batch = collate_velocity_phase([record])
        unit = batch["unit_ids"][0]
        log("eval_case_start", arm=cfg["arm"], index=index + 1, total=count, unit=unit, partition=ev_partition)
        b = move(batch, device)
        with torch.inference_mode():
            encoded = encode_batch(model, b, epochs, cfg["seed"], True)
            state = model.prepare_phase(encoded, b["phase"], bc=b.get("bc"), phase_chunk_size=tc["phase_chunk"], phase_indices=torch.arange(80, device=device))
            outputs = {}
            for task in cfg["tasks"]:
                q = b["queries"][task]
                parts = []
                chunk = cfg["eval"]["query_chunk"]
                for left in range(0, len(q["pos"]), chunk):
                    sl = slice(left, left + chunk)
                    pred = model.decode(encoded, task, q["pos"][sl], q["x"][sl], q["batch"][sl],
                                        phase_state=state, phase_chunk_size=tc["phase_chunk"],
                                        query_frame=q["frame"][sl], query_geom=q["geom_extra"][sl])
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
            met = case_metrics(true, raw, weight, q["frame"].numpy(), q["frame_valid"].numpy(),
                               cfg["eval"]["angle_speed_floor_m_s"], tail_spec["q80_m_s"])
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
               "velocity_phase_contract": getattr(model, "velocity_phase_contract", None),
               "direction_lambda": direction_lambda,
               "tail_threshold_scope": "206 training units, unit/volume weighted per-phase q80",
               "geometry_manifest_sha256": sha(Path(cfg["data"]["geometry_root"]) / "manifest.json"),
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
    parser.add_argument("--probe-direction-lambda", action="store_true")
    args = parser.parse_args()
    try:
        if args.probe_direction_lambda:
            probe_direction_lambda(args.config)
        else:
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

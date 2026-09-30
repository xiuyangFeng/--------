#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""配置驱动的 V5.1 时间 Transformer 探针（独立实验路径）。

本文件只读取 P0.10 产生的同折缓存，不修改既有 X5D、T0、时间适配器或评估代码。
每个病例的静态 3D 特征只编码一次：点特征先形成病例级几何 token，时间 token
沿 81 帧做 self-attention，再将时间上下文与点特征解码为逐点 WSS。输出采用
T-null 基底加峰值锚定残差，保证 peak 帧逐点恒等。

这是一个可控的架构探针：当前 V5.1 没有逐帧速度/压力输入，因此 full token
只含静态几何和相位信息；它验证跨帧注意力的增量，不宣称已经恢复病例特异动力学。
"""
from __future__ import annotations

import argparse
import copy
import json
import math
import os
import random
import time
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np
import torch
from torch import nn

from . import time_adapter_probe as TA

SCHEMA = "time_transformer_v51_v1"
N_FRAMES = 81
INPUT_DIMS = {"query_x": 32, "raw": 27, "peak_ln": 1}
DEFAULTS: Dict = {
    "schema": SCHEMA, "name": None, "experiment": None, "arm": None, "fold": None,
    "seed": 1234, "title": "", "cache_dir": None, "split_path": None, "data_root": None,
    "frame_stats_path": None, "waveform_path": None, "out_dir": None, "peak_index": 21,
    "inputs": ["query_x", "raw", "peak_ln"],
    "architecture": {"temporal": "transformer", "d_model": 64, "nhead": 4, "num_layers": 2,
                     "dim_feedforward": 128, "dropout": 0.0, "spatial_hidden": 128,
                     "decoder_hidden": [128, 128], "global_pool": "mean"},
    "train": {"epochs": 60, "steps_per_epoch": 30, "batch_cases": 4, "query_points": 1024,
              "train_pool_points": 8192, "stats_points": 4096, "lr": 1e-3, "weight_decay": 1e-4,
              "warmup_steps": 100, "min_lr_ratio": 0.01, "grad_clip": 1.0,
              "selection": "last", "allow_tf32": True, "log_every_epochs": 5},
    "eval": {"partition": "test", "chunk_points": 4096},
    "notes": "",
}


def merge(defaults: Dict, override: Dict, where: str = "") -> Dict:
    out = copy.deepcopy(defaults)
    for key, value in override.items():
        if key not in defaults:
            raise KeyError(f"unknown config key {where}{key!r}")
        if isinstance(defaults[key], dict):
            if not isinstance(value, dict):
                raise TypeError(f"config key {where}{key!r} must be an object")
            out[key] = merge(defaults[key], value, f"{where}{key}.")
        else:
            out[key] = value
    return out


def load_config(path: str) -> Dict:
    raw = json.loads(Path(path).read_text())
    if raw.get("schema") != SCHEMA:
        raise ValueError(f"{path}: schema must be {SCHEMA!r}")
    cfg = merge(DEFAULTS, raw)
    required = ("name", "experiment", "arm", "fold", "cache_dir", "split_path", "data_root",
                "frame_stats_path", "waveform_path", "out_dir")
    for key in required:
        if cfg[key] is None:
            raise ValueError(f"{path}: missing required key {key!r}")
    if set(cfg["inputs"]) - set(INPUT_DIMS) or not cfg["inputs"]:
        raise ValueError(f"inputs must be a non-empty subset of {tuple(INPUT_DIMS)}")
    if cfg["architecture"]["temporal"] not in ("transformer", "frame_mlp"):
        raise ValueError("architecture.temporal must be transformer or frame_mlp")
    if cfg["architecture"]["global_pool"] != "mean":
        raise ValueError("only global_pool='mean' is registered")
    if cfg["train"]["selection"] != "last":
        raise ValueError("only train.selection='last' is registered")
    return cfg


def _seed_all(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _case_key(uid: str) -> str:
    return uid.replace("/", "__")


def _load_fold(cfg: Dict) -> Dict:
    split = json.loads(Path(cfg["split_path"]).read_text())
    train_ids = list(split["train_cases"])
    test_ids = list(split["test_cases"])
    fs = json.loads(Path(cfg["frame_stats_path"]).read_text())
    if set(fs["train_units"]) != set(train_ids):
        raise ValueError("frame stats are not fitted on exactly this fold's training cases")
    wf = json.loads(Path(cfg["waveform_path"]).read_text())
    if int(wf["peak_index"]) != int(cfg["peak_index"]):
        raise ValueError("waveform peak_index differs from config")
    if list(wf["steps"]) != list(fs["frame"]["steps"]):
        raise ValueError("waveform steps differ from frame stats")
    manifest = json.loads((Path(cfg["cache_dir"]) / "manifest.json").read_text())
    if Path(manifest["config"]["split_path"]).resolve() != Path(cfg["split_path"]).resolve():
        raise ValueError("cache was built for a different split")
    return {"train_ids": train_ids, "test_ids": test_ids, "steps": np.asarray(fs["frame"]["steps"], np.int64),
            "mu": np.asarray(fs["frame"]["log_mean"], np.float32),
            "sd": np.asarray(fs["frame"]["log_std"], np.float32), "floor": float(fs["floor"]),
            "eps": float(fs["eps"]), "phase": np.stack([np.asarray(wf[k], np.float32)
            for k in ("q_norm", "dq_norm", "t_sin", "t_cos")], axis=1),
            "q_norm": np.asarray(wf["q_norm"], np.float32), "peak_index": int(cfg["peak_index"]),
            "manifest": manifest}


def _load_case(cfg: Dict, uid: str, with_label: bool, pool_points: int = 0, rng=None,
               floor: float = 0.05, eps: float = 1e-6) -> Dict:
    path = Path(cfg["cache_dir"]) / f"{_case_key(uid)}.npz"
    with np.load(path, allow_pickle=False) as z:
        n = int(z["peak_ln"].shape[0])
        if pool_points and n > pool_points:
            if rng is None:
                rng = np.random.default_rng(0)
            idx = np.sort(rng.choice(n, size=pool_points, replace=False))
        else:
            idx = np.arange(n, dtype=np.int64)
        out = {"unit_id": uid, "idx": idx}
        for key in ("query_x", "raw", "peak_ln"):
            out[key] = np.asarray(z[key][idx], dtype=np.float32)
    if with_label:
        with np.load(Path(cfg["data_root"]) / uid / "bundle.npz", allow_pickle=False) as z:
            tau = np.asarray(z["wall_wss"][:, idx], dtype=np.float32)
            if tau.shape[0] != N_FRAMES:
                raise ValueError(f"{uid}: expected {N_FRAMES} frames, got {tau.shape}")
        out["ln_y"] = TA.ln_floor(tau, floor, eps).astype(np.float32)
        if not pool_points:
            out["tau_raw"] = tau.astype(np.float32)
    return out


class TemporalFieldModel(nn.Module):
    def __init__(self, input_dim: int, arch: Dict, phase_dim: int = 4):
        super().__init__()
        d = int(arch["d_model"])
        h = int(arch["spatial_hidden"])
        self.temporal_kind = arch["temporal"]
        self.d_model = d
        self.point = nn.Sequential(nn.Linear(input_dim, h), nn.GELU(), nn.Linear(h, d), nn.LayerNorm(d))
        self.phase = nn.Sequential(nn.Linear(phase_dim, d), nn.GELU(), nn.Linear(d, d))
        if self.temporal_kind == "transformer":
            layer = nn.TransformerEncoderLayer(d_model=d, nhead=int(arch["nhead"]),
                dim_feedforward=int(arch["dim_feedforward"]), dropout=float(arch["dropout"]),
                activation="gelu", batch_first=True, norm_first=True)
            self.temporal = nn.TransformerEncoder(layer, num_layers=int(arch["num_layers"]))
        else:
            self.temporal = nn.Sequential(nn.Linear(d, d), nn.GELU(), nn.Linear(d, d), nn.LayerNorm(d))
        dec: List[nn.Module] = []
        last = 2 * d
        for width in arch["decoder_hidden"]:
            dec += [nn.Linear(last, int(width)), nn.GELU()]
            last = int(width)
        dec += [nn.Linear(last, 1)]
        self.decoder = nn.Sequential(*dec)
        nn.init.zeros_(self.decoder[-1].weight)
        nn.init.zeros_(self.decoder[-1].bias)

    def temporal_context(self, point_feat: torch.Tensor, phase: torch.Tensor) -> torch.Tensor:
        # point_feat: B,P,D; attention is factorised over a case-level geometry token and 81 times.
        g = point_feat.mean(dim=1)
        seq = g[:, None, :] + self.phase(phase)[None, :, :]
        if self.temporal_kind == "transformer":
            return self.temporal(seq)
        return self.temporal(seq)

    def decode(self, point_feat: torch.Tensor, z: torch.Tensor, peak_index: int) -> torch.Tensor:
        # Returns anchored residual in B,P,F; point chunks keep memory bounded.
        b, p, d = point_feat.shape
        f = z.shape[1]
        x = torch.cat([point_feat[:, :, None, :].expand(-1, -1, f, -1),
                       z[:, None, :, :].expand(-1, p, -1, -1)], dim=-1)
        r = self.decoder(x.reshape(b * p * f, 2 * d)).reshape(b, p, f)
        return r - r[:, :, peak_index:peak_index + 1]

    def forward(self, x: torch.Tensor, phase: torch.Tensor, peak_index: int) -> torch.Tensor:
        feat = self.point(x)
        z = self.temporal_context(feat, phase)
        return self.decode(feat, z, peak_index)


def _stats(pool: Sequence[Dict], inputs: Sequence[str], max_points: int, seed: int) -> Tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed + 991)
    blocks = []
    for c in pool:
        n = len(c["peak_ln"])
        idx = np.arange(n) if n <= max_points else rng.choice(n, size=max_points, replace=False)
        parts = []
        for key in inputs:
            a = c[key][idx]
            if key == "peak_ln":
                a = a[:, None]
            parts.append(a)
        blocks.append(np.concatenate(parts, axis=1).astype(np.float64))
    arr = np.concatenate(blocks, axis=0)
    mean, std = arr.mean(0).astype(np.float32), arr.std(0).astype(np.float32)
    return mean, np.where(std > 1e-6, std, 1.0).astype(np.float32)


def _input(c: Dict, idx: np.ndarray, inputs: Sequence[str], mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    parts = []
    for key in inputs:
        a = c[key][idx]
        if key == "peak_ln":
            a = a[:, None]
        parts.append(a)
    return (np.concatenate(parts, axis=1) - mean) / std


def _lr(step: int, total: int, warm: int, floor: float) -> float:
    if step < warm:
        return (step + 1) / max(warm, 1)
    p = min(max((step - warm) / max(total - warm, 1), 0.0), 1.0)
    return floor + (1 - floor) * 0.5 * (1 + math.cos(math.pi * p))


def _batch(pool: Sequence[Dict], ids: Sequence[int], inputs: Sequence[str], mean, std, qpoints, rng):
    xs, ys, peaks = [], [], []
    for i in ids:
        c = pool[i]
        idx = rng.choice(len(c["peak_ln"]), size=min(qpoints, len(c["peak_ln"])), replace=False)
        x = _input(c, idx, inputs, mean, std)
        xs.append(x)
        ys.append(c["ln_y"][:, idx].T)
        peaks.append(c["peak_ln"][idx])
    return (torch.from_numpy(np.stack(xs).astype(np.float32)),
            torch.from_numpy(np.stack(ys).astype(np.float32)),
            torch.from_numpy(np.stack(peaks).astype(np.float32)))


@torch.no_grad()
def _predict_case(model, c, phase, mean, std, mu, sd, peak_index, device, chunk):
    n = len(c["peak_ln"])
    # Compute one case-level geometry token over all points, then decode point chunks.
    feats = []
    for s in range(0, n, chunk):
        x = torch.from_numpy(_input(c, np.arange(s, min(n, s + chunk)), model._inputs, mean, std)).to(device)
        feats.append(model.point(x))
    gfeat = torch.cat(feats, 0)
    z = model.temporal_context(gfeat[None, :, :], phase).squeeze(0)
    out = []
    peak = torch.from_numpy(c["peak_ln"]).to(device)
    base = torch.from_numpy(TA.tnull_base(c["peak_ln"], mu, sd, peak_index).astype(np.float32)).to(device)
    for s in range(0, n, chunk):
        ids = np.arange(s, min(n, s + chunk))
        x = torch.from_numpy(_input(c, ids, model._inputs, mean, std)).to(device)
        pf = model.point(x)[None, :, :]
        r = model.decode(pf, z[None, :, :], peak_index).squeeze(0)
        out.append((base[ids] + r).cpu().numpy())
    return np.concatenate(out, axis=0).T


def run(config_path: str) -> Dict:
    cfg = load_config(config_path)
    out_dir = Path(cfg["out_dir"])
    if (out_dir / "metrics.json").exists():
        raise FileExistsError(f"{out_dir}/metrics.json exists; refuse to overwrite")
    out_dir.mkdir(parents=True, exist_ok=True)
    _seed_all(int(cfg["seed"]))
    tc, arch = cfg["train"], cfg["architecture"]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.backends.cuda.matmul.allow_tf32 = bool(tc["allow_tf32"])
    torch.backends.cudnn.allow_tf32 = bool(tc["allow_tf32"])
    fold = _load_fold(cfg)
    rng_pool = np.random.default_rng(int(cfg["seed"]) + 17)
    pool = [_load_case(cfg, uid, True, int(tc["train_pool_points"]), rng_pool,
                       fold["floor"], fold["eps"]) for uid in fold["train_ids"]]
    mean, std = _stats(pool, cfg["inputs"], int(tc["stats_points"]), int(cfg["seed"]))
    input_dim = sum(INPUT_DIMS[k] for k in cfg["inputs"])
    model = TemporalFieldModel(input_dim, arch).to(device)
    model._inputs = list(cfg["inputs"])
    phase = torch.from_numpy(fold["phase"]).to(device)
    mu, sd = fold["mu"], fold["sd"]
    opt = torch.optim.AdamW(model.parameters(), lr=float(tc["lr"]), weight_decay=float(tc["weight_decay"]))
    total = int(tc["epochs"]) * int(tc["steps_per_epoch"])
    step = 0
    history = []
    t0 = time.perf_counter()
    for epoch in range(int(tc["epochs"])):
        model.train()
        ep_loss = 0.0
        for _ in range(int(tc["steps_per_epoch"])):
            ids = np.random.default_rng(int(cfg["seed"]) + step * 7919).choice(len(pool), int(tc["batch_cases"]), replace=False)
            xb, yb, peakb = _batch(pool, ids, cfg["inputs"], mean, std, int(tc["query_points"]),
                                   np.random.default_rng(int(cfg["seed"]) + step * 3571 + 7))
            xb, yb, peakb = xb.to(device), yb.to(device), peakb.to(device)
            r = model(xb, phase, int(cfg["peak_index"]))
            peak_np = peakb.detach().cpu().numpy()
            base_np = mu[None, None, :] + sd[None, None, :] * ((peak_np - mu[int(cfg["peak_index"])]) / sd[int(cfg["peak_index"])])[:, :, None]
            base = torch.from_numpy(base_np.astype(np.float32)).to(device)
            pred = base + r
            loss = (((pred - yb) / torch.from_numpy(sd).to(device)[None, None, :]) ** 2).mean()
            opt.zero_grad(set_to_none=True)
            loss.backward()
            grad = float(torch.nn.utils.clip_grad_norm_(model.parameters(), float(tc["grad_clip"])))
            scale = _lr(step, total, int(tc["warmup_steps"]), float(tc["min_lr_ratio"]))
            for group in opt.param_groups:
                group["lr"] = float(tc["lr"]) * scale
            opt.step()
            ep_loss += float(loss.detach())
            step += 1
        rec = {"epoch": epoch, "loss_z": ep_loss / int(tc["steps_per_epoch"]), "step": step,
               "lr": opt.param_groups[0]["lr"], "grad_norm_last": grad,
               "elapsed_s": time.perf_counter() - t0}
        history.append(rec)
        if epoch % int(tc["log_every_epochs"]) == 0 or epoch == int(tc["epochs"]) - 1:
            print(f"[tt] epoch {epoch} loss_z {rec['loss_z']:.5f} lr {rec['lr']:.2e} "
                  f"elapsed {rec['elapsed_s'] / 60:.1f} min", flush=True)
    with (out_dir / "history.jsonl").open("w") as fh:
        for rec in history:
            fh.write(json.dumps(rec) + "\n")
    torch.save({"state_dict": model.state_dict(), "input_mean": mean, "input_std": std, "config": cfg}, out_dir / "ckpt_last.pt")

    model.eval()
    true, pred, anchor_devs = [], [], []
    eval_cases = [_load_case(cfg, uid, True, 0, None, fold["floor"], fold["eps"]) for uid in fold["test_ids"]]
    for i, c in enumerate(eval_cases):
        lp = _predict_case(model, c, phase, mean, std, mu, sd, int(cfg["peak_index"]), device, int(cfg["eval"]["chunk_points"]))
        true.append(c["tau_raw"])
        pred.append(lp)
        anchor_devs.append(float(np.max(np.abs(lp[int(cfg["peak_index"])] - c["peak_ln"]))))
        print(f"[tt] eval {i + 1}/{len(eval_cases)} {c['unit_id']} n={len(c['peak_ln'])}", flush=True)
    report = TA.cycle_report(true, pred, fold["steps"],
                             int(cfg["peak_index"]), fold["q_norm"], fold["floor"], fold["eps"])
    # cycle_report expects frame steps only for length; cache manifest has no steps, use 0..80.
    result = {"schema": SCHEMA, "name": cfg["name"], "arm": cfg["arm"], "fold": cfg["fold"],
              "seed": cfg["seed"], "partition": cfg["eval"]["partition"], "n_cases": len(eval_cases),
              "config": cfg, "config_sha256": TA.sha256_file(config_path),
              "cache_manifest_sha256": TA.sha256_file(Path(cfg["cache_dir"]) / "manifest.json"),
              "frame_stats_sha256": TA.sha256_file(cfg["frame_stats_path"]),
              "split_sha256": TA.sha256_file(cfg["split_path"]),
              "code_sha256": {"time_transformer.py": TA.sha256_file(Path(__file__))},
              "input_mean": mean.tolist(), "input_std": std.tolist(),
              "train_pool_points": int(tc["train_pool_points"]), "steps": step,
              "final_loss_z": history[-1]["loss_z"], "train_seconds": time.perf_counter() - t0,
              "anchor_check": {"max_abs_peak_ln_vs_anchor": float(max(anchor_devs)),
                               "definition": "max |predicted log peak frame - cached donor peak log| over held-out points"},
              "metrics": report, "execution": TA.execution_record()}
    TA.write_json(out_dir / "metrics.json", result)
    ec = report["eval_compatible"]
    print(f"[{cfg['name']}] cycle {ec['cycle_r2cb_pa']:.4f} trough {ec['trough_r2cb_pa']:.4f} "
          f"TAWSS {ec['tawss_r2cb_pa']:.4f} | train {result['train_seconds'] / 60:.1f} min", flush=True)
    return result


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    run(ap.parse_args().config)


if __name__ == "__main__":
    main()

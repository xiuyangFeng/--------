#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""峰值预训练 → 时间适配器（导师方案，矩阵 P0.8/P0.9/P0.10）的冻结表示探针。

独立工具、完全配置驱动；不修改任何既有训练/评估代码路径（旧配置逐位不变）。

cache：用同折冻结的 X5D 峰值 checkpoint（eval 模式；fixed_support + full_cloud_query，
       与 evaluate.predict_case_norm 同一前向、同一 support 种子）逐病例缓存
       - query_x：FP 解码 + 局部壁面分支后插值到 query 点、进入 head 之前的逐点特征；
       - patch_context：最粗层 max 池化的病例向量；
       - peak_ln：峰值帧预测 ln(max(τ̂, floor) + eps)（含 query-patch FiLM 项，= 冻结峰值锚）；
       - raw：27 维标准化输入（对照臂用）。
fit：  在折训练病例上拟合锚定时间头
         ln τ̂(x,t) = L0(x,t) + R(h(x), φ(t)) − R(h(x), φ(t_peak))
         L0 = T-null(B_scale) = μ(t) + σ(t)·(peak_ln − μ(t_peak)) / σ(t_peak)，μ/σ = 训练折逐帧统计；
       R ∈ {none（= T-null 复现）, ridge（逐帧线性读出）, mlp（几何投影 + 独立时间 encoder → 融合 → decoder）}；
       损失 = frame_stats z 空间 MSE（帧权均匀、病例等点）；在留出折 81 帧上评估，
       口径 = evaluate._cycle_metrics（与阶段 2 T0 同）+ offline/common.TimeMetrics（与 D2 T-null 同）。

用法：
  python -m training_wss_min.time_adapter_probe cache --config configs/<exp>/cache_f0.json
  python -m training_wss_min.time_adapter_probe fit   --config configs/<exp>/<arm>_f0_s1234.json
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import os
import platform
import socket
import time
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np
import torch
from torch import nn

from . import config as C
from . import dataset as D
from . import evaluate as E
from . import metrics as M

CACHE_SCHEMA = "time_adapter_cache_v1"
PROBE_SCHEMA = "time_adapter_probe_v1"
HEAD_TYPES = ("none", "ridge", "mlp")
POINT_INPUTS = ("query_x", "peak_ln", "raw")
CASE_INPUTS = ("patch_context",)
PHASE_KEYS = ("q_norm", "dq_norm", "t_sin", "t_cos")
N_FRAMES = 81
CODE_FILES = ("time_adapter_probe.py", "evaluate.py", "dataset.py", "metrics.py", "config.py",
              "baseline_models.py", "models.py", "local_refinement.py", "longitudinal_geometry.py",
              "next_geometry.py", "pointnext.py", "surface.py", "runtime.py")

PROBE_DEFAULTS: Dict = {
    "schema": PROBE_SCHEMA,
    "name": None, "experiment": None, "arm": None, "fold": None, "seed": 1234, "title": "",
    "cache_dir": None, "split_path": None, "data_root": None, "frame_stats_path": None, "waveform_path": None,
    "out_dir": None, "peak_index": 21,
    "phase_features": list(PHASE_KEYS),
    "head": {"type": "mlp", "inputs": ["query_x", "patch_context", "peak_ln"], "anchor": True,
             "geom_hidden": 128, "time_hidden": 64, "decoder_hidden": [128, 128], "zero_init_output": True},
    "train": {"steps": 4000, "cases_per_step": 8, "points_per_case": 2048, "lr": 1e-3, "weight_decay": 1e-4,
              "warmup_steps": 200, "grad_clip": 1.0, "selection": "last", "log_every": 100},
    "ridge": {"points_per_case": 8192, "lambdas_rel": [1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1.0, 10.0], "cv_folds": 5},
    "eval": {"partition": "test", "chunk_points": 8192, "save_predictions": False},
    "notes": "",
}
CACHE_KEYS = {"schema", "name", "donor_run", "checkpoint", "split_path", "partitions", "out_dir",
              "reference_predictions", "notes"}


# ---------------------------------------------------------------- utilities
def sha256_file(path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def code_fingerprint() -> Dict[str, str]:
    here = Path(__file__).resolve().parent
    return {name: sha256_file(here / name) for name in CODE_FILES if (here / name).is_file()}


def case_key(unit_id: str) -> str:
    if ".." in unit_id or unit_id.startswith("/"):
        raise ValueError(f"unsafe unit id {unit_id!r}")
    return unit_id.replace("/", "__")


def execution_record() -> Dict:
    rec = {"hostname": socket.gethostname(), "python": platform.python_version(), "torch": torch.__version__,
           "slurm_job_id": os.environ.get("SLURM_JOB_ID"), "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
           "executed_outside_slurm": os.environ.get("SLURM_JOB_ID") is None, "started_at": time.strftime("%Y-%m-%dT%H:%M:%S%z")}
    if torch.cuda.is_available():
        rec["gpu"] = torch.cuda.get_device_name(0)
    return rec


def write_json(path, obj) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(obj, indent=1, ensure_ascii=False,
                                     default=lambda o: o.tolist() if hasattr(o, "tolist") else str(o)))


def _merge(defaults: Dict, override: Dict, where: str = "") -> Dict:
    out = copy.deepcopy(defaults)
    for key, value in override.items():
        if key not in defaults:
            raise KeyError(f"unknown config key {where}{key!r}")
        if isinstance(defaults[key], dict):
            if not isinstance(value, dict):
                raise TypeError(f"config key {where}{key!r} must be an object")
            out[key] = _merge(defaults[key], value, f"{where}{key}.")
        else:
            out[key] = value
    return out


def load_probe_config(path) -> Dict:
    raw = json.loads(Path(path).read_text())
    if raw.get("schema") != PROBE_SCHEMA:
        raise ValueError(f"{path}: schema must be {PROBE_SCHEMA!r}")
    cfg = _merge(PROBE_DEFAULTS, raw)
    for key in ("name", "arm", "fold", "cache_dir", "split_path", "data_root", "frame_stats_path", "waveform_path", "out_dir"):
        if cfg[key] is None:
            raise ValueError(f"{path}: missing required key {key!r}")
    head = cfg["head"]
    if head["type"] not in HEAD_TYPES:
        raise ValueError(f"head.type must be one of {HEAD_TYPES}")
    unknown = set(head["inputs"]) - set(POINT_INPUTS) - set(CASE_INPUTS)
    if unknown:
        raise ValueError(f"unknown head.inputs {sorted(unknown)}")
    if head["type"] != "none" and not head["inputs"]:
        raise ValueError("a trained head needs at least one input")
    if not head["anchor"]:
        raise ValueError("this probe only supports anchored heads (peak identity by construction)")
    if set(cfg["phase_features"]) - set(PHASE_KEYS):
        raise ValueError(f"phase_features must be drawn from {PHASE_KEYS}")
    if cfg["train"]["selection"] != "last":
        raise ValueError("only train.selection='last' is pre-registered")
    return cfg


def load_cache_config(path) -> Dict:
    cfg = json.loads(Path(path).read_text())
    if cfg.get("schema") != CACHE_SCHEMA:
        raise ValueError(f"{path}: schema must be {CACHE_SCHEMA!r}")
    unknown = set(cfg) - CACHE_KEYS
    if unknown:
        raise KeyError(f"unknown cache config keys {sorted(unknown)}")
    cfg.setdefault("checkpoint", "best")
    cfg.setdefault("partitions", ["train", "test"])
    cfg.setdefault("reference_predictions", None)
    return cfg


def ln_floor(tau, floor: float, eps: float) -> np.ndarray:
    return np.log(np.clip(np.asarray(tau, dtype=np.float64), floor, None) + eps)


# ---------------------------------------------------------------- cache
def run_cache(config_path) -> Dict:
    cfg = load_cache_config(config_path)
    donor = Path(cfg["donor_run"])
    out = Path(cfg["out_dir"])
    out.mkdir(parents=True, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dcfg, feat_stats, model, _ = E.load_model_from_run(donor, device, cfg["checkpoint"])
    model.eval()
    if not (dcfg.eval.fixed_support and dcfg.eval.full_cloud_query):
        raise ValueError("donor must evaluate with fixed_support + full_cloud_query")
    if dcfg.data.target != "wss" or dcfg.data.timesteps != "peak" or dcfg.data.target_normalization != "global_stats":
        raise ValueError("donor must be a peak-frame scalar WSS model with global_stats normalisation")
    if Path(cfg["split_path"]).resolve() != Path(dcfg.data.split_path).resolve():
        raise ValueError("cache split_path must be the donor's training split (fold-matched donor)")
    wss_stats = E.load_wss_stats_for_run(donor)
    eps = float(wss_stats["eps"])
    floor = 0.05

    captured: Dict = {}
    orig_encode, orig_head = model.encode_support, model._predict_head

    def encode_hook(*args, **kwargs):
        encoded = orig_encode(*args, **kwargs)
        if len(encoded) < 6 or encoded[5] is None:
            raise RuntimeError("donor has no query-patch coarse context (encoded[5])")
        captured["patch_context"] = encoded[5].detach().float().cpu()
        return encoded

    def head_hook(x, input_x=None):
        captured.setdefault("query_x", []).append(x.detach().float().cpu())
        return orig_head(x, input_x)

    model.encode_support = encode_hook      # instance-level wrappers only; the class and checkpoint stay untouched
    model._predict_head = head_hook
    record = {"schema": CACHE_SCHEMA, "config": cfg, "config_sha256": sha256_file(config_path),
              "donor_checkpoint_sha256": sha256_file(donor / E.checkpoint_filename(cfg["checkpoint"])),
              "donor_config_sha256": sha256_file(donor / "config.json"),
              "split_sha256": sha256_file(cfg["split_path"]), "code_sha256": code_fingerprint(),
              "execution": execution_record(), "floor": floor, "eps": eps, "cases": {}, "reference_check": {}}
    t0 = time.perf_counter()
    for part in cfg["partitions"]:        # 数值后端保持 evaluate.py 的默认设置，便于与已存峰值预测对账
        cases = D.load_partition(
            cfg["split_path"], part, wss_stats, target=dcfg.data.target,
            target_normalization=dcfg.data.target_normalization, data_root=dcfg.data.data_root,
            required_frame_version=dcfg.data.required_frame_version,
            case_features_path=dcfg.data.case_features_path, timesteps="peak", waveform_path=None,
            extra_point_features=C.v6_point_features(dcfg),
            point_features_root=getattr(dcfg.data, "point_features_root", None))
        ref_dev = []
        for case in cases:
            uid = case["unit_id"]
            captured.clear()
            pred_norm = np.asarray(E.predict_case_norm(model, case, dcfg.data.input_features, feat_stats, device, cfg=dcfg),
                                   dtype=np.float64)
            rows = D.query_rows(case)
            if not np.array_equal(rows, np.arange(len(case["pos"]))):
                raise RuntimeError(f"{uid}: full-cloud query rows are not the identity")
            query_x = torch.cat(captured["query_x"]).numpy()
            if query_x.shape[0] != len(rows):
                raise RuntimeError(f"{uid}: captured query_x rows {query_x.shape[0]} != {len(rows)}")
            patch_context = captured["patch_context"].numpy().reshape(-1)
            raw = D.build_features(case, rows, dcfg.data.input_features, feat_stats).astype(np.float32)
            pred_pa = np.clip(D.denormalize_wss(pred_norm, wss_stats), 0, None)
            peak_ln = ln_floor(pred_pa, floor, eps)
            with np.load(Path(dcfg.data.data_root) / uid / "bundle.npz", allow_pickle=False) as z:
                wall = z["wall_wss"]
                steps = z["steps"].astype(np.int64)
                peak_index = int(np.flatnonzero(steps == int(z["peak_step"]))[0])
            if wall.shape != (N_FRAMES, len(rows)):
                raise RuntimeError(f"{uid}: wall_wss shape {wall.shape} does not match {len(rows)} query rows")
            if not np.allclose(wall[peak_index], np.asarray(case["y_raw"]), rtol=0, atol=1e-6):
                raise RuntimeError(f"{uid}: bundle peak frame does not match loaded y_raw")
            np.savez(out / f"{case_key(uid)}.npz", unit_id=np.array(uid), partition=np.array(part),
                     pred_norm=pred_norm.astype(np.float32), pred_pa=pred_pa, peak_ln=peak_ln,
                     query_x=query_x.astype(np.float32), patch_context=patch_context.astype(np.float32), raw=raw,
                     peak_index=np.array(peak_index))
            entry = {"partition": part, "n_points": int(len(rows)), "query_x_dim": int(query_x.shape[1]),
                     "patch_context_dim": int(patch_context.shape[0])}
            if part == "test" and cfg.get("reference_predictions"):
                ref = Path(cfg["reference_predictions"]) / uid / "predictions.npz"
                with np.load(ref) as z:
                    ref_pa = z["pred_pa"].astype(np.float64)
                    if not np.array_equal(z["row_index"], rows):
                        raise RuntimeError(f"{uid}: reference prediction rows differ")
                d_ln = float(np.max(np.abs(ln_floor(ref_pa, floor, eps) - peak_ln)))
                entry["reference_max_abs_dln"] = d_ln
                ref_dev.append(d_ln)
            record["cases"][uid] = entry
            print(f"[cache] {part} {uid} n={len(rows)} qx={query_x.shape[1]} pc={patch_context.shape[0]}", flush=True)
        if ref_dev:
            record["reference_check"][part] = {"n_cases": len(ref_dev), "max_abs_dln": float(max(ref_dev)),
                                               "median_max_abs_dln": float(np.median(ref_dev))}
    record["elapsed_seconds"] = time.perf_counter() - t0
    write_json(out / "manifest.json", record)
    print(json.dumps(record["reference_check"], indent=1), flush=True)
    return record


# ---------------------------------------------------------------- head
class TimeAdapterHead(nn.Module):
    """导师草图：几何投影 g(h) + 独立时间 encoder e(φ) → 融合（加性，逐 (点, 帧)）→ decoder → R(h, φ)。"""

    def __init__(self, geom_dim: int, phase_dim: int, geom_hidden: int = 128, time_hidden: int = 64,
                 decoder_hidden: Sequence[int] = (128, 128), zero_init_output: bool = True):
        super().__init__()
        decoder_hidden = list(decoder_hidden)
        self.geom = nn.Sequential(nn.Linear(geom_dim, geom_hidden), nn.GELU(), nn.Linear(geom_hidden, geom_hidden))
        self.time = nn.Sequential(nn.Linear(phase_dim, time_hidden), nn.GELU(), nn.Linear(time_hidden, time_hidden))
        self.fuse_geom = nn.Linear(geom_hidden, decoder_hidden[0])
        self.fuse_time = nn.Linear(time_hidden, decoder_hidden[0], bias=False)
        layers: List[nn.Module] = []
        for a, b in zip(decoder_hidden[:-1], decoder_hidden[1:]):
            layers += [nn.GELU(), nn.Linear(a, b)]
        layers += [nn.GELU(), nn.Linear(decoder_hidden[-1], 1)]
        self.decoder = nn.Sequential(*layers)
        if zero_init_output:
            nn.init.zeros_(self.decoder[-1].weight)
            nn.init.zeros_(self.decoder[-1].bias)

    def forward(self, geom: torch.Tensor, phase: torch.Tensor) -> torch.Tensor:
        g = self.fuse_geom(self.geom(geom))              # (P, H)
        e = self.fuse_time(self.time(phase))             # (F, H)
        return self.decoder(g[:, None, :] + e[None, :, :]).squeeze(-1)   # (P, F)


def anchored(r: torch.Tensor, peak_index: int) -> torch.Tensor:
    """R(h, φ(t)) − R(h, φ(t_peak))：峰值帧恒为 0（逐位）。"""
    return r - r[:, peak_index:peak_index + 1]


def tnull_base(peak_ln, mu, sd, peak_index: int):
    """T-null(B_scale)：L0(x,t) = μ(t) + σ(t)·(L_peak(x) − μ(t_peak))/σ(t_peak)；支持 numpy 或 torch。"""
    z = (peak_ln - mu[peak_index]) / sd[peak_index]
    return mu[None, :] + sd[None, :] * z[:, None]


# ---------------------------------------------------------------- data for fit
class FoldData:
    """一折的缓存特征 + 81 帧标签 + 逐帧统计；输入标准化统计只用训练病例。"""

    def __init__(self, cfg: Dict):
        self.cfg = cfg
        split = json.loads(Path(cfg["split_path"]).read_text())
        self.train_ids = [D.canonical_unit_id(u) for u in split["train_cases"]]
        self.test_ids = [D.canonical_unit_id(u) for u in split["test_cases"]]
        fs = json.loads(Path(cfg["frame_stats_path"]).read_text())
        if set(fs["train_units"]) != set(self.train_ids):
            raise ValueError("frame stats were not fitted on exactly this fold's training cases")
        self.floor, self.eps = float(fs["floor"]), float(fs["eps"])
        self.mu = np.asarray(fs["frame"]["log_mean"], dtype=np.float64)
        self.sd = np.asarray(fs["frame"]["log_std"], dtype=np.float64)
        self.steps = np.asarray(fs["frame"]["steps"], dtype=np.int64)
        wf = json.loads(Path(cfg["waveform_path"]).read_text())
        if list(wf["steps"]) != self.steps.tolist():
            raise ValueError("waveform steps differ from frame-stats steps")
        self.q_norm = np.asarray(wf["q_norm"], dtype=np.float64)
        self.phase = np.stack([np.asarray(wf[k], dtype=np.float64) for k in cfg["phase_features"]], axis=1)
        self.peak_index = int(cfg["peak_index"])
        if int(wf["peak_index"]) != self.peak_index:
            raise ValueError("waveform peak_index differs from config")
        manifest = json.loads((Path(cfg["cache_dir"]) / "manifest.json").read_text())
        if Path(manifest["config"]["split_path"]).resolve() != Path(cfg["split_path"]).resolve():
            raise ValueError("cache was built for a different split")
        self.cache_manifest = manifest
        if manifest["floor"] != self.floor or manifest["eps"] != self.eps:
            raise ValueError("cache floor/eps differ from frame stats")

    def load_case(self, uid: str, with_labels: bool = True) -> Dict:
        with np.load(Path(self.cfg["cache_dir"]) / f"{case_key(uid)}.npz", allow_pickle=False) as z:
            if str(z["unit_id"]) != uid or int(z["peak_index"]) != self.peak_index:
                raise ValueError(f"{uid}: cache identity mismatch")
            case = {k: z[k] for k in ("peak_ln", "query_x", "patch_context", "raw")}
        if with_labels:
            with np.load(Path(self.cfg["data_root"]) / uid / "bundle.npz", allow_pickle=False) as z:
                if not np.array_equal(z["steps"].astype(np.int64), self.steps):
                    raise ValueError(f"{uid}: bundle steps differ")
                case["tau"] = z["wall_wss"].astype(np.float64)
            if case["tau"].shape != (N_FRAMES, len(case["peak_ln"])):
                raise ValueError(f"{uid}: label/cache point count mismatch")
        return case


class InputBuilder:
    """按 head.inputs 拼几何输入；逐点量与病例量分开存，按训练病例标准化。"""

    def __init__(self, inputs: Sequence[str], train_cases: Sequence[Dict]):
        self.inputs = list(inputs)
        self.stats = {}
        for key in self.inputs:
            if key == "patch_context":
                arr = np.stack([c["patch_context"] for c in train_cases]).astype(np.float64)
            else:
                arr = np.concatenate([np.asarray(c[key], dtype=np.float64).reshape(len(c["peak_ln"]), -1) for c in train_cases])
            mean, std = arr.mean(axis=0), arr.std(axis=0)
            self.stats[key] = (mean, np.where(std > 1e-8, std, 1.0))
        self.dim = sum(len(self.stats[k][0]) for k in self.inputs)

    def point_block(self, case: Dict) -> np.ndarray:
        n = len(case["peak_ln"])
        parts = []
        for key in self.inputs:
            mean, std = self.stats[key]
            if key == "patch_context":
                parts.append(np.broadcast_to(((case[key] - mean) / std)[None, :], (n, len(mean))))
            else:
                parts.append((np.asarray(case[key], dtype=np.float64).reshape(n, -1) - mean) / std)
        return np.concatenate(parts, axis=1).astype(np.float32) if parts else np.zeros((n, 0), np.float32)

    def state(self) -> Dict:
        return {k: {"mean": m.tolist(), "std": s.tolist()} for k, (m, s) in self.stats.items()}


# ---------------------------------------------------------------- metrics
class _CaseBalanced:
    """复刻 offline/common.CaseBalanced（D2 T-null 口径）：可对 (81,N) 帧向量逐帧累计。"""

    def __init__(self):
        self.mean, self.within, self.mse = [], [], []

    def add(self, yt, yp):
        m = yt.mean(axis=-1)
        self.mean.append(m)
        self.within.append(((yt - m[..., None]) ** 2).mean(axis=-1))
        self.mse.append(((yt - yp) ** 2).mean(axis=-1))

    def r2(self):
        mean, within, mse = np.asarray(self.mean), np.asarray(self.within), np.asarray(self.mse)
        g = mean.mean(axis=0)
        return 1.0 - mse.mean(axis=0) / (within + (mean - g) ** 2).mean(axis=0)


def cycle_report(tau_true: Sequence[np.ndarray], ln_pred: Sequence[np.ndarray], steps, peak_index: int,
                 q_norm, floor: float, eps: float) -> Dict:
    """eval_compatible = evaluate._cycle_metrics + TAWSS（真值原始 Pa、预测 exp(ln)−eps 截 ≥0；阶段 2 T0 口径）；
    tnull_compatible = offline/common.TimeMetrics（真值地板化 Pa；D2 T-null 口径）。"""
    frames = list(range(len(steps)))
    pred_pa = [np.clip(np.exp(lp) - eps, 0, None) for lp in ln_pred]
    true_pa_d = {k: [t[k] for t in tau_true] for k in frames}
    pred_pa_d = {k: [p[k] for p in pred_pa] for k in frames}
    stub = [{"waveform": {"_q_norm": np.asarray(q_norm)}} for _ in tau_true]
    cyc = E._cycle_metrics(stub, frames, np.asarray(steps), peak_index, true_pa_d, pred_pa_d, {"floor": floor, "eps": eps})
    ta = M.casebalanced_field_metrics([t.mean(axis=0) for t in tau_true], [p.mean(axis=0) for p in pred_pa])
    cyc["tawss_r2cb_pa"], cyc["tawss_mae_pa"] = float(ta["r2"]), float(ta["mae"])
    # D2 口径
    trough = np.sort(np.argsort(np.asarray(q_norm))[:20])
    fr, ta2 = _CaseBalanced(), _CaseBalanced()
    for t, lp in zip(tau_true, ln_pred):
        tt = np.exp(ln_floor(t, floor, eps)) - eps
        tp = np.exp(lp) - eps
        fr.add(tt, tp)
        ta2.add(tt.mean(axis=0), tp.mean(axis=0))
    r2f = fr.r2()
    tn = {"cycle_r2cb_pa": float(r2f.mean()), "peak_r2cb_pa": float(r2f[peak_index]),
          "trough_r2cb_pa": float(r2f[trough].mean()), "tawss_r2cb": float(ta2.r2())}
    return {"eval_compatible": cyc, "tnull_compatible": tn}


# ---------------------------------------------------------------- fit
def _lr_lambda(train_cfg):
    steps, warm = int(train_cfg["steps"]), int(train_cfg["warmup_steps"])
    return lambda s: min(1.0, (s + 1) / max(warm, 1)) * 0.5 * (1.0 + math.cos(math.pi * min(s, steps) / steps))


def _fit_mlp(cfg, data: FoldData, builder: InputBuilder, train_cases, device, out_dir: Path):
    tc, hc = cfg["train"], cfg["head"]
    gen = torch.Generator(device=device).manual_seed(int(cfg["seed"]))
    torch.manual_seed(int(cfg["seed"]))
    feats = torch.from_numpy(np.concatenate([builder.point_block(c) for c in train_cases])).to(device)
    y_ln = torch.from_numpy(np.concatenate([ln_floor(c["tau"], data.floor, data.eps).T for c in train_cases]).astype(np.float32)).to(device)
    peak = torch.from_numpy(np.concatenate([c["peak_ln"] for c in train_cases]).astype(np.float32)).to(device)
    counts = torch.tensor([len(c["peak_ln"]) for c in train_cases], device=device)
    offsets = torch.cumsum(counts, 0) - counts
    mu = torch.tensor(data.mu, dtype=torch.float32, device=device)
    sd = torch.tensor(data.sd, dtype=torch.float32, device=device)
    phase = torch.tensor(data.phase, dtype=torch.float32, device=device)
    head = TimeAdapterHead(builder.dim, phase.shape[1], hc["geom_hidden"], hc["time_hidden"], hc["decoder_hidden"],
                           hc["zero_init_output"]).to(device)
    opt = torch.optim.AdamW(head.parameters(), lr=float(tc["lr"]), weight_decay=float(tc["weight_decay"]))
    sched = torch.optim.lr_scheduler.LambdaLR(opt, _lr_lambda(tc))
    n_cases, per_step, n_pts = len(train_cases), int(tc["cases_per_step"]), int(tc["points_per_case"])
    order = torch.randperm(n_cases, generator=gen, device=device)
    cursor, history = 0, []
    t0 = time.perf_counter()
    for step in range(int(tc["steps"])):
        if cursor + per_step > n_cases:
            order, cursor = torch.randperm(n_cases, generator=gen, device=device), 0
        chosen = order[cursor:cursor + per_step]
        cursor += per_step
        u = torch.rand((per_step, n_pts), generator=gen, device=device)
        idx = (offsets[chosen][:, None] + (u * counts[chosen][:, None]).long()).reshape(-1)
        base = tnull_base(peak[idx], mu, sd, data.peak_index)
        pred = base + anchored(head(feats[idx], phase), data.peak_index)
        loss = (((pred - y_ln[idx]) / sd[None, :]) ** 2).mean()
        opt.zero_grad(set_to_none=True)
        loss.backward()
        grad_norm = float(torch.nn.utils.clip_grad_norm_(head.parameters(), float(tc["grad_clip"])))
        opt.step()
        sched.step()
        if step % int(tc["log_every"]) == 0 or step == int(tc["steps"]) - 1:
            rec = {"step": step, "loss_z": float(loss), "lr": float(sched.get_last_lr()[0]), "grad_norm": grad_norm,
                   "elapsed_s": time.perf_counter() - t0}
            history.append(rec)
            print(f"[fit] step {step} loss_z {rec['loss_z']:.5f} lr {rec['lr']:.2e} gn {grad_norm:.3f}", flush=True)
    with (out_dir / "history.jsonl").open("w") as fh:
        for rec in history:
            fh.write(json.dumps(rec) + "\n")
    torch.save({"head": head.state_dict(), "inputs": builder.inputs, "input_stats": builder.state(),
                "phase_features": cfg["phase_features"], "config": cfg}, out_dir / "head_last.pt")
    head.eval()

    @torch.no_grad()
    def predict_delta(block: np.ndarray, chunk: int) -> np.ndarray:
        outs = []
        for s in range(0, len(block), chunk):
            g = torch.from_numpy(block[s:s + chunk]).to(device)
            outs.append(anchored(head(g, phase), data.peak_index).double().cpu().numpy())
        return np.concatenate(outs).T                     # (F, N)

    return predict_delta, {"train_seconds": time.perf_counter() - t0, "final_loss_z": history[-1]["loss_z"],
                           "n_parameters": int(sum(p.numel() for p in head.parameters()))}


def _fit_ridge(cfg, data: FoldData, builder: InputBuilder, train_cases, device, out_dir: Path):
    rc = cfg["ridge"]
    rng = np.random.default_rng(int(cfg["seed"]))
    X_parts, Y_parts, owner = [], [], []
    for i, c in enumerate(train_cases):
        n = len(c["peak_ln"])
        take = rng.choice(n, size=min(int(rc["points_per_case"]), n), replace=False)
        block = builder.point_block(c)[take].astype(np.float64)
        base = tnull_base(c["peak_ln"][take], data.mu, data.sd, data.peak_index)
        y = (ln_floor(c["tau"][:, take], data.floor, data.eps).T - base) / data.sd[None, :]   # z 残差
        X_parts.append(block); Y_parts.append(y); owner.append(np.full(len(take), i))
    X = torch.from_numpy(np.concatenate(X_parts)).to(device)
    Y = torch.from_numpy(np.concatenate(Y_parts)).to(device)
    owner = np.concatenate(owner)
    fit_frames = [k for k in range(N_FRAMES) if k != data.peak_index]
    lambdas = [float(v) for v in rc["lambdas_rel"]]

    def solve(Xa, Ya):
        xm, ym = Xa.mean(0), Ya.mean(0)
        Xc, Yc = Xa - xm, Ya - ym
        G, B = Xc.T @ Xc, Xc.T @ Yc
        evals, evecs = torch.linalg.eigh(G)
        VB = evecs.T @ B
        return xm, ym, evals, evecs, VB, len(Xa)

    def weights(parts, lam_rel):
        xm, ym, evals, evecs, VB, n = parts
        return evecs @ (VB / (evals + lam_rel * n)[:, None]), xm, ym

    perm = rng.permutation(len(train_cases))
    groups = np.array_split(perm, int(rc["cv_folds"]))
    cv = np.zeros((len(groups), len(lambdas)))
    for gi, grp in enumerate(groups):
        va = torch.from_numpy(np.isin(owner, grp)).to(device)
        parts = solve(X[~va], Y[~va])
        for li, lam in enumerate(lambdas):
            W, xm, ym = weights(parts, lam)
            pred = (X[va] - xm) @ W + ym
            cv[gi, li] = float(((pred - Y[va])[:, fit_frames] ** 2).mean())
    best = int(np.argmin(cv.mean(axis=0)))
    W, xm, ym = weights(solve(X, Y), lambdas[best])
    W[:, data.peak_index] = 0.0
    ym = ym.clone(); ym[data.peak_index] = 0.0
    torch.save({"W": W.cpu(), "x_mean": xm.cpu(), "y_mean": ym.cpu(), "lambda_rel": lambdas[best],
                "cv_mse_z": cv.tolist(), "lambdas_rel": lambdas, "inputs": builder.inputs,
                "input_stats": builder.state(), "config": cfg}, out_dir / "ridge.pt")
    sd = torch.tensor(data.sd, dtype=torch.float64, device=device)

    @torch.no_grad()
    def predict_delta(block: np.ndarray, chunk: int) -> np.ndarray:
        g = torch.from_numpy(block.astype(np.float64)).to(device)
        return (((g - xm) @ W + ym) * sd[None, :]).cpu().numpy().T      # z → ln；峰值帧列 = 0

    return predict_delta, {"lambda_rel": lambdas[best], "cv_mse_z_mean": cv.mean(axis=0).tolist(),
                           "n_fit_rows": int(len(X)), "n_parameters": int(W.numel() + ym.numel())}


def run_fit(config_path) -> Dict:
    cfg = load_probe_config(config_path)
    out_dir = Path(cfg["out_dir"])
    if (out_dir / "metrics.json").is_file():
        raise FileExistsError(f"{out_dir}/metrics.json exists; refuse to overwrite a finished probe")
    out_dir.mkdir(parents=True, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    t0 = time.perf_counter()
    data = FoldData(cfg)
    head_type = cfg["head"]["type"]
    fit_info: Dict = {}
    builder = None
    if head_type != "none":
        train_cases = [data.load_case(u) for u in data.train_ids]
        builder = InputBuilder(cfg["head"]["inputs"], train_cases)
        fitter = _fit_mlp if head_type == "mlp" else _fit_ridge
        predict_delta, fit_info = fitter(cfg, data, builder, train_cases, device, out_dir)
        del train_cases
    # ---- held-out evaluation (81 frames)
    part = cfg["eval"]["partition"]
    eval_ids = data.test_ids if part == "test" else data.train_ids
    tau_true, ln_pred, anchor_dev = [], [], []
    for uid in eval_ids:
        c = data.load_case(uid)
        base = tnull_base(c["peak_ln"], data.mu, data.sd, data.peak_index).T        # (F, N)
        if head_type == "none":
            lp = base
        else:
            lp = base + predict_delta(builder.point_block(c), int(cfg["eval"]["chunk_points"]))
        anchor_dev.append(float(np.max(np.abs(lp[data.peak_index] - c["peak_ln"]))))
        tau_true.append(c["tau"])
        ln_pred.append(lp)
        if cfg["eval"]["save_predictions"]:
            np.savez_compressed(out_dir / "predictions" / f"{case_key(uid)}.npz", ln_pred=lp.astype(np.float32))
    report = cycle_report(tau_true, ln_pred, data.steps, data.peak_index, data.q_norm, data.floor, data.eps)
    result = {
        "schema": PROBE_SCHEMA, "name": cfg["name"], "arm": cfg["arm"], "fold": cfg["fold"], "seed": cfg["seed"],
        "partition": part, "n_cases": len(eval_ids), "config": cfg, "config_sha256": sha256_file(config_path),
        "cache_manifest_sha256": sha256_file(Path(cfg["cache_dir"]) / "manifest.json"),
        "frame_stats_sha256": sha256_file(cfg["frame_stats_path"]), "waveform_sha256": sha256_file(cfg["waveform_path"]),
        "split_sha256": sha256_file(cfg["split_path"]), "code_sha256": code_fingerprint(), "execution": execution_record(),
        "anchor_check": {"max_abs_peak_ln_vs_frozen_anchor": float(max(anchor_dev)),
                         "definition": "max over held-out points of |ln τ̂(t_peak) − frozen donor peak ln|; 0 by construction"},
        "fit": fit_info, "metrics": report, "elapsed_seconds": time.perf_counter() - t0,
    }
    write_json(out_dir / "metrics.json", result)
    ec, tn = report["eval_compatible"], report["tnull_compatible"]
    print(f"[{cfg['name']}] cycle {ec['cycle_r2cb_pa']:.4f} trough {ec['trough_r2cb_pa']:.4f} peak {ec['peak_r2cb_pa']:.4f} "
          f"TAWSS {ec['tawss_r2cb_pa']:.4f} | D2-口径 cycle {tn['cycle_r2cb_pa']:.4f} trough {tn['trough_r2cb_pa']:.4f} "
          f"| anchor dev {result['anchor_check']['max_abs_peak_ln_vs_frozen_anchor']:.2e}", flush=True)
    return result


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("command", choices=("cache", "fit"))
    ap.add_argument("--config", required=True)
    args = ap.parse_args()
    (run_cache if args.command == "cache" else run_fit)(args.config)


if __name__ == "__main__":
    main()

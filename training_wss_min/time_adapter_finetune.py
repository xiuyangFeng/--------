#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""导师方案第二步：解冻版时间适配器（矩阵 P0.11，v5.1 cv3）。

独立工具、完全配置驱动；复用 time_adapter_probe 的锚定头 / T-null / 指标与 dataset.WSSMinDataset 的采样管线，
不修改任何既有训练/评估代码路径（旧配置逐位不变）。

结构（与 P0.9 的 TL-warm / TL-random 一致）：
  - 冻结峰值锚：同折 X5D 峰值 ln τ̂（time_adapter_probe 的 cache，fixed_support + full_cloud），训练与评估都只查表；
  - 时间几何支路：X5D 同构网络，geometry_init='donor'（同折 checkpoint，TL-warm）或 'random'（同种子随机初始化，TL-random）；
    只有 trainable_modules（默认 fp + local_wall_branch，即"后段 decoder/FP"）可训练，其余（stem/sa）冻结且保持 eval；
  - 时间头：TimeAdapterHead（几何投影 + 独立时间 encoder → 融合 → decoder），输入 = 支路 query_x ⊕ 峰值 ln τ̂；
  - ln τ̂(x,t) = T-null(x,t) + R(h,φ(t)) − R(h,φ(t_peak))，峰值帧与冻结锚逐位恒等。
用法：python -m training_wss_min.time_adapter_finetune --config configs/<exp>/<arm>_f0_s1234.json
"""
from __future__ import annotations

import argparse
import copy
import json
import math
import time
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
from torch_geometric.nn import knn_interpolate

from . import config as C
from . import dataset as D
from . import evaluate as E
from . import surface as S
from . import time_adapter_probe as TA
from .models import build_model

SCHEMA = "time_adapter_finetune_v1"
DEFAULTS: Dict = {
    "schema": SCHEMA, "name": None, "experiment": None, "arm": None, "fold": None, "seed": 1234, "title": "",
    "donor_run": None, "donor_checkpoint": "best", "geometry_init": "donor",
    "trainable_modules": ["fp", "local_wall_branch"],
    "cache_dir": None, "split_path": None, "data_root": None, "frame_stats_path": None, "waveform_path": None,
    "out_dir": None, "peak_index": 21, "phase_features": list(TA.PHASE_KEYS),
    "head": {"inputs": ["query_x", "peak_ln"], "geom_hidden": 128, "time_hidden": 64, "decoder_hidden": [128, 128],
             "zero_init_output": True},
    "train": {"epochs": 150, "batch_cases": 8, "query_points": 2048, "lr_head": 1e-3, "lr_geometry": 1e-4,
              "weight_decay": 1e-4, "warmup_epochs": 5, "min_lr_ratio": 0.01, "grad_clip": 1.0, "selection": "last",
              "num_workers": 4, "input_stats_batches": 4, "allow_tf32": True, "max_steps": 0, "log_every_epochs": 5},
    "eval": {"partition": "test", "chunk_points": 16384},
    "notes": "",
}
HEAD_INPUTS = ("query_x", "peak_ln")


def load_config(path) -> Dict:
    raw = json.loads(Path(path).read_text())
    if raw.get("schema") != SCHEMA:
        raise ValueError(f"{path}: schema must be {SCHEMA!r}")
    cfg = TA._merge(DEFAULTS, raw)
    for key in ("name", "arm", "fold", "donor_run", "cache_dir", "split_path", "data_root", "frame_stats_path",
                "waveform_path", "out_dir"):
        if cfg[key] is None:
            raise ValueError(f"{path}: missing required key {key!r}")
    if cfg["geometry_init"] not in ("donor", "random"):
        raise ValueError("geometry_init must be 'donor' or 'random'")
    if not cfg["head"]["inputs"] or set(cfg["head"]["inputs"]) - set(HEAD_INPUTS):
        raise ValueError(f"head.inputs must be a non-empty subset of {HEAD_INPUTS}")
    if cfg["train"]["selection"] != "last":
        raise ValueError("only train.selection='last' is pre-registered")
    return cfg


class TimeBranchModel(nn.Module):
    """时间几何支路（X5D 同构）+ 固定输入标准化 + 锚定时间头。"""

    def __init__(self, geometry: nn.Module, head: TA.TimeAdapterHead, inputs: List[str], trainable: List[str]):
        super().__init__()
        if geometry.query_decoder != "interpolate":
            raise ValueError("time branch supports query_decoder='interpolate' (X5D) only")
        self.geometry, self.head, self.inputs, self.trainable = geometry, head, list(inputs), list(trainable)
        dim = {"query_x": int(geometry.head[0].in_features), "peak_ln": 1}
        self.register_buffer("in_mean", torch.zeros(sum(dim[k] for k in self.inputs)))
        self.register_buffer("in_std", torch.ones(sum(dim[k] for k in self.inputs)))
        for p in self.geometry.parameters():
            p.requires_grad_(False)
        for name in self.trainable:
            for p in getattr(self.geometry, name).parameters():
                p.requires_grad_(True)

    def set_train_mode(self):
        """冻结部分恒为 eval（BN 统计、drop_path 固定），只有可训练模块与头进入 train。"""
        self.eval()
        for name in self.trainable:
            getattr(self.geometry, name).train()
        self.head.train()

    def query_x(self, support_pos, support_x, support_batch, pos, batch, *, unit_ids, epoch, global_seed, evaluation,
                support_geometry=None):
        extra = {"geometry": support_geometry} if support_geometry is not None else {}
        encoded = self.geometry.encode_support(support_pos, support_x, support_batch, unit_ids=unit_ids, epoch=epoch,
                                               global_seed=global_seed, evaluation=evaluation, **extra)
        return knn_interpolate(encoded[1], encoded[0], pos, encoded[2], batch, k=self.geometry.query_interpolation_k)

    def raw_inputs(self, qx, peak_ln):
        parts = {"query_x": qx, "peak_ln": peak_ln[:, None]}
        return torch.cat([parts[k] for k in self.inputs], dim=1)

    def delta(self, qx, peak_ln, phase, peak_index):
        g = (self.raw_inputs(qx, peak_ln) - self.in_mean) / self.in_std
        return TA.anchored(self.head(g, phase), peak_index)


def _lr_lambda(tc, steps_per_epoch):
    total = max(int(tc["epochs"]) * steps_per_epoch, 1)
    warm = max(int(tc["warmup_epochs"]) * steps_per_epoch, 1)
    floor = float(tc["min_lr_ratio"])

    def f(step):
        if step < warm:
            return (step + 1) / warm
        progress = min((step - warm) / max(total - warm, 1), 1.0)
        return floor + (1 - floor) * 0.5 * (1 + math.cos(math.pi * progress))
    return f


def run(config_path) -> Dict:
    cfg = load_config(config_path)
    out_dir = Path(cfg["out_dir"])
    if (out_dir / "metrics.json").is_file():
        raise FileExistsError(f"{out_dir}/metrics.json exists; refuse to overwrite")
    out_dir.mkdir(parents=True, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tc = cfg["train"]
    seed = int(cfg["seed"])
    t_start = time.perf_counter()

    def numerics(train: bool):
        """训练按配置开 TF32；初始检查与评估恢复 PyTorch 默认（matmul 关、cuDNN 开），与缓存/evaluate 同口径。"""
        torch.backends.cuda.matmul.allow_tf32 = bool(tc["allow_tf32"]) if train else False
        torch.backends.cudnn.allow_tf32 = bool(tc["allow_tf32"]) if train else True

    numerics(False)

    fold = TA.FoldData({**cfg, "phase_features": cfg["phase_features"]})
    donor = Path(cfg["donor_run"])
    dcfg, feat_stats, donor_model, _ = E.load_model_from_run(donor, "cpu", cfg["donor_checkpoint"])
    if Path(dcfg.data.split_path).resolve() != Path(cfg["split_path"]).resolve():
        raise ValueError("donor must be trained on this fold's split")
    wss_stats = E.load_wss_stats_for_run(donor)
    torch.manual_seed(seed)
    geometry = build_model(dcfg.model, C.input_dim(dcfg))
    init_record = {"geometry_init": cfg["geometry_init"], "trainable_modules": cfg["trainable_modules"]}
    if cfg["geometry_init"] == "donor":
        geometry.load_state_dict(donor_model.state_dict())
        init_record["checkpoint_sha256"] = TA.sha256_file(donor / E.checkpoint_filename(cfg["donor_checkpoint"]))
    else:
        init_record["random_seed"] = seed
    del donor_model
    head_cfg = cfg["head"]
    width = {"query_x": int(geometry.head[0].in_features), "peak_ln": 1}
    head = TA.TimeAdapterHead(sum(width[k] for k in head_cfg["inputs"]), len(cfg["phase_features"]),
                              head_cfg["geom_hidden"], head_cfg["time_hidden"], head_cfg["decoder_hidden"],
                              head_cfg["zero_init_output"])
    model = TimeBranchModel(geometry, head, head_cfg["inputs"], cfg["trainable_modules"]).to(device)
    init_record["n_trainable_geometry"] = int(sum(p.numel() for p in model.geometry.parameters() if p.requires_grad))
    init_record["n_frozen_geometry"] = int(sum(p.numel() for p in model.geometry.parameters() if not p.requires_grad))
    init_record["n_head"] = int(sum(p.numel() for p in model.head.parameters()))

    # ---- data: donor 的采样合同，关掉密度/噪声增广与 query patch（本支路不用），query 点数按配置
    data_cfg = copy.deepcopy(dcfg.data)
    data_cfg.density_aug_prob, data_cfg.noise_aug_prob = 0.0, 0.0
    data_cfg.query_patch_nsample = 0
    data_cfg.query_n_points = int(tc["query_points"])
    load_kw = dict(target=dcfg.data.target, target_normalization=dcfg.data.target_normalization, data_root=dcfg.data.data_root,
                   required_frame_version=dcfg.data.required_frame_version, case_features_path=dcfg.data.case_features_path,
                   timesteps="peak", waveform_path=None, extra_point_features=C.v6_point_features(dcfg),
                   point_features_root=getattr(dcfg.data, "point_features_root", None))
    tr_cases = D.load_partition(cfg["split_path"], "train", wss_stats, **load_kw)
    if [c["unit_id"] for c in tr_cases] != fold.train_ids:
        raise ValueError("training case order/identity differs from the split")
    anchor = {}
    labels = {}
    for c in tr_cases:
        rec = fold.load_case(c["unit_id"])
        if len(rec["peak_ln"]) != len(c["pos"]):
            raise ValueError(f"{c['unit_id']}: cache/bundle point count mismatch")
        anchor[c["unit_id"]] = torch.from_numpy(rec["peak_ln"].astype(np.float32)).to(device)
        labels[c["unit_id"]] = torch.from_numpy(TA.ln_floor(rec["tau"], fold.floor, fold.eps).astype(np.float32)).to(device)
    train_ds = D.WSSMinDataset(tr_cases, data_cfg, feat_stats, training=True, base_seed=seed)
    gen = torch.Generator().manual_seed(seed)
    loader = DataLoader(train_ds, batch_size=int(tc["batch_cases"]), shuffle=True, collate_fn=D.collate,
                        num_workers=int(tc["num_workers"]), drop_last=False, persistent_workers=False, generator=gen,
                        worker_init_fn=D.worker_init_fn if int(tc["num_workers"]) > 0 else None)
    mu = torch.tensor(fold.mu, dtype=torch.float32, device=device)
    sd = torch.tensor(fold.sd, dtype=torch.float32, device=device)
    phase = torch.tensor(fold.phase, dtype=torch.float32, device=device)
    pk = fold.peak_index

    def batch_tensors(batch, epoch, evaluation):
        qb = batch["batch"].to(device)
        ctx = D.local_model_context(batch, device)
        qx = model.query_x(batch["support_pos"].to(device), batch["support_x"].to(device), batch["support_batch"].to(device),
                           batch["pos"].to(device), qb, unit_ids=batch["unit_ids"], epoch=epoch, global_seed=seed,
                           evaluation=evaluation, support_geometry=ctx.get("support_geometry"))
        peak = torch.cat([anchor[u][qi.to(device)] for u, qi in zip(batch["unit_ids"], batch["query_indices"])])
        y = torch.cat([labels[u][:, qi.to(device)] for u, qi in zip(batch["unit_ids"], batch["query_indices"])], dim=1).T
        return qx, peak, y

    # ---- 留出折单例推理（fixed_support + full_cloud；与 evaluate.predict_case_norm / cache 同一 support 种子）
    part = cfg["eval"]["partition"]
    ev_cases = D.load_partition(cfg["split_path"], part, wss_stats, **load_kw)
    support_n = int(dcfg.data.support_n_points or dcfg.data.wall_n_points)
    support_sampling = dcfg.data.support_sampling or dcfg.data.sampling

    @torch.no_grad()
    def predict_case(c, rec):
        uid = c["unit_id"]
        sseed = S.stable_seed(dcfg.eval.support_seed, uid, "eval_support")
        sidx = D.sample_support_indices(c, dcfg.data, sseed, n_points=support_n, sampling=support_sampling, stream="eval_support")
        geom = D.case_geometry(c) if dcfg.data.local_geometry else None
        extra = {"geometry": torch.from_numpy(np.ascontiguousarray(geom[sidx])).to(device)} if geom is not None else {}
        sx = torch.from_numpy(D.build_features(c, sidx, dcfg.data.input_features, feat_stats)).to(device)
        spos = torch.from_numpy(np.ascontiguousarray(c["pos"][sidx])).to(device)
        sb = torch.zeros(len(sidx), dtype=torch.long, device=device)
        encoded = model.geometry.encode_support(spos, sx, sb, unit_ids=[uid], epoch=0, global_seed=dcfg.eval.support_seed,
                                                evaluation=True, **extra)
        n = len(c["pos"])
        peak_ln = torch.from_numpy(rec["peak_ln"].astype(np.float32)).to(device)
        chunk = int(cfg["eval"]["chunk_points"])
        qxs, deltas = [], []
        for s0 in range(0, n, chunk):
            m = min(chunk, n - s0)
            pos = torch.from_numpy(np.ascontiguousarray(c["pos"][s0:s0 + m])).to(device)
            qx = knn_interpolate(encoded[1], encoded[0], pos, encoded[2], torch.zeros(m, dtype=torch.long, device=device),
                                 k=model.geometry.query_interpolation_k)
            qxs.append(qx.float().cpu())
            deltas.append(model.delta(qx, peak_ln[s0:s0 + m], phase, pk).double().cpu().numpy())
        return torch.cat(qxs).numpy(), np.concatenate(deltas).T

    # 工程检查（训练前，全 eval 模式、默认数值后端）：donor 初始化的支路在首个留出病例上必须复现缓存的 query_x；零初始化头 → Δ≡0
    model.eval()
    numerics(False)
    rec0 = fold.load_case(ev_cases[0]["unit_id"])
    qx0, d0 = predict_case(ev_cases[0], rec0)
    init_record["init_check_case"] = ev_cases[0]["unit_id"]
    init_record["init_query_x_max_abs_diff_vs_cache"] = float(np.max(np.abs(qx0 - rec0["query_x"])))
    init_record["init_delta_max_abs"] = float(np.max(np.abs(d0)))
    print(f"[ft] init check: query_x vs cache {init_record['init_query_x_max_abs_diff_vs_cache']:.2e} "
          f"(meaningful for geometry_init=donor), init |delta| {init_record['init_delta_max_abs']:.1e}", flush=True)
    del qx0, d0, rec0

    # ---- 输入标准化：用初始支路在前几个训练 batch 上的统计（两臂同法），此后固定
    numerics(True)
    model.set_train_mode()
    train_ds.set_epoch(0)
    feats = []
    with torch.no_grad():
        for i, batch in enumerate(loader):
            if i >= int(tc["input_stats_batches"]):
                break
            qx, peak, _ = batch_tensors(batch, 0, False)
            feats.append(model.raw_inputs(qx, peak))
    feats = torch.cat(feats)
    model.in_mean.copy_(feats.mean(0))
    model.in_std.copy_(torch.where(feats.std(0) > 1e-6, feats.std(0), torch.ones_like(feats[0])))
    del feats

    geo_params = [p for p in model.geometry.parameters() if p.requires_grad]
    opt = torch.optim.AdamW([{"params": geo_params, "lr": float(tc["lr_geometry"])},
                             {"params": list(model.head.parameters()), "lr": float(tc["lr_head"])}],
                            weight_decay=float(tc["weight_decay"]))
    steps_per_epoch = len(loader)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, _lr_lambda(tc, steps_per_epoch))
    history, step = [], 0
    t_train = time.perf_counter()
    max_steps = int(tc["max_steps"])
    for epoch in range(int(tc["epochs"])):
        model.set_train_mode()
        train_ds.set_epoch(epoch)
        ep_loss, nb = 0.0, 0
        for batch in loader:
            qx, peak, y = batch_tensors(batch, epoch, False)
            base = TA.tnull_base(peak, mu, sd, pk)
            pred = base + model.delta(qx, peak, phase, pk)
            loss = (((pred - y) / sd[None, :]) ** 2).mean()
            opt.zero_grad(set_to_none=True)
            loss.backward()
            gn = float(torch.nn.utils.clip_grad_norm_([p for g in opt.param_groups for p in g["params"]], float(tc["grad_clip"])))
            opt.step()
            sched.step()
            ep_loss += float(loss)
            nb += 1
            step += 1
            if max_steps and step >= max_steps:
                break
        rec = {"epoch": epoch, "loss_z": ep_loss / max(nb, 1), "lr_geometry": sched.get_last_lr()[0], "lr_head": sched.get_last_lr()[1],
               "grad_norm_last": gn, "elapsed_s": time.perf_counter() - t_train, "steps": step}
        history.append(rec)
        if epoch % int(tc["log_every_epochs"]) == 0 or epoch == int(tc["epochs"]) - 1 or (max_steps and step >= max_steps):
            print(f"[ft] epoch {epoch} loss_z {rec['loss_z']:.5f} lr_g {rec['lr_geometry']:.2e} lr_h {rec['lr_head']:.2e} "
                  f"{rec['elapsed_s'] / (epoch + 1):.1f} s/epoch", flush=True)
        if max_steps and step >= max_steps:
            break
    with (out_dir / "history.jsonl").open("w") as fh:
        for rec in history:
            fh.write(json.dumps(rec) + "\n")
    torch.save({"trainable_state": {k: v for k, v in model.state_dict().items()
                                    if k.startswith("head.") or k in ("in_mean", "in_std")
                                    or any(k.startswith(f"geometry.{n}.") for n in cfg["trainable_modules"])},
                "config": cfg, "init": init_record}, out_dir / "ckpt_last.pt")
    train_seconds = time.perf_counter() - t_train

    # ---- held-out 81 帧评估（fixed_support + full_cloud，与 evaluate / cache 同一 support 种子）
    model.eval()
    numerics(False)
    tau_true, ln_pred, anchor_dev = [], [], []
    for c in ev_cases:
        rec = fold.load_case(c["unit_id"])
        _, delta = predict_case(c, rec)
        base = TA.tnull_base(rec["peak_ln"], fold.mu, fold.sd, pk).T
        lp = base + delta
        anchor_dev.append(float(np.max(np.abs(lp[pk] - rec["peak_ln"]))))
        tau_true.append(rec["tau"])
        ln_pred.append(lp)
    report = TA.cycle_report(tau_true, ln_pred, fold.steps, pk, fold.q_norm, fold.floor, fold.eps)
    result = {
        "schema": SCHEMA, "name": cfg["name"], "arm": cfg["arm"], "fold": cfg["fold"], "seed": seed, "partition": part,
        "n_cases": len(ev_cases), "config": cfg, "config_sha256": TA.sha256_file(config_path),
        "cache_manifest_sha256": TA.sha256_file(Path(cfg["cache_dir"]) / "manifest.json"),
        "frame_stats_sha256": TA.sha256_file(cfg["frame_stats_path"]), "split_sha256": TA.sha256_file(cfg["split_path"]),
        "code_sha256": {**TA.code_fingerprint(), "time_adapter_finetune.py": TA.sha256_file(Path(__file__))},
        "execution": TA.execution_record(), "initialization": init_record,
        "anchor_check": {"max_abs_peak_ln_vs_frozen_anchor": float(max(anchor_dev)),
                         "definition": "max |ln τ̂(t_peak) − frozen donor peak ln| over held-out points; 0 by construction"},
        "fit": {"train_seconds": train_seconds, "steps": step, "steps_per_epoch": steps_per_epoch,
                "final_loss_z": history[-1]["loss_z"], "first_loss_z": history[0]["loss_z"]},
        "metrics": report, "elapsed_seconds": time.perf_counter() - t_start,
    }
    TA.write_json(out_dir / "metrics.json", result)
    ec = report["eval_compatible"]
    print(f"[{cfg['name']}] cycle {ec['cycle_r2cb_pa']:.4f} trough {ec['trough_r2cb_pa']:.4f} peak {ec['peak_r2cb_pa']:.4f} "
          f"TAWSS {ec['tawss_r2cb_pa']:.4f} | anchor dev {max(anchor_dev):.2e} | train {train_seconds / 60:.1f} min", flush=True)
    return result


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--config", required=True)
    run(ap.parse_args().config)


if __name__ == "__main__":
    main()

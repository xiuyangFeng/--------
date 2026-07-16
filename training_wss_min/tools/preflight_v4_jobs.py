#!/usr/bin/env python3
"""两项 v4 正式作业的只读数据加载、统计清单与 CUDA AMP 前后向门禁。"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch

from training_wss_min import config as C, dataset as D
from training_wss_min.models import build_model
from training_wss_min.objectives import compute_loss


def sha(path: Path) -> str:
    h=hashlib.sha256()
    with path.open("rb") as f:
        for b in iter(lambda:f.read(1024*1024),b""): h.update(b)
    return h.hexdigest()


def one(config_path: Path) -> dict:
    cfg=C.ExpConfig.from_json(config_path)
    if cfg.run_dir.exists(): raise FileExistsError(f"formal run dir already exists: {cfg.run_dir}")
    if cfg.data.required_frame_version != "stl_landmarks_v4": raise RuntimeError("v4 frame gate missing")
    stats_path=Path(cfg.data.wss_stats_path); stats=D.load_wss_stats(stats_path)
    tr=D.load_partition(cfg.data.split_path,"train",stats,strict=True,target=cfg.data.target,
        target_normalization=cfg.data.target_normalization,data_root=cfg.data.data_root,
        required_frame_version=cfg.data.required_frame_version)
    te=D.load_partition(cfg.data.split_path,"test",stats,strict=True,target=cfg.data.target,
        target_normalization=cfg.data.target_normalization,data_root=cfg.data.data_root,
        required_frame_version=cfg.data.required_frame_version)
    split=D.load_split(cfg.data.split_path); units=split["train_cases"]
    if stats["train_units"] != units or stats["n_cases"] != len(tr):
        raise RuntimeError("stats train unit manifest differs from split")
    for item in stats["bundle_manifest"]:
        if sha(Path(item["bundle_path"])) != item["bundle_sha256"]:
            raise RuntimeError(f"stats bundle hash drift: {item['unit_id']}")
    excluded=set(split.get("excluded_cases",[])); partitions=set(split["train_cases"]+split["val_cases"]+split["test_cases"])
    if excluded & partitions: raise RuntimeError("excluded case leaked into partition")
    feat=D.compute_feature_stats(tr,cfg.data.input_features,cfg.data.curvature_transform)
    quant=D.compute_train_weight_quantiles(tr)
    ds=D.WSSMinDataset(tr,cfg.data,feat,training=True,base_seed=cfg.train.seed)
    batch=D.collate([ds[i] for i in range(min(cfg.train.batch_cases,len(ds)))])
    device="cuda"
    if not torch.cuda.is_available(): raise RuntimeError("CUDA is not available for formal GPU preflight")
    model=build_model(cfg.model,C.input_dim(cfg)).to(device); opt=torch.optim.AdamW(model.parameters(),lr=cfg.train.lr)
    opt.zero_grad(set_to_none=True)
    with torch.amp.autocast("cuda",enabled=cfg.train.amp):
        pred=model(batch["pos"].to(device),batch["x"].to(device),batch["batch"].to(device))
        loss=compute_loss(pred,batch,cfg.train,device,stats)
    if not torch.isfinite(loss): raise RuntimeError("non-finite CUDA smoke loss")
    loss.backward(); opt.step(); torch.cuda.synchronize()
    result={"config":str(config_path.resolve()),"config_sha256":sha(config_path),
        "run_dir":str(cfg.run_dir),"train_cases":len(tr),"test_cases":len(te),
        "canonical_ids":all(len(x.split('/'))==3 for x in units+split["test_cases"]),
        "required_frame_version":cfg.data.required_frame_version,"stats_path":str(stats_path.resolve()),
        "stats_sha256":sha(stats_path),"stats_manifest_hashes":"all_match",
        "feature_stats_computed_for_train_only":sorted(feat),"fixed_loss_quantiles":quant,
        "cuda_device":torch.cuda.get_device_name(torch.cuda.current_device()),
        "cuda_amp_forward_backward":"passed","smoke_loss":float(loss.detach().cpu())}
    del model,opt,ds,batch,tr,te; gc.collect(); torch.cuda.empty_cache()
    return result


def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--configs",nargs="+",type=Path,required=True)
    ap.add_argument("--output",type=Path,required=True); a=ap.parse_args()
    payload={"generated_at":datetime.now(timezone.utc).isoformat(),"status":"passed",
             "jobs":[one(p.resolve()) for p in a.configs]}
    a.output.parent.mkdir(parents=True,exist_ok=True)
    a.output.write_text(json.dumps(payload,indent=2,ensure_ascii=False)+"\n")
    print(json.dumps(payload,indent=2,ensure_ascii=False))


if __name__=="__main__": main()

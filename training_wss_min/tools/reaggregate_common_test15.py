#!/usr/bin/env python3
"""从旧 E2 已保存的逐点预测重汇总 common-test15；不加载模型。"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path

import numpy as np


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for b in iter(lambda: f.read(1024 * 1024), b""): h.update(b)
    return h.hexdigest()


def metrics(y: np.ndarray, p: np.ndarray) -> dict:
    err = p - y; ss_res = float(np.sum(err * err)); ss_tot = float(np.sum((y-y.mean())**2))
    rmse = float(np.sqrt(np.mean(err * err)))
    return {"r2": float(1 - ss_res / ss_tot) if ss_tot > 0 else float("nan"),
            "mae": float(np.mean(np.abs(err))), "rmse": rmse,
            "nrmse_range": float(rmse / max(float(y.max()-y.min()), 1e-12)), "n": len(y)}


def run(repo: Path, checkpoint: str) -> dict:
    run_dir = repo / "training_wss_min/runs/pointnet_distribution_matrix/outputs/e2_global"
    split_path = repo / "training/splits/split_AG_wss_min_v4_traintest.json"
    split = json.loads(split_path.read_text()); expected = split["test_cases"]
    source_root = run_dir / f"postview/ckpt_{checkpoint}/test"
    out = run_dir / f"eval/ckpt_{checkpoint}/common_test15_from_saved_predictions"
    if out.exists(): raise FileExistsError(out)
    out.mkdir(parents=True)
    all_y=[]; all_p=[]; rows=[]; sources=[]
    for unit in expected:
        _, subset, case = unit.split("/")
        path = source_root / f"{subset}__{case}__peak_wss/_export/{case}__wall.csv"
        with path.open(newline="", encoding="utf-8") as f:
            data=list(csv.DictReader(f))
        y=np.asarray([float(x["wss_cfd"]) for x in data]); p=np.asarray([float(x["wss_pred"]) for x in data])
        row={"case":unit,**metrics(y,p),"source_csv":str(path.resolve()),"source_sha256":sha(path)}
        rows.append(row); sources.append({"unit_id":unit,"path":str(path.resolve()),"sha256":sha(path)})
        all_y.append(y); all_p.append(p)
    if len(rows)!=15 or any("WANG_DENG_FENG" in x["case"] for x in rows):
        raise RuntimeError("common-test15 contract failed")
    y=np.concatenate(all_y); p=np.concatenate(all_p); field=metrics(y,p)
    aggregate={"n_cases":15,"r2_casemean":float(np.mean([x["r2"] for x in rows])),
               "mae_casemean":float(np.mean([x["mae"] for x in rows])),
               "rmse_casemean":float(np.mean([x["rmse"] for x in rows])),
               "nrmse_casemean":float(np.mean([x["nrmse_range"] for x in rows]))}
    payload={"schema_version":1,"generated_at":datetime.now(timezone.utc).isoformat(),
             "protocol":"v3 E2 saved-prediction common-test15 fair anchor",
             "no_model_inference":True,"excluded":"AG/slow/WANG_DENG_FENG",
             "source_run":str(run_dir.resolve()),"source_checkpoint":f"ckpt_{checkpoint}.pt",
             "source_checkpoint_sha256":sha(run_dir/f"ckpt_{checkpoint}.pt"),
             "split_path":str(split_path.resolve()),"split_sha256":sha(split_path),
             "aggregate":aggregate,"field":field,"source_predictions":sources}
    (out/"metrics.json").write_text(json.dumps(payload,indent=2,ensure_ascii=False)+"\n")
    with (out/"per_case_metrics.csv").open("w",newline="",encoding="utf-8") as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    return {"output":str(out),"aggregate":aggregate,"field":field}


def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--repo",type=Path,default=Path.cwd())
    ap.add_argument("--checkpoint",choices=("best","last"),default="best")
    a=ap.parse_args(); print(json.dumps(run(a.repo.resolve(),a.checkpoint),indent=2))


if __name__ == "__main__": main()

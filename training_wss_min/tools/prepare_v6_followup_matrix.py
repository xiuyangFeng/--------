#!/usr/bin/env python3
"""Generate the authorised 15-arm, single-seed A5 follow-up matrix.

No geometry rebuilding or new-geometry arms are included. ``--check`` is read-only.
All arm deltas are measured against the frozen A5 config, with secondary parent
comparisons recorded explicitly for the decoder ladder and loss factorial.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
from pathlib import Path

from training_wss_min import config as C
from training_wss_min.tools.prepare_v6_singleframe_matrix import realised_diff, set_path

ROOT = Path(__file__).resolve().parents[2]
ANCHOR = ROOT / "training_wss_min/configs/v6_singleframe_20260909/A5_r4_localbranch_diffgeom_s1234.json"
OUT_DIR = ROOT / "training_wss_min/configs/v6_followup_20260909"
RUN_PREFIX = "v6_followup_20260909"
MASK_GROUPS = {
    "D1": ("notube", [7, 8, 9], "管道坐标 rho/sin(theta)/cos(theta)"),
    "D2": ("nodist", [11, 12, 13], "分叉/端点距离与端区"),
    "D3": ("nodrds", [10], "半径梯度 dr/ds"),
    "D4": ("nonormal", [14, 15, 16], "显式法向及局部分支法向差分；预计算曲率保留"),
    "D5": ("nofine", [17, 18, 19, 20], "fine 主曲率/高斯曲率/curvedness"),
    "D6": ("nocoarse", [21, 22, 23], "coarse 主曲率/高斯曲率"),
    "D7": ("notndot", [24], "中心线切向与壁面法向点积"),
}
ARMS = [
    {"id": aid, "slug": f"{aid}_a5_{slug}_s1234", "category": "ablation",
     "hypothesis": f"标准化后屏蔽{label}，保留固定25通道和所有索引",
     "changes": {"model.input_feature_mask_indices": indices}}
    for aid, (slug, indices, label) in MASK_GROUPS.items()
] + [
    {"id": "L1", "slug": "L1_a5_mse_s1234", "category": "loss",
     "hypothesis": "固定原A5输入/模型，仅去除pinball，使用纯MSE",
     "changes": {"train.loss_pinball_lambda": 0.0}},
    {"id": "L2", "slug": "L2_a5_pin_rawhuber_s1234", "category": "loss",
     "hypothesis": "固定原A5，保留q90 pinball并增加0.1 Pa Huber",
     "changes": {"train.loss_raw_huber_lambda": 0.1, "train.raw_huber_delta": 1.0}},
    {"id": "L3", "slug": "L3_a5_mse_rawhuber_s1234", "category": "loss",
     "hypothesis": "固定原A5，MSE+0.1 Pa Huber，无pinball", "baseline": "L1",
     "changes": {"train.loss_pinball_lambda": 0.0,
                 "train.loss_raw_huber_lambda": 0.1, "train.raw_huber_delta": 1.0}},
    {"id": "M1", "slug": "M1_a5_pointwise_s1234", "category": "model",
     "hypothesis": "同注入位置/零初始化的逐点残差容量对照，替代空间邻域分支",
     "changes": {"model.local_branch_mode": "pointwise", "model.pointwise_branch_hidden": 34},
     "note": "pointwise分支10640参数，原邻域分支10496，差+1.37%；逐点MLP不读取邻域/相对几何"},
    {"id": "M2", "slug": "M2_a5_independent_k3_s1234", "category": "model",
     "hypothesis": "只将SAME改为独立query，保持固定3NN解码",
     "changes": {"data.query_mode": "independent"}},
    {"id": "M3", "slug": "M3_a5_independent_k16_s1234", "category": "model", "baseline": "M2",
     "hypothesis": "独立query条件下，仅将固定query插值3NN改为16NN；encoder FP仍3NN",
     "changes": {"data.query_mode": "independent", "model.query_interpolation_k": 16}},
    {"id": "M4", "slug": "M4_a5_localdecoder_s1234", "category": "model", "baseline": "M3",
     "hypothesis": "固定16邻居基础上增加可学几何条件权重、query自身几何和残差头",
     "changes": {"data.query_mode": "independent", "model.query_decoder": "local_attn",
                 "model.query_decoder_k": 16, "model.query_decoder_hidden": 32,
                 "model.query_decoder_residual": True,
                 "model.query_decoder_feature_indices": [4, 14, 15, 16]},
     "note": "整个局部解码配方对照；可学权重/输入条件/残差不作单项归因"},
    {"id": "M5", "slug": "M5_a5_singleradius_s1234", "category": "model",
     "hypothesis": "三条局部分支均设r=0.03，参数量严格相同，检验多尺度邻域作用",
     "changes": {"model.local_branch_radii": [0.03, 0.03, 0.03]}},
]


def payloads() -> tuple[dict, dict[str, dict]]:
    anchor = json.loads(ANCHOR.read_text())
    if len(anchor["data"]["input_features"]) != 25:
        raise ValueError("A5 feature contract changed: expected fixed 25 channels")
    expected_names = {
        7: "rho", 8: "theta_sin", 9: "theta_cos", 10: "dr_ds",
        11: "dist_to_junction_mm", 12: "dist_to_endpoint_mm", 13: "end_zone",
        14: "nx_aligned", 15: "ny_aligned", 16: "nz_aligned", 17: "curv_k1",
        18: "curv_k2", 19: "curv_gauss", 20: "curvedness", 21: "curv_k1_c",
        22: "curv_k2_c", 23: "curv_gauss_c", 24: "tn_dot",
    }
    for i, key in expected_names.items():
        if anchor["data"]["input_features"][i] != key:
            raise ValueError(f"A5 feature index {i} must remain {key}")
    results = {}
    for arm in ARMS:
        p = copy.deepcopy(anchor)
        p["name"] = f"{RUN_PREFIX}/{arm['slug']}"
        p["notes"] = f"A5 follow-up {arm['id']}: {arm['hypothesis']}。单帧WSS，s1234，400epoch，仅单seed筛选。"
        for key, value in arm["changes"].items():
            set_path(p, key, value)
        C.ExpConfig.from_dict(p)
        diff = realised_diff(anchor, p)
        if set(diff) - set(arm["changes"]):
            raise ValueError(f"undeclared changes in {arm['id']}: {diff}")
        # Loss experiments cannot silently include geometry or model changes.
        if arm["category"] == "loss" and any(not k.startswith("train.") for k in diff):
            raise ValueError(f"loss-only arm {arm['id']} changed non-loss state")
        results[arm["id"]] = p
    return anchor, results


def build(check: bool = False) -> int:
    anchor, configs = payloads()
    entries = []
    for arm in ARMS:
        p = configs[arm["id"]]
        parent_id = arm.get("baseline", "B0")
        parent = anchor if parent_id == "B0" else configs[parent_id]
        # Expand defaults on both sides so inherited-to-default changes appear too.
        parent_full = json.loads(json.dumps(C.ExpConfig.from_dict(parent).to_dict()))
        child_full = json.loads(json.dumps(C.ExpConfig.from_dict(p).to_dict()))
        entries.append({
            "id": arm["id"], "category": arm["category"], "config": arm["slug"] + ".json",
            "run_name": p["name"], "anchor": "A5", "baseline": parent_id,
            "hypothesis": arm["hypothesis"], "note": arm.get("note", ""),
            "changes": realised_diff(anchor, p),
            "changes_vs_parent": realised_diff(parent_full, child_full),
            "input_dim": 25, "masked_features": [anchor["data"]["input_features"][i]
                for i in p["model"].get("input_feature_mask_indices", [])],
        })
    manifest = {
        "matrix_id": RUN_PREFIX, "anchor_config": str(ANCHOR.relative_to(ROOT)),
        "anchor_sha256": hashlib.sha256(ANCHOR.read_bytes()).hexdigest(),
        "authorised_scope": "D1-D7, L1-L3, M1-M5 only; WSS output only; no new geometry or centreline changes",
        "protocol": {"seed": 1234, "epochs": 400, "target": "wss", "timesteps": "peak",
                     "support_n_points": 5000, "query_n_points": 5000, "batch_cases": 8,
                     "selection_rule": "train_loss", "eval_support_seed": 1234,
                     "single_seed": "screening only; no statistical superiority claims",
                     "raw_huber": "Pa/p90(train138); lambda=0.1; delta=1.0; fixed before test evaluation"},
        "reference": {"id": "B0", "run_name": anchor["name"],
                      "reuse": ["R4", "A1", "A4", "A5", "A5b"]},
        "arms": entries,
    }
    outputs = {OUT_DIR / (arm["slug"] + ".json"): configs[arm["id"]] for arm in ARMS}
    outputs[OUT_DIR / "matrix.json"] = manifest
    if not check:
        OUT_DIR.mkdir(parents=True, exist_ok=True)
    for path, payload in outputs.items():
        if check:
            if not path.is_file() or json.loads(path.read_text()) != payload:
                raise ValueError(f"stale or missing matrix artifact: {path}")
        else:
            path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n")
    print(f"{'checked' if check else 'wrote'} {len(entries)} arms: {OUT_DIR}")
    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--check", action="store_true")
    raise SystemExit(build(**vars(parser.parse_args())))

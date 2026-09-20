#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Derive the V6 single-frame WSS matrix from the frozen R4 anchor config.

Every arm is produced by applying **one** recorded change to the same anchor
(`v5_rerun_20260906/r4_lsa2_h2_logradius_v5feat_s1234`), so "only one variable
moved" is mechanical rather than a claim: the script writes each arm's diff into
`matrix.json` and refuses to emit an arm whose realised diff does not match.

    python -m training_wss_min.tools.prepare_v6_singleframe_matrix [--check]
"""
from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path

from training_wss_min import config as C

ROOT = Path(__file__).resolve().parents[2]
ANCHOR = ROOT / "training_wss_min/configs/v5_rerun_20260906/r4_lsa2_h2_logradius_v5feat_s1234.json"
OUT_DIR = ROOT / "training_wss_min/configs/v6_singleframe_20260909"
RUN_PREFIX = "v6_singleframe_20260909"
GEOM_ROOT = str(ROOT / "data_wss_v5/views/wss_min_geom_v2")

# R4 input feature positions used by the new modules (see the anchor config)
IDX_LOCAL_RADIUS = 4
IDX_NORMAL = (14, 15, 16)

DIFF_GEOM_FEATURES = (
    "curv_k1", "curv_k2", "curv_gauss", "curvedness",
    "curv_k1_c", "curv_k2_c", "curv_gauss_c", "tn_dot",
)
BRANCH_FEATURES = C.V6_SEMANTIC_FEATURE_KEYS + ("s_local_frac",)

INDEPENDENT_QUERY = {
    "data.query_mode": "independent",
    "data.query_n_points": 5000,
    "data.query_sampling": "random",
}

ARMS = [
    {
        "id": "A0", "slug": "A0_r4nll_jensen_s1234",
        "hypothesis": "高斯 NLL 头 + 逐点 Jensen 回变换是更好的部署口径（Wave 3 单 seed 0.600 复核）",
        "changes": {
            "train.loss": "gaussian_nll", "train.loss_pinball_lambda": 0.0,
            "model.out_dim": 2, "eval.pointwise_jensen": True,
        },
        "note": "out_dim=2 与 pinball=0 是 gaussian_nll 的配套约束，不是独立变量",
    },
    {
        "id": "A1", "slug": "A1_r4_localbranch_s1234",
        "hypothesis": "2–10 mm 局部壁面结构需要 support 分辨率上的多尺度邻域（SA1 的 0.05 半径太粗）",
        "changes": {
            "model.local_branch": True,
            "model.local_branch_radii": [0.015, 0.03, 0.06],
            "model.local_branch_nsample": [16, 16, 16],
            "model.local_branch_channels": 32,
            "model.local_branch_feature_indices": list(IDX_NORMAL),
        },
        "note": "分支的半径/邻居数/通道/法向差分是同一个模块的定义，父模型完全没有该模块",
    },
    {
        "id": "A1a", "slug": "A1a_r4_sa1c256_s1234",
        "hypothesis": "只加密 SA1 中心（125→256）即可覆盖局部尺度",
        "changes": {"model.sa_center_counts": [256, 125, 32]},
    },
    {
        "id": "A1b", "slug": "A1b_r4_sa1r003_s1234",
        "hypothesis": "只缩小 SA1 半径（0.05→0.03，≈8.5→5 mm）即可对准局部尺度",
        "changes": {"model.sa_radius": [0.03, 0.1, 0.2]},
    },
    {
        "id": "A2a", "slug": "A2a_r4_indepquery_s1234",
        "hypothesis": "训练时 query==support 使解码器只在恒等状态下被训练，与评估的 5000→全壁面插值不一致",
        "changes": dict(INDEPENDENT_QUERY),
    },
    {
        "id": "A2b", "slug": "A2b_r4_localattn_s1234",
        "hypothesis": "3-NN 凸插值抹平峰值：改用 16 邻域几何条件权重 + 零初始化局部残差头",
        "changes": {
            **INDEPENDENT_QUERY,
            "model.query_decoder": "local_attn",
            "model.query_decoder_k": 16,
            "model.query_decoder_hidden": 32,
            "model.query_decoder_residual": True,
            "model.query_decoder_feature_indices": [IDX_LOCAL_RADIUS, *IDX_NORMAL],
        },
        "baseline": "A2a",
        "note": "相对 A2a 只多了解码器本身；independent query 是 local_attn 的训练前提",
    },
    {
        "id": "A3", "slug": "A3_r4_casescale_s1234",
        "hypothesis": "逐例 oracle 仿射还剩 0.05–0.07 R²：让网络自己预测一个病例级幅值",
        "changes": {"model.case_scale_head": True, "model.case_scale_hidden": 64},
    },
    {
        "id": "A4", "slug": "A4_r4_diffgeom_s1234",
        "hypothesis": "局部微分几何（主曲率/高斯曲率/curvedness/切向-法向夹角）补充局部流动形态",
        "changes": {
            "data.input_features": "+" + ",".join(DIFF_GEOM_FEATURES),
            "data.point_features_root": GEOM_ROOT,
        },
        "note": "point_features_root 是新特征的数据来源，不是独立变量",
    },
    {
        "id": "A4b", "slug": "A4b_r4_branchsem_s1234",
        "hypothesis": "分支语义 one-hot + 分支内相对弧长给出分支身份与位置",
        "changes": {"data.input_features": "+" + ",".join(BRANCH_FEATURES)},
    },
]


SECTIONS = {"data": C.DataConfig, "model": C.ModelConfig, "train": C.TrainConfig, "eval": C.EvalConfig}


def default_of(section: str, key: str):
    """锚点 JSON 里没写的字段 = 该 dataclass 的默认值（也就是旧行为）。"""
    return getattr(SECTIONS[section](), key)


def set_path(payload: dict, dotted: str, value) -> None:
    section, key = dotted.split(".", 1)
    block = payload[section]
    if key not in SECTIONS[section].__dataclass_fields__:
        raise KeyError(f"{dotted} is not a config field (typo or missing dataclass entry)")
    if isinstance(value, str) and value.startswith("+") and key == "input_features":
        block[key] = list(block[key]) + value[1:].split(",")
        return
    block[key] = value


def realised_diff(anchor: dict, arm: dict) -> dict:
    out = {}
    for section in SECTIONS:
        for key, value in arm[section].items():
            before = anchor[section].get(key, default_of(section, key))
            if isinstance(before, tuple):
                before = list(before)
            if before != value:
                out[f"{section}.{key}"] = {"from": before, "to": value}
    return out


def combo_arm(ids: list[str], slug: str | None, arm_id: str) -> dict:
    """A5: merge the winning arms' changes into one config.

    Conflicting fields are refused rather than silently resolved - two arms that move the same
    field cannot be combined and still be attributed.
    """
    parts = [a for a in ARMS if a["id"] in ids]
    if len(parts) != len(ids):
        raise SystemExit(f"unknown arm id in {ids}; known: {[a['id'] for a in ARMS]}")
    changes: dict = {}
    for part in parts:
        for key, value in part["changes"].items():
            if key in changes and changes[key] != value:
                raise SystemExit(f"{arm_id}: {key} is set differently by two arms; not combinable")
            if key == "data.input_features" and key in changes:
                changes[key] = changes[key] + "," + value.lstrip("+")   # both append feature blocks
            else:
                changes[key] = value
    return {
        "id": arm_id,
        "slug": slug or f"{arm_id}_" + "_".join(i.lower() for i in ids) + "_s1234",
        "hypothesis": "组合验证互补性：" + " + ".join(f"{a['id']}（{a['hypothesis']}）" for a in parts),
        "changes": changes,
        "baseline": "+".join(ids),
        "note": "组合臂，相对 R4 有多处变化；只用于检验互补性，不作单变量归因",
    }


def build(check: bool = False, combo: list[str] | None = None, combo_id: str = "A5",
          combo_slug: str | None = None) -> int:
    anchor = json.loads(ANCHOR.read_text())
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    arms = list(ARMS)
    if combo:
        arms = arms + [combo_arm(combo, combo_slug, combo_id)]
    entries = []
    for arm in arms:
        payload = copy.deepcopy(anchor)
        payload["name"] = f"{RUN_PREFIX}/{arm['slug']}"
        payload["notes"] = (
            f"V6 单帧 WSS 矩阵 {arm['id']}：{arm['hypothesis']}。"
            f"锚点 = {ANCHOR.stem}（R4：PointNeXt-R + LocalGeoPE + L-SA2 + V5 十维几何特征，"
            f"train138/test34，峰值单帧，seed 1234，400 epoch）。"
            + (f" 说明：{arm['note']}。" if arm.get("note") else "")
        )
        for dotted, value in arm["changes"].items():
            set_path(payload, dotted, value)
        cfg = C.ExpConfig.from_dict(payload)   # validates the contract
        diff = realised_diff(anchor, payload)
        expected = set(arm["changes"])
        unexpected = sorted(set(diff) - expected)
        if unexpected:
            raise SystemExit(f"{arm['id']}: config moved fields that were never declared: {unexpected}")
        # declared fields that already equal the dataclass default are explicit no-ops (kept for readability)
        already_default = sorted(expected - set(diff))
        path = OUT_DIR / f"{arm['slug']}.json"
        if check:
            if not path.is_file() or json.loads(path.read_text()) != payload:
                raise SystemExit(f"{arm['id']}: {path} is stale; rerun without --check")
        else:
            path.write_text(json.dumps(payload, indent=2, ensure_ascii=False))
        entries.append({
            "id": arm["id"], "config": path.name, "run_name": payload["name"],
            "baseline": arm.get("baseline", "R4"), "hypothesis": arm["hypothesis"],
            "changes": diff, "declared_but_already_default": already_default,
            "note": arm.get("note", ""),
            "input_dim": len(cfg.data.input_features),
        })
    existing = json.loads((OUT_DIR / "matrix.json").read_text()) if (OUT_DIR / "matrix.json").is_file() else {}
    known = {e["id"] for e in entries}
    entries += [e for e in existing.get("arms", []) if e["id"] not in known]
    matrix = {
        "matrix_id": RUN_PREFIX,
        "objective": "峰值单帧壁面 WSS：只用几何/网络/解码/校准的改动提高精度，不引入时间维 t",
        "anchor_config": str(ANCHOR.relative_to(ROOT)),
        "protocol": {
            "seed": 1234, "timesteps": "peak", "target": "wss", "epochs": 400,
            "split": "data_wss_v5/views/wss_min_view_v1/split_V5_train138_test34.json",
            "support_n_points": 5000, "selection_rule": "train_loss",
            "eval": "test34 全壁面 query、固定 support（seed 1234）、best 与 last 两个 checkpoint",
            "single_seed": "筛选轮每臂只跑 seed 1234",
        },
        "reference": {
            "r4_three_seed_mean_physical_r2_cb": 0.5560,
            "r4_three_seed_mean_normalized_r2_cb": 0.7823,
            "r4_seed1234_physical_r2_cb": 0.572146,
            "r4_seed1234_normalized_r2_cb": 0.786347,
            "seed_noise_sd_physical": 0.0164, "seed_noise_sd_normalized": 0.0040,
            "single_seed_vs_single_seed_95_band_physical": 0.045,
            "single_seed_vs_single_seed_95_band_normalized": 0.011,
            "decision_rule": ("单 seed 只做筛选，不下结论；晋级臂必须补到三 seed 后按"
                              "三 seed 均值 ±0.026(物理)/±0.006(归一化) 判定"),
        },
        "arms": entries,
    }
    (OUT_DIR / "matrix.json").write_text(json.dumps(matrix, indent=2, ensure_ascii=False))
    print(f"{'checked' if check else 'wrote'} {len(entries)} arms -> {OUT_DIR}")
    for entry in entries:
        print(f"  {entry['id']:4s} {entry['config']:32s} in_dim={entry['input_dim']:2d} "
              f"changes={list(entry['changes'])}")
    return 0


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true", help="只校验磁盘上的配置与本脚本一致")
    ap.add_argument("--combo", nargs="+", default=None, help="生成组合臂，例如 --combo A1 A3")
    ap.add_argument("--combo-id", default="A5", dest="combo_id")
    ap.add_argument("--combo-slug", default=None, dest="combo_slug")
    raise SystemExit(build(**vars(ap.parse_args())))

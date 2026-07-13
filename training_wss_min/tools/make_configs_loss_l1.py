#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Generate W2 / L1 loss-matrix configs (log-z MSE + raw-space scaled Huber).

第六轮 W2 loss 小矩阵的 L1 分支：固定 D 输入（xyz+geom）、PointNeXt-S、FPS-2000、
split 与 seed，只扫 ``loss_raw_huber_lambda`` 的少量预注册值。L0 对照为
``r6_scale_D_xyzgeom_s1234``（同输入、λ=0，已完成三 seed），因此本文件只落 L1。

先在单 seed（1234）做廉价 Gate 确定 λ 区间；确认性结论再对选中的 λ 补三 seed，
另行提交，不在此扩表。
"""

from __future__ import annotations

import copy
import json

from training_wss_min.tools.config_paths import (
    CONFIG_ROOT,
    SWEEPS_DIR,
    config_path,
    repo_config_path,
)

OUT_DIR = CONFIG_ROOT / "loss_aug_ablation"
# D 输入、B1/dev1/FPS2000 协议、λ=0 的冻结 loss-control。
BASE = config_path("r6_scale_D_xyzgeom_s1234")
LAMBDAS = (0.1, 0.3, 1.0)
SEED = 1234


def main() -> None:
    base = json.loads(BASE.read_text(encoding="utf-8"))
    assert base["data"]["input_features"] == [
        "x", "y", "z", "abscissa_norm", "local_radius", "curvature"
    ], "L1 必须固定 D 输入（xyz+geom）"
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    SWEEPS_DIR.mkdir(parents=True, exist_ok=True)
    manifest = []
    for lam in LAMBDAS:
        cfg = copy.deepcopy(base)
        tag = f"{lam:g}".replace(".", "p")
        name = f"r6_l1_rawhuber_lam{tag}_s{SEED}"
        cfg["name"] = name
        cfg["notes"] = (
            "W2 L1: log-z MSE main loss + raw-space scaled Huber aux; "
            f"lambda={lam}; fixed D(xyz+geom) input; frozen B1/dev1/FPS2000 protocol; "
            f"val-only; single-seed Gate; seed={SEED}. "
            "L0 loss-control = r6_scale_D_xyzgeom (lambda=0)."
        )
        cfg["train"]["seed"] = SEED
        cfg["train"]["loss_raw_huber_lambda"] = lam
        cfg["train"]["raw_huber_delta"] = 1.0
        path = config_path(name)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(cfg, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        manifest.append(repo_config_path(name))

    manifest_path = SWEEPS_DIR / "loss_l1_lambda.txt"
    manifest_path.write_text("\n".join(manifest) + "\n", encoding="utf-8")

    protocol = {
        "protocol_id": "loss_l1_rawhuber_v1",
        "status": "PRE_REGISTERED",
        "purpose": "W2 L1: does a raw-space scaled Huber auxiliary on top of log-z MSE improve high-WSS amplitude/localization without hurting field/casemean R2?",
        "fixed_input": "D (xyz + abscissa_norm + local_radius + curvature)",
        "loss_control_L0": repo_config_path("r6_scale_D_xyzgeom_s1234"),
        "lambdas": list(LAMBDAS),
        "seed_gate": SEED,
        "raw_huber_delta": 1.0,
        "primary_metrics": ["r2_field_casebalanced", "r2_casemean", "r2_field_raw"],
        "guardrails": [
            "top10_pred_true_ratio",
            "p99_pred_true_ratio",
            "top10_iou",
            "high_wss_mae",
            "negative_r2_count",
        ],
        "decision_rules": {
            "gate": "single-seed screening only; pick the lambda interval where high-WSS amplitude/IoU improve and field/casemean R2 do not regress > seed noise vs L0",
            "confirm": "add three seeds for at most one selected lambda in a separate submission; never promote from single seed",
            "no_go": "if every lambda hurts field/casemean R2 or peaks (max_pred_true_ratio) without improving hotspot, stop L1",
            "test_policy": "No legacy test16 access during this diagnostic",
        },
    }
    (OUT_DIR / "loss_l1_rawhuber_protocol.json").write_text(
        json.dumps(protocol, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(f"generated {len(manifest)} L1 configs -> {manifest_path}")


if __name__ == "__main__":
    main()

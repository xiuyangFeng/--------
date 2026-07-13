#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""第六轮 W3 确认阶段配置生成（预注册于 W1/W2 单 seed 结果之后）。

单 seed 结果（dev1, val）：
- W1：E(`xyz+scale+geom`) 为最佳可部署输入，field_cb 0.330，E−D field_cb +0.021 过 Gate-1；
  但交互 E−D−B+A<0，尺度/几何冗余。
- W2-L1：raw-Huber λ∈{0.1,0.3,1.0} 全面优于 L0；λ=1.0 最佳（field 0.385/casemean 0.253/top10 0.463），
  无爆峰（max_ratio 0.16）。单 seed 仅筛选，选中 λ=1.0 补三 seed。

本文件落两批确认性作业（seed 全部 1234/7/2025，其余协议冻结）：
1. **L1 λ=1.0 三 seed 确认**（loss-control 用于 W3）：D 输入 + raw-Huber λ=1.0。
   s1234 已存在（r6_l1_rawhuber_lam1_s1234），此处补 s7 / s2025。
2. **W3 唯一组合**：E 输入 + raw-Huber λ=1.0，三 seed。W3 Gate = 同时优于
   input-control（E, λ=0，已完成三 seed）与 loss-control（D + λ=1.0 三 seed）。

用法：
    python -m training_wss_min.tools.make_configs_round6_w3
"""

from __future__ import annotations

import copy
import json

from training_wss_min.tools.config_paths import (
    SWEEPS_DIR,
    config_path,
    repo_config_path,
)

SEEDS = (1234, 7, 2025)
CONFIRM_LAMBDA = 1.0


def _emit(name: str, base_name: str, features: list, notes: str, seed: int) -> str:
    base = json.loads(config_path(base_name).read_text(encoding="utf-8"))
    cfg = copy.deepcopy(base)
    cfg["name"] = name
    cfg["notes"] = notes
    cfg["data"]["input_features"] = features
    cfg["train"]["seed"] = seed
    cfg["train"]["loss_raw_huber_lambda"] = CONFIRM_LAMBDA
    cfg["train"]["raw_huber_delta"] = 1.0
    p = config_path(name)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(cfg, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return repo_config_path(name)


def main() -> None:
    D_FEATS = ["x", "y", "z", "abscissa_norm", "local_radius", "curvature"]
    E_FEATS = ["x", "y", "z", "coord_scale", "abscissa_norm", "local_radius", "curvature"]

    l1_confirm = []
    for seed in (7, 2025):  # s1234 已存在
        name = f"r6_l1_rawhuber_lam1_s{seed}"
        note = (f"W2-L1 confirm: log-z MSE + raw-Huber lambda={CONFIRM_LAMBDA}; fixed D input; "
                f"frozen B1/dev1/FPS2000; val-only; three-seed confirm of single-seed winner; seed={seed}")
        l1_confirm.append(_emit(name, f"r6_scale_D_xyzgeom_s{seed}", D_FEATS, note, seed))

    w3 = []
    for seed in SEEDS:
        name = f"r6_w3_e_rawhuber_lam1_s{seed}"
        note = (f"W3 unique combo: best input E(xyz+scale+geom) x best loss raw-Huber lambda={CONFIRM_LAMBDA}; "
                f"frozen B1/dev1/FPS2000; val-only; Gate vs input-control E(lambda=0) and loss-control D+lambda=1; seed={seed}")
        w3.append(_emit(name, f"r6_scale_E_xyzscalegeom_s{seed}", E_FEATS, note, seed))

    manifest = l1_confirm + w3
    manifest_path = SWEEPS_DIR / "round6_w3_confirm.txt"
    manifest_path.write_text("\n".join(manifest) + "\n", encoding="utf-8")
    print(f"L1 λ=1.0 confirm (s7/s2025): {len(l1_confirm)}")
    print(f"W3 E×rawHuber-λ1.0 (3 seed): {len(w3)}")
    print(f"manifest -> {manifest_path}")
    for m in manifest:
        print("  ", m)


if __name__ == "__main__":
    main()

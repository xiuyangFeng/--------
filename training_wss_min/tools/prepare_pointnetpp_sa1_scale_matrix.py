#!/usr/bin/env python3
"""Materialize the frozen 17-run SA1-scale matrix (points x centers x k x stem, seed 1234).

方向（2026-07-21，锚点 Q1V/SAME + KNN-cover 家族）：
- D1-fixed  全点支撑 + 固定 500/125/32 center + SA1 knn_cover k∈{64,128,256}
- D1-prop   全点支撑 + 三层比例 center（ratios 0.1/0.25/0.25）+ 同一 k 网格
- D2        random5000 + SA1 center {250,125} x k {64,128} 全 2x2（SA2/3 保持 125/32）
- D3        random10000（欠点例回退全点）+ 固定 center + 同一 k 网格
- D4        random5000 + knn{8,10}_cover 的 width=64 对照
- D5-A      D4-k8 锚点上的 Stem 6->32->64 变种（w64）
- bridge    random5000 + 固定 center + k64（分离“加点数”与“加 k”）
条件触发 D5-B（Stem 6->256->512->64）由 --variant-b 单独物化，不进主 manifest。
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

from training_wss_min.config import ExpConfig


REPO = Path(__file__).resolve().parents[2]
Q1V = REPO / "training_wss_min/configs/pointnetpp_v4/ag_aaa_v4_stratified_sa3_random5000_fpscenter_same.json"
GROUPING_DIR = REPO / "training_wss_min/configs/pointnetpp_sa_grouping_single_seed_20260720"
OUT_DIR = REPO / "training_wss_min/configs/pointnetpp_sa1_scale_single_seed_20260721"
MANIFEST = REPO / "training_wss_min/preflight/pointnetpp_sa1_scale_prepared.json"
VARIANT_B_MANIFEST = REPO / "training_wss_min/preflight/pointnetpp_sa1_scale_variant_b_prepared.json"
RUN_PREFIX = "pointnetpp_sa1_scale_single_seed/outputs"


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def make(base: dict, experiment_id: str, family: str, note: str, *,
         wall_points=5000, sa1_k=8, center_counts=(500, 125, 32),
         ratios=(0.25, 0.25, 0.25), width=32, stem=(), batch=8,
         allow_undersized=False) -> tuple[dict, dict]:
    cfg = copy.deepcopy(base)
    cfg["name"] = f"{RUN_PREFIX}/{experiment_id}"
    cfg["notes"] = (
        f"2026-07-21 SA1-scale matrix; one seed only (1234); exact split 106/0/27. "
        f"{note}"
    )
    cfg["train"]["seed"] = 1234
    cfg["train"]["batch_cases"] = int(batch)
    all_points = int(wall_points) <= 0
    cfg["data"]["wall_n_points"] = int(wall_points)
    cfg["data"]["sampling"] = "random"
    cfg["data"]["support_n_points"] = int(wall_points)
    cfg["data"]["support_sampling"] = "random"
    cfg["data"]["query_mode"] = "same"
    cfg["data"]["query_n_points"] = int(wall_points)
    cfg["data"]["query_sampling"] = "random"
    # 全点支撑下重采样是 no-op，显式关闭以免误读
    cfg["data"]["resample_each_epoch"] = not all_points
    if allow_undersized:
        cfg["data"]["support_allow_undersized"] = True
    cfg["model"]["width"] = int(width)
    cfg["model"]["sa_nsample"] = [int(sa1_k), 16, 16]
    cfg["model"]["sa_grouping"] = ["knn_cover", "ball", "ball"]
    cfg["model"]["sa_center_counts"] = [int(n) for n in center_counts]
    cfg["model"]["sa_ratios"] = [float(r) for r in ratios]
    if stem:
        cfg["model"]["stem_channels"] = [int(c) for c in stem]
    row = {
        "experiment_id": experiment_id,
        "family": family,
        "support_points": "all" if all_points else int(wall_points),
        "sa1_centers": int(center_counts[0]) if center_counts else f"prop{ratios[0]}",
        "sa1_k": int(sa1_k),
        "width": int(width),
        "stem_channels": [int(c) for c in stem],
        "batch_cases": int(batch),
        "seed": 1234,
    }
    return cfg, row


def build_specs(base: dict) -> list[tuple[dict, dict]]:
    specs: list[tuple[dict, dict]] = []
    add = lambda *args, **kwargs: specs.append(make(*args, **kwargs))

    # D1-fixed：全点支撑，固定 center，与 Q1V 唯一差 = 支撑点数 + SA1 k/grouping
    for k in (64, 128, 256):
        add(base, f"d1_allpts_fixed500_k{k}", "d1_fixed",
            f"D1-fixed: all wall points as support; fixed centers 500/125/32; SA1 knn_cover k={k}.",
            wall_points=0, sa1_k=k)
    # D1-prop：全点支撑，三层比例 center（0.1N/0.25/0.25）。2-epoch 干跑实测
    # k256 在 batch4 时峰值 23.3GB/24GB，按预注册门槛整族降 batch 2。
    for k in (64, 128, 256):
        add(base, f"d1_allpts_prop10pct_k{k}", "d1_prop",
            f"D1-prop: all wall points; proportional centers ratios 0.1/0.25/0.25 (all SA stages); "
            f"SA1 knn_cover k={k}; batch_cases=2 for worst-batch memory (smoke: 23.3GB at batch4).",
            wall_points=0, sa1_k=k, center_counts=(), ratios=(0.1, 0.25, 0.25), batch=2)
    # D2：random5000，降 SA1 center x 升 k 全 2x2；SA2/3 保持 125/32
    for c in (250, 125):
        for k in (64, 128):
            add(base, f"d2_rand5000_c{c}_k{k}", "d2_centers",
                f"D2: random5000; SA1 centers {c} (SA2/3 stay 125/32); SA1 knn_cover k={k}. "
                f"250x64 and 125x128 are budget-matched (16000 assignments) vs baseline 500x8.",
                sa1_k=k, center_counts=(c, 125, 32))
    # D3：random10000；5 个 <10k 的 AG 例经 support_allow_undersized 回退全点
    for k in (64, 128, 256):
        add(base, f"d3_rand10000_fixed500_k{k}", "d3_points",
            f"D3: random10000 support (undersized AG cases fall back to all points); "
            f"fixed centers 500/125/32; SA1 knn_cover k={k}.",
            wall_points=10000, sa1_k=k, allow_undersized=True)
    # D4：cover 系 width=64 对照
    for k in (8, 10):
        add(base, f"d4_rand5000_knn{k}_cover_w64", "d4_width",
            f"D4: width=64 control of q1v_sa1_knn{k}_cover (random5000, SA1 knn_cover k={k}).",
            sa1_k=k, width=64)
    # D5-A：KNN-8-cover w64 锚点上的 Stem 6->32->64 变种
    add(base, "d5_rand5000_knn8_cover_w64_stem32_64", "d5_stem",
        "D5-A: stem bottleneck 6->32->64 under width=64 KNN-8-cover; standard-stem control "
        "is d4_rand5000_knn8_cover_w64 (6->64->64).",
        sa1_k=8, width=64, stem=(32, 64))
    # bridge：random5000 + 固定 center + k64，分离“点数”与“k”两因素
    add(base, "bridge_rand5000_fixed500_k64", "bridge",
        "Bridge control: random5000 with fixed centers 500/125/32 and SA1 knn_cover k=64; "
        "separates points-scaling from k-scaling vs D1/D3.",
        sa1_k=64)
    return specs


def build_variant_b(base: dict) -> list[tuple[dict, dict]]:
    return [make(
        base, "d5_rand5000_knn8_cover_w64_stem256_512", "d5_stem",
        "D5-B (conditional): heavy per-point front stem 6->256->512->64 feeding the fixed "
        "64->128->256->512 SA/FP backbone; parent d5_rand5000_knn8_cover_w64_stem32_64.",
        sa1_k=8, width=64, stem=(256, 512, 64),
    )]


def materialize(specs: list[tuple[dict, dict]], manifest_path: Path,
                expected: int, policy_note: str, base_split_path: str) -> None:
    if len(specs) != expected:
        raise AssertionError(f"expected {expected} specs, got {len(specs)}")
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    rows = []
    for cfg_dict, row in specs:
        path = OUT_DIR / f"{row['experiment_id']}.json"
        path.write_text(json.dumps(cfg_dict, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        cfg = ExpConfig.from_json(path)
        if cfg.run_dir.exists():
            raise FileExistsError(f"refusing existing run dir: {cfg.run_dir}")
        rows.append({**row, "config": str(path.resolve()), "sha256": sha(path),
                     "run_dir": str(cfg.run_dir)})

    split = load(Path(base_split_path))
    counts = {p: len(split.get(f"{p}_cases", [])) for p in ("train", "val", "test")}
    if counts != {"train": 106, "val": 0, "test": 27}:
        raise RuntimeError(f"unexpected split counts: {counts}")
    payload = {
        "schema_version": 1,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "status": "prepared",
        "policy": {"seeds": [1234], "split_counts": counts, "new_jobs": expected,
                   "note": policy_note,
                   "knn_cover_gates": {"sa1_coverage": 1.0}},
        "reused_controls": [
            {"id": "Q1V_random5000_ball16_anchor", "config": str(Q1V),
             "run_dir": str(ExpConfig.from_json(Q1V).run_dir)},
            {"id": "q1v_sa1_knn8_cover", "config": str(GROUPING_DIR / "q1v_sa1_knn8_cover.json"),
             "run_dir": str(ExpConfig.from_json(GROUPING_DIR / "q1v_sa1_knn8_cover.json").run_dir)},
            {"id": "q1v_sa1_knn10_cover", "config": str(GROUPING_DIR / "q1v_sa1_knn10_cover.json"),
             "run_dir": str(ExpConfig.from_json(GROUPING_DIR / "q1v_sa1_knn10_cover.json").run_dir)},
            {"id": "q1v_n16_w64", "config": str(GROUPING_DIR / "q1v_n16_w64.json"),
             "run_dir": str(ExpConfig.from_json(GROUPING_DIR / "q1v_n16_w64.json").run_dir)},
        ],
        "configs": rows,
    }
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": "prepared", "configs": len(rows),
                      "manifest": str(manifest_path)}, ensure_ascii=False, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--variant-b", action="store_true",
                        help="物化条件触发的 D5-B（stem 6->256->512->64），写独立 manifest")
    args = parser.parse_args()
    base = load(Q1V)
    if args.variant_b:
        materialize(build_variant_b(base), VARIANT_B_MANIFEST, 1,
                    "conditional D5-B; run only if D5-A beats d4_rand5000_knn8_cover_w64 "
                    "on physical R2_cb with remaining train-fit headroom",
                    base["data"]["split_path"])
    else:
        materialize(build_specs(base), MANIFEST, 17,
                    "17 new jobs: 3 d1_fixed + 3 d1_prop + 4 d2_centers + 3 d3_points "
                    "+ 2 d4_width + 1 d5_stem + 1 bridge",
                    base["data"]["split_path"])


if __name__ == "__main__":
    main()

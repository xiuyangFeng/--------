"""Generate the complete, preregistered P00-P11/V00-V11/J00-J01 matrix."""
from __future__ import annotations

import argparse
import copy
import json

from training_wss_min import config as C
from training_wss_min.tools.prepare_v6_singleframe_matrix import realised_diff
from training_wss_min.tools.volume_attention_common import CONFIGS, EXP, MATRIX, NAME, OLD, ROOT, RUNS, save_json, sha256

DESCRIPTIONS = [
    "原R5体场配置同期从零对照", "原R5加M2局部壁面分支L", "原R5原SA3后直接加全局BT",
    "L加原SA3后全局BT（直接接入主候选）", "L加仅中心条件编码（layers=0）",
    "L加中心条件及近等参数逐token FFN", "L加新SA3单半径0.20接入，无BT",
    "新SA3单半径0.20加BT", "新SA3三半径0.10/0.20/0.40，无BT",
    "新SA3三半径独立BT（老师完整主候选）", "相同0.20半径三支独立BT",
    "新SA3三半径近等参数逐token FFN",
]
PARENTS = [None, 0, 0, 1, 1, 4, 1, 6, 6, 8, 9, 8]
COMPARISONS = [[], [0], [0], [1, 2, 4, 5], [1, 3], [3, 4], [1], [6, 3], [6], [8, 7, 3, 10, 11], [9], [9, 8]]


def base_config(prefix):
    original = ROOT / "training_wss_min/configs/v5_rerun_20260906" / (OLD[prefix].split("/")[-1] + ".json")
    cfg = json.loads(original.read_text())
    cfg["data"]["feature_stats_path"] = str(RUNS / OLD[prefix] / "feature_stats.json")
    cfg["eval"]["stability_seeds"] = [1234]
    cfg["train"]["log_loss_components"] = True
    cfg["model"].update(bottleneck_transformer=False, bottleneck_mode="attention",
                        output_head="single", multiradius_values=[], local_branch=False)
    return cfg


def apply_structure(cfg, index):
    m = cfg["model"]
    m.update(bottleneck_layers=1, bottleneck_heads=8, bottleneck_ffn_ratio=2,
             bottleneck_dropout=0., bottleneck_pos_enc="input_features", bottleneck_geo_bias=True,
             bottleneck_residual_scale_init=.001)
    if index not in (0, 2):
        m.update(local_branch=True, local_branch_radii=[.015, .03, .06],
                 local_branch_nsample=[16, 16, 16], local_branch_channels=32,
                 local_branch_feature_indices=[14, 15, 16])
    if index in (2, 3, 4, 5):
        m["bottleneck_transformer"] = True
        if index == 4:
            m["bottleneck_layers"] = 0
        if index == 5:
            m["bottleneck_mode"] = "token_ffn"
    if index >= 6:
        m["sa_nsample"][-1] = 32
        m["multiradius_values"] = [.2] if index <= 7 else [.2, .2, .2] if index == 10 else [.1, .2, .4]
        m["multiradius_bottleneck"] = index in (7, 9, 10)
        m["multiradius_ffn"] = index == 11


def build_payloads():
    configs, arms, payloads = {}, [], {}
    for prefix in ("P", "V"):
        base = base_config(prefix)
        for i in range(12):
            aid = f"{prefix}{i:02d}"
            cfg = copy.deepcopy(base)
            cfg["name"] = f"{NAME}/{aid}_s1234"
            cfg["notes"] = f"{aid}: {DESCRIPTIONS[i]}。18D、峰值1162、单seed1234；体场纯MSE；test34开发暴露集。"
            apply_structure(cfg, i)
            if i:
                cfg["train"]["init_reference_config"] = str(CONFIGS / f"{prefix}00_s1234.json")
            configs[aid] = cfg
            payloads[CONFIGS / f"{aid}_s1234.json"] = cfg
            parent = f"{prefix}{PARENTS[i]:02d}" if PARENTS[i] is not None else f"R5{prefix}"
            arms.append(dict(id=aid, config=f"{aid}_s1234.json", run_name=cfg["name"],
                             target=cfg["data"]["target"], task="pressure" if prefix == "P" else "velocity",
                             parent=parent, comparisons=[f"{prefix}{n:02d}" for n in COMPARISONS[i]],
                             phase=1 if i < 6 else 2, hypothesis=DESCRIPTIONS[i],
                             changes_vs_base=realised_diff(configs[f"{prefix}00"], cfg),
                             changes_vs_parent=realised_diff(configs.get(parent, base), cfg)))
    # Same ordered 18D fields and frozen feature statistics in both tasks.
    pstats = json.loads((RUNS / OLD["P"] / "feature_stats.json").read_text())
    vstats = json.loads((RUNS / OLD["V"] / "feature_stats.json").read_text())
    if pstats != vstats:
        raise ValueError("P/V historical feature statistics differ; joint cannot silently choose new scales")
    joint_base = base_config("V")
    joint_base["name"] = f"{NAME}/references/J_base_untrained"
    joint_base["notes"] = "仅用于同seed fresh initialization的无L无BT联合结构；不训练、不计入26臂。"
    joint_base["data"].update(target="velocity_pressure", query_wall_fraction=.5)
    joint_base["model"].update(output_head="velocity_pressure", out_dim=4)
    payloads[CONFIGS / "references/J_base.json"] = joint_base
    for i in range(2):
        aid = f"J{i:02d}"
        cfg = copy.deepcopy(joint_base)
        apply_structure(cfg, 1 if i == 0 else 3)
        cfg["name"] = f"{NAME}/{aid}_s1234"
        cfg["notes"] = "联合速度压力，等权任务z-MSE，分别5000监督点union；联合对单任务last400主比较。"
        cfg["train"]["init_reference_config"] = str(CONFIGS / ("references/J_base.json" if i == 0 else "J00_s1234.json"))
        payloads[CONFIGS / f"{aid}_s1234.json"] = cfg
        arms.append(dict(id=aid, config=f"{aid}_s1234.json", run_name=cfg["name"],
                         target="velocity_pressure", task="joint", phase=3, parent="P01+V01" if i == 0 else "J00",
                         comparisons=["P01", "V01"] if i == 0 else ["J00", "P03", "V03"],
                         hypothesis="共享结构双任务" if i == 0 else "共享结构双任务加全局BT",
                         changes_vs_parent=realised_diff(joint_base if i == 0 else payloads[CONFIGS / "J00_s1234.json"], cfg)))
    manifest = dict(matrix_id=NAME, training_runs=26, seeds=[1234], arms=arms,
                    protocol=dict(input_dim=18, epochs=400, batch_cases=8, peak=1162, support=5000,
                                  queries_per_task=5000, pressure_wall_fraction=.5, split="train138/test34",
                                  best="minimum training loss", last="epoch399 (400 completed epochs)",
                                  joint_comparison="last400 primary; best supplementary",
                                  evaluation="fixed support seed1234; all query points; no multi-seed repeat"),
                    references={f"R5{k}": dict(run_name=v, feature_stats_sha256=sha256(RUNS/v/"feature_stats.json"),
                                               best_sha256=sha256(RUNS/v/"ckpt_best.pt"), last_sha256=sha256(RUNS/v/"ckpt_last.pt")) for k,v in OLD.items()},
                    gates=dict(best_r2_gain=.01, pressure_mae_reduction=.03, velocity_vector_rmse_reduction=.03,
                               secondary_component_r2_max_drop=.01, last="same direction", longwave_debiased_mse_reduction=.10,
                               longwave_sigmas_mm=[20,40], inference="single-seed exposed-development screening only"),
                    factorials=[["00","01","02","03"], ["06","07","08","09"]])
    payloads[MATRIX] = manifest
    return payloads


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true")
    args = ap.parse_args()
    payloads = build_payloads()
    for path, cfg in payloads.items():
        if path != MATRIX:
            C.ExpConfig.from_dict(cfg)
        if args.check:
            if not path.is_file() or json.loads(path.read_text()) != cfg:
                raise ValueError(f"Stale/missing matrix artifact: {path}")
        else:
            save_json(path, cfg)
    print(f"{'Checked' if args.check else 'Generated'} 26 single-seed runs + 1 untrained initialization reference")


if __name__ == "__main__":
    main()

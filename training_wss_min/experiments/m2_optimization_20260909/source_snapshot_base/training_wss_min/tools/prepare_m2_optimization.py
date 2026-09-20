"""Prepare/check the 20 authorised independent M2 optimisation runs."""
from __future__ import annotations

import argparse
import copy
import json

from training_wss_min import config as C
from training_wss_min.tools.m2_optimization_common import (
    ANCHOR_CONFIG, ANCHOR_RUN, CONFIGS, EXP, MATRIX, PREFIX, ROOT, save_json, sha,
)
from training_wss_min.tools.prepare_v6_singleframe_matrix import realised_diff, set_path


def specifications():
    arms = [("MO0", "control", {}, "原M2同期从零训练对照", "M2")]
    for i, value in enumerate((.01, .03, .10), 1):
        arms.append((f"MO-L{i}", "L", {"train.loss_raw_mse_lambda": value},
                     f"原M2损失增加Pa-MSE，lambda={value}", "MO0"))
    for i, (weight, delta) in enumerate(((.01, 1.), (.03, 1.), (.03, 3.)), 4):
        arms.append((f"MO-L{i}", "L", {"train.loss_raw_huber_lambda": weight,
                     "train.raw_huber_delta": delta},
                     f"原M2损失增加Pa-Huber，lambda={weight}，delta={delta}", "MO0"))
    for i, alpha in enumerate((.5, 1.), 7):
        arms.append((f"MO-L{i}", "L", {"train.loss_weight_target": True,
                     "train.loss_weight_target_alpha": alpha},
                     f"log-MSE目标幅值加权alpha={alpha}，pinball不变", "MO0"))
    for i, (key, value) in enumerate((("train.lr", .0005), ("train.lr", .0015),
            ("train.weight_decay", .0005), ("model.drop_path_rate", .05),
            ("model.drop_path_rate", .20)), 1):
        arms.append((f"MO-P{i}", "P", {key: value}, f"只改{key}={value}", "MO0"))
    decoder = {"model.query_decoder": "local_attn", "model.query_decoder_k": 3,
               "model.query_decoder_hidden": 32, "model.query_decoder_residual": False,
               "model.query_decoder_feature_indices": [4, 14, 15, 16], "model.sep_beta_init": 2.}
    arms.extend([
        ("MO-S1", "S", decoder, "保持3NN，可学几何条件插值权重，无query残差", "MO0"),
        ("MO-S2", "S", {**decoder, "model.query_decoder_residual": True},
         "S1仅增加零初始化query残差", "MO-S1"),
        ("MO-S3", "S", {"model.local_branch_modulation": "gate", "model.local_branch_modulation_hidden": 16},
         "原三尺度96维池化特征乘性门控96→16→96，末层零初始化", "MO0"),
        ("MO-S4", "S", {"model.local_branch_modulation": "additive", "model.local_branch_modulation_hidden": 16},
         "与S3同容量/初始化的96维加性残差对照", "MO0"),
        ("MO-S5", "S", {"model.local_branch_channels": 64}, "局部分支通道32→64", "MO0"),
        ("MO-S6", "S", {"model.sa_blocks": [2, 1, 0]}, "SA1增加一个InvRes块；原SA1/SA2 DropPath率保持0/.1", "MO0"),
    ])
    return arms


def payloads():
    original = json.loads(ANCHOR_CONFIG.read_text())
    baseline = copy.deepcopy(original)
    baseline["train"].update(init_reference_config=str(ANCHOR_CONFIG), log_loss_components=True)
    assert len(baseline["data"]["input_features"]) == 25
    assert baseline["data"]["point_features_root"].endswith("wss_min_geom_v2")
    entries, configs = [], {}
    for aid, family, changes, hypothesis, parent in specifications():
        config = copy.deepcopy(baseline)
        config["name"] = f"{PREFIX}/{aid}_s1234"
        config["notes"] = hypothesis + "；400epoch/s1234；原M2数据与几何；已暴露test34探索。"
        for key, value in changes.items():
            set_path(config, key, value)
        C.ExpConfig.from_dict(config)
        diff = realised_diff(baseline, config)
        if set(diff) - set(changes):
            raise ValueError(f"Undeclared delta {aid}: {diff}")
        assert config["data"] == original["data"]
        assert config["eval"] == original["eval"]
        configs[aid] = config
        entries.append(dict(id=aid, family=family, category=family, parent=parent, baseline=parent,
            phase="base", config=f"{aid}_s1234.json", run_name=config["name"], hypothesis=hypothesis,
            changes=changes, resolved_config=config))
    for arm in entries:
        parent = original if arm["parent"] == "M2" else configs[arm["parent"]]
        arm["changes_vs_parent"] = realised_diff(
            json.loads(json.dumps(C.ExpConfig.from_dict(parent).to_dict())),
            json.loads(json.dumps(C.ExpConfig.from_dict(arm["resolved_config"]).to_dict())))
    return dict(matrix_id=PREFIX, phase="base", anchor_config=str(ANCHOR_CONFIG),
        anchor_sha256=sha(ANCHOR_CONFIG), reference=dict(id="M2", run_name=str(ANCHOR_RUN.relative_to(ROOT / "training_wss_min/runs"))),
        protocol=dict(seed=1234, epochs=400, target="wss", timesteps="peak", train_cases=138,
            test_cases=34, test_wall_points=1231295, input_dim=25, batch_cases=8,
            support_n_points=5000, query_n_points=5000, selection_rule="train_loss",
            new_geometry_allowed=False, maximum_total_training_runs=24),
        selection=dict(delta_best=.01, delta_last=.005, mae_relative_increase=.02,
            iou_decrease=.01, case_p10_decrease=.02, maximum_negative_cases=0),
        arms=entries)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true")
    args = ap.parse_args()
    matrix = json.loads(json.dumps(payloads(), ensure_ascii=False))
    outputs = {CONFIGS / a["config"]: a["resolved_config"] for a in matrix["arms"]}
    outputs[MATRIX] = matrix
    for path, payload in outputs.items():
        if args.check:
            if not path.exists() or json.loads(path.read_text()) != payload:
                raise ValueError(f"Stale or missing config: {path}")
        else:
            if path.exists() and json.loads(path.read_text()) != payload:
                raise RuntimeError(f"Refusing to overwrite different matrix content: {path}")
            save_json(path, payload)
    print(f"{'Checked' if args.check else 'Prepared'} {len(matrix['arms'])} M2 arms")


if __name__ == "__main__":
    main()

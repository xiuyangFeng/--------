"""Summarize completed M2 mechanism diagnostics without model inference.

The producer's complete per-case distributions are reduced with equal case
weights. Activations describe the saved models, not causes of prediction gains.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from statistics import fmean, pstdev

from training_wss_min.tools import report_m2_optimization as R
from training_wss_min.tools.m2_optimization_common import (
    COMBINATIONS, EXP, MATRIX, save_json, sha, stamp,
)


LIMITS = (
    "单 seed1234、已暴露 test34 的训练后描述。所有病例统计先在每例完整 query 或固定 support 上计算，"
    "再对 34 例等权；未按病例点数加权，未平均 query chunk。"
    "参数与 β/gamma 是各 checkpoint 的模型标量，不是病例平均。"
    "激活、门控和残差幅度不表示因果重要性，也不解释模型性能改善的原因。"
)


def require(value, message):
    if not value:
        raise ValueError(message)


def finite_tree(value, label="provenance"):
    if isinstance(value, dict):
        for key, child in value.items():
            finite_tree(child, f"{label}.{key}")
    elif isinstance(value, list):
        for index, child in enumerate(value):
            finite_tree(child, f"{label}[{index}]")
    elif isinstance(value, (int, float)) and not isinstance(value, bool):
        require(math.isfinite(value), f"nonfinite diagnostic value: {label}")


def case_stats(values):
    values = list(values)
    require(len(values) == 34, "case aggregate must contain all 34 cases")
    return {"n_cases": 34, "mean": fmean(values), "between_case_std": pstdev(values),
            "min": min(values), "max": max(values)}


def distribution(value, expected_n=None):
    require(isinstance(value["n"], int) and value["n"] > 0, "empty diagnostic distribution")
    if expected_n is not None:
        require(value["n"] == expected_n, "distribution does not cover the expected points")
    require(value["std"] >= 0 and value["rms"] >= 0 and value["mean_absolute"] >= 0,
            "invalid diagnostic distribution magnitude")
    require(value["min"] - 1e-6 <= value["mean"] <= value["max"] + 1e-6,
            "distribution mean outside recorded bounds")
    return value


def summarize_checkpoint(label, checkpoint, item):
    model, cases = item["model"], item["cases"]
    require(isinstance(model["parameters"], int) and model["parameters"] > 0
            and isinstance(model["local_wall_parameters"], int)
            and 0 <= model["local_wall_parameters"] <= model["parameters"], "invalid parameter counts")
    output = {"checkpoint_epoch": item["checkpoint_epoch"], "parameters": model["parameters"],
              "local_wall_parameters": model["local_wall_parameters"], "query": None,
              "local_modulation": None, "new_sa1_blocks": model["new_sa1_blocks"]}
    query = model.get("query")
    modes = {case["mechanisms"].get("local_modulation", {}).get("mode") for case in cases.values()}
    require(len(modes) == 1, f"{label}/{checkpoint}: inconsistent modulation coverage")
    mode = next(iter(modes))
    require(mode in {None, "gate", "additive"}, "unknown local modulation mode")
    if label in {"MO-S1", "MO-S2"}:
        require(query is not None and query["k"] == 3
                and query["residual_enabled"] is (label == "MO-S2"), f"{label}: query mechanism differs")
    if label in {"MO-S3", "MO-S4"}:
        require(mode == ("gate" if label == "MO-S3" else "additive"), f"{label}: modulation differs")
    if label == "MO-S6":
        require([block["index"] for block in model["new_sa1_blocks"]] == [1], "S6 added SA1 block missing")
    query_cases, modulation_cases = {}, {}
    for case_id, case in cases.items():
        mechanism, points = case["mechanisms"], case["wall_points"]
        require(("query_attention" in mechanism) == (query is not None), f"{label}/{case_id}: query coverage differs")
        residual = mechanism.get("query_residual")
        require((residual is not None) == bool(query and query["residual_enabled"]),
                f"{label}/{case_id}: residual coverage differs")
        if query is not None:
            attention = mechanism["query_attention"]
            entropy = distribution(attention["entropy"], points)
            effective = distribution(attention["effective_neighbors"], points)
            require(-1e-5 <= entropy["min"] <= entropy["max"] <= math.log(query["k"]) + 1e-5
                    and 1 - 1e-5 <= effective["min"] <= effective["max"] <= query["k"] + 1e-5,
                    "query entropy/effective-neighbor range differs")
            row = {"wall_points": points, "entropy_mean_nats": entropy["mean"],
                   "entropy_within_case_std": entropy["std"], "effective_neighbors_mean": effective["mean"],
                   "effective_neighbors_within_case_std": effective["std"],
                   "hidden_reconstruction_max_abs": attention["hidden_reconstruction_max_abs"]}
            if residual is not None:
                physical = distribution(residual["same_weights_output_impact_pa"], points)
                high = distribution(residual["true_high_wss_output_impact_pa"])
                normalized = distribution(residual["log_z"], points)
                require(high["n"] <= points, "high-WSS residual region exceeds wall")
                row.update(residual_rms_pa=physical["rms"], residual_mean_pa=physical["mean"],
                           residual_mean_absolute_pa=physical["mean_absolute"],
                           residual_high_wss_mean_pa=high["mean"], residual_high_wss_rms_pa=high["rms"],
                           residual_high_wss_points=high["n"], residual_rms_log_z=normalized["rms"])
            query_cases[case_id] = row
        if mode is not None:
            modulation = mechanism["local_modulation"]
            support = modulation["support_points"]
            require(support == 5000, "modulation must cover the original 5000 support points")
            activation = modulation["activation"]
            module_output = distribution(modulation["module_output"])
            require(module_output["n"] == support * 96, "original three-scale 96-channel modulation shape differs")
            row = {"support_points": support, "module_output_mean": module_output["mean"],
                   "module_output_rms": module_output["rms"]}
            for key in ("modulation_residual_relative_l2", "modulation_residual_l2", "fused_modulation_delta_l2"):
                require(activation[key] >= 0, "negative modulation norm")
                row[key] = activation[key]
            gate = modulation.get("spatial_gate")
            require((gate is not None) == (mode == "gate"), "gate is missing or incorrectly assigned to additive control")
            if gate is not None:
                require(len(gate["point_mean_gate_per_scale"]) == 3, "three-scale gate summary missing")
                row["gate_scales"] = []
                for index, scale in enumerate(gate["point_mean_gate_per_scale"]):
                    distribution(scale, support)
                    require(-1e-6 <= scale["min"] <= scale["max"] <= 2 + 1e-6, "gate outside 2*sigmoid range")
                    row["gate_scales"].append({"scale_index": index, "mean": scale["mean"],
                        "mean_minus_one": scale["mean"] - 1, "within_case_spatial_std": scale["std"],
                        "point_rms_deviation_from_one": math.hypot(scale["std"], scale["mean"] - 1)})
            modulation_cases[case_id] = row
    if query is not None:
        require(math.isclose(query["beta_effective"], max(0., query["beta_parameter"]), abs_tol=1e-7),
                "effective beta differs from clamped parameter")
        stats = {key: case_stats(row[key] for row in query_cases.values()) for key in next(iter(query_cases.values()))
                 if key not in {"wall_points", "residual_high_wss_points", "hidden_reconstruction_max_abs"}}
        output["query"] = {"model_scalars": query, "case_equal": stats, "per_case": query_cases,
            "maximum_hidden_reconstruction_error": max(row["hidden_reconstruction_max_abs"] for row in query_cases.values()),
            "entropy_definition": "natural-log entropy per complete wall query; average of per-case means",
            "effective_neighbors_definition": "exp(entropy) per query, then per-case mean, then equal-case mean; not exp(mean entropy)",
            "residual_definition": "Pa prediction minus prediction after removing the additive log_z query output at the same weights, including physical back-transform/clipping; not a retrained ablation",
            "residual_rms_definition": "arithmetic mean of 34 per-case Pa RMS values; not a pooled RMS",
            "high_wss_impact_definition": "signed mean Pa difference on each case's true-WSS q90 region, then equal-case mean"}
    if mode is not None:
        scalar_keys = ("module_output_mean", "module_output_rms", "modulation_residual_relative_l2",
                       "modulation_residual_l2", "fused_modulation_delta_l2")
        output["local_modulation"] = {"mode": mode, "case_equal": {
            key: case_stats(row[key] for row in modulation_cases.values()) for key in scalar_keys},
            "per_case": modulation_cases, "gate_scales": None,
            "relative_l2_definition": "per case ||modulated-base||_2 / max(||base||_2,1e-12) across all support points/channels, then equal-case mean",
            "module_output_definition": "gate mode: pre-sigmoid logits; additive mode: signed feature additions; neither is in Pa"}
        if mode == "gate":
            output["local_modulation"]["gate_scales"] = [{"scale_index": index,
                **{key: case_stats(row["gate_scales"][index][key] for row in modulation_cases.values())
                   for key in ("mean", "mean_minus_one", "within_case_spatial_std", "point_rms_deviation_from_one")}}
                for index in range(3)]
            output["local_modulation"]["gate_definition"] = (
                "gate=2*sigmoid(logits), identity=1; first average 32 channels within each scale at each support point. "
                "Compute per-case spatial statistics on those 5000 point values, then equal-case means. "
                "Within-case spatial std differs from between-case std; channel averaging can hide opposing channel variation.")
    return output


def summarize(provenance, expected_runs):
    finite_tree(provenance)
    require(provenance["passed"] is True and provenance["smoke"] is False
            and provenance["partition"] == "test34", "completed full diagnostics required")
    require(21 <= len(expected_runs) <= 25 and set(provenance["runs"]) == set(expected_runs),
            "expected historical M2 plus all 20–24 registered trainings")
    require(set(R.BASE_IDS) | {"M2"} <= set(expected_runs), "required base experiment IDs missing")
    cases = set(provenance["geometry"])
    require(provenance["case_count"] == len(cases) == 34, "diagnostic test34 population incomplete")
    points = {case: provenance["geometry"][case]["wall_points"] for case in cases}
    require(all(isinstance(n, int) and n > 0 for n in points.values()) and sum(points.values()) == 1231295,
            "original full-wall point population differs")
    require(bool(provenance["source_sha256_before"])
            and provenance["source_sha256_before"] == provenance["source_sha256_after"], "diagnostic source drift")
    evaluations, checks = 0, 0
    output = {"passed": True, "created_at": stamp(), "limits": LIMITS, "case_count": len(cases),
              "wall_points_each_evaluation": sum(points.values()), "runs": {},
              "interpretation": "Descriptive mechanism statistics only; no performance ranking, threshold or combination decision"}
    for label, expected_name in expected_runs.items():
        checkpoints = provenance["runs"][label]
        require(set(checkpoints) == {"best", "last"}, f"{label}: both diagnostic checkpoints required")
        output["runs"][label] = {}
        for checkpoint, item in checkpoints.items():
            require(item["passed"] is True and item["run_name"] == expected_name and set(item["cases"]) == cases,
                    f"{label}/{checkpoint}: incomplete run/case provenance")
            for case_id, case in item["cases"].items():
                require(case["wall_points"] == points[case_id], f"{case_id}: diagnostic wall population differs")
                verified = case["metric_verification"]
                require(set(verified) == {"Pa", "log_z"} and all(v["passed"] is True
                        and isinstance(v["checks"], int) and v["checks"] > 0 for v in verified.values()),
                        "original metric verification incomplete")
                checks += sum(v["checks"] for v in verified.values())
            output["runs"][label][checkpoint] = summarize_checkpoint(label, checkpoint, item)
            evaluations += 1
        best, last = output["runs"][label]["best"], output["runs"][label]["last"]
        require((best["parameters"], best["local_wall_parameters"]) == (last["parameters"], last["local_wall_parameters"]),
                f"{label}: checkpoint architectures differ")
        require((best["query"] is None) == (last["query"] is None)
                and (best["local_modulation"] is None) == (last["local_modulation"] is None), "checkpoint mechanism mismatch")
        deltas = {"definition": "last minus best within one training trajectory; not independent replications"}
        if best["query"] is not None:
            require(set(best["query"]["case_equal"]) == set(last["query"]["case_equal"]), "checkpoint query fields differ")
            deltas["query"] = {key: last["query"]["case_equal"][key]["mean"] - value["mean"]
                               for key, value in best["query"]["case_equal"].items()}
            deltas["query"].update({key: last["query"]["model_scalars"][key] - best["query"]["model_scalars"][key]
                                    for key in ("beta_parameter", "beta_effective")})
        if best["local_modulation"] is not None:
            b, a = best["local_modulation"], last["local_modulation"]
            require(b["mode"] == a["mode"], "checkpoint modulation mode differs")
            deltas["local_modulation"] = {key: a["case_equal"][key]["mean"] - value["mean"] for key, value in b["case_equal"].items()}
            if b["gate_scales"] is not None:
                deltas["gate_scales"] = [{"scale_index": i, **{key: a["gate_scales"][i][key]["mean"] - value["mean"]
                    for key, value in scale.items() if key != "scale_index"}} for i, scale in enumerate(b["gate_scales"])]
        output["runs"][label]["best_to_last"] = deltas
    require(evaluations == provenance["evaluations_verified"] == 2 * len(expected_runs)
            and 42 <= evaluations <= 50, "42–50 complete checkpoint evaluations required")
    require(checks == provenance["original_per_case_numeric_checks"], "diagnostic numeric-check count differs")
    output.update(evaluations_summarized=evaluations, original_per_case_numeric_checks=checks,
                  validation_boundary="Checks diagnostic JSON coverage, schema and finite values; final acceptance separately verifies source files and saved array/figure hashes. No inference or raw-array recomputation.")
    return output


def markdown(summary):
    lines = ["# M2 训练后机制诊断摘要", "", LIMITS, "",
             f"已完成 {summary['evaluations_summarized']} 次 checkpoint 诊断；每次 34 例、1,231,295 个原壁面点。", "",
             "表内统计为 34 个病例统计量的算术平均。JSON 同时保留逐病例值、病例间总体标准差及极值。"
             "best→last 变化来自同一次训练，不能作为独立重复或性能趋势。", "", "## 参数量", "",
             "| 模型 | 总参数 | 局部分支参数 | 相对 MO0 参数增量 | best/last epoch |",
             "|---|---:|---:|---:|---|"]
    baseline = summary["runs"]["MO0"]["best"]["parameters"]
    for label, run in summary["runs"].items():
        b, a = run["best"], run["last"]
        lines.append(f"| {label} | {b['parameters']} | {b['local_wall_parameters']} | {b['parameters'] - baseline:+d} | {b['checkpoint_epoch']}/{a['checkpoint_epoch']} |")
    lines += ["", "## S1/S2 与启用同机制的组合：完整 query 解码", "",
              "β 是模型标量；熵使用自然对数。有效邻居数先对每个 query 算 exp(H)，再逐病例汇总，不能用 exp(表内平均熵) 替代。", "",
              "| 模型/checkpoint | β 原值/有效值 | query 熵 nats | 有效邻居数 | Pa 残差 RMS | 真值高 WSS 区残差均值 Pa |",
              "|---|---:|---:|---:|---:|---:|"]
    for label, run in summary["runs"].items():
        for checkpoint in ("best", "last"):
            query = run[checkpoint]["query"]
            if query is None:
                continue
            scalar, stats = query["model_scalars"], query["case_equal"]
            rms = f"{stats['residual_rms_pa']['mean']:.6g}" if scalar["residual_enabled"] else "未启用"
            high = f"{stats['residual_high_wss_mean_pa']['mean']:.6g}" if scalar["residual_enabled"] else "未启用"
            lines.append(f"| {label}/{checkpoint} | {scalar['beta_parameter']:.6g}/{scalar['beta_effective']:.6g} | {stats['entropy_mean_nats']['mean']:.6g} | {stats['effective_neighbors_mean']['mean']:.6g} | {rms} | {high} |")
    lines += ["", "Pa 残差是相同权重下移除 query 的 log_z 加性输出后，经过原物理反变换与截断得到的预测差。"
              "RMS 列为每例 Pa RMS 的等权平均，不是全点 pooled RMS；高值列为每例真值 q90 区域的有符号均值，正数表示此输出项提高该区域预测幅值。"
              "这不是重训消融，也不是误差改善量。", "", "## S3/S4 与启用同机制的组合：局部调制", "",
              "relative L2 先在每例的全部 5000 support 点、96 通道上计算 ||modulated−base||₂/max(||base||₂,1e−12)，再病例等权平均。"
              "S4 为加性对照，没有 gate；模块输出 RMS 为特征空间幅度，S3 输出的是 sigmoid 前 logits，不能与 S4 加性输出直接视为同一种门值。", "",
              "| 模型/checkpoint | 模式 | modulation relative L2 | 模块输出 RMS | fused 特征变化 L2 |",
              "|---|---|---:|---:|---:|"]
    for label, run in summary["runs"].items():
        for checkpoint in ("best", "last"):
            modulation = run[checkpoint]["local_modulation"]
            if modulation is not None:
                stats = modulation["case_equal"]
                lines.append(f"| {label}/{checkpoint} | {modulation['mode']} | {stats['modulation_residual_relative_l2']['mean']:.6g} | {stats['module_output_rms']['mean']:.6g} | {stats['fused_modulation_delta_l2']['mean']:.6g} |")
    lines += ["", "gate=2×sigmoid(logits)，恒等门为 1。每个尺度先在各 support 点内平均其 32 通道，再计算病例内空间统计。"
              "空间 std 列是病例内 std 的病例平均，不能解释为病例间离散度；通道平均可能抵消相反变化。尺度 1/2/3 对应原半径 .015/.03/.06。", "",
              "| 模型/checkpoint | 尺度 | gate 均值 | 均值−1 | 病例内空间 std | 对 1 的点 RMS 偏离 |",
              "|---|---:|---:|---:|---:|---:|"]
    for label, run in summary["runs"].items():
        for checkpoint in ("best", "last"):
            modulation = run[checkpoint]["local_modulation"]
            if modulation is not None:
                for scale in modulation["gate_scales"] or []:
                    lines.append(f"| {label}/{checkpoint} | {scale['scale_index'] + 1} | " + " | ".join(
                        f"{scale[key]['mean']:.6g}" for key in ("mean", "mean_minus_one", "within_case_spatial_std", "point_rms_deviation_from_one")) + " |")
    lines += ["", "| 模型/尺度 | last−best gate 均值 | last−best 空间 std |", "|---|---:|---:|"]
    for label, run in summary["runs"].items():
        for scale in run["best_to_last"].get("gate_scales", []):
            lines.append(f"| {label}/{scale['scale_index'] + 1} | {scale['mean']:+.6g} | {scale['within_case_spatial_std']:+.6g} |")
    lines += ["", "## S6 与含新增 SA1 块的组合", "", "gamma 是训练得到的残差系数；幅值不等于该块的因果贡献。", "",
              "| 模型/checkpoint | SA1 块索引（从 0 起） | gamma local | gamma pointwise | DropPath rate |",
              "|---|---:|---:|---:|---:|"]
    for label, run in summary["runs"].items():
        for checkpoint in ("best", "last"):
            for block in run[checkpoint]["new_sa1_blocks"]:
                lines.append(f"| {label}/{checkpoint} | {block['index']} | {block['gamma_local']:.6g} | {block['gamma_pointwise']:.6g} | {block['drop_path_rate']:.6g} |")
    lines += ["", "本摘要不重推理、不重算原预测、不执行模型筛选。完成性与数值检查基于诊断 JSON；原检查点、预测数组和图件哈希由最终验收另行核验。",
              "", f"来源 provenance SHA256：`{summary['source']['sha256']}`。", ""]
    return "\n".join(lines)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--provenance", type=Path, default=EXP / "diagnostics/provenance.json")
    parser.add_argument("--output-dir", type=Path, help="Defaults to the provenance directory")
    parser.add_argument("--matrix", type=Path, default=MATRIX)
    parser.add_argument("--combinations", type=Path, default=COMBINATIONS)
    args = parser.parse_args(argv)
    raw = args.provenance.read_bytes()
    expected = {"M2": R.ANCHOR, **{arm["id"]: arm["run_name"] for arm in R.matrix_arms(args.matrix, args.combinations)}}
    summary = summarize(json.loads(raw), expected)
    summary["source"] = {"path": str(args.provenance.resolve()), "sha256": hashlib.sha256(raw).hexdigest()}
    summary["tool"] = {"path": str(Path(__file__).resolve()), "sha256": sha(__file__)}
    summary["manifests"] = {str(path.resolve()): sha(path) for path in (args.matrix, args.combinations) if path.exists()}
    content = markdown(summary)
    require(args.provenance.read_bytes() == raw, "diagnostic provenance changed during summarization")
    output = args.output_dir or args.provenance.parent
    output.mkdir(parents=True, exist_ok=True)
    save_json(output / "mechanism_summary.json", summary)
    temporary = output / ".mechanism_summary.md.tmp"
    temporary.write_text(content, encoding="utf-8")
    temporary.replace(output / "mechanism_summary.md")
    print(json.dumps({"passed": True, "evaluations": summary["evaluations_summarized"], "output": str(output)}, ensure_ascii=False))


if __name__ == "__main__":
    main()

"""Backfill the owned Markdown section from the verified M2 reports."""
from __future__ import annotations

import hashlib
import csv
import io
import json
import os
from pathlib import Path
import subprocess

from training_wss_min.tools import report_m2_optimization as R
from training_wss_min.tools.m2_optimization_common import EXP, ROOT, RUNS, COMBINATIONS, save_json, sha, stamp
from training_wss_min.tools.tracker_section import replace_or_append, split_section

TRACKER_HEADING = "## 16. M2优化：20项完整筛选与组合结论（2026-09-09–10）"
TRACKER = ROOT / "docs/02-推进与变更/00-V5设计与历史跟踪/WSS_V5_训练实验跟踪_历史卷_2026-09-06至09-20.md"
LINK = "../../../training_wss_min/experiments/m2_optimization_20260909/"


def read(path):
    return json.loads(Path(path).read_text())


def main():
    reports = R.report_all(write=False)
    if not all(reports[ck]["matrix_finalized"] for ck in ("best", "last")):
        raise RuntimeError("All registered evaluations must be finalized before document backfill")
    diagnostic = read(EXP / "diagnostics/provenance.json")
    workbook = read(EXP / "workbook_update.json")
    mechanism = read(EXP / "diagnostics/mechanism_summary.json")
    if not diagnostic["passed"] or not mechanism["passed"] or sha(workbook["book"]) != workbook["after_sha256"]:
        raise RuntimeError("Diagnostics, mechanism summary and workbook must pass first")
    diagnostic_job = str(diagnostic["job_id"])
    attempts = read(EXP / "diagnostic_attempts.json")["attempts"]
    expected_jobs = {str(attempt["job_id"]): (attempt["state"], attempt["exit_code"]) for attempt in attempts}
    expected_jobs[diagnostic_job] = ("COMPLETED", "0:0")
    accounting = subprocess.check_output([
        "/public/slurm/bin/sacct", "-j", ",".join(expected_jobs), "-P",
        "--format=JobID,State,ExitCode,Start,End,Elapsed",
    ], text=True, timeout=60)
    diagnostic_exits = {row["JobID"]: row for row in csv.DictReader(io.StringIO(accounting), delimiter="|")
                        if row["JobID"] in expected_jobs}
    if set(diagnostic_exits) != set(expected_jobs) or any(
        (diagnostic_exits[job]["State"], diagnostic_exits[job]["ExitCode"]) != expected
        for job, expected in expected_jobs.items()
    ):
        raise RuntimeError("Diagnostic retry/probe/failure Slurm accounting differs")
    verifications = [value for run in diagnostic["runs"].values() for checkpoint in run.values()
                     for case in checkpoint["cases"].values() for value in case["metric_verification"].values()]
    strict_checks = sum(value["strict_passed_checks"] for value in verifications)
    reviewed_checks = sum(len(value["boundary_reviews"]) for value in verifications)
    if strict_checks + reviewed_checks != diagnostic["original_per_case_numeric_checks"]:
        raise RuntimeError("Strict and reviewed diagnostic checks do not reconcile")
    arms = R.matrix_arms()
    selection = reports["selection"]
    decision = read(COMBINATIONS)
    base_count = sum(a["phase"] == "base" for a in arms)
    combo_count = len(decision["arms"])
    best, last = reports["best"], reports["last"]
    observed = selection["highest_observed_candidate"]
    preferred = selection["highest_final_priority_candidate"]
    top_b, top_l = best["arms"][observed], last["arms"][observed]
    conclusion = (f"优先复核候选为 {preferred}，仅属于预设门槛内的单seed探索结果。" if preferred else
                  "没有通过完整预设门槛的候选；继续保留历史M2主基准，不用原始最高值替代合格判断。")
    outcome = (f"**已完成 {base_count} 次基础训练、{combo_count} 次追加组合，共 {len(arms)} 次400轮训练。** "
               f"best/last各{len(arms)}次评估全部有效；{conclusion}")
    observed_text = (f"本轮按best Pa R²排序的观测最高为 **{observed}：best {top_b['physical_r2_cb']:.5f}、last {top_l['physical_r2_cb']:.5f}**；"
        f"相对历史M2分别{top_b['deltas_vs_historical_m2']['physical_r2_cb']:+.5f}/{top_l['deltas_vs_historical_m2']['physical_r2_cb']:+.5f}，"
        f"相对MO0分别{top_b['deltas_vs_mo0']['physical_r2_cb']:+.5f}/{top_l['deltas_vs_mo0']['physical_r2_cb']:+.5f}。")
    fixed = ("固定M2原25D输入、V5数据与既有几何sidecar、train138/test34、峰值单帧标量WSS；seed1234从零初始化，"
        "400轮、batch8、5000 support＋5000独立query、warmup10、cosine最低学习率1e-5、AMP与梯度裁剪1。"
        "best按各臂自身训练总损失选择，last固定第400轮；同一固定support与test34全壁面，沿用legacy_vertex口径。"
        "新几何、G/M6/B1及相关几何组合继续等待用户审查。")
    rules = ("相对历史M2及同期MO0均要求Pa R²_cb绝对增量best≥0.010、last≥0.005；两个checkpoint分别检查"
        "MAE增加≤2%、IoU下降≤0.010、病例R² P10下降≤0.020、负R²病例数为0。族内按best R²、last R²、"
        "较低best MAE、编号排序。组合只使用合格族冠军，按实际配置去重且不覆盖冲突；组合成为最终优先项还须"
        "best严格超过全部有效单项。门槛在结果后未调整。")
    control = (f"同期MO0 best/last为{best['arms']['MO0']['physical_r2_cb']:.5f}/{last['arms']['MO0']['physical_r2_cb']:.5f}，"
        "历史M2为0.62841/0.62429。配置、数据统计、初始化重建、采样和旧checkpoint评估复现均已核对；"
        "同随机状态的旧代码GPU短步重复也从首个feature-propagation模块出现数值分叉，AMP/FP32均可观察到。"
        "未发现新版独有的随机消费或首次分叉位置，但没有定量解释完整400轮的R²差距，不能直接以GPU噪声结案。")
    limitations = ("本轮为单seed1234、已暴露test34上的探索。best/last来自同一轨迹，不是独立重复；"
        "预设保护线与单项排名不构成独立确认或稳定提升证据。注意力、门控及残差激活不代表因果重要性。")
    table = ["| 实验 | 变化 | best Pa R² | last Pa R² | Δbest 历史M2 | Δlast 历史M2 | Δbest MO0 | Δlast MO0 | 筛选 |",
             "|---|---|---:|---:|---:|---:|---:|---:|---|"]
    for arm in arms:
        aid = arm["id"]
        b, l = best["arms"][aid], last["arms"][aid]
        status = "同期对照" if aid == "MO0" else "通过" if selection["candidates"][aid]["qualified"] else "未通过"
        table.append(f"| {aid} | {arm['hypothesis']} | {b['physical_r2_cb']:.5f} | {l['physical_r2_cb']:.5f} | "
            f"{b['deltas_vs_historical_m2']['physical_r2_cb']:+.5f} | {l['deltas_vs_historical_m2']['physical_r2_cb']:+.5f} | "
            f"{b['deltas_vs_mo0']['physical_r2_cb']:+.5f} | {l['deltas_vs_mo0']['physical_r2_cb']:+.5f} | {status} |")
    family_text = "；".join(f"{family}族：" + ("无合格项，保留M2设置" if row["fallback"] else row["winner"])
                           for family, row in selection["families"].items()) + "。"
    combo_text = (f"已冻结{combo_count}个组合，均从零训练400轮并完成双checkpoint评估。" if combo_count else
                  "合格族不足两个，按预登记规则不追加组合；四个预设组合的跳过原因已冻结。没有临时换次优项凑数，交互量不适用。")
    evidence = (f"四个基础Slurm worker 13974/13975/13976/13977均COMPLETED / 0:0；60个训练/评估阶段全部退出0。"
        f"每次评估均为同一34病例、1,231,295有效壁面点，400轮history与best/last checkpoint选模规则核验通过。"
        f"独立汇总重算{reports['audit']['aggregation_checks']:,}项，检查{reports['audit']['per_case_numeric_fields_checked']:,}个"
        f"逐病例数值字段；训练后{diagnostic['evaluations_verified']}次全壁面重推理（含历史M2两个checkpoint），"
        f"原逐病例指标共核对{diagnostic['original_per_case_numeric_checks']:,}项，其中{strict_checks:,}项通过原数值容差，"
        f"{reviewed_checks}项经逐点证据支持的离散边界审查。"
        f"GPU完整捕获来自Slurm {diagnostic['capture']['job_id']}，原始捕获记录与所有严格比较结果完整保留；"
        f"CPU对保存数组的核验、边界审查与绘图Slurm {diagnostic_job}为COMPLETED / 0:0。")
    boundary_text = (
        "首次诊断13979因MO-P4/best的ZHANG_YONG_ZHI在log_z空间pred_above_one_fraction严格核验失败而退出1:0；"
        "13981在MO-P2/last的ZHANG_JIN_CHUN同一字段的Pa空间核验失败，亦退出1:0。每处占比均相差一个壁面顶点，"
        "原失败记录和源码快照保留。随后一次完整捕获全部42个checkpoint/34例预测，完整收集差异；"
        "完整捕获中MO-P2该字段通过原容差，MO-S3/last的YU_XIANG_SHENG在log_z空间的pred_below_zero_fraction"
        "新增一个顶点计数差异，完整原始数组也保留并接受独立重复证据审查。"
        "在同一批保存数组上复核和绘图，没有靠重新推理挑选能过容差的结果。每项后验审查均绑定具体"
        "病例、checkpoint、数值空间、原始指标、模型及重复推理数组，仅审核有逐点证据的单顶点阈值差异，保留strict_passed=false；"
        "未放宽共有连续指标容差，未改原始评估、主指标、保护线或工作簿数值。"
        "原正式评估与13979首次失败的逐顶点预测未保存，不表示直接恢复这些历史数组或逐位复现；13981及完整捕获的原始数组已保存。"
        f"最终完整捕获实际使用{reviewed_checks}项边界审查，其余检查按原数值容差核验。")
    workbook_text = (f"工作簿已先备份，向“实验矩阵总览、教师汇报视图、汇总对比”追加本轮结果；主表沿用best，"
        f"last完整保存在配套报告且汇总表同时列主指标。{workbook['transcription_checks']:,}项转录核对通过，"
        f"历史{workbook['preserved_nonempty_cells']:,}个非空单元格及{workbook['preserved_formulas']:,}个公式未改变，未删插历史行。")
    interpretation = (
        "MO-S2加入query残差后，相对MO-S1的best/last Pa R²增加0.02350/0.01713；对历史M2的两条增益线仍不足，"
        "且best P10只比允许下限高约0.000165。不能把这一结果概括为注意力本身已超过M2。\n\n"
        "MO-L3（Pa-MSE λ=0.10）best/last为0.63134/0.62998，last与全部保护指标通过，唯一否决项是历史M2 best增益不足。"
        "本次Pa-MSE权重序列的主指标和高值RMSE改善可作描述，但P10并不单调，L2 best出现负R²病例。\n\n"
        "同容量S3门控相对S4加性残差的best/last Pa R²高0.00320/0.00752，但两者仍低于历史M2。"
        "增宽S5和增深S6未通过保护线；S5 last新增负R²病例。完整正负结果和病例贡献均保留。"
    )
    efficiency = ["| 实验 | 参数量 | 训练分钟 | 训练峰值allocated MiB | 训练峰值reserved MiB | AMP溢出跳步 |",
                  "|---|---:|---:|---:|---:|---:|"]
    for arm in arms:
        diag = read(RUNS / arm["run_name"] / "training_diagnostics.json")
        efficiency.append(f"| {arm['id']} | {diag['parameter_count']:,} | {diag['elapsed_seconds']/60:.2f} | "
            f"{diag['peak_cuda_allocated_mb']:.1f} | {diag['peak_cuda_reserved_mb']:.1f} | {diag['amp_overflow_steps']} |")
    lines = ["# M2 优化实验：完整结果与验收（2026-09-09–10）", "", outcome, "", observed_text, "",
        "## 固定协议与筛选", "", fixed, "", rules, "", family_text, "", combo_text, "",
        "[冻结组合清单](../../configs/m2_optimization_20260909/combinations.json) · [逐项筛选依据](selection.json)", "",
        "## 全部实验结果", "", *table, "",
        "[完整best主指标/保护指标](matrix_tables_best.md) · [完整last主指标/保护指标](matrix_tables_last.md) · "
        "[best JSON](matrix_summary_best.json) · [last JSON](matrix_summary_last.json) · [34例配对CSV](per_case_comparisons.csv)", "",
        "Pa R²_cb为病例等权共享均值R²；MAE为壁面顶点pool，P10来自逐病例R²。高值区域按每病例真值q90定义。"
        "幅值比应按距离1的远近解释，不能替代热点位置或误差指标。", "",
        "## 结果判读", "", interpretation, "",
        "[完整损失/参数分析](loss_parameter_analysis.md) · [完整结构分析](structure_analysis.md)", "",
        "## 同期对照偏差", "", control, "",
        "[专门审计](contemporary_control_diagnosis.md) · [旧/新GPU训练轨迹分析](training_trajectory/analysis_job_13974.md) · "
        "[初始化与采样重放](initialization_rng_diagnostic.json)", "",
        "![同期对照训练曲线及病例贡献](contemporary_control_diagnosis.png)", "",
        "## 图件与机制诊断", "",
        "[指标比较 PNG](m2_optimization_comparison.png) · [PDF](m2_optimization_comparison.pdf)\n\n"
        "[训练曲线 PNG](m2_training_curves.png) · [PDF](m2_training_curves.pdf)\n\n"
        "[best 34病例贡献](m2_case_contributions_best.png) · [last 34病例贡献](m2_case_contributions_last.png)\n\n"
        "[best三例原壁面对照](diagnostics/wall_figures/wall_overview_best.png) · [last三例原壁面对照](diagnostics/wall_figures/wall_overview_last.png)\n\n"
        "[逐例WSS/误差/热点双视角及PDF](diagnostics/README.md) · [注意力、门控与残差汇总](diagnostics/mechanism_summary.md)", "",
        "三例MA_TIAN_YI、SUN_XU_XIA、ZHANG_JIN_CHUN预先固定；原始壁面全部顶点，不重建/平滑几何，同病例best/last共用完整色标。"
        "逐点预测、support门控和query诊断数组保存在diagnostics/predictions，原点ID/坐标/真值在diagnostics/geometry。", "",
        "## 资源与工程核验", "", evidence, "", boundary_text, "",
        "[首次诊断失败记录](diagnostics_failed_13979/provenance.json) · "
        "[第二次诊断失败记录](diagnostics_failed_13981/provenance.json) · "
        "[完整捕获原始记录](diagnostics_collection/provenance.json) · "
        "[八次逐点边界探针](threshold_probe/job_13980/probe.json) · "
        "[离散边界审查与逐项证据](diagnostics/provenance.json) · "
        "[全部诊断作业记录](diagnostic_attempts.json)", "", *efficiency, "",
        "峰值为真实训练PyTorch allocator统计；reserved含其缓存，不等同于nvidia-smi进程全部显存。每次启动按"
        "max(7000 MiB, 1.25×实测峰值＋2048 MiB)检查空闲显存；每GPU最多两个本轮任务，不调整同学进程。"
        "AMP梯度溢出由原GradScaler处理并逐步记录，非有限物理损失直接报错；没有裁剪标签或改损失掩盖问题。", "",
        "CPU兼容/损失梯度/配对初始化检查通过；Slurm13972真实20臂AMP与显存预检通过；13973五臂各2轮CLI冒烟及"
        "5622项历史M2完整test34数值复现通过。13970因预检工具错误要求feature_stats_path而失败，修复后重验；"
        "13971取消后以13973匹配冻结源码重跑。13975.1是nvidia-smi设备探测失败（6:0），后续探测通过，不是训练阶段失败。"
        "13978在排队时取消，短步诊断移入13974.1空槽，后者0:0完成；协调器恢复记录与隔离lease均保留。", "",
        "[队列及60阶段记录](queue_status.json) · [源码快照](source_snapshot_base/) · [518文件输入清单](data_provenance.json) · "
        "[独立指标核验](independent_metric_audit.json) · [全壁面诊断证据](diagnostics/provenance.json) · "
        "[最终验收](final_acceptance.json)", "",
        "## 文档与工作簿", "", workbook_text, "",
        f"[工作簿备份]({Path(workbook['backup']).name}) · [转录与历史保留证据](workbook_update.json)", "", limitations, ""]
    tracker_section = [TRACKER_HEADING, "", outcome, "",
        observed_text, "", fixed, "", "### 16.1 全部best/last结果", "", *table, "",
        "### 16.2 筛选结论", "", rules, "", family_text, "", combo_text, "", interpretation, "",
        "### 16.3 同期对照与验收", "", control, "", evidence, "", boundary_text, "", workbook_text, "",
        f"完整[实验报告]({LINK}README.md)、[损失/参数分析]({LINK}loss_parameter_analysis.md)、"
        f"[结构分析]({LINK}structure_analysis.md)、[同期对照审计]({LINK}contemporary_control_diagnosis.md)。", "",
        f"[指标比较]({LINK}m2_optimization_comparison.png) · [训练曲线]({LINK}m2_training_curves.png) · "
        f"[best壁面对照]({LINK}diagnostics/wall_figures/wall_overview_best.png) · "
        f"[last壁面对照]({LINK}diagnostics/wall_figures/wall_overview_last.png) · "
        f"[注意力/门控诊断]({LINK}diagnostics/mechanism_summary.md) · [最终验收]({LINK}final_acceptance.json)", "",
        limitations, ""]
    original = TRACKER.read_text()
    prefix, _, suffix = split_section(original, TRACKER_HEADING)
    updated = replace_or_append(original, TRACKER_HEADING, "\n".join(tracker_section))
    backup = EXP / "tracker_before_final_m2.md"
    if not backup.exists():
        backup.write_text(original)
    if TRACKER.read_text() != original:
        raise RuntimeError("Tracker edited concurrently; reread and merge")
    temporary = TRACKER.with_name(TRACKER.name + ".m2.tmp")
    temporary.write_text(updated)
    os.replace(temporary, TRACKER)
    (EXP / "README.md").write_text("\n".join(lines))
    saved_prefix, _, saved_suffix = split_section(TRACKER.read_text(), TRACKER_HEADING)
    if saved_prefix != prefix or saved_suffix != suffix:
        raise RuntimeError("Historical tracker sections changed")
    save_json(EXP / "document_backfill.json", {"completed_at": stamp(), "script_sha256": sha(__file__),
        "tracker_before_sha256": hashlib.sha256(original.encode()).hexdigest(), "tracker_after_sha256": sha(TRACKER),
        "historical_tracker_sections_byte_preserved": True, "registered_result_rows": len(arms),
        "readme_sha256": sha(EXP / "README.md"), "highest_observed": observed, "highest_final_priority": preferred,
        "diagnostic_slurm_exits": diagnostic_exits, "diagnostic_strict_checks": strict_checks,
        "diagnostic_reviewed_checks": reviewed_checks})
    print(json.dumps({"written": [str(TRACKER), str(EXP / 'README.md')], "rows": len(arms)}, ensure_ascii=False))


if __name__ == "__main__":
    main()

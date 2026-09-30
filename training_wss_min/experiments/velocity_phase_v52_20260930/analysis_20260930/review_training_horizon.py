"""Audit small existing training logs only; no checkpoint loading or inference.

Run with the GNN Python. Outputs are descriptive, not validation/early-stop evidence.
"""
from pathlib import Path
import csv
import hashlib
import json
import math
import statistics

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
WINDOWS = [(81, 100), (101, 120), (121, 140), (141, 150)]
ARMS = ["U0", "D1", "D2", "A0", "A1", "G00", "G01", "G10", "G11", "P1", "W1"]


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def lr_after_epoch(cfg, epoch, total_epochs=None):
    # Mirrors the frozen runners' LambdaLR, called after each optimizer step.
    tc = cfg["train"]
    steps = math.ceil(206 / tc["batch_cases"])
    warm = max(tc["warmup_epochs"] * steps, 1)
    total = (total_epochs or tc["epochs"]) * steps
    step = epoch * steps
    if step < warm:
        factor = (step + 1) / warm
    else:
        progress = min(max((step - warm) / max(total - warm, 1), 0), 1)
        factor = tc["min_lr_ratio"] + (1 - tc["min_lr_ratio"]) * (1 + math.cos(math.pi * progress)) / 2
    return tc["lr"] * factor


records, sources, summary = [], {}, {}
for arm in ARMS:
    batch = "joint_cycle_round2_v52_20260929" if arm in ["U0", "P1", "W1"] else "velocity_phase_v52_20260930"
    run = ROOT / "training_wss_min/runs" / batch / (arm + "_f0_s1234")
    cfg = json.loads((run / "config.json").read_text())
    history = list(map(json.loads, (run / "history.jsonl").read_text().splitlines()))
    task, = cfg["tasks"]
    assert [h["epoch"] for h in history] == list(range(1, 151))
    assert history[-1]["step"] == 7800
    assert cfg["seed"] == 1234 and cfg["data"]["fold"] == 0
    assert cfg["train"]["selection"] == "last"
    assert cfg["data"]["split_sha256"] == "57862a4f09de4c4bec9adf0eed66cf2ce0d9e7da0528c0ee1d38e0fb0f202269"
    frozen = ROOT / "training_wss_min/experiments" / batch / "frozen/f0_s1234_v1/training_wss_min" / ("joint_cycle_round2.py" if arm in ["U0", "P1", "W1"] else "velocity_phase.py")
    code = frozen.read_text()
    assert 'total_steps = tc["epochs"] * steps_per_epoch' in code
    assert 'progress = min(max((step - warm_steps)' in code
    sources[arm] = {"run": str(run), "history_sha256": sha(run / "history.jsonl"),
                    "config_sha256": sha(run / "config.json"), "frozen_runner": str(frozen),
                    "frozen_runner_sha256": sha(frozen),
                    "checkpoint_files": sorted(p.name for p in run.glob("*.pt")),
                    "history_has_validation_fields": any(any(k.startswith(("val", "test")) for k in h) for h in history),
                    "base_loss_field": "objective.z_mse" if "objective" in history[0] else "losses." + task}
    def base(h):
        return h.get("objective", {}).get("z_mse", h["losses"][task])
    fields = {"shared_z_mse": base}
    fields.update({"phase_z_mse." + phase: (lambda h, p=phase: h["phase_z_mse"][p])
                   for phase in history[0].get("phase_z_mse", {})})
    summary[arm] = {"task": task, "windows": {},
                    "lr_reconstructed_after_epoch": {str(e): lr_after_epoch(cfg, e) for e in [80, 100, 120, 140, 150]},
                    "lr_if_fresh300_after_epoch150": lr_after_epoch(cfg, 150, 300)}
    for field, getter in fields.items():
        means = []
        for lo, hi in WINDOWS:
            vals = [getter(h) for h in history if lo <= h["epoch"] <= hi]
            assert len(vals) == hi - lo + 1 and all(math.isfinite(v) for v in vals)
            mean = statistics.mean(vals)
            r = dict(arm=arm, task=task, field=field, epoch_start=lo, epoch_end=hi,
                     n=len(vals), mean=mean, minimum=min(vals), maximum=max(vals),
                     stdev=statistics.stdev(vals),
                     reduction_from_previous_window_pct=None if not means else 100 * (means[-1] - mean) / means[-1])
            records.append(r)
            means.append(mean)
        summary[arm]["windows"][field] = {"means": means,
            "reduction_81_100_to_141_150_pct": 100 * (means[0] - means[-1]) / means[0],
            "reduction_121_140_to_141_150_pct": 100 * (means[2] - means[3]) / means[2]}

with (HERE / "training_horizon_review.csv").open("w") as f:
    writer = csv.DictWriter(f, fieldnames=list(records[0]))
    writer.writeheader()
    writer.writerows(records)
output = {"scope": "Existing small logs/configs/frozen source only; no new training, checkpoint evaluation or raw-data access",
          "comparison_contract": "V5.2 fold0, train206/development holdout55, seed1234, 150epochs/7800steps, last, 80 phases",
          "loss_note": "Shared train normalized z-MSE only; not total D1/D2 objective. Pressure/WSS loss values must not be compared numerically with velocity.",
          "window_note": "Descriptive epoch-window mean/min/max/stdev. Windows121–140 vs141–150 differ in length; not paired significance tests. Query/support resampling and dropout imply train loss is stochastic.",
          "lr_note": "History lacks LR fields; LR reconstructed from exact frozen LambdaLR and actual config. Optimizer audit for eight velocity arms independently confirms7800 scheduler steps.",
          "validation_limit": "No per-epoch holdout metrics. Each existing run keeps only overwritten ckpt_last.pt. Cannot infer validation trajectory, overfit onset, optimal epoch, or benefit of extra epochs.",
          "sources": sources, "summary": summary, "records": records}
(HERE / "training_horizon_review.json").write_text(json.dumps(output, ensure_ascii=False, indent=2) + "\n")


def table(headers, rows):
    return "\n".join(["|" + "|".join(headers) + "|", "|" + "|".join(["---"] * len(headers)) + "|"] +
                     ["|" + "|".join(map(str, r)) + "|" for r in rows])


lines = ["# 训练轮次复核：150 轮是否值得延长（2026-09-30）", "",
         "结论：增加训练轮次可能有帮助，但现有证据不能保证开发留出精度提高。速度候选在当前退火日程末段已近训练平台，直接用末值学习率继续训练，预期更像小幅微调；压力 P1 的训练下降空间更明显。无需把 8 个速度实验全部翻倍，优先用 U0/G11 成对验证。此次仅分析现有小日志和配置，未启动训练或新推理。", "",
         "## 1. 合同与实测训练趋势", "",
         "所有表项均为 V5.2 / patient-group fold0 / train206、开发留出55 / seed1234 / 150轮、7800步 / last / 80相位。速度用共同 z-MSE，D1/D2 不用含辅助项/权重的 total loss 跨臂比较。P1、W1 是独立任务的归一化 MSE，其数值不能与速度作精度排名。", "",
         table(["臂", "任务", "81–100", "101–120", "121–140", "141–150", "末两窗口下降"],
               [[a, summary[a]["task"]] + [f"{v:.6f}" for v in summary[a]["windows"]["shared_z_mse"]["means"]] +
                [f"{summary[a]['windows']['shared_z_mse']['reduction_121_140_to_141_150_pct']:.3f}%"] for a in ARMS]), "",
         "末两窗口分别是 20 轮与 10 轮均值；完整 min/max/标准差在 CSV。这里的下降比例是描述性训练趋势，不能当作统计显著性或留出提升。随机查询/支撑点、dropout 与训练轨迹会产生逐轮波动。", "",
         table(["臂", "训练窗口", "81–100", "101–120", "121–140", "141–150", "末两窗口下降"],
               [[a, p] + [f"{v:.6f}" for v in summary[a]["windows"]["phase_z_mse." + p]["means"]] +
                [f"{summary[a]['windows']['phase_z_mse.' + p]['reduction_121_140_to_141_150_pct']:.3f}%"]
                for a in ["D1", "G10", "G11"] for p in ["trough", "decel"]]), "",
         "U0/P1/W1 历史没有 phase_z_mse，不能补造其分段训练曲线。G11 末10轮谷底 z-MSE 0.436697、减速 0.187240；谷底依然更难，但末段下降仅 0.286%/0.426%。共同 z-MSE 末10轮标准差 U0=0.002420、G11=0.002172，均大于相邻两窗口均值差；这描述波动尺度，不证明收敛或无收益。", "",
         "## 2. 学习率和证据边界", "",
         "冻结代码采用 5 轮 warmup + cosine，峰值学习率 1e-3、min_lr_ratio=0.01。history 未直接记录 LR，下表按冻结源码和真实配置重建；8 个速度新臂既有 optimizer_audit 已确认 scheduler 到 7800 步。", "",
         table(["已完成轮次", "当前150轮日程的下一步LR"], [[e, f"{lr:.9g}"] for e, lr in summary["U0"]["lr_reconstructed_after_epoch"].items()]), "",
         f"将总轮数设为 300 会从第 6 轮起改变退火速度：新日程在第150轮末 LR={summary['U0']['lr_if_fresh300_after_epoch150']:.9g}，现有日程为1e-5。两者在第150轮的模型并不是同一个优化过程。当前平台可能部分源于低学习率，不能据此证明更慢退火无效。", "",
         "所有既有 run 都只在训练结束后评估开发留出；虽然 checkpoint_every=5，保存函数每次覆盖同一个 ckpt_last.pt，没有可供逐轮留出比较的历史 checkpoint。不能识别早期过拟合、最佳轮次，也不能把训练下降解释成谷底/减速 R² 必然提高。现有 55 单元已参与多轮设计，它仍是开发集。", "",
         "## 3. 最小下一步设计（建议，尚未提交）", "",
         "优先只做 U0 与 G11 两臂，各 seed1234：从头训练 300 轮、保持原 warmup/峰值 LR/min_lr_ratio，并将 cosine 总长度改为 300；数据、输入、采样、损失、模型均保持对应父臂。它回答更长训练预算与较慢退火这一整套策略是否更好，不能声称只增加轮数带来因果收益。D1/G10 不在首批重复，D2/A1 暂不扩展。", "",
         table(["建议ID", "父臂", "起点/日程", "固定主终点", "回答的问题"], [
             ["H-U0-300", "U0", "配对初始化；300轮cosine", "last300 / 15600步", "基础结构能否同样受益"],
             ["H-G11-300", "G11", "配对初始化；300轮cosine", "last300 / 15600步", "当前候选在更长训练策略下能否继续提高"],
             ["H-U0-tail（可选）", "U0", "原last150；151–300固定1e-5", "last300 / 累计15600步", "相同前150轮轨迹后追加低LR更新"],
             ["H-G11-tail（可选）", "G11", "原last150；151–300固定1e-5", "last300 / 累计15600步", "追加低LR更新是否足够，区别于慢退火"]]), "",
         "若目标是严格回答已有模型再多训是否更好，可将可选 tail 两臂作为首批；它们保持前150轮不变，但受 LR 已降到底影响，不能用其失败否定 fresh300。tail 必须从只读父 checkpoint 恢复 model、AdamW、scaler、RNG 和 epoch/step，并延续 epoch采样种子；不能仅加载模型权重后重置 optimizer，也不能偷偷恢复为高 LR。若改变 LR，另命名为 restart 策略。", "",
         "每臂使用独立配置、入口与输出。现有 runner 校验硬锁150轮并禁止原地resume，需要独立的延长方案实现；不改冻结代码、不覆盖原last150。fresh300 建议保留150/225/300三个固定节点；225仅作训练诊断，主结论只比较预注册last300，不按55单元最优值回选checkpoint。可事先固定150/300两端留出评估，且必须报告两个节点而非只挑更好者。", "",
         "报告固定峰17–26、谷5–9及43–57、减速27–42（0-based）与整周期：R²、分量MAE、向量RMSE、方向余弦，另列早/晚谷、P10、负R²数、改善单元数与训练成本。先比较每臂300与自己的150父方案，再比较同预算G11与U0；不能用G11-300单独对U0-150证明结构收益。沿用开发筛选门作参照，不把单seed结果升级为独立确认。", "",
         "压力/壁面任务后续：P1末段训练损失仍下降3.261%，值得列为下一批长训练候选；W1下降0.910%可随后验证。它们要保持各自目标、物理单位和评价规则，不能拿速度延长结果推断压力/WSS也会提高。", "",
         "## 4. 可复算证据", "",
         "运行 `review_training_horizon.py` 仅读取现有 history/config/冻结Python文本，输出本报告、JSON 和 CSV；JSON 保存每个源路径及 SHA256。未读大 checkpoint、原始场或预测数组。", "",
         "- [完整窗口统计 CSV](training_horizon_review.csv)",
         "- [来源与精确数值 JSON](training_horizon_review.json)",
         "- [可复算脚本](review_training_horizon.py)",
         "- [既有优化器步数验收](optimizer_audit.json)",
         "- [既有150轮正式精度](results_matrix.md)"]
(HERE / "training_horizon_review.md").write_text("\n".join(lines) + "\n")
print(json.dumps({"artifacts": [str(HERE / ("training_horizon_review." + e)) for e in ["md", "json", "csv"]],
                  "last_window_relative_reductions_pct": {a: summary[a]["windows"]["shared_z_mse"]["reduction_121_140_to_141_150_pct"] for a in ARMS}}, ensure_ascii=False, indent=2))

"""Export existing metrics and completion evidence; no inference or training."""
import csv
import hashlib
import json
from datetime import datetime
from pathlib import Path
import statistics
import subprocess

ROOT = Path(__file__).resolve().parents[4]
OUT = Path(__file__).resolve().parent
EXP = OUT.parent
OLD = ROOT / "training_wss_min/runs/joint_cycle_v52_20260929"
NEW = ROOT / "training_wss_min/runs/joint_cycle_round2_v52_20260929"
OLD_ARMS = ["Iu", "Ip", "Iw", "J0", "J1c", "J1"]
NEW_ARMS = ["UT0", "UT1", "WT0", "WT1", "U0", "U1", "W0", "W1", "P1", "J2"]
PHASES = ["peak", "trough", "cycle", "accel", "decel", "plateau"]


def main():
    records, metrics, completed, rows, frames = {}, {}, {}, [], []
    for arm in OLD_ARMS + NEW_ARMS:
        run = (OLD if arm in OLD_ARMS else NEW) / f"{arm}_f0_s1234"
        d = json.loads((run / "metrics.json").read_text())
        per = [json.loads(x) for x in (run / "per_case.jsonl").read_text().splitlines()]
        assert d["status"] == "complete" and d["n_units"] == 55 and d["selection"] == "last"
        records[arm] = {r["unit_id"]: r for r in per}
        metrics[arm] = d
        assert len(records[arm]) == len(per) == 55
        assert set(records[arm]) == set(records["Iu"])
        for task, task_metrics in d["tasks"].items():
            for phase in PHASES:
                m = task_metrics[phase]
                rows.append({"arm": arm, "task": task, "phase": phase,
                             **{k: m[k] for k in ["r2", "mae", "rmse", "relative_l2"]},
                             "source": str(run / "metrics.json")})
            for frame, m in task_metrics.get("per_frame", {}).items():
                frames.append({"arm": arm, "task": task, "frame": int(frame),
                               **{k: m[k] for k in ["r2", "mae", "rmse", "relative_l2"]}})
        if arm in NEW_ARMS:
            h = [json.loads(x) for x in (run / "history.jsonl").read_text().splitlines()]
            complete = json.loads((run / "training_complete.json").read_text())
            execution = json.loads((run / "execution.json").read_text())
            assert [r["epoch"] for r in h] == list(range(1, 151))
            assert h[-1]["step"] == complete["steps"] == 7800 and not complete["smoke"]
            assert execution["returncode"] == 0 and (run / "ckpt_last.pt").is_file()
            completed[arm] = {"status": "completed", "epochs": 150, "steps": 7800,
                              "evaluated_units": 55, "phase_count": d["phase_count"],
                              "metric_source": str(run / "metrics.json"),
                              "execution_source": str(run / "execution.json")}
    for name, data in [("results_matrix.csv", rows), ("per_frame_metrics.csv", frames)]:
        with (OUT / name).open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(data[0]))
            w.writeheader(); w.writerows(data)
    frozen = EXP / "frozen/f0_s1234_v1"
    fingerprints = json.loads((frozen / "fingerprints.json").read_text())
    assert all(hashlib.sha256(Path(p).read_bytes()).hexdigest() == h for p,h in fingerprints.items())
    accounting = subprocess.check_output(["/public/slurm/bin/sacct", "-j", "16165,16166",
        "--format=JobID,State,ExitCode,Elapsed,Start,End", "-P"], text=True)
    proof = {"checked_at": datetime.now().astimezone().isoformat(), "status": "completed",
             "arms": completed, "seed": 1234, "fold": 0, "frozen_fingerprints_verified": len(fingerprints),
             "independent_checkpoint_audit": "All ten ckpt_last: epoch149 (0-based), step7800, config SHA matches provenance",
             "slurm_accounting": accounting, "new_inference_or_training": False}
    (EXP / "completion_verified.json").write_text(json.dumps(proof, ensure_ascii=False, indent=2)+"\n")
    history = [json.loads(x) for x in (NEW / "J2_f0_s1234/history.jsonl").read_text().splitlines()]
    grad = {"task_pair_conflict_fraction_mean": statistics.mean(r["gradient_diagnostics"]["conflict_fraction"] for r in history),
            "projected_mean_gradient_relative_change": statistics.mean(r["gradient_diagnostics"]["relative_shared_update_change"] for r in history),
            "gradient_clip_fraction": statistics.mean(r["gradient_clip_fraction"] for r in history),
            "train_time_change_vs_J1": metrics["J2"]["train_seconds"] / metrics["J1"]["train_seconds"] - 1}
    # Cost accounting only: no new ensemble/service inference, no speed claim.
    arms = ["U0", "P1", "W1"]
    sums = [sum(records[a][u]["cache_load_encode_decode_seconds"] for a in arms) for u in records["Iu"]]
    cost = {"arms": arms, "method": "sum stored per-unit single-model times, then take median; not a measured combined deployment",
            "median_sum_s": statistics.median(sums),
            "parameter_sum": sum(metrics[a]["parameter_count"] for a in arms)}
    (OUT / "diagnostics.json").write_text(json.dumps({"pcgrad": grad, "candidate_cost_accounting": cost},indent=2)+"\n")
    names = {"velocity": "速度向量", "pressure": "相对压力", "wss": "标量WSS"}
    groups = {"velocity": ["Iu", "UT0", "UT1", "U0", "U1", "J1", "J2"],
              "pressure": ["Ip", "J1c", "J1", "P1", "J2"],
              "wss": ["Iw", "WT0", "WT1", "W0", "W1", "J1", "J2"]}
    lines = ["# 第二轮完整结果矩阵", "", "R²为数据单元等权均值；MAE速度为分量m/s，压力与WSS为Pa。峰值窗17–26、谷底窗5–9及43–57、周期0–79，均0-based。", "",
             "| 任务 | 模型 | 峰值 R² / MAE | 谷底 R² / MAE | 周期 R² / MAE |",
             "|---|---|---:|---:|---:|"]
    for task, arms in groups.items():
        for arm in arms:
            vals = [f'{metrics[arm]["tasks"][task][p]["r2"]:.4f} / {metrics[arm]["tasks"][task][p]["mae"]:.5f}' for p in PHASES[:3]]
            lines.append(f'| {names[task]} | {arm} | '+" | ".join(vals)+" |")
    lines += ["", "全部六阶段144行指标见[CSV](results_matrix.csv)；新实验960行逐帧结果见[逐帧CSV](per_frame_metrics.csv)。旧基线的缺失单帧指标没有重算或填造。", ""]
    (OUT / "results_matrix.md").write_text("\n".join(lines))
    print(f"Verified 10 completed runs; wrote {len(rows)} phase rows and {len(frames)} frame rows.")
    print(json.dumps({"pcgrad":grad,"candidate_cost_accounting":cost},indent=2))


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Replace the preregistration status blocks with actual submitted Job IDs."""

from __future__ import annotations

import argparse
import json
import re
import subprocess
from pathlib import Path


REPO = Path(__file__).resolve().parents[2]
SUBMISSION = REPO / "training_wss_min/preflight/q2v_ilo_arch_matrix_submission.json"
CODE_LOG = REPO / "docs/02-推进与变更/WSS最小化_代码修改与实验推进记录.md"
TRAIN_LOG = REPO / "docs/02-推进与变更/WSS最小化_训练实验跟踪.md"
MATRIX_LOG = REPO / "docs/02-推进与变更/WSS最小化_PointNet_baseline实验矩阵与进度跟踪.md"
START = "<!-- Q2V_ILO_20260718_START -->"
END = "<!-- Q2V_ILO_20260718_END -->"


def replace_block(path: Path, block: str) -> None:
    text = path.read_text(encoding="utf-8")
    pattern = re.compile(re.escape(START) + r".*?" + re.escape(END), re.S)
    if len(pattern.findall(text)) != 1:
        raise RuntimeError(f"expected exactly one Q2V status block in {path}")
    path.write_text(pattern.sub(f"{START}\n{block.rstrip()}\n{END}", text), encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--submission", type=Path, default=SUBMISSION)
    args = ap.parse_args()
    payload = json.loads(args.submission.read_text(encoding="utf-8"))
    if payload.get("status") != "submitted" or len(payload.get("jobs", [])) != 4:
        raise RuntimeError("submission manifest is not complete")
    jobs = {row["role"]: row for row in payload["jobs"]}
    data = jobs["data_training_array"]
    arch = jobs["architecture_dev_array_val_only"]
    export = jobs["q2v_10477_postview_export_only_27"]
    zero = jobs["q2v_10477_zero_shot_extended_test36"]
    ids = [data["job_id"], arch["job_id"], export["job_id"], zero["job_id"]]
    queue = subprocess.run(
        ["/public/slurm/bin/squeue", "-h", "-j", ",".join(ids), "-o", "%i|%T|%R"],
        text=True, capture_output=True, check=True,
    ).stdout.strip().replace("\n", "; ") or "already left squeue; use sacct as truth"

    mappings = ", ".join(
        f"`{data['job_id']}_{row['task_id']}`={row['experiment_id']}" for row in data["tasks"]
    )
    arch_mappings = ", ".join(
        f"`{arch['job_id']}_{row['task_id']}`={row['experiment_id']}" for row in arch["tasks"]
    )
    code_block = f"""## 2026-07-18｜Q2V 数据扩容6组 + Point++ 架构6组 🚀正式门禁通过并提交

**本次主要修改**：ILO-before41 以独立训练晋级 manifest 接入，不改写原审核文件 `training_promotion_authorized=false` 的历史事实；canonical ID 仅接受 `ILO/<patient>-<0|1>/before` 并拒绝 `after`。新增冻结 feature stats、评估只读 split override、五层 grouped 精确 quota、12配置 GPU preflight、两类 `%4` array runner、只读评估和 partial-submission-safe 提交/记录链；AreaRandom 六组保持冻结。

**对应代码/文档**：`training_wss_min/{{config.py,dataset.py,train.py,evaluate.py}}`、`training_wss_min/tools/{{prepare_q2v_ilo_matrix.py,preflight_v4_jobs.py,finalize_q2v_submission_records.py,update_q2v_matrix_xlsx_uno.py}}`、`training_wss_min/cluster/` 的Q2V脚本、`training_wss_min/tests/test_q2v_ilo_matrix.py`、`training/splits/*q2v*`、`data_wss_min/fold_stats/q2v_ilo_20260718/`、12份正式config、`training_wss_min/preflight/q2v_ilo_arch_matrix_{{prepared,gpu_preflight,submission}}.json`、本页/训练跟踪/PointNet矩阵/xlsx。

**推进到实验步骤**：CPU Job `10484` `COMPLETED (0:0)`，174/174 bundle/frame/hash、五层配额、零泄漏和统计来源通过，新增测试11/11通过；GPU preflight Job `10485` 为12/12 passed且提交时配置哈希未变化。数据数组 Job `{data['job_id']}`（`0-5%4`）、架构val-only数组 Job `{arch['job_id']}`（`0-5%4`）、Q2V export-only Job `{export['job_id']}`、Q2V D2 test36零样本 Job `{zero['job_id']}` 已提交。当前队列快照：`{queue}`。

**当前状态判断**：训练/评估已正常入队但尚无结果，不预填任何指标、不将单seed写成确认性结论。架构数组只评val21且runner无 `--allow-test`；原test27确认任务继续未提交。submission manifest 固化了config/split/stats哈希、控制组、唯一变化和预期run路径。"""

    train_block = f"""## Q2V 数据扩容与 Point++ 架构预注册（2026-07-18｜QUEUED）

共同冻结 Q2V-10477：Point++、vertex random5000 SEP、FPS centers `500/125/32`、radius `0.05/0.10/0.20`、MSE、400 epoch、train-loss选模、seed1234、legacy vertex。CPU `10484` 和GPU preflight `10485` 已通过；结果列保持空白。

| 家族 | Slurm | 任务映射 / 口径 | 状态 |
| --- | --- | --- | --- |
| 数据矩阵6组 | `{data['job_id']}_[0-5]%4` | {mappings} | QUEUED/RUNNING；各组自动best/last test+比较+PostView验证 |
| 架构dev6组 | `{arch['job_id']}_[0-5]%4` | {arch_mappings} | QUEUED/RUNNING；**只评val21，不访问test27** |
| Q2V export-only | `{export['job_id']}` | 补齐 Q2V-10477 PostView test27；checkpoint/metrics前后哈希不变 | QUEUED/RUNNING |
| Q2V zero-shot | `{zero['job_id']}` | 原Q2V best对D2 test36只读split override | QUEUED/RUNNING |

当前队列快照：`{queue}`。五层 D3 固定计数为 AG `61/15`、AAA rupture `21/6`、AAA unrupture `24/6`、ILO-0 `22/6`、ILO-1 `10/3`。架构胜者test27确认任务与AreaRandom六组均未提交。"""

    matrix_block = f"""## 0D. Q2V 数据扩容与 Point++ 架构矩阵（2026-07-18｜QUEUED，结果留空）

正式门禁：CPU `10484` passed；GPU preflight `10485` 12/12 passed；提交时12份配置哈希与preflight完全一致且run目录均不存在。

| ID | train/val/test | target / feature stats | Job | 主对照与唯一变化 |
| --- | ---: | --- | --- | --- |
| `q2v_ilo_d1_fixed_frozen` | `147/0/27` | 冻结Q2V / 冻结Q2V | `{data['job_id']}_0` | 原test27不变，新增ILO41训练 |
| `q2v_ilo_d1_fixed_refit` | `147/0/27` | train147重算 / train147重算 | `{data['job_id']}_1` | 相对D1-frozen只改统计重算 |
| `q2v_ilo_d2_extended_frozen` | `138/0/36` | 冻结Q2V / 冻结Q2V | `{data['job_id']}_2` | ILO32训练+ILO9测试 |
| `q2v_ilo_d3_pool2025_control` | `106/0/36` | control106 / control106 | `{data['job_id']}_3` | 五层test36；ILO32登记unused |
| `q2v_ilo_d3_pool2025_frozen` | `138/0/36` | 冻结control106 / 冻结control106 | `{data['job_id']}_4` | 相对control只加入ILO32训练 |
| `q2v_ilo_d3_pool2025_refit` | `138/0/36` | train138重算 / train138重算 | `{data['job_id']}_5` | 相对frozen只改统计重算 |
| `q2v_arch_dev_n16_w32` | `85/21/27` | dev85 train-only | `{arch['job_id']}_0` | 架构锚格；val21 only |
| `q2v_arch_dev_n16_w64` | `85/21/27` | dev85 train-only | `{arch['job_id']}_1` | 只改width32→64；val21 only |
| `q2v_arch_dev_n32_w32` | `85/21/27` | dev85 train-only | `{arch['job_id']}_2` | 只改nsample16→32；val21 only |
| `q2v_arch_dev_n32_w64` | `85/21/27` | dev85 train-only | `{arch['job_id']}_3` | nsample32×width64因子格；val21 only |
| `q2v_arch_dev_n64_w32` | `85/21/27` | dev85 train-only | `{arch['job_id']}_4` | 只改nsample16→64；val21 only |
| `q2v_arch_dev_n64_w64` | `85/21/27` | dev85 train-only | `{arch['job_id']}_5` | nsample64×width64因子格；val21 only |

另有只读 Q2V PostView `27/27` Job `{export['job_id']}` 与原Q2V对D2 test36零样本 Job `{zero['job_id']}`。当前队列：`{queue}`。本节不得回填结果前提前排名；架构胜者test27确认与AreaRandom六组继续冻结。"""

    replace_block(CODE_LOG, code_block)
    replace_block(TRAIN_LOG, train_block)
    replace_block(MATRIX_LOG, matrix_block)
    print(json.dumps({"status": "updated", "jobs": ids, "queue": queue}, ensure_ascii=False))


if __name__ == "__main__":
    main()

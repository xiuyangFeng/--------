"""One-off CPU completion stage for the frozen 26-arm training/evaluation job.

This report-formatting stage is separate from the frozen trainer/evaluator.
It runs only after Slurm releases the parent job and preserves failures.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
import re
import subprocess
import sys
import traceback

ROOT=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT))
from training_wss_min.tools.volume_attention_common import EXP,NAME,save_json,sha256,stamp
from training_wss_min.tools.tracker_section import replace_or_append

HEADING="## 17. 压力/速度全局注意力：26臂单seed适配（2026-09-10）"
TRACKER=ROOT/"docs/02-推进与变更/WSS_PINN/WSS_V5_训练实验跟踪.md"
CHANGELOG=ROOT/"docs/02-推进与变更/WSS最小化_代码修改与实验推进记录.md"


def f(value,digits=4):return "—" if value is None else f"{value:.{digits}f}"


def update_readme_status(document):
    """Replace the owned introduction status, including a live queue update."""
    lines=document.splitlines(keepends=True)
    boundary=next((i for i,line in enumerate(lines) if line.startswith("## ")),len(lines))
    matches=[i for i,line in enumerate(lines[:boundary]) if "状态：" in line]
    if len(matches)!=1:
        raise ValueError(f"Expected one introduction status in experiment README, got {len(matches)}")
    i=matches[0]
    lines[i]=lines[i].split("状态：",1)[0]+"状态：26臂400轮及全部best/last已完成，独立数值验收与工作簿回填通过。完整结果见[results.md](results.md)，完成证据见[completion_status.json](completion_status.json)。\n"
    status="""## 当前执行状态

- 正式队列 **14040** 已完成全部26臂400轮、best/last全量评估和34病例分支长波诊断；训练源码与配置保持冻结，详见[queue_status.json](execution/queue_status.json)。
- 自动汇总 **14041** 已通过60组目标/checkpoint预测数值验收、26条训练history验收和工作簿回读核对，详见[report.json](report.json)及[xlsx_acceptance.json](xlsx_acceptance.json)。
- 真实batch8预检14034、历史复核14036及26臂两轮冒烟14038均通过；单测104项及40子测试、历史2948项检查通过。
- 最初12小时分配14039的四份部分轨迹保留[归档](interrupted_initial_allocation_14039/reason.json)，不计正式成绩；全部正式训练仅seed1234。

提交与完成依据见[submission.json](submission.json)、[completion_status.json](completion_status.json)。资源表反映单GPU四槽并发执行条件；[联合query成本](joint_query_costs.csv)单独记录联合执行和对应单任务成本，联合运行时间只计一次。
"""
    return replace_or_append("".join(lines),"## 当前执行状态",status)


def guarded_write(path, original, updated):
    if path.read_text()!=original:
        raise RuntimeError(f"Document changed concurrently; refusing overwrite: {path}")
    path.write_text(updated)


def render(report):
    complete=report["complete"]
    lines=[HEADING,"",f"状态：{'26臂及全部best/last已完成并通过独立数值验收' if complete else '执行中或尚未通过完整验收，不作最终结论'}。更新时间：{stamp()}。", "",
           "用户确认的P00–P11/V00–V11及J00/J01共26次正式训练，均从零初始化、seed1234、400轮。固定18D、5000壁面support、每任务5000query、QAD、纯MSE；联合双mask各任务等权，联合对单任务last400主比较。", "",
           "单测104项及40子测试通过；真实batch8预检14034、26臂两轮冒烟14038、原R5复核14036完成。历史两目标best/last共2948项复核通过。只有一张GPU可分配，正式14040采用四槽并发、48小时上限；最初14039因12小时限时调整，在12/9/8/7轮中止并归档，不计正式成绩。", ""]
    for task,label in (("pressure","压力，Pa（混合query）"),("velocity","速度，m/s（内部）")):
        lines.extend([f"### {label}","","| 臂 | best R²_cb | last R²_cb | best MAE | best向量RMSE | best径向R² | best周向R² | 判定 |",
                      "|---|---:|---:|---:|---:|---:|---:|---|"])
        indexed={(r["id"],r["checkpoint"]):r for r in report["rows"] if r["task"]==task}
        for aid in dict.fromkeys(r["id"] for r in report["rows"] if r["task"]==task):
            b,l=indexed[(aid,"best")],indexed[(aid,"last")]
            gate=report["gates"].get(aid+"/"+task)
            judgement="历史参照" if gate is None else "待核验" if not complete else "通过开发筛选" if gate["qualified"] else "未通过全部门槛"
            lines.append(f"| {aid} | {f(b['r2_cb'])} | {f(l['r2_cb'])} | {f(b['mae'],3)} | {f(b['vector_rmse'])} | {f(b['radial_r2'])} | {f(b['circ_r2'])} | {judgement} |")
        lines.append("")
    lines.extend(["### 预设比较与长波判读","",
                  "00/01/02/03比较局部结构与直接BT交互；06/07/08/09比较单/多半径与BT交互；03对04/05及09对10/11排查条件注入、尺度和容量。FFN残差结构有差异，不称纯注意力算子因果隔离。", ""])
    if complete:
        for task in ("pressure","velocity"):
            qualified=[key.split('/')[0] for key,g in report["gates"].items() if key.endswith('/'+task) and g["qualified"]]
            longwave=[key.split('/')[0] for key,g in report["gates"].items() if key.endswith('/'+task) and g["longwave_improved"]]
            lines.append(f"- {task}：相对各自父臂通过精度筛选的臂：{'、'.join(qualified) or '无'}；同时通过20/40mm去偏置长波门槛：{'、'.join(longwave) or '无'}。")
            for aid in ("P03","P09") if task=="pressure" else ("V03","V09"):
                compare=[c for c in report["comparisons"] if c["id"]==aid and c["task"]==task and c["checkpoint"]=="best" and c["reference"]==("R5P" if task=="pressure" else "R5V")]
                if compare:
                    c=compare[0];lines.append(f"- {aid}相对历史体场模型：best ΔR²={c['delta_r2']:+.5f}，MAE相对降低={f(100*c['mae_reduction'],2)}%；20/40mm去偏置残差MSE相对降低={f(100*c['lw20_reduction'],2)}/{f(100*c['lw40_reduction'],2)}%。")
        both=all(report["gates"][f"J01/{task}"]["qualified"] for task in ("pressure","velocity"))
        lines.append(f"- J01相对J00：{'两任务均通过，支持本次联合提升' if both else '未同时通过两任务门槛，不称为整体联合提升'}。联合对各自单任务的last判定见comparisons.csv。")
    lines.extend(["", "长波沿既有segment/s按5mm体积分箱，Gaussian sigma5/10/20/40mm同分支平滑、边界重归一；先去病例体积均值偏差，再汇总平滑残差MSE。它是尺度诊断，不是正交频带或理论精度上限。压力同时保留内部/壁面与低压差病例Pa误差；速度保留轴/径/周向与向量误差。", "",
                  "[完整矩阵与协议](../../../training_wss_min/experiments/volume_attention_20260910/README.md) · [全部best/last表](../../../training_wss_min/experiments/volume_attention_20260910/results.md) · [预设配对比较](../../../training_wss_min/experiments/volume_attention_20260910/comparisons.csv) · [逐病例配对变化](../../../training_wss_min/experiments/volume_attention_20260910/per_case_comparisons.csv) · [资源开销](../../../training_wss_min/experiments/volume_attention_20260910/resources.csv) · [联合query成本](../../../training_wss_min/experiments/volume_attention_20260910/joint_query_costs.csv) · [独立验收](../../../training_wss_min/experiments/volume_attention_20260910/report.json) · [工作簿验收](../../../training_wss_min/experiments/volume_attention_20260910/xlsx_acceptance.json)。", "",
                  "以上仅为单seed1234、已暴露test34上的开发筛选。best/last来自同一轨迹；不作显著性或稳定泛化声明。", ""])
    return "\n".join(lines)


def write_paired_cases(report):
    indexed={(r["id"],r["task"],r["checkpoint"]):r for r in report["rows"]}
    cache={};rows=[]
    for pair in report["comparisons"]:
        key=(pair["id"],pair["task"],pair["checkpoint"])
        ref=(pair["reference"],pair["task"],pair["checkpoint"])
        for k in (key,ref):
            if k not in cache:
                row=indexed[k]
                metric=json.loads(Path(row["metrics_path"]).read_text())["test"]
                diagnostic=Path(row["run_dir"])/"eval"/f"ckpt_{k[2]}"/"diagnostics/summary.json"
                diag=json.loads(diagnostic.read_text())["tasks"][k[1]]
                identities=set(metric["per_case"])
                if identities!=set(diag["longwave"]["per_case"]):
                    raise ValueError(f"Longwave case identities differ: {k}")
                if k[1]=="velocity" and identities!=set(diag["per_case_components"]):
                    raise ValueError(f"Velocity component case identities differ: {k}")
                cache[k]=(metric,diag)
        actual,parent=cache[key][0]["per_case"],cache[ref][0]["per_case"]
        if set(actual)!=set(parent):raise ValueError("Paired comparison case identities differ")
        for case in sorted(actual):
            a,b=actual[case]["overall"],parent[case]["overall"]
            if a["n"]!=b["n"]:raise ValueError(f"Paired point counts differ: {key}/{ref}/{case}")
            row=dict(id=key[0],task=key[1],checkpoint=key[2],reference=ref[0],case=case,
                     domain=case.split("/",1)[0],n_points=a["n"],
                     scalar_quantity="pressure_mixed" if key[1]=="pressure" else "speed",
                     scalar_unit="Pa" if key[1]=="pressure" else "m/s")
            def add(name,x,y):
                # Ill-conditioned per-case R² is represented as blank, never silently zero.
                x=float(x) if x is not None and math.isfinite(float(x)) else None
                y=float(y) if y is not None and math.isfinite(float(y)) else None
                row[name]=x;row["reference_"+name]=y
                row["delta_"+name]=x-y if x is not None and y is not None else None
            for field in ("r2","mae","rmse"):add(field,a[field],b[field])
            for field in ("vector_rmse",*[f"{c}_{m}" for c in ("u","v","w","axial","radial","circ") for m in ("r2","rmse")]):
                if key[1]=="velocity":
                    x,y=(cache[k][1]["per_case_components"][case] for k in (key,ref))
                    if field=="vector_rmse":add(field,x["vector_rmse_m_s"],y["vector_rmse_m_s"])
                    else:
                        component,stat=field.rsplit("_",1)
                        stat="r2_casebalanced" if stat=="r2" else stat
                        add(field,x[component][stat],y[component][stat])
                else:add(field,None,None)
            x,y=(cache[k][1]["longwave"]["per_case"][case] for k in (key,ref))
            for field in ("case_bias","full_residual_mse","binned_residual_mse"):
                add(field,x[field],y[field])
            for sigma in (5,10,20,40):
                for field in ("residual_mse","debiased_residual_mse"):
                    add(f"lw{sigma}_{field}",x["sigmas_mm"][str(sigma)][field],y["sigmas_mm"][str(sigma)][field])
            rows.append(row)
    if not rows:raise ValueError("No completed paired case comparisons to write")
    with (EXP/"per_case_comparisons.csv").open("w",newline="") as stream:
        writer=csv.DictWriter(stream,list(rows[0]));writer.writeheader();writer.writerows(rows)
    return len(rows)


def main():
    ap=argparse.ArgumentParser();ap.add_argument("--render-preview",action="store_true");args=ap.parse_args()
    if args.render_preview:
        report=json.loads((EXP/"report.json").read_text());(EXP/"completion_preview.md").write_text(render(report));return
    state=dict(status="running",started_at=stamp(),source_sha256=sha256(__file__),training_job=14040,
               resource_helper_sha256=sha256(Path(__file__).with_name("joint_query_costs.py")))
    save_json(EXP/"completion_status.json",state)
    try:
        subprocess.run([sys.executable,"-u","-m","training_wss_min.tools.report_volume_attention","--validate"],cwd=ROOT,check=True)
        report=json.loads((EXP/"report.json").read_text())
        if not report["complete"]:raise ValueError("Independent acceptance did not complete")
        state["paired_case_rows"]=write_paired_cases(report)
        from joint_query_costs import write_joint_query_costs
        state["joint_query_costs"]=write_joint_query_costs(report,EXP)
        subprocess.run([sys.executable,"-u","-m","training_wss_min.tools.update_volume_attention_xlsx"],cwd=ROOT,check=True)
        document=TRACKER.read_text();updated=replace_or_append(document,HEADING,render(report))
        guarded_write(TRACKER,document,updated)
        readme=EXP/"README.md";text=readme.read_text()
        guarded_write(readme,text,update_readme_status(text))
        heading="## 2026-09-10｜压力/速度26臂完成、独立验收与工作簿回填"
        entry=heading+"\n\n26臂正式400轮及best/last全量评估、分支长波诊断已完成；独立NumPy从保存预测重算60组目标/checkpoint的核心物理指标，检查26条history及共同checkpoint口径；工作簿独立分节回填并核对历史cell/formula/merge保留。详见[跟踪§17](WSS_PINN/WSS_V5_训练实验跟踪.md)。原14039仅为限时调整前中止记录，不计正式成绩。单seed与暴露test34边界保留。\n"
        log=CHANGELOG.read_text()
        if heading not in log:
            first_section=re.search(r"^## ",log,re.MULTILINE)
            i=first_section.start() if first_section else len(log)
            guarded_write(CHANGELOG,log,log[:i].rstrip("\n")+"\n\n"+entry+"\n"+log[i:])
        state.update(status="complete",ended_at=stamp(),validated_task_checkpoints=len(report["validation"]),validated_training_runs=len(report["history_checks"]),workbook_acceptance=str(EXP/"xlsx_acceptance.json"))
        submission_path=EXP/"submission.json"
        submission_document=submission_path.read_text()
        submission=json.loads(submission_document)
        submission["completion"]=dict(status="complete",completed_at=state["ended_at"],
                                      training_job=14040,finalizer_job=14041,
                                      validated_task_checkpoints=state["validated_task_checkpoints"],
                                      validated_training_runs=state["validated_training_runs"],
                                      paired_case_rows=state["paired_case_rows"],
                                      evidence=str(EXP/"completion_status.json"))
        guarded_write(submission_path,submission_document,json.dumps(submission,ensure_ascii=False,indent=2)+"\n")
    except BaseException as exc:
        state.update(status="failed",ended_at=stamp(),error=f"{type(exc).__name__}: {exc}",traceback=traceback.format_exc())
        save_json(EXP/"completion_status.json",state)
        raise
    save_json(EXP/"completion_status.json",state)
    print(json.dumps(state,ensure_ascii=False,indent=2))


if __name__=="__main__":
    import fcntl
    with (EXP/".reporting.lock").open("a") as reporting_lock:
        fcntl.flock(reporting_lock,fcntl.LOCK_EX)
        main()

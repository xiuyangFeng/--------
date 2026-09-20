"""Read-only scientific aggregation; writes only the new experiment's reports."""
from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

import numpy as np

from training_wss_min.tools.volume_attention_common import CONFIGS, EXP, ROOT, RUNS, fingerprints, manifest, reference_dir, save_json, stamp, task_metrics_path


def read_json(path, default=None):
    return json.loads(Path(path).read_text()) if Path(path).is_file() else default


def finite(value):
    return float(value) if value is not None and math.isfinite(float(value)) else None


def reduction(value, reference):
    return 1-value/reference if value is not None and reference is not None and reference>0 else None


def metrics_row(aid,task,ck,run,parent,status,record):
    metric_path=task_metrics_path(run,ck,task,aid.startswith("J"))
    metric=read_json(metric_path,{})
    m=metric.get("test",{})
    diag=read_json(Path(run)/"eval"/f"ckpt_{ck}"/"diagnostics/summary.json",{}).get("tasks",{}).get(task,{})
    get=lambda section,key:finite(m.get(section,{}).get(key))
    components=diag.get("components",{})
    lw=diag.get("longwave",{}).get("summary",{}).get("sigmas_mm",{})
    training_resources=read_json(Path(run)/"training_diagnostics.json",{})
    row=dict(id=aid,task=task,checkpoint=ck,status=status,run_dir=str(run),parent=parent,
             r2_cb=get("field_casebalanced","r2"),mae=get("field","mae"),rmse=get("field","rmse"),
             case_median=get("aggregate","r2_casemed"),case_p10=get("aggregate","r2_casep10"),
             negative_cases=get("aggregate","r2_negative_cases"),
             vector_rmse=finite(m.get("vector",{}).get("vector_rmse_m_s")),
             parameters=get("efficiency","parameters"),eval_seconds=get("efficiency","elapsed_seconds"),
             train_seconds=record.get("stages",{}).get("train",{}).get("elapsed_seconds"),
             train_peak_cuda_allocated_mib=finite(training_resources.get("peak_cuda_allocated_mb")),
             eval_peak_cuda_allocated_mib=(get("efficiency","peak_cuda_memory_bytes") or 0)/1024**2 if m else None,
             delta_r2_parent=None,gate="待完成",metrics_path=str(metric_path),
             groups=m.get("group_casebalanced",{}),pressure_query_groups=m.get("query_groups",{}))
    for key in ("axial","radial","circ"):
        row[f"{key}_r2"]=finite(components.get(key,{}).get("r2_casebalanced"))
    for sigma in (20,40):
        row[f"lw{sigma}"]=finite(lw.get(str(sigma),{}).get("debiased_residual_mse_casemean"))
    if diag:
        row["longwave_summary"]=diag["longwave"]["summary"]
    return row,m


def evaluate_gate(best,last,parent_best,parent_last,task,primary_checkpoint="best"):
    if any(r.get("r2_cb") is None for r in (best,last,parent_best,parent_last)):
        return {"qualified":False,"issues":["missing metrics"],"longwave_improved":False}
    issues=[]
    principal,reference=(last,parent_last) if primary_checkpoint=="last" else (best,parent_best)
    if principal["r2_cb"]-reference["r2_cb"]<.01-1e-12:issues.append(f"{primary_checkpoint} ΔR²<0.01")
    key="mae" if task=="pressure" else "vector_rmse"
    if (reduction(principal.get(key),reference.get(key)) or 0)<.03-1e-12:issues.append(f"{primary_checkpoint} {key} reduction<3%")
    if last["r2_cb"]<=parent_last["r2_cb"]:issues.append("last R² not improved")
    if last.get(key) is None or parent_last.get(key) is None or last[key]>=parent_last[key]:issues.append(f"last {key} not improved")
    if task=="velocity":
        protected=(("last",last,parent_last),) if primary_checkpoint=="last" else (("best",best,parent_best),("last",last,parent_last))
        for ck,a,b in protected:
            for component in ("radial_r2","circ_r2"):
                if a.get(component) is None or b.get(component) is None or a[component]-b[component]<-.01-1e-12:
                    issues.append(f"{ck} {component} protection failed")
    low={str(s):reduction(principal.get(f"lw{s}"),reference.get(f"lw{s}")) for s in (20,40)}
    lw_pass=all(v is not None and v>=.10-1e-12 for v in low.values()) and not issues
    return dict(qualified=not issues,issues=issues,primary_checkpoint=primary_checkpoint,longwave_reductions=low,longwave_improved=lw_pass)


def independently_validate_predictions(run,ck,task,metric):
    """Independent NumPy moments, reading saved predictions once per case."""
    base=Path(run)/"eval"/f"ckpt_{ck}"/"predictions/test"
    saved=read_json(base/"manifest.json",{})
    moments=[];vectors=[];identities=[]
    for case in saved.get("cases",[]):
        with np.load(base/case["file"],allow_pickle=False) as data:
            identity=str(data["unit_id"].item());identities.append(identity)
            query=np.asarray(data["query_idx"]);kind=np.asarray(data["point_kind"]);nwall=int(data["n_wall"])
            if identity!=case["unit_id"] or len(np.unique(query))!=len(query) or len(query)!=case["n_query"]:
                raise ValueError("Saved case identity/query rows are inconsistent or duplicated")
            if not np.array_equal(kind,(query>=nwall).astype(kind.dtype)):
                raise ValueError("Saved point_kind does not match original wall/interior rows")
            with np.load(Path(case["bundle_path"]).parent/"volume.npz") as volume:
                ncells=int(volume["n_cells"])
            target=str(data["target"].item())
            expected=np.arange(nwall,nwall+ncells) if target=="velocity" else np.arange(nwall+ncells)
            if not np.array_equal(query,expected):raise ValueError("Saved full-query coverage mismatch")
            p=np.asarray(data["pred_raw"],dtype=np.float64);y=np.asarray(data["true_raw"],dtype=np.float64)
            if task=="velocity":
                keep=data["point_kind"]==1
                p,y=p[keep,:3],y[keep,:3]
                vectors.append((len(y),float(np.square(p-y).sum())))
                p,y=np.linalg.norm(p,axis=1),np.linalg.norm(y,axis=1)
            else:
                p=p[:,3] if p.ndim==2 and p.shape[1]==4 else p.reshape(-1)
                y=y[:,3] if y.ndim==2 and y.shape[1]==4 else y.reshape(-1)
            if not np.isfinite(p).all() or not np.isfinite(y).all():raise ValueError("nonfinite saved predictions")
            moments.append([len(y),y.mean(),np.square(y).mean(),np.square(p-y).mean(),np.abs(p-y).mean()])
    if not moments:return dict(passed=False,issues=["no saved prediction manifest"])
    if len(set(identities))!=len(identities) or set(identities)!=set(metric["per_case"]):
        raise ValueError("Saved case set differs from metric cases or has duplicates")
    a=np.asarray(moments);gm=a[:,1].mean();pm=np.average(a[:,1],weights=a[:,0])
    expected={"cb_r2":1-a[:,3].mean()/np.mean(a[:,2]-2*gm*a[:,1]+gm*gm),
              "pooled_r2":1-np.average(a[:,3],weights=a[:,0])/(np.average(a[:,2],weights=a[:,0])-pm*pm),
              "pooled_mae":np.average(a[:,4],weights=a[:,0]),
              "pooled_rmse":np.sqrt(np.average(a[:,3],weights=a[:,0]))}
    actual={"cb_r2":metric["field_casebalanced"]["r2"],"pooled_r2":metric["field"]["r2"],
            "pooled_mae":metric["field"]["mae"],"pooled_rmse":metric["field"]["rmse"]}
    if vectors:
        expected["vector_rmse"]=np.sqrt(sum(v[1] for v in vectors)/sum(v[0] for v in vectors))
        actual["vector_rmse"]=metric["vector"]["vector_rmse_m_s"]
    checks={k:dict(recomputed=float(v),reported=actual[k],passed=bool(np.isclose(v,actual[k],rtol=2e-5,atol=2e-6))) for k,v in expected.items()}
    count_ok=int(a[:,0].sum())==metric["field"]["n"]
    return dict(passed=len(moments)==34 and count_ok and all(c["passed"] for c in checks.values()),n_cases=len(moments),n_points=int(a[:,0].sum()),count_matches_metrics=count_ok,checks=checks)


def build_report(validate=False):
    matrix=manifest();state=read_json(EXP/"execution/queue_status.json",{})
    complete=state.get("status")=="complete" and not state.get("source_changed",True)
    records=state.get("arms",{});rows=[];metrics={};validations={}
    arms=[dict(id=f"R5{k}",task="pressure" if k=="P" else "velocity",parent=None,run_dir=reference_dir(k)) for k in ("P","V")]+matrix["arms"]
    for arm in arms:
        aid=arm["id"];run=arm.get("run_dir",RUNS/arm.get("run_name",""))
        tasks=("pressure","velocity") if arm["task"]=="joint" else (arm["task"],)
        status="历史参照" if aid.startswith("R5") else records.get(aid,{}).get("status","pending")
        if status=="complete" and not complete:status="已输出，整批待核验"
        for task in tasks:
            parent=arm.get("parent")
            if parent=="P01+V01":parent="P01" if task=="pressure" else "V01"
            for ck in ("best","last"):
                row,m=metrics_row(aid,task,ck,run,parent,status,records.get(aid,{}))
                rows.append(row);metrics[(aid,task,ck)]=m
                if validate and m:
                    validations[f"{aid}/{task}/{ck}"]=independently_validate_predictions(run,ck,task,m)
    index={(r["id"],r["task"],r["checkpoint"]):r for r in rows}
    gates={};comparisons=[]
    for arm in matrix["arms"]:
        aid=arm["id"]
        for task in (("pressure","velocity") if arm["task"]=="joint" else (arm["task"],)):
            a=index[(aid,task,"best")];last=index[(aid,task,"last")];parent=a["parent"]
            base=index[(parent,task,"best")];base_last=index[(parent,task,"last")]
            gate=evaluate_gate(a,last,base,base_last,task,primary_checkpoint="last" if aid.startswith("J") and not parent.startswith("J") else "best")
            gate["interpretation"]="single-seed development screening"
            gates[f"{aid}/{task}"]=gate
            for row,ref in ((a,base),(last,base_last)):
                if row["r2_cb"] is not None and ref["r2_cb"] is not None:row["delta_r2_parent"]=row["r2_cb"]-ref["r2_cb"]
                row["gate"]=("通过开发筛选" if gate["qualified"] else "未通过："+"; ".join(gate["issues"])) if complete else "整批待核验"
            candidates=set([parent,*arm.get("comparisons",[]),"P00" if task=="pressure" else "V00","R5P" if task=="pressure" else "R5V"])
            for refid in sorted(candidates):
                if (refid,task,"best") not in index:continue
                for ck in ("best","last"):
                    x,y=index[(aid,task,ck)],index[(refid,task,ck)]
                    if x["r2_cb"] is None or y["r2_cb"] is None:continue
                    cross_primary="last" if aid.startswith("J") and not refid.startswith("J") else "best"
                    cross_gate=evaluate_gate(a,last,index[(refid,task,"best")],index[(refid,task,"last")],task,primary_checkpoint=cross_primary)
                    comparisons.append(dict(id=aid,task=task,reference=refid,checkpoint=ck,
                                            primary_for_joint_vs_single=ck=="last" if aid.startswith("J") and not refid.startswith("J") else ck=="best",
                                            primary_checkpoint=cross_primary,qualified=cross_gate["qualified"],gate_issues="; ".join(cross_gate["issues"]),
                                            delta_r2=x["r2_cb"]-y["r2_cb"],mae_reduction=reduction(x["mae"],y["mae"]),
                                            vector_rmse_reduction=reduction(x["vector_rmse"],y["vector_rmse"]),
                                            lw20_reduction=reduction(x["lw20"],y["lw20"]),lw40_reduction=reduction(x["lw40"],y["lw40"])))
    factorials=[]
    for prefix,task in (("P","pressure"),("V","velocity")):
        for ids in matrix["factorials"]:
            for ck in ("best","last"):
                vals=[index[(prefix+n,task,ck)]["r2_cb"] for n in ids]
                if all(v is not None for v in vals):
                    factorials.append(dict(task=task,arms=[prefix+n for n in ids],checkpoint=ck,r2_interaction=vals[3]-vals[2]-vals[1]+vals[0]))
    history_checks={}
    if validate:
        import torch
        for arm in matrix["arms"]:
            run=RUNS/arm["run_name"]
            if not (run/"history.jsonl").exists():continue
            history=[json.loads(line) for line in (run/"history.jsonl").read_text().splitlines()]
            best=torch.load(run/"ckpt_best.pt",map_location="cpu",weights_only=False)
            last=torch.load(run/"ckpt_last.pt",map_location="cpu",weights_only=False)
            expected=min(history,key=lambda h:h["train_loss"])["epoch"]
            history_checks[arm["id"]]=dict(passed=[h["epoch"] for h in history]==list(range(400)) and best["epoch"]==expected and last["epoch"]==399,
                                            epochs=len(history),best_epoch=best["epoch"],minimum_loss_epoch=expected,last_epoch=last["epoch"])
        complete=complete and len(validations)==60 and all(v["passed"] for v in validations.values()) and len(history_checks)==26 and all(v["passed"] for v in history_checks.values())
    complete=bool(complete and validate)
    for row in rows:
        key=f"{row['id']}/{row['task']}"
        if key not in gates:continue
        g=gates[key]
        row["gate"]=("通过开发筛选" if g["qualified"] else "未通过："+"; ".join(g["issues"])) if complete else "整批待核验"
        if not complete and row["status"]=="complete":row["status"]="已输出，整批待核验"
    for g in gates.values():g["accepted_scientific_result"]=complete
    return dict(matrix_id=matrix["matrix_id"],timestamp=stamp(),complete=complete,rows=rows,gates=gates,
                comparisons=comparisons,factorials=factorials,validation=validations,history_checks=history_checks,
                queue_status=state.get("status","not_started"),limitations="seed1234 only; test34 exposed development; best/last are not independent repetitions")


def fmt(v):return "—" if v is None else f"{v:.4f}" if isinstance(v,(float,int)) else str(v).replace("|","\\|")


def write_outputs(report):
    def clean(value):
        if isinstance(value,dict):return {k:clean(v) for k,v in value.items()}
        if isinstance(value,list):return [clean(v) for v in value]
        if isinstance(value,float) and not math.isfinite(value):return None
        return value
    save_json(EXP/"report.json",clean(report))
    lines=["# 压力/速度全局注意力：26臂结果", "",f"更新时间：{report['timestamp']}；整批核验完成：{report['complete']}。", "",
           "固定18D、seed1234、400epoch；best按训练损失，last固定400轮；联合对单任务以last为主。", ""]
    for task,title in (("pressure","压力（Pa，混合query；内部/壁面分组见metrics.json）"),("velocity","速度（m/s，幅值及向量分别评价）")):
        lines.extend([f"## {title}","","| ID/ckpt | 状态 | R²_cb | Δ父臂 | MAE | RMSE | 向量RMSE | 径向R² | 周向R² | 20mm去bias MSE | 40mm去bias MSE |","|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|"])
        for row in report["rows"]:
            if row["task"]!=task:continue
            values=[row["id"]+"/"+row["checkpoint"],row["status"],*[row[k] for k in ("r2_cb","delta_r2_parent","mae","rmse","vector_rmse","radial_r2","circ_r2","lw20","lw40")]]
            lines.append("| "+" | ".join(fmt(v) for v in values)+" |")
        lines.append("")
    lines.extend(["## 预登记筛选结果","","| 臂/任务 | 精度筛选 | 长波筛选 | 未通过原因 |","|---|---|---|---|"])
    for key,g in report["gates"].items():
        lines.append(f"| {key} | {g['qualified'] if report['complete'] else '待核验'} | {g['longwave_improved'] if report['complete'] else '待核验'} | {'；'.join(g['issues'])} |")
    lines.extend(["","单seed、暴露test34仅支持开发筛选。FFN与BT残差拓扑不同；注意力激活不等于因果有效。", ""])
    (EXP/"results.md").write_text("\n".join(lines))
    columns=["id","task","checkpoint","status","r2_cb","delta_r2_parent","mae","rmse","vector_rmse","case_median","case_p10","negative_cases","axial_r2","radial_r2","circ_r2","lw20","lw40","parameters","train_seconds","eval_seconds","parent","gate","run_dir"]
    with (EXP/"metrics.csv").open("w",newline="") as f:
        writer=csv.DictWriter(f,columns,extrasaction="ignore");writer.writeheader();writer.writerows(report["rows"])
    with (EXP/"comparisons.csv").open("w",newline="") as f:
        if report["comparisons"]:
            writer=csv.DictWriter(f,list(report["comparisons"][0]));writer.writeheader();writer.writerows(report["comparisons"])
    resource_columns=["id","task","checkpoint","parameters","train_seconds","eval_seconds","train_peak_cuda_allocated_mib","eval_peak_cuda_allocated_mib"]
    with (EXP/"resources.csv").open("w",newline="") as f:
        writer=csv.DictWriter(f,resource_columns,extrasaction="ignore");writer.writeheader();writer.writerows(report["rows"])
    if any(r["r2_cb"] is not None for r in report["rows"]):
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig,axes=plt.subplots(1,2,figsize=(15,5))
        for ax,task in zip(axes,("pressure","velocity")):
            rows=[r for r in report["rows"] if r["task"]==task and r["checkpoint"]=="best" and r["r2_cb"] is not None]
            ax.bar([r["id"] for r in rows],[r["r2_cb"] for r in rows]);ax.tick_params(axis="x",rotation=65)
            ax.set(title=task+" / best",ylabel="Physical case-balanced R2")
        fig.tight_layout();fig.savefig(EXP/"comparison.png",dpi=180);fig.savefig(EXP/"comparison.pdf");plt.close(fig)


def main():
    ap=argparse.ArgumentParser();ap.add_argument("--validate",action="store_true");args=ap.parse_args()
    report=build_report(args.validate);write_outputs(report)
    print(f"rows={len(report['rows'])}, complete={report['complete']}, queue={report['queue_status']}")
    if args.validate and not report["complete"]:raise SystemExit("Incomplete or failed independent acceptance; inspect report.json")


if __name__=="__main__":main()

"""入库前分布预检（只读原始导出，不建库）：新 95 单元 vs 在库 170，按队列比较峰值步 1162 的壁面 WSS 分位、壁面节点数/点间距、包围盒、壁面压力、UDF 入口面积。
混合区导出（整网格节点）按 wall-shear>0 取壁面子集（仅用于统计，入库时按坐标匹配）。输出 precheck_distribution.csv + 汇总表。"""
import json, glob, sys, os
import numpy as np, pandas as pd
from scipy.spatial import cKDTree
from concurrent.futures import ProcessPoolExecutor
ROOT="/public/newhome/cy/Digital_twin/GNN"; HERE=os.path.dirname(os.path.abspath(__file__))
split=json.load(open(f"{ROOT}/data_wss_v5/views_v5_1/wss_min_view_v1/split_V5_train136_test34.json"))
lib=split["train_cases"]+split["test_cases"]; man=json.load(open(f"{HERE}/new_units_manifest.json")); new=man["new_units"]
def one(args):
    cid,group=args; d=f"{ROOT}/data_new/{cid}"
    try:
        f=sorted(glob.glob(d+"/ascii/*-1162"))[0]
        hdr=open(f).readline(); sep="," if "," in hdr else r"\s+"
        df=pd.read_csv(f,sep=sep,engine="c" if sep=="," else "python",usecols=[1,2,3,4,5],header=0,names=["x","y","z","p","wss"],skiprows=1,dtype=float)
        n_raw=len(df); m=df.wss.values>1e-9; mixed=m.mean()<0.5
        if mixed: df=df[m]
        xyz=df[["x","y","z"]].values*1000.0; w=df.wss.values; p=df.p.values
        idx=np.random.default_rng(0).choice(len(xyz),min(len(xyz),8000),replace=False); dd,_=cKDTree(xyz).query(xyz[idx],k=2); sp=float(np.median(dd[:,1]))
        bb=np.ptp(xyz,0)
        out=dict(case=cid,group=group,cohort=cid.split("/")[0]+("/"+cid.split("/")[2] if cid.startswith("ILO") else ""),n_raw=n_raw,n_wall=len(df),mixed=mixed,
                 spacing_mm=sp,bbox_x=bb[0],bbox_y=bb[1],bbox_z=bb[2],
                 wss_p50=np.percentile(w,50),wss_p90=np.percentile(w,90),wss_p99=np.percentile(w,99),wss_max=w.max(),wss_mean=w.mean(),
                 p_mean=p.mean(),p_p99=np.percentile(p,99))
        u=(glob.glob(d+"/udf-inlet4.c")+glob.glob(d+"/udf-inlet.c"))
        if u:
            import re; t=open(u[0],errors="ignore").read(); mm=re.search(r"b8 \* sin\(8 \* tt \* w\)\)\s*/\s*([0-9.eE+-]+)",t); out["inlet_area_mm2"]=float(mm.group(1))*1e6 if mm else np.nan
            rs=re.findall(r"R1\s*=\s*([0-9.E+-]+);\s*R2\s*=\s*([0-9.E+-]+);",t); G=[1/(float(a)+float(b)) for a,b in rs]; out["R_total"]=1/sum(G) if G else np.nan
        return out
    except Exception as e:
        return dict(case=cid,group=group,error=f"{type(e).__name__}: {e}")
if __name__=="__main__":
    jobs=[(c,"lib170") for c in lib]+[(c,"new95") for c in new]
    with ProcessPoolExecutor(16) as ex: rows=list(ex.map(one,jobs))
    df=pd.DataFrame(rows); df.to_csv(f"{HERE}/precheck_distribution.csv",index=False)
    err=df[df.get("error").notna()] if "error" in df else df.iloc[0:0]
    print("errors:",len(err)); print(err[["case","error"]].to_string() if len(err) else "")
    ok=df[df.get("error").isna()] if "error" in df else df
    cols=["n_wall","spacing_mm","bbox_z","wss_p50","wss_p90","wss_p99","wss_max","p_p99","inlet_area_mm2","R_total"]
    def q(s): return f"{s.quantile(.1):.3g}/{s.median():.3g}/{s.quantile(.9):.3g}"
    for coh in ["AG","AAA","ILO/before","ILO/after"]:
        for grp in ["lib170","new95"]:
            sub=ok[(ok.cohort==coh)&(ok.group==grp)]
            if not len(sub): continue
            print(f"\n[{coh} {grp} n={len(sub)}]  (p10/中位/p90)")
            print("  "+"  ".join(f"{c}={q(sub[c])}" for c in cols if c in sub))
    # outliers: new units outside library [p1,p99] per cohort family for key columns
    print("\n=== 新单元超出同队列在库 [min,max] 的项")
    for _,r in ok[ok.group=="new95"].iterrows():
        fam=r.cohort.split("/")[0]; ref=ok[(ok.group=="lib170")&(ok.cohort.str.startswith(fam))]
        flags=[]
        for c in ["n_wall","spacing_mm","wss_p99","wss_max","p_p99","inlet_area_mm2","R_total","bbox_z"]:
            lo,hi=ref[c].min(),ref[c].max()
            if pd.notna(r[c]) and (r[c]<lo or r[c]>hi): flags.append(f"{c}={r[c]:.3g}∉[{lo:.3g},{hi:.3g}]")
        if flags: print(f"  {r.case:30s} "+"; ".join(flags))

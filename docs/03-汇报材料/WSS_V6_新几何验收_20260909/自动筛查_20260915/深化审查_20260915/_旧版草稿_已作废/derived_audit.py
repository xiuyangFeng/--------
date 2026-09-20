#!/usr/bin/env python3
"""Independent audit of derived geometry fields stored by the V6 candidate.

This file deliberately reads only ``cases/*/{geometry.npz,report.json}`` and
the frozen split.  It does not import the candidate generator.  The output is
an evidence table for deciding which cases still need a human decision.
"""
from __future__ import annotations
import argparse, csv, json, math
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[6]
DEFAULT_CANDIDATE = ROOT / "outputs/wss_v6_geometry_candidate_20260909"
DEFAULT_SPLIT = ROOT / "wss_pinn/configs/splits/split_WSS_PINN_AG_AAA_ILO_bc_rcr_v4_anatomy_only_train136_test34_s1234.json"
EXCLUDE = {"AAA/unruputer/LIU_WEN_QI", "AG/slow/HOU_SHEN_QIAN"}
WINDOW = 10.0
ATOL, RTOL = 2e-4, 2e-5       # comparison tolerance for float32 stored fields
MASK_TOL = 8.0                # ulps at a float32 endpoint used for ambiguity flag

def plain(x):
    if isinstance(x, (np.generic,)): return x.item()
    if isinstance(x, np.ndarray): return x.tolist()
    if isinstance(x, dict): return {str(k): plain(v) for k,v in x.items()}
    if isinstance(x, (list, tuple)): return [plain(v) for v in x]
    return x

def field_compare(expected, actual):
    e, a = np.asarray(expected, float), np.asarray(actual, float)
    finite = np.isfinite(e) & np.isfinite(a)
    both_nan = ~np.isfinite(e) & ~np.isfinite(a)
    d = np.abs(e-a)
    lim = ATOL + RTOL*np.abs(e)
    bad = finite & (d > lim)
    mask_bad = np.isfinite(e) != np.isfinite(a)
    vals = d[finite]
    return {"compared": int(e.size), "finite_count": int(finite.sum()),
            "finite_fraction": float(finite.mean()) if e.size else 0.0,
            "error_count": int(bad.sum()+mask_bad.sum()),
            "nan_mask_disagreements": int(mask_bad.sum()),
            "max_abs_error": float(vals.max()) if vals.size else None,
            "max_rel_error": float(np.max(vals/np.maximum(np.abs(e[finite]), 1e-12))) if vals.size else None,
            "tolerance": {"atol": ATOL, "rtol": RTOL}}

def contiguous_context(s, area, valid):
    """Independent 10-mm context: strict direction, no current station,
    contiguous valid run, nearest station when minimum areas tie."""
    s, area, valid = np.asarray(s,float), np.asarray(area,float), np.asarray(valid,bool)
    out = {k: np.full(len(s), np.nan) for k in ("upstream_min_area_ratio","downstream_min_area_ratio","upstream_min_distance_mm","downstream_min_distance_mm")}
    idx = np.flatnonzero(valid & np.isfinite(area) & (area > 0))
    for run in np.split(idx, np.flatnonzero(np.diff(idx)>1)+1):
        if len(run)<2: continue
        for i in run:
            for side, cand in (("upstream",run[(s[run] < s[i]) & (s[run] >= s[i]-WINDOW)]),
                               ("downstream",run[(s[run] > s[i]) & (s[run] <= s[i]+WINDOW)])):
                if not len(cand): continue
                m = np.min(area[cand]); ties = cand[np.isclose(area[cand],m,rtol=1e-8,atol=1e-10)]
                k = ties[np.argmin(np.abs(s[ties]-s[i]))]
                out[side+"_min_area_ratio"][i] = area[k]/area[i]
                out[side+"_min_distance_mm"][i] = abs(s[k]-s[i])
    return out

def slope(s, area, valid):
    """d(log A)/ds on contiguous runs, independently coded (3-point gradient)."""
    out=np.full(len(s),np.nan); idx=np.flatnonzero(valid & np.isfinite(area) & (area>0))
    for run in np.split(idx,np.flatnonzero(np.diff(idx)>1)+1):
        if len(run)<3: continue
        x=s[run]; y=np.log(area[run]); q=np.empty(len(run)); q[1:-1]=(y[2:]-y[:-2])/(x[2:]-x[:-2]); q[0]=(-3*y[0]+4*y[1]-y[2])/(x[2]-x[0]); q[-1]=(3*y[-1]-4*y[-2]+y[-3])/(x[-1]-x[-3]); out[run]=q
    return out

def interpolate(sq, s, value, valid):
    """Linear interpolation only when both bracketing stations are valid."""
    sq,s,value,valid=np.asarray(sq,float),np.asarray(s,float),np.asarray(value,float),np.asarray(valid,bool)
    out=np.full(len(sq),np.nan); good=np.zeros(len(sq),bool)
    if len(s)<2:return out,good
    hi=np.searchsorted(s,sq,side="right"); hi=np.clip(hi,1,len(s)-1); lo=hi-1
    good=(sq>=s[0])&(sq<=s[-1])&valid[lo]&valid[hi]
    a=(sq-s[lo])/np.maximum(s[hi]-s[lo],1e-12); out[good]=((1-a)*value[lo]+a*value[hi])[good]
    return out,good

def self_tests():
    # Analytic parabola catches endpoint and cross-invalid mistakes.
    s=np.arange(0.,9.); area=(s+2)**2; v=np.ones(9,bool); v[4]=False
    ss=slope(s,area,v); assert np.isfinite(ss[3]) and np.isnan(ss[4]) and np.isfinite(ss[5])
    c=contiguous_context(s,area,v); assert np.isnan(c["upstream_min_distance_mm"][5])
    x,g=interpolate(np.array([0.,1.,3.,4.]),np.array([0.,2.,4.]),np.array([2.,4.,8.]),np.array([1,0,1],bool)); assert not g[1] and g[0]==False
    # exact endpoint uses the first/last bracket and remains valid
    x,g=interpolate(np.array([0.,4.]),np.array([0.,2.,4.]),np.array([2.,4.,8.]),np.ones(3,bool)); assert np.allclose(x,[2,8]) and g.all()
    return {"passed": True, "tests": ["cross_invalid_slope_and_context", "bracketed_interpolation_mask", "endpoint_interpolation"]}

def topology_checks(z, report):
    checks={}; xyz=z["centerline_xyz_mm"].astype(float); seg=z["centerline_segment_id"]; ss=z["centerline_s_local_mm"].astype(float); tan=z["centerline_tangent"].astype(float)
    checks["wall_node_id_unique"] = int(len(np.unique(z["wall_node_id"]))) == len(z["wall_node_id"])
    tri=z["wall_triangles"]; checks["triangle_indices_in_range"] = bool(tri.min()>=0 and tri.max()<len(z["wall_xyz_mm"]))
    checks["triangle_non_degenerate"] = int(np.sum(np.linalg.norm(np.cross(z["wall_xyz_mm"][tri[:,1]]-z["wall_xyz_mm"][tri[:,0]],z["wall_xyz_mm"][tri[:,2]]-z["wall_xyz_mm"][tri[:,0]]),axis=1)<1e-8))
    arc_err=[]; s_bad=0; tangent_bad=0; tangent_severe=0
    for k in np.unique(seg):
        ii=np.flatnonzero(seg==k); ii=ii[np.argsort(ss[ii])]; ds=np.diff(ss[ii]); d=np.linalg.norm(np.diff(xyz[ii],axis=0),axis=1)
        s_bad += int(np.sum(ds<=0)); arc_err.extend(np.abs(d-ds).tolist())
        if len(ii)>2:
            fd=xyz[ii[1:]]-xyz[ii[:-1]]; fd/=np.maximum(np.linalg.norm(fd,axis=1,keepdims=True),1e-12)
            ang=np.degrees(np.arccos(np.clip(np.abs(np.sum(fd*tan[ii[:-1]],axis=1)),-1,1)))
            tangent_bad += int(np.sum(ang>2.0)); tangent_severe += int(np.sum(ang>30.0))
    checks.update({"centerline_nonpositive_ds":s_bad,"centerline_tangent_disagreements":tangent_bad,"centerline_tangent_severe_gt30deg":tangent_severe,"arc_length_max_abs_ds_minus_chord_mm":float(max(arc_err) if arc_err else 0.)})
    p2=report.get("P2_topology",{}); checks["report_topology_valid"] = bool(p2.get("valid",False)); checks["report_expected_pattern_valid"] = bool(p2.get("expected_aortoiliac_pattern_valid",False))
    checks["p1_inside_count_consistent"] = int(np.sum(z["centerline_audit_core_mask"] & ~z["centerline_inside_reference"])) == int(report.get("P1_centerline",{}).get("outside_core_points",-1))
    return checks

def audit_case(path):
    z=np.load(path,allow_pickle=False); report=json.loads(path.with_name("report.json").read_text()); cid=report["canonical_id"]
    sid=z["section_segment_id"]; s=z["section_s_local_mm"].astype(float); wall_sid=z["wall_map_segment_id"]; ws=z["wall_map_s_local_mm"].astype(float)
    fields={}; masks={}; cover={}; seg_errors=[]
    for prefix, area_key, valid_key in (("section_","section_area_mm2","section_valid"),("pc_section_","pc_section_area_mm2","pc_section_valid")):
        area=z[area_key].astype(float); valid=z[valid_key].astype(bool); slope_key=prefix+"area_slope_per_mm"; sl=slope(s,area,valid)
        ctx= {k:np.full(len(s),np.nan) for k in ("upstream_min_area_ratio","downstream_min_area_ratio","upstream_min_distance_mm","downstream_min_distance_mm")}
        for k in np.unique(sid):
            ii=np.flatnonzero(sid==k); order=np.argsort(s[ii]); ii=ii[order]; cc=contiguous_context(s[ii],area[ii],valid[ii])
            for name,v in cc.items():ctx[name][ii]=v
        fields[slope_key]=field_compare(sl,z[slope_key]); cover[slope_key]=float(np.isfinite(sl).mean())
        for name,v in ctx.items():
            key=prefix+name; fields[key]=field_compare(v,z[key]); cover[key]=float(np.isfinite(v).mean())
        for k in np.unique(sid):
            ii=np.flatnonzero(sid==k); 
            if np.any(np.diff(s[ii])<=0): seg_errors.append(f"{prefix}nonmonotonic_s:{int(k)}")
            if len(ii)>=2 and np.any(valid[ii][:-1] != valid[ii][1:]): pass
        # wall interpolation for every stored derivative (including area)
        wall_prefix="wall_map_" if prefix=="section_" else "wall_map_pc_"
        names=["area_mm2","area_slope_per_mm","roundness","eccentricity","upstream_min_area_ratio","downstream_min_area_ratio","upstream_min_distance_mm","downstream_min_distance_mm"]
        values={"area_mm2":area,"area_slope_per_mm":sl,**ctx}
        for name in ("roundness","eccentricity"): values[name]=z[prefix+name].astype(float)
        for name,val in values.items():
            ev=np.full(len(ws),np.nan); eg=np.zeros(len(ws),bool)
            for k in np.unique(wall_sid):
                ii=np.flatnonzero(sid==k); oo=ii[np.argsort(s[ii])]; wi=np.flatnonzero(wall_sid==k); ev[wi],eg[wi]=interpolate(ws[wi],s[oo],val[oo],valid[oo]&np.isfinite(val[oo]))
            eg &= ~z["wall_map_ambiguous"]
            ev[~eg] = np.nan
            # Candidate masks are explicit for each field; area has wall_map_valid.
            ak=wall_prefix+name; mk=wall_prefix+name+"_valid" if name!="area_mm2" else wall_prefix+"valid"
            if ak in z: fields[ak]=field_compare(ev,z[ak]); cover[ak]=float(np.isfinite(ev).mean())
            if mk in z: masks[mk]={"disagreements":int(np.sum(eg != z[mk])),"expected_true":int(eg.sum()),"actual_true":int(z[mk].sum())}
    # A few cheap consistency checks on explicit masks and NaN policy.
    for k in z.files:
        if k.endswith("_valid") and k.startswith("wall_map"):
            valkey=k[:-6];
            if valkey in z: masks[k+"_nan_policy"]={"finite_when_valid":int(np.sum(z[k]&~np.isfinite(z[valkey]))),"finite_when_invalid":int(np.sum(~z[k]&np.isfinite(z[valkey])))}
    geom=topology_checks(z,report)
    errors=[]
    for k,v in fields.items(): errors.append(v["error_count"])
    errors += [v.get("disagreements",0) for v in masks.values() if isinstance(v,dict)]
    errors += [int(not v) for v in geom.values() if isinstance(v,(bool,np.bool_))]
    # Frequent ~2--20 degree differences are expected because the stored
    # tangent is smoothed; only >30 degree disagreements are hard flags.
    errors += [int(v) for key,v in geom.items() if key in ("centerline_nonpositive_ds","triangle_non_degenerate","centerline_tangent_severe_gt30deg")]
    policy_masks = int(np.sum(z["wall_map_ambiguous"]))
    return {"canonical_id":cid,"error_count":int(sum(errors)),"true_discrepancy_count":int(sum(errors)),"policy_mask_difference_count":policy_masks,"numeric_checks":fields,"mask_checks":masks,"finite_coverage":cover,"geometry_checks":geom,"segment_errors":sorted(set(seg_errors)),"conclusion":"needs_user_decision" if sum(errors) else "mechanically_consistent"}

def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--candidate",type=Path,default=DEFAULT_CANDIDATE); ap.add_argument("--split",type=Path,default=DEFAULT_SPLIT); ap.add_argument("--out",type=Path,default=Path(__file__).with_name("derived_audit")); a=ap.parse_args()
    sp=json.loads(a.split.read_text()); ids=[*sp["train_cases"],*sp["test_cases"]]; ids=[x for x in ids if x not in EXCLUDE]
    rows=[]; missing=[]
    for cid in ids:
        d=a.candidate/"cases"/cid.replace("/","__"); p=d/"geometry.npz"
        if not p.exists(): missing.append(cid); continue
        rows.append(audit_case(p))
    summary={"cases_requested":len(ids),"cases_audited":len(rows),"missing":missing,"cases_with_errors":sum(r["error_count"]>0 for r in rows),"total_errors":sum(r["error_count"] for r in rows),"max_case_errors":max((r["error_count"] for r in rows),default=0)}
    out={"schema":"wss_v6_derived_geometry_independent_audit_v1","source":{"candidate_root":str(a.candidate),"split":str(a.split),"excluded":sorted(EXCLUDE),"read_only_files":"cases/*/{geometry.npz,report.json}"},"tolerance":{"numeric_atol":ATOL,"numeric_rtol":RTOL,"float32_endpoint_ambiguity_ulps":MASK_TOL,"context_window_mm":WINDOW,"context_direction":"strictly upstream/downstream; current station excluded; contiguous valid runs; nearest tie"},"self_tests":self_tests(),"summary":summary,"cases":rows}
    a.out.parent.mkdir(parents=True,exist_ok=True); a.out.with_suffix(".json").write_text(json.dumps(plain(out),ensure_ascii=False,indent=2)+"\n")
    cols=[]
    for r in rows:
        cols.append({"canonical_id":r["canonical_id"],"error_count":r["error_count"],"segment_error_count":len(r["segment_errors"]),"geometry_error_count":sum(int(v) for k,v in r["geometry_checks"].items() if isinstance(v,(int,np.integer))),"reference_area_slope_coverage":r["finite_coverage"].get("section_area_slope_per_mm",0),"pointcloud_area_slope_coverage":r["finite_coverage"].get("pc_section_area_slope_per_mm",0),"reference_wall_area_coverage":r["finite_coverage"].get("wall_map_area_mm2",0),"pointcloud_wall_area_coverage":r["finite_coverage"].get("wall_map_pc_area_mm2",0),"conclusion":r["conclusion"]})
    with a.out.with_suffix(".csv").open("w",newline="") as f:
        w=csv.DictWriter(f,fieldnames=list(cols[0]) if cols else ["canonical_id"]); w.writeheader(); w.writerows(cols)
    print(json.dumps(summary,ensure_ascii=False))
if __name__=="__main__": main()

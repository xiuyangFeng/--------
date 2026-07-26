#!/usr/bin/env python3
"""Analyze xyz-only LocalGeoPE against xyz and xyz+geom S2 controls."""
from __future__ import annotations
import csv, json
from pathlib import Path
from training_wss_min.tools.analyze_d2_k64_ilo_structure_results import AGG_KEYS, sha256
from training_wss_min.tools.analyze_s3_rootcause_results import build_record, comparison

ROOT=Path(__file__).resolve().parents[2]; PREFLIGHT=ROOT/"training_wss_min/preflight"; MANIFEST=PREFLIGHT/"xyz_local_geope_20260725_prepared.json"; OUT_JSON=PREFLIGHT/"xyz_local_geope_20260725_results_analysis.json"; OUT_SUMMARY=PREFLIGHT/"xyz_local_geope_20260725_results_summary.csv"; OUT_PAIRS=PREFLIGHT/"xyz_local_geope_20260725_paired_case_stats.csv"
XYZ=ROOT/"training_wss_min/runs/pointnetpp_xyz_input_ablation/outputs/s2_d2k64_pnxr_mixed_xyz"; XYZGEOM=ROOT/"training_wss_min/runs/pointnetpp_d2_k64_ilo_structure/outputs/s2_d2k64_pnxr_mixed"
def main()->None:
 m=json.loads(MANIFEST.read_text(encoding="utf-8"));row=m["configs"][0]
 xyz=build_record("s2_pnxr_xyz",XYZ,1234,"xyz_control"); geom=build_record("s2_pnxr_xyzgeom",XYZGEOM,1234,"xyzgeom_control"); geo=build_record(row["experiment_id"],Path(row["run_dir"]),1234,"xyz_local_geope",Path(row["config"]),row["sha256"])
 records={r["experiment_id"]:r for r in (xyz,geom,geo)}
 if any(r["integrity"]!="passed" for r in records.values()):raise RuntimeError("integrity failed")
 vs_xyz=comparison("xyz_local_geope_vs_xyz",xyz,geo,20260725);vs_geom=comparison("xyz_local_geope_vs_xyzgeom",geom,geo,20260825)
 p={"schema_version":1,"analysis_date":"2026-07-25","scope":"S2 PointNeXt-R mixed138/test36; xyz-only 4D relative-position LocalGeoPE","manifest":str(MANIFEST.resolve()),"manifest_sha256":sha256(MANIFEST),"records":records,"paired_comparisons":[vs_xyz,vs_geom],"limits":["single seed1234","historical test36 engineering screen"]}
 OUT_JSON.write_text(json.dumps(p,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
 with OUT_SUMMARY.open("w",newline="",encoding="utf-8") as h:
  f=["experiment_id","variant","best_epoch",*AGG_KEYS,"AG_r2","AAA_r2","ILO_r2","integrity"];w=csv.DictWriter(h,fieldnames=f);w.writeheader()
  for r in records.values():w.writerow({"experiment_id":r["experiment_id"],"variant":r["variant"],"best_epoch":r["history"].get("best_epoch"),**{k:r["best"][k] for k in AGG_KEYS},"AG_r2":r["domains"]["AG"],"AAA_r2":r["domains"]["AAA"],"ILO_r2":r["domains"]["ILO"],"integrity":r["integrity"]})
 with OUT_PAIRS.open("w",newline="",encoding="utf-8") as h:
  w=csv.DictWriter(h,fieldnames=["comparison","delta_r2cb","delta_mae","delta_rmse","delta_high_wss_r2","delta_top10_iou","case_r2_mean_delta","case_r2_ci95_low","case_r2_ci95_high"]);w.writeheader()
  for r in (vs_xyz,vs_geom):
   d,c=r["aggregate_treatment_minus_control"],r["paired_case_stats"]["case_r2"];w.writerow({"comparison":r["comparison"],"delta_r2cb":d["physical_r2_casebalanced"],"delta_mae":d["physical_mae"],"delta_rmse":d["physical_rmse"],"delta_high_wss_r2":d["high_wss_r2"],"delta_top10_iou":d["physical_top10_iou"],"case_r2_mean_delta":c["mean_delta"],"case_r2_ci95_low":c["ci95_low"],"case_r2_ci95_high":c["ci95_high"]})
 print(json.dumps({"status":"passed","json":str(OUT_JSON)},ensure_ascii=False))
if __name__=="__main__":main()

"""Independent structural audit of the coverage-recovery candidate."""
from __future__ import annotations
import argparse,json,hashlib
from collections import Counter
from pathlib import Path
import numpy as np

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main():
    ap=argparse.ArgumentParser();ap.add_argument('--candidate',type=Path,required=True);ap.add_argument('--source',type=Path,required=True);ap.add_argument('--split',type=Path,required=True);ap.add_argument('--code-root',type=Path);a=ap.parse_args()
    m=json.loads((a.candidate/'coverage_manifest.json').read_text());sp=json.loads(a.split.read_text());ids=set(sp['train_cases'])|set(sp['test_cases']);rows=[];errors=[]
    for cid in sorted(ids):
        key=cid.replace('/','__');cd=a.candidate/'cases'/key;sd=a.source/'cases'/key
        try:
            r=json.loads((cd/'report.json').read_text());cv=json.loads((cd/'coverage.json').read_text())
            with np.load(cd/'geometry.npz',allow_pickle=False) as q:z={k:q[k] for k in q.files}
            with np.load(sd/'geometry.npz',allow_pickle=False) as q:o={k:q[k] for k in q.files}
            n=len(z['section_valid']);err=[]
            for p in ('section_','pc_section_'):
                v=z[p+'valid'].astype(bool);a0=z[p+'area_mm2'].astype(float)
                if np.any(v & (~np.isfinite(a0)| (a0<=0))):err.append(p+'positive_area')
                if not np.array_equal(v,z[p+'reason']=='ok'):err.append(p+'reason_mask')
                off=z[p+'polygon_offsets'];poly=z[p+'polygon_xyz_mm']
                if off.shape!=(n+1,) or off[-1]!=len(poly) or not np.all(np.diff(off)>=0):err.append(p+'polygon_pack')
            if np.any(z['pc_section_valid'] & ~z['section_valid']):err.append('pc_implies_reference')
            for name in ('coverage_reference_status','coverage_pointcloud_status','coverage_recovery_candidate_mask','coverage_pre_shift_valid','coverage_pre_shift_area_mm2','coverage_pointcloud_method'):
                if name not in z:err.append('missing_'+name)
            if not np.array_equal(o['wall_node_id'],z['wall_node_id']) or not np.array_equal(o['wall_xyz_mm'],z['wall_xyz_mm']):err.append('frozen_wall_source')
            if cv['reference_after']!=int(z['section_valid'].sum()) or cv['pointcloud_after']!=int(z['pc_section_valid'].sum()):err.append('coverage_counts')
            rows.append({'canonical_id':cid,'mechanical_checks_pass':not err,'errors':err,'n_sections':n,'reference_before':cv['reference_before'],'reference_after':cv['reference_after'],'pointcloud_before':cv['pointcloud_before'],'pointcloud_after':cv['pointcloud_after'],'coverage_counts':cv['counts'],'pc_counts':cv['pc_counts']})
        except Exception as e:errors.append({'canonical_id':cid,'error':f'{type(e).__name__}: {e}'})
    summary={'expected_cases':len(ids),'audited_cases':len(rows),'cases_with_errors':sum(not r['mechanical_checks_pass'] for r in rows)+len(errors),'structural_errors':sum(not r['mechanical_checks_pass'] for r in rows)+len(errors),'numeric_errors':0,'all_mechanical_checks_pass':not errors and all(r['mechanical_checks_pass'] for r in rows) and len(rows)==len(ids),'recovered_reference':sum(r['coverage_counts'].get('recovered_missing_reference',0) for r in rows),'recovered_pointcloud':sum(r['pc_counts'].get('recovered_missing_pointcloud',0) for r in rows)}
    result={'schema':'wss_v6_geometry_coverage_independent_audit_v1','source':str(a.source.resolve()),'candidate':str(a.candidate.resolve()),'code_root':str((a.code_root or Path(__file__).parent).resolve()),'summary':summary,'errors':errors,'cases':rows}
    (a.candidate/'audit_coverage_geometry.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n');print(json.dumps(summary,ensure_ascii=False));
    if not summary['all_mechanical_checks_pass']:raise SystemExit(1)
if __name__=='__main__':main()

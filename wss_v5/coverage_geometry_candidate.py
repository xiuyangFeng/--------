"""Recover missing sections with exact branch intersections and explicit masks.

This second stage reads an independently validated v3 candidate. It never
substitutes radius-derived areas or copies a surface polygon into pointcloud
features. Existing validated geometry is retained unless the exact branch
crossing or final continuous-neighbor checks contradict its validity.
"""
from __future__ import annotations
import argparse
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
import copy
import json
import os
from pathlib import Path
import time
import numpy as np
from scipy.spatial import cKDTree

from .geometry_candidate import section_geometry_gate, packed_polygons, write_json
from .refine_geometry_candidate import (atlas_center, normals, remap_fields, sha,
                                       verify_final_shifts)
from .section_features import (SurfaceSlicer, frame, contains_origin,
    polyline_plane_intersection_groups, _line_intersects_polygon, pointcloud_section)

SCHEMA='wss_v6_geometry_coverage_v1'
POLICY={'branch_gate':'continuous_centerline_edge_plane_intersections',
        'search_angles_deg':[0.,5.,10.,15.,20.], 'tilt_deg':5.,'tilt_limit':.20,
        'holdout_azimuth_offset_deg':45.,'source_relative_limit':.20,
        'shift_mm':1.,'shift_residual_limit':.25,
        'final_shift_center':'original_atlas_arclength_interpolation',
        'final_shift_profile':'simultaneous_monotone_masks_until_fixed_point',
        'recovery_minimum_final_shift_probes':1,
        'pointcloud_requires_verified_reference':True,
        'missing_value_policy':'per_field_validity_not_case_level_disable',
        'pointcloud_fallback':'crust_contour_from_points_only',
        'reference_recovery_does_not_require_pointcloud_available':True}


def exact_other_branch_hits(result, center, normal, xyz, sid, atlas_s, segment_id):
    """Intersect continuous edges; near-plane samples alone are not crossings."""
    if not result.get('valid'):return []
    e1,e2=frame(normal);poly=np.asarray(result['polygon_xyz_mm'])-center
    p2=np.column_stack((poly@e1,poly@e2));hits=[]
    for seg in np.unique(sid):
        if seg==segment_id:continue
        ii=np.flatnonzero(sid==seg);ii=ii[np.argsort(atlas_s[ii])]
        for g in polyline_plane_intersection_groups(xyz[ii],atlas_s[ii],center,normal):
            points=np.asarray(g['points_xyz_mm'])-center
            q=np.column_stack((points@e1,points@e2))
            inside=any(contains_origin(p2-p) for p in q)
            if not inside and len(q)>1:
                inside=any(_line_intersects_polygon(a,b,p2) for a,b in zip(q[:-1],q[1:]))
            if inside:hits.append({'segment_id':int(seg),**g})
    return hits


def exact_gate(raw,center,normal,seg,s,xyz,sid,cfg,atlas_s):
    # Existing gate applies unchanged endpoint, graph-junction and same-branch
    # nonlocal checks. Only its +/-0.6 mm sampled-other-branch heuristic is
    # replaced, not merely removed or widened.
    mine=sid==seg['segment_id']
    result=section_geometry_gate(raw,center,normal,seg,s,xyz[mine],sid[mine],cfg,atlas_s[mine])
    hits=exact_other_branch_hits(raw,center,normal,xyz,sid,atlas_s,seg['segment_id'])
    result['other_branch_plane_crossings']=hits
    if hits:
        reason='section_contains_other_centerline_branch'
        reasons=set(result['reason'].split(';'))-{'ok'};reasons.add(reason)
        result.update(valid=False,pipeline_valid=False,reason=';'.join(sorted(reasons)),pipeline_reason=';'.join(sorted(reasons)))
    return result


def load_rows(z,p):
    names=['area_mm2','req_mm','perimeter_mm','roundness','eccentricity','centroid_mm',
           'valid','reason','raw_contour_valid','raw_contour_reason',
           'same_segment_crossing_group_count','same_segment_nonlocal_crossing_count']
    rows=[];off=z[p+'polygon_offsets']
    for k in range(len(z[p+'valid'])):
        row={name:z[p+name][k].item() if z[p+name][k].ndim==0 else z[p+name][k].copy() for name in names}
        row['polygon_xyz_mm']=z[p+'polygon_xyz_mm'][off[k]:off[k+1]].copy()
        row['pipeline_valid']=row['valid'];row['pipeline_reason']=row['reason'];rows.append(row)
    return rows


def pack_rows(z,p,rows):
    for name in ['area_mm2','req_mm','perimeter_mm','roundness','eccentricity','centroid_mm']:
        default=np.full(3,np.nan) if name=='centroid_mm' else np.nan
        z[p+name]=np.asarray([r.get(name,default) for r in rows],np.float32)
    for name in ['valid','raw_contour_valid']:z[p+name]=np.asarray([r[name] for r in rows],bool)
    for name in ['reason','raw_contour_reason']:z[p+name]=np.asarray([r[name] for r in rows])
    for name in ['same_segment_crossing_group_count','same_segment_nonlocal_crossing_count']:
        z[p+name]=np.asarray([r.get(name,0) for r in rows],np.int16)
    z[p+'polygon_xyz_mm'],z[p+'polygon_offsets']=packed_polygons(rows,'polygon_xyz_mm')


def reject(row,reason):
    row.update(valid=False,pipeline_valid=False,reason=reason,pipeline_reason=reason)


def recover_case(task):
    source,output,enable_fallback=task;started=time.monotonic()
    r=json.loads((source/'report.json').read_text());previous=copy.deepcopy(r)
    with np.load(source/'geometry.npz',allow_pickle=False) as f:z={k:f[k] for k in f.files}
    wall=z['wall_xyz_mm'].astype(float);slicer=SurfaceSlicer(wall,z['wall_triangles']);tree=cKDTree(wall)
    xyz=z['centerline_xyz_mm'].astype(float);sid=z['centerline_segment_id'];atlas_s=z['centerline_s_local_mm'].astype(float)
    ss=z['section_s_local_mm'].astype(float);segid=z['section_segment_id'];centers=z['section_center_mm'].astype(float)
    tangents=z['section_tangent'].astype(float);radii=z['section_radius_mis_mm'].astype(float)
    base_valid=z['section_valid'].copy();base_pc_valid=z['pc_section_valid'].copy()
    ref=load_rows(z,'section_');pc=load_rows(z,'pc_section_');segments={s['segment_id']:copy.deepcopy(s) for s in r['segments']}
    for seg,bounds in r['sampling_domain']['segment_s_bounds_mm'].items():
        segments[int(seg)]['start_s_mm'],segments[int(seg)]['end_s_mm']=bounds
    import h5py
    with h5py.File(r['source']['case_h5'],'r') as h:spacing=float(h['geometry'].attrs['wall_point_spacing_mm'])
    cfg=r['config'];half=np.random.default_rng(1234).random(len(wall))<.5
    def gate(raw,k,t,c=None,s=None):
        return exact_gate(raw,centers[k] if c is None else c,t,segments[int(segid[k])],ss[k] if s is None else s,xyz,sid,cfg,atlas_s)
    def reference(k,t,c=None,s=None):
        return gate(slicer.slice(centers[k] if c is None else c,t),k,t,c,s)
    def tilted(k,t,area,offset=0):
        probes=[]
        for j,nt in enumerate(normals(t,5.,offset_deg=offset)):
            p=reference(k,nt);error=abs(p.get('area_mm2',np.nan)/area-1) if p['valid'] else None
            probes.append({'normal':nt.tolist(),'azimuth_deg':offset+j*90.,'area_mm2':p.get('area_mm2'),
                           'valid':p['valid'],'reason':p['reason'],'relative_change':error})
        return {'baseline_area_mm2':float(area),'probes':probes,
                'stable':all(p['valid'] and p['relative_change']<=.20 for p in probes)}
    records=[];new=np.zeros(len(ss),bool);changed=np.zeros(len(ss),bool)
    status=np.where(base_valid,'retained_valid','previously_invalid').astype('<U64')
    for k in range(len(ss)):
        t=tangents[k].copy();raw=slicer.slice(centers[k],t);regate=gate(raw,k,t)
        if base_valid[k]:
            if not regate['valid']:
                reject(ref[k],regate['reason']);status[k]='withheld_exact_branch_gate'
                records.append({'section_index':k,'status':str(status[k]),'gate':regate['reason'],
                                'other_branch_plane_crossings':regate['other_branch_plane_crossings']})
            continue
        seg=segments[int(segid[k])]
        if ss[k]-seg['start_s_mm']<=3.5 or seg['end_s_mm']-ss[k]<=3.5:continue
        selected=None;attempts=0
        for angle in POLICY['search_angles_deg']:
            choices=[t] if angle==0 else normals(t,angle,8)
            for nt in choices:
                attempts+=1;cand=regate if angle==0 else reference(k,nt)
                if not cand['valid']:continue
                test=tilted(k,nt,cand['area_mm2'])
                if not test['stable']:continue
                selected=(cand,nt,test,angle);break
            if selected is not None:break
        note={'section_index':k,'segment_id':int(segid[k]),'s_mm':float(ss[k]),'original_reason':ref[k]['reason'],
              'search_attempts':attempts,'status':'not_recovered'}
        if selected is not None:
            cand,nt,test,angle=selected;holdout=tilted(k,nt,cand['area_mm2'],45.)
            note.update(selection_tilt=test,holdout_tilt=holdout,angle_deg=angle,normal=nt.tolist(),candidate_area_mm2=cand['area_mm2'])
            if holdout['stable']:
                ref[k]=cand;tangents[k]=nt;new[k]=True;changed[k]=angle>0;status[k]='recovered_missing_reference'
                note['status']=str(status[k])
        records.append(note)
    candidate_valid=np.array([x['valid'] for x in ref],bool);candidate_area=np.array([x.get('area_mm2',np.nan) for x in ref],float)
    def shifted(k,sq):
        c=atlas_center(sq,segid[k],xyz,sid,atlas_s);p=reference(k,tangents[k],c,sq)
        return {'center_mm':c.tolist(),'normal':tangents[k].tolist(),'actual_area_mm2':p.get('area_mm2'),
                'valid':p['valid'],'reason':p['reason']}
    final_valid,shift=verify_final_shifts(ss,segid,candidate_area,candidate_valid,new,shifted)
    for k in np.flatnonzero(candidate_valid&~final_valid):
        reject(ref[k],'coverage_final_shift_failed');status[k]='withheld_final_shift'
    z['coverage_reference_status']=status;z['coverage_recovery_candidate_mask']=new
    z['coverage_pre_shift_valid']=candidate_valid;z['coverage_pre_shift_area_mm2']=candidate_area
    pc_status=np.where(base_pc_valid,'retained_valid','previously_invalid').astype('<U64');pc_records=[]
    cloud_method=np.full(len(ss),'radial_v1',dtype='<U64')
    if enable_fallback:
        from .pointcloud_contours import pointcloud_contour_fallback
    for k in range(len(ss)):
        if not final_valid[k]:
            if pc[k]['valid']:reject(pc[k],'coverage_reference_unverified');pc_status[k]='withheld_reference'
            continue
        if base_pc_valid[k] and not changed[k]:
            # Recheck ownership under the same continuous crossing rule.
            regated=gate(pc[k],k,tangents[k])
            if regated['valid']:continue
        ix=np.asarray(tree.query_ball_point(centers[k],max(4*radii[k],12.)),int)
        ix=ix[z['wall_map_segment_id'][ix]==segid[k]]
        trials=[('radial_v1',pointcloud_section)]
        if enable_fallback:trials.append(('crust_contour_v1',pointcloud_contour_fallback))
        note={'section_index':k,'segment_id':int(segid[k]),'s_mm':float(ss[k]),'attempts':[],'status':'not_recovered'}
        accepted=False
        for method,fn in trials:
            raw=fn(wall[ix],centers[k],tangents[k],radii[k],spacing);p=gate(raw,k,tangents[k])
            err=abs(p.get('area_mm2',np.nan)/ref[k]['area_mm2']-1) if p['valid'] else None
            a={'method':method,'valid':p['valid'],'reason':p['reason'],'area_mm2':p.get('area_mm2'),
               'reference_relative_error':err,'construction':{key:value for key,value in raw.items()
               if key not in {'polygon_xyz_mm','centroid_mm'} and key not in
               {'area_mm2','perimeter_mm','req_mm','roundness','eccentricity','valid','reason','contains_center'}}}
            if p['valid'] and err<=.20:
                half_raw=fn(wall[ix[half[ix]]],centers[k],tangents[k],radii[k],spacing)
                ph=gate(half_raw,k,tangents[k]);error=abs(ph.get('area_mm2',np.nan)/p['area_mm2']-1) if ph['valid'] else None
                a.update(half_valid=ph['valid'],half_reason=ph['reason'],half_area_mm2=ph.get('area_mm2'),density_relative_error=error)
                a['half_construction']={key:value for key,value in half_raw.items()
                    if key not in {'polygon_xyz_mm','centroid_mm','area_mm2','perimeter_mm','req_mm',
                                   'roundness','eccentricity','valid','reason','contains_center'}}
                if ph['valid'] and error<=.20:
                    pc[k]=p;pc_status[k]='recovered_missing_pointcloud';cloud_method[k]=method;accepted=True
            note['attempts'].append(a)
            if accepted:note.update(status=str(pc_status[k]),method=method);break
        if not accepted and pc[k]['valid']:reject(pc[k],'coverage_pointcloud_unverified');pc_status[k]='withheld_verification'
        pc_records.append(note)
    z['coverage_pointcloud_status']=pc_status;z['coverage_pointcloud_method']=cloud_method
    for p,rows in [('section_',ref),('pc_section_',pc)]:pack_rows(z,p,rows)
    z['section_tangent']=tangents.astype(np.float32)
    edge_parts=[];offsets=[0];oldoff=z['section_intersection_offsets'];oldedges=z['section_intersection_segments_xyz_mm']
    for k in range(len(ss)):
        edges=slicer.slice(centers[k],tangents[k])['intersection_segments_xyz_mm'] if changed[k] else oldedges[oldoff[k]:oldoff[k+1]]
        edge_parts.extend(edges);offsets.append(len(edge_parts))
    z['section_intersection_offsets']=np.array(offsets,np.int64);z['section_intersection_segments_xyz_mm']=np.asarray(edge_parts,np.float32).reshape(-1,2,3)
    remap_fields(z)
    summary={'canonical_id':r['canonical_id'],'n_sections':len(ss),'counts':dict(Counter(status)),
             'pc_counts':dict(Counter(pc_status)),'reference_before':int(base_valid.sum()),'reference_after':int(final_valid.sum()),
             'pointcloud_before':int(base_pc_valid.sum()),'pointcloud_after':int(z['pc_section_valid'].sum()),
             'recovered_reference':int((~base_valid&final_valid).sum()),'withheld_reference':int((base_valid&~final_valid).sum()),
             'recovered_pointcloud':int((~base_pc_valid&z['pc_section_valid']).sum()),
             'reference_wall_coverage':float(z['wall_map_valid'].mean()),'pointcloud_wall_coverage':float(z['wall_map_pc_valid'].mean()),
             'seconds':time.monotonic()-started}
    for key,prefix in [('reference','section_'),('pointcloud','pc_section_')]:
        r['P4_sections'][key+'_valid']=int(z[prefix+'valid'].sum());r['P4_sections'][key+'_valid_fraction']=float(z[prefix+'valid'].mean())
        r['P4_sections'][key+'_reasons']=dict(Counter(z[prefix+'reason']))
    r['P4_sections']['paired_valid']=int((z['section_valid']&z['pc_section_valid']).sum())
    paired=z['section_valid']&z['pc_section_valid']
    error=np.abs(z['pc_section_area_mm2'][paired]/z['section_area_mm2'][paired]-1)
    r['P4_sections']['pointcloud_reference_relative_area_median_abs']=float(np.median(error)) if len(error) else None
    r['P4_sections']['pointcloud_reference_relative_area_p90_abs']=float(np.percentile(error,90)) if len(error) else None
    r['P5_wall_mapping'].update(reference_valid_fraction=summary['reference_wall_coverage'],pointcloud_valid_fraction=summary['pointcloud_wall_coverage'])
    r['schema']=SCHEMA;r['training_allowed']=False;r['coverage']={**summary,'policy':POLICY}
    r['coverage_source']={'directory':str(source.resolve()),'geometry_sha256':sha(source/'geometry.npz'),'report_sha256':sha(source/'report.json')}
    r['coverage_code']={name:sha(Path(__file__).with_name(name)) for name in ['coverage_geometry_candidate.py','refine_geometry_candidate.py','geometry_candidate.py','section_features.py']+(['pointcloud_contours.py'] if enable_fallback else [])}
    r['legacy_v3_summary']={'refinement':r.pop('refinement',{}),'P6_stability':r.pop('P6_stability',{})}
    r['P6_stability']={'scope':'v3_validated_baseline_plus_missing_section_recovery_final_shift_and_cloud_density','report':'coverage.json'}
    r['status']='coverage_candidate_with_per_field_validity';r['case_level_feature_disable']=False
    dest=output/'cases'/source.name;dest.mkdir(parents=True,exist_ok=True)
    np.savez_compressed(dest/'geometry.npz',**z);r['geometry_npz']=str(dest/'geometry.npz')
    write_json(dest/'report.json',r)
    write_json(dest/'coverage.json',{'schema':SCHEMA,'policy':POLICY,**summary,'reference_attempts':records,
                                  'pointcloud_attempts':pc_records,'final_verification':shift})
    print(json.dumps(summary),flush=True);return summary


def main():
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--source',type=Path,required=True);ap.add_argument('--output',type=Path,required=True)
    ap.add_argument('--workers',type=int,default=8);ap.add_argument('--cases',nargs='+');ap.add_argument('--no-pointcloud-fallback',action='store_true')
    a=ap.parse_args()
    if not os.environ.get('SLURM_JOB_ID'):ap.error('patient reconstruction requires Slurm CPU allocation')
    if a.source.resolve()==a.output.resolve():ap.error('output must differ from source')
    m=json.loads((a.source/'refinement_manifest.json').read_text());ids={x['canonical_id'] for x in m['cases']}
    if a.cases:
        if not set(a.cases)<=ids:ap.error('case outside source cohort')
        ids=set(a.cases)
    tasks=[(a.source/'cases'/cid.replace('/','__'),a.output,not a.no_pointcloud_fallback) for cid in sorted(ids)]
    with ProcessPoolExecutor(max_workers=a.workers) as pool:results=list(pool.map(recover_case,tasks))
    write_json(a.output/'coverage_manifest.json',{'schema':SCHEMA,'policy':POLICY,'source':str(a.source.resolve()),'cases':results,
               'job_id':os.environ['SLURM_JOB_ID'],'source_manifest_sha256':sha(a.source/'refinement_manifest.json'),'training_allowed':False})


if __name__=='__main__':main()

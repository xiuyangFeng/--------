"""Reconstruct and automatically contain unstable longitudinal geometry.

Reads a frozen candidate, writes a separate review candidate. A rejected
section stays missing: no smoothing or interpolation is used to invent an
area. Alternative planes pass the original gates and four independent tilt
probes before recovery. Surface-guided plane selection is explicitly not a
validated bare-cloud deployment algorithm.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
from collections import Counter
import copy
import hashlib
import json
import os
from pathlib import Path
import time

import numpy as np
from scipy.spatial import cKDTree

from .geometry_candidate import (branch_context, interpolate_valid_field, packed_polygons,
                                 section_geometry_gate, write_json)
from .section_features import (SurfaceSlicer, frame, log_area_slopes, pointcloud_section,
                               opening_overrun_domain, unit)
from .anatomy_overrides import correct_report

SCHEMA = "wss_v6_geometry_refined_v3"
DEFAULT_SPLIT = Path(__file__).resolve().parents[1] / "wss_pinn/configs/splits/split_WSS_PINN_AG_AAA_ILO_bc_rcr_v4_anatomy_only_train136_test34_s1234.json"
POLICY = {"tilt_probe_deg": 5., "tilt_relative_limit": .20,
          "source_relative_limit": .20, "search_angles_deg": [5., 10., 15., 20.],
          "shift_mm": 1., "shift_residual_limit": .25,
          "maximum_recovery_neighbor_ratio": 1.5,
          "holdout_azimuth_offset_deg": 45.,
          "final_shift_center": "original_atlas_arclength_interpolation",
          "final_shift_profile": "simultaneous_monotone_masks_until_fixed_point",
          "recovery_minimum_final_shift_probes": 1,
          "pointcloud_requires_verified_reference": True,
          "policy_scope": "candidate_reconstruction_quality_not_clinical_anatomy_or_training_approval"}

def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def normals(t, angle, azimuths=4, offset_deg=0.):
    e1, e2 = frame(t)
    a = np.radians(angle)
    return [unit(np.cos(a)*t + np.sin(a)*(np.cos(p)*e1+np.sin(p)*e2))
            for p in np.arange(azimuths)*2*np.pi/azimuths+np.radians(offset_deg)]

def atlas_center(query, segment, xyz, sid, atlas_s):
    ix = np.flatnonzero(sid == segment)
    return np.array([np.interp(query, atlas_s[ix], xyz[ix, axis]) for axis in range(3)])

def verify_final_shifts(ss, segid, area, valid, recovered, probe):
    """Simultaneous rejection with final continuous brackets, no gap bridging.

    Probe geometry is cached; predictions are recomputed after each masking
    round. Original stable stations without brackets remain unassessed for
    shift. A searched replacement needs at least one surviving bracket.
    """
    valid = valid.copy()
    records = {}; cache = {}; rounds = 0
    while True:
        rounds += 1; rejected = []
        for k in np.flatnonzero(valid):
            shift = []
            for sign in [-1, 1]:
                sq = ss[k]+sign*POLICY['shift_mm']
                pred = contiguous_prediction(k, sq, ss, segid, area, valid)
                if pred is None: continue
                key = (int(k), sign)
                if key not in cache: cache[key] = probe(int(k), sq)
                p = cache[key]
                residual = abs(p['actual_area_mm2']/pred-1) if p['valid'] else None
                shift.append({**p, 'delta_mm':sign*POLICY['shift_mm'], 'query_s_mm':float(sq),
                              'predicted_area_mm2':pred, 'relative_residual':residual,
                              'neighbor_index':int(k+sign), 'neighbor_area_mm2':float(area[k+sign])})
            failed = any(not p['valid'] or p['relative_residual']>POLICY['shift_residual_limit'] for p in shift)
            insufficient = recovered[k] and len(shift)<POLICY['recovery_minimum_final_shift_probes']
            stable = not failed and not insufficient
            records[int(k)] = {'section_index':int(k), 'segment_id':int(segid[k]), 's_mm':float(ss[k]),
                              'was_recovered':bool(recovered[k]), 'iteration':rounds, 'shift':shift,
                              'stable':stable, 'reason':'final_shift_failed' if failed else
                              'recovery_missing_contiguous_shift_evidence' if insufficient else
                              'verified' if shift else 'no_contiguous_shift_bracket'}
            if not stable: rejected.append(k)
        if not rejected: break
        valid[rejected] = False
    return valid, {'rounds':rounds, 'sections':[records[k] for k in sorted(records)]}

def contiguous_prediction(i, query, ss, sid, area, valid):
    """Expected log-area on the immediate valid bracket; never bridge a gap."""
    j = i + (1 if query > ss[i] else -1)
    if not 0 <= j < len(ss) or sid[j] != sid[i] or not (valid[i] and valid[j]):
        return None
    if not min(ss[i], ss[j]) <= query <= max(ss[i], ss[j]):
        return None
    t = (query-ss[i])/(ss[j]-ss[i])
    return float(np.exp((1-t)*np.log(area[i])+t*np.log(area[j])))

def remap_fields(z):
    """Recompute every area-derived descriptor and its mask after recovery."""
    sid, ss = z['section_segment_id'], z['section_s_local_mm'].astype(float)
    for p, w in [('section_', 'wall_map_'), ('pc_section_', 'wall_map_pc_')]:
        a, v = z[p+'area_mm2'].astype(float), z[p+'valid']
        slope = np.full(len(ss), np.nan)
        ctx = {k: np.full(len(ss), np.nan) for k in
               ['upstream_min_area_ratio','downstream_min_area_ratio',
                'upstream_min_distance_mm','downstream_min_distance_mm']}
        for seg in np.unique(sid):
            ix = np.flatnonzero(sid == seg)
            slope[ix] = log_area_slopes(ss[ix], a[ix], v[ix])
            c = branch_context(ss[ix], a[ix], v[ix], 10.)
            for k in ctx: ctx[k][ix] = c[k]
        z[p+'area_slope_per_mm'] = slope.astype(np.float32)
        for k, x in ctx.items(): z[p+k] = x.astype(np.float32)
        for name in ['area_mm2','area_slope_per_mm','roundness','eccentricity',*ctx]:
            values = z[p+name].astype(float)
            val = np.full(len(z['wall_xyz_mm']), np.nan)
            good = np.zeros(len(val), bool)
            for seg in np.unique(sid):
                ii = np.flatnonzero(sid == seg)
                wi = np.flatnonzero(z['wall_map_segment_id'] == seg)
                val[wi], good[wi] = interpolate_valid_field(z['wall_map_s_local_mm'][wi].astype(float),
                                                            ss[ii], values[ii], v[ii] & np.isfinite(values[ii]))
            if name == 'area_mm2': z[w+'raw_area_mm2'] = val.astype(np.float32)
            good &= ~z['wall_map_ambiguous']
            val[~good] = np.nan
            z[w+name] = val.astype(np.float32)
            z[w+('valid' if name == 'area_mm2' else name+'_valid')] = good

def refine_case(args):
    case_dir, output, full = args
    started = time.monotonic()
    original = json.loads((case_dir/'report.json').read_text())
    r = correct_report(copy.deepcopy(original))
    with np.load(case_dir/'geometry.npz', allow_pickle=False) as archive:
        z = {k: archive[k] for k in archive.files}
    wall = z['wall_xyz_mm'].astype(float)
    slicer = SurfaceSlicer(wall, z['wall_triangles'])
    tree = cKDTree(wall)
    xyz = z['centerline_xyz_mm'].astype(float)
    sid = z['centerline_segment_id']
    atlas_s = z['centerline_s_local_mm'].astype(float)
    ss, segid = z['section_s_local_mm'].astype(float), z['section_segment_id']
    centers, tangents = z['section_center_mm'].astype(float), z['section_tangent'].astype(float)
    radii = z['section_radius_mis_mm'].astype(float)
    old_a, old_v = z['section_area_mm2'].copy(), z['section_valid'].copy()
    old_pc_a, old_pc_v = z['pc_section_area_mm2'].copy(), z['pc_section_valid'].copy()
    cfg = r['config']; segments = {s['segment_id']: s for s in r['segments']}
    cfg = {**cfg, 'context_window_mm': 10.}
    inside,enclosed=slicer.enclosed(xyz)
    external_tail,domain=opening_overrun_domain(xyz,sid,atlas_s,r['segments'],r['openings'],inside=inside)
    original_core=z['centerline_audit_core_mask'].copy()
    z['centerline_original_audit_core_mask']=original_core
    z['centerline_opening_extension_mask']=external_tail
    z['centerline_audit_core_mask']=original_core & ~external_tail
    for seg_id,bounds in domain['segment_s_bounds_mm'].items():
        segments[seg_id]['start_s_mm'],segments[seg_id]['end_s_mm']=bounds
    # Segment physical lengths/CIA descriptors still belong to the frozen
    # graph. Only the anatomical sampling limits above are clipped.
    r['sampling_domain']=domain
    # V5 stores spacing in source geometry attributes; pointcloud uses the
    # same original per-case spacing instead of fitting it to outcomes.
    import h5py
    with h5py.File(r['source']['case_h5'], 'r') as h5:
        spacing = float(h5['geometry'].attrs['wall_point_spacing_mm'])
    gate = lambda raw,c,t,k,s: section_geometry_gate(raw,c,t,segments[int(segid[k])],s,
                                                   xyz,sid,cfg,atlas_s)
    def reference(k, t, c=None, s=None):
        c = centers[k] if c is None else c
        return gate(slicer.slice(c,t),c,t,k,ss[k] if s is None else s)
    def cloud(k,t, subset=None):
        ix = np.asarray(tree.query_ball_point(centers[k],max(4*radii[k],12.)),int)
        ix = ix[z['wall_map_segment_id'][ix] == segid[k]]
        if subset is not None: ix = ix[subset[ix]]
        raw = pointcloud_section(wall[ix],centers[k],t,radii[k],spacing)
        return gate(raw,centers[k],t,k,ss[k])
    def tilt_check(k,t,base,offset_deg=0.):
        probe_normals = normals(t,POLICY['tilt_probe_deg'],offset_deg=offset_deg)
        probes = [reference(k,nt) for nt in probe_normals]
        valid = [p['pipeline_valid'] for p in probes]
        ratios = [abs(p['area_mm2']/base['area_mm2']-1) for p in probes if p['pipeline_valid']]
        result = {'all_valid':all(valid), 'n_valid':sum(valid),
                'max_relative_change':max(ratios) if ratios else None,
                'reasons':dict(Counter(p['pipeline_reason'] for p in probes if not p['pipeline_valid'])),
                'stable':all(valid) and max(ratios,default=np.inf)<=POLICY['tilt_relative_limit']}
        if offset_deg:
            result['baseline_area_mm2']=float(base['area_mm2'])
            result['probes'] = [{'azimuth_deg':offset_deg+j*90., 'normal':nt.tolist(),
                                 'tilt_deg':POLICY['tilt_probe_deg'],
                                 'center_mm':centers[k].tolist(), 'valid':p['pipeline_valid'],
                                 'reason':p['pipeline_reason'], 'area_mm2':p.get('area_mm2'),
                                 'relative_change':abs(p['area_mm2']/base['area_mm2']-1) if p['pipeline_valid'] else None}
                                for j,(nt,p) in enumerate(zip(probe_normals,probes))]
        return result
    def load_row(k,p):
        row = {name: z[p+name][k].item() if z[p+name][k].ndim == 0 else z[p+name][k].copy()
               for name in ['area_mm2','req_mm','perimeter_mm','roundness','eccentricity','centroid_mm',
                            'valid','reason','raw_contour_valid','raw_contour_reason',
                            'same_segment_crossing_group_count','same_segment_nonlocal_crossing_count']}
        off=z[p+'polygon_offsets']; row['polygon_xyz_mm']=z[p+'polygon_xyz_mm'][off[k]:off[k+1]]
        row['pipeline_valid']=row['valid'];row['pipeline_reason']=row['reason']
        return row
    ref_rows = [load_row(k,'section_') for k in range(len(ss))]
    pc_rows = [load_row(k,'pc_section_') for k in range(len(ss))]
    diagnostics=[]
    rng=np.random.default_rng(1234); half=rng.random(len(wall))<.5
    statuses=np.full(len(ss),'previously_invalid',dtype='<U48')
    tilt_error=np.full(len(ss),np.nan); angle_change=np.zeros(len(ss))
    # Full mode tests every valid plane, not only the historical P6 sample.
    for k in range(len(ss)):
        seg=segments[int(segid[k])]
        if not old_v[k]: continue
        baseline=ref_rows[k]; t=tangents[k]
        regated=gate(baseline,centers[k],t,k,ss[k])
        if not regated['pipeline_valid']:
            statuses[k]='automatically_withheld_opening_extension'
            for rows in [ref_rows,pc_rows]:
                rows[k]=dict(rows[k],valid=False,pipeline_valid=False,
                    reason=regated['reason'],pipeline_reason=regated['reason'])
            diagnostics.append({'section_index':k,'segment_id':int(segid[k]),'s_mm':ss[k],
                'original_area_mm2':float(old_a[k]),'status':str(statuses[k]),
                'new_area_mm2':None,'domain_reason':regated['reason']})
            continue
        probe=tilt_check(k,t,baseline)
        tilt_error[k]=probe['max_relative_change'] if probe['max_relative_change'] is not None else np.nan
        disagreement=bool(old_pc_v[k] and abs(old_pc_a[k]/old_a[k]-1)>.20)
        shift_bad=False; shift=[]
        for sign in [-1,1]:
            sq=ss[k]+sign*POLICY['shift_mm']; pred=contiguous_prediction(k,sq,ss,segid,old_a,old_v)
            if pred is None: continue
            p=reference(k,t,centers[k]+sign*POLICY['shift_mm']*t,sq)
            residual=abs(p['area_mm2']/pred-1) if p['pipeline_valid'] else None
            # The original gate already contains invalid shifted planes.
            shift_bad |= bool(residual is not None and residual>POLICY['shift_residual_limit'])
            shift.append({'delta_mm':sign,'valid':p['pipeline_valid'],'reason':p['pipeline_reason'],
                          'predicted_area_mm2':pred,'actual_area_mm2':p.get('area_mm2'),'relative_residual':residual})
        unstable=not probe['stable'] or disagreement or shift_bad
        if not unstable:
            statuses[k]='stable_original'
            continue
        note={'section_index':k,'segment_id':int(segid[k]),'s_mm':ss[k],
              'original_area_mm2':float(old_a[k]),'original_pointcloud_area_mm2':float(old_pc_a[k]),
              'original_tilt':probe,'original_shift':shift,'source_disagreement':disagreement}
        recovered=None; attempts=0
        # A geometrically unstable slice near an actual opening/junction is
        # automatically withheld; it is not rescued by cutting outside it.
        distance=min(ss[k]-seg['start_s_mm'],seg['end_s_mm']-ss[k])
        if distance > 3.5:
            for angle in POLICY['search_angles_deg']:
                candidates=[]
                for nt in normals(t,angle,8):
                    attempts+=1; cand=reference(k,nt)
                    if not cand['pipeline_valid']: continue
                    pc=cloud(k,nt)
                    if not pc['pipeline_valid'] or abs(pc['area_mm2']/cand['area_mm2']-1)>.20: continue
                    # Prefer local continuity, but never change a stable
                    # baseline merely because a genuine stenosis is sharp.
                    neighbors=[j for j in [k-1,k+1] if 0<=j<len(ss) and segid[j]==segid[k] and old_v[j]]
                    if neighbors:
                        lo=min(old_a[j] for j in neighbors)/POLICY['maximum_recovery_neighbor_ratio']
                        hi=max(old_a[j] for j in neighbors)*POLICY['maximum_recovery_neighbor_ratio']
                        if not lo<=cand['area_mm2']<=hi: continue
                    test=tilt_check(k,nt,cand)
                    if not test['stable']: continue
                    ph=cloud(k,nt,half)
                    if not ph['pipeline_valid'] or abs(ph['area_mm2']/pc['area_mm2']-1)>.20: continue
                    candidates.append((test['max_relative_change'],cand,pc,nt,test,angle))
                if candidates:
                    recovered=min(candidates,key=lambda v:v[0]);break
        note['search_attempts']=attempts
        if recovered is not None:
            _,rr,pc,nt,test,angle=recovered
            ref_rows[k],pc_rows[k]=rr,pc
            tangents[k]=nt;angle_change[k]=angle;tilt_error[k]=test['max_relative_change']
            statuses[k]='recovered_stable_plane'
            note.update(status=str(statuses[k]),new_area_mm2=rr['area_mm2'],new_pointcloud_area_mm2=pc['area_mm2'],
                        normal_change_deg=angle,new_tilt=test)
        else:
            statuses[k]='automatically_withheld_unstable'
            for rows in [ref_rows,pc_rows]:
                rows[k]=dict(rows[k],valid=False,pipeline_valid=False,
                    reason='refinement_unstable_plane',pipeline_reason='refinement_unstable_plane')
            note.update(status=str(statuses[k]),new_area_mm2=None)
        diagnostics.append(note)
    # Recheck all adjacent final planes. No replacement may introduce a new
    # isolated area spike; recovered points participating in one are withheld.
    for k in np.flatnonzero(statuses=='recovered_stable_plane'):
        tests=[]
        for j in [k-1,k+1]:
            if 0<=j<len(ss) and segid[j]==segid[k] and ref_rows[j]['valid']:
                tests.append(max(ref_rows[j]['area_mm2']/ref_rows[k]['area_mm2'],ref_rows[k]['area_mm2']/ref_rows[j]['area_mm2']))
        if tests and min(tests)>POLICY['maximum_recovery_neighbor_ratio']:
            statuses[k]='automatically_withheld_isolated_recovery'
            for rows in [ref_rows,pc_rows]:
                rows[k].update(valid=False,pipeline_valid=False,reason='refinement_isolated_recovery',pipeline_reason='refinement_isolated_recovery')
            for d in diagnostics:
                if d['section_index']==k:d['status']=str(statuses[k]);d['new_area_mm2']=None
    # Freeze the searched geometry before the independent acceptance pass.
    # Holdout probes must never feed back into the orientation search.
    z['final_probe_profile_area_mm2']=np.array([x['area_mm2'] for x in ref_rows],float)
    z['final_probe_profile_valid']=np.array([x['valid'] for x in ref_rows],bool)
    z['final_probe_recovered_mask']=statuses=='recovered_stable_plane'
    notes={d['section_index']:d for d in diagnostics}
    for k in np.flatnonzero(z['final_probe_recovered_mask']):
        test=tilt_check(k,tangents[k],ref_rows[k],POLICY['holdout_azimuth_offset_deg'])
        notes[k]['new_holdout_tilt']=test
        if not test['stable']:
            statuses[k]='automatically_withheld_holdout_tilt'
            for rows in [ref_rows,pc_rows]:
                rows[k].update(valid=False,pipeline_valid=False,reason='refinement_holdout_tilt_failed',pipeline_reason='refinement_holdout_tilt_failed')
            notes[k].update(status=str(statuses[k]),new_area_mm2=None)
    def final_probe(k,sq):
        c=atlas_center(sq,segid[k],xyz,sid,atlas_s)
        p=reference(k,tangents[k],c,sq)
        return {'center_mm':c.tolist(),'normal':tangents[k].tolist(),
                'actual_area_mm2':p.get('area_mm2'),'valid':p['pipeline_valid'],'reason':p['pipeline_reason']}
    before_shift=np.array([x['valid'] for x in ref_rows],bool)
    final_valid,final_verification=verify_final_shifts(ss,segid,z['final_probe_profile_area_mm2'],
        before_shift,z['final_probe_recovered_mask'],final_probe)
    for k in np.flatnonzero(before_shift & ~final_valid):
        statuses[k]='automatically_withheld_final_shift'
        for rows in [ref_rows,pc_rows]:
            rows[k].update(valid=False,pipeline_valid=False,reason='refinement_final_shift_failed',pipeline_reason='refinement_final_shift_failed')
        if k not in notes:
            notes[k]={'section_index':int(k),'segment_id':int(segid[k]),'s_mm':ss[k],
                      'original_area_mm2':float(old_a[k])}
            diagnostics.append(notes[k])
        notes[k].update(status=str(statuses[k]),new_area_mm2=None)
    density_error=np.full(len(ss),np.nan)
    pc_status=np.full(len(ss),'previously_invalid',dtype='<U64')
    for k,row in enumerate(pc_rows):
        if not ref_rows[k]['valid']:
            if old_pc_v[k] or row['valid']:
                pc_status[k]='automatically_withheld_reference'
            if row['valid']:
                row.update(valid=False,pipeline_valid=False,reason='refinement_reference_unverified',pipeline_reason='refinement_reference_unverified')
                diagnostics.append({'section_index':k,'segment_id':int(segid[k]),'s_mm':ss[k],
                                    'status':str(pc_status[k]),'source':'pointcloud',
                                    'reference_reason':ref_rows[k]['reason']})
            continue
        if not row['valid']:continue
        ph=cloud(k,tangents[k],half)
        error=abs(ph['area_mm2']/row['area_mm2']-1) if ph['pipeline_valid'] else None
        density_error[k]=error if error is not None else np.nan
        if ph['pipeline_valid'] and error<=POLICY['source_relative_limit']:
            pc_status[k]='density_verified'
        else:
            pc_status[k]='automatically_withheld_density'
            row.update(valid=False,pipeline_valid=False,reason='refinement_density_unstable',pipeline_reason='refinement_density_unstable')
            diagnostics.append({'section_index':k,'segment_id':int(segid[k]),'s_mm':ss[k],
                                'status':str(pc_status[k]),'source':'pointcloud',
                                'half_density_valid':ph['pipeline_valid'],'half_density_reason':ph['reason'],
                                'density_relative_change':error})
    for p,rows in [('section_',ref_rows),('pc_section_',pc_rows)]:
        for name in ['area_mm2','req_mm','perimeter_mm','roundness','eccentricity','centroid_mm']:
            shape=(len(rows),3) if name=='centroid_mm' else (len(rows),)
            z[p+name]=np.asarray([x.get(name,np.full(3,np.nan) if name=='centroid_mm' else np.nan) for x in rows],np.float32).reshape(shape)
        for name in ['valid','raw_contour_valid']:
            z[p+name]=np.array([x[name] for x in rows],bool)
        for name in ['reason','raw_contour_reason']:
            z[p+name]=np.array([x[name] for x in rows])
        for name in ['same_segment_crossing_group_count','same_segment_nonlocal_crossing_count']:
            z[p+name]=np.array([x[name] for x in rows],np.int16)
        z[p+'polygon_xyz_mm'],z[p+'polygon_offsets']=packed_polygons(rows,'polygon_xyz_mm')
    z['section_tangent']=tangents.astype(np.float32)
    z['refinement_status']=statuses
    z['refinement_normal_change_deg']=angle_change.astype(np.float32)
    z['refinement_tilt_max_relative_change']=tilt_error.astype(np.float32)
    z['pc_refinement_status']=pc_status
    z['pc_refinement_density_relative_change']=density_error.astype(np.float32)
    # Actual intersections must match the new plane in downstream viewers.
    all_edges=[];offsets=[0]
    for k in range(len(ss)):
        if angle_change[k]>0:
            edges=slicer.slice(centers[k],tangents[k])['intersection_segments_xyz_mm']
        else:
            off=z['section_intersection_offsets'];edges=z['section_intersection_segments_xyz_mm'][off[k]:off[k+1]]
        all_edges.extend(edges);offsets.append(len(all_edges))
    z['section_intersection_offsets']=np.array(offsets,np.int64)
    z['section_intersection_segments_xyz_mm']=np.asarray(all_edges,np.float32).reshape(-1,2,3)
    remap_fields(z)
    z['centerline_inside_reference']=inside
    core=z['centerline_audit_core_mask']; outside=np.flatnonzero(core&~inside)
    r['P1_centerline']={**r['P1_centerline'],**enclosed,'outside_core_points':len(outside),
                        'core_points':int(core.sum()),'raw_outside_original_core_points':int((original_core&~inside).sum()),
                        'opening_extension_original_core_points':int((original_core&external_tail).sum()),
                        'anatomy_domain':domain,
                        'outside_core_atlas_rows':outside.tolist(),'outside_core_fraction':float((core&~inside).sum()/max(core.sum(),1))}
    paired=z['section_valid']&z['pc_section_valid']; err=np.abs(z['pc_section_area_mm2'][paired]/z['section_area_mm2'][paired]-1)
    r['P4_sections']={**r['P4_sections'],'reference_valid':int(z['section_valid'].sum()),
                     'pointcloud_valid':int(z['pc_section_valid'].sum()),'reference_valid_fraction':float(z['section_valid'].mean()),
                     'pointcloud_valid_fraction':float(z['pc_section_valid'].mean()),'paired_valid':int(paired.sum()),
                     'reference_reasons':dict(Counter(z['section_reason'])),'pointcloud_reasons':dict(Counter(z['pc_section_reason'])),
                     'pointcloud_reference_relative_area_p90_abs':float(np.percentile(err,90)) if len(err) else None,
                     'pointcloud_reference_relative_area_median_abs':float(np.median(err)) if len(err) else None}
    r['P5_wall_mapping']={**r['P5_wall_mapping'],'reference_valid_fraction':float(z['wall_map_valid'].mean()),
                         'pointcloud_valid_fraction':float(z['wall_map_pc_valid'].mean())}
    r['schema']=SCHEMA;r['training_allowed']=False;r['user_approved']=False
    r['source_candidate']={'directory':str(case_dir.resolve()),'report_sha256':sha(case_dir/'report.json'),
                           'geometry_sha256':sha(case_dir/'geometry.npz'),'schema':original['schema']}
    r['algorithm']['pointcloud_mask_source']='refined_reference_plane_quality_AND_pointcloud_polygon; surface_guided_candidate'
    r['source']['pointcloud_portability_proven']=False
    r['legacy_P6_source_candidate_only']={'schema':original['schema'],'stability':r.pop('stability',[]),'P6_stability':r.pop('P6_stability',{})}
    r['stability']=[]
    r['P6_stability']={'scope':'original_tilts_plus_recovery_holdout_tilts_and_final_atlas_shift_fixed_point',
                       'refinement_report':'refinement.json','legacy_sampling_not_current':True}
    counts=dict(Counter(statuses))
    summary={'canonical_id':r['canonical_id'],'n_sections':len(ss),'counts':counts,'pc_counts':dict(Counter(pc_status)),
             'original_reference_valid':int(old_v.sum()),'final_reference_valid':int(z['section_valid'].sum()),
             'original_pointcloud_valid':int(old_pc_v.sum()),'final_pointcloud_valid':int(z['pc_section_valid'].sum()),
             'original_wall_coverage':original['P5_wall_mapping']['reference_valid_fraction'],
             'final_wall_coverage':r['P5_wall_mapping']['reference_valid_fraction'],
             'original_outside_core':original['P1_centerline']['outside_core_points'],'final_outside_core':len(outside),
             'source_closed':enclosed['closed_reference_valid'], 'seconds':time.monotonic()-started}
    r['refinement']={**summary,'policy':POLICY}
    r['warnings']=[w for w in r['warnings'] if w not in ['centerline_inside_audit_needs_review','reference_section_coverage_low','pointcloud_section_coverage_low']]
    if len(outside) or not enclosed['closed_reference_valid']:r['warnings'].append('centerline_inside_audit_needs_review')
    if z['section_valid'].mean()<.60:r['warnings'].append('reference_section_coverage_low')
    if z['pc_section_valid'].mean()<.50:r['warnings'].append('pointcloud_section_coverage_low')
    r['status']='refined_geometry_candidate_with_explicit_missing_regions'
    dest=output/'cases'/case_dir.name;dest.mkdir(parents=True,exist_ok=True)
    np.savez_compressed(dest/'geometry.npz',**z)
    r['geometry_npz']=str(dest/'geometry.npz')
    r['refinement_code']={p.name:sha(p) for p in [Path(__file__),Path(__file__).with_name('section_features.py'),Path(__file__).with_name('geometry_candidate.py')]}
    write_json(dest/'refinement.json',{'schema':SCHEMA,'policy':POLICY,**summary,'sections':diagnostics,
                                     'final_verification':final_verification})
    write_json(dest/'report.json',r)
    print(json.dumps(summary,ensure_ascii=False),flush=True)
    return summary

def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--source',type=Path,required=True);ap.add_argument('--output',type=Path,required=True)
    ap.add_argument('--split',type=Path,default=DEFAULT_SPLIT);ap.add_argument('--workers',type=int,default=4)
    ap.add_argument('--cases',nargs='+');a=ap.parse_args()
    if not os.environ.get('SLURM_JOB_ID'):ap.error('patient geometry reconstruction requires a Slurm CPU allocation')
    if a.source.resolve()==a.output.resolve():ap.error('output must be separate from the source')
    split=json.loads(a.split.read_text());cases=set(split['test_cases'])
    if 'train_cases' in split:cases.update(split['train_cases'])
    else:
        for fold in split['cv'].values():cases.update(fold.get('train_cases',[]));cases.update(fold.get('val_cases',[]))
    if a.cases:
        if not set(a.cases)<=cases:ap.error('requested cases outside active split')
        cases=set(a.cases)
    tasks=[(a.source/'cases'/c.replace('/','__'),a.output,True) for c in sorted(cases)]
    started=time.monotonic()
    with ProcessPoolExecutor(max_workers=a.workers) as pool: results=list(pool.map(refine_case,tasks))
    write_json(a.output/'refinement_manifest.json',{'schema':SCHEMA,'source':str(a.source.resolve()),
              'split':str(a.split.resolve()),'split_sha256':sha(a.split),'job_id':os.environ.get('SLURM_JOB_ID'),
              'policy':POLICY,'cases':results,'seconds':time.monotonic()-started,'training_allowed':False})

if __name__=='__main__':main()

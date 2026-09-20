"""Independent numerical and provenance audit of refined geometry candidates.

Only NumPy and standard-library geometry formulas are used. No candidate
geometry implementation or historical derived-field audit is imported.
Algorithmically masked regions are counted, never reported as user decisions.
"""
from __future__ import annotations
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import numpy as np

SCHEMA = 'wss_v6_geometry_refined_independent_audit_v1'
CONTEXT_NAMES = ('upstream_min_area_ratio', 'downstream_min_area_ratio',
                 'upstream_min_distance_mm', 'downstream_min_distance_mm')
WALL_NAMES = ('area_mm2', 'area_slope_per_mm', 'roundness', 'eccentricity', *CONTEXT_NAMES)
RTOL, ATOL, PLANE_ATOL = 2e-4, 2e-4, 2e-4


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for part in iter(lambda: handle.read(1024*1024), b''): h.update(part)
    return h.hexdigest()


def contiguous_runs(sid, valid):
    """Runs terminate at either an invalid station or a segment boundary."""
    run = []
    for i in range(len(valid)):
        if not valid[i] or (run and sid[i] != sid[run[-1]]):
            if run: yield np.asarray(run, int)
            run = []
        if valid[i]: run.append(i)
    if run: yield np.asarray(run, int)


def independent_longitudinal(s, sid, area, valid, window=10.):
    s, area = np.asarray(s, float), np.asarray(area, float)
    valid = np.asarray(valid, bool) & np.isfinite(area) & (area > 0)
    out = {name: np.full(len(s), np.nan) for name in ('area_slope_per_mm', *CONTEXT_NAMES)}
    for run in contiguous_runs(sid, valid):
        if len(run) >= 3:
            for n, i in enumerate(run):
                triple = run[max(0, min(n-1, len(run)-3)):max(0, min(n-1, len(run)-3))+3]
                # Derivative of the three-point Lagrange interpolant, valid
                # for unequal station spacing and one-sided endpoints.
                x = s[triple] - s[i]
                weights = np.array([-(x[(j+1)%3]+x[(j+2)%3]) /
                                    ((x[j]-x[(j+1)%3])*(x[j]-x[(j+2)%3])) for j in range(3)])
                out['area_slope_per_mm'][i] = weights @ np.log(area[triple])
        for n, i in enumerate(run):
            for name, indices in [('upstream', run[:n][::-1]), ('downstream', run[n+1:])]:
                candidates = [j for j in indices if 0 < abs(s[j]-s[i]) <= window]
                if not candidates: continue
                smallest = min(area[j] for j in candidates)
                # candidates are visited nearest first; ties keep the first.
                j = next(j for j in candidates if abs(area[j]-smallest) <= 1e-10+1e-8*abs(smallest))
                out[name+'_min_area_ratio'][i] = area[j]/area[i]
                out[name+'_min_distance_mm'][i] = abs(s[j]-s[i])
    return out


def independent_interpolation(query, s, value, valid):
    query, s, value = np.asarray(query, float), np.asarray(s, float), np.asarray(value, float)
    out, accepted = np.full(len(query), np.nan), np.zeros(len(query), bool)
    if len(s) < 2: return out, accepted
    hi = np.digitize(query, s, right=False)
    hi = np.maximum(1, np.minimum(hi, len(s)-1)); lo = hi-1
    accepted = ((query >= s[0]) & (query <= s[-1]) & valid[lo] & valid[hi]
                & np.isfinite(value[lo]) & np.isfinite(value[hi]))
    out[accepted] = np.interp(query[accepted], s, value)
    return out, accepted


def independent_shift_prediction(index, delta_mm, s, sid, area, valid):
    """Log-linear prediction from the immediate same-segment valid neighbor.

    Returning None represents absent evidence, including when a query extends
    beyond that neighbor. An isolated valid station never acquires a forecast.
    """
    if delta_mm == 0:return None
    neighbor=index+(1 if delta_mm>0 else -1)
    if not 0<=neighbor<len(s) or sid[neighbor]!=sid[index]:return None
    if not valid[index] or not valid[neighbor]:return None
    a,b=float(area[index]),float(area[neighbor])
    if not (np.isfinite(a) and np.isfinite(b) and a>0 and b>0):return None
    distance=float(s[neighbor]-s[index])
    fraction=delta_mm/distance if distance else np.nan
    if not 0<=fraction<=1:return None
    return float(a**(1-fraction)*b**fraction)


def independent_recovery_probe_decision(shift_observations, tilt_observations,
                                       shift_limit=.25, tilt_limit=.20,
                                       require_shift=True):
    """A missing forecast is neutral; an evaluable but invalid slice fails.

    The shift list contains independently calculated predictions. A recovery
    requires at least one positive finite shift comparison plus four valid
    positive finite diagonal-tilt comparisons.
    """
    compared=0;shift_ok=True
    for item in shift_observations:
        predicted=item['predicted_area_mm2']
        if predicted is None:continue
        actual=item.get('actual_area_mm2')
        usable=(item['valid'] and actual is not None and np.isfinite(actual)
                and actual>0 and np.isfinite(predicted) and predicted>0)
        if not usable:
            shift_ok=False
            continue
        compared+=1
        shift_ok &= abs(actual/predicted-1)<=shift_limit
    tilt_ok=len(tilt_observations)==4
    for item in tilt_observations:
        actual=item.get('actual_area_mm2');baseline=item['baseline_area_mm2']
        usable=(item['valid'] and actual is not None and np.isfinite(actual)
                and actual>0 and np.isfinite(baseline) and baseline>0)
        tilt_ok &= bool(usable and abs(actual/baseline-1)<=tilt_limit)
    evidence=compared>0 or not require_shift
    return {'shift_compared':compared,'shift_pass':bool(shift_ok and evidence),
            'tilt_pass':bool(tilt_ok),'pass':bool(shift_ok and evidence and tilt_ok)}


def independent_atlas_position(segment, query_s, xyz, sid, s):
    rows=np.flatnonzero(sid==segment)
    rows=rows[np.argsort(s[rows])]
    if not len(rows) or query_s<s[rows[0]] or query_s>s[rows[-1]]:return None
    return np.array([np.interp(query_s,s[rows],xyz[rows,axis]) for axis in range(3)])


def independent_diagonal_tilt_normal(normal, azimuth_deg, tilt_deg=5.):
    normal=np.asarray(normal,float);normal=normal/np.linalg.norm(normal)
    basis=np.eye(3)[np.argmin(np.abs(normal))]
    basis=basis-(basis@normal)*normal;basis/=np.linalg.norm(basis)
    transverse=np.cross(normal,basis)
    azimuth,tilt=np.radians([azimuth_deg,tilt_deg])
    return np.cos(tilt)*normal+np.sin(tilt)*(np.cos(azimuth)*basis+np.sin(azimuth)*transverse)


def polygon_measure(vertices, center, normal):
    vertices = np.asarray(vertices, float)
    if len(vertices) > 1 and np.linalg.norm(vertices[0]-vertices[-1]) < 1e-7: vertices = vertices[:-1]
    if len(vertices) < 3 or not np.isfinite(vertices).all(): return None
    normal = normal / np.linalg.norm(normal)
    axis = np.eye(3)[np.argmin(np.abs(normal))]
    u = np.cross(normal, axis); u /= np.linalg.norm(u); v = np.cross(normal, u)
    rel = vertices-center; pts = np.column_stack((rel@u, rel@v))
    following = np.roll(pts, -1, axis=0)
    cross = pts[:,0]*following[:,1] - following[:,0]*pts[:,1]
    signed = cross.sum()/2
    if abs(signed) <= 1e-10: return None
    centroid_2d = ((pts+following)*cross[:,None]).sum(axis=0)/(6*signed)
    area = abs(signed); perim = np.linalg.norm(following-pts, axis=1).sum()
    req = np.sqrt(area/np.pi)
    # Winding angle is independent of the generator's ray parity test.
    angle = np.arctan2(cross, np.einsum('ij,ij->i', pts, following)).sum()
    inside = abs(angle) > np.pi or np.min(np.linalg.norm(pts, axis=1)) < 1e-7
    i, j = np.triu_indices(len(pts), 2)
    keep = ~((i == 0) & (j == len(pts)-1)); i, j = i[keep], j[keep]
    def turn(a,b,c): return (b[:,0]-a[:,0])*(c[:,1]-a[:,1])-(b[:,1]-a[:,1])*(c[:,0]-a[:,0])
    a,b,c,d = pts[i],following[i],pts[j],following[j]
    crossing = (turn(a,b,c)*turn(a,b,d) < -1e-18) & (turn(c,d,a)*turn(c,d,b) < -1e-18)
    return {'area_mm2':area, 'perimeter_mm':perim, 'req_mm':req,
            'roundness':4*np.pi*area/perim**2, 'eccentricity':np.linalg.norm(centroid_2d)/req,
            'centroid_mm':center+centroid_2d[0]*u+centroid_2d[1]*v,
            'max_plane_residual_mm':float(np.max(np.abs(rel@normal))),
            'contains_center':bool(inside), 'self_intersects':bool(crossing.any()),
            'nonzero_edges':bool(np.all(np.linalg.norm(following-pts,axis=1)>1e-9))}


def independent_fan_center(polygon):
    polygon=np.asarray(polygon,float);mean=polygon.mean(axis=0)
    _,vectors=np.linalg.eigh((polygon-mean).T@(polygon-mean));normal=vectors[:,0]
    metric=polygon_measure(polygon,mean,normal)
    if metric is None or metric['self_intersects'] or not metric['contains_center']:return None
    center=metric['centroid_mm'];relative=polygon-center;following=np.roll(relative,-1,axis=0)
    signed=np.cross(relative,following)@normal
    projected=relative-(relative@normal)[:,None]*normal
    following_projected=np.roll(projected,-1,axis=0)
    eps=64*np.finfo(float).eps*max(np.max(np.sum(projected**2,axis=1)),1.)
    winding=np.sum(np.arctan2(signed,np.einsum('ij,ij->i',projected,following_projected)))/(2*np.pi)
    if not (np.all(signed>eps) or np.all(signed < -eps)) or abs(abs(winding)-1)>1e-8:return None
    return center


def independent_fan_line_hits(point, direction, polygon, fan_center):
    """Solve barycentric coordinates and line parameter as a 3x3 system."""
    hits=[]
    for a,b in zip(polygon,np.roll(polygon,-1,axis=0)):
        matrix=np.column_stack((a-fan_center,b-fan_center,-direction))
        if abs(np.linalg.det(matrix))<1e-14:continue
        u,v,t=np.linalg.solve(matrix,point-fan_center)
        if u>=-1e-9 and v>=-1e-9 and u+v<=1+1e-9:hits.append(float(t))
    return np.asarray(hits,float)


def audit_fin_cleanup(z, report, ck):
    cleanup=report['P1_centerline'].get('cap_audit',{}).get('slicer_zero_thickness_fin_cleanup')
    if cleanup is None:return
    triangles=z['wall_triangles'];rows=cleanup['removed_triangle_rows']
    ck.require(cleanup['removed_triangles']==len(rows) and len(set(rows))==len(rows),'fin_cleanup_counts')
    ck.require(cleanup['frozen_source_modified'] is False,'fin_source_not_modified')
    if not rows:
        ck.require(not cleanup['pairs'],'empty_fin_cleanup_proof')
        return
    before=Counter(tuple(sorted(edge)) for t in triangles for edge in [(int(t[0]),int(t[1])),(int(t[1]),int(t[2])),(int(t[2]),int(t[0]))])
    keep=np.ones(len(triangles),bool);keep[rows]=False
    after=Counter(tuple(sorted(edge)) for t in triangles[keep] for edge in [(int(t[0]),int(t[1])),(int(t[1]),int(t[2])),(int(t[2]),int(t[0]))])
    proved=[]
    for pair in cleanup['pairs']:
        indices=pair['source_triangle_rows'];proved.extend(indices)
        ck.require(len(indices)==2,'fin_is_triangle_pair')
        a,b=triangles[indices]
        same=set(a)==set(b) and len(set(a))==3
        if same:
            permutation=[list(a).index(v) for v in b]
            odd=sum(permutation[i]>permutation[j] for i in range(3) for j in range(i+1,3))%2==1
        else:odd=False
        ck.require(same and odd,'fin_pair_exact_opposite',rows=indices)
        edges=[tuple(sorted(edge)) for edge in [(int(a[0]),int(a[1])),(int(a[1]),int(a[2])),(int(a[2]),int(a[0]))]]
        pre=[before[e] for e in edges];post=[after[e] for e in edges]
        ck.require(all(n in [2,4] for n in pre) and 4 in pre and all(y==x-2 for x,y in zip(pre,post)),'fin_incidence_repair_proof',rows=indices)
        ck.require(pre==pair['edge_incidence_before'] and post==pair['edge_incidence_after'],'fin_incidence_report',rows=indices)
    ck.require(sorted(proved)==rows,'fin_removed_rows_proved')
    ck.require({e for e,n in before.items() if n==1}=={e for e,n in after.items() if n==1},'fin_cleanup_preserves_opening_edges')


def audit_opening_domain(z, report, old, old_report, ck):
    if 'sampling_domain' not in report:
        ck.require(np.array_equal(z['centerline_audit_core_mask'],old['centerline_audit_core_mask']),'unchanged_legacy_core_mask')
        return
    domain=report['sampling_domain']; xyz=z['centerline_xyz_mm'].astype(float)
    s=z['centerline_s_local_mm'].astype(float); sid=z['centerline_segment_id']
    external=z['centerline_opening_extension_mask']; original=z['centerline_original_audit_core_mask']
    ck.require(np.array_equal(original,old['centerline_audit_core_mask']),'original_core_mask_frozen')
    ck.require(np.array_equal(z['centerline_audit_core_mask'],original&~external),'opening_extension_only_core_mask_change')
    ck.require(domain['excluded_atlas_rows']==np.flatnonzero(external).tolist() and domain['excluded_points']==int(external.sum()),'opening_extension_domain_counts')
    ck.require(domain==report['P1_centerline']['anatomy_domain'],'P1_sampling_domain_matches')
    expected_mask=np.zeros(len(xyz),bool)
    bounds={int(seg['segment_id']):[seg['start_s_mm'],seg['end_s_mm']] for seg in old_report['segments']}
    by={int(seg['segment_id']):seg for seg in old_report['segments']}
    openings={int(row['segment_id']):row for row in report['openings']}
    for ext in domain['extensions']:
        segment=int(ext['segment_id']); seg=by[segment]; opening=openings[segment]
        rows=np.flatnonzero(sid==segment);rows=rows[np.argsort(s[rows])]
        inlet=ext['endpoint']=='start'
        ck.require((inlet and seg['parent_id']==-1) or (not inlet and not seg['children']),'extension_own_terminal_opening',segment=segment)
        if not inlet:rows=rows[::-1]
        matched=np.asarray(ext['atlas_rows'],int);next_row=ext['first_interior_atlas_row']
        ck.require(np.array_equal(matched,rows[:len(matched)]) and len(matched)<len(rows) and rows[len(matched)]==next_row,'extension_endpoint_contiguous',segment=segment)
        expected_mask[matched]=True
        normal=np.asarray(opening['normal'],float);normal/=np.linalg.norm(normal)
        center=np.asarray(opening['center_mm'],float);poly=np.asarray(opening['polygon_xyz_mm'],float)
        residual=np.max(np.abs((poly-center)@normal));axial=(xyz[matched]-center)@normal
        method=ext.get('crossing_method','fitted_rim_plane')
        ck.require(method in ['fitted_rim_plane','verified_3d_cap_triangle_intersection'],'known_opening_crossing_method',segment=segment)
        fan=independent_fan_center(poly) if method=='verified_3d_cap_triangle_intersection' else None
        if method=='verified_3d_cap_triangle_intersection':
            ck.require(fan is not None and ext.get('fan_proof',{}).get('fan_verified') is True,'actual_cap_fan_independently_verified',segment=segment)
        offsets=[]
        if fan is not None:
            for row in matched:
                hit=independent_fan_line_hits(xyz[row],normal,poly,fan)
                offsets.append(-float(np.max(hit)) if len(hit) else np.nan)
            offsets=np.asarray(offsets)
            stored=ext.get('actual_cap_outward_offsets_mm',ext.get('verified_cap_outward_offsets_mm'))
            ck.compare(offsets,np.asarray([np.nan if v is None else v for v in stored]),'actual_cap_offsets:'+str(segment),atol=2e-6,rtol=2e-6)
            cap_outward=np.isfinite(offsets)&(offsets>1e-7)&~z['centerline_inside_reference'][matched]
            ck.require(np.all((axial>residual+1e-7)|cap_outward),'extension_outward_of_verified_rim_or_actual_cap',segment=segment)
        else:
            ck.require(np.all(axial>residual+1e-7),'extension_strictly_outward_of_rim_band',segment=segment)
        ck.compare(axial,ext['outward_offsets_mm'],'extension_offsets:'+str(segment),atol=2e-6,rtol=2e-6)
        ck.compare(residual,ext['rim_plane_residual_max_mm'],'extension_rim_residual:'+str(segment))
        for row in matched:
            metric=polygon_measure(poly,xyz[row],normal)
            ck.require(metric is not None and metric['contains_center'],'extension_projects_inside_own_opening',atlas_row=int(row))
        next_d=float((xyz[next_row]-center)@normal)
        ck.require(z['centerline_inside_reference'][next_row] and next_d<=residual+1e-7,'extension_next_sample_verified_inside',segment=segment)
        d0=float(axial[-1]);alpha=float(np.clip(d0/(d0-next_d),0.,1.))
        if fan is not None:
            hits=independent_fan_line_hits(xyz[matched[-1]],xyz[next_row]-xyz[matched[-1]],poly,fan)
            hits=hits[(hits>=-1e-9)&(hits<=1+1e-9)]
            ck.require(len(hits)>0 and np.ptp(hits)<1e-7,'unique_actual_cap_segment_intersection',segment=segment)
            if len(hits):alpha=float(np.median(hits))
        crossing_s=s[matched[-1]]+alpha*(s[next_row]-s[matched[-1]])
        crossing_xyz=xyz[matched[-1]]+alpha*(xyz[next_row]-xyz[matched[-1]])
        ck.compare(crossing_s,ext['crossing_s_mm'],'extension_crossing_s:'+str(segment),atol=2e-6,rtol=2e-6)
        ck.compare(crossing_xyz,ext['crossing_xyz_mm'],'extension_crossing_xyz:'+str(segment))
        bounds[segment][0 if inlet else 1]=crossing_s
    ck.require(np.array_equal(expected_mask,external),'all_extension_mask_has_geometric_evidence')
    for seg in report['segments']:
        k=int(seg['segment_id']);actual=domain['segment_s_bounds_mm'][str(k)]
        ck.compare(bounds[k],actual,'sampling_bounds_evidence:'+str(k),atol=2e-6,rtol=2e-6)
        ck.compare(actual,[seg['start_s_mm'],seg['end_s_mm']],'segment_sampling_bounds:'+str(k),atol=1e-9,rtol=0.)
    p1=report['P1_centerline']
    ck.require(p1['raw_outside_original_core_points']==int(np.sum(original&~z['centerline_inside_reference'])),'P1_original_core_outside_count')
    ck.require(p1['opening_extension_original_core_points']==int(np.sum(original&external)),'P1_external_original_core_count')


class Recorder:
    def __init__(self): self.errors=[]; self.numeric=[]; self.roundoff=[]; self.checks=0
    def require(self, condition, name, **evidence):
        self.checks += 1
        if not condition: self.errors.append({'check':name,'category':'structural',**evidence})
    def compare(self, expected, actual, name, atol=ATOL, rtol=RTOL):
        self.checks += 1
        expected, actual = np.asarray(expected), np.asarray(actual)
        if expected.shape != actual.shape:
            self.errors.append({'check':name,'category':'structural','expected_shape':list(expected.shape),'actual_shape':list(actual.shape)}); return
        ef, af = np.isfinite(expected), np.isfinite(actual)
        mismatch = int(np.sum(ef != af))
        if mismatch: self.errors.append({'check':name,'category':'structural','finite_mask_mismatches':mismatch})
        both = ef & af
        if not np.any(both): return
        diff = np.abs(expected[both]-actual[both]); lim = atol+rtol*np.abs(expected[both])
        summary = {'check':name,'compared':int(both.sum()),'max_abs_difference':float(diff.max()),'atol':atol,'rtol':rtol}
        bad = int(np.sum(diff>lim))
        if bad: self.errors.append({**summary,'category':'numeric','out_of_tolerance':bad})
        elif diff.max() > 1e-7: self.roundoff.append(summary)
        self.numeric.append(summary)


def audit_final_verification(z, ref, ck):
    """Validate unused tilt probes and synchronous final-mask shift evidence.

    Each rejected station is checked against its recorded rejection round's
    mask. Retained stations are checked against the actual final mask, so a
    subsequently rejected neighbor cannot remain their prediction support.
    """
    if 'final_verification' not in ref:
        ck.require(ref.get('schema')!='wss_v6_geometry_refined_v3','v3_requires_final_verification')
        return {'available':False}
    ss=z['section_s_local_mm'].astype(float);sid=z['section_segment_id'];ns=len(ss)
    status=z['refinement_status'];normals=z['section_tangent'].astype(float)
    areas=z['final_probe_profile_area_mm2'].astype(float)
    profile=z['final_probe_profile_valid'].astype(bool); recovered=z['final_probe_recovered_mask'].astype(bool)
    ck.require(areas.shape==profile.shape==recovered.shape==(ns,),'final_probe_profile_shapes')
    ck.compare(areas,z['section_area_mm2'],'final_probe_area_snapshot_preserved')
    ck.require(not np.any(recovered&~profile),'preverification_recovery_requires_valid_profile')
    ck.require(np.array_equal(recovered,(z['refinement_normal_change_deg']>0)&profile),'preverification_recovered_identity')
    expected_prevalid=np.isin(status,['stable_original','recovered_stable_plane',
                                    'automatically_withheld_holdout_tilt','automatically_withheld_final_shift'])
    ck.require(np.array_equal(profile,expected_prevalid),'preverification_valid_profile_identity')
    policy=ref['policy'];shift_limit=policy['shift_residual_limit'];tilt_limit=policy['tilt_relative_limit']
    ck.require(policy['shift_mm']==1. and policy['tilt_probe_deg']==5. and policy['holdout_azimuth_offset_deg']==45.,'final_probe_physical_recipe')
    ck.require(policy['recovery_minimum_final_shift_probes']==1,'recovered_shift_evidence_minimum')
    diagnostics={d['section_index']:d for d in ref['sections'] if d.get('source')!='pointcloud'}
    heldout=np.zeros(ns,bool);tilt_count=0
    for k in np.flatnonzero(recovered):
        note=diagnostics.get(int(k),{});test=note.get('new_holdout_tilt')
        ck.require(test is not None,'every_recovery_has_unused_tilt_probes',station=int(k))
        if test is None:continue
        ck.compare(areas[k],test['baseline_area_mm2'],'holdout_baseline:'+str(k),atol=1e-8,rtol=1e-8)
        probes=test['probes'];azimuths=[p['azimuth_deg'] for p in probes]
        ck.require(len(probes)==4 and sorted(azimuths)==[45.,135.,225.,315.],'holdout_four_diagonal_directions',station=int(k))
        changes=[];observations=[];reasons=Counter()
        for p in probes:
            tilt_count+=1
            expected=independent_diagonal_tilt_normal(normals[k],p['azimuth_deg'],5.)
            ck.compare(expected,p['normal'],'holdout_normal:'+str(k)+':'+str(p['azimuth_deg']),atol=2e-6,rtol=0.)
            ck.compare(z['section_center_mm'][k],p['center_mm'],'holdout_center:'+str(k),atol=2e-6,rtol=0.)
            ck.require(p['tilt_deg']==5.,'holdout_tilt_angle',station=int(k))
            ck.require(p['valid']==(p['reason']=='ok'),'holdout_valid_reason',station=int(k))
            actual=p['area_mm2']
            if p['valid']:
                ck.require(actual is not None and np.isfinite(actual) and actual>0,'holdout_valid_positive_area',station=int(k))
                change=abs(actual/areas[k]-1);changes.append(change)
                ck.compare(change,p['relative_change'],'holdout_relative_change:'+str(k),atol=1e-8,rtol=1e-8)
            else:
                reasons[p['reason']]+=1
                ck.require(p['relative_change'] is None,'invalid_holdout_no_comparison',station=int(k))
            observations.append({'valid':p['valid'],'actual_area_mm2':actual,'baseline_area_mm2':areas[k]})
        decision=independent_recovery_probe_decision([],observations,shift_limit,tilt_limit,require_shift=False)
        passed=decision['tilt_pass'];heldout[k]=not passed
        ck.require(test['stable']==passed and test['n_valid']==len(changes) and test['all_valid']==(len(changes)==4),'holdout_summary_decision',station=int(k))
        ck.require(test['reasons']==dict(reasons),'holdout_failure_reason_counts',station=int(k))
        ck.compare(max(changes) if changes else np.nan,np.nan if test['max_relative_change'] is None else test['max_relative_change'],'holdout_max_change:'+str(k),atol=1e-8,rtol=1e-8)
    ck.require(np.array_equal(heldout,status=='automatically_withheld_holdout_tilt'),'holdout_failure_masks_exactly')
    initial=profile&~heldout;fv=ref['final_verification'];rounds=fv['rounds'];records=fv['sections']
    ck.require(isinstance(rounds,int) and rounds>=1,'final_shift_round_count')
    indices=[p['section_index'] for p in records]
    ck.require(len(indices)==len(set(indices)) and set(indices)==set(np.flatnonzero(initial)),'final_shift_record_coverage')
    by_index={p['section_index']:p for p in records}
    failed={p['section_index']:p['iteration'] for p in records if not p['stable']}
    failed_mask=np.zeros(ns,bool)
    if failed:failed_mask[list(failed)]=True
    ck.require(np.array_equal(failed_mask,status=='automatically_withheld_final_shift'),'final_shift_failure_status_identity')
    ck.require(np.array_equal(initial&~failed_mask,z['section_valid']),'final_shift_mask_closure')
    ck.require(all(1<=iteration<rounds for iteration in failed.values()),'failure_precedes_final_converged_round')
    ck.require(rounds==max(failed.values(),default=0)+1,'final_shift_rounds_end_at_first_no_rejection')
    for iteration in range(1,rounds):
        ck.require(any(t==iteration for t in failed.values()),'each_nonfinal_round_rejects_stations',iteration=iteration)
    actual_comparisons=0;final_recovered_with_evidence=0;unassessed_original=0
    for k,d in by_index.items():
        iteration=d['iteration']
        ck.require(1<=iteration<=rounds and (not d['stable'] or iteration==rounds),'last_record_iteration',station=k)
        ck.require(d['was_recovered']==bool(recovered[k]) and d['segment_id']==int(sid[k]),'final_shift_station_identity',station=k)
        ck.compare(ss[k],d['s_mm'],'final_shift_local_s:'+str(k),atol=1e-8,rtol=1e-8)
        mask=initial.copy()
        prior=[i for i,t in failed.items() if t<iteration]
        if prior:mask[prior]=False
        expected={}
        for delta in [-1.,1.]:
            pred=independent_shift_prediction(k,delta,ss,sid,areas,mask)
            if pred is not None:expected[delta]=pred
        shifts=d['shift'];deltas=[p['delta_mm'] for p in shifts]
        ck.require(len(deltas)==len(set(deltas)) and set(deltas)==set(expected),'final_shift_predictable_directions',station=k)
        observations=[]
        for p in shifts:
            delta=p['delta_mm'];pred=expected.get(delta)
            if pred is None:continue
            query=ss[k]+delta;neighbor=k+(1 if delta>0 else -1)
            ck.compare(query,p['query_s_mm'],'final_shift_query_s:'+str(k),atol=1e-8,rtol=1e-8)
            ck.require(p['neighbor_index']==neighbor and mask[neighbor] and sid[neighbor]==sid[k],'final_shift_neighbor_identity',station=k)
            ck.compare(areas[neighbor],p['neighbor_area_mm2'],'final_shift_neighbor_area:'+str(k),atol=1e-8,rtol=1e-8)
            ck.compare(pred,p['predicted_area_mm2'],'final_shift_prediction:'+str(k),atol=1e-8,rtol=1e-8)
            center=independent_atlas_position(sid[k],query,z['centerline_xyz_mm'].astype(float),z['centerline_segment_id'],z['centerline_s_local_mm'].astype(float))
            ck.require(center is not None,'final_shift_query_within_atlas',station=k)
            if center is not None:ck.compare(center,p['center_mm'],'final_shift_center_follows_atlas:'+str(k),atol=2e-6,rtol=0.)
            ck.compare(normals[k],p['normal'],'final_shift_keeps_final_normal:'+str(k),atol=2e-6,rtol=0.)
            ck.require(p['valid']==(p['reason']=='ok'),'final_shift_gate_reason',station=k)
            actual=p['actual_area_mm2']
            if p['valid']:
                actual_comparisons+=1
                ck.require(actual is not None and np.isfinite(actual) and actual>0,'final_shift_valid_positive_area',station=k)
                ck.compare(abs(actual/pred-1),p['relative_residual'],'final_shift_residual:'+str(k),atol=1e-8,rtol=1e-8)
            else:ck.require(p['relative_residual'] is None,'invalid_final_shift_not_comparable',station=k)
            observations.append({'valid':p['valid'],'actual_area_mm2':actual,'predicted_area_mm2':pred})
        result=independent_recovery_probe_decision(observations,[],shift_limit,tilt_limit,require_shift=bool(recovered[k]))
        ck.require(d['stable']==result['shift_pass'],'final_shift_independent_decision',station=k)
        if d['stable'] and recovered[k]:
            ck.require(result['shift_compared']>=1,'retained_recovery_has_final_shift_evidence',station=k)
            final_recovered_with_evidence+=1
        if d['stable'] and not recovered[k] and not observations:
            unassessed_original+=1
            ck.require(d['reason']=='no_contiguous_shift_bracket','unassessed_original_explicit_reason',station=k)
    return {'available':True,'candidate_recoveries':int(recovered.sum()),'holdout_probes':tilt_count,
            'holdout_rejected':int(heldout.sum()),'shift_rejected':int(failed_mask.sum()),
            'shift_rounds':rounds,'shift_stations':len(records),'shift_valid_comparisons':actual_comparisons,
            'retained_recoveries_with_final_shift_evidence':final_recovered_with_evidence,
            'original_valid_without_shift_brackets':unassessed_original}


def audit_pointcloud_reference_dependency(z, old, ref, ck):
    if not ref['policy'].get('pointcloud_requires_verified_reference'):return
    valid=z['pc_section_valid'];reference=z['section_valid'];pcs=z['pc_refinement_status']
    ck.require(not np.any(valid&~reference),'pointcloud_valid_requires_verified_reference')
    expected_withheld=old['pc_section_valid']&~reference
    ck.require(np.array_equal(pcs=='automatically_withheld_reference',expected_withheld),'pointcloud_reference_withheld_identity')
    recovered=z.get('final_probe_recovered_mask',z['refinement_status']=='recovered_stable_plane')
    eligible=reference&(old['pc_section_valid']|recovered)
    density_checked=np.isin(pcs,['density_verified','automatically_withheld_density'])
    ck.require(np.array_equal(eligible,density_checked),'pointcloud_density_eligible_population')
    ck.require(np.isnan(z['pc_refinement_density_relative_change'][~eligible]).all(),'unassessed_density_values_are_nan')
    diagnostics=[d for d in ref['sections'] if d.get('source')=='pointcloud' and d['status']=='automatically_withheld_reference']
    indices=[d['section_index'] for d in diagnostics]
    expected_diag=np.flatnonzero(z['pc_section_reason']=='refinement_reference_unverified')
    ck.require(len(indices)==len(set(indices)) and set(indices)==set(expected_diag),'pointcloud_reference_diagnostic_coverage')
    for d in diagnostics:
        i=d['section_index']
        ck.require(expected_withheld[i] and d['reference_reason']==z['section_reason'][i],'pointcloud_reference_diagnostic_reason',station=i)


def audit_case(candidate_dir, source_dir, code_root=None):
    r = json.loads((candidate_dir/'report.json').read_text())
    old_r = json.loads((source_dir/'report.json').read_text())
    ref = json.loads((candidate_dir/'refinement.json').read_text())
    with np.load(candidate_dir/'geometry.npz',allow_pickle=False) as f: z={k:f[k] for k in f.files}
    with np.load(source_dir/'geometry.npz',allow_pickle=False) as f: old={k:f[k] for k in f.files}
    ck=Recorder(); cid=r['canonical_id']; ns=len(z['section_segment_id']); nw=len(z['wall_xyz_mm'])
    ck.require(cid==old_r['canonical_id']==ref['canonical_id'], 'canonical_identity')
    ck.require(r.get('training_allowed') is False and r.get('user_approved') is False and r.get('awaiting_user_review') is True, 'training_review_gates')
    source=r['source_candidate']
    ck.require(source.get('report_sha256')==sha(source_dir/'report.json'), 'source_report_sha256')
    ck.require(source.get('geometry_sha256')==sha(source_dir/'geometry.npz'), 'source_geometry_sha256')
    ck.require(Path(source['directory']).resolve()==source_dir.resolve(), 'source_directory')
    codes=r.get('refinement_code',{})
    ck.require({'refine_geometry_candidate.py','geometry_candidate.py','section_features.py'}<=set(codes), 'refinement_code_required_files')
    for name, expected in codes.items():
        local=(code_root or Path(__file__).parent)/Path(name).name
        ck.require(name==Path(name).name and local.is_file() and sha(local)==expected, 'refinement_code_sha256',file=name)
    # Every original array except the explicitly recomputed fields is frozen.
    mutable={'section_tangent','centerline_inside_reference','centerline_audit_core_mask','section_intersection_offsets','section_intersection_segments_xyz_mm'}
    for p in ['section_','pc_section_']:
        mutable.update(p+k for k in ['area_mm2','req_mm','perimeter_mm','roundness','eccentricity','centroid_mm','valid','reason','raw_contour_valid','raw_contour_reason','same_segment_crossing_group_count','same_segment_nonlocal_crossing_count','polygon_offsets','polygon_xyz_mm','area_slope_per_mm',*CONTEXT_NAMES])
    for w in ['wall_map_','wall_map_pc_']:
        mutable.update(w+k for k in [*WALL_NAMES,'valid','raw_area_mm2'])
        mutable.update(w+k+'_valid' for k in WALL_NAMES if k!='area_mm2')
    for name, original in old.items():
        ck.require(name in z,'source_field_present',field=name)
        if name in z and name not in mutable:
            same=(z[name].dtype==original.dtype and z[name].shape==original.shape and z[name].tobytes()==original.tobytes())
            ck.require(same,'frozen_source_array_identical',field=name)
    ss=z['section_s_local_mm'].astype(float); sid=z['section_segment_id']; normal=z['section_tangent'].astype(float)
    ck.require(normal.shape==(ns,3) and np.isfinite(normal).all(),'normal_shape_finite')
    ck.compare(np.ones(ns),np.linalg.norm(normal,axis=1),'unit_normals',atol=2e-6,rtol=0.)
    for seg in np.unique(sid): ck.require(np.all(np.diff(ss[sid==seg])>0),'strict_section_s',segment=int(seg))
    ck.require(len(np.unique(z['wall_node_id']))==nw,'wall_node_ids_unique')
    tri=z['wall_triangles'];ck.require(tri.ndim==2 and tri.shape[1]==3 and np.all((tri>=0)&(tri<nw)),'wall_triangle_indices')
    raw_retained={}; coverage={}; polygon_count=0
    for p,w in [('section_','wall_map_'),('pc_section_','wall_map_pc_')]:
        accepted=z[p+'valid'].astype(bool); a=z[p+'area_mm2'].astype(float)
        ck.require(accepted.shape==(ns,),'section_valid_shape',source=p)
        ck.require(np.all(np.isfinite(a[accepted])&(a[accepted]>0)),'valid_positive_area',source=p)
        ck.require(np.array_equal(accepted,z[p+'reason']=='ok'),'valid_reason_consistency',source=p)
        ck.require(not np.any(accepted&~z[p+'raw_contour_valid']),'valid_requires_raw_contour',source=p)
        ck.require(not np.any(accepted&(z[p+'same_segment_nonlocal_crossing_count']>0)),'valid_excludes_nonlocal_crossing',source=p)
        off=z[p+'polygon_offsets']; verts=z[p+'polygon_xyz_mm']
        good_offsets=(off.shape==(ns+1,) and np.issubdtype(off.dtype,np.integer) and off[0]==0 and np.all(np.diff(off)>=0) and off[-1]==len(verts) and verts.ndim==2 and verts.shape[1]==3)
        ck.require(good_offsets,'polygon_packing',source=p)
        if good_offsets:
            for i in np.flatnonzero(accepted):
                metric=polygon_measure(verts[off[i]:off[i+1]],z['section_center_mm'][i].astype(float),normal[i])
                ck.require(metric is not None,'valid_polygon_nondegenerate',source=p,station=int(i))
                if metric is None:continue
                polygon_count+=1
                ck.require(metric['contains_center'] and not metric['self_intersects'] and metric['nonzero_edges'],'simple_closed_polygon_contains_center',source=p,station=int(i))
                ck.require(metric['max_plane_residual_mm']<=PLANE_ATOL,'polygon_plane',source=p,station=int(i),residual_mm=metric['max_plane_residual_mm'])
                for name in ['area_mm2','req_mm','perimeter_mm','roundness','eccentricity','centroid_mm']:
                    ck.compare(metric[name],z[p+name][i],p+name+':'+str(i))
        derived=independent_longitudinal(ss,sid,a,accepted)
        for name, value in derived.items():ck.compare(value,z[p+name],p+name)
        values={'area_mm2':a,'roundness':z[p+'roundness'],'eccentricity':z[p+'eccentricity'],**derived}
        for name,value in values.items():
            ev=np.full(nw,np.nan); eg=np.zeros(nw,bool)
            for seg in np.unique(sid):
                stations=np.flatnonzero(sid==seg); points=np.flatnonzero(z['wall_map_segment_id']==seg)
                ev[points],eg[points]=independent_interpolation(z['wall_map_s_local_mm'][points],ss[stations],value[stations],accepted[stations]&np.isfinite(value[stations]))
            if name=='area_mm2':ck.compare(ev,z[w+'raw_area_mm2'],w+'raw_area_mm2')
            eg &= ~z['wall_map_ambiguous']; ev[~eg]=np.nan
            key=w+('valid' if name=='area_mm2' else name+'_valid')
            ck.require(np.array_equal(eg,z[key]),'wall_field_mask',field=key,disagreements=int(np.sum(eg!=z[key])))
            ck.compare(ev,z[w+name],w+name)
            ck.require(np.isnan(z[w+name][~z[key]]).all(),'invalid_wall_values_are_nan',field=w+name)
        raw_retained[p]=int(np.sum(~accepted & np.isfinite(a)))
        coverage[p]=float(accepted.mean()); coverage[w]=float(z[w+'valid'].mean())
    # Full intersection segments for every changed normal must be coplanar.
    edge_off=z['section_intersection_offsets'];edges=z['section_intersection_segments_xyz_mm']
    good=(edge_off.shape==(ns+1,) and np.issubdtype(edge_off.dtype,np.integer) and edge_off[0]==0 and np.all(np.diff(edge_off)>=0) and edge_off[-1]==len(edges) and edges.shape[1:]==(2,3))
    ck.require(good,'intersection_packing')
    if good:
        for i in np.flatnonzero(z['refinement_normal_change_deg']>0):
            local=edges[edge_off[i]:edge_off[i+1]].reshape(-1,3)
            ck.require(len(local)>0,'changed_plane_intersections_present',station=int(i))
            if len(local):ck.require(np.max(np.abs((local-z['section_center_mm'][i])@normal[i]))<=PLANE_ATOL,'changed_plane_intersections_coplanar',station=int(i))
    status=z['refinement_status'];counts={str(k):int(v) for k,v in Counter(status).items()}
    allowed={'previously_invalid','stable_original','recovered_stable_plane','automatically_withheld_unstable','automatically_withheld_isolated_recovery','automatically_withheld_opening_extension','automatically_withheld_holdout_tilt','automatically_withheld_final_shift'}
    ck.require(status.shape==(ns,) and set(counts)<=allowed,'refinement_status_values')
    ck.require(counts==r['refinement']['counts']==ref['counts'],'refinement_status_counts')
    ck.require(np.array_equal(status=='previously_invalid',~old['section_valid']),'previously_invalid_identity')
    stable=status=='stable_original'; recovered=status=='recovered_stable_plane'; held=np.char.startswith(status,'automatically_withheld')
    ck.require(np.all(z['section_valid'][stable|recovered]),'stable_recovered_are_valid')
    ck.require(np.all(z['pc_section_valid'][recovered]),'recovered_cloud_is_valid')
    ck.require(not np.any(z['section_valid'][held]|z['pc_section_valid'][held]),'withheld_masks_both_sources')
    ck.require(np.array_equal(z['section_tangent'][stable],old['section_tangent'][stable]),'stable_normals_unchanged')
    ck.require(np.array_equal(z['section_area_mm2'][stable],old['section_area_mm2'][stable]),'stable_area_unchanged')
    original_norm=old['section_tangent'].astype(float);original_norm/=np.linalg.norm(original_norm,axis=1,keepdims=True)
    cos=np.sum(original_norm*normal/np.linalg.norm(normal,axis=1,keepdims=True),axis=1)
    angles=np.degrees(np.arccos(np.clip(cos,-1,1)))
    ck.compare(angles,z['refinement_normal_change_deg'],'normal_change_angle_deg',atol=.003,rtol=1e-5)
    diagnostics=[d for d in ref['sections'] if d.get('source','reference') != 'pointcloud'];diag_ids=[d['section_index'] for d in diagnostics]
    ck.require(len(set(diag_ids))==len(diag_ids) and set(diag_ids)==set(np.flatnonzero(~stable & (status!='previously_invalid'))),'refinement_diagnostic_station_identity')
    for d in diagnostics:
        i=d['section_index'];ck.require(d['status']==status[i],'diagnostic_status_matches',station=i)
        if recovered[i]:
            ck.require(d['new_tilt']['stable'] and d['new_tilt']['all_valid'] and d['new_tilt']['n_valid']==4,'recovered_four_tilts',station=i)
            ck.compare(z['section_area_mm2'][i],d['new_area_mm2'],'diagnostic_recovered_area:'+str(i))
    if 'pc_refinement_status' in z:
        pcs=z['pc_refinement_status'];density=z['pc_refinement_density_relative_change']
        pc_counts={str(k):int(v) for k,v in Counter(pcs).items()}
        ck.require(set(pc_counts)<={'previously_invalid','density_verified','automatically_withheld_density','automatically_withheld_reference'},'pc_density_status_values')
        ck.require(pc_counts==ref['pc_counts']==r['refinement']['pc_counts'],'pc_density_status_counts')
        verified=pcs=='density_verified';denied=pcs=='automatically_withheld_density'
        ck.require(np.array_equal(verified,z['pc_section_valid']),'pc_valid_requires_density_verified')
        limit=ref['policy']['source_relative_limit']
        ck.require(np.all(np.isfinite(density[verified])&(density[verified]<=limit+2e-7)),'verified_density_relative_change')
        ck.require(np.all(~np.isfinite(density[denied]) | (density[denied]>limit-2e-7)),'withheld_density_failure')
        pcdiag=[d for d in ref['sections'] if d.get('source')=='pointcloud' and d['status']=='automatically_withheld_density'];indices=[d['section_index'] for d in pcdiag]
        ck.require(len(set(indices))==len(indices) and set(indices)==set(np.flatnonzero(denied)),'pc_density_diagnostic_identity')
        for d in pcdiag:
            i=d['section_index'];ck.require(d['status']==pcs[i],'pc_density_diagnostic_status',station=i)
            ck.compare(np.nan if d['density_relative_change'] is None else d['density_relative_change'],density[i],'pc_density_diagnostic_error:'+str(i))
    audit_pointcloud_reference_dependency(z,old,ref,ck)
    final_verification=audit_final_verification(z,ref,ck)
    audit_opening_domain(z,r,old,old_r,ck)
    audit_fin_cleanup(z,r,ck)
    core=z['centerline_audit_core_mask']; outside=np.flatnonzero(core&~z['centerline_inside_reference'])
    p1=r['P1_centerline'];ck.require(p1['core_points']==int(core.sum()),'P1_core_count')
    ck.require(p1['outside_core_points']==len(outside) and p1['outside_core_atlas_rows']==outside.tolist(),'P1_outside_identity')
    ck.compare(len(outside)/max(core.sum(),1),p1['outside_core_fraction'],'P1_outside_fraction',atol=1e-10,rtol=0.)
    paired=z['section_valid']&z['pc_section_valid']; paired_error=np.abs(z['pc_section_area_mm2'][paired]/z['section_area_mm2'][paired]-1)
    p4=r['P4_sections'];p5=r['P5_wall_mapping']
    ck.require(p4['count']==ns and p4['paired_valid']==int(paired.sum()),'P4_section_counts')
    for label,p,w in [('reference','section_','wall_map_'),('pointcloud','pc_section_','wall_map_pc_')]:
        ck.require(p4[label+'_valid']==int(z[p+'valid'].sum()),'P4_valid_count',source=label)
        ck.compare(coverage[p],p4[label+'_valid_fraction'],'P4_'+label+'_coverage',atol=1e-9,rtol=0.)
        ck.compare(coverage[w],p5[label+'_valid_fraction'],'P5_'+label+'_coverage',atol=1e-9,rtol=0.)
        ck.require(dict(Counter(z[p+'reason']))==p4[label+'_reasons'],'P4_reason_counts',source=label)
    if len(paired_error):
        for suffix,val in [('median_abs',np.median(paired_error)),('p90_abs',np.percentile(paired_error,90))]:ck.compare(val,p4['pointcloud_reference_relative_area_'+suffix],'P4_paired_error_'+suffix)
    expected_summary={'n_sections':ns,'original_reference_valid':int(old['section_valid'].sum()),'final_reference_valid':int(z['section_valid'].sum()),'original_pointcloud_valid':int(old['pc_section_valid'].sum()),'final_pointcloud_valid':int(z['pc_section_valid'].sum()),'original_outside_core':old_r['P1_centerline']['outside_core_points'],'final_outside_core':len(outside),'source_closed':p1['closed_reference_valid']}
    for key,val in expected_summary.items():ck.require(ref[key]==r['refinement'][key]==val,'refinement_summary',field=key)
    for key,val in [('original_wall_coverage',old_r['P5_wall_mapping']['reference_valid_fraction']),('final_wall_coverage',coverage['wall_map_'])]:
        ck.compare(val,ref[key],'refinement_'+key,atol=1e-10,rtol=0.)
        ck.compare(val,r['refinement'][key],'report_refinement_'+key,atol=1e-10,rtol=0.)
    return {'canonical_id':cid,'mechanical_checks_pass':not ck.errors,'check_count':ck.checks,'errors':ck.errors,
            'structural_error_count':sum(e['category']=='structural' for e in ck.errors),
            'numeric_error_count':sum(e['category']=='numeric' for e in ck.errors),
            'accepted_float_roundoff_checks':len(ck.roundoff),'maximum_accepted_roundoff':sorted(ck.roundoff,key=lambda d:d['max_abs_difference'],reverse=True)[:5],
            'polygons_audited':polygon_count,'refinement_counts':counts,'coverage':coverage,'final_verification':final_verification,
            'normal_masked_raw_diagnostic_sections':raw_retained,'outside_core_points':len(outside),
            'masked_regions_are_not_audit_errors_or_requests_for_human_decision':True}


def run_audit(source, candidate, cases, split_path=None, code_root=None):
    rows=[]; errors=[]
    for cid in sorted(cases):
        key=cid.replace('/','__')
        try: rows.append(audit_case(candidate/'cases'/key, source/'cases'/key, code_root=code_root))
        except Exception as exc: errors.append({'canonical_id':cid,'category':'structural','error':f'{type(exc).__name__}: {exc}'})
    manifest_path=candidate/'refinement_manifest.json'
    if manifest_path.is_file():
        manifest=json.loads(manifest_path.read_text());manids=[r['canonical_id'] for r in manifest['cases']]
        if len(set(manids))!=len(manids) or set(manids)!=set(cases):errors.append({'category':'structural','error':'manifest case identities differ from audit selection'})
        if split_path is not None and manifest.get('split_sha256')!=sha(split_path):errors.append({'category':'structural','error':'manifest split SHA256 mismatch'})
        by={r['canonical_id']:r for r in rows}
        for summary in manifest['cases']:
            if summary['canonical_id'] in by and summary['counts']!=by[summary['canonical_id']]['refinement_counts']:errors.append({'category':'structural','error':'manifest section counts mismatch','canonical_id':summary['canonical_id']})
    else:errors.append({'category':'structural','error':'missing refinement manifest'})
    summary={'expected_cases':len(cases),'audited_cases':len(rows),'cases_with_errors':sum(not r['mechanical_checks_pass'] for r in rows)+len(errors),
             'structural_errors':sum(r['structural_error_count'] for r in rows)+len(errors),'numeric_errors':sum(r['numeric_error_count'] for r in rows),
             'accepted_float_roundoff_checks':sum(r['accepted_float_roundoff_checks'] for r in rows),
             'normal_masked_sections':int(sum(sum(n for k,n in r['refinement_counts'].items() if k.startswith('automatically_withheld')) for r in rows)),
             'cases_with_final_probe_verification':sum(r['final_verification']['available'] for r in rows),
             'retained_recoveries_with_final_shift_evidence':sum(r['final_verification'].get('retained_recoveries_with_final_shift_evidence',0) for r in rows),
             'all_mechanical_checks_pass':not errors and all(r['mechanical_checks_pass'] for r in rows) and len(rows)==len(cases)}
    return {'schema':SCHEMA,'source':str(source.resolve()),'candidate':str(candidate.resolve()),'code_root':str((code_root or Path(__file__).parent).resolve()),'audit_code_sha256':sha(Path(__file__)),'tolerances':{'atol':ATOL,'rtol':RTOL,'plane_mm':PLANE_ATOL},
            'interpretation':'mechanical consistency only; explicit missing geometry is normal masking and never a human-decision flag; anatomical acceptance and training approval are not inferred',
            'summary':summary,'errors':errors,'cases':rows}


def main():
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--source',type=Path,required=True)
    ap.add_argument('--candidate','--root','--output',dest='candidate',type=Path,required=True)
    ap.add_argument('--split',type=Path,required=True);ap.add_argument('--cases',nargs='+');ap.add_argument('--out',type=Path);ap.add_argument('--code-root',type=Path,help='directory containing the three frozen refinement Python source files')
    a=ap.parse_args();sp=json.loads(a.split.read_text());cases=set(sp['test_cases'])|set(sp['train_cases'])
    if a.cases:
        if not set(a.cases)<=cases:ap.error('cases outside active split')
        cases=set(a.cases)
    result=run_audit(a.source,a.candidate,cases,a.split,a.code_root)
    out=a.out or a.candidate/'audit_refined_geometry.json';out.parent.mkdir(parents=True,exist_ok=True)
    out.write_text(json.dumps(result,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
    print(json.dumps(result['summary'],ensure_ascii=False),flush=True)
    if not result['summary']['all_mechanical_checks_pass']:raise SystemExit(1)

if __name__=='__main__':main()

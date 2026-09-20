"""Analytic oracles for the independent refinement audit, without real data."""
import numpy as np
from wss_v5.audit_refined_geometry import (Recorder, independent_interpolation,
                                         independent_longitudinal, polygon_measure)


def test_unequal_station_spacing_exact_quadratic_log_area():
    s=np.array([0.,.4,1.7,4.,7.5])
    loga=.013*s*s-.2*s+1.
    out=independent_longitudinal(s,np.zeros(len(s)),np.exp(loga),np.ones(len(s),bool))
    assert np.allclose(out['area_slope_per_mm'],.026*s-.2,atol=1e-13)


def test_segment_and_invalid_gap_never_enter_derivatives_or_context():
    s=np.array([0.,1.,3.,4.,8.,9.,11.,0.,2.])
    sid=np.array([0]*7+[1]*2)
    v=np.array([1,1,1,0,1,1,1,1,1],bool)
    # The two sides of the gap have unrelated log-area offsets.
    area=np.exp(np.array([0.,1.,3.,100.,-5.,-4.,-2.,20.,24.]))
    out=independent_longitudinal(s,sid,area,v)
    assert np.allclose(out['area_slope_per_mm'][[0,1,2,4,5,6]],1.)
    assert np.isnan(out['area_slope_per_mm'][[3,7,8]]).all()
    assert np.isnan(out['upstream_min_area_ratio'][[0,3,4,7]]).all()
    assert np.isnan(out['downstream_min_area_ratio'][[2,3,6,8]]).all()


def test_context_strict_direction_nearest_ties_and_window_boundary():
    s=np.array([0.,2.,5.,10.,12.,15.])
    area=np.array([1.,1.,8.,9.,1.,1.])
    out=independent_longitudinal(s,np.zeros(6),area,np.ones(6,bool))
    assert out['upstream_min_distance_mm'][3]==8.  # nearest of 0 and 2
    assert out['downstream_min_distance_mm'][2]==7. # nearest of 12 and 15
    assert out['downstream_min_area_ratio'][4]==1. # excludes current, retains equal area
    assert out['upstream_min_distance_mm'][5]==3.  # 2 is outside ten mm


def test_interpolation_exact_endpoints_missing_brackets_and_unequal_s():
    s=np.array([0.,2.,5.,9.]);value=2*s+3
    q=np.array([-1.,0.,1.,2.,4.,5.,8.,9.,10.])
    actual,valid=independent_interpolation(q,s,value,np.ones(4,bool))
    assert np.array_equal(valid,(q>=0)&(q<=9))
    assert np.allclose(actual[valid],2*q[valid]+3)
    _,bad=independent_interpolation(q,s,value,np.array([1,0,1,1],bool))
    assert not bad[(q>=0)&(q<5)].any()
    assert bad[q==5].all() and bad[q==9].all()


def test_rotated_polygon_integrals_and_self_crossing():
    center=np.array([120.,-30.,50.]);normal=np.array([0.,1.,1.])/np.sqrt(2)
    u=np.array([1.,0.,0.]);v=np.cross(normal,u)
    xy=np.array([[-2.,-1.],[2.,-1.],[2.,1.],[-2.,1.]])
    poly=center+xy[:,0,None]*u+xy[:,1,None]*v
    m=polygon_measure(poly,center,normal)
    assert np.isclose(m['area_mm2'],8.) and np.isclose(m['perimeter_mm'],12.)
    assert np.allclose(m['centroid_mm'],center)
    assert m['contains_center'] and not m['self_intersects']
    assert m['max_plane_residual_mm']<1e-10
    # Asymmetric bow tie has nonzero signed area, so intersection detection
    # cannot be accidentally satisfied by the zero-area rejection alone.
    bow=np.array([[-2.,-1.],[2.,1.],[-2.,1.],[1.,-1.]])
    m=polygon_measure(center+bow[:,0,None]*u+bow[:,1,None]*v,center,normal)
    assert m['self_intersects']


def test_mask_leakage_is_structural_while_roundoff_is_accepted():
    check=Recorder();check.compare(np.array([1.,np.nan]),np.array([1.+1e-6,3.]),'field')
    assert len(check.errors)==1 and check.errors[0]['category']=='structural'
    assert check.errors[0]['finite_mask_mismatches']==1
    assert len(check.roundoff)==1


def test_opening_extension_requires_geometric_evidence_and_preserves_core():
    from copy import deepcopy
    from wss_v5.audit_refined_geometry import audit_opening_domain
    xyz=np.array([[0.,0.,-2.],[0.,0.,-1.],[0.,0.,1.],[0.,0.,3.]])
    original=np.ones(4,bool);external=np.array([1,1,0,0],bool)
    old={'centerline_audit_core_mask':original}
    z={'centerline_xyz_mm':xyz,'centerline_s_local_mm':np.array([0.,1.,3.,5.]),
       'centerline_segment_id':np.zeros(4,int),'centerline_opening_extension_mask':external,
       'centerline_original_audit_core_mask':original,'centerline_audit_core_mask':original&~external,
       'centerline_inside_reference':~external}
    segment={'segment_id':0,'parent_id':-1,'children':[1,2],'start_s_mm':0.,'end_s_mm':5.}
    opening={'segment_id':0,'normal':[0.,0.,-1.],'center_mm':[0.,0.,0.],
             'polygon_xyz_mm':[[-1.,-1.,0.],[1.,-1.,0.],[1.,1.,0.],[-1.,1.,0.]]}
    domain={'excluded_atlas_rows':[0,1],'excluded_points':2,'segment_s_bounds_mm':{'0':[2.,5.]},
            'extensions':[{'segment_id':0,'endpoint':'start','atlas_rows':[0,1],
            'first_interior_atlas_row':2,'outward_offsets_mm':[2.,1.],
            'rim_plane_residual_max_mm':0.,'crossing_s_mm':2.,'crossing_xyz_mm':[0.,0.,0.]}]}
    report={'sampling_domain':domain,'openings':[opening],
            'segments':[{**segment,'start_s_mm':2.}],
            'P1_centerline':{'anatomy_domain':domain,'raw_outside_original_core_points':2,
                             'opening_extension_original_core_points':2}}
    ck=Recorder();audit_opening_domain(z,report,old,{'segments':[segment]},ck)
    assert not ck.errors
    bad=deepcopy(z);bad['centerline_xyz_mm'][0,0]=3.
    ck=Recorder();audit_opening_domain(bad,report,old,{'segments':[segment]},ck)
    assert any(e['check']=='extension_projects_inside_own_opening' for e in ck.errors)


def test_nonplanar_actual_cap_crossing_and_signed_offset():
    from wss_v5.audit_refined_geometry import independent_fan_center, independent_fan_line_hits
    polygon=np.array([[-1.,-1.,0.],[1.,-1.,0.],[1.,1.,.8],[-1.,1.,0.]])
    fan=independent_fan_center(polygon)
    assert fan is not None
    # A known barycentric point inside one actual nonplanar fan triangle.
    target=.2*fan+.4*polygon[1]+.4*polygon[2]
    direction=np.array([0.,0.,1.]);start=target+.1*direction;end=target-.3*direction
    hits=independent_fan_line_hits(start,end-start,polygon,fan)
    hits=hits[(hits>=0)&(hits<=1)]
    assert len(hits)==1 and np.isclose(hits[0],.25)
    offset=-max(independent_fan_line_hits(start,direction,polygon,fan))
    assert np.isclose(offset,.1)


def test_fin_proof_preserves_source_and_rejects_same_orientation_duplicate():
    from wss_v5.audit_refined_geometry import audit_fin_cleanup
    triangles=np.array([[0,1,2],[0,2,1],[0,1,3],[1,0,4]])
    # Fin pair shares edge 0--1 with two real faces; remaining pair edges vanish.
    cleanup={'removed_triangle_rows':[0,1],'removed_triangles':2,'frozen_source_modified':False,
             'pairs':[{'source_triangle_rows':[0,1],'edge_incidence_before':[4,2,2],
                       'edge_incidence_after':[2,0,0]}]}
    report={'P1_centerline':{'cap_audit':{'slicer_zero_thickness_fin_cleanup':cleanup}}}
    ck=Recorder();audit_fin_cleanup({'wall_triangles':triangles},report,ck)
    assert not ck.errors
    bad=triangles.copy();bad[1]=bad[0]
    ck=Recorder();audit_fin_cleanup({'wall_triangles':bad},report,ck)
    assert any(e['check']=='fin_pair_exact_opposite' for e in ck.errors)


def test_final_shift_prediction_uses_adjacent_log_profile_without_crossing_gap():
    from wss_v5.audit_refined_geometry import independent_shift_prediction
    s=np.array([0.,2.,5.,0.,3.]);sid=np.array([0,0,0,1,1]);area=np.exp(.1*s)
    valid=np.ones(5,bool)
    assert np.isclose(independent_shift_prediction(1,1.,s,sid,area,valid),np.exp(.3))
    assert np.isclose(independent_shift_prediction(1,-1.,s,sid,area,valid),np.exp(.1))
    valid[2]=False
    assert independent_shift_prediction(1,1.,s,sid,area,valid) is None
    assert independent_shift_prediction(2,1.,s,sid,area,np.ones(5,bool)) is None
    assert independent_shift_prediction(0,3.,s,sid,area,np.ones(5,bool)) is None


def test_final_recovery_shift_invalid_and_missing_evidence_cannot_pass():
    from copy import deepcopy
    from wss_v5.audit_refined_geometry import independent_recovery_probe_decision
    tilts=[{'valid':True,'actual_area_mm2':100.,'baseline_area_mm2':100.} for _ in range(4)]
    shifts=[{'valid':True,'actual_area_mm2':110.,'predicted_area_mm2':100.},
            {'valid':False,'actual_area_mm2':None,'predicted_area_mm2':None}]
    assert independent_recovery_probe_decision(shifts,tilts)['pass']
    bad=deepcopy(shifts);bad[1]['predicted_area_mm2']=100.
    assert not independent_recovery_probe_decision(bad,tilts)['pass']
    bad=deepcopy(shifts);bad[0]['predicted_area_mm2']=None
    assert not independent_recovery_probe_decision(bad,tilts)['pass']
    bad=deepcopy(tilts);bad[2]['actual_area_mm2']=121.
    assert not independent_recovery_probe_decision(shifts,bad)['pass']
    assert not independent_recovery_probe_decision(shifts,tilts[:3])['pass']


def test_final_shift_center_follows_bent_atlas_and_tilts_use_diagonal_phase():
    from wss_v5.audit_refined_geometry import independent_atlas_position, independent_diagonal_tilt_normal
    xyz=np.array([[0.,0.,0.],[1.,0.,0.],[1.,1.,0.]])
    got=independent_atlas_position(0,1.5,xyz,np.zeros(3,int),np.array([0.,1.,2.]))
    assert np.allclose(got,[1.,.5,0.])
    assert not np.allclose(got,[1.5,0.,0.])
    normal=independent_diagonal_tilt_normal(np.array([0.,0.,1.]),45.)
    expected=np.array([np.sin(np.radians(5))/np.sqrt(2)]*2+[np.cos(np.radians(5))])
    assert np.allclose(normal,expected)
    assert not np.allclose(normal,independent_diagonal_tilt_normal(np.array([0.,0.,1.]),0.))


def final_verification_fixture():
    from wss_v5.audit_refined_geometry import independent_diagonal_tilt_normal
    s=np.array([0.,2.,4.,6.]);normal=np.tile([0.,0.,1.],(4,1))
    normal[1]=[np.sin(np.radians(10.)),0.,np.cos(np.radians(10.))]
    centers=np.column_stack([s,np.zeros((4,2))])
    z={'section_s_local_mm':s,'section_segment_id':np.zeros(4,int),'section_tangent':normal,
       'section_center_mm':centers,'centerline_xyz_mm':centers,'centerline_s_local_mm':s,
       'centerline_segment_id':np.zeros(4,int),'section_area_mm2':np.full(4,100.),
       'section_valid':np.array([1,1,0,1],bool),'refinement_normal_change_deg':np.array([0.,10.,0.,0.]),
       'final_probe_profile_area_mm2':np.full(4,100.),'final_probe_profile_valid':np.ones(4,bool),
       'final_probe_recovered_mask':np.array([0,1,0,0],bool),
       'refinement_status':np.array(['stable_original','recovered_stable_plane','automatically_withheld_final_shift','stable_original'])}
    def probe(k,delta,actual=100.):
        return {'delta_mm':delta,'query_s_mm':s[k]+delta,'center_mm':[s[k]+delta,0.,0.],
                'normal':normal[k].tolist(),'predicted_area_mm2':100.,'actual_area_mm2':actual,
                'valid':True,'reason':'ok','relative_residual':abs(actual/100.-1),
                'neighbor_index':k+(1 if delta>0 else -1),'neighbor_area_mm2':100.}
    records=[{'section_index':k,'segment_id':0,'s_mm':s[k],'was_recovered':k==1,
              'iteration':1 if k==2 else 2,'stable':k!=2,
              'reason':'final_shift_failed' if k==2 else 'no_contiguous_shift_bracket' if k==3 else 'verified',
              'shift':[probe(0,1.)] if k==0 else [probe(1,-1.)] if k==1 else
                      [probe(2,-1.,160.),probe(2,1.)] if k==2 else []} for k in range(4)]
    tilt={'stable':True,'baseline_area_mm2':100.,'all_valid':True,'n_valid':4,
          'max_relative_change':0.,'reasons':{},'probes':[
              {'azimuth_deg':az,'tilt_deg':5.,'normal':independent_diagonal_tilt_normal(normal[1],az).tolist(),
               'center_mm':centers[1].tolist(),'valid':True,'reason':'ok','area_mm2':100.,'relative_change':0.}
              for az in [45.,135.,225.,315.]]}
    ref={'schema':'wss_v6_geometry_refined_v3','policy':{'shift_mm':1.,'shift_residual_limit':.25,
         'tilt_probe_deg':5.,'tilt_relative_limit':.20,'holdout_azimuth_offset_deg':45.,
         'recovery_minimum_final_shift_probes':1},'sections':[{'section_index':1,'new_holdout_tilt':tilt}],
         'final_verification':{'rounds':2,'sections':records}}
    return z,ref,probe


def test_final_verification_replays_rejection_round_and_rejects_stale_neighbor():
    from copy import deepcopy
    from wss_v5.audit_refined_geometry import audit_final_verification
    z,ref,probe=final_verification_fixture();ck=Recorder()
    summary=audit_final_verification(z,ref,ck)
    assert not ck.errors
    assert summary['shift_rejected']==1 and summary['retained_recoveries_with_final_shift_evidence']==1
    bad=deepcopy(ref);bad['final_verification']['sections'][1]['shift'].append(probe(1,1.))
    ck=Recorder();audit_final_verification(z,bad,ck)
    assert any(e['check']=='final_shift_predictable_directions' for e in ck.errors)


def test_final_verification_rejects_search_phase_tilt_reused_as_holdout():
    from wss_v5.audit_refined_geometry import audit_final_verification, independent_diagonal_tilt_normal
    z,ref,_=final_verification_fixture()
    ref['sections'][0]['new_holdout_tilt']['probes'][0]['normal']=independent_diagonal_tilt_normal(z['section_tangent'][1],0.).tolist()
    ck=Recorder();audit_final_verification(z,ref,ck)
    assert any(e['check'].startswith('holdout_normal:') for e in ck.errors)


def test_reference_dependency_blocks_old_valid_cloud_without_reference():
    from copy import deepcopy
    from wss_v5.audit_refined_geometry import audit_pointcloud_reference_dependency
    old={'pc_section_valid':np.array([1,1,0,0],bool)}
    z={'section_valid':np.array([0,1,1,0],bool),'pc_section_valid':np.array([0,1,1,0],bool),
       'pc_refinement_status':np.array(['automatically_withheld_reference','density_verified','density_verified','previously_invalid']),
       'final_probe_recovered_mask':np.array([0,0,1,0],bool),
       'refinement_status':np.array(['previously_invalid','stable_original','recovered_stable_plane','previously_invalid']),
       'pc_refinement_density_relative_change':np.array([np.nan,.01,.02,np.nan]),
       'pc_section_reason':np.array(['refinement_reference_unverified','ok','ok','bad']),
       'section_reason':np.array(['no_closed_contour_containing_center','ok','ok','bad'])}
    ref={'policy':{'pointcloud_requires_verified_reference':True},'sections':[
        {'source':'pointcloud','status':'automatically_withheld_reference','section_index':0,
         'reference_reason':'no_closed_contour_containing_center'}]}
    ck=Recorder();audit_pointcloud_reference_dependency(z,old,ref,ck)
    assert not ck.errors
    bad=deepcopy(z);bad['pc_section_valid'][0]=True
    ck=Recorder();audit_pointcloud_reference_dependency(bad,old,ref,ck)
    assert any(e['check']=='pointcloud_valid_requires_verified_reference' for e in ck.errors)

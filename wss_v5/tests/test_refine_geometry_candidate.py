"""Regression tests for retained geometry, gaps, and physical probe units."""
import numpy as np
from wss_v5.refine_geometry_candidate import (atlas_center, contiguous_prediction,
                                            normals, remap_fields, verify_final_shifts)
from wss_v5.section_features import SurfaceSlicer
from wss_v5.tests.test_geometry_candidate import tube


def test_shift_explanation_preserves_true_exponential_area_gradient():
    # A strong, continuous stenosis is not a discontinuity just because two
    # stations differ by more than 50 percent.
    s=np.arange(0.,10.,2.);a=np.exp(.5*s);v=np.ones(len(s),bool);seg=np.zeros(len(s),int)
    assert np.isclose(contiguous_prediction(2,5.,s,seg,a,v),np.exp(2.5))
    v[3]=False
    assert contiguous_prediction(2,5.,s,seg,a,v) is None
    v[3]=True;seg[3]=1
    assert contiguous_prediction(2,5.,s,seg,a,v) is None


def test_orthogonal_tilt_probes_match_analytical_cylinder_area():
    xyz,tri=tube(radius=2.,length=20.,n=360)
    cutter=SurfaceSlicer(xyz,tri)
    for t in normals(np.array([0.,0.,1.]),5.):
        result=cutter.slice(np.array([0.,0.,10.]),t)
        assert result['valid']
        assert abs(result['area_mm2']/(4*np.pi/np.cos(np.radians(5.)))-1)<1e-4


def test_rejected_slice_propagates_to_all_context_and_wall_masks():
    n=9;s=np.arange(n,dtype=float)
    z={'section_segment_id':np.zeros(n,int),'section_s_local_mm':s,
       'wall_xyz_mm':np.zeros((8,3)),'wall_map_segment_id':np.zeros(8,int),
       'wall_map_s_local_mm':np.arange(.5,8.,1.),'wall_map_ambiguous':np.zeros(8,bool)}
    for p in ['section_','pc_section_']:
        z[p+'area_mm2']=np.exp(.1*s)
        z[p+'valid']=np.ones(n,bool);z[p+'valid'][4]=False
        z[p+'roundness']=np.full(n,.9);z[p+'eccentricity']=np.full(n,.1)
    remap_fields(z)
    for w in ['wall_map_','wall_map_pc_']:
        assert not z[w+'valid'][3:5].any()
        assert np.isnan(z[w+'area_slope_per_mm'][3:5]).all()
        assert np.isnan(z[w+'upstream_min_area_ratio'][4:6]).all()
        assert np.isnan(z[w+'downstream_min_area_ratio'][2:4]).all()
        assert np.allclose(z[w+'area_slope_per_mm'][[0,1,2,5,6,7]],.1,atol=1e-6)


def test_final_shift_rejects_failed_recovery_and_rechecks_its_neighbors():
    s=np.arange(0.,10.,2.);seg=np.zeros(5,int);area=np.full(5,100.)
    v=np.ones(5,bool);recovered=np.array([False,True,True,True,False])
    def probe(k,q):
        # Station 2 must fail even if the orientation-search tilts passed.
        return {'valid':True,'actual_area_mm2':200. if k==2 else 100.,'reason':'ok'}
    kept,evidence=verify_final_shifts(s,seg,area,v,recovered,probe)
    assert kept.tolist()==[True,True,False,True,True]
    assert evidence['rounds']==2
    records={r['section_index']:r for r in evidence['sections']}
    assert records[2]['reason']=='final_shift_failed'
    assert len(records[1]['shift'])==1 and records[1]['shift'][0]['neighbor_index']==0
    assert len(records[3]['shift'])==1 and records[3]['shift'][0]['neighbor_index']==4


def test_missing_bracket_is_not_shift_pass_or_recursive_original_deletion():
    s=np.arange(0.,6.,2.);seg=np.zeros(3,int);area=np.full(3,100.)
    def probe(k,q): return {'valid':False,'actual_area_mm2':None,'reason':'bad_loop'}
    kept,evidence=verify_final_shifts(s,seg,area,np.array([True,False,True]),
                                     np.array([True,False,False]),probe)
    assert kept.tolist()==[False,False,True]
    assert evidence['sections'][0]['reason']=='recovery_missing_contiguous_shift_evidence'
    assert evidence['sections'][1]['reason']=='no_contiguous_shift_bracket'
    kept,evidence=verify_final_shifts(s,seg,area,np.ones(3,bool),np.zeros(3,bool),probe)
    assert not kept.any()  # Invalid probed geometry is not a passing residual.


def test_final_shift_uses_atlas_arclength_and_holdout_azimuths_are_distinct():
    xyz=np.array([[0.,0.,0.],[2.,0.,0.],[2.,2.,0.]])
    assert np.allclose(atlas_center(3.,0,xyz,np.zeros(3,int),np.array([0.,2.,4.])),[2.,1.,0.])
    t=np.array([0.,0.,1.]);search=normals(t,5.);holdout=normals(t,5.,offset_deg=45.)
    assert all(not np.allclose(a,b) for a in search for b in holdout)

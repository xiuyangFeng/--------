import numpy as np
from wss_v5.coverage_geometry_candidate import exact_other_branch_hits, exact_gate
from wss_v5.section_features import polygon_metrics


def circle():
    t=np.linspace(0,2*np.pi,128,endpoint=False)
    return polygon_metrics(np.column_stack((2*np.cos(t),2*np.sin(t),np.zeros(len(t)))),np.zeros(3),np.array([0.,0.,1.]))


def test_nearby_branch_that_does_not_cross_plane_is_not_mixed_lumen():
    xyz=np.array([[0,0,-2],[0,0,2],[1,0,.2],[1,0,.5]],float)
    sid=np.array([0,0,1,1]);s=np.array([0.,4.,0.,.3])
    assert not exact_other_branch_hits(circle(),np.zeros(3),np.array([0,0,1.]),xyz,sid,s,0)


def test_continuous_crossing_between_far_samples_is_detected():
    xyz=np.array([[0,0,-2],[0,0,2],[1,0,-2],[1,0,2]],float)
    sid=np.array([0,0,1,1]);s=np.array([0.,4.,0.,4.])
    hits=exact_other_branch_hits(circle(),np.zeros(3),np.array([0,0,1.]),xyz,sid,s,0)
    assert len(hits)==1 and hits[0]['segment_id']==1
    assert np.allclose(hits[0]['points_xyz_mm'],[[1,0,0]])


def test_coplanar_branch_crossing_polygon_without_vertex_inside_is_rejected():
    xyz=np.array([[0,0,-2],[0,0,2],[-3,0,0],[3,0,0]],float)
    sid=np.array([0,0,1,1]);s=np.array([0.,4.,0.,6.])
    assert exact_other_branch_hits(circle(),np.zeros(3),np.array([0,0,1.]),xyz,sid,s,0)


def test_exact_gate_keeps_endpoint_and_own_branch_checks():
    xyz=np.array([[0,0,0],[0,0,10]],float);sid=np.zeros(2,int);s=np.array([0.,10.])
    seg={'segment_id':0,'start_s_mm':0.,'end_s_mm':10.,'parent_id':-1,'children':[]}
    cfg={'junction_exclusion_mm':2.5,'endpoint_exclusion_mm':1.,'same_segment_nonlocal_s_gap_mm':5.,'plane_intersection_tolerance_mm':1e-5}
    out=exact_gate(circle(),np.zeros(3),np.array([0,0,1.]),seg,0.,xyz,sid,cfg,s)
    assert not out['valid'] and 'endpoint_exclusion' in out['reason']

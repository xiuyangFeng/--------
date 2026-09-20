import numpy as np
from wss_v5.audit_coverage_geometry import independent_other_branch_hits,point_in_polygon,line_hits_polygon

POLYGON=np.array([[-1.,-1.,0.],[1.,-1.,0.],[1.,1.,0.],[-1.,1.,0.]])

def test_near_plane_parallel_samples_are_not_true_crossings():
    xyz=np.array([[-2.,0.,.1],[2.,0.,.1]])
    assert not independent_other_branch_hits(POLYGON,np.zeros(3),np.array([0.,0.,1.]),xyz,np.ones(2,int),np.array([0.,4.]),0)


def test_actual_crossing_is_detected_even_without_near_plane_samples():
    xyz=np.array([[0.,0.,-5.],[0.,0.,5.]])
    hits=independent_other_branch_hits(POLYGON,np.zeros(3),np.array([0.,0.,1.]),xyz,np.ones(2,int),np.array([0.,10.]),0)
    assert len(hits)==1 and np.allclose(hits[0]['points_xyz_mm'],[[0.,0.,0.]])


def test_coplanar_edge_crossing_is_detected_when_both_endpoints_outside():
    xyz=np.array([[-2.,0.,0.],[2.,0.,0.]])
    hits=independent_other_branch_hits(POLYGON,np.zeros(3),np.array([0.,0.,1.]),xyz,np.ones(2,int),np.array([0.,4.]),0)
    assert len(hits)==1 and hits[0]['coplanar']


def test_edge_outside_polygon_and_own_branch_are_not_false_hits():
    outside=np.array([[3.,0.,-5.],[3.,0.,5.]])
    assert not independent_other_branch_hits(POLYGON,np.zeros(3),np.array([0.,0.,1.]),outside,np.ones(2,int),np.array([0.,10.]),0)
    inside=np.array([[0.,0.,-5.],[0.,0.,5.]])
    assert not independent_other_branch_hits(POLYGON,np.zeros(3),np.array([0.,0.,1.]),inside,np.zeros(2,int),np.array([0.,10.]),0)


def test_polygon_boundary_and_collinear_overlap_are_retained_as_crossings():
    p=POLYGON[:,:2]
    assert point_in_polygon(np.array([1.,.5]),p)
    assert line_hits_polygon(np.array([-2.,1.]),np.array([2.,1.]),p)
    assert not line_hits_polygon(np.array([-2.,1.2]),np.array([2.,1.2]),p)

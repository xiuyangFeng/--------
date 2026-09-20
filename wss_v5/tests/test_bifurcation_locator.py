import numpy as np
from wss_v5.bifurcation_locator import _record, _scan
from wss_v5.section_features import SurfaceSlicer


def square(cx, cy, r=2):
    return np.array([[cx-r, cy-r, 0], [cx+r, cy-r, 0], [cx+r, cy+r, 0],
                     [cx-r, cy+r, 0], [cx-r, cy-r, 0]], float)


class FakeSlicer:
    def __init__(self, loops):
        self.loops = [{"polygon_xyz_mm": p} for p in loops]

    def contours(self, center, normal):
        from wss_v5.section_features import polygon_metrics
        return [polygon_metrics(x["polygon_xyz_mm"], center, normal) for x in self.loops]


def test_one_contour_contains_both_daughters():
    r = _record(FakeSlicer([square(0, 0, 5)]), np.zeros(3), [0, 0, 1],
                np.array([[-1, 0, 0], [1, 0, 0]]))
    assert r["state"] == "one"


def test_two_daughters_must_be_in_distinct_contours():
    r = _record(FakeSlicer([square(-2, 0, 1), square(2, 0, 1), square(30, 30, 2)]),
                np.zeros(3), [0, 0, 1], np.array([[-2, 0, 0], [2, 0, 0]]))
    assert r["state"] == "two"
    assert len(r["daughter_contour_ids"]) == 2


def test_unrelated_third_contour_does_not_trigger_two():
    r = _record(FakeSlicer([square(0, 0, 5), square(30, 30, 2)]), np.zeros(3), [0, 0, 1],
                np.array([[-1, 0, 0], [1, 0, 0]]))
    assert r["state"] == "one"


def test_missing_or_nonfinite_daughter_points_are_ambiguous():
    r = _record(FakeSlicer([square(0, 0, 5)]), np.zeros(3), [0, 0, 1],
                np.empty((0, 3)))
    assert r["state"] == "ambiguous"


def tube_mesh(z0, z1, centers, radius, n=24):
    """Open triangulated 3-D tubes for testing real VTK plane cutting."""
    rings = []
    for z, (cx, cy) in zip(np.linspace(z0, z1, 7), centers):
        rings.append(np.array([[cx + radius*np.cos(a), cy + radius*np.sin(a), z]
                               for a in np.linspace(0, 2*np.pi, n, endpoint=False)]))
    xyz = np.concatenate(rings)
    tri = []
    for k in range(len(rings)-1):
        for i in range(n):
            j=(i+1)%n; a=k*n+i; b=k*n+j; c=(k+1)*n+i; d=(k+1)*n+j
            tri.extend([[a,b,c],[b,d,c]])
    return xyz, np.asarray(tri, dtype=np.int64)


def test_real_surface_slicer_y_split_and_unrelated_third_loop():
    trunk_xyz, trunk_tri = tube_mesh(-4, 1, [(0, 0)]*7, 2.0)
    left_xyz, left_tri = tube_mesh(1, 5, [(-1.3, 0)]*7, .65)
    right_xyz, right_tri = tube_mesh(1, 5, [(1.3, 0)]*7, .65)
    third_xyz, third_tri = tube_mesh(1, 5, [(15, 15)]*7, .6)
    xyz=np.concatenate([trunk_xyz,left_xyz,right_xyz,third_xyz]); tri=[]; off=0
    for t, q in ((trunk_tri,0),(left_tri,len(trunk_xyz)),(right_tri,len(trunk_xyz)+len(left_xyz)),
                 (third_tri,len(trunk_xyz)+len(left_xyz)+len(right_xyz))): tri.append(t+q)
    slicer=SurfaceSlicer(xyz,np.concatenate(tri))
    assert len(slicer.contours(np.array([0,0,0.]), [0,0,1])) == 1
    # The unrelated third tube makes three loops, but cannot affect daughter
    # ownership because both daughter probes remain in the two local loops.
    loops=slicer.contours(np.array([0,0,2.]), [0,0,1])
    assert len(loops) == 3
    r=_record(slicer,np.array([0,0,2.]),[0,0,1],np.array([[-1.3,0,2],[1.3,0,2]]))
    assert r["state"] == "two"


def test_four_two_samples_do_not_meet_one_mm_persistence():
    class ChangingSlicer:
        def contours(self, center, normal):
            z=float(center[2])
            return ([{"polygon_xyz_mm": square(-1,0,1)}, {"polygon_xyz_mm": square(1,0,1)}]
                    if z >= 2 else [{"polygon_xyz_mm": square(0,0,4)}])
    # Directly validate the physical persistence conversion used by _scan.
    xyz=[]; sid=[]; ss=[]
    for seg,x in ((1,-1),(2,1)):
        for s in np.arange(0,4.01,.25):
            xyz.append([x,0,s+1]); sid.append(seg); ss.append(s)
    out=_scan(np.asarray(xyz,float),np.asarray(sid),np.asarray(ss,float),0,[1,2],ChangingSlicer(),step=.25)
    assert out["persistence_samples"] == 5


def test_real_mesh_scan_finds_local_transition():
    parts = [tube_mesh(-4,1,[(0,0)]*7,2),
             tube_mesh(1,5,[(-1.3,0)]*7,.65),
             tube_mesh(1,5,[(1.3,0)]*7,.65),
             tube_mesh(-4,5,[(15,15)]*7,.6)]
    vertices=[]; triangles=[]; count=0
    for p,t in parts:
        vertices.append(p);triangles.append(t+count);count+=len(p)
    slicer=SurfaceSlicer(np.concatenate(vertices),np.concatenate(triangles))
    arc=np.arange(0,6.01,.25)
    xyz=np.concatenate([np.column_stack((np.full(len(arc),x),np.zeros(len(arc)),arc-1)) for x in (-1.3,1.3)])
    result=_scan(xyz,np.repeat([1,2],len(arc)),np.tile(arc,2),0,[1,2],slicer)
    assert result['candidate'] is not None
    assert result['candidate']['bracket_mm'][0] <= 2 <= result['candidate']['bracket_mm'][1]
    for r in result['records']:
        if len(r['daughter_points_mm']):
            assert np.max(np.abs((r['daughter_points_mm']-r['center_mm'])@r['normal']))<1e-6


def test_persistence_and_preceding_shared_lumen_are_required():
    arc=np.arange(0,4.01,.25)
    xyz=np.concatenate([np.column_stack((np.full(len(arc),x),np.zeros(len(arc)),arc)) for x in (-1.3,1.3)])
    class SliceSequence:
        def __init__(self,start): self.start=start
        def contours(self,center,normal):
            polys=[square(-1.3,0,.6),square(1.3,0,.6)] if center[2]>=self.start else [square(0,0,3)]
            return [{'polygon_xyz_mm':p+np.array([0,0,center[2]])} for p in polys]
    args=(xyz,np.repeat([1,2],len(arc)),np.tile(arc,2),0,[1,2])
    four=_scan(*args,SliceSequence(2.25))
    assert sum(r['state']=='two' for r in four['records'])==4
    assert four['candidate'] is None
    five=_scan(*args,SliceSequence(2.0))
    assert five['candidate']['s_mm']==2.0
    assert five['candidate']['bracket_mm']==[1.75,2.0]
    no_prior=_scan(*args,SliceSequence(-1))
    assert no_prior['candidate'] is None
    assert no_prior['status']=='unconfirmed_no_preceding_one'


def test_non_coplanar_and_nan_probes_are_not_projected_into_lumen():
    for probes in (np.array([[0,0,1],[1,0,0]]),np.array([[np.nan,0,0],[1,0,0]])):
        r=_record(FakeSlicer([square(0,0,5)]),np.zeros(3),[0,0,1],probes)
        assert r['state']=='ambiguous'

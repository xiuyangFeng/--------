import numpy as np
from wss_v5.pointcloud_contours import pointcloud_contour_fallback


def tube_points(n=6000, seed=4, half=False, radius=9.):
    rng=np.random.default_rng(seed);theta=rng.uniform(0,2*np.pi,n);z=rng.uniform(-4,4,n)
    # Smooth non-star, mildly concave silhouette with dense longitudinal wall.
    r=radius*(1+0.35*np.cos(2*theta))
    p=np.c_[r*np.cos(theta),.72*r*np.sin(theta),z]
    p+=rng.normal(0,.025,p.shape)
    return p[rng.random(n)>.5] if half else p


def test_nonstar_crust_recovers_one_closed_loop_without_radial_or_hull():
    p=tube_points();out=pointcloud_contour_fallback(p,[0,0,1],[0,0,1],12.,.35)
    assert out['method'].startswith('local_surface_projection_planar_crust')
    assert out['reference_geometry_consumed'] is False
    assert out['topology']['path_completion_performed'] is False
    assert out['topology']['closed_components']==1
    assert out['valid'] and out['area_mm2']>0 and len(out['polygon_xyz_mm'])>=24


def test_half_density_is_explicitly_supported_but_never_promoted_without_comparison():
    full=tube_points(seed=8);half=tube_points(seed=8,half=True)
    a=pointcloud_contour_fallback(full,[0,0,1],[0,0,1],12.,.35)
    b=pointcloud_contour_fallback(half,[0,0,1],[0,0,1],12.,.35)
    assert a['requires_independent_half_density_verification'] is True
    assert b['requires_independent_half_density_verification'] is True
    assert a['valid'] and b['valid']
    assert abs(b['area_mm2']/a['area_mm2']-1)<.20


def test_adjacent_two_branches_are_not_joined_by_path_completion():
    rng=np.random.default_rng(12);parts=[]
    for shift in [0.,16.]:
        n=3000;theta=rng.uniform(0,2*np.pi,n);z=rng.uniform(-4,4,n)
        r=5.;parts.append(np.c_[shift+r*np.cos(theta),r*np.sin(theta),z])
    out=pointcloud_contour_fallback(np.concatenate(parts),[0,0,1],[0,0,1],8.,.35)
    assert out['valid'] is True
    assert out['topology']['path_completion_performed'] is False
    assert out['topology']['closed_components'] == 2
    assert np.max(out['selected_sample_xyz_mm'][:,0]) < 8.


def test_sparse_or_gap_sampling_fails_closed_and_does_not_close_the_gap():
    p=tube_points(n=2000,seed=13)
    # remove an angular sector from every z slice
    theta=np.arctan2(p[:,1]/.72,p[:,0]);p=p[(theta<2.4)|(theta>3.2)]
    out=pointcloud_contour_fallback(p,[0,0,1],[0,0,1],12.,.35)
    assert out['valid'] is False
    assert out['topology']['path_completion_performed'] is False

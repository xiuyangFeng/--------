"""Non-star point-cloud sections from a locally supported planar curve crust.

This fallback consumes coordinates only. Local surface PCA extrapolates nearby
wall samples onto the requested plane; a 2-D Voronoi/Delaunay crust reconstructs
sample-to-sample curve edges without radial ordering or a convex hull. The
algorithm never connects open paths to manufacture a loop. Every component
must already be a sampled closed curve, and every retained edge must have local
surface/tangent support. Close branches or missing samples therefore fail
conservatively rather than becoming an artificial combined lumen.

Use only after the established radial method or its independent density check
fails. The caller must keep the original branch/endpoint gates and rerun this
same fallback on a half-density input before accepting a recovered contour.
A valid return is a reconstruction candidate, not anatomical/clinical approval.
"""
from __future__ import annotations
from collections import Counter
import numpy as np
from scipy.spatial import cKDTree, Delaunay, QhullError, Voronoi

METHOD = 'local_surface_projection_planar_crust_v1'
POLICY = {
    'normal_neighbors': 16, 'minimum_normal_neighbors': 6,
    'normal_support_radius_spacing': 4., 'maximum_pca_normal_variance_ratio': .25,
    'minimum_transverse_normal': .4, 'minimum_projection_support_fraction': .85,
    'packing_radius_spacing': .8, 'maximum_edge_spacing': 4.,
    'maximum_edge_support_distance_spacing': 1.25,
    'maximum_edge_normal_component': .60,
    'minimum_cycle_vertices': 24,
    'no_path_completion': True,
    'input_dependency': 'point_coordinates_center_plane_radius_hint_spacing_only',
}


def _frame(normal):
    normal=np.asarray(normal,float);normal/=np.linalg.norm(normal)
    e=np.eye(3)[np.argmin(np.abs(normal))]
    first=np.cross(normal,e);first/=np.linalg.norm(first)
    return normal,first,np.cross(normal,first)


def _metrics(vertices,center,normal):
    """Standard closed-polygon integrals, compatible with pointcloud_section."""
    normal,u,v=_frame(normal);rel=vertices-center
    p=np.column_stack((rel@u,rel@v));q=np.roll(p,-1,axis=0)
    cr=p[:,0]*q[:,1]-q[:,0]*p[:,1];signed=.5*cr.sum();area=abs(signed)
    if area<=1e-9:return None
    cen=((p+q)*cr[:,None]).sum(axis=0)/(6*signed)
    perimeter=np.linalg.norm(q-p,axis=1).sum();req=np.sqrt(area/np.pi)
    angle=np.arctan2(cr,np.sum(p*q,axis=1)).sum()
    return {'valid':True,'reason':'ok','area_mm2':float(area),'perimeter_mm':float(perimeter),
            'req_mm':float(req),'roundness':float(4*np.pi*area/perimeter**2),
            'eccentricity':float(np.linalg.norm(cen)/req),
            'centroid_mm':center+cen[0]*u+cen[1]*v,
            'contains_center':bool(abs(angle)>np.pi),'polygon_xyz_mm':vertices}


def _crust_edges(points):
    """The planar crust: original-original edges after adding Voronoi vertices.

    Delaunay edges that could merely bridge a cavity are eliminated by the
    Voronoi vertices. Local-support gates further reject under-sampled edges.
    No long edges or endpoints are repaired after this construction.
    """
    # Translation/scaling avoid a dependence on the patient's coordinate origin.
    shift=points.mean(axis=0);scale=max(np.ptp(points,axis=0).max(),1e-9)
    p=(points-shift)/scale
    voronoi=Voronoi(p)
    extra=voronoi.vertices[np.isfinite(voronoi.vertices).all(axis=1)]
    triangles=Delaunay(np.concatenate((p,extra))).simplices
    edges=np.sort(np.concatenate((triangles[:,[0,1]],triangles[:,[1,2]],triangles[:,[2,0]])),axis=1)
    return np.unique(edges[np.all(edges<len(p),axis=1)],axis=0)


def _cycle_components(n,edges):
    neighbors=[[] for _ in range(n)]
    for a,b in edges:neighbors[a].append(int(b));neighbors[b].append(int(a))
    degree=np.asarray([len(row) for row in neighbors]);cycles=[]
    if np.any(degree!=2):return [],degree
    unvisited=set(range(n))
    while unvisited:
        start=min(unvisited);cycle=[start];previous=-1;current=start
        while True:
            candidate=next(j for j in neighbors[current] if j!=previous)
            if candidate==start:break
            if candidate in cycle:return [],degree
            cycle.append(candidate);previous,current=current,candidate
        unvisited.difference_update(cycle);cycles.append(np.asarray(cycle,int))
    return cycles,degree


def _simple_polygon(p):
    if len(p)<3:return False
    q=np.roll(p,-1,axis=0);i,j=np.triu_indices(len(p),2)
    ok=~((i==0)&(j==len(p)-1));i,j=i[ok],j[ok]
    def cross(a,b):return a[:,0]*b[:,1]-a[:,1]*b[:,0]
    d1=q[i]-p[i];d2=q[j]-p[j];delta=p[j]-p[i];den=cross(d1,d2)
    t=cross(delta,d2)/np.where(abs(den)>1e-14,den,np.nan)
    s=cross(delta,d1)/np.where(abs(den)>1e-14,den,np.nan)
    return not bool(np.any((t>1e-9)&(t<1-1e-9)&(s>1e-9)&(s<1-1e-9)))


def pointcloud_contour_fallback(points,center,normal,radius_hint,spacing_mm):
    """Return a geometry-only non-star reconstruction or an explicit failure.

    Parameters match ``pointcloud_section(points, center, normal, radius_hint,
    spacing_mm)``. ``points`` must contain enough axial neighborhoods for PCA;
    the caller may preselect a branch using point/centerline coordinates only.
    No surface triangles, CFD values, reference polygon/area or outcome enters
    this API. Radius is a reported hint, never an area target or hull radius.
    """
    out={'valid':False,'reason':'crust_insufficient_points','polygon_xyz_mm':np.empty((0,3)),
         'method':METHOD,'applicability':'sampled_simple_closed_curve_in_section_plane',
         'policy':dict(POLICY),'reference_geometry_consumed':False,
         'requires_independent_half_density_verification':True}
    points=np.asarray(points,float);center=np.asarray(center,float);normal=np.asarray(normal,float)
    if (points.ndim!=2 or points.shape[1:]!=(3,) or center.shape!=(3,) or normal.shape!=(3,)
            or not np.isfinite(points).all() or not np.isfinite(center).all()
            or not np.isfinite(normal).all() or np.linalg.norm(normal)<1e-9
            or not np.isfinite(spacing_mm) or spacing_mm<=0 or not np.isfinite(radius_hint) or radius_hint<=0):
        return {**out,'reason':'crust_invalid_input'}
    points=np.unique(points,axis=0)
    normal,u,v=_frame(normal);spacing=float(spacing_mm)
    slab=max(.75,min(2.,2.*spacing));axial=(points-center)@normal
    near=np.flatnonzero(np.abs(axial)<=slab)
    out.update(input_points=len(points),slab_points=len(near),slab_halfwidth_mm=slab,radius_hint_mm=float(radius_hint))
    if len(near)<POLICY['minimum_cycle_vertices'] or len(points)<POLICY['normal_neighbors']:return out
    tree=cKDTree(points);distance,ids=tree.query(points[near],k=POLICY['normal_neighbors'])
    neighborhoods=points[ids];local=neighborhoods-neighborhoods.mean(axis=1,keepdims=True)
    eigen,basis=np.linalg.eigh(np.einsum('nki,nkj->nij',local,local))
    surface_normal=basis[:,:,0];n_axial=surface_normal@normal
    n_plane=surface_normal-n_axial[:,None]*normal;transverse=np.linalg.norm(n_plane,axis=1)
    ratio=eigen[:,0]/np.maximum(eigen[:,1],1e-14)
    counts=np.sum(distance<=POLICY['normal_support_radius_spacing']*spacing,axis=1)
    supported=((counts>=POLICY['minimum_normal_neighbors'])
               &(distance[:,-1]<=POLICY['normal_support_radius_spacing']*spacing)
               &(ratio<=POLICY['maximum_pca_normal_variance_ratio'])
               &(transverse>=POLICY['minimum_transverse_normal']))
    out['projection_support']={'accepted':int(supported.sum()),'fraction':float(supported.mean()),
         'maximum_pca_variance_ratio':float(ratio.max()),'minimum_transverse_normal':float(transverse.min())}
    if supported.mean()<POLICY['minimum_projection_support_fraction']:
        return {**out,'reason':'crust_local_surface_support_insufficient'}
    sample=near[supported];n_plane=n_plane[supported];n_axial=n_axial[supported]
    transverse=transverse[supported]
    projected=(points[sample]-axial[sample,None]*normal
               +(axial[sample]*n_axial/transverse**2)[:,None]*n_plane)
    xy=np.column_stack(((projected-center)@u,(projected-center)@v))
    plane_normals=np.column_stack((n_plane@u,n_plane@v));plane_normals/=np.linalg.norm(plane_normals,axis=1,keepdims=True)
    # Prefer actual observations closest to the plane; fixed-distance packing
    # avoids fitting a second, density-dependent radial envelope.
    support_tree=cKDTree(xy);available=np.ones(len(xy),bool);chosen=[]
    order=np.lexsort((xy[:,1],xy[:,0],np.abs(axial[sample])))
    for i in order:
        if not available[i]:continue
        available[support_tree.query_ball_point(xy[i],POLICY['packing_radius_spacing']*spacing)]=False
        chosen.append(int(i))
    chosen=np.asarray(chosen,int);q=xy[chosen]
    out['packed_points']=len(q)
    if len(q)<POLICY['minimum_cycle_vertices']:return {**out,'reason':'crust_insufficient_packed_samples'}
    try:edges=_crust_edges(q)
    except QhullError:return {**out,'reason':'crust_planar_triangulation_degenerate'}
    lengths=np.linalg.norm(q[edges[:,1]]-q[edges[:,0]],axis=1)
    midpoint=(q[edges[:,1]]+q[edges[:,0]])/2
    support_distance,support_id=support_tree.query(midpoint,k=1)
    direction=(q[edges[:,1]]-q[edges[:,0]])/np.maximum(lengths[:,None],1e-12)
    tangent_error=np.abs(np.sum(direction*plane_normals[support_id],axis=1))
    keep=((lengths<=POLICY['maximum_edge_spacing']*spacing)
          &(support_distance<=POLICY['maximum_edge_support_distance_spacing']*spacing)
          &(tangent_error<=POLICY['maximum_edge_normal_component']))
    out['edge_support']={'crust_edges':len(edges),'accepted_edges':int(keep.sum()),
          'maximum_length_mm':float(lengths.max()) if len(lengths) else None,
          'maximum_midpoint_support_mm':float(support_distance.max()) if len(lengths) else None,
          'maximum_edge_normal_component':float(tangent_error.max()) if len(lengths) else None,
          'long_edges_rejected':int(np.sum(lengths>POLICY['maximum_edge_spacing']*spacing)),
          'unsupported_edges_rejected':int(np.sum(support_distance>POLICY['maximum_edge_support_distance_spacing']*spacing)),
          'transverse_edges_rejected':int(np.sum(tangent_error>POLICY['maximum_edge_normal_component']))}
    cycles,degree=_cycle_components(len(q),edges[keep])
    out['topology']={'degree_counts':{str(k):int(vv) for k,vv in Counter(degree).items()},
                     'closed_components':len(cycles),'path_completion_performed':False}
    if not cycles:return {**out,'reason':'crust_open_or_nonmanifold_sampling_graph'}
    candidates=[]
    for cycle in cycles:
        p=q[cycle]
        if not _simple_polygon(p):return {**out,'reason':'crust_self_intersection'}
        vertices=center+p[:,0,None]*u+p[:,1,None]*v
        metrics=_metrics(vertices,center,normal)
        if metrics is not None and metrics['contains_center']:
            if len(cycle)<POLICY['minimum_cycle_vertices']:return {**out,'reason':'crust_enclosing_loop_under_sampled'}
            candidates.append((cycle,metrics))
    if len(candidates)!=1:
        return {**out,'reason':'crust_no_unique_enclosing_loop','containing_components':len(candidates)}
    cycle,metrics=candidates[0]
    out.update(metrics)
    out['selected_sample_input_rows']=sample[chosen[cycle]]
    out['selected_sample_xyz_mm']=points[sample[chosen[cycle]]]
    out['selected_sample_plane_distance_mm']=axial[sample[chosen[cycle]]]
    out['topology']['containing_components']=1
    out['topology']['selected_cycle_vertices']=len(cycle)
    out['topology']['other_closed_components']=len(cycles)-1
    out['source']='bare_pointcloud_local_surface_projection_and_crust'
    return out

#!/usr/bin/env python3
"""Rebuild selected cached fields, native walls and Gaussian STL surfaces; no inference."""
from pathlib import Path
import argparse, csv, gzip, hashlib, json, sys
import h5py
import numpy as np
import vtk
from vtk.util.numpy_support import numpy_to_vtk, numpy_to_vtkIdTypeArray, vtk_to_numpy
from scipy.spatial import cKDTree

OUT = Path(__file__).resolve().parent
ROOT = next(p for p in OUT.parents if (p/'training_wss_min').is_dir())
sys.path.insert(0, str(ROOT))
from tools.cfdpost_cloud_export.map_to_stl_surface import _load_stl_mesh, _interp_gaussian
from tools.cfdpost_cloud_export.display_fields import fit_selfmax, selfmax_fields, normalization_field_data
SUMMARY = OUT.parent/'启发式v2_R2分布_20260914/summary.json'
DATA = ROOT/'data_wss_v5/views/wss_min_view_v1'
SELECT = {'X5X11': {'best':'AG/fast/ZHANG_LIANG', 'median':'AG/fast/YAO_CUN_HONG', 'worst':'AAA/unruputer/SUN_SHU_MING'},
          'VF6': {'best':'AG/fast/ZHANG_LIANG', 'median':'AG/slow/LI_HUAN_GE', 'worst':'AAA/unruputer/SUN_SHU_MING'},
          'PF6': {'best':'AG/fast/YAO_CUN_HONG', 'median':'AG/fast/ZHANG_CHUN', 'worst':'AAA/unruputer/SUN_SHU_MING'}}

def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(1<<20),b''):h.update(b)
    return h.hexdigest()

def dump(p,d):
    p.parent.mkdir(parents=True,exist_ok=True)
    p.write_text(json.dumps(d,ensure_ascii=False,indent=2,allow_nan=False)+'\n')

def loadnp(p):
    with np.load(p,allow_pickle=False) as z:return {k:z[k] for k in z.files}

def write_vtp(path,xyz,arrays,tris=None,description='',field_data=None):
    pts=vtk.vtkPoints();pts.SetData(numpy_to_vtk(np.ascontiguousarray(xyz,np.float64),deep=True))
    poly=vtk.vtkPolyData();poly.SetPoints(pts)
    ca=vtk.vtkCellArray()
    if tris is None:
        packed=np.column_stack([np.ones(len(xyz),np.int64),np.arange(len(xyz))]).ravel()
        ca.SetCells(len(xyz),numpy_to_vtkIdTypeArray(packed,deep=True));poly.SetVerts(ca)
    else:
        packed=np.column_stack([np.full(len(tris),3,np.int64),tris]).ravel()
        ca.SetCells(len(tris),numpy_to_vtkIdTypeArray(packed,deep=True));poly.SetPolys(ca)
    for k,v in arrays.items():
        v=np.asarray(v);assert len(v)==len(xyz),(k,v.shape,len(xyz))
        a=numpy_to_vtk(np.ascontiguousarray(v),deep=True);a.SetName(k);poly.GetPointData().AddArray(a)
    if arrays:poly.GetPointData().SetActiveScalars(next(iter(arrays)))
    a=vtk.vtkStringArray();a.SetName('display_contract');a.InsertNextValue(description);poly.GetFieldData().AddArray(a)
    for key,value in (field_data or {}).items():
        a=vtk.vtkStringArray() if isinstance(value,str) else vtk.vtkDoubleArray()
        a.SetName(key);a.InsertNextValue(value);poly.GetFieldData().AddArray(a)
    path.parent.mkdir(parents=True,exist_ok=True)
    w=vtk.vtkXMLPolyDataWriter();w.SetFileName(str(path));w.SetInputData(poly);w.SetDataModeToBinary();w.SetCompressorTypeToZLib()
    assert w.Write()==1
    r=vtk.vtkXMLPolyDataReader();r.SetFileName(str(path));r.Update();q=r.GetOutput()
    assert q.GetNumberOfPoints()==len(xyz)
    assert q.GetNumberOfPolys()==(0 if tris is None else len(tris))
    np.testing.assert_array_equal(vtk_to_numpy(q.GetPoints().GetData()),xyz)
    for k,v in arrays.items():np.testing.assert_array_equal(vtk_to_numpy(q.GetPointData().GetArray(k)),v)
    for k,v in (field_data or {}).items():
        a=q.GetFieldData().GetAbstractArray(k);assert a is not None
        if isinstance(v,str):assert a.GetValue(0)==v
        else:np.testing.assert_equal(a.GetValue(0),v)

def write_csv(path,xyz,arrays):
    cols={'x_mm':xyz[:,0],'y_mm':xyz[:,1],'z_mm':xyz[:,2]}
    for k,v in arrays.items():
        if v.ndim==1:cols[k]=v
        else:
            for j in range(v.shape[1]):cols[k+'_'+str(j)]=v[:,j]
    path.parent.mkdir(exist_ok=True,parents=True)
    opener=(lambda:gzip.open(path,'wt',compresslevel=1,newline='')) if path.suffix=='.gz' else (lambda:path.open('w',newline=''))
    with opener() as f:
        f.write(','.join(cols)+'\n');np.savetxt(f,np.column_stack(list(cols.values())),delimiter=',',fmt='%.12g')

def stats(t,p):
    t=np.asarray(t,np.float64);p=np.asarray(p,np.float64)
    return {'r2':float(1-np.sum((p-t)**2)/np.sum((t-t.mean())**2)), 'mae':float(np.mean(np.abs(p-t))), 'n':len(t)}

def geometry(cid,b):
    stl=ROOT/'data_new'/cid/(cid.split('/')[-1]+'.stl')
    v,t=_load_stl_mesh(stl);wall=b['wall_coords_aligned_mm'].astype(float);raw=b['wall_coords_raw'].astype(float)
    d,idx=cKDTree(raw).query(v)
    assert d.max()<1e-3 and len(np.unique(idx))==len(raw)==len(v),'STL unit/identity mismatch'
    R=b['transform_rotation'].astype(float);C=b['transform_centroid'].astype(float)
    aligned=(v-C)@R.T
    derr=np.abs((raw-C)@R.T-wall).max();assert derr<1e-3 and np.linalg.det(R)>0
    assert np.max(np.linalg.norm(aligned-wall[idx],axis=1))<1e-3
    rep=json.loads((DATA/cid/'view_report.json').read_text());hpath=Path(rep['source']['case_h5'])
    with h5py.File(hpath) as h:
        valid=h['wall_static/valid'][()].astype(bool);xyz=h['wall_static/xyz_mm'][()][valid]
        ids=h['wall_static/node_id_cas'][()][valid];tri=h['topology/wall_triangles'][()].astype(np.int64)
        normals=h['wall_static/normal_out_mesh'][()][valid]@R.T
    np.testing.assert_array_equal(ids,b['wall_node_id_cas'])
    assert np.max(np.abs(xyz-raw))<1e-3
    remap=np.full(len(valid),-1,np.int64);remap[valid]=np.arange(valid.sum());tri=remap[tri];keep=np.all(tri>=0,axis=1);tri=tri[keep]
    assert len(tri)>0 and tri.min()>=0 and tri.max()<len(wall)
    report={'stl_source':str(stl),'stl_sha256':sha(stl),'snapshot_source':str(hpath),
       'stl_to_mm_scale':1.,'scale_source':'identity units verified by bijective STL-to-CFD wall mapping; bbox not fitted',
       'bbox_ratio_qc_only':float(np.linalg.norm(np.ptp(raw,axis=0))/np.linalg.norm(np.ptp(v,axis=0))),
       'rotation_convention':'(raw_mm - centroid) @ rotation.T','rotation_determinant':float(np.linalg.det(R)),
       'centroid_mm':C.tolist(),'rotation':R.tolist(),'reconstruction_max_abs_mm':float(derr),
       'stl_wall_identity_max_distance_mm':float(d.max()),'aligned_max_distance_mm':float(np.max(np.linalg.norm(aligned-wall[idx],axis=1))),
       'bijective_node_count':len(idx),'n_stl_triangles':len(t),'n_cfd_triangles':len(tri),'triangles_dropped_invalid':int((~keep).sum()),
       'wall_crop_applied':bool(b['wall_crop_applied']),'frame_version':str(b['transform_frame_version']),
       'coordinate_unit':'mm','tolerance_mm':1e-3}
    return aligned,t,tri,idx,normals,report

def gaussian(xyz,values,target,wall_idx=None,normals=None,tri=None):
    """Same repository Gaussian weights, compute all fields with one bounded stencil."""
    tree=cKDTree(xyz);out=np.full((len(target),values.shape[1]),np.nan)
    nn=tree.query(target,workers=2)[0];valid=np.zeros(len(target),np.uint8);count=np.zeros(len(target),np.int32)
    max_distance=np.full(len(target),np.nan);mean_distance=np.full(len(target),np.nan)
    opposite_mass=np.zeros(len(target));disconnected_mass=np.zeros(len(target))
    adjacency=None
    if wall_idx is not None:
        adjacency=[set() for _ in xyz]
        for a,b,c in tri:adjacency[a].update((b,c));adjacency[b].update((a,c));adjacency[c].update((a,b))
    for start in range(0,len(target),256):
        lists=tree.query_ball_point(target[start:start+256],3.,workers=2)
        for j,inds in enumerate(lists,start):
            if not inds or nn[j]>3.:continue
            ix=np.asarray(inds);dist=np.linalg.norm(xyz[ix]-target[j],axis=1);w=np.exp(-2*(dist/3)**2);w/=w.sum()
            out[j]=w@values[ix];valid[j]=1;count[j]=len(ix);max_distance[j]=dist.max();mean_distance[j]=w@dist
            if wall_idx is not None:
                center=int(wall_idx[j]);opposite_mass[j]=w[(normals[ix]@normals[center])<0].sum()
                allowed=set(inds);seen={center};stack=[center]
                while stack:
                    cur=stack.pop();new=(adjacency[cur]&allowed)-seen;seen.update(new);stack.extend(new)
                disconnected_mass[j]=sum(weight for i,weight in zip(ix,w) if i not in seen)
    # Independent repository implementation check on spatially spread queries.
    check=np.linspace(0,len(target)-1,min(32,len(target)),dtype=int)
    ref,_,refvalid=_interp_gaussian(target[check],xyz,values[:,0],radius=3,sharpness=2,max_dist=3,fallback='mask')
    np.testing.assert_allclose(out[check,0],ref,rtol=1e-12,atol=1e-10,equal_nan=True)
    np.testing.assert_array_equal(valid[check],refvalid)
    extra={'map_dist_mm':nn,'map_valid':valid,'neighbor_count':count,'kernel_max_distance_mm':max_distance,'kernel_mean_distance_mm':mean_distance}
    report={'method':'gaussian','params':{'radius_mm':3.,'sharpness':2.,'max_dist_mm':3.,'fallback':'mask'},
      'kernel':'exp(-2*(d/3 mm)^2) inside 3 mm; normalized; same stencil for CFD and Pred',
      'coverage':{'target_points':len(target),'valid_points':int(valid.sum()),'valid_ratio':float(valid.mean())},
      'distance_mm':{k:float(np.percentile(nn,q)) for k,q in [('p50',50),('p95',95),('max',100)]},
      'independent_kernel_check':{'n_queries':len(check),'implementation':'tools.cfdpost_cloud_export.map_to_stl_surface._interp_gaussian','passed':True},
      'metric_basis':'original cache only, not smoothed STL','surface_area_metric_gate':'not requested; coverage is not area-mapping validation'}
    if wall_idx is not None:
        extra['opposite_normal_weight_fraction']=opposite_mass;extra['disconnected_local_patch_weight_fraction']=disconnected_mass
        report['cross_wall_checks']={'definition':'disconnected = Gaussian neighbors not connected to nearest CFD node inside local wall triangle graph; opposite = negative wall-normal dot',
          'disconnected_stencil_count':int((disconnected_mass>1e-12).sum()),'max_disconnected_weight':float(disconnected_mass.max()),
          'opposite_normal_stencil_count':int((opposite_mass>1e-12).sum()),'max_opposite_normal_weight':float(opposite_mass.max()),
          'interpretation':'geometric diagnostics, not proof of no cross-wall mixing; use native CFD surface to check smoothing'}
    else:report['cross_wall_checks']={'interpretation':'volume Euclidean projection can mix nearby branches; no wall-velocity or derivative claim'}
    return out,extra,report

def build(model,role,cid,summary):
    d=OUT/model/role;d.mkdir(parents=True,exist_ok=True)
    run=ROOT/'training_wss_min/runs/wss_local_wave2_20260912'/(model+'_s1234')
    pp=run/'eval/ckpt_best/predictions/test'/cid/'predictions.npz';bp=DATA/cid/'bundle.npz';vp=DATA/cid/'volume.npz'
    b=loadnp(bp);p=loadnp(pp);vol=loadnp(vp) if model!='X5X11' else None
    peak=int(b['peak_step']);peakidx=int(np.flatnonzero(b['steps']==peak)[0]);assert peak==1162
    cfg=json.loads((run/'config.json').read_text());assert cfg['train']['seed']==1234
    metrics_path=run/'eval/ckpt_best/metrics.json';official=json.loads(metrics_path.read_text())['test']['per_case'][cid]['overall']
    pm_path=run/'eval/ckpt_best/predictions/manifest.json';pm=json.loads(pm_path.read_text())
    assert pm['checkpoint']=='best'
    for key,path in [('checkpoint_sha256',run/'ckpt_best.pt'),('config_sha256',run/'config.json')]:
        if key in pm:assert sha(path)==pm[key],key
    ds=next(x for x in summary['distributions'] if x['key']==('A_' if model=='X5X11' else 'B_')+model)
    sel=next(x for x in ds['cases'] if x['case_id']==cid)
    wall=b['wall_coords_aligned_mm'].astype(float);stlv,stlt,cfdtri,wallidx,normals,geom=geometry(cid,b)
    point_arrays={};unit='Pa';pressure_ref=None;wall_metrics=None
    if model=='X5X11':
        idx=p['row_index'].astype(np.int64);assert np.array_equal(np.sort(idx),np.arange(len(wall)))
        xyz=wall[idx];true=p['true_pa'].astype(float);pred=p['pred_pa'].astype(float)
        np.testing.assert_array_equal(true,b['wall_wss'][peakidx,idx]);assert np.isfinite(pred).all() and pred.min()>=0
        point_arrays={'wss_cfd_pa':true,'wss_pred_pa':pred,'wss_error_pred_minus_cfd_pa':pred-true,'wss_abs_error_pa':abs(pred-true),
           'wss_cfd_norm':p['true_norm'].astype(float),'wss_pred_norm':p['pred_norm'].astype(float),'row_index':idx}
        pointname=f'{model}__{role}__{cid.replace("/","__")}__wall_wss.vtp'
        sort=np.argsort(idx);sx=xyz[sort];base=point_arrays;cfkey='wss_cfd_pa';prkey='wss_pred_pa';erkey='wss_error_pred_minus_cfd_pa'
        sa={k:v[sort] for k,v in base.items()};vals=np.column_stack([sa[cfkey],sa[prkey]])
        cache_metrics=stats(true,pred);native=sa
    else:
        assert int(vol['peak_step'])==peak and str(vol['transform_frame_version'])==str(b['transform_frame_version'])
        idx=p['query_idx'].astype(np.int64);nwall=int(p['n_wall']);assert nwall==len(wall)
        allxyz=np.concatenate([wall,vol['vol_coords_aligned_mm'].astype(float)])
        assert len(np.unique(idx))==len(idx) and idx.min()>=0 and idx.max()<len(allxyz)
        xyz=allxyz[idx];kind=p['point_kind'];np.testing.assert_array_equal(kind,(idx>=nwall).astype(kind.dtype))
        dist=np.r_[np.zeros(nwall),vol['vol_dist_to_wall_mm']][idx];true=p['true_raw'].astype(float);pred=p['pred_raw'].astype(float)
        assert np.isfinite(xyz).all() and np.isfinite(pred).all()
        if model=='PF6':
            truth_all=np.r_[vol['wall_pressure_rel_peak'],vol['vol_pressure_rel_peak']]
            np.testing.assert_array_equal(true,truth_all[idx])
            point_arrays={'pressure_cfd_pa':true,'pressure_pred_pa':pred,'pressure_error_pred_minus_cfd_pa':pred-true,'pressure_abs_error_pa':abs(pred-true)}
            mask=kind==0;wallq=idx[mask];sort=np.argsort(wallq);assert np.array_equal(wallq[sort],np.arange(nwall))
            sx=xyz[mask][sort];np.testing.assert_array_equal(sx,wall)
            sa={k:v[mask][sort] for k,v in point_arrays.items()};vals=np.column_stack([sa['pressure_cfd_pa'],sa['pressure_pred_pa']])
            cfkey='pressure_cfd_pa';prkey='pressure_pred_pa';erkey='pressure_error_pred_minus_cfd_pa';native=sa
            cache_metrics=stats(true,pred);wall_metrics=stats(sa[cfkey],sa[prkey])
            pressure_ref={'p_ref_pa':float(vol['p_ref_pa']),'kind':str(vol['p_ref_kind']),'posthoc_offset_correction':False}
            pointname=f'{model}__{role}__{cid.replace("/","__")}__volume_pressure.vtp'
        else:
            assert np.all(kind==1);np.testing.assert_array_equal(true,vol['vol_velocity_aligned_peak'][idx-nwall])
            ts=np.linalg.norm(true,axis=1);ps=np.linalg.norm(pred,axis=1)
            point_arrays={'speed_cfd':ts,'speed_pred':ps,'speed_error_pred_minus_cfd':ps-ts,'speed_abs_error':abs(ps-ts),
              'velocity_cfd_vector':true,'velocity_pred_vector':pred,'velocity_error_vector':pred-true,'velocity_vector_error_norm':np.linalg.norm(pred-true,axis=1)}
            for j,letter in enumerate('xyz'):point_arrays['velocity_cfd_'+letter]=true[:,j];point_arrays['velocity_pred_'+letter]=pred[:,j]
            cfkey='speed_cfd';prkey='speed_pred';erkey='speed_error_pred_minus_cfd'
            cache_metrics=stats(ts,ps);native=None;unit='m/s'
            pointname=f'{model}__{role}__{cid.replace("/","__")}__volume_velocity.vtp'
        point_arrays.update(query_index=idx,point_kind=kind,dist_to_wall_mm=dist)
    assert abs(cache_metrics['r2']-official['r2'])<1e-8,(model,cache_metrics,official)
    # The saved raw cache is float32; the evaluator can use higher precision before saving.
    assert abs(cache_metrics['mae']-official['mae'])<1e-6,(model,cache_metrics,official)
    prefix={'X5X11':'wss','VF6':'speed','PF6':'pressure'}[model]
    scope={'X5X11':'wall','VF6':'interior','PF6':'wall_union_interior'}[model]
    normalization=fit_selfmax(point_arrays[cfkey],point_arrays[prkey],prefix,scope,unit)
    field_data=normalization_field_data(normalization)
    point_arrays.update(selfmax_fields(point_arrays[cfkey],point_arrays[prkey],normalization))
    if native is not None:native.update(selfmax_fields(native[cfkey],native[prkey],normalization))
    description=f'{model} {cid}; test; seed1234; ckpt_best; peak{peak}; atlas mm; cached same-point prediction'
    write_vtp(d/pointname,xyz,point_arrays,description=description,field_data=field_data)
    write_csv(d/'_export'/'same_point_fields.csv.gz',xyz,point_arrays)
    name=None;report=None;purpose='Interior velocity pointcloud; no wall interpolation'
    if native is not None:
        write_vtp(d/'aligned_geometry.vtp',stlv,{},stlt,description='Aligned STL geometry; no field interpolation')
        write_vtp(d/'cfd_wall_native.vtp',wall,native,cfdtri,description=description+'; native CFD triangles; no smoothing',field_data=field_data)
        mapped,extra,report=gaussian(sx,vals,stlv,wallidx,normals,cfdtri)
        sf={cfkey:mapped[:,0],prkey:mapped[:,1],erkey:mapped[:,1]-mapped[:,0],erkey.replace('error','abs_error'):np.abs(mapped[:,1]-mapped[:,0]),**extra}
        sf.update(selfmax_fields(sf[cfkey],sf[prkey],normalization))
        name='surface_gaussian.vtp';purpose='Wall scalar interpolated to original STL vertices'
        write_vtp(d/name,stlv,sf,stlt,description=description+'; '+purpose,field_data=field_data)
        write_csv(d/'surface_gaussian'/'mapped_vertices.csv',stlv,sf)
        source_arrays={k:vals[:,i] for i,k in enumerate([cfkey,prkey])}
        source_arrays.update({erkey:vals[:,1]-vals[:,0],erkey.replace('error','abs_error'):np.abs(vals[:,1]-vals[:,0])})
        source_arrays.update(selfmax_fields(vals[:,0],vals[:,1],normalization))
        write_csv(d/'_export'/'gaussian_source.csv.gz',sx,source_arrays)
        assert report['coverage']['valid_ratio']==1.,'Incomplete wall mapping'
        report.update(purpose=purpose,source_points=len(sx),source_csv='_export/gaussian_source.csv.gz',target_geometry='aligned_geometry.vtp',geometry=geom,
           units=unit,topology={'triangles':len(stlt),'valid_triangles':int(np.all(extra['map_valid'][stlt],axis=1).sum())},
           errors='signed error = interpolated Pred - interpolated CFD; absolute = abs(signed)',display_normalization=normalization)
        dump(d/'mapping_report.json',report)
    src={str(path):sha(path) for path in [pp,bp,metrics_path,run/'config.json',run/'ckpt_best.pt',pm_path]}
    if vol is not None:src[str(vp)]=sha(vp)
    visual=dict(official,seed=1234,metric_source=str(metrics_path),metric_key=f'test.per_case[{cid}].overall',basis='whole evaluated case, not smoothed surface',cache_recomputed=cache_metrics)
    if wall_metrics is not None:visual['wall_only_reference']=wall_metrics
    manifest={'schema_version':3,'model':model,'role':role,'case_id':cid,'run':str(run),'checkpoint':'ckpt_best.pt','seed':1234,'split':'test',
      'peak_step':peak,'peak_index':peakidx,'frame':str(b['transform_frame_version']),'coordinate_unit':'mm','unit':unit,
      'selection':{'basis':'fixed three-seed mean of per-case R2 (1234,7,2025); not ensemble', 'r2_case_mean':sel['r2'],'statistical_median':ds['stats']['case_median'],
          'median_rule':'closest actual case to statistical median; selected IDs retained from original package'},
      'visualization_metrics':visual,'pressure_reference':pressure_ref,'geometry':geom,'source_sha256':src,'display_normalization':normalization,
      'verification':{'truth_vs_mother_data':'exact','point_index_and_coordinate_identity':'exact','r2_matches_official_abs_tol':1e-8,'mae_matches_official_abs_tol':1e-6,'metric_tolerance_reason':'saved float32 raw cache versus evaluator precision','written_vtp_all_arrays_roundtrip':'exact'},
      'files':{'pointcloud':pointname,'gaussian_surface':name,'mapping_report':'mapping_report.json' if native is not None else None,'native_cfd_wall':'cfd_wall_native.vtp' if native is not None else None,
          'aligned_geometry':'aligned_geometry.vtp' if native is not None else None,'original_points_csv':'_export/same_point_fields.csv.gz'},'surface_display_purpose':purpose}
    manifest['output_sha256']={k:sha(d/k) for k in [pointname]+([name,'aligned_geometry.vtp','cfd_wall_native.vtp'] if native is not None else [])}
    dump(d/'manifest.json',manifest)
    note=f'# {model} / {role} / {cid}\n\nParaView 打开 **{name or pointname}**，点击 Apply，Representation 选 {"Surface" if native is not None else "Points"}。\n\nCFD：`{cfkey}`；Pred：`{prkey}`；有符号误差：`{erkey}`（{unit}）。CFD/Pred 使用相同色标，误差用对称色标。\n\n选例 R²（三种子均值）：{sel["r2"]:.6f}；当前 s1234 全病例 R²：{official["r2"]:.6f}。peak {peak}；v5_atlas_frame_v1；坐标 mm。\n\n'
    if native is not None:
        note+=f'壁面 Gaussian 插值：r=3 mm，sharpness=2，max_dist=3 mm，fallback=mask。覆盖率 {report["coverage"]["valid_ratio"]:.3%}。面片内含 `map_valid`、`map_dist_mm`；配准和跨壁诊断见 `mapping_report.json`。\n\n`cfd_wall_native.vtp` 保留原生 CFD 三角面上的同点场值，无高斯平滑；用于核对局部热点。原点云：`{pointname}`。\n'
        if model=='PF6':note+='\n压力点云包含壁面和体内点（`point_kind`：0 壁面，1 体内）；压力面片仅使用壁面点插值。相对压力 p−p_ref，参考值保存在 manifest.json。\n'
    else:note+='仅保留体内速度点云，不做壁面插值。含速度大小和 CFD/Pred 三分量向量；`dist_to_wall_mm` 可筛选近壁区域。点云没有体单元，不能当作连续体网格直接 Slice。\n'
    cmax_label='invalid' if normalization['cfd_max'] is None else f'{normalization["cfd_max"]:.12g}'
    pmax_label='invalid' if normalization['pred_max'] is None else f'{normalization["pred_max"]:.12g}'
    note+=f'\n自身最大值归一化：`{prefix}_cfd_selfmax` = CFD / 原始 CFD max；`{prefix}_pred_selfmax` = Pred / 原始 Pred max；差值 `{prefix}_selfmax_error_pred_minus_cfd`，绝对差 `{prefix}_selfmax_abs_error`。均无量纲，仅作空间分布对照，幅值差已移除。\n\n原始同点域 `{scope}`：CFD max = {cmax_label} {unit}，Pred max = {pmax_label} {unit}。点云/原生面/Gaussian 共享此对分母，插值后不再重求最大值；元数据嵌入 VTP FieldData 和 manifest.json。\n'
    if model=='PF6':note+='\n压力 selfmax 按带符号相对压力除以各自 max，保留原 p_ref，不改成 maxabs/minmax；允许负值和小于 −1，不强制使用 0–1 色标。\n'
    (d/'README_打开说明.md').write_text(note)
    print(json.dumps({'model':model,'role':role,'case':cid,'points':len(xyz),'surface_triangles':len(stlt) if native is not None else 0,'coverage':report['coverage']['valid_ratio'] if report else None,'s1234_r2':official['r2']},ensure_ascii=False),flush=True)
    return manifest

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--models',nargs='+',default=list(SELECT),choices=list(SELECT));args=ap.parse_args()
    summary=json.loads(SUMMARY.read_text())
    for model in args.models:
        for role,cid in SELECT[model].items():build(model,role,cid,summary)
    cases=[]
    for model in SELECT:
        for role in SELECT[model]:
            p=OUT/model/role/'manifest.json'
            if p.exists():cases.append(json.loads(p.read_text()))
    dump(OUT/'manifest.json',{'schema_version':3,'note':'raw physical plus selfmax display fields rebuilt from caches; package paths relative to each case for relocation',
      'selection_summary_source':str(SUMMARY),'selection_summary_sha256':sha(SUMMARY),'cases':cases})
    with (OUT/'打开文件清单.csv').open('w',encoding='utf-8-sig',newline='') as f:
        w=csv.writer(f);w.writerow(['model','role','case_id','s1234_whole_case_r2','three_seed_selection_r2','surface_file','original_points','usage'])
        for c in cases:
            if c.get('schema_version')!=3:continue
            rel=Path(c['model'])/c['role'];w.writerow([c['model'],c['role'],c['case_id'],c['visualization_metrics']['r2'],c['selection']['r2_case_mean'],str(rel/c['files']['gaussian_surface']) if c['files']['gaussian_surface'] else '',str(rel/c['files']['pointcloud']),c['surface_display_purpose']])
    with (OUT/'归一化分母清单.csv').open('w',encoding='utf-8-sig',newline='') as f:
        w=csv.writer(f);w.writerow(['model','role','case_id','seed','peak_step','denominator_scope','source_unit','cfd_max','pred_max','cfd_selfmax_field','pred_selfmax_field','cfd_valid','pred_valid'])
        for c in cases:
            if c.get('schema_version')!=3:continue
            n=c['display_normalization'];prefix=n['prefix']
            w.writerow([c['model'],c['role'],c['case_id'],c['seed'],c['peak_step'],n['denominator_scope'],n['source_unit'],n['cfd_max'],n['pred_max'],prefix+'_cfd_selfmax',prefix+'_pred_selfmax',n['cfd_denominator']['valid'],n['pred_denominator']['valid']])
if __name__=='__main__':main()

"""Auditable V5.2 joint u/p'/|tau| cycle data, with bounded-memory preparation.

Preparation reads ONE CFD frame at a time, never an 81-frame volume array.
The assembly plan, not a search/fallback, selects each original H5. Cached
labels are fixed uniform candidate pools; training resamples candidates by
case/epoch/seed. Evaluation uses a separately seeded, disjoint query pool.
Only training candidate pools fit normalization. Pressure uses the stored
per-frame volume reference; velocity vectors rotate into the atlas frame.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Mapping

import h5py
import numpy as np
import torch
from torch.utils.data import Dataset

ROOT = Path(os.environ.get('GNN_JOINT_SOURCE_ROOT', '/public/newhome/cy/Digital_twin/GNN')).resolve()
VIEW_ROOT = ROOT / 'data_wss_v5/views_v5_2_full_20260923'
DEFAULT_CACHE = ROOT / 'training_wss_min/experiments/joint_cycle_v52_20260929/data_audit_cache'
SPLIT_PATH = Path(os.environ.get('GNN_JOINT_SPLIT_PATH', str(VIEW_ROOT / 'wss_min_view_v1/cv5_v52/fold0.json'))).resolve()
WAVE_PATH = ROOT / 'training_wss_min/experiments/wss_time_ecc_20260918/offline/protocol_inlet_waveform_v51.json'
SCHEMA = 'joint_cycle_v52_v1'
TASKS = ('velocity', 'pressure', 'wss')
N_FRAMES = 80
PHASE_FEATURES = ('q_norm', 'dq_norm', 't_sin', 't_cos')
BASE_FEATURES = ('x','y','z','abscissa_norm','local_radius','curvature',
                 'log_local_radius','rho','theta_sin','theta_cos','dr_ds',
                 'dist_to_junction_mm','dist_to_endpoint_mm','end_zone')
WALL_FEATURES = BASE_FEATURES + ('nx_aligned','ny_aligned','nz_aligned',
    'curv_k1','curv_k2','curv_gauss','curvedness','curv_k1_c','curv_k2_c','curv_gauss_c',
    'tn_dot','log_q_branch_murray_cap','log_tau0_murray_cap')
VOLUME_FEATURES = BASE_FEATURES + ('radial_x','radial_y','radial_z',
    'log_q_branch_murray_cap','log_tau0_murray_cap','dist_to_wall_mm')
CURVATURE_FEATURES = {'curvature','curv_k1','curv_k2','curv_gauss','curvedness',
                      'curv_k1_c','curv_k2_c','curv_gauss_c'}
OUTLETS = ('out-le','out-li','out-re','out-ri')
RHEOLOGY_FEATURES = ('mu_inf_pa_s','mu_zero_pa_s','lambda_s','a','n')
BC_FEATURES = tuple(f'log_{o}_{v}' for o in OUTLETS for v in ('R1','R2','C')) + (
    'period_s','rho_kg_m3',*RHEOLOGY_FEATURES,'A_inlet_udf_m2')


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def stable_seed(seed, cid, stream, epoch=0):
    return int.from_bytes(hashlib.sha256(f'{seed}|{cid}|{stream}|{epoch}'.encode()).digest()[:8], 'little')


def _write_json(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(data, indent=2, ensure_ascii=False) + '\n')
    tmp.replace(path)


def _signature(path):
    p = Path(path).resolve()
    s = p.stat()
    return {'path': str(p), 'size': s.st_size, 'mtime_ns': s.st_mtime_ns}


def _boundary_conditions(h):
    """Frozen simulation inputs stored with this exact CFD export, not labels."""
    attrs=h['conditions'].attrs
    rcr={v['outlet']:v for v in json.loads(attrs['rcr'])}
    rheology=json.loads(attrs['rheology'])
    values=[np.log(float(rcr[o][v])) for o in OUTLETS for v in ('R1','R2','C')]
    values += [float(attrs['period_s']),float(attrs['rho_kg_m3'])]
    values += [float(rheology[v]) for v in RHEOLOGY_FEATURES]
    values += [float(attrs['A_inlet_udf_m2'])]
    arr=np.asarray(values,np.float32)
    assert np.isfinite(arr).all()
    return arr,{'udf':json.loads(attrs['udf']),'rcr':list(rcr.values()),'rheology':rheology,
        'period_s':float(attrs['period_s']),'rho_kg_m3':float(attrs['rho_kg_m3']),
        'A_inlet_udf_m2':float(attrs['A_inlet_udf_m2']),
        'resistance_units':'Pa*s/kg','capacitance_units':'kg/Pa',
        'source':'case.h5 conditions attributes; frozen prescribed inputs, no CFD labels'}


def source_paths(cid, cache_root=DEFAULT_CACHE, *, require_volume=True):
    plan = json.loads((VIEW_ROOT / 'assembly_plan.json').read_text())['plan']
    tag = plan[cid]
    if tag == 'v5.1':
        base = ROOT / 'data_wss_v5/anatomy_pointcloud_v5_1_20260916/cases'
    elif tag in ('v5.2-new', 'v5.2-rebuilt'):
        base = ROOT / 'data_wss_v5/anatomy_pointcloud_v5_2_20260922/cases'
    else:
        raise ValueError(f'{cid}: unsupported assembly provenance {tag}')
    paths = {'h5': base / cid.replace('/', '__') / 'case.h5',
             'bundle': VIEW_ROOT / 'wss_min_view_v1' / cid / 'bundle.npz',
             'volume': VIEW_ROOT / 'wss_min_view_v1' / cid / 'volume.npz',
             'geom': VIEW_ROOT / 'wss_min_geom_v2' / cid / 'features.npz',
             'flowref': VIEW_ROOT / 'wss_min_flowref_v1' / cid / 'features.npz'}
    if not paths['volume'].is_file():
        derived=Path(cache_root)/'derived_volume'/cid/'volume.npz'
        provenance=derived.with_name('source.json')
        if derived.is_file() and provenance.is_file():
            expected={k:_signature(paths[k]) for k in ('h5','bundle')}
            if json.loads(provenance.read_text())['sources'] != expected:
                raise ValueError(f'{cid}: derived volume source changed')
            paths['volume']=derived
    for name, p in paths.items():
        if name=='volume' and not require_volume:
            continue
        if not p.is_file():
            raise FileNotFoundError(f'{cid}: assembly={tag}, missing {name}: {p}; fallback is forbidden')
    return tag, paths


def _wall_source_rows(h,b):
    source=h['wall_static/node_id_cas'][()]
    desired=b['wall_node_id_cas']
    order=np.argsort(source)
    where=np.searchsorted(source[order],desired)
    if np.any(where>=len(source)) or not np.array_equal(source[order[np.minimum(where,len(source)-1)]],desired):
        raise ValueError('bundle wall IDs are not a subset of the authoritative H5 wall IDs')
    return order[where]


def build_missing_volume(cid,cache_root=DEFAULT_CACHE):
    tag,paths=source_paths(cid,cache_root,require_volume=False)
    if paths['volume'].is_file():
        return {'canonical_id':cid,'status':'existing'}
    from wss_v5.views.volume_view import build_case_volume
    out=Path(cache_root)/'derived_volume'
    dest=out/cid
    dest.mkdir(parents=True,exist_ok=True)
    link=dest/'bundle.npz'
    if not link.exists():
        link.symlink_to(paths['bundle'].resolve())
    elif link.resolve()!=paths['bundle'].resolve():
        raise ValueError(f'{cid}: stale derived view bundle link')
    result=build_case_volume(cid,paths['h5'].parents[2],out)
    _write_json(dest/'source.json',{'assembly_source':tag,'sources':{k:_signature(paths[k]) for k in ('h5','bundle')},
        'builder':'wss_v5.views.volume_view.build_case_volume','builder_sha256':sha256(ROOT/'wss_v5/views/volume_view.py'),
        'report':result})
    return result


def inspect_case(cid,cache_root=DEFAULT_CACHE):
    """Read metadata plus identity arrays, without reading temporal fields."""
    tag, paths = source_paths(cid,cache_root)
    with h5py.File(paths['h5'], 'r') as h, np.load(paths['bundle']) as b, np.load(paths['volume']) as v:
        assert str(h.attrs['canonical_id']) == cid
        assert str(b['canonical_id']) == cid == str(v['canonical_id'])
        assert str(b['transform_frame_version']) == 'v5_atlas_frame_v1'
        assert str(v['transform_frame_version']) == 'v5_atlas_frame_v1'
        steps = h['wall_temporal/step'][()]
        assert np.array_equal(steps, np.arange(1120,1281,2))
        assert np.array_equal(steps, b['steps'])
        source_rows=_wall_source_rows(h,b)
        nw = len(source_rows)
        nv = len(h['volume_static/xyz_mm'])
        assert int(v['n_cells']) == nv
        assert h['volume_temporal/velocity_m_s'].shape == (81,nv,3)
        assert h['volume_temporal/pressure_pa'].shape == (81,nv)
        assert h['wall_temporal/wss_scalar_pa'].shape == (81,len(h['wall_static/node_id_cas']))
        assert h['pressure_reference/p_volume_mean_pa'].shape == (81,)
        rows = np.unique(np.linspace(0,nv-1,min(nv,33)).astype(int))
        assert np.allclose(h['volume_static/xyz_mm'][rows],v['vol_coords_raw'][rows],rtol=0,atol=2e-4)
        rot = b['transform_rotation']
        assert np.allclose(rot @ rot.T,np.eye(3),atol=1e-8)
        assert np.linalg.det(rot) > .999999
        wf = json.loads(WAVE_PATH.read_text())
        assert np.allclose(h['conditions/q_nom_m3s'][()],wf['q_nom_m3s'],rtol=1e-10,atol=1e-14)
        _,bc_meta=_boundary_conditions(h)
    return {'canonical_id':cid,'assembly_source':tag,'n_wall':nw,'n_volume':nv,
            'sources':{k:_signature(p) for k,p in paths.items()},'boundary_conditions':bc_meta}


def audit(cache_root=DEFAULT_CACHE):
    split = json.loads(SPLIT_PATH.read_text())
    ids = split['train_cases'] + split['test_cases']
    assert len(set(ids)) == 261
    assert len(ids) == 261
    def patient(cid):
        q=cid.split('/')
        return q[1].rsplit('-',1)[0] if q[0]=='ILO' else q[-1]
    groups={c:patient(c) for c in ids}
    related=split.get('duplicate_geometry_groups',[])+split.get('explicit_related_groups',[])
    for group in related:
        keys={groups[c] for c in group if c in groups}
        if keys:
            root=min(keys)
            groups={c:(root if g in keys else g) for c,g in groups.items()}
    overlap=sorted({groups[c] for c in split['train_cases']} & {groups[c] for c in split['test_cases']})
    result = {'schema':SCHEMA,'split_path':str(SPLIT_PATH),'split_sha256':sha256(SPLIT_PATH),
              'train_cases':split['train_cases'],'test_cases':split['test_cases'],'cases':[],'errors':[],
              'patient_group_overlap':overlap,'patient_independent':not overlap,
              'patient_group_overlap_cases':{g:{p:[c for c in split[p+'_cases'] if groups[c]==g] for p in ('train','test')} for g in overlap},
              'group_rule':'ILO strip terminal -0/-1 and before/after; otherwise case name; union duplicate_geometry_groups and explicit_related_groups'}
    if overlap:
        result['errors'].append({'error':'patient/related geometry groups cross train and test',
                                 'overlap_groups':overlap})
    for i,cid in enumerate(ids):
        try:
            result['cases'].append(inspect_case(cid,cache_root))
        except Exception as e:
            result['errors'].append({'canonical_id':cid,'error':repr(e),'traceback':traceback.format_exc()})
            print(result['errors'][-1]['traceback'],flush=True)
        print(f'audit {i+1}/{len(ids)} {cid} errors={len(result["errors"])}',flush=True)
    _write_json(Path(cache_root)/'audit.json',result)
    if result['errors']:
        raise RuntimeError(f'coverage audit failed: {result["errors"]}')
    return result


def _raw_features(b,v,g,f):
    cols = {}
    for j,name in enumerate(('x','y','z')):
        cols[name] = b['wall_coords_norm'][:,j]
    for name in BASE_FEATURES[3:]:
        cols[name] = np.log(np.maximum(b['wall_local_radius'],1e-6)) if name == 'log_local_radius' else b['wall_'+name]
    for j,name in enumerate(('nx_aligned','ny_aligned','nz_aligned')):
        cols[name] = b['wall_normal_pca_aligned'][:,j]
    for name in WALL_FEATURES[17:]:
        cols[name] = (g if 'wall_'+name in g.files else f)['wall_'+name]
    wall = np.stack([cols[n] for n in WALL_FEATURES],axis=1).astype(np.float32)
    vc = {n:v['vol_coords_norm'][:,j] for j,n in enumerate(('x','y','z'))}
    for name in BASE_FEATURES[3:]:
        vc[name] = np.log(np.maximum(v['vol_local_radius'],1e-6)) if name == 'log_local_radius' else v['vol_'+name]
    for j,name in enumerate(('radial_x','radial_y','radial_z')):
        vc[name] = v['vol_radial_aligned'][:,j]
    ws,vs = b['wall_segment_id'],v['vol_segment_id']
    shares = {int(k):np.median(cols['log_q_branch_murray_cap'][ws == k]) for k in np.unique(ws)}
    missing = set(np.unique(vs).tolist()) - set(shares)
    if missing:
        raise ValueError(f'volume segments without wall Murray geometry: {missing}')
    vc['log_q_branch_murray_cap'] = np.asarray([shares[int(k)] for k in vs],dtype=np.float32)
    vc['log_tau0_murray_cap'] = vc['log_q_branch_murray_cap'] - 3*np.log(np.maximum(v['vol_local_radius'],1e-6))
    vc['dist_to_wall_mm'] = v['vol_dist_to_wall_mm']
    vol = np.stack([vc[n] for n in VOLUME_FEATURES],axis=1).astype(np.float32)
    for x,names in ((wall,WALL_FEATURES),(vol,VOLUME_FEATURES)):
        for j,n in enumerate(names):
            if n in CURVATURE_FEATURES:
                x[:,j] = np.sign(x[:,j])*np.log1p(np.abs(x[:,j]))
        if not np.isfinite(x).all():
            raise ValueError('non-finite geometry features')
    return wall,vol


def _pool_rows(n,ntrain,neval,seed,cid,domain):
    if n < 2 or ntrain < 1 or neval < 1:
        raise ValueError(f'{cid}: {domain} requires two nonempty disjoint pools')
    if ntrain+neval > n:
        ntrain=max(1,min(n-1,round(n*ntrain/(ntrain+neval))))
        neval=n-ntrain
    rows = np.random.default_rng(stable_seed(seed,cid,domain+'-pools')).permutation(n)
    return np.sort(rows[:ntrain]),np.sort(rows[ntrain:ntrain+neval])


def prepare_case(cid,cache_root=DEFAULT_CACHE,*,seed=1234,train_volume=8192,
                 train_wall=8192,eval_volume=16384,eval_wall=8192):
    cache_root = Path(cache_root)
    record = inspect_case(cid,cache_root)
    _,p = source_paths(cid,cache_root)
    out = cache_root/'cases'/cid
    settings = dict(seed=seed,train_volume=train_volume,train_wall=train_wall,
                    eval_volume=eval_volume,eval_wall=eval_wall)
    marker = out/'metadata.json'
    if marker.exists():
        previous = json.loads(marker.read_text())
        if previous.get('schema') != SCHEMA or previous['settings'] != settings or previous['sources'] != record['sources']:
            raise ValueError(f'{cid}: refusing incompatible/stale cache; use a new cache root')
        return previous
    out.mkdir(parents=True,exist_ok=True)
    with h5py.File(p['h5'],'r') as h, np.load(p['bundle']) as b, np.load(p['volume']) as v, np.load(p['geom']) as g, np.load(p['flowref']) as f:
        for z in (g,f):
            assert np.array_equal(b['wall_node_id_cas'],z['wall_node_id_cas'])
        wx,vx = _raw_features(b,v,g,f)
        wr = _pool_rows(len(wx),train_wall,eval_wall,seed,cid,'wall')
        vr = _pool_rows(len(vx),train_volume,eval_volume,seed,cid,'volume')
        arrays = {'support_pos':b['wall_coords_norm'],'support_x':wx}
        source_wall_rows=_wall_source_rows(h,b)
        area = h['wall_static/area_m2'][()][source_wall_rows]
        volume = v['vol_volume_m3']
        assert np.isfinite(area).all() and area.min() >= 0 and area.sum() > 0
        assert np.isfinite(volume).all() and volume.min() > 0
        pref = h['pressure_reference/p_volume_mean_pa'][()]
        rot = b['transform_rotation']
        for mode,wi,vi in zip(('train','eval'),wr,vr):
            arrays.update({f'{mode}_wall_pos':b['wall_coords_norm'][wi],f'{mode}_wall_x':wx[wi],
                f'{mode}_wall_rows':wi,f'{mode}_wall_weight':area[wi],
                f'{mode}_volume_pos':v['vol_coords_norm'][vi],f'{mode}_volume_x':vx[vi],
                f'{mode}_volume_rows':vi,f'{mode}_volume_weight':volume[vi],
                f'{mode}_velocity':np.empty((len(vi),80,3),np.float32),
                f'{mode}_pressure':np.empty((len(vi),80,1),np.float32),
                f'{mode}_wss':np.empty((len(wi),80,1),np.float32)})
        errors = {'peak_velocity_abs_max':0.,'peak_pressure_abs_max':0.,'peak_wss_abs_max':0.}
        endpoint = {}
        for k in range(81):
            # Contiguous one-frame reads avoid pathological h5py fancy-index scans.
            u = h['volume_temporal/velocity_m_s'][k]
            pr = h['volume_temporal/pressure_pa'][k]
            tau = h['wall_temporal/wss_scalar_pa'][k][source_wall_rows]
            for mode,wi,vi in zip(('train','eval'),wr,vr):
                uq = (u[vi].astype(np.float64) @ rot.T).astype(np.float32)
                pq = (pr[vi].astype(np.float64)-pref[k]).astype(np.float32)
                wq = tau[wi]
                if k < 80:
                    arrays[f'{mode}_velocity'][:,k] = uq
                    arrays[f'{mode}_pressure'][:,k,0] = pq
                    arrays[f'{mode}_wss'][:,k,0] = wq
                else:
                    endpoint[mode] = {name:float(np.max(np.abs(a-arrays[f'{mode}_{name}'][:,0])))
                        for name,a in (('velocity',uq),('pressure',pq[:,None]),('wss',wq[:,None]))}
                if k == 21:
                    errors['peak_velocity_abs_max'] = max(errors['peak_velocity_abs_max'],float(np.max(np.abs(uq-v['vol_velocity_aligned_peak'][vi]))))
                    errors['peak_pressure_abs_max'] = max(errors['peak_pressure_abs_max'],float(np.max(np.abs(pq-v['vol_pressure_rel_peak'][vi]))))
                    errors['peak_wss_abs_max'] = max(errors['peak_wss_abs_max'],float(np.max(np.abs(wq-b['wall_wss'][21,wi]))))
        assert errors['peak_velocity_abs_max'] < 1e-5,errors
        assert errors['peak_pressure_abs_max'] < .02,errors
        assert errors['peak_wss_abs_max'] < 1e-5,errors
        wf = json.loads(WAVE_PATH.read_text())
        arrays['phase'] = np.stack([wf[n][:80] for n in PHASE_FEATURES],axis=1).astype(np.float32)
        arrays['bc'],_ = _boundary_conditions(h)
        for name,a in arrays.items():
            if not np.isfinite(a).all():
                raise ValueError(f'{cid}/{name}: nonfinite cache')
            np.save(out/(name+'.npy'),a,allow_pickle=False)
    record.update(schema=SCHEMA,settings=settings,peak_consistency=errors,endpoint_80_vs_0_abs_max=endpoint,
        actual_pool_counts={'train_wall':len(wr[0]),'eval_wall':len(wr[1]),'train_volume':len(vr[0]),'eval_volume':len(vr[1])},
        normalization_input='raw physical labels; WSS floor is applied only by normalizer',
        bc_note='20 frozen prescribed input channels from conditions attrs; RCR resistance Pa*s/kg and capacitance kg/Pa; train-only normalization')
    _write_json(marker,record)
    return record


def fit_stats(cache_root=DEFAULT_CACHE,train_ids=None):
    cache_root = Path(cache_root)
    split = json.loads(SPLIT_PATH.read_text())
    train_ids = split['train_cases'] if train_ids is None else list(train_ids)
    if not set(train_ids) <= set(split['train_cases']):
        raise ValueError('normalization attempted to include a non-training case')
    stats = {'schema':SCHEMA,'split_path':str(SPLIT_PATH),'split_sha256':sha256(SPLIT_PATH),'train_ids':train_ids,
        'complete_train_partition':train_ids == split['train_cases'],
        'scope':'uniform fixed train candidate pools only; point pooled per-frame targets',
        'dimensions':{'support':27,'velocity':20,'pressure':20,'wss':27},
        'phase_dims':4,'bc_dims':len(BC_FEATURES),'bc_features':list(BC_FEATURES),
        'phase_features':list(PHASE_FEATURES),'frames':list(range(80)),
        'features':{},'targets':{}}
    bc_values=np.stack([np.load(cache_root/'cases'/cid/'bc.npy') for cid in train_ids]).astype(np.float64)
    stats['bc']={'mean':bc_values.mean(0).tolist(),'std':np.maximum(bc_values.std(0),1e-6).tolist()}
    for domain,names in (('wall',WALL_FEATURES),('volume',VOLUME_FEATURES)):
        values = np.concatenate([np.load(cache_root/'cases'/cid/f'train_{domain}_x.npy') for cid in train_ids])
        clips = np.full(len(names),np.inf)
        for j,n in enumerate(names):
            if n in CURVATURE_FEATURES:
                clips[j] = np.quantile(np.abs(values[:,j]),.99)
                values[:,j] = np.clip(values[:,j],-clips[j],clips[j])
        mean,std = values.mean(0,dtype=np.float64),values.std(0,dtype=np.float64)+1e-6
        mean[:3],std[:3] = 0.,1.  # preserve case-normalized coordinates
        stats['features'][domain] = {'names':list(names),'mean':mean.tolist(),'std':std.tolist(),
             'clip':[float(v) if np.isfinite(v) else None for v in clips],
             'curvature_transform':'signed_log1p applied before caching'}
    for task in TASKS:
        total = sq = None
        count = 0
        for cid in train_ids:
            a = np.load(cache_root/'cases'/cid/f'train_{task}.npy').astype(np.float64)
            if task == 'wss':
                a = np.log(np.maximum(a,.05)+1e-6)
            s,ss = a.sum(0),np.square(a).sum(0)
            total = s if total is None else total+s
            sq = ss if sq is None else sq+ss
            count += len(a)
        mean = total/count
        std = np.sqrt(np.maximum(sq/count-mean**2,0))+1e-6
        stats['targets'][task] = {'mean':mean.tolist(),'std':std.tolist(),'n_points_per_frame':count,
            'transform':'log' if task=='wss' else 'linear','floor':.05 if task=='wss' else None,
            'eps':1e-6 if task=='wss' else None}
    _write_json(cache_root/'stats.json',stats)
    return stats


def encode_target(task,raw,stats):
    s=stats['targets'][task]
    a=np.asarray(raw,dtype=np.float64)
    if s['transform']=='log':
        a=np.log(np.maximum(a,s['floor'])+s['eps'])
    return ((a-np.asarray(s['mean']))/np.asarray(s['std'])).astype(np.float32)


def decode_target(task,pred,stats):
    """Invert per-frame target normalization; supports NumPy or Torch."""
    s=stats['targets'][task]
    if isinstance(pred,torch.Tensor):
        a=pred*pred.new_tensor(s['std'])+pred.new_tensor(s['mean'])
        return torch.exp(a)-s['eps'] if s['transform']=='log' else a
    a=np.asarray(pred)*np.asarray(s['std'])+np.asarray(s['mean'])
    return np.exp(a)-s['eps'] if s['transform']=='log' else a


class JointCycleDataset(Dataset):
    """Sample cached geometry and all 80 phases; no CFD reads during training."""
    def __init__(self,cache_root=DEFAULT_CACHE,partition='train',stats_path=None,*,
                 support_n=5000,query_n=512,training=True,seed=1234,case_ids=None,
                 allow_partial_stats=False):
        self.cache_root=Path(cache_root)
        self.stats=json.loads(Path(stats_path or self.cache_root/'stats.json').read_text())
        if self.stats['split_sha256'] != sha256(SPLIT_PATH):
            raise ValueError('split changed after fitting cache statistics')
        if not self.stats['complete_train_partition'] and not allow_partial_stats:
            raise ValueError('partial statistics are for smoke tests only')
        split=json.loads(SPLIT_PATH.read_text())
        self.ids=list(case_ids) if case_ids is not None else split[partition+'_cases']
        if not set(self.ids) <= set(split[partition+'_cases']):
            raise ValueError('case_ids cross the specified split partition')
        self.support_n=int(support_n)
        self.query_n={t:int(query_n[t]) for t in TASKS} if isinstance(query_n,Mapping) else {t:int(query_n) for t in TASKS}
        if self.query_n['velocity'] != self.query_n['pressure']:
            raise ValueError('velocity and pressure must share equal-size query rows')
        self.training,self.seed,self.epoch=bool(training),int(seed),0
        self._cache={}
        for cid in self.ids:
            if not (self.cache_root/'cases'/cid/'metadata.json').is_file():
                raise FileNotFoundError(f'incomplete cache case {cid}')

    def __len__(self):
        return len(self.ids)

    def set_epoch(self,epoch):
        self.epoch=int(epoch)

    def _array(self,cid,name):
        key=(cid,name)
        if key not in self._cache:
            self._cache[key]=np.load(self.cache_root/'cases'/cid/(name+'.npy'),mmap_mode='r')
        return self._cache[key]

    def __getstate__(self):
        state=self.__dict__.copy();state['_cache']={};return state

    def _x(self,raw,domain):
        s=self.stats['features'][domain]
        clips=np.asarray([np.inf if v is None else v for v in s['clip']])
        return ((np.clip(raw,-clips,clips)-np.asarray(s['mean']))/np.asarray(s['std'])).astype(np.float32)

    def _rows(self,n,k,cid,stream):
        if k<=0 or k>=n:
            return np.arange(n)
        epoch=self.epoch if self.training else 0
        return np.sort(np.random.default_rng(stable_seed(self.seed,cid,stream,epoch)).choice(n,k,replace=False))

    def __getitem__(self,index):
        cid=self.ids[index]; mode='train' if self.training else 'eval'
        sr=self._rows(len(self._array(cid,'support_pos')),self.support_n,cid,'support-'+mode)
        item={'unit_id':cid,'support':{'pos':np.asarray(self._array(cid,'support_pos')[sr],np.float32),
             'x':self._x(self._array(cid,'support_x')[sr],'wall'),'rows':sr},
             'phase':np.array(self._array(cid,'phase')),
             'bc':((self._array(cid,'bc')-np.asarray(self.stats['bc']['mean']))/np.asarray(self.stats['bc']['std'])).astype(np.float32),
             'queries':{}}
        rowsets={d:self._rows(len(self._array(cid,f'{mode}_{d}_pos')),self.query_n[t],cid,d+'-query-'+mode)
                 for d,t in (('volume','velocity'),('wall','wss'))}
        for task in TASKS:
            domain='wall' if task=='wss' else 'volume';rows=rowsets[domain]
            raw=np.array(self._array(cid,f'{mode}_{task}')[rows])
            item['queries'][task]={'pos':np.asarray(self._array(cid,f'{mode}_{domain}_pos')[rows],np.float32),
                'x':self._x(self._array(cid,f'{mode}_{domain}_x')[rows],domain),
                'y':encode_target(task,raw,self.stats),'y_raw':raw,
                'weight':np.asarray(self._array(cid,f'{mode}_{domain}_weight')[rows],np.float32),
                'rows':np.array(self._array(cid,f'{mode}_{domain}_rows')[rows])}
        return item


def collate_joint_cycle(items):
    def pack(parts):
        out={k:torch.from_numpy(np.concatenate([a[k] for a in parts])) for k in parts[0]}
        out['batch']=torch.cat([torch.full((len(a['pos']),),i,dtype=torch.long) for i,a in enumerate(parts)])
        return out
    return {'unit_ids':[i['unit_id'] for i in items],
        'support':pack([i['support'] for i in items]),
        'queries':{t:pack([i['queries'][t] for i in items]) for t in TASKS},
        'phase':torch.from_numpy(np.stack([i['phase'] for i in items])),
        'bc':torch.from_numpy(np.stack([i['bc'] for i in items]))}


def main():
    global SPLIT_PATH
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('command',choices=('views','audit','prepare','stats','smoke'))
    p.add_argument('--cache-root',type=Path,default=DEFAULT_CACHE)
    p.add_argument('--split-path',type=Path,default=SPLIT_PATH)
    p.add_argument('--limit',type=int,default=0)
    p.add_argument('--seed',type=int,default=1234)
    p.add_argument('--workers',type=int,default=1)
    for name,default in [('train-volume',8192),('train-wall',8192),('eval-volume',16384),('eval-wall',8192)]:
        p.add_argument('--'+name,type=int,default=default)
    a=p.parse_args()
    SPLIT_PATH=a.split_path.resolve()
    if a.command=='audit':
        audit(a.cache_root);return
    split=json.loads(SPLIT_PATH.read_text());ids=split['train_cases']+split['test_cases']
    if a.limit: ids=ids[:a.limit]
    if a.command=='views':
        with ProcessPoolExecutor(max_workers=a.workers) as pool:
            jobs={pool.submit(build_missing_volume,cid,a.cache_root):cid for cid in ids}
            reports=[]
            for i,future in enumerate(as_completed(jobs)):
                reports.append(future.result())
                print(f'volume view {i+1}/{len(ids)} {jobs[future]}',flush=True)
        _write_json(a.cache_root/'volume_view_report.json',reports)
        return
    if a.command=='prepare':
        kwargs=dict(seed=a.seed,train_volume=a.train_volume,train_wall=a.train_wall,
                    eval_volume=a.eval_volume,eval_wall=a.eval_wall)
        if a.workers > 1:
            with ProcessPoolExecutor(max_workers=a.workers) as pool:
                jobs={pool.submit(prepare_case,cid,a.cache_root,**kwargs):cid for cid in ids}
                for i,future in enumerate(as_completed(jobs)):
                    future.result()
                    print(f'cache {i+1}/{len(ids)} {jobs[future]}',flush=True)
        else:
            for i,cid in enumerate(ids):
                prepare_case(cid,a.cache_root,**kwargs)
                print(f'cache {i+1}/{len(ids)} {cid}',flush=True)
        _write_json(a.cache_root/'manifest.json',{'schema':SCHEMA,'split_sha256':sha256(SPLIT_PATH),
            'case_ids':ids,'complete':len(ids)==261,'settings':vars(a)|{'cache_root':str(a.cache_root),'split_path':str(SPLIT_PATH)}})
        fit_stats(a.cache_root,[c for c in ids if c in split['train_cases']]);return
    if a.command=='stats':
        fit_stats(a.cache_root);return
    ds=JointCycleDataset(a.cache_root,case_ids=ids[:1],allow_partial_stats=True)
    x=ds[0];batch=collate_joint_cycle([x,x])
    assert np.array_equal(x['queries']['velocity']['rows'],x['queries']['pressure']['rows'])
    for task in TASKS:
        q=x['queries'][task]
        expected=np.maximum(q['y_raw'],.05) if task=='wss' else q['y_raw']
        assert np.allclose(decode_target(task,q['y'],ds.stats),expected,atol=1e-4,rtol=1e-5)
    assert np.array_equal(ds[0]['queries']['velocity']['rows'],x['queries']['velocity']['rows'])
    ds.set_epoch(1)
    assert not np.array_equal(ds[0]['queries']['velocity']['rows'],x['queries']['velocity']['rows'])
    result={'unit_id':x['unit_id'],'support':{k:list(v.shape) for k,v in batch['support'].items()},
        'queries':{t:{k:list(v.shape) for k,v in q.items()} for t,q in batch['queries'].items()},
        'phase':list(batch['phase'].shape),'bc':list(batch['bc'].shape),'passed':True}
    # Candidate pools are disjoint and held-out query selection ignores epoch.
    ev=JointCycleDataset(a.cache_root,case_ids=ids[:1],training=False,allow_partial_stats=True)
    ex=ev[0]
    for task in TASKS:
        assert not np.intersect1d(x['queries'][task]['rows'],ex['queries'][task]['rows']).size
    ev.set_epoch(7)
    assert np.array_equal(ex['queries']['velocity']['rows'],ev[0]['queries']['velocity']['rows'])
    result['train_eval_queries_disjoint']=True
    result['eval_epoch_invariant']=True
    _write_json(a.cache_root/'smoke.json',result);print(json.dumps(result,indent=2))


if __name__=='__main__':
    main()

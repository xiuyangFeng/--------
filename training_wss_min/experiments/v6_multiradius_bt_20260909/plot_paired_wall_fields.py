#!/usr/bin/env python3
"""Reproduce and verify M2/MS4 best fields before paired wall visualization.

Read-only run/checkpoint/data inputs. GPU inference requires Slurm. The frozen
single-frame evaluator supplies both metrics and the exact predictions plotted;
all per-case physical/log_z metric leaves must agree with saved original results.
No geometry candidate, new training, velocity prediction or checkpoint selection.
"""
from __future__ import annotations

import gc
import hashlib
import json
import math
import os
from pathlib import Path
import time

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import colors, font_manager
from matplotlib.lines import Line2D
import numpy as np
import torch

from training_wss_min import config as C, dataset as D, evaluate as E
from training_wss_min.tools.run_v6_multiradius_bt_queue import fingerprints

EXP = Path(__file__).resolve().parent
ROOT = EXP.parents[2]
OUT = EXP/'paired_wall_fields_best'
CASES = ('AG/slow/MA_TIAN_YI', 'ILO/ZHANG_JIN_CHUN-1/before', 'ILO/SUN_XU_XIA-1/before')
RUNS = {'M2': ROOT/'training_wss_min/runs/v6_followup_20260909/M2_a5_independent_k3_s1234',
        'MS4': ROOT/'training_wss_min/runs/v6_multiradius_bt_20260909/MS4_m2_multi_r010_r020_r040_bt_s1234'}
VIEWS = ((10., -60.), (10., 120.))
ATOL, RTOL = 5e-6, 5e-6


def digest(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
    return h.hexdigest()


def file_record(path):
    return {'path':str(path),'sha256':digest(path)}


def save_json(path,payload):
    temp=path.with_suffix(path.suffix+'.tmp')
    temp.write_text(json.dumps(payload,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
    temp.replace(path)


def leaves(value,prefix=''):
    if isinstance(value,dict):
        for key,v in value.items():yield from leaves(v,prefix+'.'+key if prefix else key)
    elif isinstance(value,(int,float)) and not isinstance(value,bool):
        yield prefix,value


def verify_metrics(actual,saved):
    a,b=dict(leaves(actual)),dict(leaves(saved))
    checks=[]
    if a.keys()!=b.keys():raise ValueError('per-case metric keys differ')
    for key,reference in b.items():
        value=a[key]
        if isinstance(reference,int):ok=value==reference;tol=0.
        elif not math.isfinite(reference):ok=not math.isfinite(value);tol=0.
        else:tol=ATOL+RTOL*abs(reference);ok=math.isfinite(value) and abs(value-reference)<=tol
        checks.append({'metric':key,'reference':reference if math.isfinite(reference) else None,
                       'recomputed':value if math.isfinite(value) else None,
                       'absolute_difference':abs(value-reference) if math.isfinite(reference) and math.isfinite(value) else None,
                       'tolerance':tol,'passed':bool(ok)})
    return {'passed':all(x['passed'] for x in checks),'checks':checks}


def configure_font():
    path=ROOT/'training_wss_min/experiments/v6_followup_20260909/assets/NotoSansCJK-Regular.ttc'
    if not path.exists():raise FileNotFoundError(path)
    font_manager.fontManager.addfont(str(path))
    plt.rcParams['font.family']=font_manager.FontProperties(fname=path).get_name()
    plt.rcParams['axes.unicode_minus']=False
    plt.rcParams['pdf.fonttype']=42
    return file_record(path)


def orient(pos_mm):
    center=pos_mm.mean(axis=0)
    _,vectors=np.linalg.eigh(np.cov((pos_mm-center).T))
    longitudinal=vectors[:,-1]
    if longitudinal[np.argmax(np.abs(longitudinal))]<0:longitudinal=-longitudinal
    transverse=vectors[:,-2]
    if transverse[np.argmax(np.abs(transverse))]<0:transverse=-transverse
    normal=np.cross(longitudinal,transverse)
    basis=np.column_stack([transverse,normal,longitudinal])
    return (pos_mm-center)@basis,{'center_mm':center.tolist(),'basis_columns':basis.tolist(),
           'method':'PCA geometry-only display rotation; not a model input or anatomical side label'}


def wall(ax,xyz,values,norm,cmap,view,title):
    collection=ax.scatter(*xyz.T,c=values,cmap=cmap,norm=norm,s=1.1,linewidths=0,
                          depthshade=False,rasterized=True)
    low,high=xyz.min(0),xyz.max(0);span=np.maximum(high-low,1e-6)
    pad=.035*span
    ax.set(xlim=(low[0]-pad[0],high[0]+pad[0]),ylim=(low[1]-pad[1],high[1]+pad[1]),zlim=(low[2]-pad[2],high[2]+pad[2]))
    ax.set_box_aspect(span)
    ax.set_proj_type('ortho')
    ax.view_init(elev=view[0],azim=view[1])
    ax.set_axis_off()
    ax.set_title(title,fontsize=10,pad=0)
    return collection


def metric_label(case,model):
    m=case['metrics'][model]
    return f"{model}  R²={m['overall']['r2']:.3f} · MAE={m['overall']['mae']:.2f} Pa"


def save_figure(fig,name):
    paths=[]
    for extension in ('png','pdf'):
        p=OUT/(name+'.'+extension)
        fig.savefig(p,dpi=190,facecolor='white')
        paths.append(file_record(p))
    plt.close(fig)
    return paths


def render_case(uid,case,wss_norm,error_norm):
    xyz,y,m2,ms4=case['display_xyz'],case['truth_pa'],case['M2_pa'],case['MS4_pa']
    short=uid.replace('/','__')
    fig=plt.figure(figsize=(19,10.2));axes=[]
    titles=('CFD 真值',metric_label(case,'M2'),metric_label(case,'MS4'),'M2 − CFD','MS4 − CFD')
    for row,view in enumerate(VIEWS):
        for col,value in enumerate((y,m2,ms4,m2-y,ms4-y)):
            ax=fig.add_subplot(2,5,row*5+col+1,projection='3d');axes.append(ax)
            wall(ax,xyz,value,wss_norm if col<3 else error_norm,'turbo' if col<3 else 'coolwarm',view,titles[col] if row==0 else f'反面 · {titles[col].split("  ")[0]}')
    fig.subplots_adjust(left=.015,right=.985,bottom=.13,top=.89,wspace=-.09,hspace=.0)
    cax=fig.add_axes([.10,.078,.46,.018]);cb=fig.colorbar(plt.cm.ScalarMappable(norm=wss_norm,cmap='turbo'),cax=cax,orientation='horizontal');cb.set_label('WSS (Pa) · 同病例三种场共用色标；颜色按 √WSS 映射，完整范围',fontsize=9)
    cax=fig.add_axes([.65,.078,.27,.018]);cb=fig.colorbar(plt.cm.ScalarMappable(norm=error_norm,cmap='coolwarm'),cax=cax,orientation='horizontal');cb.set_label('有符号误差 (Pa) · 蓝=低估，红=高估；共用对称线性色标',fontsize=9)
    fig.suptitle(f'{uid}\n峰值帧 {case["peak_step"]} · best · 全部 {len(y):,} 个壁面点 · 两个相对视角',fontsize=14,y=.97)
    fig.text(.5,.018,'仅旋转显示几何，不重建或平滑壁面；所有预测已与原始逐例指标核验。单种子例图，不代表总体收益。',ha='center',fontsize=9)
    files=save_figure(fig,short+'__wss_errors')
    true_hot=y>=np.percentile(y,90.)
    codes=[]
    ious=[]
    for pred,label in ((m2,'M2'),(ms4,'MS4')):
        ph=pred>=np.percentile(pred,90.)
        code=np.zeros(len(y),dtype=np.int8)
        code[true_hot&ph]=1;code[true_hot&~ph]=2;code[~true_hot&ph]=3
        computed=float(np.sum(true_hot&ph)/np.sum(true_hot|ph))
        reference=case['metrics'][label]['hotspot']['top10_iou']
        if abs(computed-reference)>1e-12:raise ValueError('Hotspot plot mask differs from recomputed evaluator')
        codes.append(code);ious.append(computed)
    cmap=colors.ListedColormap(['#d5d9df','#009e73','#d55e00','#0072b2'])
    norm=colors.BoundaryNorm([-.5,.5,1.5,2.5,3.5],cmap.N)
    fig=plt.figure(figsize=(19,10.2))
    pred_hots=[(p>=np.percentile(p,90.)).astype(np.int8) for p in (m2,ms4)]
    for row,view in enumerate(VIEWS):
        for col,value in enumerate((true_hot.astype(np.int8),*pred_hots,*codes)):
            ax=fig.add_subplot(2,5,row*5+col+1,projection='3d')
            title=('CFD 高值前 10%', 'M2 自身高值前 10%', 'MS4 自身高值前 10%', f'M2 命中/漏检 · IoU={ious[0]:.3f}',f'MS4 命中/漏检 · IoU={ious[1]:.3f}')[col]
            wall(ax,xyz,value,norm,cmap,view,title if row==0 else '反面 · '+title)
    fig.subplots_adjust(left=.015,right=.985,bottom=.10,top=.89,wspace=-.14,hspace=.0)
    legends=[Line2D([],[],marker='o',linestyle='',markersize=7,color=c,label=l) for c,l in zip(cmap.colors,['其余壁面','前3列：自身热点；后2列：命中','漏检CFD热点','额外预测热点'])]
    fig.legend(handles=legends,loc='lower center',bbox_to_anchor=(.5,.043),ncol=4,frameon=False,fontsize=9)
    fig.suptitle(f'{uid}\n峰值壁面热点位置 · best · 两个相对视角',fontsize=14,y=.97)
    fig.text(.5,.018,'每种场按其自身第90百分位确定热点，与原始 legacy_vertex IoU 一致；热点图不表示幅值准确。',ha='center',fontsize=9)
    files+=save_figure(fig,short+'__hotspots')
    return files


def render_overview(cases,norms):
    fig=plt.figure(figsize=(12,15))
    for row,(uid,case) in enumerate(cases.items()):
        for col,(label,key) in enumerate((('CFD','truth_pa'),('M2','M2_pa'),('MS4','MS4_pa'))):
            ax=fig.add_subplot(3,3,row*3+col+1,projection='3d')
            wall(ax,case['display_xyz'],case[key],norms[uid][0],'turbo',VIEWS[0],label if label=='CFD' else metric_label(case,label))
        fig.text(.01,.88-row*.272,uid,fontsize=10,rotation=90,va='top')
        cax=fig.add_axes([.91,.69-row*.279,.013,.15]);cb=fig.colorbar(plt.cm.ScalarMappable(norm=norms[uid][0],cmap='turbo'),cax=cax);cb.set_label('WSS (Pa)',fontsize=9)
    fig.subplots_adjust(left=.045,right=.90,bottom=.075,top=.91,hspace=.015,wspace=-.16)
    fig.text(.5,.04,'每行同病例三场共用完整范围色标，病例间范围可不同；颜色按 √WSS 映射。',ha='center',fontsize=10)
    fig.suptitle('预先指定病例：CFD、原 M2、老师 MS4 方案\n峰值单帧 WSS · best · 同点全壁面预测已与原始指标核验',fontsize=15,y=.97)
    return save_figure(fig,'paired_wall_overview')


def main():
    if not os.environ.get('SLURM_JOB_ID'):raise RuntimeError('GPU reproduction must run through Slurm')
    if not torch.cuda.is_available():raise RuntimeError('This wrapper must allocate a GPU')
    torch.set_num_threads(2)
    OUT.mkdir(parents=True,exist_ok=True)
    queue=json.loads((EXP/'queue_status.json').read_text());before=fingerprints()
    if queue.get('status')!='complete' or queue.get('source_changed') is not False or queue['source_sha256']!=queue['source_sha256_end'] or before!=queue['source_sha256']:
        raise RuntimeError('Formal queue must be complete with unchanged training sources before plotting')
    font=configure_font()
    provenance={'job_id':os.environ['SLURM_JOB_ID'],'started_at':time.strftime('%Y-%m-%dT%H:%M:%S%z'),
                'script':file_record(Path(__file__).resolve()),'font':font,'cases_preselected':list(CASES),
                'checkpoint':'best','inference_function':'training_wss_min.evaluate._evaluate_partition_frame(return_predictions=True)',
                'metric_tolerance':{'absolute':ATOL,'relative':RTOL,'integer_counts':'exact'},
                'source_sha256_before':before,'runs':{},'cases':{},'passed':False}
    assembled={}
    for label,run in RUNS.items():
        cfg,feat_stats,model,checkpoint=E.load_model_from_run(run,'cuda','best')
        stats=E.load_wss_stats_for_run(run)
        if (cfg.data.target,cfg.data.timesteps,cfg.model.out_dim,cfg.eval.fixed_support,cfg.eval.support_seed,cfg.eval.query_chunk_size,cfg.model.fp_knn,cfg.model.query_decoder,cfg.eval.surface_metric_mode)!=("wss","peak",1,True,1234,16384,3,"interpolate","legacy_vertex"):
            raise ValueError('Plot reproduction protocol differs from original M2/MS4')
        if not set(CASES)<=set(json.loads(Path(cfg.data.split_path).read_text())['test_cases']):raise ValueError('Preselected cases are not all in fixed test partition')
        cases=[]
        for uid in CASES:
            cohort,name=uid.rsplit('/',1)
            cases.append(D.load_case(cohort,name,stats,target=cfg.data.target,target_normalization=cfg.data.target_normalization,
                         data_root=cfg.data.data_root,required_frame_version=cfg.data.required_frame_version,
                         extra_point_features=C.v6_point_features(cfg),point_features_root=cfg.data.point_features_root))
        result=E._evaluate_partition_frame(model,cases,cfg,feat_stats,stats,'cuda',return_predictions=True)
        saved=json.loads((run/'eval/ckpt_best/metrics.json').read_text())['test']
        rec={'checkpoint_epoch':int(checkpoint['epoch']),'checkpoint':file_record(run/'ckpt_best.pt'),
             'original_metrics':file_record(run/'eval/ckpt_best/metrics.json'),'run_config':file_record(run/'config.json'),
             'feature_stats':file_record(run/'feature_stats.json'),'target_stats':file_record(run/'wss_global_stats.json'),
             'metric_verification':{}}
        for case,pred_norm in zip(cases,result['_pred_norm_by_case']):
            uid=case['unit_id']
            check={space:verify_metrics(result['per_case'][uid] if space=='Pa' else result['normalized']['per_case'][uid],
                                      saved['per_case'][uid] if space=='Pa' else saved['normalized']['per_case'][uid]) for space in ('Pa','log_z')}
            rec['metric_verification'][uid]=check
            if not all(v['passed'] for v in check.values()):
                provenance['runs'][label]=rec;save_json(OUT/'verification_failed.json',provenance)
                raise RuntimeError(f'{label}/{uid}: original metric reproduction failed; no figures written')
            pred_pa=np.asarray(D.denormalize_wss(np.asarray(pred_norm,dtype=np.float64),stats),dtype=np.float64)
            if stats.get('method')=='log_z':pred_pa=np.clip(pred_pa,0,None)
            if uid not in assembled:
                xyz=np.asarray(case['pos'],dtype=np.float64)*case['coord_scale_scalar'];display,orientation=orient(xyz)
                assembled[uid]={'truth_pa':np.asarray(case['y_raw'],dtype=np.float64),'truth_norm':np.asarray(case['y_norm'],dtype=np.float64),
                                'pos_norm':np.asarray(case['pos']),'display_xyz':display,'orientation':orientation,
                                'peak_step':case['peak_step'],'coord_scale':case['coord_scale_scalar'],'metrics':{}}
                provenance['cases'][uid]={'bundle':file_record(Path(case['bundle_path'])),'orientation':orientation,'n_wall_points':len(xyz)}
            elif not np.array_equal(assembled[uid]['truth_pa'],case['y_raw']) or not np.array_equal(assembled[uid]['pos_norm'],case['pos']):
                raise ValueError('M2 and MS4 wall point identities/truth differ')
            assembled[uid][label+'_pa']=pred_pa;assembled[uid][label+'_norm']=pred_norm
            assembled[uid]['metrics'][label]=result['per_case'][uid]
        provenance['runs'][label]=rec
        del model,checkpoint,result,cases
        gc.collect();torch.cuda.empty_cache()
    # Every requested original metric has passed before any plot is generated.
    provenance['all_original_case_metric_checks_passed']=True
    provenance['color_scales']={}
    norms={}
    for uid,case in assembled.items():
        max_wss=max(float(case[key].max()) for key in ('truth_pa','M2_pa','MS4_pa'))
        max_error=max(float(np.abs(case[key]-case['truth_pa']).max()) for key in ('M2_pa','MS4_pa'))
        wss_norm=colors.PowerNorm(gamma=.5,vmin=0.,vmax=max_wss,clip=False)
        error_norm=colors.Normalize(vmin=-max_error,vmax=max_error,clip=False)
        norms[uid]=(wss_norm,error_norm)
        provenance['color_scales'][uid]={'WSS_Pa':{'vmin':0.,'vmax':max_wss,'normalization':'PowerNorm gamma0.5 (sqrt color mapping)','shared_across':'this case / CFD / M2 / MS4','display_clipping':False},
            'signed_error_Pa':{'vmin':-max_error,'vmax':max_error,'normalization':'linear symmetric','shared_across':'this case / both models','display_clipping':False},
            'hotspots':'value >= each field own90th percentile, exactly as legacy_vertex evaluator; not a common Pa threshold'}
        path=OUT/(uid.replace('/','__')+'__predictions.npz')
        np.savez_compressed(path,**{k:v for k,v in case.items() if isinstance(v,np.ndarray)})
        provenance['cases'][uid]['arrays']=file_record(path)
        provenance['cases'][uid]['figures']=render_case(uid,case,wss_norm,error_norm)
    provenance['overview']=render_overview(assembled,norms)
    after=fingerprints()
    if after!=before:raise RuntimeError('Frozen source changed during plotting')
    for rec in provenance['runs'].values():
        for key in ('checkpoint','original_metrics','run_config','feature_stats','target_stats'):
            if digest(Path(rec[key]['path']))!=rec[key]['sha256']:raise RuntimeError('Source run artifact changed during plotting')
    provenance.update(passed=True,ended_at=time.strftime('%Y-%m-%dT%H:%M:%S%z'),source_sha256_after=after,
        views=[{'elevation':e,'azimuth':a} for e,a in VIEWS],
        limits='Three cases were specified before this round finished, based on prior-round issues. These visualizations explain local behavior and are not representative performance evidence. Point scatter shows all original wall nodes without remeshing; opposite views expose opposite faces. Hotspot membership is a rank-region comparison, not amplitude accuracy.')
    save_json(OUT/'provenance.json',provenance)
    print(json.dumps({'output':str(OUT),'passed':True,'cases':len(assembled),'models':list(RUNS),'figure_pairs':7},ensure_ascii=False))


if __name__=='__main__':main()

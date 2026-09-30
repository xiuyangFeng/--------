"""Evaluation only: restore historical frozen temporal models and add physical MAE.
No optimizer, no training, no writes to original runs, caches, configs, or sources.
"""
import argparse, hashlib, json, os, socket, sys, time
from pathlib import Path
ap=argparse.ArgumentParser();ap.add_argument('--family',choices=['probe','ft','tt'],required=True);ap.add_argument('--fold',type=int,required=True);ap.add_argument('--arms',nargs='*');ap.add_argument('--device',default='cuda');ap.add_argument('--threads',type=int,default=2);a=ap.parse_args()
ROOT=Path('/public/newhome/cy/Digital_twin/GNN');OUT=Path(__file__).resolve().parent
FAMILIES={
 'probe':('wss_time_adapter_v51_20260923','GNN_timeadapter_frozen_20260923',['Tnull','P-mlp','P-mlp-qx','C-raw']),
 'ft':('wss_time_adapter_ft_v51_20260923','GNN_timeadapter_ft_frozen_20260923',['TL-warm','TL-random']),
 'tt':('wss_time_transformer_v51_20260924_r3','GNN_time_transformer_frozen_20260924_r3',['TT-warm','TT-warm-noattn','TT-raw','TT-raw-noattn'])}
exp,frozen,default_arms=FAMILIES[a.family];SOURCE=ROOT.parent/frozen
sys.path.insert(0,str(SOURCE))
import numpy as np
import torch
from training_wss_min import time_adapter_probe as TA
from training_wss_min import metrics as M
from training_wss_min import dataset as D, config as C, evaluate as E, surface as S
assert Path(TA.__file__).resolve().parent.parent==SOURCE
os.environ.setdefault('OMP_NUM_THREADS','2');torch.set_num_threads(a.threads)
torch.backends.cuda.matmul.allow_tf32=False
torch.backends.cudnn.allow_tf32=(a.family!='probe')
device=torch.device(a.device)

def digest(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(p,x):
 p.parent.mkdir(parents=True,exist_ok=True);tmp=p.with_suffix('.tmp');tmp.write_text(json.dumps(x,indent=2,ensure_ascii=False,allow_nan=False));tmp.replace(p)
def finite(x):assert np.isfinite(x).all()
class Stats:
 def __init__(self):self.rows=[]
 def add(self,uid,t,lp,eps,pk,q):
  p=np.clip(np.exp(lp)-eps,0,None);finite(t);finite(p);assert t.shape==p.shape and len(t)==81
  err=p-t;tm=t.mean(1);var=((t-tm[:,None])**2).mean(1)
  tt=t.mean(0);pp=p.mean(0)
  row=dict(unit_id=uid,n_points=t.shape[1],true_mean=tm.tolist(),true_var=var.tolist(),mse=(err**2).mean(1).tolist(),mae=np.abs(err).mean(1).tolist(),tawss_true_mean=float(tt.mean()),tawss_true_var=float(tt.var()),tawss_mse=float(((pp-tt)**2).mean()),tawss_mae=float(np.abs(pp-tt).mean()))
  self.rows.append(row)
 def result(self,pk,q):
  tm=np.array([r['true_mean'] for r in self.rows]);var=np.array([r['true_var'] for r in self.rows]);mse=np.array([r['mse'] for r in self.rows]);mae=np.array([r['mae'] for r in self.rows]);n=np.array([r['n_points'] for r in self.rows])
  r2=1-mse.mean(0)/(var+(tm-tm.mean(0))**2).mean(0)
  tmean=np.array([r['tawss_true_mean'] for r in self.rows]);tv=np.array([r['tawss_true_var'] for r in self.rows]);te=np.array([r['tawss_mse'] for r in self.rows]);low=np.argsort(q)[:20]
  return dict(cycle_r2cb_pa=float(r2.mean()),trough_r2cb_pa=float(r2[low].mean()),peak_r2cb_pa=float(r2[pk]),tawss_r2cb_pa=float(1-te.mean()/(tv+(tmean-tmean.mean())**2).mean()),tawss_mae_pa=float(np.mean([r['tawss_mae'] for r in self.rows])),cycle_mae_pa=float(mae.mean()),trough_mae_pa=float(mae[:,low].mean()),peak_mae_pa=float(mae[:,pk].mean()),cycle_mae_pooled_pa=float(np.average(mae.mean(1),weights=n)),per_frame_r2cb_pa=r2.tolist(),per_frame_mae_pa=mae.mean(0).tolist(),n_cases=len(self.rows),n_points=int(n.sum()))

arms=a.arms or default_arms;records={};models={};started=time.time()
for arm in arms:
 name=f'{arm}_f{a.fold}_s1234';target=OUT/'results'/exp/(name+'.json')
 if target.exists():
  old=json.loads(target.read_text());assert old['validation']['passed'];print('[skip verified]',name,flush=True);continue
 path=ROOT/'training_wss_min/runs'/exp/name;metric=path/'metrics.json';j=json.loads(metric.read_text());cfg=j['config'];bad=[]
 for n,h in j['code_sha256'].items():
  if digest(SOURCE/'training_wss_min'/n)!=h:bad.append(n)
 assert not bad,('historical code fingerprint changed',bad)
 for key,field in [('split_path','split_sha256'),('frame_stats_path','frame_stats_sha256')]:
  assert digest(cfg[key])==j[field],(name,key)
 assert digest(Path(cfg['cache_dir'])/'manifest.json')==j['cache_manifest_sha256']
 rec=dict(name=name,arm=arm,path=path,original=j,cfg=cfg,target=target,stats=Stats(),anchor=0.,checkpoint_sha256=None,checkpoint=None)
 if a.family=='probe':
  fd=TA.FoldData(cfg);phase=torch.tensor(fd.phase,dtype=torch.float32,device=device)
  if arm!='Tnull':
   ck=path/'head_last.pt';state=torch.load(ck,map_location='cpu',weights_only=False);hc=cfg['head']
   builder=object.__new__(TA.InputBuilder);builder.inputs=state['inputs'];builder.stats={k:(np.array(v['mean']),np.array(v['std'])) for k,v in state['input_stats'].items()};builder.dim=sum(len(v[0]) for v in builder.stats.values())
   model=TA.TimeAdapterHead(builder.dim,phase.shape[1],hc['geom_hidden'],hc['time_hidden'],hc['decoder_hidden'],hc['zero_init_output']).to(device);model.load_state_dict(state['head'],strict=True);model.eval();rec.update(model=model,builder=builder)
  else:ck=None
 elif a.family=='tt':
  from training_wss_min import time_transformer as TT
  ck=path/'ckpt_last.pt';state=torch.load(ck,map_location='cpu',weights_only=False)
  fold=TT._load_fold(cfg);fd=TA.FoldData({**cfg,'phase_features':list(TA.PHASE_KEYS)});phase=torch.tensor(fold['phase'],device=device)
  mean=np.asarray(state['input_mean'],np.float32);std=np.asarray(state['input_std'],np.float32)
  model=TT.TemporalFieldModel(sum(TT.INPUT_DIMS.values()),cfg['architecture']).to(device);model.load_state_dict(state['state_dict'],strict=True);model.eval();rec.update(model=model,mean=mean,std=std,ttfold=fold)
 else:
  from training_wss_min import time_adapter_finetune as FT
  from training_wss_min.models import build_model
  from torch_geometric.nn import knn_interpolate
  ck=path/'ckpt_last.pt';state=torch.load(ck,map_location='cpu',weights_only=False);fd=TA.FoldData(cfg);phase=torch.tensor(fd.phase,dtype=torch.float32,device=device)
  donor=Path(cfg['donor_run']);dcfg,feat,donor_model,_=E.load_model_from_run(donor,'cpu',cfg['donor_checkpoint']);wstats=E.load_wss_stats_for_run(donor)
  if cfg['geometry_init']=='donor':assert digest(donor/E.checkpoint_filename(cfg['donor_checkpoint']))==j['initialization']['checkpoint_sha256']
  torch.manual_seed(int(cfg['seed']));geometry=build_model(dcfg.model,C.input_dim(dcfg))
  if cfg['geometry_init']=='donor':geometry.load_state_dict(donor_model.state_dict())
  del donor_model
  hc=cfg['head'];width={'query_x':int(geometry.head[0].in_features),'peak_ln':1}
  head=TA.TimeAdapterHead(sum(width[k] for k in hc['inputs']),len(cfg['phase_features']),hc['geom_hidden'],hc['time_hidden'],hc['decoder_hidden'],hc['zero_init_output'])
  model=FT.TimeBranchModel(geometry,head,hc['inputs'],cfg['trainable_modules']).to(device)
  mismatch=model.load_state_dict(state['trainable_state'],strict=False);assert not mismatch.unexpected_keys
  assert all(k.startswith('geometry.') and not any(k.startswith('geometry.'+n+'.') for n in cfg['trainable_modules']) for k in mismatch.missing_keys),mismatch
  model.eval();rec.update(model=model,dcfg=dcfg,feat=feat,wstats=wstats)
 if ck:rec.update(checkpoint=str(ck),checkpoint_sha256=digest(ck))
 rec.update(fd=fd,phase=phase);records[arm]=rec
if not records:sys.exit(0)
ids=next(iter(records.values()))['fd'].test_ids
assert all(r['fd'].test_ids==ids for r in records.values())
@torch.no_grad()
def predict(rec,c,uid):
 cfg=rec['cfg'];fd=rec['fd'];pk=fd.peak_index;phase=rec['phase'];base=TA.tnull_base(c['peak_ln'],fd.mu,fd.sd,pk).T
 if a.family=='probe':
  if rec['arm']=='Tnull':return base
  block=rec['builder'].point_block(c);outs=[]
  for s in range(0,len(block),int(cfg['eval']['chunk_points'])):
   g=torch.from_numpy(block[s:s+int(cfg['eval']['chunk_points'])]).to(device);outs.append(TA.anchored(rec['model'](g,phase),pk).double().cpu().numpy())
  return base+np.concatenate(outs).T
 if a.family=='tt':
  ctxhash=hashlib.sha256(c['context_idx'].tobytes()).hexdigest();assert ctxhash==rec['original']['context_contract']['holdout_index_sha256'][uid]
  return TT._predict_case(rec['model'],c,phase,rec['mean'],rec['std'],rec['ttfold'],cfg,device)
 dcfg=rec['dcfg'];cohort,case=uid.rsplit('/',1)
 cc=D.load_case(cohort,case,rec['wstats'],target=dcfg.data.target,target_normalization=dcfg.data.target_normalization,data_root=dcfg.data.data_root,required_frame_version=dcfg.data.required_frame_version,timesteps='peak',extra_point_features=C.v6_point_features(dcfg),point_features_root=getattr(dcfg.data,'point_features_root',None))
 support_n=int(dcfg.data.support_n_points or dcfg.data.wall_n_points);sampling=dcfg.data.support_sampling or dcfg.data.sampling
 sseed=S.stable_seed(dcfg.eval.support_seed,uid,'eval_support');sidx=D.sample_support_indices(cc,dcfg.data,sseed,n_points=support_n,sampling=sampling,stream='eval_support')
 geom=D.case_geometry(cc) if dcfg.data.local_geometry else None;extra={'geometry':torch.from_numpy(np.ascontiguousarray(geom[sidx])).to(device)} if geom is not None else {}
 sx=torch.from_numpy(D.build_features(cc,sidx,dcfg.data.input_features,rec['feat'])).to(device);spos=torch.from_numpy(np.ascontiguousarray(cc['pos'][sidx])).to(device);sb=torch.zeros(len(sidx),dtype=torch.long,device=device)
 model=rec['model'];encoded=model.geometry.encode_support(spos,sx,sb,unit_ids=[uid],epoch=0,global_seed=dcfg.eval.support_seed,evaluation=True,**extra)
 peak=torch.from_numpy(c['peak_ln'].astype(np.float32)).to(device);outs=[];chunk=int(cfg['eval']['chunk_points'])
 for s in range(0,len(peak),chunk):
  m=min(chunk,len(peak)-s);pos=torch.from_numpy(np.ascontiguousarray(cc['pos'][s:s+m])).to(device)
  qx=knn_interpolate(encoded[1],encoded[0],pos,encoded[2],torch.zeros(m,dtype=torch.long,device=device),k=model.geometry.query_interpolation_k)
  outs.append(model.delta(qx,peak[s:s+m],phase,pk).double().cpu().numpy())
 return base+np.concatenate(outs).T

print('[start]',a.family,a.fold,list(records),'cases',len(ids),'gpu',torch.cuda.get_device_name() if device.type=='cuda' else 'cpu',flush=True)
for i,uid in enumerate(ids):
 ts=time.time();first=next(iter(records.values()))
 c=TT._load_case(first['cfg'],uid,first['ttfold'],False) if a.family=='tt' else first['fd'].load_case(uid)
 for arm,rec in records.items():
  lp=predict(rec,c,uid);t=c['tau_raw'] if a.family=='tt' else c['tau'];pk=rec['fd'].peak_index
  dev=float(np.max(np.abs(lp[pk]-c['peak_ln'])));assert dev<1e-8;(rec.update(anchor=max(rec['anchor'],dev)))
  rec['stats'].add(uid,t,lp,rec['fd'].eps,pk,rec['fd'].q_norm)
  # Independently verify streaming statistics against the original metric function on first case.
  if i==0:
   check=TA.cycle_report([t],[lp],rec['fd'].steps,pk,rec['fd'].q_norm,rec['fd'].floor,rec['fd'].eps)['eval_compatible'];mine=rec['stats'].result(pk,rec['fd'].q_norm)
   assert max(abs(mine[k]-check[k]) for k in ['cycle_r2cb_pa','trough_r2cb_pa','peak_r2cb_pa','tawss_r2cb_pa','tawss_mae_pa'])<1e-10
  del lp
 print(f'[case {i+1}/{len(ids)}] {uid} points={len(c["peak_ln"])} seconds={time.time()-ts:.2f}',flush=True)
 del c
for arm,rec in records.items():
 fd=rec['fd'];got=rec['stats'].result(fd.peak_index,fd.q_norm);old=rec['original']['metrics']['eval_compatible'];keys=['cycle_r2cb_pa','trough_r2cb_pa','peak_r2cb_pa','tawss_r2cb_pa','tawss_mae_pa']
 diffs={k:got[k]-old[k] for k in keys};passed=max(abs(x) for x in diffs.values())<=5e-5
 result=dict(schema='time_mae_saved_weights_v1',family=a.family,arm=arm,fold=a.fold,seed=1234,original_run=str(rec['path']),original_metrics_sha256=digest(rec['path']/'metrics.json'),frozen_source=str(SOURCE),code_sha256=rec['original']['code_sha256'],checkpoint=rec['checkpoint'],checkpoint_sha256=rec['checkpoint_sha256'],evaluation=dict(partition='test',n_frames=81,frame_steps=fd.steps.tolist(),truth='raw Pa, not floored',prediction='clip(exp(ln_pred)-eps,0,None)',aggregation='point mean within each case and frame; mean over 81 frames and cases; final report arithmetic mean over 3 folds',checkpoint_selection='original last; no training or model selection'),metrics=got,validation=dict(passed=passed,abs_tolerance=5e-5,differences=diffs,anchor_max_abs=rec['anchor']),per_case=rec['stats'].rows,execution=dict(host=socket.gethostname(),pid=os.getpid(),gpu_uuid=os.environ.get('WSS_GPU_UUID'),cuda_visible_devices=os.environ.get('CUDA_VISIBLE_DEVICES'),elapsed_seconds=time.time()-started,completed_at=time.strftime('%Y-%m-%dT%H:%M:%S%z')))
 write(rec['target'],result);print('[result]',rec['name'],'MAE',got['cycle_mae_pa'],'maxdiff',max(abs(v) for v in diffs.values()),'passed',passed,flush=True)
 if not passed:raise RuntimeError('Original metric reproduction failed for '+rec['name'])

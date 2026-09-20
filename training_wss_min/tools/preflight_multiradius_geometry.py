"""Train138-only neighbor geometry audit; reads coordinates and scale, no CFD labels.

Uses epoch-specific support seeds from the existing sampler. Random-start FPS
uses a separate fixed diagnostic seed per case/epoch, not a claim of replaying
the training RNG stream after stochastic model operations.
"""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
import torch
from torch_geometric.nn import fps
from training_wss_min.dataset import load_split_cases, sample_support_indices
from training_wss_min.surface import stable_seed
from training_wss_min.config import ExpConfig

ROOT=Path(__file__).resolve().parents[2]
EXP=ROOT/'training_wss_min/experiments/v6_multiradius_bt_20260909'
CONFIG=ROOT/'training_wss_min/configs/v6_multiradius_bt_20260909/MS4_m2_multi_r010_r020_r040_bt_s1234.json'

def quantiles(x):return dict(zip(('p10','median','p90'),np.quantile(x,[.1,.5,.9]).tolist()))

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--output',type=Path,default=EXP/'geometry_preflight.json')
    ap.add_argument('--device',default='cpu');ap.add_argument('--epochs',nargs='+',type=int,default=[0,199,399])
    ap.add_argument('--cap',type=int,default=None)
    args=ap.parse_args();torch.set_num_threads(2)
    cfg=ExpConfig.from_json(CONFIG);radii=cfg.model.multiradius_values;cap=args.cap or cfg.model.sa_nsample[-1]
    rows=[];sources={};allcounts=[[] for _ in radii];allsame=[[],[]]
    cases=load_split_cases(cfg.data.split_path,'train');assert len(cases)==138
    for ci,(cohort,name) in enumerate(cases):
        uid=f'{cohort}/{name}';p=Path(cfg.data.data_root)/uid/'bundle.npz'
        with np.load(p,allow_pickle=False) as bundle:
            pos=bundle['wall_coords_norm'].astype(np.float32);scale=float(bundle['coord_scale'])
        sources[uid]=hashlib.sha256(pos.tobytes()+np.float64(scale).tobytes()).hexdigest()
        for epoch in args.epochs:
            idx=sample_support_indices(dict(pos=pos,unit_id=uid),cfg.data,stable_seed(1234,epoch,uid,'support'),n_points=5000,sampling='random',stream='support',epoch=epoch,case_index=ci,run_seed=1234)
            torch.manual_seed(stable_seed(1234,epoch,uid,'geometry_fps_audit')%(2**63-1))
            x=torch.from_numpy(pos[idx]).to(args.device)
            for n in [125,125]:
                ii=fps(x,ratio=n/len(x),random_start=True)[:n];assert len(ii)==n
                x=x[ii]
            ii=fps(x,ratio=32/len(x),random_start=True)[:32];assert len(ii)==32
            # Compute direct differences so centre-self distances are exactly zero.
            d=torch.linalg.vector_norm(x[ii,None,:]-x[None,:,:],dim=-1).cpu().numpy()
            ordered=np.argsort(d,axis=1,kind='stable');groups=[];scale_stats=[]
            for ri,r in enumerate(radii):
                g=[ordered[j][d[j,ordered[j]]<=r][:cap] for j in range(32)]
                count=np.array([len(v) for v in g]);assert np.all(count>=1)
                nonself=count-1;allcounts[ri].extend(nonself.tolist());groups.append(g)
                furthest=np.array([d[j,v].max() for j,v in enumerate(g)])
                scale_stats.append(dict(radius_norm=r,radius_mm=r*scale,
                    nonself=quantiles(nonself),only_self_fraction=float(np.mean(nonself==0)),
                    at_most_one_nonself_fraction=float(np.mean(nonself<=1)),
                    cap_saturated_fraction=float(np.mean(count==cap)),
                    furthest_over_radius=quantiles(furthest/r)))
            pair_stats=[]
            for i in range(len(radii)-1):
                same=[np.array_equal(a,b) for a,b in zip(groups[i],groups[i+1])]
                jac=[len(set(a)&set(b))/len(set(a)|set(b)) for a,b in zip(groups[i],groups[i+1])]
                allsame[i].extend(same)
                pair_stats.append(dict(radii=[radii[i],radii[i+1]],same_fraction=float(np.mean(same)),jaccard_mean=float(np.mean(jac))))
            rows.append(dict(unit_id=uid,epoch=epoch,coord_scale=scale,scales=scale_stats,pairs=pair_stats))
        if (ci+1)%20==0:print(f'geometry {ci+1}/138',flush=True)
    summary=[]
    for ri,r in enumerate(radii):
        v=np.array(allcounts[ri]);summary.append(dict(radius_norm=r,nonself=quantiles(v),only_self_fraction=float(np.mean(v==0)),cap_saturated_fraction=float(np.mean(v==cap-1))))
    strata=[]
    bounds=np.quantile([r['coord_scale'] for r in rows],[1/3,2/3])
    for bucket in range(3):
        subset=[r for r in rows if int(np.searchsorted(bounds,r['coord_scale'],side='right'))==bucket]
        strata.append(dict(coord_scale_tertile=bucket+1,n_case_epochs=len(subset),
            only_self_by_radius=[float(np.mean([r['scales'][i]['only_self_fraction'] for r in subset])) for i in range(len(radii))],
            same_neighbors_by_adjacent_pair=[float(np.mean([r['pairs'][i]['same_fraction'] for r in subset])) for i in range(len(radii)-1)]))
    payload=dict(schema='ms_geometry_preflight_v1',partition='train138',n_cases=len(cases),n_case_epochs=len(rows),
        labels_read=False,radii=list(radii),cap=cap,sampling_epochs=args.epochs,
        fps_note=__doc__,summary=summary,pair_same_fraction=[float(np.mean(s)) for s in allsame],
        coord_scale_tertile_boundaries=bounds.tolist(),strata=strata,
        input_geometry_sha256=sources,config_sha256=hashlib.sha256(CONFIG.read_bytes()).hexdigest(),rows=rows)
    args.output.parent.mkdir(parents=True,exist_ok=True);args.output.write_text(json.dumps(payload,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps({k:payload[k] for k in ('n_cases','n_case_epochs','summary','pair_same_fraction')},ensure_ascii=False))

if __name__=='__main__':main()

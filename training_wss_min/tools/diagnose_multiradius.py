"""Read-only best-checkpoint branch/attention diagnostics on eight train cases.

Does not evaluate fit or select a radius, checkpoint or case-dependent mask.
Distances are reported per case using its existing coordinate scale. Projection
norms describe activation of a route, not causal feature importance.
"""
from __future__ import annotations
import gc
import hashlib
import json
import os
import torch
from training_wss_min import config as C, dataset as D
from training_wss_min.baseline_models import build_baseline_model
from training_wss_min.tools.run_v6_multiradius_bt_queue import EXP,MATRIX,ROOT,fingerprints,save_json

def main():
    if not os.environ.get('SLURM_JOB_ID'):raise RuntimeError('Use Slurm for GPU diagnostics')
    torch.set_num_threads(2)
    hashes=fingerprints();q=json.loads((EXP/'queue_status.json').read_text())
    assert q['status']=='complete' and q['source_sha256']==q['source_sha256_end']==hashes
    ids=json.loads((EXP/'runtime_preflight.json').read_text())['cases']
    matrix=json.loads(MATRIX.read_text());cfg=C.ExpConfig.from_json(MATRIX.parent/matrix['arms'][0]['config'])
    stats=D.load_wss_stats(cfg.data.wss_stats_path)
    alltrain={f'{c}/{n}' for c,n in D.load_split_cases(cfg.data.split_path,'train')};assert set(ids)<=alltrain
    cases=[]
    for uid in ids:
        cohort,name=uid.split('/',2)[:2],uid.split('/',2)[2]
        cases.append(D.load_case('/'.join(cohort),name,stats,data_root=cfg.data.data_root,
            required_frame_version=cfg.data.required_frame_version,extra_point_features=cfg.data.input_features,
            point_features_root=cfg.data.point_features_root))
    results={}
    for arm in matrix['arms']:
        ac=C.ExpConfig.from_json(MATRIX.parent/arm['config']);run=ROOT/'training_wss_min/runs'/arm['run_name']
        checkpoint=run/'ckpt_best.pt';ck=torch.load(checkpoint,map_location='cpu',weights_only=False)
        state=ck.get('model_state',ck.get('model',ck.get('state_dict',ck)))
        m=build_baseline_model(ac.model,25);m.load_state_dict(state,strict=True);m.eval().cuda()
        feat_stats=json.loads((run/'feature_stats.json').read_text());ds=D.WSSMinDataset(cases,ac.data,feat_stats,training=False,base_seed=1234)
        sa=m.sa[-1];c=sa.channels
        fusion=None if sa.fuse is None else dict(block_frobenius_norm=[float(sa.fuse.weight[:,i*c:(i+1)*c].norm()) for i in range(len(sa.radii))],bias_norm=float(sa.fuse.bias.norm()))
        percase=[]
        for j,case in enumerate(cases):
            sample=ds[j];pos=sample['support_pos'].cuda();x=sample['support_x'].cuda();batch=torch.zeros(len(pos),dtype=torch.long,device=pos.device)
            captured={}
            hooks=[]
            for i,context in enumerate(sa.contexts):
                output_module=sa.pointwise[i] if sa.pointwise is not None else context
                hooks.append(output_module.register_forward_hook(lambda mod,args,out,i=i:captured.__setitem__(i,out.detach())))
            with torch.no_grad():m.encode_support(pos,x,batch,unit_ids=[case['unit_id']],global_seed=1234,evaluation=True)
            for hook in hooks:hook.remove()
            branches=[]
            for i,context in enumerate(sa.contexts):
                layers=[dict(gamma_attention=float(layer.gamma_attention),gamma_ffn=float(layer.gamma_ffn),
                    attention_entropy=float(layer.last_attention_entropy),
                    attention_mean_distance_norm=float(layer.last_attention_mean_distance),
                    attention_mean_distance_mm=float(layer.last_attention_mean_distance)*case['coord_scale_scalar']) for layer in context.layers]
                row,col=sa.last_group_edges[i];counts=torch.bincount(row,minlength=32)
                branches.append(dict(scale=i,radius_norm=sa.radii[i],radius_mm=sa.radii[i]*case['coord_scale_scalar'],
                    actual_neighbors_mean=float(counts.float().mean()),only_self_fraction=float((counts==1).float().mean()),
                    tokens_rms=float(captured[i].float().square().mean().sqrt()),layers=layers,
                    ffn_gammas=[float(layer.gamma_ffn) for layer in sa.pointwise[i]] if sa.pointwise is not None else []))
            percase.append(dict(unit_id=case['unit_id'],coord_scale=case['coord_scale_scalar'],branches=branches))
        results[arm['id']]=dict(checkpoint=str(checkpoint),checkpoint_sha256=hashlib.sha256(checkpoint.read_bytes()).hexdigest(),fusion=fusion,cases=percase)
        del m,sa,state,ck;gc.collect();torch.cuda.empty_cache()
        print(f"diagnosed {arm['id']}",flush=True)
    assert fingerprints()==hashes
    save_json(EXP/'branch_diagnostics_best.json',dict(passed=True,job_id=os.environ['SLURM_JOB_ID'],
        partition='train',n_cases=len(cases),checkpoint='best',fit_labels_used=False,
        source_sha256=hashes,arms=results,limits=__doc__))

if __name__=='__main__':main()

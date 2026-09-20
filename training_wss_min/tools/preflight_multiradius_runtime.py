"""Real batch8/support5000 GPU wiring, backward and legacy compatibility checks."""
from __future__ import annotations
import gc
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys
import torch
from training_wss_min import config as C
from training_wss_min import dataset as D
from training_wss_min.baseline_models import build_baseline_model
from training_wss_min.tools.run_v6_multiradius_bt_queue import fingerprints, save_json, EXP, MATRIX, ROOT

ANCHOR=ROOT/'training_wss_min/configs/v6_followup_20260909/M2_a5_independent_k3_s1234.json'
LEGACY=ROOT/'training_wss_min/experiments/v6_followup_20260909/source_snapshot_13205/training_wss_min/baseline_models.py'
RUN=ROOT/'training_wss_min/runs/v6_followup_20260909/M2_a5_independent_k3_s1234'

def parameter_count(module):return sum(p.numel() for p in module.parameters())

def main():
    if not os.environ.get('SLURM_JOB_ID'):raise RuntimeError('Use Slurm for GPU preflight')
    torch.set_num_threads(2);start_hashes=fingerprints()
    device=torch.device('cuda');cfg=C.ExpConfig.from_json(ANCHOR)
    stats=D.load_wss_stats(cfg.data.wss_stats_path)
    pairs=D.load_split_cases(cfg.data.split_path,'train')
    # Select across cohort families, then fill eight cases from the fixed train list.
    chosen=[]
    for family in ('AG','AAA','ILO'):
        chosen.extend([p for p in pairs if p[0].startswith(family+'/')][:2])
    chosen.extend([p for p in pairs if p not in chosen][:8-len(chosen)])
    cases=[D.load_case(c,n,stats,target=cfg.data.target,target_normalization=cfg.data.target_normalization,
        data_root=cfg.data.data_root,required_frame_version=cfg.data.required_frame_version,
        extra_point_features=cfg.data.input_features,point_features_root=cfg.data.point_features_root) for c,n in chosen]
    feat_stats=json.loads((RUN/'feature_stats.json').read_text())
    ds=D.WSSMinDataset(cases,cfg.data,feat_stats,training=True,base_seed=1234)
    b=D.collate([ds[i] for i in range(len(ds))]);assert len(ds)==8
    for k,v in b.items():
        if isinstance(v,torch.Tensor):b[k]=v.to(device)
    # Frozen pre-edit module provides a direct legacy oracle, not a self-comparison.
    spec=importlib.util.spec_from_file_location('training_wss_min._ms_legacy_models',LEGACY)
    legacy=importlib.util.module_from_spec(spec);sys.modules[spec.name]=legacy;spec.loader.exec_module(legacy)
    torch.manual_seed(1234);old=legacy.build_baseline_model(cfg.model,25).eval().to(device)
    torch.manual_seed(1234);new=build_baseline_model(cfg.model,25).eval().to(device)
    old_state=old.state_dict();new_state=new.state_dict()
    assert old_state.keys()==new_state.keys()
    assert all(torch.equal(old_state[k],new_state[k]) for k in old_state)
    def forward(m):
        return m.forward_support_query(b['support_pos'],b['support_x'],b['support_batch'],
            b['pos'],b['x'],b['batch'],unit_ids=b['unit_ids'],epoch=0,global_seed=1234,evaluation=not m.training)
    with torch.no_grad():
        o,n=forward(old),forward(new)
        initialization_output_max_abs=float((o-n).abs().max())
        initialization_repeat_max_abs=float((o-forward(old)).abs().max())
        torch.testing.assert_close(o,n,rtol=2e-5,atol=2e-6)
        checkpoint=torch.load(RUN/'ckpt_best.pt',map_location='cpu',weights_only=False)
        state=checkpoint.get('model_state',checkpoint.get('model',checkpoint.get('state_dict',checkpoint)))
        old.load_state_dict(state,strict=True);new.load_state_dict(state,strict=True)
        o,n=forward(old),forward(new)
        checkpoint_output_max_abs=float((o-n).abs().max())
        checkpoint_repeat_max_abs=float((o-forward(old)).abs().max())
        torch.testing.assert_close(o,n,rtol=2e-5,atol=2e-6)
    torch.manual_seed(1234);base=build_baseline_model(cfg.model,25)
    common={k:v.clone() for k,v in base.state_dict().items() if not k.startswith('sa.2.')}
    del old,new,base,old_state,new_state,o,n,checkpoint,state;gc.collect();torch.cuda.empty_cache()
    matrix=json.loads(MATRIX.read_text());records={};states={}
    for arm in matrix['arms']:
        aid=arm['id'];ac=C.ExpConfig.from_json(MATRIX.parent/arm['config'])
        torch.manual_seed(1234);m=build_baseline_model(ac.model,25)
        initial=m.state_dict()
        differences=[k for k,v in common.items() if k not in initial or not torch.equal(v,initial[k])]
        assert not differences,(aid,'inherited initialization differences',differences)
        states[aid]={k:v.clone() for k,v in initial.items() if k.startswith('sa.2.')}
        m=m.to(device);optim=torch.optim.AdamW(m.parameters(),lr=.001,weight_decay=.0001)
        torch.cuda.reset_peak_memory_stats();steps=[]
        for step in range(3):
            m.train();optim.zero_grad(set_to_none=True)
            with torch.autocast('cuda',dtype=torch.float16):
                out=forward(m);target=b['y'];residual=target-out
                loss=((out-target)**2).mean()+.2*torch.maximum(.9*residual,-.1*residual).mean()
            assert torch.isfinite(out).all() and torch.isfinite(loss)
            loss.backward()
            finite=all(torch.isfinite(p.grad).all() for p in m.parameters() if p.grad is not None)
            assert finite,(aid,'nonfinite gradients')
            tracked={k:float(p.grad.detach().float().norm()) if p.grad is not None else None for k,p in m.named_parameters() if k.startswith('sa.2.')}
            steps.append(dict(step=step,loss=float(loss),gradients=tracked))
            torch.nn.utils.clip_grad_norm_(m.parameters(),1.);optim.step()
        assert any(v is not None and v>0 for k,v in tracked.items() if 'branches.0.' in k or 'locals.0.' in k), (aid,'first scale inactive after3steps',list(tracked))
        m.eval()
        with torch.no_grad():assert torch.isfinite(forward(m)).all()
        records[aid]=dict(parameters=parameter_count(m),sa3_parameters=parameter_count(m.sa[-1]),
            max_memory_allocated_mib=torch.cuda.max_memory_allocated()/1024**2,
            common_initialization_exact=True,steps=steps)
        print(aid,records[aid]['parameters'],records[aid]['max_memory_allocated_mib'],flush=True)
        del m,optim,out,loss,initial;gc.collect();torch.cuda.empty_cache()
    assert states['MS4'].keys()==states['MS5'].keys()
    assert all(torch.equal(states['MS4'][k],states['MS5'][k]) for k in states['MS4'])
    ratio=abs(records['MS6']['sa3_parameters']/records['MS4']['sa3_parameters']-1)
    assert ratio<=.01,('FFN parameter mismatch',ratio)
    end_hashes=fingerprints();assert start_hashes==end_hashes,'source changed during runtime preflight'
    save_json(EXP/'runtime_preflight.json',dict(passed=True,job_id=os.environ['SLURM_JOB_ID'],
        cases=[c['unit_id'] for c in cases],batch_cases=8,support_per_case=5000,query_per_case=5000,
        old_initialization_weights_exact=True,
        legacy_gpu_output_check=dict(rtol=2e-5,atol=2e-6,
            initialization_max_abs=initialization_output_max_abs,initialization_repeat_max_abs=initialization_repeat_max_abs,
            checkpoint_max_abs=checkpoint_output_max_abs,checkpoint_repeat_max_abs=checkpoint_repeat_max_abs,
            note='GPU scatter/interpolation numerical nondeterminism; CPU frozen-source bit-exact comparison is in module tests'),
        same_radius_initialization_exact=True,ffn_parameter_relative_difference=ratio,
        source_sha256=start_hashes,source_sha256_end=end_hashes,arms=records))

if __name__=='__main__':main()

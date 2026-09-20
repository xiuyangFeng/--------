"""Check all geometry input tensors and one CPU backward pass, without fitting.

No optimizer step, training run, checkpoint or test metric is produced.
Statistics use only explicit train136; every test34 and low-coverage wall row
is retained and checked with the same feature/mask schema.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import numpy as np
import torch
from training_wss_min import config as C, dataset as D, longitudinal_geometry as L
from training_wss_min.baseline_models import MLPRegressor


def main():
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--sidecar',type=Path,required=True)
    ap.add_argument('--view-root',type=Path,required=True);ap.add_argument('--split',type=Path,required=True)
    ap.add_argument('--out',type=Path,required=True);a=ap.parse_args();a.out.mkdir(parents=True,exist_ok=True)
    split=json.loads(a.split.read_text());wss=D.load_wss_stats(a.view_root/'wss_global_stats_train136.json')
    names=('x','y','z',*L.FEATURE_KEYS)
    # Read actual bundles with the production loader, not an imitation of it.
    tr=D.load_partition(str(a.split),'train',wss,data_root=a.view_root,
        extra_point_features=L.FEATURE_KEYS,point_features_root=a.sidecar)
    stats=D.compute_feature_stats(tr,names);L.validate_frozen_stats(stats,names,split['train_cases'])
    (a.out/'feature_stats.json').write_text(json.dumps(stats,indent=2)+'\n')
    rows=[];smoke=None
    def check(c,partition):
        nonlocal smoke
        n=len(c['pos']);x=D.build_features(c,np.arange(n),names,stats)
        if not np.isfinite(x).all():raise ValueError(f'nonfinite input {c["unit_id"]}')
        if x.shape!=(n,len(names)) or len(c['y_raw'])!=n or len(c['y_norm'])!=n:
            raise ValueError('feature or WSS supervision rows were dropped')
        counts={}
        for name in L.VALUE_KEYS:
            v=c[name+'_valid'].astype(bool);value=x[:,names.index(name)];mask=x[:,names.index(name+'_valid')]
            if not np.array_equal(mask,v.astype(float)) or np.any(value[~v]!=0):raise ValueError('mask/value encoding mismatch')
            counts[name]=int(v.sum())
        rows.append({'canonical_id':c['unit_id'],'partition':partition,'wall_points':n,'features':x.shape[1],
                     'all_finite':True,'supervision_rows_retained':True,'valid_counts':counts})
        if c['unit_id']=='AAA/ruputer/FENG_LI_XIN':
            # Include valid and missing geometry points; neither is filtered.
            valid=c['geom_ref_log_area_valid'];ix=np.r_[np.flatnonzero(valid)[:32],np.flatnonzero(~valid)[:32]]
            model=MLPRegressor(in_dim=len(names),hidden=(32,32),out_dim=1)
            features=torch.tensor(x[ix],requires_grad=True);pos=torch.tensor(c['pos'][ix]);batch=torch.zeros(len(ix),dtype=torch.long)
            pred=model(pos,features,batch);target=torch.tensor(c['y_norm'][ix]).reshape_as(pred)
            loss=torch.mean((pred-target)**2);loss.backward()
            if not torch.isfinite(loss) or any(p.grad is not None and not torch.isfinite(p.grad).all() for p in model.parameters()):
                raise ValueError('nonfinite backward pass')
            smoke={'case':c['unit_id'],'points':len(ix),'valid_geometry_points':int(valid[ix].sum()),
                   'missing_geometry_points':int((~valid[ix]).sum()),'finite_loss':True,'finite_gradients':True,
                   'supervision_used_for_missing_geometry':True,'optimizer_steps':0,'model':'MLPRegressor CPU interface smoke'}
    for c in tr:check(c,'train')
    del tr
    for cid in split['test_cases']:
        cohort,case=cid.rsplit('/',1)
        c=D.load_case(cohort,case,wss,data_root=a.view_root,extra_point_features=L.FEATURE_KEYS,point_features_root=a.sidecar)
        check(c,'test')
    result={'status':'passed','case_count':len(rows),'train_cases':len(split['train_cases']),'test_cases':len(split['test_cases']),
            'feature_count':len(names),'longitudinal_value_mask_channels':len(L.FEATURE_KEYS),'training_started':False,
            'statistics_train_only':True,'all_cases_same_schema':True,'all_wall_supervision_retained':True,
            'sidecar':str(a.sidecar.resolve()),'view_root':str(a.view_root.resolve()),'split_sha256':L.sha256_file(a.split),
            'cpu_backward_smoke':smoke,'cases':rows}
    if len(rows)!=170 or smoke is None:raise ValueError('incomplete cohort or smoke')
    (a.out/'preflight.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k!='cases'},ensure_ascii=False))


if __name__=='__main__':main()

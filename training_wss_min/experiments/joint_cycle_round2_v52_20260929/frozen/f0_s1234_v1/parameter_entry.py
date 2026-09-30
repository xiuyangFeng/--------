import json
from pathlib import Path
import torch
from training_wss_min.joint_cycle_round2 import build, validate_config
root=Path(__file__).resolve().parent
torch.set_num_threads(2)
arms={}
for path in json.loads((root/'configs.json').read_text()):
    cfg=json.loads(Path(path).read_text())
    validate_config(cfg)
    model=build(cfg,json.loads(Path(cfg['data']['stats_path']).read_text()))
    contract=getattr(model,'round2_contract',{'module':'none','residual_parameters':0})
    arms[cfg['arm']]={'total_trainable_parameters':sum(p.numel() for p in model.parameters() if p.requires_grad),
                      'added_parameters':contract['residual_parameters'],'module_contract':contract}
    del model
pairs=[]
for control,treatment in [('UT0','UT1'),('WT0','WT1'),('U0','U1'),('W0','W1')]:
    n0,n1=arms[control]['added_parameters'],arms[treatment]['added_parameters']
    relative=abs(n0-n1)/max(n0,n1)
    if relative>0.05:raise ValueError('Added parameter mismatch: '+control+'/'+treatment)
    pairs.append({'control':control,'treatment':treatment,'control_added_parameters':n0,
                  'treatment_added_parameters':n1,'relative_difference':relative,'passed':True})
out={'status':'passed','device':'cpu','seed':1234,'arms':arms,'pairs':pairs,
     'scope':'parameter/config construction only; GPU smoke still required'}
(root/'parameter_validation.json').write_text(json.dumps(out,indent=2)+'\n')
print(json.dumps(out,indent=2))

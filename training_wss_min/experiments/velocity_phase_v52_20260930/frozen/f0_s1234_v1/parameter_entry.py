import json
from pathlib import Path
import torch
from training_wss_min.velocity_phase import build,validate_config
root=Path(__file__).resolve().parent
torch.set_num_threads(2)
arms={}
for filename in json.loads((root/'configs.json').read_text()):
    cfg=json.loads(Path(filename).read_text()); validate_config(cfg)
    model=build(cfg,json.loads(Path(cfg['data']['stats_path']).read_text()))
    contract=getattr(model,'velocity_contract',None)
    if contract is None: contract=getattr(model,'velocity_phase_contract',None)
    if not isinstance(contract,dict): raise ValueError('missing velocity model parameter contract')
    total=sum(p.numel() for p in model.parameters() if p.requires_grad)
    added=contract.get('added_parameters')
    if not isinstance(added,int): raise ValueError('missing active added-parameter count')
    arms[cfg['arm']]={'total_parameters':total,'added_parameters':added,
                      'graph_added_parameters':contract.get('graph_added_parameters'),'module_contract':contract}
    del model
pairs=[]
for group in [('A0','A1'),('G00','G01','G10','G11')]:
    for i,a in enumerate(group):
        for b in group[i+1:]:
            key='graph_added_parameters' if a.startswith('G') else 'added_parameters'
            x,y=arms[a][key],arms[b][key]
            if not isinstance(x,int) or not isinstance(y,int):raise ValueError('missing active '+key)
            delta=abs(x-y)/max(x,y,1)
            if delta>.05:raise ValueError('active parameter mismatch '+a+'/'+b)
            pairs.append({'arms':[a,b],'relative_difference':delta,'passed':True})
out={'status':'passed','arms':arms,'pairs':pairs,'scope':'CPU construction and active parameter gate; GPU smoke pending'}
(root/'parameter_validation.json').write_text(json.dumps(out,indent=2)+'\n')
print(json.dumps(out,indent=2))

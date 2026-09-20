import json,sys
import numpy as np
from pathlib import Path
sys.path.insert(0,'/public/newhome/cy/Digital_twin/GNN')
from training_wss_min import longitudinal_geometry as L
SIDE=Path('/public/newhome/cy/Digital_twin/GNN/training_wss_min/experiments/wss_x5d_longitudinal_20260917/frozen_longitudinal_features')
split=json.load(open('/public/newhome/cy/Digital_twin/GNN/data_wss_v5/views_v5_1/wss_min_view_v1/split_V5_train136_test34.json'))
V=list(L.VALUE_KEYS)
REF=[v for v in V if v.startswith('geom_ref_')]
NEW4=['geom_ref_roundness','geom_ref_eccentricity','geom_ref_upstream_min_distance','geom_ref_downstream_min_distance']
SUB={'全部 32 列（16 value）':V,'ref 8':REF,'ref 8 + pc_roundness':REF+['geom_pc_roundness'],
     '"新"4（圆度/偏心/上下游距离, ref）':NEW4,
     'ref 4（log_area/坡度/上下游面积比）':['geom_ref_log_area','geom_ref_area_slope','geom_ref_upstream_min_area_ratio','geom_ref_downstream_min_area_ratio'],
     '只 geom_ref_log_area':['geom_ref_log_area']}
for part in ('train_cases','test_cases'):
    tot={k:[0,0] for k in SUB}
    for cid in split[part]:
        with np.load(SIDE/cid/'features.npz',allow_pickle=False) as z:
            m={v:z[f'wall_{v}_valid'].astype(bool) for v in V}
        n=len(next(iter(m.values())))
        for k,keys in SUB.items():
            ok=np.ones(n,bool)
            for v in keys: ok&=m[v]
            tot[k][0]+=int(ok.sum()); tot[k][1]+=n
    print(f'--- {part}（{len(split[part])} 例）逐点"全部所选通道同时有效"占比')
    for k,(a,b) in tot.items(): print(f'  {k:38s} {a/b:.4f}')
    print()

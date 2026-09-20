"""Generate/check the six authorized MS arms; never submit a job here."""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
from pathlib import Path

from training_wss_min import config as C
from training_wss_min.tools.prepare_v6_singleframe_matrix import realised_diff

ROOT = Path(__file__).resolve().parents[2]
PREFIX = "v6_multiradius_bt_20260909"
OUT = ROOT / "training_wss_min/configs" / PREFIX
ANCHOR = ROOT / "training_wss_min/configs/v6_followup_20260909/M2_a5_independent_k3_s1234.json"
SPECS = [
    ("MS1", "single_r020", "MS0", [.2], False, False, "单半径新SA3及中心条件桥接；无新增全局BT"),
    ("MS2", "single_r020_bt", "MS1", [.2], True, False, "单半径新SA3加全局BT；与MS1比较"),
    ("MS3", "multi_r010_r020_r040", "MS1", [.1,.2,.4], False, False, "多半径分组及融合；无新增全局BT"),
    ("MS4", "multi_r010_r020_r040_bt", "MS3", [.1,.2,.4], True, False, "老师主方案：多半径各支独立BT后按中心融合"),
    ("MS5", "multi_r020x3_bt", "MS4", [.2,.2,.2], True, False, "相同半径三支BT；与MS4固定容量检验多尺度"),
    ("MS6", "multi_r010_r020_r040_ffn", "MS4", [.1,.2,.4], False, True, "逐token近等容量FFN替代BT；保持分组及融合"),
]

def build(check=False):
    anchor=json.loads(ANCHOR.read_text()); configs={}; arms=[]
    for aid, slug, parent, radii, bt, ffn, hypothesis in SPECS:
        cfg=copy.deepcopy(anchor)
        cfg['name']=f"{PREFIX}/{aid}_m2_{slug}_s1234"
        cfg['notes']=f"{aid}: {hypothesis}；仅峰值WSS，seed1234，400epoch；G/M6未启用。"
        cfg['model'].update(multiradius_values=radii, multiradius_bottleneck=bt,
            multiradius_ffn=ffn, bottleneck_transformer=False,
            bottleneck_layers=1, bottleneck_heads=8, bottleneck_ffn_ratio=2,
            bottleneck_dropout=0., bottleneck_pos_enc='input_features',
            bottleneck_geo_bias=True, bottleneck_residual_scale_init=.001)
        # Train138-only geometry preflight: cap16 made .2/.4 identical at55.52%
        # of centres; uniform cap32 reduces this to4.88%, before any MS training.
        cfg['model']['sa_nsample'][-1]=32
        C.ExpConfig.from_dict(cfg)
        assert cfg['data']==anchor['data'] and cfg['train']==anchor['train'] and cfg['eval']==anchor['eval']
        configs[aid]=cfg
        arms.append(dict(id=aid,config=Path(cfg['name']).name+'.json',run_name=cfg['name'],
            baseline=parent,category='model',hypothesis=hypothesis,anchor='M2',
            changes=realised_diff(anchor,cfg),
            changes_vs_parent=realised_diff(anchor if parent=='MS0' else configs[parent],cfg),
            input_dim=25))
    manifest=dict(matrix_id=PREFIX,anchor_config=str(ANCHOR.relative_to(ROOT)),
        anchor_sha256=hashlib.sha256(ANCHOR.read_bytes()).hexdigest(),
        reference=dict(id='MS0',run_name=anchor['name']),
        authorised_scope='MS1-MS6 only; no G/M6/new-geometry/conditional arms',
        protocol=dict(seed=1234,epochs=400,target='wss',timesteps='peak',batch_cases=8,
            support_n_points=5000,query_n_points=5000,selection_rule='train_loss',
            evaluation='best and last; test34 exposed development comparison',
            grouping='shared FPS32; nearest within normalized radii; cap32; shared distance ordering'),arms=arms)
    payloads={OUT/a['config']:configs[a['id']] for a in arms}
    payloads[OUT/'matrix.json']=manifest
    if not check: OUT.mkdir(parents=True,exist_ok=True)
    for path,payload in payloads.items():
        if check:
            if not path.is_file() or json.loads(path.read_text())!=payload: raise ValueError(f'Stale artifact: {path}')
        else:path.write_text(json.dumps(payload,indent=2,ensure_ascii=False)+'\n')
    print(f"{'Checked' if check else 'Generated'} 6 arms at {OUT}")

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--check',action='store_true')
    build(**vars(ap.parse_args()))

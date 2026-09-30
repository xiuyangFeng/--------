"""把 wss_v5/contract.py 切到 v5.1：新快照根、train136 split、schema 版本。只在 26 例重跑完成并瘦身后执行一次（幂等）。"""
import sys, time
from pathlib import Path
ROOT = Path('/public/newhome/cy/Digital_twin/GNN'); p = ROOT / 'wss_v5/contract.py'; s = p.read_text()
snap = sys.argv[1] if len(sys.argv) > 1 else f'anatomy_pointcloud_v5_1_{time.strftime("%Y%m%d")}'
pairs = [
    ('SCHEMA_VERSION = "wss_v5.0"', 'SCHEMA_VERSION = "wss_v5.1"  # 2026-09: 26 例 RCR 出口面积修正重算 + 剔除 2 例重复（train136/test34）'),
    ('SNAPSHOT_NAME = "anatomy_pointcloud_v5_20260906"', f'SNAPSHOT_NAME = "{snap}"  # v5.0 快照 anatomy_pointcloud_v5_20260906 原样保留'),
    ('SPLIT_PATH = ROOT / "wss_pinn/configs/splits/split_WSS_PINN_AG_AAA_ILO_bc_rcr_v4_anatomy_only_train138_test34_s1234.json"',
     'SPLIT_PATH = ROOT / "wss_pinn/configs/splits/split_WSS_PINN_AG_AAA_ILO_bc_rcr_v4_anatomy_only_train136_test34_s1234.json"'),
]
done = 0
for old, new in pairs:
    if old in s: s = s.replace(old, new); done += 1
    else: assert new.split('#')[0].strip() in s, old
p.write_text(s); print(f'contract.py: {done} 处已切换（其余已是 v5.1）; SNAPSHOT_NAME={snap}')

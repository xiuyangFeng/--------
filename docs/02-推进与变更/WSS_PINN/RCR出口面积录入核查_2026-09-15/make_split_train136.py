"""从冻结的 train138/test34 源 split 派生 train136/test34：剔除 LIU_WEN_QI（CFD 即 ZHU_ZI_HAI 的副本）与 HOU_SHEN_QIAN（KANG_XI_MING 同面的另一次求解），
登记两组重复几何；cv 折里同步移除并重算分层计数。只生成文件，不改 wss_v5/contract.py（母库刷新时再切换）。"""
import hashlib, json, time
from pathlib import Path
ROOT = Path('/public/newhome/cy/Digital_twin/GNN'); SRC = ROOT / 'wss_pinn/configs/splits/split_WSS_PINN_AG_AAA_ILO_bc_rcr_v4_anatomy_only_train138_test34_s1234.json'
DST = SRC.with_name(SRC.name.replace('train138', 'train136'))
DROP = {
    'AAA/unruputer/LIU_WEN_QI': '2026-09-15: H5 壁面坐标/节点号/81 帧 WSS 与压力/体网格/入口通量与 ZHU_ZI_HAI 逐位相同，LIU_WEN_QI.stl 是另一副解剖 → .cas 是 ZHU_ZI_HAI 网格的副本，标签不属于本例；真实解剖未算过，可从 STL 重划网重算后收回',
    'AG/slow/HOU_SHEN_QIAN': '2026-09-15: 与 KANG_XI_MING 同一张面（11379 点互距中位 0 mm）两次划网求解，WSS 逐帧相关 0.47–0.98；KANG_XI_MING.stl 为原始导出、HOU_SHEN_QIAN.stl 为 VTK 重导出，且松弛因子 0.5/0.5 异常 → 同一解剖两套矛盾标签，剔除本例',
}
EVID = 'docs/02-推进与变更/WSS_PINN/WSS_V5_训练实验跟踪.md §24'
def stratum(c):
    if c.startswith('AAA/'): return '/'.join(c.split('/')[:2])
    if c.startswith('AG/'): return 'AG'
    return 'ILO/' + c.split('/')[1].rsplit('-', 1)[-1]
d = json.loads(SRC.read_text()); assert all(c in d['train_cases'] for c in DROP)
d['train_cases'] = [c for c in d['train_cases'] if c not in DROP]
for c, why in DROP.items(): d['excluded_cases'].append({'canonical_id': c, 'evidence': EVID, 'previous_role': 'train', 'reason': why})
for f in d['cv']['folds']:
    for k in list(f.keys()):
        if isinstance(f[k], list) and any(isinstance(x, str) and x in DROP for x in f[k]): f[k] = [x for x in f[k] if x not in DROP]
    if 'stratum_counts' in f and 'validation_cases' in f:
        sc = {}
        for c in f['validation_cases']: sc[stratum(c)] = sc.get(stratum(c), 0) + 1
        f['stratum_counts'] = sc
d['duplicate_geometry_groups'] = [['AAA/unruputer/LIU_WEN_QI', 'AAA/unruputer/ZHU_ZI_HAI'], ['AG/slow/HOU_SHEN_QIAN', 'AG/slow/KANG_XI_MING']]
d['duplicate_geometry_note'] = '每组第一例已剔除（见 excluded_cases）；保留 ZHU_ZI_HAI、KANG_XI_MING'
d['counts']['train'] = len(d['train_cases']); d['schema_version'] = 3
d['source_split'] = str(SRC); d['source_split_sha256'] = hashlib.sha256(SRC.read_bytes()).hexdigest(); d['created_at'] = time.strftime('%Y-%m-%dT%H:%M:%S%z')
d['derivation'] = '派生自 train138/test34 冻结 split，只剔除 2 例、登记重复组、重算 cv 分层计数；test34 逐字不变'
d.pop('content_sha256', None); d['content_sha256_recipe'] = 'sha256(json.dumps(payload_without_content_sha256, sort_keys=True, ensure_ascii=False))'
d['content_sha256'] = hashlib.sha256(json.dumps(d, sort_keys=True, ensure_ascii=False).encode()).hexdigest()
DST.write_text(json.dumps(d, indent=1, ensure_ascii=False), encoding='utf-8')
src = json.loads(SRC.read_text())
print(f'写出 {DST.name}: train {len(d["train_cases"])} test {len(d["test_cases"])} excluded {len(d["excluded_cases"])}; test34 逐字不变 {src["test_cases"] == d["test_cases"]}; 折内 val 数 {[len(f.get("validation_cases", [])) for f in d["cv"]["folds"]]}（原 {[len(f.get("validation_cases", [])) for f in src["cv"]["folds"]]}）')

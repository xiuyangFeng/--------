"""172 例壁面几何重复扫描（2026-09-15）：按点数/质心/包围盒指纹找候选，再算最近距离；对候选对比 H5 里的节点号/WSS/压力/体网格是否逐位相同，并把各自 STL 与壁面叠合。"""
import itertools, struct, numpy as np, h5py
from pathlib import Path
from scipy.spatial import cKDTree
ROOT = Path('/public/newhome/cy/Digital_twin/GNN'); H5 = ROOT / 'data_wss_v5/anatomy_pointcloud_v5_20260906/cases'
def nn(a, b): d, _ = cKDTree(b).query(a); return np.median(d), np.quantile(d, .9), d.max()
def stl_pts(p):
    b = open(p, 'rb').read()
    if b[:5].lower() == b'solid' and b'facet' in b[:1000]:
        v = np.array([[float(x) for x in l.split()[1:4]] for l in b.decode('latin-1').splitlines() if l.strip().startswith('vertex')])
    else:
        n = struct.unpack('<I', b[80:84])[0]; v = np.frombuffer(b[84:84 + n * 50], dtype=np.dtype([('n', '<3f4'), ('v', '<9f4'), ('a', '<u2')]))['v'].reshape(-1, 3).astype(float)
    return np.unique(np.round(v, 4), axis=0)
fp = {}
for d in sorted(H5.glob('*')):
    with h5py.File(d / 'case.h5', 'r') as h: x = h['wall_static/xyz_mm'][:]; fp[d.name] = (x, str(h.attrs['role']), np.r_[len(x), x.mean(0), x.min(0), x.max(0)])
names = list(fp)
for i, j in itertools.combinations(range(len(names)), 2):
    fi, fj = fp[names[i]][2], fp[names[j]][2]
    if abs(fi[0] - fj[0]) <= 0.02 * max(fi[0], fj[0]) and np.abs(fi[1:] - fj[1:]).max() < 2.0:
        a, b = names[i], names[j]; m, p90, mx = nn(fp[a][0], fp[b][0])
        print(f'{a} ({fp[a][1]}) ~ {b} ({fp[b][1]}): 点数 {len(fp[a][0])}/{len(fp[b][0])}, 壁面最近距离 中位 {m:.3f} p90 {p90:.3f} max {mx:.3f} mm')
        with h5py.File(H5 / a / 'case.h5', 'r') as A, h5py.File(H5 / b / 'case.h5', 'r') as B:
            same = lambda k: A[k].shape == B[k].shape and np.array_equal(A[k][:], B[k][:])
            print('   逐位相同: 节点号', same('wall_static/node_id_cas'), '| WSS', same('wall_temporal/wss_scalar_pa'), '| 压力', same('wall_temporal/pressure_pa'), '| 体网格', same('volume_static/xyz_mm'), '| 入口通量', same('interfaces/inlet/flux_outward_m3s'))
        for cid in (a, b):
            cdir = ROOT / 'data_new' / cid.replace('__', '/'); stls = sorted(cdir.glob('*.stl'))
            if stls: print(f'   {cid} 自己的 STL → 自己的 CFD 壁面 中位/p90/max: %.3f/%.3f/%.3f mm' % nn(stl_pts(stls[0]), fp[cid][0]))

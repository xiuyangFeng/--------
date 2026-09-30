"""按文献口径（逐例 R²、MAE、NMAE_max、NMAE_mean、相对 L2）重算 cv3 折外与周期量预测。
运行位置：仓库根目录。只读已保存的 predictions.npz，不做推理。
"""
import glob
import numpy as np

def pcm(t, p):
    e = p - t; ae = np.abs(e); var = np.sum((t - t.mean()) ** 2)
    return dict(r2=1 - np.sum(e ** 2) / var if var > 0 else np.nan, mae=ae.mean(),
                nmae_max=ae.mean() / np.abs(t).max(), nmae_mean=ae.mean() / np.abs(t).mean(),
                rel_l2=np.linalg.norm(e) / np.linalg.norm(t))

def run(root, arm, ck):
    cases, T, P = {}, [], []
    for f in range(3):
        for fn in sorted(glob.glob(f'{root}/{arm}_f{f}_s1234/eval/{ck}/predictions/test/*/*/*/predictions.npz')):
            u = fn.split('/predictions/test/')[1].rsplit('/', 1)[0]
            z = np.load(fn); t = z['true_pa'].astype(float); p = z['pred_pa'].astype(float)
            cases[u] = pcm(t, p); T.append(t); P.append(p)
    if not cases:
        return
    print(f'== {arm} {ck} OOF cases={len(cases)}')
    for k in next(iter(cases.values())):
        v = np.array([c[k] for c in cases.values()])
        print(f'  {k:9s} mean {np.nanmean(v):.4f} sd {np.nanstd(v, ddof=1):.4f} median {np.nanmedian(v):.4f}')
    t = np.concatenate(T); p = np.concatenate(P)
    print('  pooled', {k: round(float(v), 4) for k, v in pcm(t, p).items()})
    mse = np.mean([np.mean((pp - tt) ** 2) for tt, pp in zip(T, P)]); gm = t.mean()
    var = np.mean([np.mean((tt - gm) ** 2) for tt in T]); print('  R2_cb', round(1 - mse / var, 4))

run('training_wss_min/runs/wss_v51_wave2a_20260916', 'X5D_v51', 'ckpt_best')   # 峰值 WSS 折外
run('training_wss_min/runs/wss_cycle_20260920', 'A1', 'ckpt_best')             # TAWSS 折外
run('training_wss_min/runs/wss_cycle_20260920', 'O1', 'ckpt_best')             # OSI 折外

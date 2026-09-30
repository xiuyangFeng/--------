# 用 trend_eval 已存的逐例逐帧矩 (mean, mean_sq, mse)，估算不同"时间归一化"下各相位段占总损失的份额
import json, glob, numpy as np
base='/public/newhome/cy/Digital_twin/GNN/training_wss_min/experiments'
w=json.load(open(f'{base}/wss_time_ecc_20260918/offline/protocol_inlet_waveform_v51.json'))
q=np.array(w['q_norm'])[:80]
SEG={"plateau":list(range(0,5))+list(range(58,80)),"accel":list(range(10,17)),"peak":list(range(17,27)),
     "decel":list(range(27,43)),"trough":list(range(5,10))+list(range(43,58))}
seg_of=np.empty(80,dtype=object)
for s,idx in SEG.items(): seg_of[idx]=s
sd={f:np.array(json.load(open(f'{base}/wss_time_ecc_20260918/offline/wss_frame_stats_fold{f}.json'))['frame']['log_std'])[:80] for f in range(3)}
def load(arm):
    rows=[]
    for fn in sorted(glob.glob(f'{base}/wss_time_trend_20260927/per_case/{arm}_f*.jsonl')):
        for l in open(fn):
            r=json.loads(l); rows.append(r)
    return rows
print('frames per segment', {s:len(i) for s,i in SEG.items()})
print('q_norm mean per segment', {s:round(float(q[i].mean()),3) for s,i in SEG.items()})
for arm in ['Tnull','C-raw','TT-warm-noattn']:
    rows=load(arm)
    if not rows: print(arm,'missing'); continue
    ln=np.array([r['moments']['ln'] for r in rows])[:,:,:80]   # case,3,frame
    pa=np.array([r['moments']['pa'] for r in rows])[:,:,:80]
    folds=np.array([r['fold'] for r in rows])
    sdt=np.stack([sd[f] for f in folds])                         # case,frame
    var_ln=ln[:,1]-ln[:,0]**2; var_pa=pa[:,1]-pa[:,0]**2
    schemes={
      'S0 当前: ln MSE / 训练折帧 σ(t)²': (ln[:,2]/sdt**2),
      'S1 Pa MSE (不归一)':            pa[:,2],
      'S2 ln MSE 不归一':              ln[:,2],
      'S3 Pa 逐帧相对 L2 (÷均方)':      pa[:,2]/pa[:,1],
      'S4 逐例逐帧 1-R² (ln, ÷病例内方差)': ln[:,2]/var_ln,
      'S5 逐例逐帧 1-R² (Pa)':          pa[:,2]/var_pa,
    }
    print(f'\n== {arm}  n={len(rows)}')
    print('scheme'.ljust(38), ' '.join(s.rjust(8) for s in SEG), '  | 每帧份额 max/min')
    for name,v in schemes.items():
        per_frame=v.mean(0)            # 病例等权
        share=per_frame/per_frame.sum()
        segshare={s:share[i].sum() for s,i in SEG.items()}
        print(name.ljust(38), ' '.join(f'{segshare[s]*100:7.1f}%' for s in SEG), f'  | {share.max()/share.min():6.1f}x')
    # 帧均匀分布基准
    print('（帧数比例）'.ljust(38), ' '.join(f'{len(SEG[s])/80*100:7.1f}%' for s in SEG))
    # 每段 per-frame 平均的 z 空间误差
    z=(ln[:,2]/sdt**2).mean(0)
    print('S0 每帧平均 z-MSE 按段:', {s:round(float(z[i].mean()),3) for s,i in SEG.items()})
    r2ln=1-(ln[:,2]/var_ln).mean(0)
    print('逐例 ln R² 按段(病例均):', {s:round(float(r2ln[i].mean()),3) for s,i in SEG.items()})

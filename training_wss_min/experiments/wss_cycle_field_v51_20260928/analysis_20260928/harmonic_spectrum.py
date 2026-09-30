# 5 例导出：逐点 ln τ(t) 的 80 帧 DFT，按谐波比较真值/预测能量（面积加权），看时间方向是否过平滑（只读）
import numpy as np, glob, os
root='/public/newhome/cy/Digital_twin/GNN/training_wss_min/experiments/wss_time_trend_20260927/export'
FLOOR,EPS=0.05,1e-6
ln=lambda x: np.log(np.clip(x,FLOOR,None)+EPS)
bands={'h1':[1],'h2':[2],'h3-4':[3,4],'h5-8':list(range(5,9)),'h9-20':list(range(9,21))}
arms=['Tnull','C-raw','TT-warm']
print('病例'.ljust(34),'arm'.ljust(8),' '.join(f'{b:>7}' for b in bands),'   (能量比 pred/true)  | 谐波误差/真值能量')
for d in sorted(glob.glob(root+'/*')):
    t=np.load(d+'/truth.npz'); a=t['area']/t['area'].sum()
    Ft=np.fft.rfft(ln(t['tau']),axis=0)             # 41,N
    Et={b:(a*(np.abs(Ft[h])**2).sum(0)).sum() for b,h in bands.items()}
    for arm in arms:
        f=d+f'/{arm}.npz'
        if not os.path.exists(f): continue
        Fp=np.fft.rfft(ln(np.load(f)['pred_pa']),axis=0)
        ratio=[(a*(np.abs(Fp[h])**2).sum(0)).sum()/Et[b] for b,h in bands.items()]
        err=[(a*(np.abs(Fp[h]-Ft[h])**2).sum(0)).sum()/Et[b] for b,h in bands.items()]
        print(os.path.basename(d)[:34].ljust(34),arm.ljust(8),' '.join(f'{r:7.2f}' for r in ratio),'  |',' '.join(f'{e:5.2f}' for e in err))
    # 真值能量分布
    tot=sum(Et.values()); print(' '*34,'真值能量占比'.ljust(8),' '.join(f'{Et[b]/tot:7.2f}' for b in bands))

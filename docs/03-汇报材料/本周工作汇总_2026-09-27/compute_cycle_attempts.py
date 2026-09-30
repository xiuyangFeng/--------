from pathlib import Path
exec(Path(__file__).with_name('compute_cycle_metrics.py').read_text().split('cv51={}')[0])
for exp,arm,channel in [('wss_osi_tail_20260924',a,'osi') for a in ('M1r','K1','K2','K5')]+[('wss_osi_struct_20260925','V1','osi_derived'),('wss_osi_struct_20260925','V1','osi'),('wss_osi_struct_20260925','V2','osi')]:
    groups=[[RUNS/exp/('%s_f%d_s1234'%(arm,k))/P/channel for k in range(3)]]
    calc('v5.1 '+arm+' '+channel+' cv3 s1234','osi',groups,'v5.1 cv3 OOF train136; 3 folds merged; single seed1234')
with open(OUT/'cycle_attempts.json','w') as f:json.dump(dict(rows=rows),f,indent=2)
cols=['label','target','protocol','n_cases','r2_cb','mae_cb','approx_disp_casemean','cosine_spatial_casemean','spearman_casemean']
with open(OUT/'cycle_attempts.tsv','w') as f:
    w=csv.DictWriter(f,fieldnames=cols,delimiter='\t',extrasaction='ignore');w.writeheader();w.writerows(rows)
print('OUTPUT',str(OUT/'cycle_attempts.json'),flush=True)

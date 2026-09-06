#!/usr/bin/env python3
"""Extract the every-50-step gradient diagnostics from training_progress.jsonl into compact CSVs.

Only records whose gradient diagnostics were actually computed
(global_step % gradient_check_interval_steps == 0) are kept.  Read-only on the run directories.
"""
import csv, json, os, sys
BASE = '/public/newhome/cy/Digital_twin/GNN/outputs/wss_pinn/volume_uvwp_bc_rcr_v4'
OUT = os.path.dirname(os.path.abspath(__file__))
ARMS = [
    ('steady_peak', 'V4-SP-PN-BC-s1234'),
    ('steady_peak', 'V4-SP-PN-BC-PDE-F-s1234'),
    ('steady_peak', 'V4-SP-PN-BC-PDE-EMA-s1234'),
    ('transient_autograd', 'V4-TR-PN-BC-s1234'),
    ('transient_autograd', 'V4-TR-PN-BC-PDE-F-s1234'),
    ('transient_autograd', 'V4-TR-PN-BC-PDE-EMA-s1234'),
]
FIELDS = ['epoch', 'global_step', 'cos_grad_data_no_slip', 'cos_grad_data_momentum',
          'cos_grad_data_continuity', 'cos_grad_data_inlet_bc', 'cos_grad_data_rcr_bc',
          'grad_norm_data', 'grad_norm_physics', 'gradient_norm', 'gradient_clip_flag',
          'no_slip_raw', 'inlet_bc_raw', 'data_total', 'pde_total', 'loss_ratio_raw',
          'lambda_phy', 'lambda_target', 'lambda_clamped_high', 'ema_data', 'ema_physics']
for sub, run in ARMS:
    src = f'{BASE}/{sub}/{run}/training_progress.jsonl'
    dst = f'{OUT}/gradcos_{run}.csv'
    n = 0
    with open(src) as fh, open(dst, 'w', newline='') as out:
        w = csv.DictWriter(out, fieldnames=FIELDS)
        w.writeheader()
        for line in fh:
            if '"global_step": ' not in line:
                continue
            # cheap prefilter: global_step multiple of 50 -> ends with "0," and second-last digit 0 or 5
            i = line.find('"global_step": ') + 15
            j = line.find(',', i)
            gs = int(line[i:j])
            if gs % 50:
                continue
            d = json.loads(line)
            w.writerow({k: d.get(k) for k in FIELDS})
            n += 1
    print(run, n, flush=True)
print('DONE')

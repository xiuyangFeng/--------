"""M1 三头门控读数（矩阵 §14）：M1_f{k} 峰值通道对同折 X5D_v51 峰值底座；TAWSS / OSI 头对单头 A1_f{k} / O2_f{k}（归一化 R²_cb）。
    python report_m1.py [best|last]
"""
from __future__ import annotations
import json, sys
from pathlib import Path
import numpy as np

ROOT = Path('/public/newhome/cy/Digital_twin/GNN')
RUNS = ROOT / 'training_wss_min/runs'
OUT = Path(__file__).resolve().parent
CKPT = sys.argv[1] if len(sys.argv) > 1 else 'best'


def load(path):
    return json.loads(path.read_text())['test'] if path.is_file() else None


def block(m):
    ca = m.get('cycle_agreement') or {}
    return {'norm': m['normalized']['field_casebalanced']['r2'], 'pa': m['field_casebalanced']['r2'], 'casemean': m['aggregate']['r2_casemean'],
            'p10': m['aggregate']['r2_casep10'], 'neg': int(m['aggregate']['r2_negative_cases']), 'sp': m['hotspot'].get('spearman_all_casemean'),
            'masks': {k: v['iou_casemean'] for k, v in (m.get('threshold_masks') or {}).items()},
            'ccc': (ca.get('ccc') or {}).get('med'), 'seg': (ca.get('seg_rel_err_med') or {}).get('med')}


rows = []
for k in range(3):
    m = load(RUNS / f'wss_cycle_m1_20260921/M1_f{k}_s1234/eval/ckpt_{CKPT}/metrics.json')
    if m is None:
        rows.append({'fold': k, 'status': 'missing'}); continue
    base = load(RUNS / f'wss_v51_wave2a_20260916/X5D_v51_f{k}_s1234/eval/ckpt_best/metrics.json')
    a1 = load(RUNS / f'wss_cycle_20260920/A1_f{k}_s1234/eval/ckpt_best/metrics.json')
    o2 = load(RUNS / f'wss_cycle_20260920/O2_f{k}_s1234/eval/ckpt_best/metrics.json')
    r = {'fold': k, 'peak': block(m), 'tawss': block(m['heads']['tawss']), 'osi': block(m['heads']['osi']),
         'base_peak': block(base), 'A1': block(a1), 'O2': block(o2)}
    r['d_peak_norm'] = r['peak']['norm'] - r['base_peak']['norm']; r['d_peak_pa'] = r['peak']['pa'] - r['base_peak']['pa']
    r['d_tawss_norm'] = r['tawss']['norm'] - r['A1']['norm']; r['d_osi_norm'] = r['osi']['norm'] - r['O2']['norm']
    r['d_tawss_pa'] = r['tawss']['pa'] - r['A1']['pa']; r['d_osi_pa'] = r['osi']['pa'] - r['O2']['pa']
    rows.append(r)
done = [r for r in rows if 'peak' in r]
lines = [f'=== M1 三头（{CKPT}）===']
for r in done:
    lines.append(f"fold{r['fold']}: 峰值 norm {r['peak']['norm']:.4f} (底座 {r['base_peak']['norm']:.4f}, Δ {r['d_peak_norm']:+.4f}) Pa {r['peak']['pa']:.4f} (Δ {r['d_peak_pa']:+.4f}) | "
                 f"TAWSS norm {r['tawss']['norm']:.4f} (A1 {r['A1']['norm']:.4f}, Δ {r['d_tawss_norm']:+.4f}) Pa {r['tawss']['pa']:.4f} (Δ {r['d_tawss_pa']:+.4f}) CCC {r['tawss']['ccc']} seg {r['tawss']['seg']} | "
                 f"OSI norm {r['osi']['norm']:.4f} (O2 {r['O2']['norm']:.4f}, Δ {r['d_osi_norm']:+.4f}) R² {r['osi']['pa']:.4f} (Δ {r['d_osi_pa']:+.4f}) IoU>0.1 {r['osi']['masks'].get('above_0.1')}")
gate = {}
if len(done) == 3:
    dp = [r['d_peak_norm'] for r in done]; dt = [r['d_tawss_norm'] for r in done]; do = [r['d_osi_norm'] for r in done]
    gate = {'peak_drop_le_0.02_all_folds': all(d >= -0.02 for d in dp), 'peak_delta_mean': float(np.mean(dp)),
            'tawss_head_ge_A1_minus_0.01': all(d >= -0.01 for d in dt), 'tawss_delta_mean': float(np.mean(dt)),
            'osi_head_ge_O2_minus_0.01': all(d >= -0.01 for d in do), 'osi_delta_mean': float(np.mean(do))}
    lines.append(f"三折均值：峰值 Δnorm {np.mean(dp):+.4f}（{sum(d >= -0.02 for d in dp)}/3 ≥ −0.02）→ {'PASS' if gate['peak_drop_le_0.02_all_folds'] else 'FAIL'}；"
                 f"TAWSS 头 Δ {np.mean(dt):+.4f}（{sum(d >= -0.01 for d in dt)}/3）→ {'PASS' if gate['tawss_head_ge_A1_minus_0.01'] else 'FAIL'}；"
                 f"OSI 头 Δ {np.mean(do):+.4f}（{sum(d >= -0.01 for d in do)}/3）→ {'PASS' if gate['osi_head_ge_O2_minus_0.01'] else 'FAIL'}")
print('\n'.join(lines))
(OUT / f'm1_report_{CKPT}.json').write_text(json.dumps({'checkpoint': CKPT, 'folds': rows, 'gate': gate}, indent=1, ensure_ascii=False, default=float))
(OUT / f'm1_report_{CKPT}.txt').write_text('\n'.join(lines) + '\n')

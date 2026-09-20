"""诊断二：训练好的 X5D_long 模型，推理时把沿程通道改成"无效"，看预测变多少。

变体（只改输入，不重训）：
  full    原样
  no_long 32 个沿程通道全部置为 invalid（value 标准化 0 + mask 0）
  ref_only 只屏蔽 pc 的 16 通道
  pc_only  只屏蔽 ref 的 16 通道
如果 no_long 与 full 几乎相同，说明模型本来就没怎么用这些通道。
"""
import copy, json, sys
import numpy as np
import torch
from pathlib import Path
sys.path.insert(0, '/public/newhome/cy/Digital_twin/GNN')
from training_wss_min import config as C
from training_wss_min import dataset as D
from training_wss_min import evaluate as E
from training_wss_min import longitudinal_geometry as L
M = E.M

ROOT = Path('/public/newhome/cy/Digital_twin/GNN')
RUNS = ROOT / 'training_wss_min/runs/wss_x5d_longitudinal_20260917'
OUT = Path(__file__).resolve().parent
SEEDS = (1234, 7, 2025)
REF = tuple(k for k in L.VALUE_KEYS if k.startswith('geom_ref_'))
PC = tuple(k for k in L.VALUE_KEYS if k.startswith('geom_pc_'))
VARIANTS = {'full': (), 'no_long': REF + PC, 'ref_only': PC, 'pc_only': REF}


def blank(case, keys):
    if not keys:
        return case
    out = dict(case)
    n = len(case['pos'])
    for k in keys:
        out[k] = np.zeros(n, dtype=np.float32)
        out[k + '_valid'] = np.zeros(n, dtype=bool)
    return out


def main():
    dev = 'cuda' if torch.cuda.is_available() else 'cpu'
    res = {}
    for seed in SEEDS:
        run = RUNS / f'X5D_long_s{seed}'
        cfg, feat_stats, model, _ = E.load_model_from_run(run, dev, 'best')
        model.eval()
        wss_stats = E.load_wss_stats_for_run(run)
        cases = D.load_partition(cfg.data.split_path, 'test', wss_stats, target=cfg.data.target,
                                 target_normalization=cfg.data.target_normalization,
                                 data_root=cfg.data.data_root,
                                 required_frame_version=cfg.data.required_frame_version,
                                 timesteps=cfg.data.timesteps,
                                 extra_point_features=C.v6_point_features(cfg),
                                 point_features_root=cfg.data.point_features_root)
        truth = [c['y_raw'] for c in cases]
        coh = [c['unit_id'].split('/')[0] for c in cases]
        preds = {}
        with torch.no_grad():
            for vname, keys in VARIANTS.items():
                out = []
                for case in cases:
                    pn = E.predict_case_norm(model, blank(case, keys), cfg.data.input_features,
                                             feat_stats, dev, cfg)
                    out.append(np.clip(D.denormalize_wss(np.asarray(pn, dtype=np.float64), wss_stats), 0, None))
                preds[vname] = out
                print(f'  s{seed} {vname} done', flush=True)
        row = {}
        for vname, p in preds.items():
            r2 = M.casebalanced_field_metrics(truth, p)['r2']
            d = float(np.mean([np.mean(np.abs(np.log(np.maximum(a, 1e-6)) - np.log(np.maximum(b, 1e-6))))
                               for a, b in zip(p, preds['full'])]))
            per = {c: M.casebalanced_field_metrics([truth[i] for i in range(len(cases)) if coh[i] == c],
                                                   [p[i] for i in range(len(cases)) if coh[i] == c])['r2']
                   for c in ('AG', 'AAA', 'ILO')}
            row[vname] = {'pa_r2_cb': r2, 'mean_abs_dlog_vs_full': d, 'per_cohort': per}
            print(f'  s{seed} {vname:9s} Pa R2_cb {r2:.4f}  |Δln pred| vs full {d:.4f}  '
                  + 'AG/AAA/ILO ' + '/'.join(f'{v:.3f}' for v in per.values()), flush=True)
        res[f's{seed}'] = row
        del model
        torch.cuda.empty_cache()
    print('\n=== 三 seed 均值')
    for vname in VARIANTS:
        r = np.mean([res[f's{s}'][vname]['pa_r2_cb'] for s in SEEDS])
        d = np.mean([res[f's{s}'][vname]['mean_abs_dlog_vs_full'] for s in SEEDS])
        base = np.mean([res[f's{s}']['full']['pa_r2_cb'] for s in SEEDS])
        print(f'  {vname:9s} Pa R2_cb {r:.4f} (Δ vs full {r-base:+.4f})  |Δln pred| {d:.4f}')
    (OUT / 'diagnose_ablation.json').write_text(json.dumps(res, indent=1, ensure_ascii=False, default=float))
    print('\nwrote', OUT / 'diagnose_ablation.json')


if __name__ == '__main__':
    main()

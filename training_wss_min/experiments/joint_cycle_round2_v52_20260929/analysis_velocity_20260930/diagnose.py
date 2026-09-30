"""CPU-only diagnostics of stored predictions, metrics and audited label caches.

No model import, checkpoint loading, training, inference or original HDF5 reads.
All reductions give equal weight to data units, then volume/phase within unit.
"""
import csv
import json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[4]
OUT = Path(__file__).resolve().parent
RUNS = ROOT / 'training_wss_min/runs'
ARMS = ['Iu', 'UT0', 'UT1', 'U0', 'U1', 'J1', 'J2']
SEGMENTS = {'cycle': list(range(80)), 'peak': list(range(17, 27)),
            'accel': list(range(10, 17)), 'decel': list(range(27, 43)),
            'trough': list(range(5, 10)) + list(range(43, 58)),
            'early_trough': list(range(5, 10)), 'late_trough': list(range(43, 58)),
            'plateau': list(range(0, 5)) + list(range(58, 80))}


def run(arm):
    batch = 'joint_cycle_v52_20260929' if arm in ['Iu', 'J1'] else 'joint_cycle_round2_v52_20260929'
    return RUNS / batch / (arm + '_f0_s1234')


def read_json(p):
    return json.loads(p.read_text())


def write_csv(name, rows):
    with (OUT / name).open('w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


def reduce_rows(rows, group_keys, drop_keys=('unit_id',)):
    groups = {}
    for r in rows:
        groups.setdefault(tuple(r[k] for k in group_keys), []).append(r)
    result = []
    for group, rr in groups.items():
        out = dict(zip(group_keys, group))
        out['n_units'] = len(rr)
        if 'bin' in group_keys:
            out['n_units_total'] = 55
        for k in rr[0]:
            if k in group_keys or k in drop_keys:
                continue
            vals = [r[k] for r in rr if r[k] is not None]
            # Empty conditional bins contribute zero coverage/error energy;
            # conditional means themselves are undefined and remain excluded.
            if 'bin' in group_keys and k in ('volume_phase_coverage','vector_error_energy_contribution'):
                out[k] = float(np.sum(vals)/55)
            else:
                out[k] = float(np.mean(vals)) if vals else None
        result.append(out)
    return result


def main():
    records, rows, inventory = {}, [], {}
    for arm in ARMS:
        rr = [json.loads(s) for s in (run(arm) / 'per_case.jsonl').read_text().splitlines()]
        records[arm] = {r['unit_id']: r for r in rr}
        assert len(rr) == len(records[arm]) == 55
        assert set(records[arm]) == set(records['Iu'])
        h = [json.loads(s) for s in (run(arm) / 'history.jsonl').read_text().splitlines()]
        inventory[arm] = {'prediction_npz_count': len(list((run(arm) / 'predictions').glob('*.npz'))),
                          'epochs': len(h), 'final_loss': h[-1]['losses']['velocity'],
                          'first10_loss_mean': np.mean([r['losses']['velocity'] for r in h[:10]]),
                          'last10_loss_mean': np.mean([r['losses']['velocity'] for r in h[-10:]])}
        for r in rr:
            v = r['tasks']['velocity']
            for phase in ['peak', 'accel', 'decel', 'trough', 'plateau', 'cycle']:
                a, s = v[phase], v['speed'][phase]
                vmse = 3 * a['mse']
                ang = vmse - s['mse']
                assert ang >= -1e-12
                rows.append({'arm': arm, 'unit_id': r['unit_id'], 'phase': phase,
                             'vector_r2': a['r2'], 'component_mae_m_s': a['mae'],
                             'vector_mse_m2_s2': vmse, 'speed_mse_m2_s2': s['mse'],
                             'angular_mse_m2_s2': ang, 'angular_fraction': ang / vmse,
                             'speed_r2': s['r2'], 'speed_mae_m_s': s['mae'],
                             'direction_cosine': a['direction_cosine'],
                             'direction_coverage': a['direction_weight_coverage'],
                             'vector_truth_energy_m2_s2': 3 * a['truth_energy'],
                             'vector_truth_variance_m2_s2': vmse / (1 - a['r2'])})
    write_csv('phase_per_unit.csv', rows)
    agg = reduce_rows(rows, ('arm', 'phase'))
    for r in agg:
        r['angular_fraction_ratio_of_means'] = r['angular_mse_m2_s2'] / r['vector_mse_m2_s2']
    write_csv('phase_summary.csv', agg)
    paired = []
    for arm, base in [('U0','Iu'), ('U1','U0'), ('UT1','UT0'), ('J2','J1')]:
        for phase in ['peak','accel','decel','trough','plateau','cycle']:
            a = np.array([records[arm][u]['tasks']['velocity'][phase]['r2'] for u in records[arm]])
            b = np.array([records[base][u]['tasks']['velocity'][phase]['r2'] for u in records[arm]])
            paired.append({'arm':arm,'base':base,'phase':phase,'mean_delta_r2':float(np.mean(a-b)),
                           'improved_units':int(np.sum(a>b)), 'total_units':len(a),
                           'r2_p10':float(np.quantile(a,.1)), 'base_r2_p10':float(np.quantile(b,.1)),
                           'negative_r2_units':int(np.sum(a<0))})
    write_csv('paired_summary.csv', paired)

    # Saved predictions are physical atlas velocity; match authoritative row IDs,
    # never assume a prediction index is an original CFD row.
    point_rows, frame_rows = [], []
    max_identity_error, max_metrics_mse_error = 0., 0.
    for unit in records['U0']:
        cfg = read_json(run('U0') / 'config.json')
        cache = Path(cfg['data']['cache_root']) / 'cases' / unit
        cached_rows = np.load(cache / 'eval_volume_rows.npy')
        order = np.argsort(cached_rows)
        y0 = np.load(cache / 'eval_velocity.npy', mmap_mode='r')
        w0 = np.load(cache / 'eval_volume_weight.npy', mmap_mode='r')
        x0 = np.load(cache / 'eval_volume_x.npy', mmap_mode='r')
        for arm in ['Iu','U0']:
            index = records[arm][unit]['index']
            with np.load(run(arm) / 'predictions' / f'{index:03d}.npz') as z:
                assert str(z['unit_id']) == unit
                query_rows = z['velocity_rows']
                where = np.searchsorted(cached_rows[order], query_rows)
                assert np.all(where < len(cached_rows))
                take = order[where]
                assert np.array_equal(cached_rows[take], query_rows)
                p = z['velocity_prediction'].astype(np.float64)
            y = np.asarray(y0[take], np.float64)
            w = np.asarray(w0[take], np.float64)
            w /= w.sum()
            x = np.asarray(x0[take], np.float64)
            sy, sp = np.linalg.norm(y, axis=2), np.linalg.norm(p, axis=2)
            dot = np.einsum('nfc,nfc->nf', y, p)
            cos = np.clip(dot / np.maximum(sy * sp, 1e-20), -1., 1.)
            vecerr = np.sum((y-p)**2, axis=2)
            sperr = (sy-sp)**2
            angerr = vecerr-sperr
            identity = 2 * (sy*sp-dot)
            max_identity_error = max(max_identity_error, float(np.max(np.abs(angerr-identity))))
            cycle_mse = float(np.einsum('n,nf->', w, vecerr) / 80 / 3)
            max_metrics_mse_error = max(max_metrics_mse_error, abs(cycle_mse-records[arm][unit]['tasks']['velocity']['cycle']['mse']))
            masks = {
                'all': np.ones(sy.shape, bool),
                'truth_speed_lt_0.01': sy < .01,
                'truth_speed_0.01_0.03': (sy >= .01) & (sy < .03),
                'truth_speed_0.03_0.10': (sy >= .03) & (sy < .10),
                'truth_speed_0.10_0.30': (sy >= .10) & (sy < .30),
                'truth_speed_ge_0.30': sy >= .30,
                'wall_distance_lt_1mm': np.broadcast_to((x[:,19] < 1)[:,None], sy.shape),
                'wall_distance_1_3mm': np.broadcast_to(((x[:,19] >= 1)&(x[:,19] < 3))[:,None], sy.shape),
                'wall_distance_ge_3mm': np.broadcast_to((x[:,19] >= 3)[:,None], sy.shape),
            }
            for phase, ids in SEGMENTS.items():
                for name, mask in masks.items():
                    ww = w[:,None] * mask[:,ids] / len(ids)
                    den = float(ww.sum())
                    if den <= 1e-20:
                        continue
                    mean = lambda a: float((ww * a[:,ids]).sum()/den)
                    direction_mask = sy[:,ids] >= .01
                    dw = ww * direction_mask
                    dd = float(dw.sum())
                    point_rows.append({'arm':arm,'unit_id':unit,'phase':phase,'bin':name,
                        'volume_phase_coverage':den,
                        'vector_mse_m2_s2':mean(vecerr), 'speed_mse_m2_s2':mean(sperr),
                        'angular_mse_m2_s2':mean(angerr),
                        'vector_error_energy_contribution':float((ww*vecerr[:,ids]).sum()),
                        'truth_speed_mean_m_s':mean(sy), 'pred_speed_mean_m_s':mean(sp),
                        'speed_bias_m_s':mean(sp-sy),
                        'direction_cosine_gt_0.01':float((dw*cos[:,ids]).sum()/dd) if dd else None,
                        'direction_coverage_in_bin':dd/den,
                        'pred_lt_0.001_given_truth_ge_0.01':float((dw*(sp[:,ids]<.001)).sum()/dd) if dd else None,
                        'opposing_direction_given_truth_ge_0.01':float((dw*(cos[:,ids]<0)).sum()/dd) if dd else None,
                        'speed_under_half_truth_rate':mean(sp < .5*sy),
                        'speed_over_1.5_truth_rate':mean(sp > 1.5*sy)})
            for frame in range(80):
                wf = w * (sy[:,frame] >= .01)
                cov = float(wf.sum())
                vmse = float(w @ vecerr[:,frame])
                truth_energy = float(w @ (sy[:,frame]**2))
                meanvec = w @ y[:,frame]
                variance = truth_energy-float(meanvec@meanvec)
                frame_rows.append({'arm':arm,'unit_id':unit,'frame':frame,
                                   'vector_r2':1-vmse/variance,
                                   'component_mae_m_s':float(np.einsum('n,nc->', w, np.abs(y[:,frame]-p[:,frame]))/3),
                                   'vector_mse_m2_s2':vmse,'speed_mse_m2_s2':float(w@sperr[:,frame]),
                                   'angular_mse_m2_s2':float(w@angerr[:,frame]),
                                   'direction_cosine_gt_0.01':float(wf@cos[:,frame]/cov) if cov else None,
                                   'direction_coverage':cov,
                                   'truth_speed_mean_m_s':float(w@sy[:,frame]),
                                   'pred_speed_mean_m_s':float(w@sp[:,frame])})
        print(f'completed stored predictions {unit}', flush=True)
    write_csv('point_bins_per_unit.csv', point_rows)
    write_csv('point_bins_summary.csv', reduce_rows(point_rows, ('arm','phase','bin')))
    write_csv('frames_per_unit.csv', frame_rows)
    write_csv('frames_summary.csv', reduce_rows(frame_rows, ('arm','frame')))
    proof = {'no_training_or_inference': True, 'source': 'saved physical predictions + audited raw eval cache',
             'arms_point_diagnostics': ['Iu','U0'], 'n_units':55,
             'coordinate_frame': 'v5_atlas_frame_v1; y transformed by rotation.T, prediction physical m/s',
             'weight': 'eval_volume_weight, physical cell volumes, normalized within unit',
             'row_mapping': 'prediction velocity_rows exactly matched eval_volume_rows',
             'max_point_decomposition_abs_error_m2_s2':max_identity_error,
             'max_cycle_mse_error_vs_stored_metric':max_metrics_mse_error,
             'prediction_rounding': 'saved p is float32 while original metric used decoded float64',
             'inventory':inventory,
             'missing': ['cached local centerline tangent for signed axial/reversal diagnosis',
                         'training per-phase losses and gradient norms',
                         'pointwise physical gradient/divergence; no mesh neighborhood quadrature in cache']}
    (OUT/'provenance.json').write_text(json.dumps(proof,indent=2)+'\n')
    assert max_identity_error < 1e-12
    assert max_metrics_mse_error < 1e-8
    print(json.dumps(proof,indent=2))


if __name__ == '__main__':
    main()

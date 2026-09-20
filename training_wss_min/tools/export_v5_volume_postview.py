"""Reproduce R5P/R5V best fields and prepare same-point ParaView/report artifacts."""
from __future__ import annotations

import argparse
import csv
import gc
import hashlib
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / 'outputs/field/postview/wss_v5_r5_volume_best_worst_20260909'
REPORT = ROOT / 'docs/03-汇报材料/启发式实验汇报_2026-09-13/WSS_V5_体场_最好最差与横向R2_20260909'
RUNS = ROOT / 'training_wss_min/runs/v5_rerun_20260906/outputs'
ARMS = {'pressure': 'r5p_pressure_mixed_qad_s1234', 'speed': 'r5v_velocity_qad_s1234'}


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def write_json(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False) + '\n')


def data_stage(device):
    import torch
    from training_wss_min import dataset as D, evaluate as E, metrics as M
    torch.set_num_threads(2)
    source_files = list((ROOT / 'training_wss_min').glob('*.py'))
    before = {str(p): sha(p) for p in source_files}
    result = {'status': 'running', 'peak_step': 1162, 'selection_metric': 'official per_case.overall.r2',
              'coordinate_frame': 'V5 atlas aligned millimetres', 'arms': {}, 'source_sha256': before}
    for field, run_name in ARMS.items():
        run = RUNS / run_name
        official = json.loads((run / 'eval/ckpt_best/metrics.json').read_text())['test']
        ordered = sorted(official['per_case'], key=lambda uid: (official['per_case'][uid]['overall']['r2'], uid))
        selected = {ordered[-1]: 'best', ordered[0]: 'worst'}
        cfg, features, model, ckpt = E.load_model_from_run(run, device, 'best')
        model.eval()
        stats = E.load_wss_stats_for_run(run)
        split_ids = {f'{c}/{n}' for c, n in D.load_split_cases(cfg.data.split_path, 'test')}
        assert split_ids == set(ordered) and len(ordered) == 34
        assert cfg.eval.fixed_support and cfg.eval.support_seed == 1234
        arm = {'run': str(run), 'checkpoint_epoch': int(ckpt['epoch']), 'selected': selected,
               'input_sha256': {str(run / n): sha(run / n) for n in
                                ('config.json', 'ckpt_best.pt', 'feature_stats.json', 'wss_global_stats.json', 'eval/ckpt_best/metrics.json')},
               'split_sha256': sha(cfg.data.split_path), 'cases': []}
        if field == 'speed':
            cache_root = ROOT / 'training_wss_min/experiments/v5_velocity_to_wss_20260909'
            cache_manifest = json.loads((cache_root / 'velocity_reproduction.json').read_text())
            assert cache_manifest['passed'] and cache_manifest['status'] == 'complete'
            assert cache_manifest['source_sha256'][str((run / 'ckpt_best.pt').relative_to(ROOT))] == sha(run / 'ckpt_best.pt')
        for i, uid in enumerate(ordered):
            cohort, name = uid.rsplit('/', 1)
            case = D.load_case(cohort, name, stats, target=cfg.data.target,
                               target_normalization=cfg.data.target_normalization,
                               data_root=cfg.data.data_root, required_frame_version=cfg.data.required_frame_version)
            case['unit_id'] = uid
            rows = D.query_rows(case)
            truth = np.asarray(case['y_raw'][rows], dtype=np.float64)
            cache_info = None
            if field == 'pressure':
                pred_norm = np.asarray(E.predict_case_norm(model, case, cfg.data.input_features,
                                                          features, device, cfg=cfg), dtype=np.float64)
                pred = D.denormalize_wss(pred_norm, stats)
                true_scalar, pred_scalar = truth, pred
            else:
                cache_info = cache_manifest['cases'][uid]
                assert sha(cache_info['path']) == cache_info['sha256']
                assert sha(case['bundle_path']) == cache_info['bundle_sha256']
                assert sha(Path(cfg.data.data_root) / uid / 'volume.npz') == cache_info['volume_sha256']
                with np.load(cache_info['path']) as z:
                    pred = z['velocity_pred_aligned_m_s'].copy()
                    assert np.allclose(z['velocity_cfd_raw_m_s'] @ z['transform_rotation'].T, truth, atol=1e-10)
                true_scalar, pred_scalar = np.linalg.norm(truth, axis=1), np.linalg.norm(pred, axis=1)
            assert true_scalar.shape == pred_scalar.shape == (len(rows),)
            assert np.isfinite(true_scalar).all() and np.isfinite(pred_scalar).all()
            measured = {**M.basic_metrics(true_scalar, pred_scalar), **M.linear_fit_metrics(true_scalar, pred_scalar)}
            checks = []
            for key, value in official['per_case'][uid]['overall'].items():
                tolerance = 0 if key == 'n' else 5e-6 + 5e-6 * abs(value)
                delta = abs(measured[key] - value)
                assert delta <= tolerance, (field, uid, key, measured[key], value)
                checks.append({'metric': key, 'difference': delta, 'tolerance': tolerance})
            assert abs(measured['linear_fit_slope'] - official['per_case'][uid]['calibration']['calibration_slope']) < 2e-5
            record = {'case_id': uid, **measured, 'selection': selected.get(uid, ''), 'verification': checks}
            if cache_info:
                record['prediction_cache'] = {'path': cache_info['path'], 'sha256': cache_info['sha256']}
            if uid in selected:
                folder = OUT / field / (selected[uid] + '__' + uid.replace('/', '__'))
                folder.mkdir(parents=True, exist_ok=True)
                n_wall = int(case['n_wall'])
                wall_xyz = np.asarray(case['pos'][:n_wall], dtype=np.float64) * case['coord_scale_scalar']
                xyz = np.asarray(case['pos'][rows], dtype=np.float64) * case['coord_scale_scalar']
                source_index = rows - n_wall if field == 'speed' else rows
                np.savez_compressed(folder / 'same_point_fields.npz', xyz_mm=xyz, wall_xyz_mm=wall_xyz,
                                    truth=true_scalar, pred=pred_scalar, source_index=source_index,
                                    point_kind=case['point_kind'][rows], truth_components=truth, pred_components=pred,
                                    n_wall=np.int64(n_wall), p_ref_pa=np.float64(case['p_ref_pa']))
                record.update(folder=str(folder), bundle=str(case['bundle_path']),
                              bundle_sha256=sha(case['bundle_path']),
                              volume_sha256=sha(Path(cfg.data.data_root) / uid / 'volume.npz'),
                              archive_sha256=sha(folder / 'same_point_fields.npz'),
                              p_ref_pa=case['p_ref_pa'], truth_std=float(true_scalar.std()))
                if field == 'pressure':
                    record['groups'] = {label: {**M.basic_metrics(true_scalar[mask], pred_scalar[mask]),
                                               **M.linear_fit_metrics(true_scalar[mask], pred_scalar[mask])}
                                        for label, mask in [('wall', case['point_kind'][rows] == 0),
                                                            ('interior', case['point_kind'][rows] == 1)]}
            arm['cases'].append(record)
            result['arms'][field] = arm
            write_json(OUT / 'manifest.json', result)
            print(f'{field} {i + 1}/34 {uid}: R2={measured["r2"]:.6f}, fit={measured["r2_linear_fit"]:.6f} verified', flush=True)
            del case, truth, pred, true_scalar, pred_scalar
            gc.collect()
        del model
        torch.cuda.empty_cache()
    assert before == {str(p): sha(p) for p in source_files}
    for arm in result['arms'].values():
        assert all(sha(p) == h for p, h in arm['input_sha256'].items())
    result.update(status='complete', numerical_checks=sum(len(c['verification']) for a in result['arms'].values() for c in a['cases']))
    write_json(OUT / 'manifest.json', result)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--device', default='cuda:0')
    args = parser.parse_args()
    data_stage(args.device)

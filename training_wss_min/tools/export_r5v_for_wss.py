"""Reproduce the existing R5V best prediction, without training or changing its run."""
from __future__ import annotations

import gc
import hashlib
import json
import math
import os
from pathlib import Path
import time

import numpy as np
import torch

from training_wss_min import dataset as D, evaluate as E

ROOT = Path(__file__).resolve().parents[2]
EXP = ROOT / 'training_wss_min/experiments/v5_velocity_to_wss_20260909'
RUN = ROOT / 'training_wss_min/runs/v5_rerun_20260906/outputs/r5v_velocity_qad_s1234'


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda: f.read(1024 * 1024), b''):
            h.update(b)
    return h.hexdigest()


def write_json(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + '.tmp')
    temp.write_text(json.dumps(obj, ensure_ascii=False, indent=2, allow_nan=False) + '\n')
    temp.replace(path)


def leaves(obj, prefix=''):
    if isinstance(obj, dict):
        for key, value in obj.items():
            yield from leaves(value, f'{prefix}.{key}' if prefix else key)
    elif isinstance(obj, (int, float)) and not isinstance(obj, bool):
        yield prefix, obj


def check_metrics(actual, reference):
    actual, reference = dict(leaves(actual)), dict(leaves(reference))
    if actual.keys() != reference.keys():
        raise ValueError('Original/reproduced per-case metric keys differ')
    rows = []
    for key, old in reference.items():
        new = actual[key]
        tol = 0.0 if isinstance(old, int) else 5e-6 + 5e-6 * abs(old)
        tolerance_basis = 'exact integer' if isinstance(old, int) else '5e-6 absolute + 5e-6 relative'
        prefix, _, field = key.rpartition('.')
        sibling = lambda name: reference.get(f'{prefix}.{name}' if prefix else name)
        if field in ('top10_precision', 'top10_recall', 'top10_iou'):
            nt, npred = sibling('n_high_true'), sibling('n_high_pred')
            if nt and npred:
                if field == 'top10_precision':
                    resolution = 1.0 / npred
                elif field == 'top10_recall':
                    resolution = 1.0 / nt
                else:
                    intersection = old * (nt + npred) / (1.0 + old)
                    candidates = [max(0, intersection - 1), min(nt, npred, intersection + 1)]
                    resolution = max(abs(k / (nt + npred - k) - old) for k in candidates)
                tol = max(tol, resolution + 1e-12)
                tolerance_basis = 'at most one hotspot intersection vertex; counts remain exact'
        passed = (math.isfinite(new) and abs(new - old) <= tol) if math.isfinite(old) else (math.isnan(new) and math.isnan(old))
        rows.append(dict(metric=key, passed=passed,
                         old=old if math.isfinite(old) else None,
                         reproduced=new if math.isfinite(new) else None,
                         abs_difference=abs(new-old) if math.isfinite(old) and math.isfinite(new) else None,
                         tolerance=tol if math.isfinite(tol) else None,
                         tolerance_basis=tolerance_basis))
    return dict(passed=all(r['passed'] for r in rows), checks=rows)


def main():
    if not os.environ.get('SLURM_JOB_ID') or not torch.cuda.is_available():
        raise RuntimeError('Use a Slurm GPU allocation for checkpoint inference')
    torch.set_num_threads(2)
    files = list((ROOT/'training_wss_min').glob('*.py'))
    files += [Path(__file__), *(RUN/name for name in ('config.json', 'ckpt_best.pt', 'feature_stats.json', 'wss_global_stats.json', 'eval/ckpt_best/metrics.json'))]
    before = {str(f.relative_to(ROOT)): sha(f) for f in files}
    cfg, feature_stats, model, checkpoint = E.load_model_from_run(RUN, 'cuda', 'best')
    model.eval()
    stats = E.load_wss_stats_for_run(RUN)
    assert (cfg.data.target, cfg.data.timesteps, cfg.model.out_dim, cfg.model.query_decoder,
            cfg.eval.fixed_support, cfg.eval.support_seed, cfg.eval.query_chunk_size) == (
                'velocity', 'peak', 3, 'qad_lite', True, 1234, 16384)
    ids = [f'{c}/{n}' for c, n in D.load_split_cases(cfg.data.split_path, 'test')]
    assert len(ids) == 34
    saved = json.loads((RUN/'eval/ckpt_best/metrics.json').read_text())['test']
    assert set(ids) == set(saved['per_case'])
    mean, std = (np.asarray(stats['velocity'][key], dtype=np.float64) for key in ('mean', 'std'))
    outdir = EXP/'velocity_best'
    outdir.mkdir(parents=True, exist_ok=True)
    record = dict(status='running', checkpoint='best', checkpoint_epoch=int(checkpoint['epoch']),
                  job_id=os.environ['SLURM_JOB_ID'], run=str(RUN), source_sha256=before,
                  split_path=cfg.data.split_path, split_sha256=sha(cfg.data.split_path),
                  expected_cases=ids, cases={}, frozen_support_seed=1234,
                  reused_prediction_archive=False,
                  prediction_note='Original run retained metrics but no pointwise archive; reproduced from original best checkpoint and frozen support protocol.')
    write_json(EXP/'velocity_reproduction.json', record)
    started = time.time()
    for i, uid in enumerate(ids):
        cohort, name = uid.rsplit('/', 1)
        case = D.load_case(cohort, name, stats, target='velocity',
                          target_normalization=cfg.data.target_normalization,
                          data_root=cfg.data.data_root, required_frame_version=cfg.data.required_frame_version)
        case['unit_id'] = uid
        rows = D.query_rows(case)
        prediction_norm = np.asarray(E.predict_case_norm(model, case, cfg.data.input_features,
                                                         feature_stats, 'cuda', cfg=cfg,
                                                         return_all_channels=True), dtype=np.float64)
        assert prediction_norm.shape == (len(rows), 3) and np.isfinite(prediction_norm).all()
        sub = D.subset_case(case, rows)
        metrics = E._evaluate_velocity([sub], [np.asarray(sub['y_norm'], dtype=np.float64)],
                                      [prediction_norm], cfg, stats, save_dir=None, make_plots=False)
        verification = {key: check_metrics(metrics['per_case'][uid] if key == 'physical' else metrics['normalized']['per_case'][uid],
                                           saved['per_case'][uid] if key == 'physical' else saved['normalized']['per_case'][uid])
                        for key in ('physical', 'normalized')}
        if not all(v['passed'] for v in verification.values()):
            write_json(EXP/f'velocity_reproduction_failed_{os.environ["SLURM_JOB_ID"]}.json', dict(case=uid, verification=verification))
            raise RuntimeError(f'R5V metric reproduction failed: {uid}')
        bundle = Path(case['bundle_path'])
        volume = Path(cfg.data.data_root)/uid/'volume.npz'
        with np.load(bundle, allow_pickle=True) as z:
            rotation = np.asarray(z['transform_rotation'], dtype=np.float64)
            origin = np.asarray(z['transform_centroid'], dtype=np.float64)
            scale = float(z['coord_scale'])
            wall = np.asarray(z['wall_coords_raw'], dtype=np.float64)
            step = int(z['peak_step']); idx = list(z['steps']).index(step)
            truth_wss = np.asarray(z['wall_wss'][idx], dtype=np.float64)
            truth_wss_vec = np.asarray(z['wall_wss_vec'][idx], dtype=np.float64)
        with np.load(volume, allow_pickle=False) as z:
            interior = np.asarray(z['vol_coords_raw'], dtype=np.float64)
            truth_aligned = np.asarray(z['vol_velocity_aligned_peak'], dtype=np.float64)
            assert step == int(z['peak_step']) == 1162
        assert np.allclose(rotation @ rotation.T, np.eye(3), atol=1e-7, rtol=0)
        assert len(interior) == len(rows) and np.array_equal(truth_aligned, sub['y_raw'])
        coordinate_error = float(np.max(np.abs((interior-origin) @ rotation.T / scale - sub['pos'])))
        assert coordinate_error * scale < 2e-4, coordinate_error * scale
        pred_aligned = prediction_norm * std + mean
        pred_raw = pred_aligned @ rotation
        truth_raw = truth_aligned @ rotation
        assert np.allclose(np.linalg.norm(pred_aligned, axis=1), np.linalg.norm(pred_raw, axis=1), atol=1e-10, rtol=1e-10)
        target = outdir/(uid.replace('/', '__')+'.npz')
        tmp = target.with_suffix('.tmp.npz')
        np.savez(tmp, canonical_id=np.asarray(uid), peak_step=np.int32(step),
                 wall_mm=wall, interior_mm=interior, velocity_pred_raw_m_s=pred_raw,
                 velocity_cfd_raw_m_s=truth_raw, velocity_pred_aligned_m_s=pred_aligned,
                 velocity_prediction_norm=prediction_norm, truth_wss_pa=truth_wss,
                 truth_wss_vec_pa=truth_wss_vec, transform_rotation=rotation,
                 transform_centroid=origin, coord_scale=np.float64(scale))
        tmp.replace(target)
        record['cases'][uid] = dict(path=str(target), sha256=sha(target),
                                   bundle=str(bundle), bundle_sha256=sha(bundle),
                                   volume=str(volume), volume_sha256=sha(volume),
                                   n_wall=len(wall), n_interior=len(interior),
                                   coordinate_roundtrip_max_error_mm=coordinate_error*scale,
                                   verification=verification)
        write_json(EXP/'velocity_reproduction.json', record)
        print(f'[{i+1}/34] {uid}: {len(interior)} vectors; original speed metrics verified', flush=True)
        del case, sub, metrics, prediction_norm, pred_aligned, pred_raw, truth_raw
        gc.collect()
    after = {str(f.relative_to(ROOT)): sha(f) for f in files}
    assert before == after
    record.update(status='complete', passed=True, n_cases=34, source_sha256_end=after,
                  elapsed_seconds=time.time()-started)
    write_json(EXP/'velocity_reproduction.json', record)


if __name__ == '__main__':
    main()

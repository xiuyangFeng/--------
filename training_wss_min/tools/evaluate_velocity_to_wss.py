"""Frozen R5V -> Profile-Secant V3 evaluation. Run substantial stages on Slurm."""
from __future__ import annotations

import argparse
import gc
import importlib
import json
import os
from pathlib import Path
import time

import numpy as np
from scipy.spatial import cKDTree

from training_wss_min.tools.export_r5v_for_wss import ROOT, EXP, RUN, sha, write_json
from training_wss_min.tools import profile_secant_adapter as A

DATA = ROOT / 'data_wss_v5/views/wss_min_view_v1'
STATS = DATA / 'wss_global_stats_train138.json'
CHUNK = 8192


def read(path):
    return json.loads(Path(path).read_text())


def save_npz(path, **arrays):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix('.tmp.npz')
    np.savez(tmp, **arrays)
    tmp.replace(path)


def paths(uid):
    slug = uid.replace('/', '__')
    return EXP/'velocity_best'/f'{slug}.npz', EXP/'geometry'/f'{slug}.npz'


def source_hashes():
    files = [Path(__file__), ROOT/'training_wss_min/tools/export_r5v_for_wss.py',
             ROOT/'wss_mri_calculator/src/run_v3_experiment.py', STATS,
             DATA/'split_V5_train138_test34.json',
             *sorted((ROOT/'training_wss_min').glob('*.py')),
             *sorted((ROOT/'wss_pinn/physics').glob('*.py'))]
    return {str(p): sha(p) for p in files}


def verify_sources(provenance):
    assert source_hashes() == provenance['evaluation_source_sha256'], 'Evaluation sources changed'
    assert A.frozen_sources() == provenance['frozen_sources'], 'Frozen algorithm changed'
    reproduction = read(EXP/'velocity_reproduction.json')
    assert reproduction['passed'] and reproduction['n_cases'] == 34
    assert sha(EXP/'velocity_reproduction.json') == provenance['velocity_reproduction_sha256']
    for path, digest in reproduction['source_sha256'].items():
        assert sha(ROOT/path) == digest, path


def prepare():
    reproduction = read(EXP/'velocity_reproduction.json')
    assert reproduction['status'] == 'complete' and reproduction['passed']
    contracts = A.validate_adapter_contracts()
    runtime = A._runtime()
    radius_module = importlib.import_module('run_v3_experiment')
    provenance = dict(experiment='VELWSS1', checkpoint='best', seed=1234,
                      new_training=False, new_centerlines=False,
                      calibration='frozen; historically WSS-supervised on train138',
                      frozen_sources=A.frozen_sources(), adapter_contracts=contracts,
                      evaluation_source_sha256=source_hashes(),
                      velocity_reproduction_sha256=sha(EXP/'velocity_reproduction.json'),
                      normalization_path=str(STATS), normalization_sha256=sha(STATS),
                      n_cases=34, geometry={}, chunks=[], job_id=os.environ['SLURM_JOB_ID'])
    for uid in reproduction['expected_cases']:
        source, geometry_path = paths(uid)
        assert sha(source) == reproduction['cases'][uid]['sha256']
        with np.load(source, allow_pickle=False) as z:
            wall, interior = z['wall_mm'], z['interior_mm']
        normals = runtime['calculate_wss_cfd'].pca_wall_normals(wall, interior, cKDTree(interior))
        radius, radius_source = radius_module._resolve_radius(
            'bundle_then_wall', ROOT/'data_new'/uid, wall, normals,
            wall, normals, cKDTree(wall),
            {'wall_radius_search_neighbors': 512, 'wall_radius_opposite_normal_cosine': -0.6})
        assert radius is not None and np.isfinite(radius).all() and (radius > 0).all()
        old_bundle = ROOT/'data_wss_min'/uid/'bundle.npz'
        radius_evidence = dict(source=radius_source, median_mm=float(np.median(radius)),
                               min_mm=float(radius.min()), max_mm=float(radius.max()))
        if radius_source == 'bundle_centerline_radius':
            with np.load(old_bundle, allow_pickle=False) as z:
                old_wall = np.asarray(z['wall_coords_raw'], dtype=np.float64)
            rescale = float(np.nanmedian(np.linalg.norm(old_wall, axis=1))) < 20
            if rescale:
                old_wall *= 1000
            distance, _ = cKDTree(old_wall).query(wall)
            radius_evidence.update(bundle=str(old_bundle), bundle_sha256=sha(old_bundle),
                                   legacy_coordinate_scale1000=rescale,
                                   mapping_distance_mm_p50=float(np.median(distance)),
                                   mapping_distance_mm_max=float(distance.max()))
        save_npz(geometry_path, canonical_id=np.asarray(uid), normals=normals, radius_mm=radius)
        provenance['geometry'][uid] = dict(path=str(geometry_path), sha256=sha(geometry_path),
                                           n_wall=len(wall), radius=radius_evidence)
        for start in range(0, len(wall), CHUNK):
            provenance['chunks'].append(dict(index=len(provenance['chunks']), uid=uid,
                                              start=start, stop=min(start+CHUNK, len(wall))))
        print(f'geometry {uid}: {len(wall)} wall targets; {radius_source}', flush=True)
        del wall, interior, normals, radius
        gc.collect()
    provenance['n_wall'] = sum(g['n_wall'] for g in provenance['geometry'].values())
    assert provenance['evaluation_source_sha256'] == source_hashes()
    write_json(EXP/'provenance.json', provenance)
    print(f"Prepared {len(provenance['chunks'])} chunks / {provenance['n_wall']} wall targets", flush=True)


def worker(index):
    started = time.time()
    p = read(EXP/'provenance.json')
    verify_sources(p)
    row = p['chunks'][index]
    uid = row['uid']
    source, geometry_path = paths(uid)
    record = read(EXP/'velocity_reproduction.json')['cases'][uid]
    assert sha(source) == record['sha256']
    assert sha(geometry_path) == p['geometry'][uid]['sha256']
    with np.load(source, allow_pickle=False) as z:
        wall, interior = z['wall_mm'], z['interior_mm']
        velocities = {key: z[f'velocity_{key}_raw_m_s'] for key in ('pred', 'cfd')}
    with np.load(geometry_path, allow_pickle=False) as z:
        normals, radius = z['normals'], z['radius_mm']
    indices = np.arange(row['start'], row['stop'], dtype=np.int64)
    arrays = {}
    for source_name, velocity in velocities.items():
        cache = A.build_cache(uid, wall, normals, interior, velocity, radius, indices)
        arrays.update({f'{source_name}__{key}': value for key, value in cache.items()})
        print(f'chunk {index} {source_name} {uid} [{row["start"]}:{row["stop"]}] elapsed={time.time()-started:.1f}s', flush=True)
    target = EXP/'chunks'/f'{index:04d}.npz'
    verify_sources(p)
    save_npz(target, **arrays)
    write_json(target.with_suffix('.json'), dict(**row, sha256=sha(target),
                elapsed_seconds=time.time()-started, job_id=os.environ['SLURM_JOB_ID'],
                task_id=os.environ.get('SLURM_ARRAY_TASK_ID'), passed=True))


def clean_json(value):
    if isinstance(value, dict):
        return {k: clean_json(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [clean_json(v) for v in value]
    if isinstance(value, (float, np.floating)):
        return float(value) if np.isfinite(value) else None
    if isinstance(value, np.integer):
        return int(value)
    return value


def assemble():
    from training_wss_min import dataset as D, evaluate as E
    p = read(EXP/'provenance.json')
    verify_sources(p)
    audit = read(EXP/'radius_split_audit.json')
    stats = read(STATS)
    cases, truths = [], []
    predictions = {key: [] for key in ('test', 'oracle', 'physics_only')}
    coverage = {}
    artifacts = {}
    for uid in read(EXP/'velocity_reproduction.json')['expected_cases']:
        shards = [r for r in p['chunks'] if r['uid'] == uid]
        caches = {key: {} for key in ('pred', 'cfd')}
        for row in shards:
            target = EXP/'chunks'/f'{row["index"]:04d}.npz'
            status = read(target.with_suffix('.json'))
            assert status['passed'] and status['sha256'] == sha(target)
            with np.load(target, allow_pickle=False) as z:
                for source_name in caches:
                    for name in z.files:
                        if name.startswith(source_name+'__'):
                            key = name[len(source_name)+2:]
                            caches[source_name].setdefault(key, []).append(z[name])
        results = {}
        for source_name, cache in caches.items():
            merged = {}
            for key, values in cache.items():
                if values[0].ndim == 0:
                    assert all(np.array_equal(values[0], v) for v in values)
                    merged[key] = values[0]
                else:
                    merged[key] = np.concatenate(values)
            assert np.array_equal(merged['sample_index'], np.arange(p['geometry'][uid]['n_wall']))
            results[source_name] = A.calibrate_case(merged)
        source, _ = paths(uid)
        with np.load(source, allow_pickle=False) as z:
            truth, wall = z['truth_wss_pa'], z['wall_mm']
            true_vec = z['truth_wss_vec_pa']
        cohort, name = uid.rsplit('/', 1)
        case = D.load_case(cohort, name, stats, target='wss', data_root=str(DATA),
                           required_frame_version='v5_atlas_frame_v1')
        case['unit_id'] = uid
        assert np.array_equal(case['y_raw'].astype(np.float64), truth)
        assert len(truth) == p['geometry'][uid]['n_wall']
        outputs = dict(test=results['pred']['wss_mag_pa'], oracle=results['cfd']['wss_mag_pa'],
                       physics_only=results['pred']['physics_wss_mag_pa'])
        valid = np.isfinite(truth)
        for values in outputs.values():
            valid &= np.isfinite(values) & (values >= 0)
        coverage[uid] = dict(n_wall=len(truth), n_valid=int(valid.sum()), coverage=float(valid.mean()),
                             case_scales_pred=results['pred']['case_scales'],
                             case_scales_oracle=results['cfd']['case_scales'])
        if not valid.all():
            write_json(EXP/'invalid_coverage.json', coverage)
            raise RuntimeError(f'Incomplete full-wall coverage: {uid}; no silent masking allowed')
        target = EXP/'predictions'/f'{uid.replace("/", "__")}.npz'
        save_npz(target, canonical_id=np.asarray(uid), wall_mm=wall, truth_wss_pa=truth,
                 truth_wss_vec_pa=true_vec, pred_wss_pa=outputs['test'], oracle_wss_pa=outputs['oracle'],
                 physics_wss_pa=outputs['physics_only'], pred_wss_vec_pa=results['pred']['wss_vec_pa'],
                 oracle_wss_vec_pa=results['cfd']['wss_vec_pa'],
                 physics_wss_vec_pa=results['pred']['physics_wss_vec_pa'],
                 pred_calibration_ratio=results['pred']['high_tail_ratio'],
                 oracle_calibration_ratio=results['cfd']['high_tail_ratio'], valid_mask=valid)
        artifacts[uid] = dict(path=str(target), sha256=sha(target))
        truths.append(truth)
        cases.append(case)
        for key in predictions:
            predictions[key].append(outputs[key])
        print(f'assembled {uid}: full-wall coverage {valid.sum()}/{len(truth)}', flush=True)
        del caches, merged, results
        gc.collect()
    metrics = {}
    for key, preds in predictions.items():
        metrics[key] = E._evaluate_space(cases, truths, preds, save_dir=None, make_plots=False,
                                         plot_space='physical target', include_area=False)
        metrics[key]['normalized'] = E._evaluate_space(
            cases, [D.normalize_wss(v, stats) for v in truths],
            [D.normalize_wss(v, stats) for v in preds], save_dir=None, make_plots=False,
            plot_space='normalized', include_area=False)
        print(key, metrics[key]['field_casebalanced'], flush=True)
    verify_sources(p)
    write_json(EXP/'metrics.json', clean_json(metrics))
    write_json(EXP/'coverage.json', clean_json(coverage))
    write_json(EXP/'prediction_manifest.json', artifacts)
    # Leakage gate is explicitly re-derived from the audited case sets.
    audit_train = set(audit['calibrator_train_cases'])
    train_ids = {f'{c}/{n}' for c, n in D.load_split_cases(DATA/'split_V5_train138_test34.json', 'train')}
    test_ids = set(artifacts)
    assert audit_train == train_ids and len(train_ids) == 138
    assert not (audit_train & test_ids) and len(test_ids) == 34
    gate = dict(passed=True, checkpoint='best', n_cases=len(cases), n_wall=sum(len(x) for x in truths),
                valid_coverage=1.0, velocity_reproduced=True, no_new_training=True,
                calibrator_train_test_overlap=0, calibrator_train_matches_current_train138=True,
                provenance_sha256=sha(EXP/'provenance.json'), metrics_sha256=sha(EXP/'metrics.json'),
                prediction_manifest_sha256=sha(EXP/'prediction_manifest.json'),
                radius_split_audit_sha256=sha(EXP/'radius_split_audit.json'),
                job_id=os.environ['SLURM_JOB_ID'])
    write_json(EXP/'evaluation_gate.json', gate)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('stage', choices=['prepare', 'worker', 'assemble'])
    parser.add_argument('--index', type=int)
    args = parser.parse_args()
    if not os.environ.get('SLURM_JOB_ID'):
        raise RuntimeError('Run this workload through Slurm')
    if args.stage == 'prepare':
        prepare()
    elif args.stage == 'assemble':
        assemble()
    else:
        worker(args.index if args.index is not None else int(os.environ['SLURM_ARRAY_TASK_ID']))


if __name__ == '__main__':
    main()

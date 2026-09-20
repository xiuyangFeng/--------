"""VELWSS2: VF6 three-seed predicted velocity -> frozen Profile-Secant V3 -> WSS.

Same frozen operator, wall, normals and calibration protocol as VELWSS1
(``evaluate_velocity_to_wss.py``); differences are only (1) the velocity model
(VF6 seeds 1234/7/2025 instead of R5V) and (2) an additional pre-registered
radius sources reported next to the unchanged ``legacy`` radius:
``legacy_exact`` keeps the legacy bundle radius field but maps it with the
legacy wall coordinates rescaled by 1000/unit_factor (exact node match, so only
the mapping error is removed), and ``v5atlas`` uses the V5 bundle
``wall_local_radius`` directly (new atlas radius definition, no mapping).

Stages (all substantial stages run through Slurm):
  export   --seed s1234|s7|s2025          GPU: reproduce VF6 best predictions, verify vs saved metrics
  prepare                                 CPU: normals + both radius sources + chunk plan
  worker   --radius legacy|v5atlas --index i   CPU array: frozen gradient stages for cfd + 3 seeds
  assemble --radius legacy|v5atlas        CPU: merge chunks, per-case calibration, full-wall metrics
"""
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

from training_wss_min.tools.export_r5v_for_wss import ROOT, check_metrics, sha, write_json
from training_wss_min.tools import profile_secant_adapter as A

EXPERIMENT = 'VELWSS2'
EXP = ROOT / 'training_wss_min/experiments/vf6_velocity_to_wss_20260916'
VELWSS1 = ROOT / 'training_wss_min/experiments/v5_velocity_to_wss_20260909'
RUNS = {
    's1234': ROOT / 'training_wss_min/runs/wss_local_wave2_20260912/VF6_s1234',
    's7': ROOT / 'training_wss_min/runs/wss_local_wave3_20260913/VF6_s7',
    's2025': ROOT / 'training_wss_min/runs/wss_local_wave3_20260913/VF6_s2025',
}
SEEDS = tuple(RUNS)
REFERENCE_SEED = 's1234'          # wall / interior / CFD velocity arrays are taken from this export
RADIUS_SOURCES = ('legacy', 'legacy_exact', 'v5atlas')
VELOCITY_SOURCES = ('cfd', *SEEDS)
DATA = ROOT / 'data_wss_v5/views/wss_min_view_v1'
STATS = DATA / 'wss_global_stats_train138.json'
SPLIT = DATA / 'split_V5_train138_test34.json'
CHUNK = 8192
PEAK_STEP = 1162
LEGACY_GEOMETRY = {'wall_radius_search_neighbors': 512, 'wall_radius_opposite_normal_cosine': -0.6}


def read(path):
    return json.loads(Path(path).read_text())


def save_npz(path, **arrays):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix('.tmp.npz')
    np.savez(tmp, **arrays)
    tmp.replace(path)


def slug(uid):
    return uid.replace('/', '__')


def velocity_path(exp, seed, uid):
    return exp / 'velocity' / seed / f'{slug(uid)}.npz'


def reproduction_path(exp, seed):
    return exp / 'velocity' / f'{seed}_reproduction.json'


def geometry_path(exp, uid):
    return exp / 'geometry' / f'{slug(uid)}.npz'


def chunk_path(exp, radius, index):
    return exp / 'chunks' / radius / f'{index:04d}.npz'


def root_source_hashes(extra=()):
    files = [Path(__file__), ROOT / 'training_wss_min/tools/export_r5v_for_wss.py',
             ROOT / 'training_wss_min/tools/profile_secant_adapter.py',
             ROOT / 'wss_mri_calculator/src/run_v3_experiment.py', STATS, SPLIT,
             *sorted((ROOT / 'training_wss_min').glob('*.py')),
             *sorted((ROOT / 'wss_pinn/physics').glob('*.py')), *extra]
    return {str(p.relative_to(ROOT)): sha(p) for p in files}


def frozen_operator_hashes():
    """Only the files that the frozen gradient/calibration stages execute."""
    files = [Path(__file__), ROOT / 'training_wss_min/tools/profile_secant_adapter.py',
             ROOT / 'wss_mri_calculator/src/run_v3_experiment.py',
             *sorted((ROOT / 'wss_pinn/physics').glob('*.py'))]
    return {str(p.relative_to(ROOT)): sha(p) for p in files}


def verify_frozen(provenance):
    assert frozen_operator_hashes() == provenance['frozen_operator_sha256'], 'Frozen operator sources changed'
    assert A.frozen_sources() == provenance['frozen_sources'], 'Frozen algorithm changed'


# --------------------------------------------------------------------------- export

def load_velocity_case(cfg, stats, cohort, name):
    """Mirror dataset.load_partition's per-case call for a velocity run."""
    from training_wss_min import config as C, dataset as D
    extra = C.v6_point_features(cfg)
    case = D.load_case(cohort, name, stats, target=cfg.data.target,
                       target_normalization=cfg.data.target_normalization,
                       data_root=cfg.data.data_root,
                       required_frame_version=cfg.data.required_frame_version,
                       case_features=None, timesteps=getattr(cfg.data, 'timesteps', 'peak'),
                       waveform_path=getattr(cfg.data, 'waveform_path', None),
                       extra_point_features=extra,
                       point_features_root=getattr(cfg.data, 'point_features_root', None))
    lg = getattr(D, 'LG', None)
    if lg is not None and set(extra) & set(getattr(lg, 'FEATURE_KEYS', ())):
        case['_longitudinal_partition'] = 'test'
    assert cfg.data.case_features_path is None
    return case


def export(seed, exp=EXP, device='cuda', limit=None):
    import torch
    from training_wss_min import dataset as D, evaluate as E
    if device == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('CUDA requested but unavailable')
    import socket
    job_id = os.environ.get('SLURM_JOB_ID')
    execution = 'slurm' if job_id else ('login-node shell outside Slurm: master GPU gres fully held by the '
                                        'wss_v51 queue (14428) and no-gres Slurm jobs cannot see GPUs')
    job_id = job_id or f'login:{socket.gethostname()}:pid{os.getpid()}'
    gpu_free_mib = (int(torch.cuda.mem_get_info()[0] // 2**20) if device == 'cuda' else None)
    torch.set_num_threads(2)
    run = RUNS[seed]
    run_files = [run / name for name in ('config.json', 'ckpt_best.pt', 'feature_stats.json',
                                         'wss_global_stats.json', 'eval/ckpt_best/metrics.json')]
    before = root_source_hashes(run_files)
    cfg, feature_stats, model, checkpoint = E.load_model_from_run(run, device, 'best')
    model.eval()
    stats = E.load_wss_stats_for_run(run)
    assert (cfg.data.target, cfg.data.timesteps, cfg.model.out_dim, cfg.model.query_decoder,
            cfg.eval.fixed_support, cfg.eval.support_seed, cfg.eval.query_chunk_size) == (
                'velocity', 'peak', 3, 'qad_lite', True, 1234, 16384)
    assert int(cfg.train.seed) == int(seed[1:]), (cfg.train.seed, seed)
    assert Path(cfg.data.split_path).resolve() == SPLIT.resolve()
    ids = [f'{c}/{n}' for c, n in D.load_split_cases(cfg.data.split_path, 'test')]
    assert len(ids) == 34
    saved = read(run / 'eval/ckpt_best/metrics.json')['test']
    assert set(ids) == set(saved['per_case'])
    if limit:
        ids = ids[:limit]
    mean, std = (np.asarray(stats['velocity'][key], dtype=np.float64) for key in ('mean', 'std'))
    record = dict(experiment=EXPERIMENT, status='running', seed=seed, checkpoint='best',
                  checkpoint_epoch=int(checkpoint['epoch']), job_id=job_id, execution=execution,
                  hostname=socket.gethostname(), started_at=time.strftime('%Y-%m-%dT%H:%M:%S%z'),
                  run=str(run), device=device, cuda_visible_devices=os.environ.get('CUDA_VISIBLE_DEVICES'),
                  gpu_name=(torch.cuda.get_device_name(0) if device == 'cuda' else None),
                  gpu_free_mib_at_start=gpu_free_mib,
                  source_sha256=before, split_path=cfg.data.split_path, split_sha256=sha(cfg.data.split_path),
                  saved_metrics=dict(physical_r2_cb=saved['field_casebalanced']['r2'],
                                     normalized_r2_cb=saved['normalized']['field_casebalanced']['r2']),
                  expected_cases=ids, cases={}, frozen_support_seed=1234, reused_prediction_archive=False,
                  smoke=bool(limit),
                  prediction_note='VF6 run retained metrics but no pointwise archive; reproduced from the '
                                  'original best checkpoint with the frozen fixed-support protocol.')
    write_json(reproduction_path(exp, seed), record)
    started = time.time()
    for i, uid in enumerate(ids):
        cohort, name = uid.rsplit('/', 1)
        case = load_velocity_case(cfg, stats, cohort, name)
        case['unit_id'] = uid
        rows = D.query_rows(case)
        prediction_norm = np.asarray(E.predict_case_norm(model, case, cfg.data.input_features, feature_stats,
                                                         device, cfg=cfg, return_all_channels=True), dtype=np.float64)
        assert prediction_norm.shape == (len(rows), 3) and np.isfinite(prediction_norm).all()
        sub = D.subset_case(case, rows)
        metrics = E._evaluate_velocity([sub], [np.asarray(sub['y_norm'], dtype=np.float64)], [prediction_norm],
                                       cfg, stats, save_dir=None, make_plots=False)
        verification = {key: check_metrics(metrics['per_case'][uid] if key == 'physical' else metrics['normalized']['per_case'][uid],
                                           saved['per_case'][uid] if key == 'physical' else saved['normalized']['per_case'][uid])
                        for key in ('physical', 'normalized')}
        if not all(v['passed'] for v in verification.values()):
            write_json(exp / 'velocity' / f'{seed}_reproduction_failed_{job_id.replace(":", "_")}.json',
                       dict(case=uid, verification=verification))
            raise RuntimeError(f'VF6 {seed} metric reproduction failed: {uid}')
        bundle = Path(case['bundle_path'])
        volume = Path(cfg.data.data_root) / uid / 'volume.npz'
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
            assert step == int(z['peak_step']) == PEAK_STEP
        assert np.allclose(rotation @ rotation.T, np.eye(3), atol=1e-7, rtol=0)
        assert len(interior) == len(rows) and np.array_equal(truth_aligned, sub['y_raw'])
        coordinate_error = float(np.max(np.abs((interior - origin) @ rotation.T / scale - sub['pos'])))
        assert coordinate_error * scale < 2e-4, coordinate_error * scale
        pred_aligned = prediction_norm * std + mean
        pred_raw = pred_aligned @ rotation
        truth_raw = truth_aligned @ rotation
        assert np.allclose(np.linalg.norm(pred_aligned, axis=1), np.linalg.norm(pred_raw, axis=1), atol=1e-10, rtol=1e-10)
        target = velocity_path(exp, seed, uid)
        save_npz(target, canonical_id=np.asarray(uid), seed=np.asarray(seed), run=np.asarray(str(run)),
                 peak_step=np.int32(step), wall_mm=wall, interior_mm=interior,
                 velocity_pred_raw_m_s=pred_raw, velocity_cfd_raw_m_s=truth_raw,
                 velocity_pred_aligned_m_s=pred_aligned, velocity_prediction_norm=prediction_norm,
                 truth_wss_pa=truth_wss, truth_wss_vec_pa=truth_wss_vec, transform_rotation=rotation,
                 transform_centroid=origin, coord_scale=np.float64(scale))
        record['cases'][uid] = dict(path=str(target), sha256=sha(target), bundle=str(bundle), bundle_sha256=sha(bundle),
                                   volume=str(volume), volume_sha256=sha(volume), n_wall=len(wall),
                                   n_interior=len(interior), coordinate_roundtrip_max_error_mm=coordinate_error * scale,
                                   speed_r2=metrics['per_case'][uid]['overall']['r2'], verification=verification)
        write_json(reproduction_path(exp, seed), record)
        print(f'[{seed} {i + 1}/{len(ids)}] {uid}: {len(interior)} vectors; saved speed metrics reproduced', flush=True)
        del case, sub, metrics, prediction_norm, pred_aligned, pred_raw, truth_raw
        gc.collect()
    after = root_source_hashes(run_files)
    assert before == after, 'sources changed during export'
    record.update(status='complete', passed=True, n_cases=len(ids), source_sha256_end=after,
                  elapsed_seconds=time.time() - started)
    write_json(reproduction_path(exp, seed), record)


# --------------------------------------------------------------------------- prepare

def prepare(exp=EXP):
    reproductions = {seed: read(reproduction_path(exp, seed)) for seed in SEEDS}
    ids = reproductions[REFERENCE_SEED]['expected_cases']
    for seed, rep in reproductions.items():
        assert rep['status'] == 'complete' and rep['passed'] and rep['n_cases'] == 34 and not rep['smoke'], seed
        assert rep['expected_cases'] == ids, seed
    contracts = A.validate_adapter_contracts()
    runtime = A._runtime()
    radius_module = importlib.import_module('run_v3_experiment')
    audit = read(VELWSS1 / 'radius_split_audit.json')
    assert sha(SPLIT) == audit['split_evidence']['v5_split_sha256']
    assert len(audit['calibrator_train_cases']) == 138 and not (set(audit['calibrator_train_cases']) & set(ids))
    provenance = dict(experiment=EXPERIMENT, checkpoint='best', seeds=list(SEEDS), radius_sources=list(RADIUS_SOURCES),
                      velocity_sources=list(VELOCITY_SOURCES), new_training=False, new_centerlines=False,
                      calibration='frozen; historically WSS-supervised on train138 with the legacy radius mapping',
                      frozen_sources=A.frozen_sources(), adapter_contracts=contracts,
                      frozen_operator_sha256=frozen_operator_hashes(), evaluation_source_sha256=root_source_hashes(),
                      velocity_reproduction_sha256={seed: sha(reproduction_path(exp, seed)) for seed in SEEDS},
                      velocity_runs={seed: str(RUNS[seed]) for seed in SEEDS},
                      velwss1_radius_split_audit_sha256=sha(VELWSS1 / 'radius_split_audit.json'),
                      normalization_path=str(STATS), normalization_sha256=sha(STATS),
                      legacy_geometry=LEGACY_GEOMETRY, n_cases=34, geometry={}, chunks=[],
                      job_id=os.environ['SLURM_JOB_ID'])
    for uid in ids:
        ref = velocity_path(exp, REFERENCE_SEED, uid)
        assert sha(ref) == reproductions[REFERENCE_SEED]['cases'][uid]['sha256']
        with np.load(ref, allow_pickle=False) as z:
            wall, interior, cfd = z['wall_mm'], z['interior_mm'], z['velocity_cfd_raw_m_s']
        for seed in SEEDS:
            if seed == REFERENCE_SEED:
                continue
            other = velocity_path(exp, seed, uid)
            assert sha(other) == reproductions[seed]['cases'][uid]['sha256']
            with np.load(other, allow_pickle=False) as z:
                assert np.array_equal(z['wall_mm'], wall) and np.array_equal(z['interior_mm'], interior), (seed, uid)
                assert np.array_equal(z['velocity_cfd_raw_m_s'], cfd), (seed, uid)
        normals = runtime['calculate_wss_cfd'].pca_wall_normals(wall, interior, cKDTree(interior))
        legacy, legacy_source = radius_module._resolve_radius(
            'bundle_then_wall', ROOT / 'data_new' / uid, wall, normals, wall, normals, cKDTree(wall), LEGACY_GEOMETRY)
        assert legacy is not None and np.isfinite(legacy).all() and (legacy > 0).all()
        evidence = {'legacy': dict(source=legacy_source, median_mm=float(np.median(legacy)),
                                   min_mm=float(legacy.min()), max_mm=float(legacy.max()))}
        old_bundle = ROOT / 'data_wss_min' / uid / 'bundle.npz'
        unit_factor = None
        if legacy_source == 'bundle_centerline_radius':
            with np.load(old_bundle, allow_pickle=False) as z:
                old_wall = np.asarray(z['wall_coords_raw'], dtype=np.float64)
                unit_factor = float(z['unit_factor']) if 'unit_factor' in z.files else None
            rescale = float(np.nanmedian(np.linalg.norm(old_wall, axis=1))) < 20
            if rescale:
                old_wall *= 1000
            distance, _ = cKDTree(old_wall).query(wall)
            evidence['legacy'].update(bundle=str(old_bundle), bundle_sha256=sha(old_bundle), legacy_unit_factor=unit_factor,
                                      legacy_coordinate_scale1000=rescale,
                                      mapping_distance_mm_p50=float(np.median(distance)),
                                      mapping_distance_mm_max=float(distance.max()))
        assert legacy_source == 'bundle_centerline_radius' and unit_factor is not None and unit_factor > 0, uid
        with np.load(old_bundle, allow_pickle=False) as z:
            old_wall_exact = np.asarray(z['wall_coords_raw'], dtype=np.float64) * (1000.0 / unit_factor)
            old_radius = np.asarray(z['wall_local_radius'], dtype=np.float64)
        exact_distance, exact_nearest = cKDTree(old_wall_exact).query(wall)
        assert float(exact_distance.max()) < 1e-3, (uid, float(exact_distance.max()))
        assert len(np.unique(exact_nearest)) == len(wall), uid
        legacy_exact = old_radius[exact_nearest]
        assert np.isfinite(legacy_exact).all() and (legacy_exact > 0).all(), uid
        evidence['legacy_exact'] = dict(source='legacy_bundle_radius_exact_node_match', bundle=str(old_bundle),
                                        rescale='wall_coords_raw * 1000 / unit_factor', unit_factor=unit_factor,
                                        mapping_distance_mm_max=float(exact_distance.max()),
                                        median_mm=float(np.median(legacy_exact)), min_mm=float(legacy_exact.min()),
                                        max_mm=float(legacy_exact.max()))
        ratio_exact = legacy / legacy_exact
        evidence['legacy_over_legacy_exact'] = dict(
            ratio_p05=float(np.quantile(ratio_exact, 0.05)), ratio_p50=float(np.median(ratio_exact)),
            ratio_p95=float(np.quantile(ratio_exact, 0.95)),
            abs_diff_mm_p50=float(np.median(np.abs(legacy - legacy_exact))),
            abs_diff_mm_p95=float(np.quantile(np.abs(legacy - legacy_exact), 0.95)),
            fraction_off_by_20pct=float(np.mean(np.abs(ratio_exact - 1) > 0.2)))
        v5_bundle = DATA / uid / 'bundle.npz'
        with np.load(v5_bundle, allow_pickle=False) as z:
            assert np.array_equal(np.asarray(z['wall_coords_raw'], dtype=np.float64), wall), uid
            assert str(np.asarray(z['transform_frame_version']).item()) == 'v5_atlas_frame_v1'
            v5 = np.asarray(z['wall_local_radius'], dtype=np.float64)
            v5_unit = float(z['unit_factor'])
        assert v5.shape == (len(wall),) and np.isfinite(v5).all() and (v5 > 0).all(), uid
        ratio = legacy / v5
        evidence['v5atlas'] = dict(source='v5_bundle_wall_local_radius', bundle=str(v5_bundle), bundle_sha256=sha(v5_bundle),
                                   unit_factor=v5_unit, median_mm=float(np.median(v5)), min_mm=float(v5.min()),
                                   max_mm=float(v5.max()))
        evidence['legacy_over_v5atlas'] = dict(ratio_p50=float(np.median(ratio)), ratio_p05=float(np.quantile(ratio, 0.05)),
                                               ratio_p95=float(np.quantile(ratio, 0.95)),
                                               abs_diff_mm_p50=float(np.median(np.abs(legacy - v5))),
                                               abs_diff_mm_p95=float(np.quantile(np.abs(legacy - v5), 0.95)),
                                               fraction_off_by_20pct=float(np.mean(np.abs(ratio - 1) > 0.2)))
        gpath = geometry_path(exp, uid)
        save_npz(gpath, canonical_id=np.asarray(uid), normals=normals, radius_legacy_mm=legacy,
                 radius_legacy_exact_mm=legacy_exact, radius_v5atlas_mm=v5)
        provenance['geometry'][uid] = dict(path=str(gpath), sha256=sha(gpath), n_wall=len(wall), radius=evidence)
        for start in range(0, len(wall), CHUNK):
            provenance['chunks'].append(dict(index=len(provenance['chunks']), uid=uid, start=start,
                                             stop=min(start + CHUNK, len(wall))))
        print(f"geometry {uid}: {len(wall)} wall targets; legacy={legacy_source} "
              f"map_max={evidence['legacy'].get('mapping_distance_mm_max', float('nan')):.2f}mm "
              f"legacy/v5 p50={evidence['legacy_over_v5atlas']['ratio_p50']:.3f}", flush=True)
        del wall, interior, cfd, normals, legacy, legacy_exact, v5
        gc.collect()
    provenance['n_wall'] = sum(g['n_wall'] for g in provenance['geometry'].values())
    assert provenance['frozen_operator_sha256'] == frozen_operator_hashes()
    write_json(exp / 'provenance.json', provenance)
    print(f"Prepared {len(provenance['chunks'])} chunks / {provenance['n_wall']} wall targets", flush=True)


# --------------------------------------------------------------------------- worker

def worker(radius, index, exp=EXP):
    started = time.time()
    p = read(exp / 'provenance.json')
    verify_frozen(p)
    row = p['chunks'][index]
    uid = row['uid']
    reproductions = {seed: read(reproduction_path(exp, seed)) for seed in SEEDS}
    velocities = {}
    for seed in SEEDS:
        source = velocity_path(exp, seed, uid)
        assert sha(source) == reproductions[seed]['cases'][uid]['sha256']
        with np.load(source, allow_pickle=False) as z:
            if seed == REFERENCE_SEED:
                wall, interior = z['wall_mm'], z['interior_mm']
                velocities['cfd'] = z['velocity_cfd_raw_m_s']
            velocities[seed] = z['velocity_pred_raw_m_s']
    gpath = geometry_path(exp, uid)
    assert sha(gpath) == p['geometry'][uid]['sha256']
    with np.load(gpath, allow_pickle=False) as z:
        normals, radius_mm = z['normals'], z[f'radius_{radius}_mm']
    indices = np.arange(row['start'], row['stop'], dtype=np.int64)
    arrays = {}
    for source_name in VELOCITY_SOURCES:
        cache = A.build_cache(uid, wall, normals, interior, velocities[source_name], radius_mm, indices)
        arrays.update({f'{source_name}__{key}': value for key, value in cache.items()})
        print(f'chunk {radius}/{index} {source_name} {uid} [{row["start"]}:{row["stop"]}] '
              f'elapsed={time.time() - started:.1f}s', flush=True)
    target = chunk_path(exp, radius, index)
    verify_frozen(p)
    save_npz(target, **arrays)
    write_json(target.with_suffix('.json'), dict(**row, radius=radius, sha256=sha(target),
                                                 elapsed_seconds=time.time() - started,
                                                 job_id=os.environ['SLURM_JOB_ID'],
                                                 task_id=os.environ.get('SLURM_ARRAY_TASK_ID'), passed=True))


# --------------------------------------------------------------------------- assemble

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


def output_keys():
    keys = []
    for source in VELOCITY_SOURCES:
        keys += [f'{source}_calibrated', f'{source}_physics']
    return keys


def assemble(radius, exp=EXP):
    from training_wss_min import dataset as D, evaluate as E
    p = read(exp / 'provenance.json')
    verify_frozen(p)
    source_hashes_start = root_source_hashes()
    stats = read(STATS)
    reproductions = {seed: read(reproduction_path(exp, seed)) for seed in SEEDS}
    ids = reproductions[REFERENCE_SEED]['expected_cases']
    cases, truths = [], []
    predictions = {key: [] for key in output_keys()}
    coverage, artifacts = {}, {}
    for uid in ids:
        shards = [r for r in p['chunks'] if r['uid'] == uid]
        caches = {key: {} for key in VELOCITY_SOURCES}
        for row in shards:
            target = chunk_path(exp, radius, row['index'])
            status = read(target.with_suffix('.json'))
            assert status['passed'] and status['radius'] == radius and status['sha256'] == sha(target)
            with np.load(target, allow_pickle=False) as z:
                for source_name in caches:
                    prefix = source_name + '__'
                    for name in z.files:
                        if name.startswith(prefix):
                            caches[source_name].setdefault(name[len(prefix):], []).append(z[name])
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
        ref = velocity_path(exp, REFERENCE_SEED, uid)
        assert sha(ref) == reproductions[REFERENCE_SEED]['cases'][uid]['sha256']
        with np.load(ref, allow_pickle=False) as z:
            truth, wall, true_vec = z['truth_wss_pa'], z['wall_mm'], z['truth_wss_vec_pa']
        cohort, name = uid.rsplit('/', 1)
        case = D.load_case(cohort, name, stats, target='wss', data_root=str(DATA),
                           required_frame_version='v5_atlas_frame_v1')
        case['unit_id'] = uid
        assert np.array_equal(case['y_raw'].astype(np.float64), truth)
        assert len(truth) == p['geometry'][uid]['n_wall']
        outputs = {}
        for source_name in VELOCITY_SOURCES:
            outputs[f'{source_name}_calibrated'] = results[source_name]['wss_mag_pa']
            outputs[f'{source_name}_physics'] = results[source_name]['physics_wss_mag_pa']
        valid = np.isfinite(truth)
        for values in outputs.values():
            valid &= np.isfinite(values) & (values >= 0)
        coverage[uid] = dict(n_wall=len(truth), n_valid=int(valid.sum()), coverage=float(valid.mean()),
                             case_scales={s: results[s]['case_scales'] for s in VELOCITY_SOURCES})
        if not valid.all():
            write_json(exp / f'invalid_coverage_{radius}.json', coverage)
            raise RuntimeError(f'Incomplete full-wall coverage: {radius} {uid}; no silent masking allowed')
        target = exp / 'predictions' / radius / f'{slug(uid)}.npz'
        payload = dict(canonical_id=np.asarray(uid), radius_source=np.asarray(radius), wall_mm=wall,
                       truth_wss_pa=truth, truth_wss_vec_pa=true_vec, valid_mask=valid)
        for source_name in VELOCITY_SOURCES:
            payload[f'{source_name}_calibrated_wss_pa'] = outputs[f'{source_name}_calibrated']
            payload[f'{source_name}_physics_wss_pa'] = outputs[f'{source_name}_physics']
            payload[f'{source_name}_calibrated_wss_vec_pa'] = results[source_name]['wss_vec_pa']
            payload[f'{source_name}_calibration_ratio'] = results[source_name]['high_tail_ratio']
        save_npz(target, **payload)
        artifacts[uid] = dict(path=str(target), sha256=sha(target))
        truths.append(truth)
        cases.append(case)
        for key in predictions:
            predictions[key].append(outputs[key])
        print(f'assembled {radius} {uid}: full-wall coverage {valid.sum()}/{len(truth)}', flush=True)
        del caches, merged, results
        gc.collect()
    metrics = {}
    for key, preds in predictions.items():
        metrics[key] = E._evaluate_space(cases, truths, preds, save_dir=None, make_plots=False,
                                         plot_space='physical target', include_area=False)
        metrics[key]['normalized'] = E._evaluate_space(
            cases, [D.normalize_wss(v, stats) for v in truths], [D.normalize_wss(v, stats) for v in preds],
            save_dir=None, make_plots=False, plot_space='normalized', include_area=False)
        print(radius, key, metrics[key]['field_casebalanced'], flush=True)
    verify_frozen(p)
    source_hashes_end = root_source_hashes()
    (exp / 'metrics').mkdir(parents=True, exist_ok=True)
    write_json(exp / 'metrics' / f'{radius}.json', clean_json(metrics))
    write_json(exp / 'metrics' / f'{radius}_coverage.json', clean_json(coverage))
    write_json(exp / 'metrics' / f'{radius}_prediction_manifest.json', artifacts)
    audit = read(VELWSS1 / 'radius_split_audit.json')
    audit_train = set(audit['calibrator_train_cases'])
    train_ids = {f'{c}/{n}' for c, n in D.load_split_cases(SPLIT, 'train')}
    test_ids = set(artifacts)
    assert audit_train == train_ids and len(train_ids) == 138
    assert not (audit_train & test_ids) and len(test_ids) == 34
    gate = dict(passed=True, experiment=EXPERIMENT, radius_source=radius, checkpoint='best', n_cases=len(cases),
                n_wall=sum(len(x) for x in truths), valid_coverage=1.0, velocity_reproduced=True, no_new_training=True,
                calibrator_train_test_overlap=0, calibrator_train_matches_current_train138=True,
                evaluation_sources_stable_during_assemble=(source_hashes_start == source_hashes_end),
                evaluation_source_sha256=source_hashes_end,
                provenance_sha256=sha(exp / 'provenance.json'), metrics_sha256=sha(exp / 'metrics' / f'{radius}.json'),
                prediction_manifest_sha256=sha(exp / 'metrics' / f'{radius}_prediction_manifest.json'),
                job_id=os.environ['SLURM_JOB_ID'])
    write_json(exp / 'metrics' / f'{radius}_evaluation_gate.json', gate)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('stage', choices=['export', 'prepare', 'worker', 'assemble'])
    parser.add_argument('--seed', choices=SEEDS)
    parser.add_argument('--radius', choices=RADIUS_SOURCES)
    parser.add_argument('--index', type=int)
    parser.add_argument('--experiment-dir', type=Path, default=EXP)
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--limit', type=int, default=None, help='export smoke: first N cases only')
    parser.add_argument('--allow-login-node', action='store_true',
                        help='export only: run outside Slurm when GPU gres is unavailable; recorded in the JSON')
    args = parser.parse_args()
    if not os.environ.get('SLURM_JOB_ID') and not (args.stage == 'export' and args.allow_login_node):
        raise RuntimeError('Run this workload through Slurm')
    if args.stage == 'export':
        export(args.seed, exp=args.experiment_dir, device=args.device, limit=args.limit)
    elif args.stage == 'prepare':
        prepare(exp=args.experiment_dir)
    elif args.stage == 'assemble':
        assemble(args.radius, exp=args.experiment_dir)
    else:
        index = args.index if args.index is not None else int(os.environ['SLURM_ARRAY_TASK_ID'])
        worker(args.radius, index, exp=args.experiment_dir)


if __name__ == '__main__':
    main()

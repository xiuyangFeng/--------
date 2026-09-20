from __future__ import annotations
import contextlib
from copy import deepcopy
import io
import json
import os
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace as NS
from unittest.mock import patch
import numpy as np

ROOT_REAL = Path('/public/newhome/cy/Digital_twin/GNN')
sys.path.insert(0, str(ROOT_REAL))
from training_wss_min.tools import diagnose_m2_optimization as M

BASE = Path(tempfile.mkdtemp(prefix='m2_collection_main_review_'))
BASE_METRIC = {'overall': {'r2': .5, 'mae': 2.}, 'hotspot': {'top10_iou': .3}, 'distribution': {'n': 4}}

def run_scenario(name, collect=True, failures=(), fatal_at=None):
    root = BASE / name
    exp, runs_root = root/'exp', root/'runs'
    exp.mkdir(parents=True)
    inputs = root/'inputs'
    inputs.mkdir()
    cases = []
    uids = list(M.REPRESENTATIVES) + [f'MOCK/case_{i:02d}' for i in range(31)]
    for i, uid in enumerate(uids):
        bundle = inputs/f'bundle_{i:02d}.npz'
        np.savez(bundle, wall_node_id_cas=np.arange(4))
        sidecar = inputs/'cohort'/f'case_{i:02d}'/'features.npz'
        sidecar.parent.mkdir(parents=True)
        np.savez(sidecar, placeholder=np.arange(4))
        cases.append(dict(unit_id=uid, bundle_path=str(bundle), cohort='cohort', case=f'case_{i:02d}',
             pos=np.arange(12).reshape(4,3).astype(float), wall_coords_raw=np.arange(12).reshape(4,3).astype(float),
             coord_scale_scalar=2., y_raw=np.arange(4).astype(float), y_norm=np.arange(4).astype(float),
             abscissa_norm=np.linspace(0,1,4), peak_step=1))
    cfg = NS(data=NS(target='wss', timesteps='peak', input_features=['a'], data_root=str(inputs),
             point_features_root=str(inputs), wss_stats_path='unused', split_path='unused',
             target_normalization='log_z', required_frame_version='mock'),
             model=NS(out_dim=1), eval=NS(fixed_support=True, full_cloud_query=True,
             support_seed=1234, query_chunk_size=16384, surface_metric_mode='legacy_vertex'))
    run_paths = {'M2': runs_root/'M2', **{f'MO{i}': runs_root/f'MO{i}' for i in range(20)}}
    saved = {'test': {'per_case': {uid: deepcopy(BASE_METRIC) for uid in uids},
                     'normalized': {'per_case': {uid: deepcopy(BASE_METRIC) for uid in uids}}}}
    for label, run in run_paths.items():
        run.mkdir(parents=True)
        for filename in ('config.json', 'feature_stats.json', 'wss_global_stats.json'):
            (run/filename).write_text('{}')
        for ck in ('best', 'last'):
            (run/f'ckpt_{ck}.pt').write_bytes(f'{label}/{ck}'.encode())
            d = run/'eval'/f'ckpt_{ck}'
            d.mkdir(parents=True)
            (d/'metrics.json').write_text(json.dumps(saved))
    entries = {label: {'run_name': str(run.relative_to(runs_root)),
               'evidence': {'valid_scientific_result': True}, 'physical_r2_cb': .6, 'mae': 2.}
               for label,run in run_paths.items() if label != 'M2'}
    reports = {ck: {'matrix_finalized': True, 'arms': deepcopy(entries)} for ck in ('best','last')}
    reports['selection'] = {'ranking': [], 'candidates': {}}
    counts = {'evaluate': 0, 'created': 0, 'closed': 0, 'resets': 0, 'mechanisms': 0, 'renders': 0, 'empty_cache': 0}
    expected = {}
    class Collector:
        def __init__(self, model):
            counts['created'] += 1
            self.model = model
        def reset(self): counts['resets'] += 1
        def close(self): counts['closed'] += 1
        def case_result(self, cfg, case, prediction_norm, prediction_pa, stats):
            counts['mechanisms'] += 1
            return {'mock': True}, {'mock_mechanism': np.array([1.])}
    def load_model(run, device, ck):
        return cfg, {}, NS(label=run.name, ck=ck), {'epoch': 400}
    def evaluate(model, selected, cfg, features, stats, device, return_predictions):
        counts['evaluate'] += 1
        index = counts['evaluate']
        if index == fatal_at:
            raise ValueError('injected non-metric fatal error')
        uid = selected[0]['unit_id']
        prediction = np.array([index, index+.25, index+.5, index+.75], dtype=float)
        expected[(model.label, model.ck, uid)] = prediction
        actual = deepcopy(BASE_METRIC)
        if index in failures:
            actual['overall']['r2'] = .25
        return {'_pred_norm_by_case': [prediction], 'per_case': {uid: actual},
                'normalized': {'per_case': {uid: deepcopy(BASE_METRIC)}}}
    def render(*args, **kwargs):
        counts['renders'] += 1
        return {'mock': True}
    def cache(): counts['empty_cache'] += 1
    patches = [
        patch.object(M, 'ROOT', root), patch.object(M, 'EXP', exp), patch.object(M, 'OUT', exp/'diagnostics'),
        patch.object(M, 'RUNS', runs_root), patch.object(M, 'ANCHOR_RUN', run_paths['M2']),
        patch.object(M, 'ANCHOR_CONFIG', run_paths['M2']/'config.json'),
        patch.object(M, '_verify_current_training_sources', return_value={'base': {'mock': 'frozen'}}),
        patch.object(M, 'fingerprints', return_value={'mock': 'frozen'}),
        patch.object(M.R, 'report_all', return_value=reports),
        patch.object(M.C.ExpConfig, 'from_json', return_value=cfg),
        patch.object(M.C, 'v6_point_features', return_value=[]),
        patch.object(M.D, 'load_wss_stats', return_value={}),
        patch.object(M.D, 'load_partition', return_value=cases),
        patch.object(M.D, 'denormalize_wss', side_effect=lambda p,stats: p.copy()),
        patch.object(M.E, 'load_model_from_run', side_effect=load_model),
        patch.object(M.E, 'load_wss_stats_for_run', return_value={}),
        patch.object(M.E, '_evaluate_partition_frame', side_effect=evaluate),
        patch.object(M, 'MechanismCollector', Collector),
        patch.object(M, 'parameter_diagnostics', return_value={'mock': True}),
        patch.object(M, 'render_figures', side_effect=render),
        patch.object(M.torch.cuda, 'is_available', return_value=True),
        patch.object(M.torch.cuda, 'empty_cache', side_effect=cache),
        patch.object(M.gc, 'collect', return_value=0),
        patch.dict(os.environ, {'SLURM_JOB_ID': 'CPU_MOCK_NO_GPU'}),
        patch.object(sys, 'argv', ['diagnose_m2_optimization.py'] + (['--collect-failures'] if collect else [])),
    ]
    exception = None
    log = io.StringIO()
    with contextlib.ExitStack() as stack:
        for p in patches: stack.enter_context(p)
        with contextlib.redirect_stdout(log):
            try: M.main()
            except Exception as exc: exception = exc
    (root/'stdout.log').write_text(log.getvalue())
    output = exp/('diagnostics_collection' if collect else 'diagnostics')
    provenance = json.loads((output/'provenance.json').read_text())
    items = [item for arm in provenance['runs'].values() for item in arm.values()]
    case_items = [(label, ck, uid, case) for label,arm in provenance['runs'].items()
                  for ck,item in arm.items() for uid,case in item['cases'].items()]
    assert counts['renders'] == 0, counts
    assert counts['created'] == counts['closed'] == counts['empty_cache'], counts
    assert not provenance['passed'], provenance
    assert not (output/'wall_figures').exists()
    if fatal_at:
        assert counts['evaluate'] == fatal_at, counts
        assert isinstance(exception, ValueError) and str(exception) == 'injected non-metric fatal error', repr(exception)
    elif collect:
        assert isinstance(exception, RuntimeError), repr(exception)
        assert counts['evaluate'] == 1428, counts
        assert len(items) == 42 and all(item['completed'] for item in items), (len(items), [x.get('completed') for x in items])
        assert len(case_items) == 1428, len(case_items)
        failed_items = [item for item in items if not item['passed']]
        failed_cases = [(label, ck, uid, case) for label,ck,uid,case in case_items
                        if not all(v['passed'] for v in case['metric_verification'].values())]
        assert len(failed_cases) == 3, len(failed_cases)
        assert len(failed_items) == 3, len(failed_items)
        assert provenance['strict_failed_numeric_checks'] == 3, provenance['strict_failed_numeric_checks']
        for label,ck,uid,case in case_items:
            with np.load(case['arrays']['path']) as payload:
                assert np.array_equal(payload['prediction_norm'], expected[(label,ck,uid)])
                assert np.array_equal(payload['prediction_pa'], expected[(label,ck,uid)])
        for label,ck,uid,case in failed_cases:
            assert case['passed'] is False
            checks = case['metric_verification']['Pa']
            assert checks['failed_checks'][0]['metric'] == 'overall.r2'
            assert checks['failed_checks'][0]['reference'] == .5
            assert checks['failed_checks'][0]['recomputed'] == .25
        assert '"passed": true' not in log.getvalue().lower()
        assert not (exp/'diagnostics').exists(), 'collection leaked into normal output'
    else:
        assert isinstance(exception, RuntimeError), repr(exception)
        assert counts['evaluate'] == 1 and len(case_items) == 1, counts
    result = {'scenario': name, 'counts': counts, 'exception': f'{type(exception).__name__}: {exception}'[:600],
              'passed': provenance['passed'], 'completed': provenance.get('completed'),
              'evaluations_completed': provenance.get('evaluations_completed'),
              'evaluations_verified': provenance.get('evaluations_verified'),
              'strict_failed_numeric_checks': provenance['strict_failed_numeric_checks'],
              'output': str(output)}
    (root/'test_result.json').write_text(json.dumps(result, indent=2))
    return result

if __name__ == '__main__':
    results = [run_scenario('full_collection_early_middle_late', failures=(1,715,1428)),
               run_scenario('normal_mode_fails_first', collect=False, failures=(1,)),
               run_scenario('nonmetric_exception_is_fatal', failures=(1,), fatal_at=2)]
    (BASE/'test_results.json').write_text(json.dumps(results, indent=2))
    print(json.dumps({'passed': True, 'evidence': str(BASE), 'scenarios': results}, indent=2))

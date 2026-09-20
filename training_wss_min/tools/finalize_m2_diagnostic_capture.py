"""Review saved full-wall GPU captures and render them on CPU, without inference.

The GPU capture and formal metrics stay immutable. A final result may pass only
when every original strict failure has a narrowly verified boundary explanation.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
import csv
import io
import json
import math
import os
from pathlib import Path
import shutil
import subprocess

import numpy as np
import torch

from training_wss_min import config as C, dataset as D, evaluate as E, metrics as M
from training_wss_min.tools import diagnose_m2_optimization as P
from training_wss_min.tools import probe_m2_boundary_batch as B
from training_wss_min.tools.m2_optimization_common import ANCHOR_CONFIG, EXP, ROOT, RUNS, fingerprints, save_json, sha, stamp

require = P.require
record = P.record
THRESHOLDS = {'distribution.pred_above_one_fraction': (1., 'gt'),
              'distribution.pred_below_zero_fraction': (0., 'lt')}


def load_record(item):
    path = Path(item['path'])
    require(path.is_file() and sha(path) == item['sha256'], f'Changed evidence artifact: {path}')
    return path


def load_arrays(item):
    with np.load(load_record(item), allow_pickle=False) as payload:
        return {key: payload[key].copy() for key in payload.files}


def slurm_exit(job_id):
    require(str(job_id).isdigit(), 'Numeric Slurm job ID required')
    raw = subprocess.check_output(['/public/slurm/bin/sacct', '-j', str(job_id), '-n', '-P',
                                   '--format=JobID,State,ExitCode'], text=True, timeout=30)
    rows = [line.split('|') for line in raw.splitlines()]
    found = [row for row in rows if row[0] == str(job_id)]
    require(len(found) == 1, f'Missing or ambiguous Slurm accounting for {job_id}')
    return {'state': found[0][1], 'exit_code': found[0][2]}


def verify_hashes(values):
    for path, digest in values.items():
        require(sha(Path(path)) == digest, f'Changed captured source/input: {path}')


def link_record(item, destination):
    source = load_record(item)
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        require(sha(destination) == item['sha256'], f'Refusing to replace a different final artifact: {destination}')
    else:
        try:
            os.link(source, destination)
        except OSError:
            shutil.copy2(source, destination)
    require(sha(destination) == item['sha256'], 'Copied capture array changed')
    return record(destination)


def case_metrics(case, truth, prediction):
    truth, prediction = np.asarray(truth, np.float64), np.asarray(prediction, np.float64)
    result = M.regional_metrics(case['pos'], case['local_radius'], truth, prediction)
    result['calibration'] = M.calibration_metrics(truth, prediction)
    result['distribution'] = M.distribution_metrics(truth, prediction)
    result['hotspot'] = M.hotspot_localization_metrics(truth, prediction, case['pos'])
    result['legacy_vertex_hotspot'] = dict(result['hotspot'])
    return result


def exact_numeric_metrics(actual, recorded):
    a, b = dict(P.V.leaves(actual)), dict(P.V.leaves(recorded))
    require(a.keys() == b.keys(), 'Captured/recomputed metric keys differ')
    require(all((a[key] == value) or (math.isnan(a[key]) and math.isnan(value))
                for key, value in b.items()), 'CPU recomputation does not exactly recover the saved capture metrics')


def count_from_fraction(fraction, n):
    require(math.isfinite(fraction), 'Nonfinite threshold fraction')
    count = round(fraction * n)
    require(0 <= count <= n and abs(fraction * n - count) <= 1e-8,
            'Threshold fraction is not an integer count divided by n')
    return count


def boundary_core(context, failure, n, captured, repeats, stats):
    """Pure CPU proof of one captured threshold-side difference, in its native space."""
    require(failure['metric'] in THRESHOLDS and context['space'] in {'Pa', 'log_z'}, 'Unregistered threshold metric/space')
    threshold, comparison = THRESHOLDS[failure['metric']]
    require(threshold != 0. or context['space'] == 'log_z', 'Pa predictions are clipped at zero; no below-zero exception')
    side = lambda values: values > threshold if comparison == 'gt' else values < threshold
    require(type(n) is int and n > 0 and 8 <= len(repeats) <= 32, 'Exact population and 8 to 32 saved repeats required')
    reference_count = count_from_fraction(failure['reference'], n)
    capture_count = count_from_fraction(failure['recomputed'], n)
    require(abs(capture_count - reference_count) == 1, 'Boundary review is capped at one vertex')
    pred_key = 'prediction_pa' if context['space'] == 'Pa' else 'prediction_norm'
    truth_key = 'truth_pa' if context['space'] == 'Pa' else 'truth_norm'
    for bundle in [captured, *repeats]:
        for key in ('prediction_norm', 'prediction_pa', 'truth_norm', 'truth_pa', 'wall_node_id_cas'):
            require(key in bundle and bundle[key].shape == (n,) and np.isfinite(bundle[key]).all(),
                    f'Wrong or nonfinite boundary population: {key}')
        for key in ('truth_norm', 'truth_pa', 'wall_node_id_cas'):
            require(np.array_equal(bundle[key], captured[key]), f'Boundary truth/vertex ordering differs: {key}')
        expected_pa = np.clip(D.denormalize_wss(bundle['prediction_norm'], stats), 0., None)
        require(np.array_equal(expected_pa, bundle['prediction_pa']), 'Pa predictions do not match the fixed norm back-transform')
        require(np.array_equal(bundle['prediction_norm'], bundle['prediction_norm'].astype(np.float32).astype(np.float64)),
                'Boundary predictions are not exact promoted float32 outputs')
    require(int(np.count_nonzero(side(captured[pred_key]))) == capture_count
            and float(np.mean(side(captured[pred_key]))) == failure['recomputed'], 'Capture raw fraction/count does not match its recorded failure')
    counts = [int(np.count_nonzero(side(bundle[pred_key]))) for bundle in repeats]
    require(set(counts) == {reference_count, capture_count}, 'Probe must reproduce exactly both original/captured counts')
    reference_index = counts.index(reference_count)
    reference = repeats[reference_index]
    reference_side = side(reference[pred_key])
    changed = np.flatnonzero(side(captured[pred_key]) != reference_side).tolist()
    require(len(changed) == 1, 'Captured threshold sides differ at more than one probe vertex')
    index = changed[0]
    for bundle, count in zip(repeats, counts):
        expected = [] if count == reference_count else [index]
        require(np.flatnonzero(side(bundle[pred_key]) != reference_side).tolist() == expected,
                'Probe changes additional threshold-side vertices')
    norm_threshold = threshold if context['space'] == 'log_z' else float(D.normalize_wss(np.array([threshold], dtype=np.float64), stats)[0])
    ulp = abs(float(np.spacing(np.float32(norm_threshold))))
    values = np.array([float(bundle['prediction_norm'][index]) for bundle in [captured, *repeats]])
    if threshold == 0.:
        require(values[1:].min() < 0. <= values[1:].max(), 'Probe does not demonstrate both strict-zero threshold sides')
        require(values[1:].min() <= values[0] <= values[1:].max(), 'Captured zero-boundary value lies outside the observed probe envelope')
        rule = 'unique threshold-side vertex; both counts replayed; current normalized value inside the observed saved-repeat envelope (no ULP tolerance at zero)'
    else:
        require(values[1:].min() <= values[0] <= values[1:].max(),
                'Captured boundary value lies outside the observed probe envelope')
        rule = 'unique threshold-side vertex; both counts replayed; current normalized value inside the observed saved-repeat envelope; fixed analytic back-transform verified (ULP is descriptive, not a tolerance)'
    return {'review_status': 'accepted_discrete_boundary', 'metric': failure['metric'], 'context': context, 'n': n,
            'reference_count': reference_count, 'recomputed_count': capture_count, 'count_delta': capture_count-reference_count,
            'reference_fraction': failure['reference'], 'recomputed_fraction': failure['recomputed'], 'one_vertex_fraction': 1./n,
            'vertex_index': index, 'vertex_node_id': int(captured['wall_node_id_cas'][index]),
            'current_vertex_value': float(captured[pred_key][index]), 'reference_vertex_value': float(reference[pred_key][index]),
            'metric_threshold': threshold, 'comparison': comparison, 'normalized_threshold': norm_threshold,
            'current_vertex_normalized_value': float(captured['prediction_norm'][index]),
            'reference_vertex_normalized_value': float(reference['prediction_norm'][index]), 'float32_ulp': ulp if threshold else None,
            'ulp_limit_applied': False,
            'boundary_evidence_rule': rule, 'probe_vertex_normalized_min': float(values[1:].min()),
            'probe_vertex_normalized_max': float(values[1:].max()),
            'probe_vertex_normalized_values': values[1:].tolist(), 'probe_counts': counts,
            'probe_repeat_count': len(repeats),
            'max_normalized_distance_to_threshold': float(np.max(np.abs(values - norm_threshold))),
            'crossing_indices_vs_probe_reference': changed, 'probe_reference_index': reference_index,
            'max_abs_vs_probe_reference': float(np.max(np.abs(captured[pred_key] - reference[pred_key]))),
            'finite_mask_matches_probe': True, 'truth_matches_probe': True,
            'limit': 'Post-hoc review of this immutable GPU capture and saved repeated forwards; formal evaluation per-vertex predictions were not saved. Continuous tolerances, original metrics and selection stay unchanged.'}


def load_probe(path, capture_record):
    evidence_record = record(path)
    evidence = json.loads(load_record(evidence_record).read_text())
    require(evidence.get('completed') and not evidence.get('metric_accepted')
            and evidence.get('status') == 'independent_review_required', 'Batch probe did not complete as evidence only')
    require(evidence['collector'] == capture_record, 'Batch probe belongs to another primary GPU capture')
    require(slurm_exit(evidence['job_id']) == {'state': 'COMPLETED', 'exit_code': '0:0'}, 'Batch probe Slurm exit failed')
    load_record(evidence['script']); load_record(evidence['metric_verification_script'])
    verify_hashes(evidence['inputs'])
    require(evidence['source_sha256_before'] == evidence['source_sha256_after']
            == {phase: fingerprints(phase) for phase in evidence['source_sha256_before']}, 'Probe frozen sources changed')
    all_array_paths = []
    target_ids = []
    for group in evidence['groups'].values():
        require(group['completed'] and 8 <= len(group['repeats']) <= 32, 'Incomplete batch-probe group')
        require(len(group['repeats']) == evidence['repeats_per_case'], 'Probe repeat total differs from its registered command')
        require([repeat['index'] for repeat in group['repeats']] == list(range(len(group['repeats']))),
                'Batch-probe repeat indices are missing, duplicated or reordered')
        verify_hashes(group['sources'])
        load_record(group['geometry_arrays'])
        for repeat in group['repeats']:
            load_record(repeat['arrays'])
            all_array_paths.append(str(Path(repeat['arrays']['path']).resolve()))
        for target in group['targets']:
            target_ids.append(target['id'])
            load_record(target['capture_origin']['provenance'])
            load_record(target['capture_origin']['arrays'])
            load_record(target['geometry_arrays'])
    require(len(all_array_paths) == len(set(all_array_paths)), 'One probe array path was counted as multiple repeats')
    require(len(target_ids) == len(set(target_ids)) == evidence['target_count']
            and len(all_array_paths) == evidence['inference_count'], 'Batch-probe total counts are inconsistent')
    require(set(target_ids) == {target['id'] for target in evidence['selection']['selected_targets']},
            'Batch-probe selected targets were not all captured exactly once')
    return evidence, evidence_record


def find_probe_target(evidence, context, metric, capture_record, array_record):
    matches = [(group, target) for group in evidence['groups'].values() for target in group['targets']
               if target['context'] == context and target['metric'] == metric and target['capture_origin']['provenance'] == capture_record
               and target['capture_origin']['arrays'] == array_record]
    require(len(matches) == 1, 'Current capture failure has no unique matching batch-probe target')
    return matches[0]


def reverify_probe_groups(probe, by_case, capture_record):
    """Check every saved group, including historical evidence that grants no current exception."""
    ledger, inputs = [], dict(probe['inputs'])
    for item in (probe['script'], probe['metric_verification_script']):
        inputs[item['path']] = item['sha256']
    for group in probe['groups'].values():
        uid, case = group['case'], by_case[group['case']]
        run = RUNS / group['run_name']
        stats = E.load_wss_stats_for_run(run)
        require(stats == group['wss_stats'], 'Batch-probe normalization stats changed')
        geometry = load_arrays(group['geometry_arrays'])
        require(np.array_equal(geometry['truth_norm'], case['y_norm']) and np.array_equal(geometry['truth_pa'], case['y_raw']),
                'Batch-probe original truth/order differs')
        saved = json.loads((run / f"eval/ckpt_{group['checkpoint']}/metrics.json").read_text())['test']
        repeats = []
        for repeat in group['repeats']:
            bundle = load_arrays(repeat['arrays']); repeats.append(bundle)
            inputs[repeat['arrays']['path']] = repeat['arrays']['sha256']
            for key in ('truth_norm', 'truth_pa', 'wall_node_id_cas'):
                require(np.array_equal(bundle[key], geometry[key]), 'Batch-probe raw truth/IDs differ')
            require(np.array_equal(np.clip(D.denormalize_wss(bundle['prediction_norm'], stats), 0., None), bundle['prediction_pa']),
                    'Batch-probe raw Pa transformation differs')
            for space, pkey, tkey in [('Pa', 'prediction_pa', 'truth_pa'), ('log_z', 'prediction_norm', 'truth_norm')]:
                actual = case_metrics(case, bundle[tkey], bundle[pkey])
                expected = saved['per_case'][uid] if space == 'Pa' else saved['normalized']['per_case'][uid]
                require(P.V.verify_metrics(actual, expected) == repeat['metric_verification'][space],
                        'Batch-probe metric record does not match raw arrays')
        for target in group['targets']:
            context, origin = target['context'], target['capture_origin']
            require(context['arm'] == group['arm'] and context['checkpoint'] == group['checkpoint'] and context['case'] == uid,
                    'Batch-probe target/group identity differs')
            origin_payload = json.loads(load_record(origin['provenance']).read_text())
            source = origin_payload['runs'][context['arm']][context['checkpoint']]
            origin_case = source['cases'][uid]
            require(str(origin_payload['job_id']) == str(origin['job_id']) and source['sources'] == target['sources'] == group['sources']
                    and origin_case['arrays']['sha256'] == origin['arrays']['sha256']
                    and origin_payload['geometry'][uid]['arrays']['sha256'] == target['geometry_arrays']['sha256']
                    and target['original_failure'] in origin_case['metric_verification'][context['space']]['failed_checks'],
                    'Batch-probe target is not an original captured failure')
            raw = load_arrays(origin['arrays'])
            target_geometry = load_arrays(target['geometry_arrays'])
            require(all(np.array_equal(target_geometry[key], geometry[key])
                        for key in ('truth_norm', 'truth_pa', 'wall_node_id_cas')),
                    'Target capture geometry differs from its shared probe group')
            captured = {**raw, **{key: geometry[key] for key in ('truth_norm', 'truth_pa', 'wall_node_id_cas')}}
            for repeat, bundle in zip(group['repeats'], repeats):
                measurement = B.measure(bundle['prediction_norm'], stats, context['space'], bundle['wall_node_id_cas'],
                                        raw['prediction_norm'], target['metric'])
                require(measurement == repeat['measurements'][target['id']], 'Batch-probe threshold measurement metadata differs')
            summary = B.summarize_target(target, [bundle['prediction_norm'] for bundle in repeats], raw['prediction_norm'], stats)
            require(summary == group['summaries'][target['id']], 'Batch-probe summary metadata differs')
            proof, reason = None, None
            try:
                proof = boundary_core(context, target['original_failure'], target['n'], captured, repeats, stats)
            except RuntimeError as exc:
                reason = str(exc)
            ledger.append({'target_id': target['id'], 'context': context, 'metric': target['metric'],
                'capture_origin': origin, 'primary_current_capture': origin['provenance'] == capture_record,
                'raw_records_verified': True, 'boundary_evidence_sufficient': proof is not None,
                'boundary_evidence': proof, 'insufficient_reason': reason,
                'grants_current_metric_exception': False})
    return ledger, inputs


def review_from_probe(context, failure, n, capture_arrays, geometry, array_record, capture_record,
                      sources, case, stats, saved_metrics, probe, probe_record):
    group, target = find_probe_target(probe, context, failure['metric'], capture_record, array_record)
    require(target['original_failure'] == failure and target['sources'] == sources == group['sources']
            and target['n'] == n and group['wss_stats'] == stats, 'Probe target failure/source/population changed')
    captured = {**capture_arrays, **{key: geometry[key] for key in ('truth_norm', 'truth_pa', 'wall_node_id_cas')}}
    repeats = [load_arrays(repeat['arrays']) for repeat in group['repeats']]
    # Re-evaluate every repeated raw vector on CPU. Only the target fraction may
    # differ from formal metrics; a continuous-metric failure blocks this review.
    for bundle, repeat in zip(repeats, group['repeats']):
        for space, pred_key, truth_key in [('Pa', 'prediction_pa', 'truth_pa'), ('log_z', 'prediction_norm', 'truth_norm')]:
            actual = case_metrics(case, bundle[truth_key], bundle[pred_key])
            expected = saved_metrics['per_case'][context['case']] if space == 'Pa' else saved_metrics['normalized']['per_case'][context['case']]
            check = P.V.verify_metrics(actual, expected)
            require(check == repeat['metric_verification'][space], 'Repeated-probe metric metadata does not match its saved raw arrays')
            require(all(row['passed'] or (space == context['space'] and row['metric'] == failure['metric'])
                        for row in check['checks']), 'Probe has an unrelated continuous/other-space metric failure')
    reviewed = boundary_core(context, failure, n, captured, repeats, stats)
    index = reviewed['probe_reference_index']
    run = RUNS / group['run_name']
    evidence_inputs = {probe_record['path']: probe_record['sha256'],
                       probe['script']['path']: probe['script']['sha256'],
                       probe['metric_verification_script']['path']: probe['metric_verification_script']['sha256'],
                       **{r['arrays']['path']: r['arrays']['sha256'] for r in group['repeats']}}
    reviewed.update(probe=probe_record, probe_reference_array=group['repeats'][index]['arrays'],
        capture_origin={'job_id': target['capture_origin']['job_id'], 'provenance': capture_record, 'arrays': array_record},
        probe_target_id=target['id'], probe_source_script_sha256=probe['script']['sha256'], evidence_inputs=evidence_inputs,
        source_metrics_sha256=sources[str(run / f"eval/ckpt_{context['checkpoint']}/metrics.json")],
        checkpoint_sha256=sources[str(run / f"ckpt_{context['checkpoint']}.pt")], config_sha256=sources[str(run / 'config.json')])
    return reviewed


def validate_capture(capture, capture_record):
    require(capture.get('collect_failures') and capture.get('completed') and capture.get('full_inference_coverage_complete')
            and not capture.get('smoke') and not capture.get('figures_rendered'), 'A complete no-figure collection is required')
    require(capture['expected_evaluations'] == capture['evaluations_completed'] == 42
            and capture['expected_case_evaluations'] == capture['cases_evaluated'] == 1428
            and capture['case_count'] == 34 and len(capture['geometry']) == 34, 'Full 42 x 34 capture coverage required')
    require(len(capture['runs']) == 21 and all(set(run) == {'best', 'last'} for run in capture['runs'].values()), 'Capture run/checkpoint identities differ')
    for run in capture['runs'].values():
        for item in run.values():
            require(item['completed'] and set(item['cases']) == set(capture['geometry']), 'Incomplete or mixed case identities')
            verify_hashes(item['sources'])
    summary = P.metric_summary(capture)
    require(all(capture[key] == value for key, value in summary.items()), 'Capture verification summary is inconsistent')
    require(capture['passed'] == (capture['unreviewed_metric_failure_count'] == 0), 'Capture passed status does not match unresolved failures')
    expected_exit = {'state': 'COMPLETED', 'exit_code': '0:0'} if capture['passed'] else {'state': 'FAILED', 'exit_code': '1:0'}
    observed_exit = slurm_exit(capture['job_id'])
    require(observed_exit == expected_exit, 'GPU capture Slurm exit does not match saved status')
    if not capture['passed']:
        require(capture.get('error', '').startswith('RuntimeError: Complete inference captured;'), 'Capture failure was not the final unresolved-metric gate')
    require(capture['source_sha256_before'] == capture['source_sha256_after'] == P._verify_current_training_sources(), 'Captured training sources changed')
    load_record(capture['script']); load_record(capture['report_code']); verify_hashes(capture['inputs'])
    load_record(capture_record)
    return observed_exit


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--capture', type=Path, default=EXP / 'diagnostics_collection/provenance.json')
    parser.add_argument('--probe', type=Path)
    parser.add_argument('--output', type=Path, default=EXP / 'diagnostics')
    parser.add_argument('--self-test', action='store_true')
    args = parser.parse_args()
    if args.self_test:
        self_test()
        return
    require(args.probe is not None, '--probe is required for the complete historical/current boundary evidence ledger')
    require(os.environ.get('SLURM_JOB_ID', '').isdigit(), 'CPU finalization requires an accountable Slurm job')
    torch.set_num_threads(2)
    capture_record = record(args.capture)
    capture = json.loads(args.capture.read_text())
    capture_exit = validate_capture(capture, capture_record)
    probe, probe_record = load_probe(args.probe, capture_record)
    output = args.output.resolve()
    require(output != args.capture.parent.resolve(), 'Final output must be separate from the immutable capture')
    output.mkdir(parents=True, exist_ok=True)
    require(not (output / 'provenance.json').exists(), 'Final output provenance already exists; preserve it before retrying')
    provenance = deepcopy(capture)
    provenance.pop('error', None); provenance.pop('failed_at', None)
    provenance.update(passed=False, completed=False, job_id=os.environ['SLURM_JOB_ID'], script=record(__file__),
        started_at=stamp(), collect_failures=False, figures_rendered=False,
        capture={'job_id': str(capture['job_id']), 'provenance': capture_record, 'slurm': capture_exit,
                 'passed': capture['passed'], 'strict_passed': capture['strict_passed'], 'script': capture['script'],
                 'full_inference_coverage_complete': True},
        finalization={'mode': 'cpu_review_of_saved_capture', 'completed': False, 'source': record(__file__),
                      'started_at': stamp(), 'batch_probe': probe_record, 'probe_job_id': str(probe['job_id'])})
    provenance['inputs'].update({capture_record['path']: capture_record['sha256'], probe_record['path']: probe_record['sha256']})
    save_json(output / 'provenance.json', provenance)
    try:
        anchor = C.ExpConfig.from_json(ANCHOR_CONFIG)
        stats = D.load_wss_stats(anchor.data.wss_stats_path)
        cases = D.load_partition(anchor.data.split_path, 'test', stats, strict=True,
            target='wss', target_normalization=anchor.data.target_normalization, data_root=anchor.data.data_root,
            required_frame_version=anchor.data.required_frame_version, extra_point_features=C.v6_point_features(anchor),
            point_features_root=anchor.data.point_features_root)
        by_case = {case['unit_id']: case for case in cases}
        require(set(by_case) == set(capture['geometry']), 'CPU original case identities differ from capture')
        probe_ledger, probe_inputs = reverify_probe_groups(probe, by_case, capture_record)
        provenance['batch_probe_reverification'] = probe_ledger
        provenance['inputs'].update(probe_inputs)
        geometries = {}
        for uid, entry in capture['geometry'].items():
            geometry = load_arrays(entry['arrays']); geometries[uid] = geometry
            case = by_case[uid]
            require(np.array_equal(geometry['truth_norm'], case['y_norm']) and np.array_equal(geometry['truth_pa'], case['y_raw'])
                    and np.array_equal(geometry['pos_norm'], case['pos']), 'Original geometry/truth differs from captured vertex ordering')
            provenance['geometry'][uid]['arrays'] = link_record(entry['arrays'], output / 'geometry' / f'{P.slug(uid)}.npz')
        raw_checks = 0
        new_reviews = []
        for arm, run in capture['runs'].items():
            for checkpoint_name, entry in run.items():
                destination = provenance['runs'][arm][checkpoint_name]
                run_path = RUNS / entry['run_name']
                source_files = {'metrics': run_path / f'eval/ckpt_{checkpoint_name}/metrics.json',
                                'checkpoint': run_path / f'ckpt_{checkpoint_name}.pt', 'config': run_path / 'config.json'}
                source_hashes = {key: entry['sources'][str(path)] for key, path in source_files.items()}
                saved_metrics = json.loads(source_files['metrics'].read_text())['test']
                run_stats = E.load_wss_stats_for_run(run_path)
                for uid, recorded_case in entry['cases'].items():
                    case, geometry = by_case[uid], geometries[uid]
                    arrays = load_arrays(recorded_case['arrays'])
                    require(np.array_equal(np.clip(D.denormalize_wss(arrays['prediction_norm'], run_stats), 0., None), arrays['prediction_pa']),
                            'Captured Pa output is not the fixed back-transform')
                    checks = {}
                    for space, pred_key, truth_key in [('Pa', 'prediction_pa', 'truth_pa'), ('log_z', 'prediction_norm', 'truth_norm')]:
                        require(arrays[pred_key].shape == geometry[truth_key].shape == (recorded_case['wall_points'],)
                                and np.isfinite(arrays[pred_key]).all() and np.isfinite(geometry[truth_key]).all(), 'Captured finite vertex population differs')
                        recomputed = case_metrics(case, geometry[truth_key], arrays[pred_key])
                        exact_numeric_metrics(recomputed, recorded_case['recomputed_metrics'][space])
                        expected = saved_metrics['per_case'][uid] if space == 'Pa' else saved_metrics['normalized']['per_case'][uid]
                        checks[space] = P.metric_check(recomputed, expected, prediction=arrays['prediction_norm'], truth=geometry['truth_norm'],
                            context=(arm, checkpoint_name, uid, space), source_hashes=source_hashes,
                            other_space_strict_passed=checks.get('Pa', {}).get('strict_passed', False) if space == 'log_z' else False)
                        require(checks[space] == recorded_case['metric_verification'][space], 'CPU verification does not recover the original capture check and review')
                        raw_checks += checks[space]['checks']
                    pending = [(space, row) for space, check in checks.items() if not check['passed'] for row in check['failed_checks']]
                    if pending:
                        require(len(pending) == 1 and pending[0][1]['metric'] in THRESHOLDS,
                                'A case with multiple or continuous failures cannot receive a threshold-only review')
                        space, failure = pending[0]
                        context = {'arm': arm, 'checkpoint': checkpoint_name, 'case': uid, 'space': space}
                        review = review_from_probe(context, failure, recorded_case['wall_points'], arrays, geometry,
                            recorded_case['arrays'], capture_record, entry['sources'], case, run_stats, saved_metrics, probe, probe_record)
                        checks[space]['boundary_reviews'].append(review)
                        checks[space]['passed'] = True
                        provenance['inputs'].update(review['evidence_inputs'])
                        new_reviews.append(review)
                    target_case = destination['cases'][uid]
                    target_case['metric_verification'] = checks
                    target_case['passed'] = all(check['passed'] for check in checks.values())
                    target_case['arrays'] = link_record(recorded_case['arrays'], output / 'predictions' / arm / checkpoint_name / f'{P.slug(uid)}.npz')
                destination['passed'] = all(case['passed'] for case in destination['cases'].values())
                print(f'{stamp()} CPU raw metrics verified {arm}/{checkpoint_name}: 34 cases', flush=True)
        provenance.update(P.metric_summary(provenance))
        require(provenance['unreviewed_metric_failure_count'] == 0 and provenance['cases_verified'] == 1428
                and provenance['evaluations_verified'] == 42, 'Unreviewed capture discrepancies remain')
        provenance['finalization'].update(raw_metrics_recomputed=raw_checks, new_boundary_review_count=len(new_reviews),
            inherited_boundary_review_count=capture['boundary_review_count'])
        provenance['figures'] = P.render_figures(output, by_case, capture['visual_candidate'], provenance['runs'])
        provenance['figures_rendered'] = True
        verify_hashes(provenance['inputs'])
        for run in capture['runs'].values():
            for entry in run.values():
                verify_hashes(entry['sources'])
        require(capture['source_sha256_after'] == P._verify_current_training_sources(), 'Training sources changed during CPU finalization')
        for entry in provenance['geometry'].values():
            load_record(entry['arrays'])
        for run in provenance['runs'].values():
            for entry in run.values():
                for case in entry['cases'].values():
                    load_record(case['arrays'])
        load_record(capture_record); load_record(probe_record); load_record(provenance['script'])
        provenance['finalization'].update(completed=True, completed_at=stamp())
        provenance.update(passed=True, completed=True, completed_at=stamp(), source_sha256_after=capture['source_sha256_after'])
        lines = ['# M2 完整捕获的 CPU 审查与原壁面图', '', P.LIMITS, '',
            f"GPU捕获作业{capture['job_id']}：{capture_exit['state']} / {capture_exit['exit_code']}，完成42×34次推理；捕获strict_passed={capture['strict_passed']}。",
            f"CPU finalizer作业{provenance['job_id']}重算保存数组的{raw_checks:,}项逐病例指标并绘图；没有再次推理。",
            f"原容差严格通过{provenance['strict_passed_numeric_checks']:,}项，原严格失败{provenance['strict_failed_numeric_checks']}项，"
            f"后验边界审核{provenance['boundary_review_count']}项，未审查失败{provenance['unreviewed_metric_failure_count']}项。", '',
            '原容差核验不代表逐位相等；strict_passed=false和所有原失败条目仍保留。每项后验审核绑定特定捕获、病例、空间、字段、逐点probe与来源SHA；原指标、保护线和工作簿数值未改变。', '',
            '[best原壁面对照](wall_figures/wall_overview_best.png) · [last原壁面对照](wall_figures/wall_overview_last.png)', '',
            'geometry和predictions中的文件是同一次GPU捕获的原数组副本/硬链接。源diagnostics_collection及失败历史记录保持不变。',
            f"图中候选为{capture['visual_candidate']['id']}（{capture['visual_candidate']['label']}），不替换历史M2或改变筛选。", '']
        (output / 'README.md').write_text('\n'.join(lines))
        print(json.dumps({'passed': True, 'job_id': provenance['job_id'], 'capture_job_id': capture['job_id'],
                          'boundary_review_count': provenance['boundary_review_count'], 'output': str(output)}), flush=True)
    except Exception as exc:
        provenance.update(error=f'{type(exc).__name__}: {exc}', failed_at=stamp(), passed=False)
        raise
    finally:
        provenance.update(P.metric_summary(provenance))
        save_json(output / 'provenance.json', provenance)


def self_test():
    stats = {'method': 'log_z', 'eps': 1e-6, 'log': {'mean': .6408381995319768, 'std': 1.2999484463172573}}
    def bundle(values):
        norm = np.asarray(values, np.float32).astype(np.float64)
        truth_norm = np.arange(len(norm), dtype=np.float64)
        return {'prediction_norm': norm, 'prediction_pa': np.clip(D.denormalize_wss(norm, stats), 0., None),
                'truth_norm': truth_norm, 'truth_pa': D.denormalize_wss(truth_norm, stats), 'wall_node_id_cas': np.arange(len(norm))}
    context = {'arm': 'fixture', 'checkpoint': 'last', 'case': 'fixture', 'space': 'log_z'}
    above = {'metric': 'distribution.pred_above_one_fraction', 'reference': 1/3, 'recomputed': 2/3}
    lower = bundle([1., 0., 2.]); upper = bundle([np.nextafter(np.float32(1.), np.float32(np.inf)), 0., 2.])
    positive = boundary_core(context, above, 3, upper, [lower, upper]*4, stats)
    require(positive['vertex_index'] == 0 and positive['count_delta'] == 1, 'Above-one CPU boundary proof failed')
    below = {'metric': 'distribution.pred_below_zero_fraction', 'reference': 2/3, 'recomputed': 1/3}
    negative = bundle([-1e-8, 1., -1.]); nonnegative = bundle([1e-8, 1., -1.])
    zero = boundary_core(context, below, 3, nonnegative, [negative, nonnegative]*16, stats)
    require(zero['comparison'] == 'lt' and zero['float32_ulp'] is None and zero['probe_repeat_count'] == 32,
            'Zero-boundary empirical-envelope CPU proof failed')
    pa_threshold = float(D.normalize_wss(np.array([1.]), stats)[0])
    pa_high = np.float32(pa_threshold)
    pa_low = np.nextafter(pa_high, np.float32(-np.inf))
    pa_ref, pa_cap = bundle([pa_low, 0., -1.]), bundle([pa_high, 0., -1.])
    physical = boundary_core({**context, 'space': 'Pa'}, above, 3, pa_cap, [pa_ref, pa_cap]*4, stats)
    require(physical['vertex_index'] == 0 and physical['current_vertex_value'] > 1. > physical['reference_vertex_value'],
            'Pa/norm CPU threshold mapping failed')
    rejected = 0
    def reject(failure=below, current=nonnegative, repeats=None, n=3, ctx=context):
        nonlocal rejected
        try:
            boundary_core(ctx, failure, n, current, repeats if repeats is not None else [negative, nonnegative]*4, stats)
        except (RuntimeError, ValueError):
            rejected += 1
        else:
            raise RuntimeError('Invalid CPU boundary fixture was accepted')
    reject(repeats=[nonnegative]*8)
    reject(current=bundle([2e-8, 1., -1.]))
    reject(repeats=[negative, nonnegative]*3)
    reject(repeats=[negative, nonnegative]*17)
    reject(failure={**below, 'reference': 1.})
    reject(failure={**below, 'reference': .5})
    reject(failure={**below, 'metric': 'overall.r2'})
    reject(n=4)
    changed = deepcopy(negative); changed['truth_norm'][0] = 5
    reject(repeats=[changed, nonnegative]*4)
    changed = deepcopy(negative); changed['wall_node_id_cas'][0] = 10
    reject(repeats=[changed, nonnegative]*4)
    changed = deepcopy(negative); changed['prediction_norm'][0] = np.nan
    reject(repeats=[changed, nonnegative]*4)
    changed = deepcopy(negative); changed['prediction_pa'][0] += .01
    reject(repeats=[changed, nonnegative]*4)
    wrong_side = bundle([1e-8, -1., 1.])
    reject(repeats=[negative, wrong_side]*4)
    too_far = bundle([np.nextafter(np.nextafter(np.float32(1.), np.float32(np.inf)), np.float32(np.inf)), 0., 2.])
    reject(failure=above, current=too_far, repeats=[lower, upper]*4)
    print(f'CPU finalizer boundary tests passed: normalized >1, physical >1, zero-envelope <0, 8/32 repeats and {rejected} rejection cases')


if __name__ == '__main__':
    main()

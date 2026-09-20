"""User-confirmed anatomy corrections for review candidates; preserve CFD names."""
from __future__ import annotations
from copy import deepcopy
import hashlib
from pathlib import Path

VERSION = 'anatomy_review_corrections_20260911'
OVERRIDES = {'AAA/ruputer/WANG_KUI_WU': {3: 'out-ri', 5: 'out-li'}}
ANATOMY = {'out-le': 'left_external', 'out-li': 'left_internal',
           'out-re': 'right_external', 'out-ri': 'right_internal'}

def correct_topology(canonical_id, segments, topology):
    mapping = OVERRIDES.get(canonical_id)
    if not mapping:
        return segments, topology
    segments, topology = deepcopy(segments), deepcopy(topology)
    by = {int(s['segment_id']): s for s in segments}
    if {i: by[i]['parent_id'] for i in by} != {0:-1, 1:0, 2:0, 3:2, 4:2, 5:1, 6:1}:
        raise ValueError('Confirmed WANG anatomy override does not match expected segment tree')
    for sid, expected in {3:'out-li', 4:'out-re', 5:'out-ri', 6:'out-le'}.items():
        if by[sid]['cfd_zone_label'] != expected:
            raise ValueError(f'Confirmed WANG override: unexpected original CFD correspondence S{sid}')
    def labels(sid):
        row = by[sid]
        return sum((labels(c) for c in row['children']), []) if row['children'] else [mapping.get(sid, row['cfd_zone_label'])]
    mixed = []
    for sid, row in by.items():
        row['anatomy_descendant_outlet_labels'] = labels(sid)
        if row['role'] == 'trunk':
            continue
        row.setdefault('semantic_reason_before_anatomy_correction', row['semantic_reason'])
        row.setdefault('anatomy_label_before_user_override', row['anatomy_label'])
        if not row['children']:
            name = mapping.get(sid, row['cfd_zone_label'])
            row['anatomy_outlet_label'] = name
            row['anatomy_label'] = ANATOMY[name]
            row['anatomy_label_user_confirmed'] = sid in mapping
            row['label_provenance'] = 'user_confirmed_internal_side_swap_20260911' if sid in mapping else 'retained_external_CFD_correspondence'
            if sid in mapping:
                row['anatomy_label_override'] = name
        else:
            sides = {ANATOMY[name].split('_')[0] for name in labels(sid)}
            row['anatomy_label'] = next(iter(sides)) + '_cia' if len(sides) == 1 else 'unknown'
            row['anatomy_label_user_confirmed'] = False
            row['label_provenance'] = 'derived_from_corrected_terminal_anatomy_and_tree'
            if len(sides) != 1:
                mixed.append(sid)
        row['semantic_valid'] = row['anatomy_label'] != 'unknown'
        row['anatomy_semantic_valid'] = row['semantic_valid']
        row['semantic_reason'] = row['label_provenance']
    topology.setdefault('original_CFD_mixed_side_cia_segments', topology.get('mixed_side_cia_segments', []))
    topology['mixed_side_cia_segments'] = mixed
    topology['anatomy_names_user_confirmed'] = False
    topology['user_confirmed_terminal_segments'] = sorted(mapping)
    return segments, topology

def correct_openings(canonical_id, openings, segments):
    if canonical_id not in OVERRIDES:
        return openings
    openings = deepcopy(openings)
    by = {int(s['segment_id']): s for s in segments}
    for row in openings:
        segment = by[int(row['segment_id'])]
        if segment['role'] == 'trunk':
            continue
        row['anatomy_outlet_label'] = segment['anatomy_outlet_label']
        row['anatomy_label'] = segment['anatomy_label']
        row['anatomy_label_user_confirmed'] = segment['anatomy_label_user_confirmed']
        row['label_role'] = 'original_CFD_correspondence_preserved; anatomy_outlet_label_used_for_review'
    return openings

def correct_report(report):
    cid = report['canonical_id']
    if cid not in OVERRIDES:
        return report
    report = deepcopy(report)
    report['segments'], report['P2_topology'] = correct_topology(cid, report['segments'], report['P2_topology'])
    report['P3_CIA'] = [s for s in report['segments'] if s['role'] == 'cia']
    report['openings'] = correct_openings(cid, report['openings'], report['segments'])
    report['warnings'] = [w for w in report['warnings'] if w not in ('mixed_CFD_sides_anatomy_unknown', 'label_corrected_user_confirmed')]
    report['status'] = 'candidate_needs_review' if report['warnings'] else 'candidate_ready_for_visual_review'
    report['anatomy_correction'] = {'version': VERSION, 'terminal_outlet_overrides': OVERRIDES[cid],
        'user_confirmed_segments': sorted(OVERRIDES[cid]), 'original_CFD_names_preserved': True,
        'centerline_coordinates_changed': False, 'scope': 'anatomy_names_only_not_whole_geometry_approval',
        'code_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    return report

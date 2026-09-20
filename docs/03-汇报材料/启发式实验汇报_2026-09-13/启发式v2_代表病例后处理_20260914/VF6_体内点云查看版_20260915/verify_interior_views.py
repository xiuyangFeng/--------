#!/usr/bin/env python3
"""Check each VF6 VTP keeps only points farther than 1 mm from the wall."""
from pathlib import Path
import json
import numpy as np
import vtk
from vtk.util.numpy_support import vtk_to_numpy

OUT = Path(__file__).resolve().parent
PARENT = OUT.parent
ROOT = next(p for p in OUT.parents if (p / 'training_wss_min').is_dir())
PEEL_MM = 1.0
ROLES = ('best', 'median', 'worst')


def read_vtp(path):
    r = vtk.vtkXMLPolyDataReader(); r.SetFileName(str(path)); r.Update(); p = r.GetOutput()
    data = p.GetPointData()
    return p, vtk_to_numpy(p.GetPoints().GetData()).copy(), {
        data.GetArrayName(i): vtk_to_numpy(data.GetArray(i)).copy()
        for i in range(data.GetNumberOfArrays())
    }


def main():
    checks = []
    def check(label, passed, detail=None):
        rec = {'check': label, 'passed': bool(passed)}
        if detail is not None:
            rec['detail'] = detail
        checks.append(rec)
        if not passed:
            print('FAILED', label, detail)

    for role in ROLES:
        m = json.loads((OUT / role / 'manifest.json').read_text())
        src_m = json.loads((PARENT / 'VF6' / role / 'manifest.json').read_text())
        src_path = PARENT / 'VF6' / role / src_m['files']['pointcloud']
        _, sxyz, sarr = read_vtp(src_path)
        path = OUT / role / m['files']['interior']['vtp']
        poly, xyz, arr = read_vtp(path)
        check(f'{role}/no_surface', poly.GetNumberOfPolys() == 0 and poly.GetNumberOfLines() == 0)
        check(f'{role}/point_kind_interior', np.all(arr['point_kind'] == 1))
        check(f'{role}/peel', float(arr['dist_to_wall_mm'].min()) > PEEL_MM,
              float(arr['dist_to_wall_mm'].min()))
        check(f'{role}/no_halfcut_field', 'view_plane_distance_mm' not in arr)
        check(f'{role}/count', len(xyz) == m['files']['interior']['points'])
        view = ROOT / 'data_wss_v5/views/wss_min_view_v1' / m['case_id']
        with np.load(view / 'bundle.npz') as b:
            nwall = len(b['wall_coords_aligned_mm'])
        vi = arr['query_index'].astype(np.int64) - nwall
        check(f'{role}/no_wall_query', np.all(vi >= 0))
        src_mask = sarr['dist_to_wall_mm'] > PEEL_MM
        check(f'{role}/mask_count', int(src_mask.sum()) == len(xyz),
              {'source': int(src_mask.sum()), 'vtp': int(len(xyz))})
        np.testing.assert_array_equal(xyz, sxyz[src_mask])
        check(f'{role}/xyz_from_source', True)
        np.testing.assert_array_equal(arr['speed_cfd'], sarr['speed_cfd'][src_mask])
        check(f'{role}/speed_from_source', True)
        check(f'{role}/selfmax_denominator_unchanged',
              m['display_normalization']['source_point_count'] == src_m['visualization_metrics']['n'])
        check(f'{role}/csv_exists', path.with_suffix('.csv.gz').exists())
        check(f'{role}/contract_no_halfcut', m['contract'].get('halfcut') is False)
    failed = [c for c in checks if not c['passed']]
    report = {
        'status': 'failed' if failed else 'passed',
        'peel_mm': PEEL_MM,
        'halfcut': False,
        'n_checks': len(checks),
        'n_failed': len(failed),
        'checks': checks,
    }
    (OUT / 'verification.json').write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n')
    print(report['status'], len(checks), 'checks', len(failed), 'failed')
    if failed:
        raise SystemExit(1)


if __name__ == '__main__':
    main()

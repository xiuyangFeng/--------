"""Export one interior-only VTP per VF6 case: drop cells within 1 mm of the wall.

No half-cut. Clip in post-processing if needed.
"""
from pathlib import Path
import argparse, csv, gzip, json
import numpy as np
import vtk
from vtk.util.numpy_support import vtk_to_numpy, numpy_to_vtk, numpy_to_vtkIdTypeArray
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize, LinearSegmentedColormap
from matplotlib.patches import Rectangle
from matplotlib.ticker import MaxNLocator
from plot_style import OUT, ROOT, page, sha

PARENT = OUT.parent
ROLES = ('best', 'median', 'worst')
PEEL_MM = 1.0
CMAP = plt.get_cmap('viridis')
ERR = LinearSegmentedColormap.from_list('signed_error', ['#356B99', '#FAFAFA', '#AD4946'])


def dump(path, data):
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False) + '\n')


def read_vtp(path):
    r = vtk.vtkXMLPolyDataReader(); r.SetFileName(str(path)); r.Update(); p = r.GetOutput()
    assert p.GetNumberOfPolys() == 0 and p.GetNumberOfLines() == 0
    xyz = vtk_to_numpy(p.GetPoints().GetData()).copy()
    arrays = {p.GetPointData().GetArrayName(i): vtk_to_numpy(p.GetPointData().GetArray(i)).copy()
              for i in range(p.GetPointData().GetNumberOfArrays())}
    return p, xyz, arrays


def write_vtp(path, source, xyz, arrays, indices, contract):
    pts = vtk.vtkPoints(); pts.SetData(numpy_to_vtk(np.ascontiguousarray(xyz[indices]), deep=True))
    poly = vtk.vtkPolyData(); poly.SetPoints(pts)
    verts = vtk.vtkCellArray()
    verts.SetData(numpy_to_vtkIdTypeArray(np.arange(len(indices) + 1, dtype=np.int64), deep=True),
                  numpy_to_vtkIdTypeArray(np.arange(len(indices), dtype=np.int64), deep=True))
    poly.SetVerts(verts)
    for key, values in arrays.items():
        a = numpy_to_vtk(np.ascontiguousarray(values[indices]), deep=True)
        a.SetName(key); poly.GetPointData().AddArray(a)
    poly.GetPointData().SetActiveScalars('speed_cfd')
    poly.GetPointData().SetActiveVectors('velocity_cfd_vector')
    poly.GetFieldData().DeepCopy(source.GetFieldData())
    note = vtk.vtkStringArray(); note.SetName('view_contract')
    note.InsertNextValue(json.dumps(contract, ensure_ascii=False))
    poly.GetFieldData().AddArray(note)
    w = vtk.vtkXMLPolyDataWriter(); w.SetFileName(str(path)); w.SetInputData(poly)
    w.SetDataModeToBinary(); w.SetCompressorTypeToZLib()
    assert w.Write() == 1
    _, rx, ra = read_vtp(path)
    np.testing.assert_array_equal(rx, xyz[indices])
    for key in arrays:
        np.testing.assert_array_equal(ra[key], arrays[key][indices])
    assert float(ra['dist_to_wall_mm'].min()) > PEEL_MM
    with gzip.open(path.with_suffix('.csv.gz'), 'wt', compresslevel=1) as stream:
        columns = {'x_mm': rx[:, 0], 'y_mm': rx[:, 1], 'z_mm': rx[:, 2]}
        for key, value in ra.items():
            if value.ndim == 1:
                columns[key] = value
            else:
                for j in range(value.shape[1]):
                    columns[f'{key}_{j}'] = value[:, j]
        stream.write(','.join(columns) + '\n')
        np.savetxt(stream, np.column_stack(list(columns.values())), delimiter=',', fmt='%.12g')


def export_case(role):
    case_dir = PARENT / 'VF6' / role
    m = json.loads((case_dir / 'manifest.json').read_text())
    source_path = case_dir / m['files']['pointcloud']
    source, xyz, arrays = read_vtp(source_path)
    cid = m['case_id']
    dist = arrays['dist_to_wall_mm']
    assert np.all(arrays['point_kind'] == 1)
    assert np.min(dist) > 0.001
    view = ROOT / 'data_wss_v5/views/wss_min_view_v1' / cid
    with np.load(view / 'bundle.npz') as b, np.load(view / 'volume.npz') as v:
        nwall = len(b['wall_coords_aligned_mm'])
        vi = arrays['query_index'].astype(np.int64) - nwall
        assert np.all(vi >= 0)
        arrays['volume_row_index'] = vi
        arrays['segment_id'] = v['vol_segment_id'][vi]
    mask = dist > PEEL_MM
    indices = np.flatnonzero(mask)
    contract = {
        'selection': f'dist_to_wall_mm > {PEEL_MM}',
        'peel_mm': PEEL_MM,
        'original_full_count': int(len(xyz)),
        'near_wall_dropped': int((~mask).sum()),
        'selected_count': int(len(indices)),
        'halfcut': False,
        'filter_uses_velocity': False,
        'official_metrics_scope': 'original full case; no new subset R2',
    }
    out = OUT / role; out.mkdir(exist_ok=True)
    path = out / f'VF6__{role}__{cid.replace("/", "__")}__interior.vtp'
    write_vtp(path, source, xyz, arrays, indices, contract)
    files = {'interior': {
        'vtp': path.name, 'csv': path.with_suffix('.csv.gz').name,
        'points': int(len(indices)), 'vtp_sha256': sha(path),
        'csv_sha256': sha(path.with_suffix('.csv.gz')),
    }}
    result = {
        'schema_version': 3, 'model': 'VF6', 'role': role, 'case_id': cid,
        'seed': m['seed'], 'peak_step': m['peak_step'], 'frame': m['frame'],
        'coordinate_unit': 'mm', 'velocity_unit': 'm/s',
        'selection': m['selection'],
        'official_full_case_metrics': m['visualization_metrics'],
        'display_normalization': m['display_normalization'],
        'files': files, 'contract': contract,
        'audit': {
            'min_dist_to_wall_mm_kept': float(dist[indices].min()),
            'near_wall_le_1mm_dropped': int((~mask).sum()),
        },
        'sources': [{'path': str(p.resolve()), 'sha256': sha(p)} for p in
                    (source_path, case_dir / 'manifest.json')],
    }
    dump(out / 'manifest.json', result)
    print(role, files['interior']['points'], 'kept /', len(xyz), flush=True)
    return result


def preview_basis(xyz):
    origin = xyz.mean(0)
    centered = xyz - origin
    _, basis = np.linalg.eigh(centered.T @ centered)
    right, up, look = basis[:, -1], basis[:, 1], basis[:, 0]
    if right[2] > 0:
        right = -right
    if look[0] < 0:
        look = -look
    up = np.cross(look, right)
    up /= np.linalg.norm(up)
    return origin, np.column_stack((right, up, look))


def plot_case(role, selfmax=False):
    folder = OUT / role
    m = json.loads((folder / 'manifest.json').read_text())
    xyz, arrays = read_vtp(folder / m['files']['interior']['vtp'])[1:]
    origin, basis = preview_basis(xyz)
    projected = (xyz - origin) @ basis
    fields = ['speed_cfd_selfmax', 'speed_pred_selfmax', 'speed_selfmax_error_pred_minus_cfd'] if selfmax \
        else ['speed_cfd', 'speed_pred', 'speed_error_pred_minus_cfd']
    maximum = 1. if selfmax else max(m['display_normalization']['cfd_max'],
                                     m['display_normalization']['pred_max'])
    emax = np.max(np.abs(arrays[fields[2]]))
    if selfmax:
        emax = max(emax, 1e-6)
    rolelabel = {'best': 'Best', 'median': 'Median', 'worst': 'Worst'}[role]
    fig = page(
        f'VF6 {rolelabel}：管内点云',
        f'{m["case_id"]}  ·  {len(xyz):,} 点  ·  距壁 > {PEEL_MM} mm，未切开',
        kicker='体内点云 / ' + ('SELFMAX' if selfmax else '原物理量'),
        footer='s1234 / ckpt_best / peak 1162。正式 R² 仍用完整域。后处理里自行 Clip。',
    )
    lefts = [.14, .42, .70]; width = .25; height = .50; bottom = .22
    lo, hi = projected[:, :2].min(0), projected[:, :2].max(0)
    center = (lo + hi) / 2; extent = (hi - lo) * 1.12
    ratio = width * 16 / (height * 9)
    extent[0] = max(extent[0], extent[1] * ratio); extent[1] = extent[0] / ratio
    order = np.argsort(projected[:, 2], kind='stable')
    for col, (field, title) in enumerate(zip(fields, ['CFD', 'Pred', 'Error'])):
        ax = fig.add_axes([lefts[col], bottom, width, height]); ax.set_gid(chr(97 + col))
        ax.add_patch(Rectangle((0, 0), 1, 1, transform=ax.transAxes, fc='#F0F3F4', ec='none', zorder=-1))
        cmap = ERR if col == 2 else CMAP
        norm = Normalize(-emax, emax) if col == 2 else Normalize(0, maximum)
        ax.scatter(projected[order, 0], projected[order, 1], c=arrays[field][order], s=.08,
                   cmap=cmap, norm=norm, edgecolors='none', linewidths=0, rasterized=True)
        ax.set(xlim=(center[0] - extent[0] / 2, center[0] + extent[0] / 2),
               ylim=(center[1] - extent[1] / 2, center[1] + extent[1] / 2), aspect='equal')
        ax.axis('off')
        fig.text(lefts[col] + width / 2, .745, title, fontsize=17, ha='center', weight='bold')
        ax.text(.016, .96, ax.get_gid(), transform=ax.transAxes, va='top', fontsize=11, weight='bold')
    for rect, colour, n, lab in [
        ([.14, .10, .38, .012], CMAP, Normalize(0, maximum), 'Speed / own full max' if selfmax else 'Speed (m/s)'),
        ([.57, .10, .38, .012], ERR, Normalize(-emax, emax), 'Pred − CFD (normalized)' if selfmax else 'Pred − CFD (m/s)'),
    ]:
        bar = fig.colorbar(plt.cm.ScalarMappable(norm=n, cmap=colour), cax=fig.add_axes(rect), orientation='horizontal')
        bar.ax.tick_params(labelsize=10, pad=5); bar.set_label(lab, fontsize=11, labelpad=3)
        if selfmax and colour is CMAP:
            bar.set_ticks([0, .5, 1])
        else:
            bar.locator = MaxNLocator(nbins=4); bar.update_ticks()
        bar.outline.set_linewidth(.5)
    dest = OUT / 'figures' / (f'VF6_{role}_interior' + ('_selfmax' if selfmax else ''))
    dest.parent.mkdir(exist_ok=True)
    fig.savefig(str(dest) + '.png', dpi=300)
    plt.close(fig)
    print(dest.name, flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--plots-only', action='store_true')
    parser.add_argument('--roles', nargs='+', default=list(ROLES), choices=ROLES)
    args = parser.parse_args()
    (OUT / 'figures').mkdir(exist_ok=True)
    if not args.plots_only:
        cases = [export_case(role) for role in args.roles]
        dump(OUT / 'manifest.json', {
            'schema_version': 3,
            'purpose': 'one interior VTP per case: keep points farther than 1 mm from the wall',
            'peel_mm': PEEL_MM,
            'halfcut': False,
            'cases': cases,
        })
        with (OUT / '打开文件清单.csv').open('w', encoding='utf-8-sig', newline='') as f:
            w = csv.writer(f); w.writerow(['role', 'case_id', 'points', 'vtp'])
            for m in cases:
                w.writerow([m['role'], m['case_id'], m['files']['interior']['points'],
                            str(Path(m['role']) / m['files']['interior']['vtp'])])
    for role in args.roles:
        plot_case(role, False)
        plot_case(role, True)


if __name__ == '__main__':
    main()

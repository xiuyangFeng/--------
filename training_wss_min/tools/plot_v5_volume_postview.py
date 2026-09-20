"""Export full VTP fields and static scientific figures from verified R5 predictions."""
from __future__ import annotations

import csv
import html
import json
import shutil
from pathlib import Path

import h5py
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.collections import PolyCollection
from matplotlib.colors import LinearSegmentedColormap
import numpy as np
from PIL import Image
import vtk
from vtk.util.numpy_support import numpy_to_vtk, numpy_to_vtkIdTypeArray, vtk_to_numpy

from training_wss_min.tools.export_v5_volume_postview import ROOT, OUT, REPORT, sha, write_json
from training_wss_min.tools.add_cfd_wall_mesh_vtp import write_mesh_vtp
from wss_v5 import contract as C

CMAP = LinearSegmentedColormap.from_list('field_bwr', ['#1f5a9e', '#ffffff', '#c62828'])
UNITS = {'pressure': 'Pa', 'speed': 'm/s'}
LABELS = {'pressure': 'Relative pressure (wall + interior)', 'speed': 'Speed |u| (interior)'}
plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 11, 'axes.spines.top': False,
                     'axes.spines.right': False, 'savefig.facecolor': 'white'})


def write_points(path, xyz, arrays):
    points = vtk.vtkPoints()
    points.SetData(numpy_to_vtk(np.ascontiguousarray(xyz), deep=True))
    poly = vtk.vtkPolyData()
    poly.SetPoints(points)
    cells = vtk.vtkCellArray()
    cells.SetData(numpy_to_vtkIdTypeArray(np.arange(len(xyz) + 1, dtype=np.int64), deep=True),
                  numpy_to_vtkIdTypeArray(np.arange(len(xyz), dtype=np.int64), deep=True))
    poly.SetVerts(cells)
    for name, data in arrays.items():
        arr = numpy_to_vtk(np.ascontiguousarray(data), deep=True)
        arr.SetName(name)
        poly.GetPointData().AddArray(arr)
    writer = vtk.vtkXMLPolyDataWriter()
    writer.SetFileName(str(path))
    writer.SetInputData(poly)
    writer.SetDataModeToBinary()
    assert writer.Write() == 1
    reader = vtk.vtkXMLPolyDataReader()
    reader.SetFileName(str(path)); reader.Update()
    loaded = reader.GetOutput()
    assert loaded.GetNumberOfPoints() == len(xyz)
    assert np.array_equal(vtk_to_numpy(loaded.GetPoints().GetData()), xyz)
    for name, data in arrays.items():
        assert np.array_equal(vtk_to_numpy(loaded.GetPointData().GetArray(name)), data), name


def triangles(uid, wall_xyz):
    with h5py.File(C.case_dir(uid) / 'case.h5') as h5:
        valid = h5['wall_static/valid'][()].astype(bool)
        tri = h5['topology/wall_triangles'][()].astype(np.int64)
    assert valid.sum() == len(wall_xyz)
    remap = np.full(len(valid), -1, dtype=np.int64)
    remap[valid] = np.arange(valid.sum())
    tri = remap[tri]
    return tri[(tri >= 0).all(axis=1)]


def limits(truth, pred):
    return float(min(truth.min(), pred.min())), float(max(truth.max(), pred.max()))


def field_triptych(path, xyz, truth, pred, title, unit, *, tri=None, axes=(0, 2), background=None):
    fig, axs = plt.subplots(1, 3, figsize=(12.8, 6.6), constrained_layout=True)
    vmin, vmax = limits(truth, pred)
    error = pred - truth
    errmax = max(float(np.abs(error).max()), 1e-12)
    normal = next(i for i in range(3) if i not in axes)
    # Draw far-to-near, with deterministic point thinning only for PNG rendering.
    if tri is None:
        idx = np.arange(len(xyz))
        if len(idx) > 85000:
            idx = np.random.default_rng(1234).choice(idx, 85000, replace=False)
        idx = idx[np.argsort(xyz[idx, normal])]
    else:
        tri = tri[np.argsort(xyz[tri, normal].mean(axis=1))]
    for ax, value, label in zip(axs, [truth, pred, error], ['CFD', 'Prediction', 'Prediction - CFD']):
        lo, hi = (-errmax, errmax) if label == 'Prediction - CFD' else (vmin, vmax)
        if background is not None:
            ax.scatter(background[::6, axes[0]], background[::6, axes[1]], c='#dce2e8', s=.25, rasterized=True)
        if tri is None:
            artist = ax.scatter(xyz[idx, axes[0]], xyz[idx, axes[1]], c=value[idx], cmap=CMAP,
                                vmin=lo, vmax=hi, s=.65, linewidths=0, rasterized=True)
        else:
            artist = PolyCollection(xyz[tri][:, :, axes], array=value[tri].mean(axis=1), cmap=CMAP,
                                    clim=(lo, hi), edgecolors='none', linewidths=0, rasterized=True)
            ax.add_collection(artist); ax.autoscale_view()
        ax.set_aspect('equal'); ax.set_title(label, fontsize=13)
        ax.set_xlabel('XYZ'[axes[0]] + ' (mm)')
        ax.set_ylabel('XYZ'[axes[1]] + ' (mm)')
        fig.colorbar(artist, ax=ax, orientation='horizontal', fraction=.055, pad=.09, label=unit)
    fig.suptitle(title + '\nShared CFD/Pred scale; full displayed range; atlas frame', fontsize=12)
    fig.savefig(path, dpi=190)
    plt.close(fig)


def depth_mean_triptych(path, xyz, truth, pred, title, unit):
    """Average original cells along Y in occupied XZ bins, for display only."""
    spacing = .75
    xedges = np.arange(xyz[:, 0].min() - spacing, xyz[:, 0].max() + 2*spacing, spacing)
    zedges = np.arange(xyz[:, 2].min() - spacing, xyz[:, 2].max() + 2*spacing, spacing)
    count = np.histogram2d(xyz[:, 0], xyz[:, 2], bins=(xedges, zedges))[0]
    grids = [np.histogram2d(xyz[:, 0], xyz[:, 2], bins=(xedges, zedges), weights=v)[0] /
             np.maximum(count, 1) for v in [truth, pred]]
    lo = min(g[count > 0].min() for g in grids); hi = max(g[count > 0].max() for g in grids)
    error = grids[1] - grids[0]; emax = max(float(np.abs(error).max()), 1e-12)
    fig, axs = plt.subplots(1, 3, figsize=(12.8, 6.6), constrained_layout=True)
    for ax, grid, label, bounds in zip(axs, [*grids, error], ['CFD', 'Prediction', 'Prediction - CFD'], [(lo, hi), (lo, hi), (-emax, emax)]):
        artist = ax.pcolormesh(xedges, zedges, np.ma.masked_where(count == 0, grid).T,
                               cmap=CMAP, vmin=bounds[0], vmax=bounds[1], shading='flat', rasterized=True)
        ax.set_aspect('equal'); ax.set(title=label, xlabel='X (mm)', ylabel='Z (mm)')
        fig.colorbar(artist, ax=ax, orientation='horizontal', fraction=.055, pad=.09, label=unit)
    fig.suptitle(title + '\nCell-count-weighted depth mean along Y; 0.75 mm XZ bins (display only)\nShared CFD/Pred scale; all interior cells contribute', fontsize=12)
    fig.savefig(path, dpi=190); plt.close(fig)


def horizontal(ax, cases, metric, title):
    ordered = sorted(cases, key=lambda c: (c[metric], c['case_id']))
    values = np.array([c[metric] for c in ordered])
    bins = np.linspace(values.min(), values.max(), int(np.sqrt(len(values))) + 1)
    counts, edges = np.histogram(values, bins=bins)
    tied = np.flatnonzero(counts == counts.max())
    mode_bin = min(tied, key=lambda i: (abs((edges[i]+edges[i+1])/2 - np.median(values)), int(i)))
    centre = (edges[mode_bin] + edges[mode_bin+1]) / 2
    frequent = min(ordered[1:-1], key=lambda c: (abs(c[metric]-centre), c['case_id']))
    ax.plot(values, np.arange(1, len(values)+1), '-o', color='#6aaed6', markerfacecolor='#3b78a8', lw=2, ms=5)
    for label, case, color in [('worst', ordered[0], '#d62f2f'), ('most frequent', frequent, '#e69f00'), ('best', ordered[-1], '#2e9d47')]:
        value = case[metric]
        ax.axvline(value, color=color, lw=1.8, label=f'{label}: {case["case_id"]} ({value:.3f})', zorder=1)
        ax.scatter([value], [ordered.index(case)+1], s=65, color=color, edgecolor='white', zorder=4)
    ax.set(xlabel='Linear-fit R²' if metric == 'r2_linear_fit' else 'Prediction R² (1 - SSE/SST)',
           ylabel='Case rank (worst → best)', title=title, ylim=(.5, len(values)+.5))
    ax.set_yticks([1, 5, 10, 15, 20, 25, 30, 34]); ax.grid(alpha=.22)
    ax.legend(loc='upper left', fontsize=8, framealpha=.92)


def combine(paths, output, columns=1):
    imgs = [Image.open(p).convert('RGB') for p in paths]
    width, height = max(i.width for i in imgs), max(i.height for i in imgs)
    canvas = Image.new('RGB', (columns*width, ((len(imgs)+columns-1)//columns)*height), 'white')
    for k, img in enumerate(imgs):
        canvas.paste(img, ((k % columns)*width, (k // columns)*height))
        img.close()
    canvas.save(output)


def main():
    manifest = json.loads((OUT / 'manifest.json').read_text())
    assert manifest['status'] == 'complete'
    REPORT.mkdir(parents=True, exist_ok=True)
    colormap = ROOT / 'outputs/field/postview/wss_v5_r4_best_worst_20260907/AG__fast__LOU_YANG__peak_wss/GNN_blue_white_red.xml'
    shutil.copy2(colormap, OUT / colormap.name)
    all_rows, previews, report_images, case_links = [], [], [], []
    fig_ranks, rank_axes = plt.subplots(2, 2, figsize=(17, 12), constrained_layout=True)
    export_checks = []
    for arm_i, (field, arm) in enumerate(manifest['arms'].items()):
        for metric_i, metric in enumerate(['r2', 'r2_linear_fit']):
            title = f'V5 R5{"P" if field == "pressure" else "V"} | {LABELS[field]} | best checkpoint'
            horizontal(rank_axes[arm_i, metric_i], arm['cases'], metric, title)
            fig, ax = plt.subplots(figsize=(10.2, 7), constrained_layout=True)
            horizontal(ax, arm['cases'], metric, title)
            fig.savefig(REPORT / f'{field}_{metric}_horizontal.png', dpi=220)
            fig.savefig(REPORT / f'{field}_{metric}_horizontal.pdf')
            plt.close(fig)
        for record in arm['cases']:
            all_rows.append({'field': field, **{k: record[k] for k in ['case_id', 'selection', 'n', 'r2', 'r2_linear_fit', 'linear_fit_slope', 'linear_fit_intercept', 'mae', 'rmse']}})
        selected = sorted([c for c in arm['cases'] if c['selection']], key=lambda c: c['selection'])
        for case_i, rec in enumerate(selected):
            folder = Path(rec['folder']); plots = folder / 'plots'; plots.mkdir(exist_ok=True)
            report_case = REPORT / field / folder.name; report_case.mkdir(parents=True, exist_ok=True)
            assert sha(folder / 'same_point_fields.npz') == rec['archive_sha256']
            with np.load(folder / 'same_point_fields.npz') as z:
                data = {k: z[k] for k in z.files}
            xyz, wall = data['xyz_mm'], data['wall_xyz_mm']
            truth, pred, kind = data['truth'], data['pred'], data['point_kind']
            tri = triangles(rec['case_id'], wall)
            arrays = {f'{field}_cfd': truth, f'{field}_pred': pred,
                      f'err_{field}': pred-truth, f'abs_err_{field}': np.abs(pred-truth),
                      'point_kind_0_wall_1_interior': kind, 'source_index': data['source_index']}
            if field == 'speed':
                arrays.update(velocity_cfd_aligned_m_s=data['truth_components'],
                              velocity_pred_aligned_m_s=data['pred_components'],
                              velocity_error_aligned_m_s=data['pred_components']-data['truth_components'],
                              speed_cfd_over_cfd_max=truth/truth.max(), speed_pred_over_cfd_max=pred/truth.max(),
                              speed_pred_over_pred_max=pred/pred.max())
            else:
                arrays['pressure_reference_pa'] = np.full(len(truth), float(data['p_ref_pa']))
            write_points(folder / 'full_pointcloud.vtp', xyz, arrays)
            interior = kind == 1
            write_points(folder / 'interior_pointcloud.vtp', xyz[interior], {k: v[interior] for k, v in arrays.items()})
            if field == 'pressure':
                surface_arrays = {k: v[~interior] for k, v in arrays.items()}
                assert np.array_equal(xyz[~interior], wall)
                write_mesh_vtp(wall, tri, surface_arrays, folder / 'cfd_wall_mesh.vtp')
                reader = vtk.vtkXMLPolyDataReader(); reader.SetFileName(str(folder / 'cfd_wall_mesh.vtp')); reader.Update()
                mesh = reader.GetOutput()
                assert mesh.GetNumberOfPolys() == len(tri)
                assert np.array_equal(vtk_to_numpy(mesh.GetPoints().GetData()), wall)
                assert np.array_equal(vtk_to_numpy(mesh.GetPointData().GetArray('pressure_pred')), pred[~interior])
            else:
                write_mesh_vtp(wall, tri, {'geometry_only': np.ones(len(wall))}, folder / 'wall_geometry_only.vtp')
            # A thin slab of original cells, without invented volume connectivity/interpolation.
            y0 = float(np.median(xyz[interior, 1])); half_width = 1.5
            slab = interior & (np.abs(xyz[:, 1] - y0) <= half_width)
            write_points(folder / 'interior_3mm_slab.vtp', xyz[slab], {k: v[slab] for k, v in arrays.items()})
            title = f'{field.upper()} | {rec["selection"].upper()}: {rec["case_id"]}\nPrediction R² = {rec["r2"]:.4f}; MAE = {rec["mae"]:.4f} {UNITS[field]}'
            if field == 'pressure':
                field_triptych(plots / 'fig_wall_pressure.png', wall, truth[~interior], pred[~interior], title + '\nExact CFD wall triangulation', UNITS[field], tri=tri)
                previews.append(plots / 'fig_wall_pressure.png')
            else:
                depth_mean_triptych(plots / 'fig_interior_depth_mean.png', xyz[interior], truth[interior], pred[interior], title, UNITS[field])
                previews.append(plots / 'fig_interior_depth_mean.png')
            field_triptych(plots / 'fig_interior_projection.png', xyz[interior], truth[interior], pred[interior], title + '\nFull interior projection (depth overlap)', UNITS[field])
            field_triptych(plots / 'fig_interior_slab.png', xyz[slab], truth[slab], pred[slab],
                           title + f'\nOriginal-cell slab: Y = {y0:.1f} ± {half_width:.1f} mm', UNITS[field], background=wall)
            for j, q in enumerate([.25, .5, .75]):
                z0 = float(np.quantile(xyz[interior, 2], q))
                cross = interior & (np.abs(xyz[:, 2] - z0) <= 1.)
                field_triptych(plots / f'fig_cross_section_{j+1}.png', xyz[cross], truth[cross], pred[cross],
                               title + f'\nOriginal-cell transverse slab: Z = {z0:.1f} ± 1.0 mm', UNITS[field], axes=(0, 1))
            for p in sorted(plots.iterdir()):
                shutil.copy2(p, report_case / p.name)
                if p.suffix == '.png':
                    report_images.append(report_case / p.name)
            info = {'field': field, 'case_id': rec['case_id'], 'selection': rec['selection'], 'n': len(xyz),
                    'n_interior': int(interior.sum()), 'n_slab': int(slab.sum()), 'n_wall': len(wall), 'n_triangles': len(tri),
                    'slab_center_y_mm': y0, 'slab_width_mm': 3, 'vtp_roundtrip_passed': True,
                    'figure_subsampling_only': '85,000 points maximum per PNG; metrics and VTP use all points'}
            write_json(folder / 'export_verification.json', info); export_checks.append(info)
            case_links.append((field, rec, report_case))
            print(f'Exported {field} {rec["selection"]}: {len(xyz):,} points; VTP round-trip passed', flush=True)
    fig_ranks.savefig(REPORT / 'horizontal_r2_overview.png', dpi=180)
    fig_ranks.savefig(REPORT / 'horizontal_r2_overview.pdf'); plt.close(fig_ranks)
    combine(previews, REPORT / 'field_best_worst_overview.png', columns=2)
    with (REPORT / 'per_case_r2_and_linear_fit.csv').open('w', newline='', encoding='utf-8-sig') as f:
        writer = csv.DictWriter(f, fieldnames=list(all_rows[0])); writer.writeheader(); writer.writerows(all_rows)
    write_json(OUT / 'export_verification.json', {'passed': True, 'cases': export_checks})
    shutil.copy2(OUT / 'manifest.json', REPORT / 'manifest.json')
    for p in ['horizontal_r2_overview.png', 'field_best_worst_overview.png', 'per_case_r2_and_linear_fit.csv']:
        shutil.copy2(REPORT / p, OUT / p)
    readme(manifest, case_links)
    from training_wss_min.tools.plot_v5_volume_scatter import refresh
    refresh()


def readme(manifest, case_links):
    lines = ['# V5 R5P 混合压力 / R5V 内部速度：最好、最差病例后处理', '',
             '2026-09-09；峰值步 1162；原 best checkpoint；test34；固定 5000 壁面 support / seed1234；全量 query。', '',
             '两组均按原 metrics.json 的逐病例物理预测 R² 选首尾。压力是壁面∪内部的相对压力 p−p_ref（Pa）；速度是预测三分量的幅值 |u|（m/s）。', '',
             '| 目标 | 选例 | 病例 | 预测 R² | 拟合 R² | 斜率 a | 截距 b | MAE |', '|---|---|---|---:|---:|---:|---:|---:|']
    for field, r, report_case in case_links:
        lines.append(f'| {field} | {r["selection"]} | {r["case_id"]} | {r["r2"]:.6f} | {r["r2_linear_fit"]:.6f} | {r["linear_fit_slope"]:.4f} | {r["linear_fit_intercept"]:.4f} | {r["mae"]:.4f} |')
    lines += ['', '## 直接查看', '',
              '- `index.html`：全部图件导航。`field_best_worst_overview.png`：压力壁面/速度内部场总览。',
              '- `scatter_fit_best_worst.png/pdf`：四张全点密度散点拟合；单例另有独立图，压力还分壁面/内部画图。',
              '- `horizontal_r2_overview.png/pdf`：两组 × 预测R²/拟合R²；每张也单独导出 PNG/PDF。',
              '- `per_case_r2_and_linear_fit.csv`：完整68行结果；拟合 pred = a·CFD+b，以全部同点数据计算。', '',
              '## ParaView 包', '', f'完整文件目录：`{OUT}`。', '',
              '1. 压力打开每例 `cfd_wall_mesh.vtp` → Apply → Surface：真实 CFD 壁面三角网格直接挂同点压力，无插值。',
              '2. 内部场打开 `interior_pointcloud.vtp` → Apply → Points（Point Size 2–4）。完整源场为 `full_pointcloud.vtp`。',
              '3. 看腔内打开 `interior_3mm_slab.vtp`，或对内部点云用两个 Clip/Box 保留薄层。原始点云没有体单元拓扑，不能把普通 Slice 当连续 CFD 截面。',
              '4. 真值/预测字段：`pressure_cfd / pressure_pred`（Pa，相对压），`speed_cfd / speed_pred`（m/s）；误差为 `err_*`（Pred−CFD）及 `abs_err_*`。',
              '5. 速度方向可用 Glyph，Vectors 选 `velocity_pred_aligned_m_s` 或 `velocity_cfd_aligned_m_s`；`wall_geometry_only.vtp` 可叠加透明外壳。',
              '6. CFD 与 Pred 色标用同一区间；可导入根目录 `GNN_blue_white_red.xml`。全部坐标与速度向量为同一 atlas 解剖系，长度 mm、速度 m/s。',
              '7. `point_kind_0_wall_1_interior` 标明点类型；速度文件只有内部点。`source_index`：压力为壁面后接内部的query行号，速度为volume内部行号。', '',
              '## 图与指标口径', '',
              '- 预测 R² = 1−SSE/SST；拟合 R² = Pearson r²，必须结合 a、b 判断。选例依预测 R²；拟合横向图的红绿线依拟合 R² 自己排序，病例可能不同。橙线取等宽5箱中最高频箱中心最近病例，非概率密度估计。',
              '- 压力标签沿用训练视图 v1.1：壁面压力来自最近内部单元；p_ref 为该例峰值步体积平均压力，不是壁面原导出压力，也不是绝对压力。未对测试预测另做校准或重新居中。',
              '- 压力KANG_YONG真值标准差约78 Pa；尽管相关性较高，幅值偏差使预测R²为负，MAE约75 Pa。负R²没有裁掉。',
              '- 静态云图CFD与Pred使用共享范围，误差用对称范围。内部全云是沿Y投影，可能遮挡；另给Y方向3mm薄层及Z方向三个2mm横截薄层。薄层只含原始单元中心，无插值。',
              '- 速度总览使用 `fig_interior_depth_mean.png`：在0.75mm的XZ网格内，沿Y对原内部单元的值作点数等权平均，空格透明；帮助观察被近壁低速点遮挡的内部场。这是显示用深度平均，不是CFD截面或体积加权平均，不用于拟合和正式指标。原三维点场不变。',
              '- PNG内部点最多显示85,000点；散点密度统计、拟合、VTP与NPZ均使用全部同点数据。壁面静态图三角面颜色取三顶点均值，正式指标仍按原节点。',
              '- 速度另存CFDmax共同分母和Predmax自分母字段；双自最大值仅供看形状。压力为有符号相对压，不做自最大值归一化。',
              '- 每例 `same_point_fields.npz` 保存全量同点值、点索引和坐标；manifest记录checkpoint、输入及数据SHA256。', '',
              '## 验证与复现', '',
              f'- 两组34例共{manifest["numerical_checks"]}项原始逐病例基本指标复现通过，并检查全部68个原斜率；模型和原评估文件未改动。速度复用已验证全点缓存，压力按原协议重推理。',
              '- 全部点云VTP重新读回后坐标、标量、向量逐值一致；压力网格点数、三角数及标量一致。',
              '- `python -m training_wss_min.tools.export_v5_volume_postview --device cuda:0`',
              '- `python -m training_wss_min.tools.plot_v5_volume_postview`',
              '- 运行环境：`/public/newhome/cy/.conda/envs/GNN/bin/python`。单seed、已暴露test34，仅为既有模型结果展示。', '']
    content = '\n'.join(lines)
    (REPORT / 'README.md').write_text(content); (OUT / 'README.md').write_text(content)
    body = ['<!doctype html><html lang="zh-CN"><meta charset="utf-8"><title>V5体场可视化</title><style>body{font:16px system-ui;max-width:1400px;margin:32px auto;padding:0 24px;background:#f6f8fb;color:#172536}img{max-width:100%;background:white;border-radius:8px}section{margin:32px 0}a{color:#1667ac}p{line-height:1.8}</style>',
            '<h1>V5 混合压力与内部速度：最好 / 最差病例</h1><p>峰值1162 · 原best checkpoint · test34 · 按物理预测R²选例。图片点击可查看原图；完整三维点场见ParaView包。</p>',
            '<p><a href="README.md">口径与打开说明</a> · <a href="per_case_r2_and_linear_fit.csv">全部68行指标CSV</a></p>']
    for p, title in [('field_best_worst_overview.png', '场分布总览'), ('scatter_fit_best_worst.png', '最好 / 最差同点散点拟合'), ('horizontal_r2_overview.png', '预测R²与拟合R²横向排序')]:
        body.append(f'<section><h2>{title}</h2><a href="{p}"><img src="{p}" loading="lazy"></a></section>')
    for field, r, folder in case_links:
        body.append(f'<section><h2>{html.escape(field + " / " + r["selection"] + " / " + r["case_id"])}</h2>')
        body.append(f'<p>预测R² {r["r2"]:.4f} · 拟合R² {r["r2_linear_fit"]:.4f} · MAE {r["mae"]:.4f} {UNITS[field]}</p>')
        for p in sorted(folder.glob('*.png')):
            relative = p.relative_to(REPORT).as_posix()
            body.append(f'<details><summary>{p.stem}</summary><a href="{relative}"><img src="{relative}" loading="lazy"></a></details>')
        body.append('</section>')
    body.append('</html>')
    (REPORT / 'index.html').write_text('\n'.join(body))


if __name__ == '__main__':
    main()

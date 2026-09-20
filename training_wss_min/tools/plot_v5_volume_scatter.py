"""Refresh R5 volume scatters in the R4 reference style, without inference."""
from __future__ import annotations

import json
import shutil
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from training_wss_min.tools.export_v5_volume_postview import OUT, REPORT, sha, write_json
from training_wss_min.tools.plot_postview_case_scatter import panel, point_density


def refresh():
    # English scientific labels work on nodes without a CJK font installation.
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.sans-serif': ['DejaVu Sans'],
                         'axes.spines.top': True, 'axes.spines.right': True})
    manifest = json.loads((OUT / 'manifest.json').read_text())
    assert manifest['status'] == 'complete'
    report = {'style_source': 'training_wss_min/tools/plot_postview_case_scatter.py',
              'density': 'Gaussian KDE fitted to fixed-seed 6000-point sample, evaluated on 256x256 grid and interpolated for point colors only',
              'scatter_and_fit': 'all original query points; no subsampling, clipping or calibration',
              'normalization': 'train138 scalar linear z (pressure or speed), same transform for CFD and prediction', 'cases': []}
    combined, combined_axes = plt.subplots(2, 2, figsize=(11, 10.4))
    for field_i, (field, arm) in enumerate(manifest['arms'].items()):
        stats = json.loads((Path(arm['run']) / 'wss_global_stats.json').read_text())
        norm = stats['linear' if field == 'pressure' else 'speed']
        mean, std = float(norm['mean']), float(norm['std'])
        assert std > 0
        units = 'Pa' if field == 'pressure' else 'm/s'
        field_name = 'pressure' if field == 'pressure' else 'speed'
        count_label = 'wall + interior points' if field == 'pressure' else 'interior points'
        tag = 'R5P' if field == 'pressure' else 'R5V'
        fig, axes = plt.subplots(2, 2, figsize=(11, 10.4))
        selected = sorted([r for r in arm['cases'] if r['selection']], key=lambda r: r['selection'])
        for row, rec in enumerate(selected):
            folder = Path(rec['folder']); plots = folder / 'plots'
            target = REPORT / field / folder.name
            assert sha(folder / 'same_point_fields.npz') == rec['archive_sha256']
            with np.load(folder / 'same_point_fields.npz') as z:
                x, y, kinds = z['truth'], z['pred'], z['point_kind']
            density = point_density(x, y, grid_size=256)
            assert np.isfinite(density).all() and (density >= 0).all()
            title = f'[{rec["selection"]}] {rec["case_id"]}'
            paired, pair_axes = plt.subplots(1, 2, figsize=(11, 5.4))
            metrics = {}
            for col, (tx, ty, unit, space) in enumerate([
                    (x, y, units, 'physical'), ((x-mean)/std, (y-mean)/std, 'linear z', 'normalized')]):
                # Positive affine normalization preserves density ordering.
                options = dict(title=f'{title} · {space} ({unit})', unit=unit,
                               field_name=field_name, point_label=count_label, density=density, point_size=.7)
                metrics[space] = panel(axes[row, col], tx, ty, **options)
                panel(pair_axes[col], tx, ty, **options)
                if col == 0:
                    panel(combined_axes[row, field_i], tx, ty, **options)
            for key in ['r2_raw', 'r2_fit', 'slope']:
                assert abs(metrics['physical'][key] - metrics['normalized'][key]) < 1e-10
            for key, expected in [('r2_raw', rec['r2']), ('r2_fit', rec['r2_linear_fit']),
                                  ('slope', rec['linear_fit_slope']), ('intercept', rec['linear_fit_intercept'])]:
                assert np.isclose(metrics['physical'][key], expected, rtol=1e-10, atol=1e-10)
            assert np.isclose(metrics['normalized']['intercept'],
                              (metrics['physical']['intercept'] + (metrics['physical']['slope']-1)*mean)/std,
                              rtol=1e-10, atol=1e-10)
            paired.suptitle(f'{tag} s1234 best · pred = a·CFD + b · all same-point queries', fontsize=12)
            paired.tight_layout()
            for suffix in ['png', 'pdf']:
                output = plots / f'fig_scatter_fit.{suffix}'
                paired.savefig(output, dpi=170)
                shutil.copy2(output, target / output.name)
            plt.close(paired)
            if field == 'pressure':
                groups, group_axes = plt.subplots(1, 3, figsize=(16.5, 5.4))
                for ax, mask, label in zip(group_axes, [np.ones(len(x), bool), kinds == 0, kinds == 1],
                                           ['wall + interior points', 'wall nodes', 'interior points']):
                    dens = density if mask.all() else point_density(x[mask], y[mask], grid_size=256)
                    panel(ax, x[mask], y[mask], title=f'{title}\n{label}', unit=units,
                          field_name=field_name, point_label=label, density=dens, point_size=.7)
                groups.tight_layout()
                for suffix in ['png', 'pdf']:
                    output = plots / f'fig_pressure_group_scatter.{suffix}'
                    groups.savefig(output, dpi=170); shutil.copy2(output, target / output.name)
                plt.close(groups)
            report['cases'].append({'field': field, 'case': rec['case_id'], 'selection': rec['selection'],
                                    'archive_sha256': rec['archive_sha256'], 'mean': mean, 'std': std, **metrics})
            print(f'{field} {rec["selection"]}: {len(x):,} density-colored points, physical/linear-z metrics verified', flush=True)
        fig.suptitle(f'{tag} s1234 best · pred = a·CFD + b · all same-point queries\nTop: best case; bottom: worst case · Left: physical; right: train138 linear z', fontsize=12)
        fig.tight_layout()
        for suffix in ['png', 'pdf']:
            output = REPORT / f'{field}_scatter_fit_best_worst.{suffix}'
            fig.savefig(output, dpi=170); shutil.copy2(output, OUT / output.name)
        plt.close(fig)
    combined.suptitle('R5P / R5V s1234 best · pred = a·CFD + b · all same-point queries\nTop: best case; bottom: worst case · Left: pressure (Pa); right: speed (m/s)', fontsize=12)
    combined.tight_layout()
    for suffix in ['png', 'pdf']:
        output = REPORT / f'scatter_fit_best_worst.{suffix}'
        combined.savefig(output, dpi=170); shutil.copy2(output, OUT / output.name)
    plt.close(combined)
    report['verified'] = True
    write_json(REPORT / 'scatter_style_verification.json', report)
    write_json(OUT / 'scatter_style_verification.json', report)
    for root in [REPORT, OUT]:
        path = root / 'README.md'
        content = path.read_text().split('\n<!-- R5_SCATTER_STYLE -->')[0]
        content += ('\n<!-- R5_SCATTER_STYLE -->\n## 参考R4样式的散点图（2026-09-09更新）\n\n'
                    '- `pressure_scatter_fit_best_worst.png/pdf`、`speed_scatter_fit_best_worst.png/pdf`：各自上排最好、下排最差，左物理、右linear z。\n'
                    '- 复用 `plot_postview_case_scatter.py`：viridis密度着色的细散点、红色拟合直线、灰色y=x虚线、左上指标框和右下方程。每个原始query点均实际绘制。\n'
                    '- 大点云的颜色采用固定seed抽取6000点拟合Gaussian KDE，再在256×256格上求值并插值回各点，仅近似颜色。拟合、R²和点数均用全量原值。\n'
                    '- 压力和速度幅值的归一化均为训练集统计的线性z，不使用WSS的log_z。因此左右R²与斜率相同，单位和截距改变；已核对这一关系及原始物理指标。\n'
                    '- 独立重绘命令：`python -m training_wss_min.tools.plot_v5_volume_scatter`，无需重新推理或导出VTP。\n')
        path.write_text(content)
    index = REPORT / 'index.html'
    text = index.read_text()
    if '<!-- R5_SCATTER_STYLE -->' in text:
        text = text.split('<!-- R5_SCATTER_STYLE -->')[0] + '</html>'
    sections = ['<!-- R5_SCATTER_STYLE -->']
    for field, label in [('pressure', '混合压力'), ('speed', '内部速度幅值')]:
        stem = f'{field}_scatter_fit_best_worst'
        sections.append(f'<section><h2>{label}：物理 / 线性z散点（参考R4样式）</h2><p>上：最好，下：最差；左：物理，右：训练集线性z。<a href="{stem}.pdf">PDF</a></p><a href="{stem}.png"><img src="{stem}.png" loading="lazy"></a></section>')
    index.write_text(text.replace('</html>', '\n'.join(sections) + '</html>'))


if __name__ == '__main__':
    refresh()

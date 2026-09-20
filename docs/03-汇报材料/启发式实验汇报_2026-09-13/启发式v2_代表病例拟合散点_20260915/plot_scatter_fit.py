#!/usr/bin/env python3
"""Best / median / worst Pred-vs-CFD scatter with pred = a·CFD + b.

Uses the delivered same-point CSVs. Official prediction R² is 1−SSE/SST on the
identity; fit R² is the regression R² of the affine line. No inference.
"""
from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from scipy.interpolate import RegularGridInterpolator
from scipy.stats import gaussian_kde

from plot_style import (
    GREY, INK, LINE, ORANGE, OUT, RED, ROOT, TEAL,
    page, panel_label, save,
)

sys.path.insert(0, str(ROOT))
from training_wss_min.metrics import linear_fit_metrics, r2_score
from training_wss_min.tools.plot_postview_case_scatter import point_density as _point_density

CASES = ROOT / 'docs' / '03-汇报材料' / '启发式实验汇报_2026-09-13' / '启发式v2_代表病例后处理_20260914'
ROLES = ('best', 'median', 'worst')
ROLE_CN = {'best': '最好', 'median': '中位', 'worst': '最差'}
ROLE_COLOR = {'best': TEAL, 'median': ORANGE, 'worst': RED}
MODELS = {
    'X5X11': dict(
        stem='X5X11_scatter_fit',
        field='wss',
        x='wss_cfd_pa',
        y='wss_pred_pa',
        unit='Pa',
        field_cn='WSS',
        point_label='壁面节点',
        point_size=2.4,
        kicker='wave-2 · 直接 WSS · X5X11 · seed 1234',
        title='X5X11 最好例接近恒等，最差例是斜率腰斩',
        subtitle='灰虚线 y=x；红线 pred = a·CFD + b。R² 预测是正式同点指标；R² 拟合必须连 a、b 一起读。',
        claim='X5X11 最好例斜率 0.95；最差例斜率 0.45，拟合 R² 只到 0.54，不是截距能解释的偏差',
        footer_domain='监督原壁面 WSS（Pa）',
    ),
    'VF6': dict(
        stem='VF6_scatter_fit',
        field='speed',
        x='speed_cfd',
        y='speed_pred',
        unit='m/s',
        field_cn='速度大小',
        point_label='体内点',
        point_size=0.55,
        kicker='wave-2 · 体场速度 · VF6 · seed 1234',
        title='VF6 最好例接近恒等，最差例同样是斜率腰斩',
        subtitle='灰虚线 y=x；红线 pred = a·CFD + b。速度大小在体内同点上拟合；不是向量方向。',
        claim='VF6 最好例斜率 0.97；最差例斜率 0.48，拟合 R² 几乎不抬预测 R²',
        footer_domain='监督体内速度大小（m/s）',
    ),
    'PF6': dict(
        stem='PF6_scatter_fit',
        field='pressure',
        x='pressure_cfd_pa',
        y='pressure_pred_pa',
        unit='Pa',
        field_cn='相对压力',
        point_label='壁面∪体内点',
        point_size=0.55,
        kicker='wave-2 · 体场压力 · PF6 · seed 1234',
        title='PF6 中位例相关仍高但幅值压缩，最差例拟合也掉',
        subtitle='灰虚线 y=x；红线 pred = a·CFD + b。中位例说明拟合 R² 高不能代替斜率接近 1。',
        claim='PF6 最好例接近恒等；中位例拟合 R² 0.93 但斜率仅 0.69；最差例预测与拟合一起掉',
        footer_domain='监督壁面∪体内相对压力（Pa）',
    ),
    'VF6_wss': dict(
        stem='VF6_wss_scatter_fit',
        field='wss',
        x='wss_cfd_pa',
        y='wss_pred_pa',
        unit='Pa',
        field_cn='WSS',
        point_label='壁面节点',
        point_size=2.4,
        kicker='VELWSS2 · VF6 速度 → 冻结 V3 派生 WSS · seed 1234',
        title='VF6 派生 WSS 最好例仍有幅值压缩，最差例落到负 R²',
        subtitle='灰虚线 y=x；红线 pred = a·CFD + b。这是速度经冻结算子得到的壁面 WSS，不是直接回归。',
        claim='VELWSS2 派生 WSS 的预测 R² 与拟合 R² 见三面板；最差例保留负 R²',
        footer_domain='VF6 预测速度 → 冻结 Profile-Secant V3 校准 WSS（Pa）',
    ),
    'X5D_v51': dict(
        stem='X5D_v51_scatter_fit',
        field='wss',
        x='wss_cfd_pa',
        y='wss_pred_pa',
        unit='Pa',
        field_cn='WSS',
        point_label='壁面节点',
        point_size=2.4,
        kicker='v5.1 · 直接 WSS · X5D_v51 · seed 1234',
        title='X5D_v51 最好例接近恒等，最差例仍是斜率压缩',
        subtitle='灰虚线 y=x；红线 pred = a·CFD + b。R² 预测是正式同点指标；R² 拟合必须连 a、b 一起读。',
        claim='X5D_v51 最好/中位/最差的斜率与预测 R² 见三面板；选例按五 seed 均值，场来自 s1234',
        footer_domain='监督原壁面 WSS（Pa）',
        footer_selection='选例按五 seed 病例 R² 均值（1234/7/2025/11/2026）；Median 是距分布中位最近的真实病例。',
    ),
    'V4_EMA': dict(
        stem='V4_EMA_scatter_fit',
        field='wss',
        x='wss_cfd_pa',
        y='wss_pred_pa',
        unit='Pa',
        field_cn='WSS',
        point_label='壁面节点',
        point_size=2.4,
        kicker='历史 V4 · PN BC+PDE-EMA · 派生 WSS · seed 1234',
        title='V4 EMA 派生 WSS 最好例仍弱，最差例落到负 R²',
        subtitle='灰虚线 y=x；红线 pred = a·CFD + b。这是速度经冻结算子得到的壁面 WSS，不是直接回归。',
        claim='V4 EMA 派生 WSS 的预测 R² 与拟合 R² 见三面板；最差例保留负 R²',
        footer_domain='V4 预测速度 → 冻结 Profile-Secant V3 校准 WSS（Pa）',
        footer_selection='选例按 test35 单 seed 1234 全壁面派生 WSS R²；Median 是距分布中位最近的真实病例。checkpoint 名为 last_converged，实际未收敛。',
        footer_frame='s1234 / last_converged / 历史 V4 稳态 peak 同点，不是 V5 peak 1162。',
    ),
    'V4_EMA_speed': dict(
        stem='V4_EMA_speed_scatter_fit',
        folder='V4_EMA',
        csv_file='same_point_speed.csv.gz',
        metrics_key='visualization_metrics_speed',
        field='speed',
        x='speed_cfd',
        y='speed_pred',
        unit='m/s',
        field_cn='速度大小',
        point_label='体内点',
        point_size=0.55,
        kicker='历史 V4 · PN BC+PDE-EMA · 体内速度 · seed 1234',
        title='V4 EMA 速度：WSS 选例上的体内速度大小',
        subtitle='灰虚线 y=x；红线 pred = a·CFD + b。三例按派生 WSS 选取，不是速度独立排名。',
        claim='同例速度散点见三面板；正式 R² 为严格体内同点',
        footer_domain='监督体内速度大小（m/s）',
        footer_selection='病例角色沿用派生 WSS best/median/worst；速度 R² 是这些病例的官方体内指标。',
        footer_frame='s1234 / last_converged / 历史 V4 稳态 peak 同点，不是 V5 peak 1162。',
    ),
    'V4_EMA_pressure': dict(
        stem='V4_EMA_pressure_scatter_fit',
        folder='V4_EMA',
        csv_file='same_point_pressure.csv.gz',
        metrics_key='visualization_metrics_pressure',
        field='pressure',
        x='pressure_cfd_pa',
        y='pressure_pred_pa',
        unit='Pa',
        field_cn='相对压力',
        point_label='体内点',
        point_size=0.55,
        kicker='历史 V4 · PN BC+PDE-EMA · 相对压力 · seed 1234',
        title='V4 EMA 压力：WSS 选例上的体内相对压力',
        subtitle='灰虚线 y=x；红线 pred = a·CFD + b。正式 R² 为严格体内同点；壁面点仅用于可视化。',
        claim='同例压力散点见三面板；正式口径不含壁面点',
        footer_domain='监督体内相对压力（Pa）',
        footer_selection='病例角色沿用派生 WSS best/median/worst；压力 R² 是这些病例的官方体内指标。',
        footer_frame='s1234 / last_converged / 历史 V4 稳态 peak 同点，不是 V5 peak 1162。',
    ),
}


def density_colors(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    dens = _point_density(x, y, max_kde=6000, seed=0, grid_size=256)
    if dens.shape != x.shape:
        raise ValueError('density size mismatch')
    interp = RegularGridInterpolator
    # Re-evaluate with bounds_error=False if the helper's interpolator clipped.
    if not np.isfinite(dens).all():
        rng = np.random.default_rng(0)
        idx = rng.choice(len(x), size=min(6000, len(x)), replace=False)
        kde = gaussian_kde(np.vstack([x[idx], y[idx]]))
        gx = np.linspace(float(x.min()), float(x.max()), 256)
        gy = np.linspace(float(y.min()), float(y.max()), 256)
        xx, yy = np.meshgrid(gx, gy, indexing='ij')
        grid = kde(np.vstack([xx.ravel(), yy.ravel()])).reshape(xx.shape)
        dens = interp((gx, gy), grid, bounds_error=False, fill_value=None)(np.column_stack([x, y]))
    assert np.isfinite(dens).all() and np.all(dens >= 0)
    return dens


def load_case(model: str, role: str, spec: dict) -> dict:
    folder = spec.get('root', CASES) / spec.get('folder', model) / role
    manifest = json.loads((folder / 'manifest.json').read_text())
    csv_path = folder / '_export' / spec.get('csv_file', 'same_point_fields.csv.gz')
    df = pd.read_csv(csv_path, usecols=[spec['x'], spec['y']])
    x = df[spec['x']].to_numpy(np.float64)
    y = df[spec['y']].to_numpy(np.float64)
    finite = np.isfinite(x) & np.isfinite(y)
    assert finite.all(), (model, role, int((~finite).sum()))
    raw = float(r2_score(x, y))
    fit = linear_fit_metrics(x, y)
    expected = manifest[spec.get('metrics_key', 'visualization_metrics')]
    assert int(len(x)) == int(expected['n']), (model, role, len(x), expected['n'])
    assert np.isclose(raw, expected['r2'], rtol=0, atol=1e-8), (model, role, raw, expected['r2'])
    mae = float(np.mean(np.abs(y - x)))
    assert np.isclose(mae, expected['mae'], rtol=0, atol=1e-6)
    return dict(
        model=model, role=role, case_id=manifest['case_id'],
        r2_seed_mean=float(manifest['selection']['r2_case_mean']),
        x=x, y=y, csv_path=csv_path, manifest_path=folder / 'manifest.json',
        n=int(len(x)), r2_raw=raw,
        r2_fit=float(fit['r2_linear_fit']),
        slope=float(fit['linear_fit_slope']),
        intercept=float(fit['linear_fit_intercept']),
        mae=mae,
    )


def format_intercept(value: float, unit: str) -> str:
    if unit == 'Pa' and abs(value) >= 10:
        return f'{value:.1f} {unit}'
    if unit == 'Pa':
        return f'{value:.3f} {unit}'
    return f'{value:.3f} {unit}'


def draw_panel(ax, rec: dict, spec: dict) -> None:
    dens = density_colors(rec['x'], rec['y'])
    order = np.argsort(dens)
    ax.scatter(
        rec['x'][order], rec['y'][order], c=dens[order], s=spec['point_size'],
        cmap='viridis', alpha=0.78, linewidths=0, rasterized=True, zorder=0,
    )
    lo = min(float(rec['x'].min()), float(rec['y'].min()))
    hi = max(float(rec['x'].max()), float(rec['y'].max()))
    pad = 0.04 * (hi - lo if hi > lo else 1.0)
    lo -= pad
    hi += pad
    grid = np.linspace(lo, hi, 2)
    ax.plot(grid, grid, color='0.45', lw=1.15, ls='--', zorder=3)
    ax.plot(grid, rec['slope'] * grid + rec['intercept'], color='#B45F5B', lw=1.85, zorder=4)
    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    ax.set_aspect('equal', adjustable='box')
    assert rec['x'].min() >= lo and rec['y'].min() >= lo
    assert rec['x'].max() <= hi and rec['y'].max() <= hi
    ax.set_xlabel(f"CFD {spec['field_cn']}（{spec['unit']}）")
    ax.set_ylabel(f"Pred {spec['field_cn']}（{spec['unit']}）")
    parts = rec['case_id'].split('/')
    short = parts[-2] if parts[0] == 'ILO' and len(parts) >= 3 else parts[-1]
    ax.set_title(
        f"{ROLE_CN[rec['role']]}  ·  {short}",
        loc='left', pad=12, color=ROLE_COLOR[rec['role']],
    )
    box = (
        f"n = {rec['n']:,} {spec['point_label']}\n"
        f"R² 预测 = {rec['r2_raw']:.3f}\n"
        f"R² 拟合 = {rec['r2_fit']:.3f}\n"
        f"a = {rec['slope']:.3f}\n"
        f"b = {format_intercept(rec['intercept'], spec['unit'])}"
    )
    ax.text(
        0.03, 0.97, box, transform=ax.transAxes, va='top', ha='left', fontsize=11,
        color=INK, linespacing=1.35,
        bbox=dict(boxstyle='round,pad=0.28', fc='white', ec=LINE, alpha=0.92),
        zorder=5,
    )
    ax.grid(color=LINE, alpha=.45, lw=.4)
    ax.set_axisbelow(True)
    ax.set_rasterization_zorder(1)
    ax.tick_params(axis='both', pad=4)


def write_csv(rows: list[dict], path: Path) -> None:
    path.parent.mkdir(exist_ok=True)
    fields = [
        'model', 'role', 'case_id', 'n_points', 'point_domain',
        'r2_prediction', 'r2_linear_fit', 'slope_a', 'intercept_b', 'mae',
        'r2_three_seed_mean_selection', 'r2_five_seed_mean_selection', 'unit', 'source_csv',
    ]
    with path.open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def draw_model(model: str, spec: dict) -> list[dict]:
    records = [load_case(model, role, spec) for role in ROLES]
    if model in ('VF6_wss', 'X5D_v51'):
        best, med, worst = records
        spec = dict(spec)
        spec['title'] = (
            f"{model} 最好例 a={best['slope']:.2f}，最差例 a={worst['slope']:.2f}"
        )
        spec['claim'] = (
            f"最好例 a={best['slope']:.3f}，预测/拟合 R² {best['r2_raw']:.3f} / {best['r2_fit']:.3f}；"
            f"最差例 a={worst['slope']:.3f}，预测/拟合 {worst['r2_raw']:.3f} / {worst['r2_fit']:.3f}"
        )
    elif str(model).startswith('V4_EMA'):
        best, med, worst = records
        spec = dict(spec)
        spec['claim'] = (
            f"最好例 a={best['slope']:.3f}，预测/拟合 R² {best['r2_raw']:.3f} / {best['r2_fit']:.3f}；"
            f"最差例 a={worst['slope']:.3f}，预测/拟合 {worst['r2_raw']:.3f} / {worst['r2_fit']:.3f}"
        )
    csv_rows = []
    sources = []
    for rec in records:
        csv_rows.append(dict(
            model=model, role=rec['role'], case_id=rec['case_id'],
            n_points=rec['n'], point_domain=spec['point_label'],
            r2_prediction=f"{rec['r2_raw']:.10f}",
            r2_linear_fit=f"{rec['r2_fit']:.10f}",
            slope_a=f"{rec['slope']:.10f}",
            intercept_b=f"{rec['intercept']:.10f}",
            mae=f"{rec['mae']:.10f}",
            r2_three_seed_mean_selection=f"{rec['r2_seed_mean']:.10f}" if model not in ('X5D_v51',) and not str(model).startswith('V4_EMA') else '',
            r2_five_seed_mean_selection=f"{rec['r2_seed_mean']:.10f}" if model == 'X5D_v51' else '',
            unit=spec['unit'],
            source_csv=str(rec['csv_path']),
        ))
        sources.extend([rec['csv_path'], rec['manifest_path']])
    summary_csv = OUT / 'source_data' / f"{spec['stem']}.csv"
    write_csv(csv_rows, summary_csv)
    fig = page(
        spec['title'],
        spec['subtitle'],
        spec['kicker'],
        f"{spec['footer_domain']}；{spec.get('footer_frame', 's1234 / peak 1162 同点，不是多种子平均场。')}"
        + spec.get(
            'footer_selection',
            '选例按三 seed 病例 R² 均值；Median 是距分布中位最近的真实病例。',
        )
        + '\nR² 预测 = 1 − SSE/SST（相对该例 CFD 均值）；R² 拟合 = pred = a·CFD + b 的回归 R²。'
        + '全部同点保留；颜色仅 6000 点 KDE 近似。高斯插值面未参与拟合。test34 已暴露。',
    )
    gs = fig.add_gridspec(1, 3, left=.055, right=.985, bottom=.175, top=.70, wspace=.22)
    axes = [fig.add_subplot(gs[0, i]) for i in range(3)]
    for ax, rec in zip(axes, records):
        draw_panel(ax, rec, spec)
    panel_label(axes[0], 'a')
    panel_label(axes[1], 'b')
    panel_label(axes[2], 'c')
    legend = [
        Line2D([0], [0], color='0.45', lw=1.15, ls='--', label='y = x'),
        Line2D([0], [0], color='#B45F5B', lw=1.85, label='pred = a·CFD + b'),
    ]
    fig.legend(
        legend, ['y = x', 'pred = a·CFD + b'],
        loc='upper center', bbox_to_anchor=(0.52, 0.785), ncol=2,
        fontsize=12, handlelength=2.2, borderaxespad=0,
    )
    save(
        fig, spec['stem'],
        claim=spec['claim'],
        sources=sources + [summary_csv],
        axes=axes,
        note=(
            'Three equal-width identity scatters. Axis limits are independent per case so that '
            'every same-point query is retained. Density color is KDE on a 6000-point sample; '
            'slope, intercept and both R² values use all points. Prediction R² matches the '
            'package visualization_metrics.r2.'
        ),
    )
    return csv_rows


def main() -> None:
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--models', nargs='+', choices=list(MODELS), default=list(MODELS))
    args = parser.parse_args()
    (OUT / 'source_data').mkdir(exist_ok=True)
    (OUT / 'figures').mkdir(exist_ok=True)
    (OUT / 'qa').mkdir(exist_ok=True)
    all_rows: list[dict] = []
    for model in args.models:
        all_rows.extend(draw_model(model, MODELS[model]))
    all_csv = OUT / 'source_data' / 'all_scatter_fit.csv'
    if all_csv.exists() and set(args.models) != set(MODELS):
        existing = [row for row in csv.DictReader(all_csv.open()) if row['model'] not in args.models]
        write_csv(existing + all_rows, all_csv)
    elif set(args.models) == set(MODELS):
        write_csv(all_rows, all_csv)


if __name__ == '__main__':
    main()

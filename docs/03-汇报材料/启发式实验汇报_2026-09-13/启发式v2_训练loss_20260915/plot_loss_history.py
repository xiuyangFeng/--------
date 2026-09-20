#!/usr/bin/env python3
"""Plot wave-2 PF6 / VF6 / X5X11 training loss from saved history.jsonl only."""
from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np
import pandas as pd

from plot_style import (
    BLUE, GREY, INK, LINE, ORANGE, OUT, PURPLE, RED, ROOT, TEAL,
    page, panel_label, save,
)

RUNS = ROOT / 'training_wss_min' / 'runs' / 'wss_local_wave2_20260912'
SMOOTH = 21
LOG_TICKS = [0.03, 0.05, 0.1, 0.2, 0.5, 1.0]


def load_run(model: str, seed: int) -> dict:
    run = RUNS / f'{model}_s{seed}'
    history_path = run / 'history.jsonl'
    config_path = run / 'config.json'
    records = [json.loads(line) for line in history_path.read_text().splitlines() if line.strip()]
    assert [row['epoch'] for row in records] == list(range(400)), history_path
    loss = np.array([row['train_loss'] for row in records], float)
    mse = np.array([row['train_mse_norm'] for row in records], float)
    assert np.isfinite(loss).all() and np.all(loss > 0)
    assert np.isfinite(mse).all() and np.all(mse > 0)
    best = int(loss.argmin())
    flagged = [i for i, row in enumerate(records) if row.get('is_best')]
    assert flagged and flagged[-1] == best
    cfg = json.loads(config_path.read_text())
    train = cfg['train']
    assert train['epochs'] == 400
    assert train['selection_rule'] == 'train_loss'
    assert train['ckpt_metric'] == 'train_loss'
    assert train['seed'] == seed
    best_metrics = json.loads((run / 'eval' / 'ckpt_best' / 'metrics.json').read_text())
    last_metrics = json.loads((run / 'eval' / 'ckpt_last' / 'metrics.json').read_text())
    r2_best = float(best_metrics['test']['field_casebalanced']['r2'])
    r2_last = float(last_metrics['test']['field_casebalanced']['r2'])
    components = records[-1].get('loss_components') or {}
    return dict(
        model=model, seed=seed, run=run, history_path=history_path, config_path=config_path,
        records=records, epoch=np.arange(400), loss=loss, mse=mse,
        smooth=pd.Series(loss).rolling(SMOOTH, center=True, min_periods=1).mean().to_numpy(),
        best=best, last_loss=float(loss[-1]), best_loss=float(loss[best]),
        pinball=float(train.get('loss_pinball_lambda') or 0.0),
        loss_name=train['loss'], r2_best=r2_best, r2_last=r2_last,
        component_keys=sorted(components),
    )


def write_csv(rows: list[dict], path: Path) -> None:
    path.parent.mkdir(exist_ok=True)
    fields = [
        'model', 'seed', 'epoch', 'train_loss', 'train_mse_norm', 'smooth21_train_loss',
        'lr_logged_after_step', 'is_selected_best', 'is_last', 'pinball_lambda',
        'test_r2_cb_ckpt_best', 'test_r2_cb_ckpt_last',
    ]
    with path.open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def rows_for(run: dict) -> list[dict]:
    out = []
    for i, rec in enumerate(run['records']):
        out.append(dict(
            model=run['model'], seed=run['seed'], epoch=int(rec['epoch']),
            train_loss=float(run['loss'][i]), train_mse_norm=float(run['mse'][i]),
            smooth21_train_loss=float(run['smooth'][i]),
            lr_logged_after_step=float(rec['lr']),
            is_selected_best=int(i == run['best']), is_last=int(i == 399),
            pinball_lambda=run['pinball'],
            test_r2_cb_ckpt_best=run['r2_best'], test_r2_cb_ckpt_last=run['r2_last'],
        ))
    return out


def style_loss_ax(ax, ymin, ymax):
    ax.set_yscale('log')
    ax.set_xlim(-8, 407)
    ax.set_xticks([0, 100, 200, 300, 399])
    ax.set_xlabel('Epoch')
    ticks = [v for v in LOG_TICKS if ymin < v < ymax]
    ax.set_yticks(ticks, [f'{v:g}' for v in ticks])
    ax.set_ylim(ymin, ymax)
    ax.grid(axis='y', color=LINE, alpha=.55, lw=.4)
    ax.tick_params(axis='x', pad=6)
    ax.set_axisbelow(True)
    ax.set_rasterization_zorder(1)


def draw_single(ax, run, color):
    ax.plot(run['epoch'], run['loss'], color=color, alpha=.22, lw=.9, rasterized=True, zorder=0,
            label='逐轮训练 loss')
    ax.plot(run['epoch'], run['smooth'], color=color, lw=2.15, zorder=2, label=f'{SMOOTH} 轮居中均值')
    ax.scatter(run['best'], run['best_loss'], marker='D', s=52, color=RED, zorder=5, clip_on=True,
               label='ckpt_best')
    ax.scatter(399, run['last_loss'], s=38, facecolors='white', edgecolors=GREY, linewidths=1.2,
               zorder=5, clip_on=True, label='ckpt_last')


def field_loss(run, *, stem, color, title, subtitle, kicker, claim, footer_target):
    assert run['pinball'] == 0.0
    assert run['component_keys'] == ['base', 'total']
    rows = rows_for(run)
    csv_path = OUT / 'source_data' / f'{stem}.csv'
    write_csv(rows, csv_path)
    ymin = float(run['loss'].min()) * 0.72
    ymax = float(run['loss'].max()) * 1.15
    fig = page(
        title,
        subtitle,
        kicker,
        f"{footer_target}；标准化 MSE，无 PINN 项。eval_every=400，无逐轮 val。"
        f"best epoch {run['best']} · {run['best_loss']:.4f}（last {run['last_loss']:.4f}；R²_cb {run['r2_best']:.3f}）。"
        '\n淡线保留全部尖峰；粗线仅辅助读下降趋势。单 seed 1234；test34 已暴露，R²_cb 未参与选模。',
    )
    gs = fig.add_gridspec(1, 1, left=.08, right=.97, bottom=.20, top=.70)
    ax = fig.add_subplot(gs[0, 0])
    draw_single(ax, run, color)
    style_loss_ax(ax, ymin, ymax)
    assert np.all(run['loss'] > ymin) and np.all(run['loss'] < ymax)
    ax.set_ylabel('训练 loss（标准化 MSE，对数轴）')
    ax.set_title(f"ckpt_best = epoch {run['best']}", loc='left', pad=14, color=color)
    handles, labels = ax.get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.5, 0.785), ncol=4,
               fontsize=12, handlelength=1.6, borderaxespad=0)
    save(
        fig, stem,
        claim=claim,
        sources=[run['history_path'], run['config_path'], csv_path],
        axes=[ax],
        note='All 400 logged epochs retained on a log axis fitted to this run. Centered 21-epoch mean is a visual aid only. Best is argmin(train_loss); no validation curve exists.',
    )


def pf6():
    run = load_run('PF6', 1234)
    field_loss(
        run,
        stem='PF6_loss_history',
        color=BLUE,
        title='PF6 相对压力训练 loss 总体下降，ckpt_best 在第 375 轮',
        subtitle='选模信号是在线 batch 均值训练 loss，不是验证/测试 R²；压力目标重尾，淡线尖峰全部保留。',
        kicker='wave-2 · 体场训练 · 压力 · seed 1234',
        claim='PF6 400 轮相对压力训练 loss 总体下降，交付 checkpoint 按最低训练 loss 选在末期而非 last',
        footer_target='监督壁面∪体内相对压力',
    )


def vf6():
    run = load_run('VF6', 1234)
    field_loss(
        run,
        stem='VF6_loss_history',
        color=TEAL,
        title='VF6 速度训练 loss 总体下降，ckpt_best 在第 374 轮',
        subtitle='选模信号是在线 batch 均值训练 loss，不是验证/测试 R²；淡线保留全部逐轮波动，粗线仅辅助读下降趋势。',
        kicker='wave-2 · 体场训练 · 速度 · seed 1234',
        claim='VF6 400 轮速度训练 loss 总体下降，交付 checkpoint 按最低训练 loss 选在末期而非 last',
        footer_target='监督体内速度三分量',
    )


def x5x11():
    runs = [load_run('X5X11', seed) for seed in (1234, 7, 2025)]
    for run in runs:
        assert run['pinball'] == 0.2
        assert run['component_keys'] == ['base', 'pinball', 'pinball_weighted', 'total']
    rows = []
    for run in runs:
        rows.extend(rows_for(run))
    csv_path = OUT / 'source_data' / 'X5X11_loss_history.csv'
    write_csv(rows, csv_path)
    colors = {1234: BLUE, 7: PURPLE, 2025: ORANGE}
    fig = page(
        'X5X11 三个 seed 的训练 loss 同步下降，ckpt_best 都在末期',
        '曲线为选模用的 total loss（base MSE + 0.2×pinball），不是纯 MSE；s1234 是后处理云图所用 seed。',
        'wave-2 · 直接 WSS · seed 1234 / 7 / 2025',
        '固定 v2 三 seed，与病例 R² 分布同一组 run。无逐轮 val。'
        + '  '.join(
            f"s{run['seed']} best {run['best']} · {run['best_loss']:.4f}"
            f"（last {run['last_loss']:.4f}；R²_cb {run['r2_best']:.3f}）"
            for run in runs
        )
        + '\ntest34 已暴露；R²_cb 未参与选模。三 seed 同向不能当作统计显著性。',
    )
    gs = fig.add_gridspec(1, 2, left=.08, right=.97, bottom=.20, top=.70, wspace=.28)
    ax = fig.add_subplot(gs[0, 0])
    bx = fig.add_subplot(gs[0, 1])
    legend_handles = []
    for run in runs:
        color = colors[run['seed']]
        lw = 2.3 if run['seed'] == 1234 else 1.7
        alpha_raw = .28 if run['seed'] == 1234 else .16
        ax.plot(run['epoch'], run['loss'], color=color, alpha=alpha_raw, lw=.8, rasterized=True, zorder=0)
        line, = ax.plot(run['epoch'], run['smooth'], color=color, lw=lw, zorder=0, rasterized=True, label=f"s{run['seed']}")
        legend_handles.append(line)
        bx.plot(run['epoch'], run['loss'], color=color, alpha=alpha_raw, lw=.9, rasterized=True, zorder=0)
        bx.plot(run['epoch'], run['smooth'], color=color, lw=lw, zorder=0, rasterized=True)
        ax.scatter(run['best'], run['best_loss'], marker='D', s=46, color=color, zorder=5,
                   edgecolors='white', linewidths=.6, clip_on=True)
        bx.scatter(run['best'], run['best_loss'], marker='D', s=54, color=color, zorder=5,
                   edgecolors='white', linewidths=.6, clip_on=True)
        bx.scatter(399, run['last_loss'], s=32, facecolors='white', edgecolors=color,
                   linewidths=1.15, zorder=4, clip_on=True)
        bx.annotate(
            f"s{run['seed']}",
            xy=(399, run['last_loss']),
            xytext=(8, {1234: -11, 7: 1, 2025: 13}[run['seed']]),
            textcoords='offset points',
            color=color, fontsize=11, va='center', clip_on=True,
        )
    style_loss_ax(ax, 0.024, 1.25)
    ax.set_rasterization_zorder(1)
    tail = np.concatenate([run['loss'][300:] for run in runs])
    bx.set_xlim(299, 422)
    bx.set_xticks([300, 340, 379, 399])
    bx.set_xlabel('Epoch')
    bx.set_ylim(float(tail.min()) * 0.985, float(tail.max()) * 1.04)
    assert np.all(tail < bx.get_ylim()[1]) and np.all(tail > bx.get_ylim()[0])
    bx.grid(axis='y', color=LINE, alpha=.55, lw=.4)
    bx.tick_params(axis='x', pad=6)
    bx.set_axisbelow(True)
    bx.set_rasterization_zorder(1)
    ax.set_ylabel('训练 loss（total，对数轴）')
    bx.set_ylabel('训练 loss（total，线性轴）')
    ax.set_title('全程：三 seed 形态一致', loc='left', pad=14, color=INK)
    bx.set_title('末 100 轮：best 在 379–393', loc='left', pad=14, color=INK)
    fig.legend(legend_handles, [f"s{seed}" for seed in (1234, 7, 2025)],
               loc='upper left', bbox_to_anchor=(0.30, 0.785), ncol=3,
               fontsize=12, handlelength=1.8, borderaxespad=0, title='21 轮居中均值', title_fontsize=12)
    panel_label(ax, 'a')
    panel_label(bx, 'b')
    save(
        fig, 'X5X11_loss_history',
        claim='X5X11 三 seed 训练 loss 同步下降，交付 checkpoint 取末期最低 total loss 而非 last',
        sources=[run['history_path'] for run in runs] + [run['config_path'] for run in runs] + [csv_path],
        axes=[ax, bx],
        note='All 400 epochs retained. Right panel is a linear zoom of epochs 300-399; y-limits include every recorded point in that window. Diamonds mark argmin(train_loss) per seed.',
    )


def main():
    (OUT / 'source_data').mkdir(exist_ok=True)
    (OUT / 'figures').mkdir(exist_ok=True)
    (OUT / 'qa').mkdir(exist_ok=True)
    pf6()
    vf6()
    x5x11()


if __name__ == '__main__':
    main()

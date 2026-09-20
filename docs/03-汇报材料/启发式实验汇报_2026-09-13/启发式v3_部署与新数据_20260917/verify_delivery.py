"""交付核验：图件齐全 / 尺寸一致 / 对齐审计通过 / 提纲里的关键数字能回到原始产物。

用法：/public/newhome/cy/.conda/envs/GNN/bin/python verify_delivery.py
任何一条 FAIL 都会以退出码 1 结束，并把失败项逐条打印。
"""
from pathlib import Path
import json
import struct
import sys

OUT = Path(__file__).resolve().parent
OUTLINE = OUT / '启发式汇报提纲_v3_2026-09-17.md'
STEMS = ['01_deployment_decision', '02_density_curve', '03_geometry_fidelity',
         '04_rcr_audit', '05_data_v50_to_v51', '06_x5d_v51_baseline',
         '07_three_arms_paired', '08_thinning_reeval', '09_stl_and_noise_reeval',
         '10_longitudinal_paired', '11_longitudinal_why', '12_residual_structure',
         '13_ilo_and_gate', '14_residual_scale_ladder', '15_levers_used_vs_unused',
         '16_future_roadmap', '17_prune_ladder']
SIZE = (4800, 2700)

# 提纲里出现的关键数字 → 它在 source_data 里的取值路径与显示格式。
# 目的不是覆盖每个数字，而是把"页面上最容易被引用的读数"钉回原始产物。
CONTRACT = [
    ('0.7749', '06_x5d_v51_baseline', 'test34_ensemble_5seed', '{:.4f}'),
    ('0.7100', '06_x5d_v51_baseline', 'cv3_pooled', '{:.4f}'),
    ('0.6239', '13_ilo_and_gate', 'ilo_oof_r2', '{:.4f}'),
    ('0.7844', '13_ilo_and_gate', 'gate.gate8.pa_r2_cb', '{:.4f}'),
    ('0.7736', '13_ilo_and_gate', 'gate.base.pa_r2_cb', '{:.4f}'),
    ('0.7789', '10_longitudinal_paired', 'ensemble_best.long3.pa_r2_cb', '{:.4f}'),
    ('+0.0049', '10_longitudinal_paired', 'paired_best.mean_d_pa', '{:+.4f}'),
    ('0.0143', '10_longitudinal_paired', 'paired_best.sd_d_pa', '{:.4f}'),
    ('0.7465', '05_data_v50_to_v51', 'five_seed.old_weights_old_labels', '{:.4f}'),
    ('0.7652', '05_data_v50_to_v51', 'five_seed.old_weights_new_labels', '{:.4f}'),
    ('39', '04_rcr_audit', 'n_bad_outlets', '{:d}'),
    ('23', '04_rcr_audit', 'n_bad_cases', '{:d}'),
    ('26', '04_rcr_audit', 'n_rerun_cases', '{:d}'),
    ('0.00473', '15_levers_used_vs_unused', 'ceiling.oof_forward_all16', '{:.5f}'),
    ('0.00435', '15_levers_used_vs_unused', 'ceiling.oof_forward_ref8', '{:.5f}'),
    ('0.0059', '15_levers_used_vs_unused',
     'ceiling.all16_on_base_residual_pooled', '{:.4f}'),
    ('69.66', '12_residual_structure',
     'test34_base5.residual_medians.within_bin_share', '{:.2%}'),
    ('70.13', '12_residual_structure',
     'cv3_oof.residual_medians.within_bin_share', '{:.2%}'),
    ('0.5265', '12_residual_structure',
     'test34_base5.high_wss_case_p90_then_pool_r2', '{:.4f}'),
    ('0.4022', '12_residual_structure',
     'cv3_oof.high_wss_case_p90_then_pool_r2', '{:.4f}'),
    ('+0.0060', '17_prune_ladder', 'ladder.L1_s1234.d_norm', '{:+.4f}'),
    ('+0.0065', '17_prune_ladder', 'ladder.X5D_long 全32列.d_norm', '{:+.4f}'),
    ('0.788', '17_prune_ladder', 'criteria.coverage_L1', '{:.3f}'),
    ('0.651', '17_prune_ladder', 'criteria.coverage_all32', '{:.3f}'),
    ('0.178', '17_prune_ladder', 'criteria.predictability_roundness', '{:.3f}'),
    ('0.726', '17_prune_ladder', 'criteria.ref_pc_corr_roundness', '{:.3f}'),
]


def png_size(path):
    head = path.read_bytes()[:24]
    if head[:8] != b'\x89PNG\r\n\x1a\n':
        raise ValueError(f'{path} is not a PNG')
    return struct.unpack('>II', head[16:24])


def dig(obj, dotted):
    for part in dotted.split('.'):
        obj = obj[part]
    return obj


def main():
    fails, checks = [], 0
    outline = OUTLINE.read_text() if OUTLINE.exists() else ''
    if not outline:
        fails.append(f'缺少提纲 {OUTLINE.name}')

    for stem in STEMS:
        png = OUT / 'figures' / f'{stem}.png'
        align = OUT / 'qa' / f'{stem}.alignment.json'
        meta = OUT / 'qa' / f'{stem}.metadata.json'
        data = OUT / 'source_data' / f'{stem}.json'
        for p in (png, align, meta, data):
            checks += 1
            if not p.exists():
                fails.append(f'缺少 {p.relative_to(OUT)}')
        if png.exists():
            checks += 1
            size = png_size(png)
            if size != SIZE:
                fails.append(f'{stem}.png 尺寸 {size}，应为 {SIZE}')
        if align.exists():
            checks += 1
            report = json.loads(align.read_text())
            verdict = report.get('verdict', report.get('status'))
            if verdict not in ('PASS', 'NOT APPLICABLE'):
                fails.append(f'{stem} 对齐审计 = {verdict}')
        if meta.exists():
            checks += 1
            sources = json.loads(meta.read_text()).get('sources', [])
            if not sources:
                fails.append(f'{stem} 没有记录任何来源文件')
            for s in sources:
                if not Path(s['path']).exists():
                    fails.append(f'{stem} 的来源文件已不存在：{s["path"]}')
        if outline and f'figures/{stem}.png' not in outline and stem != '16_future_roadmap':
            # 16 号是路线图，允许只在 P35 通过别名引用
            fails.append(f'提纲里没有引用 figures/{stem}.png')

    for shown, stem, path, fmt in CONTRACT:
        checks += 1
        data = OUT / 'source_data' / f'{stem}.json'
        if not data.exists():
            fails.append(f'契约 {shown}：缺少 {stem}.json')
            continue
        try:
            value = dig(json.loads(data.read_text())['values'], path)
        except (KeyError, TypeError):
            fails.append(f'契约 {shown}：{stem}.json 里找不到 {path}')
            continue
        rendered = fmt.format(value)
        if fmt.endswith('%}'):
            rendered = rendered.rstrip('%')
        if rendered != shown:
            fails.append(f'契约不符：提纲写 {shown}，{stem}.json/{path} = {rendered}')
        if outline and shown not in outline:
            fails.append(f'契约 {shown} 没有出现在提纲里（路径 {stem}/{path}）')

    print(f'检查 {checks} 项，失败 {len(fails)} 项')
    for f in fails:
        print('  FAIL', f)
    return 1 if fails else 0


if __name__ == '__main__':
    sys.exit(main())

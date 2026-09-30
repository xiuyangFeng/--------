"""P0b 训练折标定（零训练）。预注册见 PREREG.md §P0b。

底座：ALL3 = O1+O2+M1 等权（主）、M1（已部署模型族，次）。留出折 k 的标定只用其余两折的折外预测与真值拟合（不碰留出折真值）；
test34 的标定用全部 136 例 cv3 折外拟合，套到 test34 对应底座（ALL3 → ALL9，M1 → M1_ens3）。
候选：
- 数值：none / iso（等渗 y ~ ŷ）/ qmap（分位映射 F_true⁻¹(F_pred(ŷ))，病例等点抽样后池化）
- 掩膜：none / qmap / thr_iou（每个 T 在训练折上取使逐例 IoU 均值最大的预测阈值 t'）/ thr_area（使训练折平均预测面积份额 = 真值份额的 t'）
选择规则（只在 cv3 上）：数值 = CCC 中位最高且 R²_cb 不低于 none − 0.01；掩膜 = IoU@0.3 最高；并列取 none。test34 只报 none 与选中项。
已知偏差：test34 底座是 3/9 模型集成，比 cv3 单 seed 折外更平滑，按单 seed 分布拟合的映射在 test34 上可能拉伸不足。
"""
from __future__ import annotations

import time

import numpy as np
from sklearn.isotonic import IsotonicRegression

from p0_common import (FOLD_HELD, OUT, OSI_THR, T34_ENSEMBLES, cv_bundle, t34_bundle, cv_ensemble, osi_summary,
                       fold_mean, flat, table, write_json)

N_FIT = 4000
GRID = np.round(np.arange(0.02, 0.4501, 0.005), 4)
rng = np.random.default_rng(20260924)
BASES = {'ALL3': (('O1', 'O2', 'M1'), 'ALL9'), 'M1': (('M1',), 'M1_ens3')}


def sample(cases, y, p):
    ys, ps = [], []
    for c in cases:
        n = len(y[c]); pick = rng.choice(n, min(N_FIT, n), replace=False)
        ys.append(y[c][pick]); ps.append(p[c][pick])
    return ys, ps


def fit(ys, ps):
    yc, pc = np.concatenate(ys), np.concatenate(ps)
    iso = IsotonicRegression(increasing=True, out_of_bounds='clip', y_min=0.0, y_max=0.5).fit(pc, yc)
    qs = np.linspace(0, 1, 2001)
    pq = np.quantile(pc, qs); yq = np.quantile(yc, qs)
    pq = pq + np.arange(len(pq)) * 1e-12          # 严格递增供 interp
    thr_iou, thr_area = {}, {}
    for T in OSI_THR:
        ious = []
        for t in GRID:
            v = []
            for y, p in zip(ys, ps):
                tm, pm = y > T, p > t
                u = np.count_nonzero(tm | pm)
                if u:
                    v.append(np.count_nonzero(tm & pm) / u)
            ious.append(np.mean(v))
        thr_iou[T] = float(GRID[int(np.argmax(ious))])
        true_frac = np.mean([(y > T).mean() for y in ys])
        pred_frac = np.array([np.mean([(p > t).mean() for p in ps]) for t in GRID])
        thr_area[T] = float(GRID[int(np.argmin(np.abs(pred_frac - true_frac)))])
    return {'iso': iso, 'qmap': (pq, yq), 'thr_iou': thr_iou, 'thr_area': thr_area}


def apply_all(cal, cases, osi_t, base_p, ta_t, ta_p):
    qm = lambda x: np.clip(np.interp(x, *cal['qmap']), 0.0, 0.5)
    out = {'none': osi_summary(cases, osi_t, base_p, ta_t, ta_p),
           'iso': osi_summary(cases, osi_t, [cal['iso'].predict(p) for p in base_p], ta_t, ta_p),
           'qmap': osi_summary(cases, osi_t, [qm(p) for p in base_p], ta_t, ta_p)}
    for m in ('thr_iou', 'thr_area'):
        out[m] = osi_summary(cases, osi_t, base_p, ta_t, ta_p, thr_override=cal[m], stag_osi_thr=cal[m][0.1])
    return out


def main():
    t0 = time.time()
    cv = cv_bundle()
    y = {c: cv[c]['osi'] for c in cv}
    res, fits = {}, {}
    for bname, (members, _) in BASES.items():
        p = {c: cv_ensemble(cv[c], members) for c in cv}
        per_method = {}
        fits[bname] = {}
        for k in range(3):
            train = [c for kk in range(3) if kk != k for c in FOLD_HELD[kk]]
            cal = fit(*sample(train, y, p))
            fits[bname][f'fold{k}'] = {'thr_iou': cal['thr_iou'], 'thr_area': cal['thr_area']}
            held = FOLD_HELD[k]
            r = apply_all(cal, held, [y[c] for c in held], [p[c] for c in held], [cv[c]['tawss'] for c in held], [cv[c]['M1_tawss'] for c in held])
            for m, s in r.items():
                per_method.setdefault(m, []).append(s)
        res[bname] = {m: {'per_fold': v, 'fold_mean': fold_mean(v)} for m, v in per_method.items()}
        print(f'{bname} cv3 done {time.time() - t0:.0f}s', flush=True)

    # 选择（只看 cv3）
    selected = {}
    for bname in BASES:
        fm = {m: res[bname][m]['fold_mean'] for m in res[bname]}
        ok = [m for m in ('none', 'iso', 'qmap') if fm[m]['r2_cb'] >= fm['none']['r2_cb'] - 0.01]
        val = max(ok, key=lambda m: (round(fm[m]['ccc_casemed'], 6), m == 'none'))
        mask = max(('none', 'qmap', 'thr_iou', 'thr_area'), key=lambda m: (round(fm[m]['iou_0.3'], 6), m == 'none'))
        selected[bname] = {'value': val, 'mask': mask}

    # test34：用全部 136 例 cv3 折外拟合
    t34 = t34_bundle()
    cases = list(t34)
    ot = [t34[c]['osi'] for c in cases]; tt = [t34[c]['tawss'] for c in cases]; tp = [t34[c]['M1_tawss'] for c in cases]
    res_t34 = {}
    for bname, (members, t34name) in BASES.items():
        p = {c: cv_ensemble(cv[c], members) for c in cv}
        cal = fit(*sample(list(cv), y, p))
        fits[bname]['train136'] = {'thr_iou': cal['thr_iou'], 'thr_area': cal['thr_area']}
        bp = [cv_ensemble(t34[c], T34_ENSEMBLES[t34name]) for c in cases]
        allr = apply_all(cal, cases, ot, bp, tt, tp)
        keep = {'none', selected[bname]['value'], selected[bname]['mask']}
        res_t34[bname] = {m: allr[m] for m in keep}

    text = ['=== P0b 训练折标定（OSI，零训练）===']
    for bname in BASES:
        text += ['', f'[cv3 底座 {bname}，三折留出均值；标定只用其余两折]',
                 table([(m, res[bname][m]['fold_mean']) for m in res[bname]], ref='none'),
                 f"  逐折预测阈值 thr_iou: {[fits[bname][f'fold{k}']['thr_iou'] for k in range(3)]}",
                 f"  逐折预测阈值 thr_area: {[fits[bname][f'fold{k}']['thr_area'] for k in range(3)]}",
                 f"  选中：数值 {selected[bname]['value']}，掩膜 {selected[bname]['mask']}"]
    for bname, (_, t34name) in BASES.items():
        text += ['', f'[test34 读一次：底座 {t34name}，标定用 train136 cv3 折外；只报 none 与选中项]',
                 table([(m, flat(res_t34[bname][m])) for m in res_t34[bname]], ref='none'),
                 f"  train136 拟合阈值 thr_iou {fits[bname]['train136']['thr_iou']} / thr_area {fits[bname]['train136']['thr_area']}"]
    text.append(f'\n耗时 {time.time() - t0:.0f}s')
    print('\n'.join(text))
    (OUT / 'p0b_calibration.txt').write_text('\n'.join(text) + '\n')
    write_json(OUT / 'p0b_calibration.json', {'cv3': res, 'selected_on_cv3': selected, 'thresholds': fits, 'test34': res_t34})


if __name__ == '__main__':
    main()

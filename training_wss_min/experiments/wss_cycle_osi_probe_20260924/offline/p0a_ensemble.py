"""P0a 异构集成（零训练，只读已保存预测）。预注册见 PREREG.md §P0a。

cv3 折外（单 seed）：M1（参照 = 已部署模型族）、O1、O2、两两组合、ALL3 = O1+O2+M1（主臂）；三折留出均值。
test34（读一次）：M1_ens3（参照 = 已部署）、O1_ens3、O2_ens3、O1O2_ens6、ALL9（主臂）；
另报「等成员数」对照：27 个 (O1_si, O2_sj, M1_sk) 异构三模型组合的均值 vs 同构三 seed 集成，区分「异构」与「成员更多」。
滞留区统一用 M1 的 TAWSS 通道（cv3：同折 M1；test34：M1 三 seed 均值），只让 OSI 变。
锚点：必须复现矩阵已报数字（O1 cv3 三折 0.469/0.511/0.470；M1 OSI 头 0.4572/0.4838/0.4539；test34 M1_ens3 0.5648、O1_ens3 0.5728、O2_ens3 0.5543）。
"""
from __future__ import annotations

import itertools
import time

import numpy as np

from p0_common import (FOLD_HELD, OUT, SEEDS3, CV_ENSEMBLES, T34_ENSEMBLES, cv_bundle, t34_bundle, cv_ensemble,
                       osi_summary, fold_mean, flat, table, write_json)

ANCHORS_CV = {'O1': (0.469, 0.511, 0.470), 'M1': (0.4572, 0.4838, 0.4539)}
ANCHORS_T34 = {'M1_ens3': 0.5648, 'O1_ens3': 0.5728, 'O2_ens3': 0.5543}


def main():
    t0 = time.time()
    cv = cv_bundle()
    print(f'cv3 bundle {len(cv)} cases {time.time() - t0:.0f}s', flush=True)
    res_cv = {}
    for name, members in CV_ENSEMBLES.items():
        per_fold = []
        for k in range(3):
            cases = FOLD_HELD[k]
            per_fold.append(osi_summary(cases, [cv[c]['osi'] for c in cases], [cv_ensemble(cv[c], members) for c in cases],
                                        [cv[c]['tawss'] for c in cases], [cv[c]['M1_tawss'] for c in cases]))
        res_cv[name] = {'per_fold': per_fold, 'fold_mean': fold_mean(per_fold)}
    for name, exp in ANCHORS_CV.items():
        got = res_cv[name]['fold_mean']['r2_cb_folds']
        if not np.allclose(got, exp, atol=2e-3):
            raise RuntimeError(f'anchor {name} cv3 {got} != {exp}')

    t34 = t34_bundle()
    print(f'test34 bundle {len(t34)} cases {time.time() - t0:.0f}s', flush=True)
    cases = list(t34)
    ot = [t34[c]['osi'] for c in cases]; tt = [t34[c]['tawss'] for c in cases]; tp = [t34[c]['M1_tawss'] for c in cases]
    res_t34 = {name: osi_summary(cases, ot, [cv_ensemble(t34[c], m) for c in cases], tt, tp) for name, m in T34_ENSEMBLES.items()}
    for name, exp in ANCHORS_T34.items():
        if abs(res_t34[name]['r2_cb'] - exp) > 2e-3:
            raise RuntimeError(f'anchor {name} test34 {res_t34[name]["r2_cb"]} != {exp}')

    # 等成员数对照：异构三模型 27 组合 vs 同构三 seed
    hetero = [flat(osi_summary(cases, ot, [cv_ensemble(t34[c], (f'O1_s{a}', f'O2_s{b}', f'M1_s{m}')) for c in cases], tt, tp))
              for a, b, m in itertools.product(SEEDS3, SEEDS3, SEEDS3)]
    keys = list(hetero[0])
    hetero_mean = {k: float(np.mean([h[k] for h in hetero])) for k in keys}
    hetero_sd = {k: float(np.std([h[k] for h in hetero], ddof=1)) for k in keys}
    homo = {n: flat(res_t34[n]) for n in ('M1_ens3', 'O1_ens3', 'O2_ens3')}
    homo_mean = {k: float(np.mean([homo[n][k] for n in homo])) for k in keys}

    rows_cv = [(n, res_cv[n]['fold_mean']) for n in CV_ENSEMBLES]
    rows_t34 = [(n, flat(res_t34[n])) for n in T34_ENSEMBLES]
    rows_eq = [('hetero3 (27 组合均值)', hetero_mean), ('hetero3 组合间 sd', hetero_sd), ('homo3 (三个同构 ens3 均值)', homo_mean)]
    text = ['=== P0a 异构集成（OSI，零训练）===', '',
            '[cv3 折外，三折留出均值；参照 M1 单 seed；滞留区 TAWSS = 同折 M1 通道]', table(rows_cv, ref='M1'), '',
            '[test34 读一次；参照 M1_ens3 = 已部署；滞留区 TAWSS = M1 三 seed 均值]', table(rows_t34, ref='M1_ens3'), '',
            '[test34 等成员数：异构三模型 vs 同构三 seed]', table(rows_eq), '',
            '分队列 R²_cb（cv3 三折均值 / test34）：']
    for n in ('M1', 'ALL3'):
        text.append(f'  cv3 {n:6s} ' + ' '.join(f"{g} {np.mean([f['r2_cb_by_cohort'][g] for f in res_cv[n]['per_fold']]):.3f}" for g in ('AG', 'AAA', 'ILO')))
    for n in ('M1_ens3', 'ALL9'):
        text.append(f'  t34 {n:8s} ' + ' '.join(f"{g} {res_t34[n]['r2_cb_by_cohort'][g]:.3f}" for g in ('AG', 'AAA', 'ILO')))
    text.append(f'\n锚点复现通过（cv3 O1/M1 逐折、test34 三个 ens3）；耗时 {time.time() - t0:.0f}s')
    print('\n'.join(text))
    (OUT / 'p0a_ensemble.txt').write_text('\n'.join(text) + '\n')
    write_json(OUT / 'p0a_ensemble.json', {'cv3': res_cv, 'test34': res_t34, 'test34_equal_members': {
        'hetero3_mean': hetero_mean, 'hetero3_sd': hetero_sd, 'homo3_mean': homo_mean, 'homo3': homo}})


if __name__ == '__main__':
    main()

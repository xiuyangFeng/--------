"""体场全周期阶段 0 公共读取与指标（只读）。

与 WSS 时间线（`wss_time_ecc_20260918/offline/common.py`）同结构，差别只在目标空间：
体场目标是**线性** z（压力 Pa、速度逐分量 m/s），不是 log_z，因此归一化、去归一化与
「Pa 空间指标」都是仿射变换，凡是 WSS 线要 exp/ln 的地方这里都不需要。

数据来源
- 几何与峰值标签：v5.1 视图 `views_v5_1/wss_min_view_v1/<cid>/volume.npz`（`wss_v5.views.volume_view` v1.1）
- 81 帧体场标签：快照 `case.h5` 的 `volume_temporal/{pressure_pa,velocity_m_s}`（逐帧 lzf 分块，懒读）
- 逐帧压力参考：`pressure_reference/p_volume_mean_pa` (81,)

压力目标的逐帧参考（合同）
    p_rel(x, t) = p(x, t) − p_volume_mean(t)
即把峰值帧视图的「相对该帧体积平均压」定义原样延伸到每一帧。**不用峰值帧的单一 p_ref**：
整周期的体积平均压摆幅是由入口波形驱动的全局 DC 项，172 例逐位相同，模型不看几何也能背出来，
用它当标签会把周期 R² 抬成伪高分。A0 审计里报出这个摆幅的实际大小作为证据。
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[4]))
# 子采样规则与周期指标定义从部署侧共享模块取，保证「上限/T-null」与「模型评估」同点集、同口径
from training_wss_min.volume_time import (  # noqa: E402,F401
    CaseBalanced, N_FRAMES, PEAK_INDEX, PEAK_STEP, SUBSAMPLE_CELLS, SUBSAMPLE_SEED, SUBSAMPLE_WALL,
    VolumeTimeMetrics, fmt_summary, subsample_rows as _subsample_rows, trough_frames,
)

ROOT = Path('/public/newhome/cy/Digital_twin/GNN')
VIEW = ROOT / 'data_wss_v5/views_v5_1/wss_min_view_v1'
H5 = ROOT / 'data_wss_v5/anatomy_pointcloud_v5_1_20260916/cases'
OUT = Path(__file__).resolve().parent
SCAN = OUT / 'scan'

TARGETS = ('pressure', 'velocity')
VELOCITY_COMPONENTS = ('u', 'v', 'w')

split = json.loads((VIEW / 'split_V5_train136_test34.json').read_text())
TRAIN, TEST = list(split['train_cases']), list(split['test_cases'])
ALL_CASES = TRAIN + TEST
cv3 = json.loads((VIEW / 'cv3_v51/cv3_summary.json').read_text())
FOLD_OF = {k: int(v) for k, v in cv3['fold_of'].items()}
FOLD_TRAIN = {f: [c for c in TRAIN if FOLD_OF[c] != f] for f in range(3)}
FOLD_HELD = {f: [c for c in TRAIN if FOLD_OF[c] == f] for f in range(3)}
SCOPES = {'train136': TRAIN, 'fold0': FOLD_TRAIN[0], 'fold1': FOLD_TRAIN[1], 'fold2': FOLD_TRAIN[2]}


def h5_path(cid: str) -> Path:
    return H5 / cid.replace('/', '__') / 'case.h5'


def volume_npz(cid: str) -> Path:
    return VIEW / cid / 'volume.npz'


def subsample_rows(n: int, k: int, cid: str) -> np.ndarray:
    """逐例固定子采样行号；实现在 training_wss_min.volume_time，离线与部署侧共用同一规则。"""
    return _subsample_rows(n, k, cid)


def load_stats(scope: str) -> dict:
    """A1 产出：volume_stats_<scope>.json（scope = train136 | fold0 | fold1 | fold2）。"""
    return json.loads((OUT / f'volume_stats_{scope}.json').read_text())


def frame_stats(scope: str, target: str) -> tuple[np.ndarray, np.ndarray, dict]:
    """逐帧 (mean, std)：压力 → (81,)（wall∪interior）；速度 → (3, 81)（内部逐分量）。"""
    d = load_stats(scope)
    f = d['frame']
    if target == 'pressure':
        return np.asarray(f['linear_mean'], np.float64), np.asarray(f['linear_std'], np.float64), d
    if target == 'velocity':
        return (np.asarray(f['velocity_mean'], np.float64).T, np.asarray(f['velocity_std'], np.float64).T, d)
    raise ValueError(target)


def write_json(path, obj) -> None:
    Path(path).write_text(json.dumps(obj, indent=1, ensure_ascii=False,
                                     default=lambda o: o.tolist() if hasattr(o, 'tolist') else str(o)))


# ----------------------------------------------------------------------------- 逐例时间矩
# 一次遍历存下每例每个通道的 (M, s, n)：M = Σ_x y(x,:) y(x,:)^T (81,81)，s = Σ_x y(x,:) (81,)。
# 逐帧均值/标准差、时间基 PCA、任意 K 的投影残差都能由它们精确推出，无需再读 78 GB 原始帧。

def moments_to_stats(M: np.ndarray, s: np.ndarray, n: int) -> tuple[np.ndarray, np.ndarray]:
    """(M, s, n) → 逐帧均值与标准差。"""
    mean = s / float(n)
    var = np.maximum(np.diag(M) / float(n) - mean ** 2, 0.0)
    return mean, np.sqrt(var)


def gram_of_normalized(M: np.ndarray, s: np.ndarray, n: int,
                       mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    """原始时间矩 → 逐帧标准化后 z 的 Gram 矩阵 Σ_x z(x,:) z(x,:)^T。

    z_k = (y_k − mean_k) / std_k ⇒
    G_kl = [M_kl − mean_k s_l − mean_l s_k + n·mean_k·mean_l] / (std_k·std_l)
    """
    inv = 1.0 / np.maximum(std, 1e-12)
    G = M - np.outer(mean, s) - np.outer(s, mean) + float(n) * np.outer(mean, mean)
    return G * np.outer(inv, inv)


class ChannelMoments:
    """把若干病例的 (M, s, n) 累加成一个分区的时间矩。"""

    def __init__(self, n_frames: int = N_FRAMES):
        self.M = np.zeros((n_frames, n_frames), dtype=np.float64)
        self.s = np.zeros(n_frames, dtype=np.float64)
        self.n = 0

    def add(self, M: np.ndarray, s: np.ndarray, n: int) -> 'ChannelMoments':
        self.M += M
        self.s += s
        self.n += int(n)
        return self

    def stats(self) -> tuple[np.ndarray, np.ndarray]:
        return moments_to_stats(self.M, self.s, self.n)

    def gram(self, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
        return gram_of_normalized(self.M, self.s, self.n, mean, std)


def centering_matrix(n_frames: int = N_FRAMES) -> np.ndarray:
    """H = I − 11ᵀ/T：去逐点时间均值 b0(x) = mean_k z(x, k) 的投影（与 time_basis.project 一致）。"""
    return np.eye(n_frames) - np.ones((n_frames, n_frames)) / float(n_frames)


def time_basis_from_gram(G: np.ndarray, n: int) -> tuple[np.ndarray, np.ndarray]:
    """由标准化 z 的 Gram 矩阵求时间基，口径与 WSS 线 d1_time_basis_ceiling.fit_basis 逐位一致：

    基拟合在**去逐点时间均值**后的 zc = H z 上（H = I − 11ᵀ/T），即 C = Σ_x zc zcᵀ / n = H G H / n，
    特征值降序、Φ 列正交归一、符号取「峰值帧分量为正」。b0(x) 由时间基头的第 0 个通道单独承担。

    返回 (特征值降序 (T,), Φ (T, T))。
    """
    H = centering_matrix(len(G))
    w, V = np.linalg.eigh(H @ G @ H / float(n))
    order = np.argsort(w)[::-1]
    w, V = w[order], V[:, order]
    V = V * np.where(V[PEAK_INDEX] < 0, -1.0, 1.0)
    return w, V


# ----------------------------------------------------------------------------- 指标
# CaseBalanced / VolumeTimeMetrics / fmt_summary / trough_frames 见 training_wss_min.volume_time

def load_waveform():
    wf = ROOT / 'training_wss_min/experiments/wss_time_ecc_20260918/offline/protocol_inlet_waveform_v51.json'
    return json.loads(wf.read_text())

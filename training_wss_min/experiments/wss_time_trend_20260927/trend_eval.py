#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""全周期「趋势」指标零训练复评（2026-09-27）。

对已训练好的时间臂在 v5.1 cv3 三折留出病例（136 例）上重建 81 帧 WSS 预测：
  1. 先按原口径复现已存主指标（cycle / trough / peak / TAWSS Pa R²_cb、峰时误差、ln 时序相关、幅值误差）做对账；
  2. 再算「趋势」指标（老师口径：幅值不准可以接受，重点看随时间的变化）：
     逐帧空间 Spearman、低 WSS（0.4 Pa）面积占比曲线与位置 Dice、最低 20 % / 最高 10 % 区位置 Dice、
     全壁面积加权平均 WSS 曲线、逐点归一化波形误差、Pa 时序相关、谷时误差、按真值同步性分层的时序相关。
不训练、不改任何权重；CFD 真值只作标签与分层（残差分析），不作任何模型输入。

子命令：
  metrics --arm A --fold f   → out/per_case/<A>_f<f>.jsonl + out/fold_summary/<A>_f<f>.json
  export  --arm A --cases …  → out/export/<case_key>/<A>.npz（动画用 80 帧 float16 预测）
  truth   --cases …          → out/export/<case_key>/truth.npz（真值、坐标、法向、面积、波形）
"""
from __future__ import annotations

import argparse
import json
import os
import socket
import sys
import time
from pathlib import Path
from typing import Dict, Iterator, List, Sequence, Tuple

import numpy as np
import torch
from scipy.stats import rankdata

REPO = Path("/public/newhome/cy/Digital_twin/GNN")
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from training_wss_min import config as C  # noqa: E402
from training_wss_min import dataset as D  # noqa: E402
from training_wss_min import evaluate as E  # noqa: E402
from training_wss_min import time_adapter_probe as TA  # noqa: E402
from training_wss_min import time_transformer as TT  # noqa: E402

EXP = Path(__file__).resolve().parent
RUNS = REPO / "training_wss_min/runs"
VIEW = REPO / "data_wss_v5/views_v5_1/wss_min_view_v1"
H5 = REPO / "data_wss_v5/anatomy_pointcloud_v5_1_20260916/cases"
WAVEFORM = REPO / "training_wss_min/experiments/wss_time_ecc_20260918/offline/protocol_inlet_waveform_v51.json"

FLOOR, EPS = 0.05, 1e-6
N_FRAMES, PEAK_INDEX, PERIOD = 81, 21, 80       # 帧 0–79 = 一个 0.8 s 周期；帧 80 = 帧 0 + T
LOW_WSS_PA = 0.4                               # 与部署页 / 周期量合同的低 WSS 阈值一致
LOW_Q, HIGH_Q = 0.20, 0.10                     # 相对阈值：每帧最低 20 % / 最高 10 %（与幅值无关）
SYNC_HI, SYNC_LO = 0.9, 0.5                    # 真值逐点 ln 波形与本例全壁平均 ln 波形的相关分层

# 相位段（帧号，0–79）；峰值窗 = 步 1154–1172、谷底 = 协议 Q 最低 20 帧（与 analysis_20260919/phase_summary.json 一致）
SEGMENTS = {
    "plateau": list(range(0, 5)) + list(range(58, 80)),
    "accel": list(range(10, 17)),
    "peak": list(range(17, 27)),
    "decel": list(range(27, 43)),
    "trough": list(range(5, 10)) + list(range(43, 58)),
}
assert sorted(sum(SEGMENTS.values(), [])) == list(range(PERIOD))

ARMS = {
    "Tnull": dict(kind="tnull", run="wss_time_adapter_v51_20260923/Tnull_f{f}_s1234",
                  label="T-null（峰值预测 × 训练折逐帧统计，不训练）"),
    "C-raw": dict(kind="adapter", run="wss_time_adapter_v51_20260923/C-raw_f{f}_s1234",
                  label="C-raw 锚定时间头（原始 27 维几何 + 峰值 ln）"),
    "TT-warm": dict(kind="tt", run="wss_time_transformer_v51_20260924_r3/TT-warm_f{f}_s1234",
                    label="TT-warm（X5D 隐藏特征 + 跨帧注意力）"),
    "TT-raw": dict(kind="tt", run="wss_time_transformer_v51_20260924_r3/TT-raw_f{f}_s1234",
                   label="TT-raw（原始几何 + 跨帧注意力）"),
    "TT-warm-noattn": dict(kind="tt", run="wss_time_transformer_v51_20260924_r3/TT-warm-noattn_f{f}_s1234",
                           label="TT-warm-noattn（同 TT-warm，注意力改对角：无跨帧交互）"),
    "TT-raw-noattn": dict(kind="tt", run="wss_time_transformer_v51_20260924_r3/TT-raw-noattn_f{f}_s1234",
                          label="TT-raw-noattn（同 TT-raw，注意力改对角）"),
    "T0": dict(kind="t0", run="wss_time_ecc_20260918/T0_f{f}_s1234",
               label="T0 端到端相位查询（ckpt_best，不锚定）"),
}


# ---------------------------------------------------------------- helpers
def ln_floor(a) -> np.ndarray:
    return np.log(np.clip(np.asarray(a, dtype=np.float64), FLOOR, None) + EPS)


def case_key(uid: str) -> str:
    return uid.replace("/", "__")


def execution_record() -> Dict:
    rec = {"hostname": socket.gethostname(), "pid": os.getpid(), "python": sys.executable,
           "torch": torch.__version__, "cuda_available": torch.cuda.is_available(),
           "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
           "slurm_job_id": os.environ.get("SLURM_JOB_ID"), "started": time.strftime("%Y-%m-%d %H:%M:%S")}
    if torch.cuda.is_available():
        rec["gpu"] = torch.cuda.get_device_name(0)
    return rec


def load_area(uid: str, n: int) -> np.ndarray:
    """逐点面积（m²）；与 bundle 行序按 node_id_cas 对齐。"""
    import h5py
    with np.load(VIEW / uid / "bundle.npz", allow_pickle=False) as z:
        ids = z["wall_node_id_cas"]
    with h5py.File(H5 / case_key(uid) / "case.h5", "r") as f:
        valid = f["wall_static/valid"][:]
        hid = f["wall_static/node_id_cas"][:][valid]
        area = f["wall_static/area_m2"][:][valid].astype(np.float64)
    if not np.array_equal(hid, ids):
        order = {int(v): i for i, v in enumerate(hid)}
        area = area[np.array([order[int(v)] for v in ids])]
    if len(area) != n or not np.all(area > 0):
        raise ValueError(f"{uid}: area rows {len(area)} != {n} or non-positive")
    return area


def rowwise_corr(a: np.ndarray, b: np.ndarray, axis: int) -> np.ndarray:
    a = a - a.mean(axis=axis, keepdims=True)
    b = b - b.mean(axis=axis, keepdims=True)
    den = np.sqrt((a * a).sum(axis=axis) * (b * b).sum(axis=axis))
    num = (a * b).sum(axis=axis)
    out = np.full(num.shape, np.nan)
    ok = den > 1e-12
    out[ok] = num[ok] / den[ok]
    return out


def cyc_diff(a: np.ndarray, b: np.ndarray, period: int) -> np.ndarray:
    d = np.abs(a - b)
    return np.minimum(d, period - d)


def weighted_dice(mt: np.ndarray, mp: np.ndarray, w: np.ndarray) -> np.ndarray:
    """(F,N) 布尔掩膜 → 逐帧面积加权 Dice；两边都空的帧为 NaN。"""
    at, ap = mt @ w, mp @ w
    inter = (mt & mp) @ w
    den = at + ap
    out = np.full(len(at), np.nan)
    ok = den > 0
    out[ok] = 2 * inter[ok] / den[ok]
    return out


def quantile_mask(x: np.ndarray, q: float, low: bool) -> np.ndarray:
    thr = np.quantile(x, q if low else 1 - q, axis=1, keepdims=True)
    return x <= thr if low else x >= thr


# ---------------------------------------------------------------- metrics
def case_metrics(tau: np.ndarray, pred: np.ndarray, area: np.ndarray) -> Tuple[Dict, Dict]:
    """tau / pred：(81, N) Pa（pred 已截 ≥0）。返回 (标量, 逐帧数组)。"""
    if tau.shape != pred.shape or tau.shape[0] != N_FRAMES:
        raise ValueError(f"shape mismatch {tau.shape} vs {pred.shape}")
    lt, lp = ln_floor(tau), ln_floor(pred)
    out: Dict = {"n_points": int(tau.shape[1])}
    # ---- 复现口径（81 帧，与 evaluate._cycle_metrics 相同）
    d = cyc_diff(lt.argmax(0), lp.argmax(0), N_FRAMES - 1)
    out["peak_time_err_med"] = float(np.median(d))
    out["peak_time_err_p90"] = float(np.quantile(d, 0.9))
    c81 = rowwise_corr(lt, lp, axis=0)
    out["ts_corr_ln"] = float(np.nanmean(c81))
    out["amp_err_ln_med"] = float(np.median(np.abs((lp.max(0) - lp.min(0)) - (lt.max(0) - lt.min(0)))))
    yt, yp = tau[PEAK_INDEX], pred[PEAK_INDEX]
    out["peak_r2_case_pa"] = float(1 - np.mean((yt - yp) ** 2) / max(np.var(yt), 1e-12))
    # ---- 趋势口径（帧 0–79 一个周期）
    T, P, LT, LP = tau[:PERIOD], pred[:PERIOD], lt[:PERIOD], lp[:PERIOD]
    w = area / area.sum()
    frame = {}
    frame["spearman"] = rowwise_corr(rankdata(T, axis=1), rankdata(P, axis=1), axis=1)
    frame["pearson_ln"] = rowwise_corr(LT, LP, axis=1)
    mt, mp = T < LOW_WSS_PA, P < LOW_WSS_PA
    frame["lowwss_frac_true"], frame["lowwss_frac_pred"] = mt @ w, mp @ w
    frame["lowwss_dice"] = weighted_dice(mt, mp, w)
    frame["lowq_dice"] = weighted_dice(quantile_mask(T, LOW_Q, True), quantile_mask(P, LOW_Q, True), w)
    frame["highq_dice"] = weighted_dice(quantile_mask(T, HIGH_Q, False), quantile_mask(P, HIGH_Q, False), w)
    frame["global_true"], frame["global_pred"] = T @ w, P @ w
    for key in ("spearman", "pearson_ln", "lowq_dice", "highq_dice", "lowwss_dice"):
        out[f"{key}_cycle"] = float(np.nanmean(frame[key]))
        for seg, idx in SEGMENTS.items():
            out[f"{key}_{seg}"] = float(np.nanmean(frame[key][idx])) if np.isfinite(frame[key][idx]).any() else float("nan")
    ft, fp = frame["lowwss_frac_true"], frame["lowwss_frac_pred"]
    out["lowwss_frac_mae"] = float(np.mean(np.abs(fp - ft)))
    out["lowwss_frac_corr"] = float(rowwise_corr(ft[None], fp[None], axis=1)[0])
    out["lowwss_frac_true_range"] = float(np.ptp(ft))
    gt, gp = frame["global_true"], frame["global_pred"]
    out["global_corr"] = float(rowwise_corr(gt[None], gp[None], axis=1)[0])
    out["global_amp_ratio"] = float(np.ptp(gp) / max(np.ptp(gt), 1e-12))
    out["global_mean_ratio"] = float(gp.mean() / max(gt.mean(), 1e-12))
    out["global_shape_nrmse"] = float(np.sqrt(np.mean((gp / gp.mean() - gt / gt.mean()) ** 2)))
    out["ts_corr_pa"] = float(np.nanmean(rowwise_corr(T, P, axis=0)))
    ok = (T.mean(0) > FLOOR) & (P.mean(0) > FLOOR)
    s_err = np.sqrt(np.mean((P[:, ok] / P[:, ok].mean(0) - T[:, ok] / T[:, ok].mean(0)) ** 2, axis=0))
    out["shape_nrmse_med"] = float(np.median(s_err))
    out["shape_nrmse_wmean"] = float(np.sum(s_err * w[ok]) / w[ok].sum())
    dt = cyc_diff(LT.argmin(0), LP.argmin(0), PERIOD)
    out["trough_time_err_med"] = float(np.median(dt))
    out["trough_time_err_p90"] = float(np.quantile(dt, 0.9))
    # ---- 真值同步性分层（只用真值定义分层，属残差分析）
    g_ln = LT @ w
    r_sync = rowwise_corr(LT, np.broadcast_to(g_ln[:, None], LT.shape), axis=0)
    c80 = rowwise_corr(LT, LP, axis=0)
    strata = {"sync": r_sync >= SYNC_HI, "mid": (r_sync >= SYNC_LO) & (r_sync < SYNC_HI),
              "async": ~(r_sync >= SYNC_LO)}           # 含 NaN（真值常值点）归入 async
    for name, m in strata.items():
        out[f"area_frac_{name}"] = float(w[m].sum())
        good = m & np.isfinite(c80)
        out[f"ts_corr_ln_{name}"] = float(np.sum(c80[good] * w[good]) / w[good].sum()) if good.any() else float("nan")
    return out, {k: np.round(np.asarray(v, dtype=np.float64), 6).tolist() for k, v in frame.items()}


class FoldAccumulator:
    """按病例累计一阶/二阶矩，复现 metrics.casebalanced_field_metrics 的逐帧 R²_cb。"""

    def __init__(self):
        self.parts = {"pa": [], "ln": [], "tawss": []}

    @staticmethod
    def moments(tau, pred) -> Dict:
        """逐例一阶/二阶矩（81 帧），可存进 JSON 再合并；R²_cb 与 metrics.casebalanced_field_metrics 同式。"""
        lt, lp = ln_floor(tau), ln_floor(pred)
        m = {}
        for name, (yt, yp) in (("pa", (tau, pred)), ("ln", (lt, lp))):
            m[name] = np.stack([yt.mean(1), (yt ** 2).mean(1), ((yt - yp) ** 2).mean(1)]).tolist()
        ta_t, ta_p = tau.mean(0), pred.mean(0)
        m["tawss"] = [float(ta_t.mean()), float((ta_t ** 2).mean()), float(((ta_t - ta_p) ** 2).mean())]
        return m

    def add_moments(self, m: Dict):
        for name in self.parts:
            self.parts[name].append(np.asarray(m[name], dtype=np.float64))

    def add(self, tau, pred):
        self.add_moments(self.moments(tau, pred))

    @staticmethod
    def _r2(stack):
        m1, m2, mse = stack[:, 0], stack[:, 1], stack[:, 2]
        g = m1.mean(0)
        return 1 - mse.mean(0) / (m2 - 2 * g * m1 + g ** 2).mean(0)

    def summary(self, q_norm) -> Dict:
        r_pa = self._r2(np.asarray(self.parts["pa"]))
        r_ln = self._r2(np.asarray(self.parts["ln"]))
        trough = np.sort(np.argsort(np.asarray(q_norm))[:20])
        return {"cycle_r2cb_pa": float(r_pa.mean()), "cycle_r2cb_ln": float(r_ln.mean()),
                "peak_r2cb_pa": float(r_pa[PEAK_INDEX]), "trough_r2cb_pa": float(r_pa[trough].mean()),
                "tawss_r2cb_pa": float(self._r2(np.asarray(self.parts["tawss"]))),
                "frame_r2cb_pa": np.round(r_pa, 6).tolist()}


# ---------------------------------------------------------------- predictors (yield uid, tau (81,N), pred_pa (81,N))
def _adapter_like(arm: str, fold: int, device: str, only: Sequence[str] | None):
    run = RUNS / ARMS[arm]["run"].format(f=fold)
    cfg = json.loads((run / "metrics.json").read_text())["config"]
    data = TA.FoldData(cfg)
    head, builder = None, None
    if ARMS[arm]["kind"] == "adapter":
        ck = torch.load(run / "head_last.pt", map_location=device, weights_only=False)
        builder = TA.InputBuilder.__new__(TA.InputBuilder)
        builder.inputs = list(ck["inputs"])
        builder.stats = {k: (np.asarray(v["mean"]), np.asarray(v["std"])) for k, v in ck["input_stats"].items()}
        builder.dim = sum(len(builder.stats[k][0]) for k in builder.inputs)
        hc = cfg["head"]
        head = TA.TimeAdapterHead(builder.dim, len(cfg["phase_features"]), hc["geom_hidden"], hc["time_hidden"],
                                  hc["decoder_hidden"], hc["zero_init_output"]).to(device)
        head.load_state_dict(ck["head"])
        head.eval()
    phase = torch.tensor(data.phase, dtype=torch.float32, device=device)
    for uid in data.test_ids:
        if only is not None and uid not in only:
            continue
        c = data.load_case(uid)
        lp = TA.tnull_base(c["peak_ln"], data.mu, data.sd, data.peak_index).T
        if head is not None:
            block = builder.point_block(c)
            outs = []
            with torch.no_grad():
                for s in range(0, len(block), 8192):
                    g = torch.from_numpy(block[s:s + 8192]).to(device)
                    outs.append(TA.anchored(head(g, phase), data.peak_index).double().cpu().numpy())
            lp = lp + np.concatenate(outs).T
        yield uid, c["tau"], np.clip(np.exp(lp) - EPS, 0, None)


def _tt(arm: str, fold: int, device: str, only):
    run = RUNS / ARMS[arm]["run"].format(f=fold)
    ck = torch.load(run / "ckpt_last.pt", map_location=device, weights_only=False)
    cfg = ck["config"]
    fd = TT._load_fold(cfg)
    model = TT.TemporalFieldModel(sum(TT.INPUT_DIMS.values()), cfg["architecture"]).to(device)
    model.load_state_dict(ck["state_dict"])
    model.eval()
    mean, std = np.asarray(ck["input_mean"], np.float32), np.asarray(ck["input_std"], np.float32)
    phase = torch.from_numpy(fd["phase"]).to(device)
    for uid in fd["test_ids"]:
        if only is not None and uid not in only:
            continue
        c = TT._load_case(cfg, uid, fd, False)
        lp = TT._predict_case(model, c, phase, mean, std, fd, cfg, device)
        yield uid, c["tau_raw"], np.clip(np.exp(lp) - EPS, 0, None)


def _t0(arm: str, fold: int, device: str, only):
    run = RUNS / ARMS[arm]["run"].format(f=fold)
    cfg, feat_stats, model, _ = E.load_model_from_run(run, device, "best")
    model.eval()
    wss_stats = E.load_wss_stats_for_run(run)
    split = json.loads(Path(cfg.data.split_path).read_text())
    for label in split["test_cases"]:
        uid = D.canonical_unit_id(label)
        if only is not None and uid not in only:
            continue
        cohort, subset, name = uid.split("/", 2)
        case = D.load_case(f"{cohort}/{subset}", name, wss_stats, target=cfg.data.target,
                           target_normalization=cfg.data.target_normalization, data_root=cfg.data.data_root,
                           required_frame_version=cfg.data.required_frame_version,
                           timesteps=cfg.data.timesteps, waveform_path=cfg.data.waveform_path,
                           extra_point_features=C.v6_point_features(cfg),
                           point_features_root=getattr(cfg.data, "point_features_root", None))
        n = len(case["pos"])
        tau = np.empty((N_FRAMES, n)); pred = np.empty((N_FRAMES, n))
        for k in range(N_FRAMES):
            D.select_frame(case, k)
            pn = np.asarray(E.predict_case_norm(model, case, cfg.data.input_features, feat_stats, device,
                                                cfg=cfg, frame_index=k), dtype=np.float64)
            pred[k] = np.clip(D.denormalize_wss(pn, D.frame_stats_view(wss_stats, k)), 0, None)
            tau[k] = np.asarray(case["y_raw"], dtype=np.float64)
        with np.load(VIEW / uid / "bundle.npz", allow_pickle=False) as z:
            if not np.allclose(z["wall_wss"], tau, rtol=0, atol=1e-6):
                raise RuntimeError(f"{uid}: T0 case labels differ from bundle wall_wss")
        yield uid, tau, pred


def iter_arm(arm: str, fold: int, device: str, only: Sequence[str] | None = None) -> Iterator:
    kind = ARMS[arm]["kind"]
    fn = {"tnull": _adapter_like, "adapter": _adapter_like, "tt": _tt, "t0": _t0}[kind]
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = kind != "adapter"
    yield from fn(arm, fold, device, only)


def stored_reference(arm: str, fold: int) -> Dict:
    run = RUNS / ARMS[arm]["run"].format(f=fold)
    if ARMS[arm]["kind"] == "t0":
        m = json.loads((run / "eval/ckpt_best/metrics.json").read_text())["test"]
        cyc = dict(m["cycle"])
        ta = m.get("tawss", {})
        cyc["tawss_r2cb_pa"] = (ta.get("field_casebalanced") or {}).get("r2", float("nan"))
        return cyc
    m = json.loads((run / "metrics.json").read_text())["metrics"]["eval_compatible"]
    return m


# ---------------------------------------------------------------- commands
def fold_test_ids(fold: int) -> List[str]:
    split = json.loads((VIEW / f"cv3_v51/fold{fold}.json").read_text())
    return [D.canonical_unit_id(u) for u in split["test_cases"]]


def cmd_metrics(args):
    """逐例指标；--nshards>1 时按留出病例序号取模分片（T0 逐帧前向慢），分片齐后由 summarize 合并对账。"""
    out = Path(args.out)
    (out / "per_case").mkdir(parents=True, exist_ok=True)
    tag = f"{args.arm}_f{args.fold}"
    shard_tag = tag if args.nshards == 1 else f"{tag}_s{args.shard}of{args.nshards}"
    per_case_path = out / "per_case" / f"{shard_tag}.jsonl"
    if per_case_path.exists() and not args.overwrite:
        raise FileExistsError(per_case_path)
    ids = fold_test_ids(args.fold)
    only = [u for i, u in enumerate(ids) if i % args.nshards == args.shard]
    if args.limit:
        only = only[:args.limit]
    t0 = time.perf_counter()
    with per_case_path.open("w") as fh:
        for i, (uid, tau, pred) in enumerate(iter_arm(args.arm, args.fold, args.device, only=only)):
            tc = time.perf_counter()
            scal, frame = case_metrics(tau, pred, load_area(uid, tau.shape[1]))
            rec = {"arm": args.arm, "fold": args.fold, "unit_id": uid, "cohort": uid.split("/")[0], **scal,
                   "moments": FoldAccumulator.moments(tau, pred), "frame": frame}
            fh.write(json.dumps(rec) + "\n"); fh.flush()
            print(f"[{shard_tag}] {i + 1}/{len(only)} {uid} spearman {scal['spearman_cycle']:.3f} ts_ln {scal['ts_corr_ln']:.3f} "
                  f"metric_s {time.perf_counter() - tc:.1f} elapsed {time.perf_counter() - t0:.0f}s", flush=True)
    meta = {"shard": args.shard, "nshards": args.nshards, "n_cases": len(only), "elapsed_seconds": time.perf_counter() - t0,
            "execution": execution_record()}
    (out / "per_case" / f"{shard_tag}.meta.json").write_text(json.dumps(meta, indent=1))
    print(f"[{shard_tag}] done n={len(only)} elapsed {meta['elapsed_seconds']:.0f}s", flush=True)
    if args.nshards == 1 and not args.limit:
        summarize(out, args.arm, args.fold)


def summarize(out: Path, arm: str, fold: int) -> Dict:
    tag = f"{arm}_f{fold}"
    files = sorted((out / "per_case").glob(f"{tag}.jsonl")) + sorted((out / "per_case").glob(f"{tag}_s*of*.jsonl"))
    recs = {}
    for f in files:
        for line in f.read_text().splitlines():
            r = json.loads(line)
            recs[r["unit_id"]] = r
    ids = fold_test_ids(fold)
    missing = [u for u in ids if u not in recs]
    if missing:
        raise RuntimeError(f"{tag}: {len(missing)} held-out cases missing, e.g. {missing[:3]}")
    rows = [recs[u] for u in ids]
    acc = FoldAccumulator()
    for r in rows:
        acc.add_moments(r["moments"])
    rep = acc.summary(json.loads(WAVEFORM.read_text())["q_norm"])
    keys = ("peak_time_err_med", "peak_time_err_p90", "ts_corr_ln", "amp_err_ln_med")
    rep.update({k: float(np.mean([r[k] for r in rows])) for k in keys})
    ref = stored_reference(arm, fold)
    ref_map = {"cycle_r2cb_pa": "cycle_r2cb_pa", "trough_r2cb_pa": "trough_r2cb_pa", "peak_r2cb_pa": "peak_r2cb_pa",
               "tawss_r2cb_pa": "tawss_r2cb_pa", "cycle_r2cb_ln": "cycle_r2cb_ln", "ts_corr_ln": "ts_corr_ln",
               "peak_time_err_med": "peak_time_err_med_frames", "peak_time_err_p90": "peak_time_err_p90_frames",
               "amp_err_ln_med": "amp_err_ln_med"}
    check = {k: {"reproduced": rep[k], "stored": ref.get(v),
                 "abs_diff": abs(rep[k] - ref[v]) if isinstance(ref.get(v), (int, float)) else None}
             for k, v in ref_map.items()}
    max_diff = max(c["abs_diff"] for c in check.values() if c["abs_diff"] is not None)
    scalar_keys = [k for k, v in rows[0].items() if isinstance(v, (int, float)) and k not in ("fold", "n_points")]
    metas = [json.loads(m.read_text()) for m in sorted((out / "per_case").glob(f"{tag}*.meta.json"))]
    summary = {"schema": "wss_time_trend_fold_v1", "arm": arm, "fold": fold, "n_cases": len(rows),
               "run": str(RUNS / ARMS[arm]["run"].format(f=fold)), "per_case_files": [f.name for f in files],
               "reproduced": rep, "reference_check": check, "reference_max_abs_diff": max_diff,
               "trend_means": {k: float(np.nanmean([r[k] for r in rows])) for k in scalar_keys},
               "shards": metas}
    (out / "fold_summary").mkdir(parents=True, exist_ok=True)
    (out / "fold_summary" / f"{tag}.json").write_text(json.dumps(summary, indent=1, ensure_ascii=False))
    print(f"[{tag}] summary n={len(rows)} reference_max_abs_diff={max_diff:.2e}", flush=True)
    return summary


def cmd_summarize(args):
    for arm in args.arm.split(","):
        for fold in (0, 1, 2):
            summarize(Path(args.out), arm, fold)


def fold_of(uid: str) -> int:
    for f in range(3):
        split = json.loads((VIEW / f"cv3_v51/fold{f}.json").read_text())
        if uid in {D.canonical_unit_id(u) for u in split["test_cases"]}:
            return f
    raise KeyError(uid)


def cmd_export(args):
    out = Path(args.out) / "export"
    cases = [c.strip() for c in args.cases.split(",") if c.strip()]
    by_fold: Dict[int, List[str]] = {}
    for uid in cases:
        by_fold.setdefault(fold_of(uid), []).append(uid)
    for fold, uids in sorted(by_fold.items()):
        for uid, tau, pred in iter_arm(args.arm, fold, args.device, only=uids):
            d = out / case_key(uid)
            d.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(d / f"{args.arm}.npz", pred_pa=pred[:PERIOD].astype(np.float16),
                                pred_peak_pa=pred[PEAK_INDEX].astype(np.float32), fold=np.array(fold))
            print(f"[export] {args.arm} fold{fold} {uid}", flush=True)


def cmd_truth(args):
    out = Path(args.out) / "export"
    wf = json.loads(WAVEFORM.read_text())
    for uid in [c.strip() for c in args.cases.split(",") if c.strip()]:
        with np.load(VIEW / uid / "bundle.npz", allow_pickle=False) as z:
            tau = z["wall_wss"].astype(np.float32)
            xyz = z["wall_coords_aligned_mm"].astype(np.float32)
            nrm = z["wall_normal_pca_aligned"].astype(np.float32)
            absc = z["wall_abscissa_norm"].astype(np.float32)
        d = out / case_key(uid)
        d.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(d / "truth.npz", tau=tau[:PERIOD].astype(np.float16), xyz=xyz, normal=nrm, abscissa=absc,
                            area=load_area(uid, tau.shape[1]).astype(np.float32), fold=np.array(fold_of(uid)),
                            q_norm=np.asarray(wf["q_norm"][:PERIOD], np.float32),
                            time_s=np.asarray(wf["time_s"][:PERIOD], np.float32), unit_id=np.array(uid))
        print(f"[truth] {uid}", flush=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    sm = sub.add_parser("summarize"); sm.add_argument("--arm", required=True, help="逗号分隔臂名")
    sm.add_argument("--out", default=str(EXP))
    m = sub.add_parser("metrics"); m.add_argument("--arm", required=True, choices=list(ARMS))
    m.add_argument("--shard", type=int, default=0); m.add_argument("--nshards", type=int, default=1)
    m.add_argument("--fold", type=int, required=True); m.add_argument("--overwrite", action="store_true")
    m.add_argument("--limit", type=int, default=0, help="smoke only: stop after N cases (reference check then meaningless)")
    x = sub.add_parser("export"); x.add_argument("--arm", required=True, choices=list(ARMS)); x.add_argument("--cases", required=True)
    t = sub.add_parser("truth"); t.add_argument("--cases", required=True)
    for p in (m, x, t):
        p.add_argument("--out", default=str(EXP))
        p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()
    {"metrics": cmd_metrics, "export": cmd_export, "truth": cmd_truth, "summarize": cmd_summarize}[args.cmd](args)


if __name__ == "__main__":
    main()

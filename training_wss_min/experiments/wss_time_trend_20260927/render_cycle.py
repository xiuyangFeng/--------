#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""全周期 WSS 并排动画 + 相位拼图（CFD 与各时间臂，同一色标）。

读 trend_eval.py export/truth 产出的 export/<case_key>/{truth,<arm>}.npz，
用点云 z-buffer 投影渲染（集群无 X，pyvista 不能离屏；壁面点约 8 万、面积均匀，splat 足够连续）。
输出 animations/<case_key>/：cycle.mp4、cycle.gif、contact_logpa.png（固定对数 Pa 色标）、
contact_rank.png（逐帧百分位色标：只看空间分布，与幅值无关）。
"""
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
from pathlib import Path
from typing import Dict, List, Sequence

import matplotlib
matplotlib.use("Agg")
import matplotlib.font_manager as fm  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.colors import LogNorm, Normalize  # noqa: E402
from PIL import Image  # noqa: E402
from scipy.stats import rankdata  # noqa: E402

EXP = Path(__file__).resolve().parent
for ttc in ("/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",):
    if Path(ttc).is_file():
        fm.fontManager.addfont(ttc)
plt.rcParams["font.family"] = ["Noto Sans CJK JP", "Droid Sans Fallback", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

ARM_STYLE = {"CFD": ("CFD 真值", "#1f1f1d"), "Tnull": ("T-null（峰值×波形）", "#2a78d6"),
             "C-raw": ("C-raw 锚定时间头", "#eb6834"), "T0": ("T0 端到端", "#1baf7a"),
             "TT-raw": ("TT-raw", "#eda100"), "TT-warm": ("TT-warm", "#e87ba4")}
PANEL_PX = (300, 420)          # 每个视图面板 宽×高（像素）
CONTACT_FRAMES = (13, 21, 30, 38, 48, 70)
LOW_WSS_PA = 0.4


def view_matrix(azim_deg: float, elev_deg: float) -> np.ndarray:
    """返回 3×3：行 0/1 = 图像 u（右）/ v（上），行 2 = 朝向观察者的深度轴。对齐坐标 z 为主轴（竖直）。"""
    a, e = np.deg2rad(azim_deg), np.deg2rad(elev_deg)
    right = np.array([np.cos(a), np.sin(a), 0.0])
    toward = np.array([-np.sin(a) * np.cos(e), np.cos(a) * np.cos(e), np.sin(e)])
    up = np.cross(toward, right)
    return np.stack([right, up, toward])


class SplatView:
    """一次性算好每个像素由哪个点着色（z-buffer）与 Lambert 明暗；逐帧只换标量。"""

    def __init__(self, xyz: np.ndarray, normal: np.ndarray, area_m2: np.ndarray, R: np.ndarray,
                 size=PANEL_PX, bounds=None):
        w, h = size
        p = xyz @ R.T
        u, v, depth = p[:, 0], p[:, 1], p[:, 2]
        if bounds is None:
            bounds = (u.min(), u.max(), v.min(), v.max())
        u0, u1, v0, v1 = bounds
        scale = 0.94 * min(w / max(u1 - u0, 1e-6), h / max(v1 - v0, 1e-6))   # px / mm
        cu, cv = (u0 + u1) / 2, (v0 + v1) / 2
        iu = np.round((u - cu) * scale + w / 2).astype(np.int64)
        iv = np.round(h / 2 - (v - cv) * scale).astype(np.int64)
        spacing_mm = np.sqrt(np.median(area_m2) * 1e6)
        r = int(np.clip(np.ceil(1.1 * spacing_mm * scale), 2, 6))
        offs = [(dx, dy) for dx in range(-r + 1, r) for dy in range(-r + 1, r)] if r > 1 else [(0, 0)]
        pix, pts, dep = [], [], []
        idx = np.arange(len(xyz))
        for dx, dy in offs:
            x, y = iu + dx, iv + dy
            ok = (x >= 0) & (x < w) & (y >= 0) & (y < h)
            pix.append(y[ok] * w + x[ok]); pts.append(idx[ok]); dep.append(depth[ok])
        pix, pts, dep = np.concatenate(pix), np.concatenate(pts), np.concatenate(dep)
        order = np.lexsort((-dep, pix))                     # 同一像素里离观察者最近者优先
        pix, pts = pix[order], pts[order]
        first = np.r_[True, pix[1:] != pix[:-1]]
        self.pix, self.pts = pix[first], pts[first]
        n = normal / np.maximum(np.linalg.norm(normal, axis=1, keepdims=True), 1e-9)
        self.shade = (0.45 + 0.55 * np.abs(n[self.pts] @ R[2]))[:, None]
        self.size, self.bounds, self.splat_r = size, bounds, r

    def image(self, values: np.ndarray, cmap, norm) -> np.ndarray:
        w, h = self.size
        img = np.ones((h * w, 3))
        rgb = cmap(norm(values[self.pts]))[:, :3]
        img[self.pix] = rgb * self.shade
        return img.reshape(h, w, 3)


def frame_rank(x: np.ndarray) -> np.ndarray:
    return (rankdata(x) - 0.5) / len(x)


def load_case(case_dir: Path, arms: Sequence[str]) -> Dict:
    t = np.load(case_dir / "truth.npz", allow_pickle=False)
    d = {"uid": str(t["unit_id"]), "fold": int(t["fold"]), "xyz": t["xyz"].astype(np.float64),
         "normal": t["normal"].astype(np.float64), "area": t["area"].astype(np.float64),
         "q": t["q_norm"].astype(np.float64), "time": t["time_s"].astype(np.float64),
         "abscissa": t["abscissa"].astype(np.float64),
         "fields": {"CFD": t["tau"].astype(np.float64)}}
    for a in arms:
        d["fields"][a] = np.load(case_dir / f"{a}.npz")["pred_pa"].astype(np.float64)
    return d


def curves(case: Dict) -> Dict:
    w = case["area"] / case["area"].sum()
    out = {}
    for name, f in case["fields"].items():
        out[name] = {"low": (f < LOW_WSS_PA) @ w, "mean": f @ w}
    return out


def spearman_frames(case: Dict) -> Dict[str, np.ndarray]:
    t = case["fields"]["CFD"]
    rt = rankdata(t, axis=1)
    out = {}
    for name, f in case["fields"].items():
        if name == "CFD":
            continue
        rp = rankdata(f, axis=1)
        a, b = rt - rt.mean(1, keepdims=True), rp - rp.mean(1, keepdims=True)
        out[name] = (a * b).sum(1) / np.sqrt((a * a).sum(1) * (b * b).sum(1))
    return out


def build_views(case: Dict) -> List[SplatView]:
    """前视 / 后视；若入口（中心线弧长 < 5 %）不在上方，则绕视线转 180°，保证入口朝上。"""
    views = []
    inlet = case["abscissa"] < 0.05
    for azim in (0.0, 180.0):
        R = view_matrix(azim, 8.0)
        v = case["xyz"] @ R[1]
        if inlet.any() and v[inlet].mean() < v.mean():
            R = R * np.array([[-1.0], [-1.0], [1.0]])
        views.append(SplatView(case["xyz"], case["normal"], case["area"], R))
    return views


def render_animation(case: Dict, out_dir: Path, arms: Sequence[str], fps: int, info: str) -> Dict:
    names = ["CFD"] + list(arms)
    views = build_views(case)
    cfd = case["fields"]["CFD"]
    vmin = max(0.05, float(np.percentile(cfd, 1)))
    vmax = float(np.percentile(cfd, 99.5))
    norm = LogNorm(vmin=vmin, vmax=vmax, clip=True)
    cmap = plt.get_cmap("turbo")
    cur, sp = curves(case), spearman_frames(case)
    nf = cfd.shape[0]
    ncol = len(names)
    fig = plt.figure(figsize=(3.0 * ncol + 1.6, 11.4), dpi=100)
    gs = fig.add_gridspec(3, ncol + 1, height_ratios=[1, 1, 0.5], width_ratios=[1] * ncol + [0.06],
                          hspace=0.22, wspace=0.05, left=0.04, right=0.92, top=0.9, bottom=0.06)
    ims, sp_txt = [], []
    for r, vname in enumerate(("前视", "后视")):
        for c, name in enumerate(names):
            ax = fig.add_subplot(gs[r, c]); ax.set_axis_off()
            im = ax.imshow(views[r].image(case["fields"][name][0], cmap, norm), interpolation="nearest")
            ims.append((im, r, name))
            if r == 0:
                ax.set_title(ARM_STYLE[name][0], fontsize=12, color=ARM_STYLE[name][1], fontweight="bold", pad=18)
                sp_txt.append((ax.text(0.5, 1.01, "", transform=ax.transAxes, ha="center", va="bottom", fontsize=9.5,
                                       color="#333"), name))
            if c == 0:
                ax.text(-0.02, 0.5, vname, transform=ax.transAxes, rotation=90, va="center", ha="right", fontsize=11)
    cax = fig.add_subplot(gs[0:2, ncol])
    cb = fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=cmap), cax=cax)
    cb.set_label("WSS (Pa，对数色标，本例固定)", fontsize=10)
    sub = gs[2, :ncol].subgridspec(1, 3, wspace=0.28)
    t = case["time"] - case["time"][0]
    ax_q = fig.add_subplot(sub[0]); ax_q.plot(t, case["q"], color="#444"); ax_q.set_title("入口流量 Q/Qpeak（协议波形）", fontsize=10)
    ax_low = fig.add_subplot(sub[1]); ax_low.set_title(f"低 WSS（<{LOW_WSS_PA} Pa）面积占比", fontsize=10)
    ax_mean = fig.add_subplot(sub[2]); ax_mean.set_title("全壁面积加权平均 WSS (Pa)", fontsize=10)
    for name in names:
        lw, ls = (2.4, "-") if name == "CFD" else (1.4, "--")
        ax_low.plot(t, cur[name]["low"], color=ARM_STYLE[name][1], lw=lw, ls=ls, label=ARM_STYLE[name][0])
        ax_mean.plot(t, cur[name]["mean"], color=ARM_STYLE[name][1], lw=lw, ls=ls)
    ax_low.legend(fontsize=8, loc="upper left", frameon=False)
    vlines = [ax.axvline(t[0], color="#888", lw=1) for ax in (ax_q, ax_low, ax_mean)]
    for ax in (ax_q, ax_low, ax_mean):
        ax.set_xlabel("t (s)", fontsize=9); ax.tick_params(labelsize=8)
    title = fig.suptitle("", fontsize=13)
    fig.text(0.5, 0.955, info, ha="center", fontsize=9, color="#555")
    tmp = out_dir / "_frames"
    if tmp.exists():
        shutil.rmtree(tmp)
    tmp.mkdir(parents=True)
    for k in range(nf):
        for im, r, name in ims:
            im.set_data(views[r].image(case["fields"][name][k], cmap, norm))
        for txt, name in sp_txt:
            txt.set_text("" if name == "CFD" else f"该帧空间 Spearman {sp[name][k]:.2f}")
        for vl in vlines:
            vl.set_xdata([t[k], t[k]])
        title.set_text(f"{case['uid']}（cv3 fold {case['fold']} 留出）  帧 {k}/{nf - 1}  t={t[k]:.2f} s  Q/Qpeak={case['q'][k]:.2f}")
        fig.savefig(tmp / f"f{k:03d}.png", dpi=100)
    plt.close(fig)
    mp4 = out_dir / "cycle.mp4"
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-framerate", str(fps), "-i", str(tmp / "f%03d.png"),
                    "-vf", "pad=ceil(iw/2)*2:ceil(ih/2)*2", "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "20", str(mp4)],
                   check=True)
    frames = [Image.open(tmp / f"f{k:03d}.png").convert("RGB") for k in range(nf)]
    small = [f.resize((f.width * 2 // 3, f.height * 2 // 3), Image.LANCZOS).quantize(colors=192, method=Image.MEDIANCUT) for f in frames]
    gif = out_dir / "cycle.gif"
    small[0].save(gif, save_all=True, append_images=small[1:], duration=int(1000 / fps), loop=0, optimize=True)
    shutil.copy2(tmp / "f021.png", out_dir / "frame_peak.png")
    shutil.copy2(tmp / "f048.png", out_dir / "frame_trough.png")
    shutil.rmtree(tmp)
    return {"vmin_pa": vmin, "vmax_pa": vmax, "splat_radius_px": [v.splat_r for v in views],
            "mp4": mp4.name, "gif": gif.name}


def render_contact(case: Dict, out_dir: Path, arms: Sequence[str], mode: str, info: str) -> str:
    names = ["CFD"] + list(arms)
    view = build_views(case)[0]
    cfd = case["fields"]["CFD"]
    cmap = plt.get_cmap("turbo")
    if mode == "logpa":
        norm = LogNorm(vmin=max(0.05, float(np.percentile(cfd, 1))), vmax=float(np.percentile(cfd, 99.5)), clip=True)
        label = "WSS (Pa，对数色标，全周期固定)"
    else:
        norm = Normalize(0, 1)
        label = "该帧内百分位（0=最低，1=最高；与幅值无关）"
    t = case["time"] - case["time"][0]
    fig, axes = plt.subplots(len(names), len(CONTACT_FRAMES), figsize=(2.3 * len(CONTACT_FRAMES) + 0.8, 3.1 * len(names)),
                             dpi=110, squeeze=False)
    for r, name in enumerate(names):
        for c, k in enumerate(CONTACT_FRAMES):
            ax = axes[r, c]; ax.set_axis_off()
            vals = case["fields"][name][k]
            if mode == "rank":
                vals = frame_rank(vals)
            ax.imshow(view.image(vals, cmap, norm), interpolation="nearest")
            if r == 0:
                ax.set_title(f"帧 {k}  t={t[k]:.2f}s\nQ/Qpk={case['q'][k]:.2f}", fontsize=9)
            if c == 0:
                ax.text(-0.03, 0.5, ARM_STYLE[name][0], transform=ax.transAxes, rotation=90, va="center", ha="right",
                        fontsize=10, color=ARM_STYLE[name][1], fontweight="bold")
    fig.subplots_adjust(left=0.05, right=0.9, top=0.9, bottom=0.02, wspace=0.03, hspace=0.12)
    cax = fig.add_axes([0.915, 0.15, 0.015, 0.65])
    fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=cmap), cax=cax).set_label(label, fontsize=9)
    fig.suptitle(f"{case['uid']} · 前视 · {'对数 Pa' if mode == 'logpa' else '逐帧百分位'}\n{info}", fontsize=11)
    path = out_dir / f"contact_{mode}.png"
    fig.savefig(path, dpi=110)
    plt.close(fig)
    return path.name


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--cases", required=True, help="逗号分隔 unit_id")
    ap.add_argument("--arms", default="Tnull,C-raw,T0")
    ap.add_argument("--fps", type=int, default=12)
    ap.add_argument("--export-dir", default=str(EXP / "export"))
    ap.add_argument("--out-dir", default=str(EXP / "animations"))
    ap.add_argument("--info-json", default=None, help="{unit_id: 说明文字}，印在图上（选例理由）")
    args = ap.parse_args()
    arms = [a for a in args.arms.split(",") if a]
    info = json.loads(Path(args.info_json).read_text()) if args.info_json else {}
    manifest = {}
    for uid in [c for c in args.cases.split(",") if c]:
        key = uid.replace("/", "__")
        case = load_case(Path(args.export_dir) / key, arms)
        out = Path(args.out_dir) / key
        out.mkdir(parents=True, exist_ok=True)
        text = info.get(uid, "")
        rec = render_animation(case, out, arms, args.fps, text)
        rec["contact"] = [render_contact(case, out, arms, m, text) for m in ("logpa", "rank")]
        manifest[uid] = rec
        print(f"[render] {uid} {rec}", flush=True)
    mpath = Path(args.out_dir) / "manifest.json"
    old = json.loads(mpath.read_text()) if mpath.is_file() else {}
    old.update(manifest)
    mpath.write_text(json.dumps(old, indent=1, ensure_ascii=False))


if __name__ == "__main__":
    main()

"""Cycle-integrated wall labels (TAWSS / OSI / RRT / reversal fraction) for the ``wss_min_view_v1`` cloud.

Matrix: docs/02-推进与变更/03-周期量TAWSS_OSI/WSS_V5_周期积分量TAWSS_OSI直接回归实验矩阵_2026-09-20.md §2 (frozen definitions).

Labels are derived from the 81-frame wall shear vector already stored in every view bundle
(``wall_wss_vec`` (81, N, 3), source case.h5 ``wall_temporal/``); no new CFD is needed.

Cycle definition (C0): frames 0..79 (Fluent steps 1120..1278) with equal weight 1/80. Frame 80
(step 1280) is frame 0 shifted by one period T = 0.8 s (``q_norm[80] == q_norm[0]``); an 81-frame
equal-weight mean would count phase 0 twice, and for cases whose solution has not converged to a
periodic state it would mix in the next cycle. With a periodic solution 1/80 equal weights equal
the trapezoidal integral over [0, T].

    TAWSS    = (1/80) Σ_k ‖τ_k‖                                     [Pa]
    OSI      = 0.5 · (1 − ‖Σ_k τ_k‖ / Σ_k ‖τ_k‖)                    [0, 0.5]
    RRT      = 1 / ((1 − 2·OSI) · max(TAWSS, floor))                [1/Pa], report only
    rev_frac = (1/80) Σ_k [τ_k · τ_peak < 0]                        diagnostic, never a target/input

Rows are exactly the rows of the view's ``bundle.npz`` (``wall_node_id_cas`` copied for checking).
Audit fields per case: period-wrap mismatch ``mean|‖τ_80‖ − ‖τ_0‖| / mean‖τ_0‖`` (> 5 % flags an
unconverged cycle; kept, never excluded), vector-vs-scalar magnitude ratio at the peak frame, share
of points with TAWSS below the 0.05 Pa floor.

    python -m wss_v5.views.wall_cycle_v1 --workers 8          # build cycle.npz for every case
    python -m wss_v5.views.wall_cycle_v1 --stats              # train136 + cv3_v51 fold statistics
"""
from __future__ import annotations

import argparse
import hashlib
import json
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import numpy as np

from wss_pinn.utils import ROOT, utc_now

PACK_NAME = "wss_min_cycle_v1"
PACK_VERSION = "v1.0"
VIEW_ROOT = ROOT / "data_wss_v5" / "views_v5_1" / "wss_min_view_v1"
OUT_ROOT = ROOT / "data_wss_v5" / "views_v5_1" / PACK_NAME
N_FRAMES = 81
N_USED = 80
PEAK_STEP = 1162
PEAK_INDEX = 21
FLOOR_PA = 0.05
EPS = 1e-6
OSI_SCALE = 0.5
OSI_LOGIT_CLIP = 1e-3
WRAP_FLAG = 0.05
DEFINITION = ("TAWSS=(1/80)sum_k|tau_k| over frames 0..79 (steps 1120..1278; frame 80 = frame 0 + T); "
              "OSI=0.5*(1-|sum_k tau_k|/sum_k|tau_k|); RRT=1/((1-2*OSI)*max(TAWSS,0.05Pa)); "
              "rev_frac=(1/80)sum_k[tau_k.tau_peak<0]")


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def cycle_labels(vec: np.ndarray, scalar: np.ndarray | None = None) -> dict[str, Any]:
    """Pure label maths on (81, N, 3) vectors (float64 internally). Returns per-point arrays + audit scalars."""
    vec = np.asarray(vec, dtype=np.float64)
    if vec.ndim != 3 or vec.shape[0] != N_FRAMES or vec.shape[2] != 3:
        raise ValueError(f"wall_wss_vec must be (81, N, 3), got {vec.shape}")
    mag = np.linalg.norm(vec, axis=2)                                   # (81, N)
    used = slice(0, N_USED)
    tawss = mag[used].mean(axis=0)
    mean_vec = vec[used].mean(axis=0)                                   # (N, 3)
    osi = 0.5 * (1.0 - np.linalg.norm(mean_vec, axis=1) / np.clip(tawss, 1e-12, None))
    osi = np.clip(osi, 0.0, OSI_SCALE)
    rrt = 1.0 / (np.clip(1.0 - 2.0 * osi, 1e-3, None) * np.clip(tawss, FLOOR_PA, None))
    rev = (np.einsum("knd,nd->kn", vec[used], vec[PEAK_INDEX]) < 0.0).mean(axis=0)
    out = {
        "wall_tawss": tawss.astype(np.float32),
        "wall_osi": osi.astype(np.float32),
        "wall_rrt": rrt.astype(np.float32),
        "wall_rev_frac": rev.astype(np.float32),
        "wall_tawss_81": mag.mean(axis=0).astype(np.float32),          # legacy 81-frame equal mean (audit only)
    }
    m0, m80 = mag[0], mag[N_FRAMES - 1]
    denom = float(m0.mean()) if float(m0.mean()) > 1e-12 else 1e-12
    audit = {
        "period_wrap_rel_diff": float(np.abs(m80 - m0).mean() / denom),
        "period_wrap_ln_corr": float(np.corrcoef(np.log(np.clip(m0, FLOOR_PA, None)), np.log(np.clip(m80, FLOOR_PA, None)))[0, 1]),
        "tawss_below_floor_frac": float(np.mean(tawss < FLOOR_PA)),
        "tawss_median_pa": float(np.median(tawss)), "tawss_p10_pa": float(np.quantile(tawss, 0.1)),
        "osi_median": float(np.median(osi)), "osi_gt_0p1_frac": float(np.mean(osi > 0.1)), "osi_gt_0p3_frac": float(np.mean(osi > 0.3)),
        "stagnation_frac": float(np.mean((tawss < 0.4) & (osi > 0.1))),
        "tawss81_vs_80_max_rel_diff": float(np.abs(out["wall_tawss_81"] - tawss).max() / max(float(tawss.mean()), 1e-12)),
    }
    if scalar is not None:
        scalar = np.asarray(scalar, dtype=np.float64)
        ratio = mag[PEAK_INDEX] / np.clip(scalar[PEAK_INDEX], 1e-9, None)
        audit["vec_scalar_ratio_median_peak"] = float(np.median(ratio))
        out["wall_tawss_scalar80"] = scalar[used].mean(axis=0).astype(np.float32)
    return {"arrays": out, "audit": audit}


def build_case(canonical_id: str, out_root: Path = OUT_ROOT, view_root: Path = VIEW_ROOT) -> dict[str, Any]:
    started = time.time()
    bundle_path = Path(view_root) / canonical_id / "bundle.npz"
    with np.load(bundle_path, allow_pickle=True) as b:
        steps = np.asarray(b["steps"]).astype(np.int64)
        peak_step = int(b["peak_step"])
        if len(steps) != N_FRAMES or peak_step != PEAK_STEP or int(np.flatnonzero(steps == peak_step)[0]) != PEAK_INDEX \
                or not np.array_equal(steps, np.arange(1120, 1281, 2)):
            raise ValueError(f"{canonical_id}: unexpected steps/peak layout")
        node_id = np.asarray(b["wall_node_id_cas"]).astype(np.int64)
        frame_version = str(np.asarray(b["transform_frame_version"]).item()) if "transform_frame_version" in b.files else ""
        vec = np.asarray(b["wall_wss_vec"], dtype=np.float32)
        scalar = np.asarray(b["wall_wss"], dtype=np.float32)
    if vec.shape[1] != len(node_id) or scalar.shape != vec.shape[:2]:
        raise ValueError(f"{canonical_id}: wall_wss_vec/wall_wss/node_id shape mismatch")
    if not np.isfinite(vec).all():
        raise ValueError(f"{canonical_id}: non-finite wall_wss_vec")
    res = cycle_labels(vec, scalar)
    out_dir = Path(out_root) / canonical_id
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "cycle.npz"
    np.savez_compressed(
        out_path, wall_node_id_cas=node_id, **res["arrays"],
        steps_used=steps[:N_USED], n_frames_used=np.int64(N_USED), peak_step=np.int64(peak_step),
        floor_pa=np.float64(FLOOR_PA), definition=np.array(DEFINITION), pack=np.array(PACK_NAME), pack_version=np.array(PACK_VERSION),
        transform_frame_version=np.array(frame_version), source_bundle=np.array(str(bundle_path)),
    )
    return {"case": canonical_id, "n_points": int(len(node_id)), "seconds": round(time.time() - started, 2),
            "frame_version": frame_version, "sha256": sha256_file(out_path),
            "source_bundle_size": int(bundle_path.stat().st_size), "source_bundle_mtime": bundle_path.stat().st_mtime,
            **res["audit"]}


def build_all(out_root: Path, view_root: Path, workers: int, cases: list[str] | None) -> dict[str, Any]:
    ids = cases or sorted(str(p.parent.relative_to(view_root)) for p in Path(view_root).glob("*/*/*/bundle.npz"))
    started = time.time()
    reports, failures = [], []
    with ProcessPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(build_case, cid, out_root, view_root): cid for cid in ids}
        for fut in as_completed(futures):
            cid = futures[fut]
            try:
                reports.append(fut.result())
            except Exception as exc:  # noqa: BLE001 - record and continue, fail at the end
                failures.append({"case": cid, "error": f"{type(exc).__name__}: {exc}"})
    reports.sort(key=lambda r: r["case"])
    flagged = [r["case"] for r in reports if r["period_wrap_rel_diff"] > WRAP_FLAG]
    manifest = {
        "pack": PACK_NAME, "version": PACK_VERSION, "created_at": utc_now(), "view_root": str(view_root), "out_root": str(out_root),
        "definition": DEFINITION, "n_frames_used": N_USED, "steps_used": [1120, 1278], "floor_pa": FLOOR_PA,
        "period_wrap_flag_threshold": WRAP_FLAG, "n_cases": len(reports), "n_failed": len(failures),
        "period_wrap_flagged": flagged, "seconds": round(time.time() - started, 1), "cases": reports, "failures": failures,
    }
    Path(out_root).mkdir(parents=True, exist_ok=True)
    (Path(out_root) / "cycle_manifest.json").write_text(json.dumps(manifest, indent=1, ensure_ascii=False), encoding="utf-8")
    return manifest


# ---------------------------------------------------------------------------
# train-only target statistics (mirrors wss_global_stats_*.json so dataset.normalize_wss works unchanged)
# ---------------------------------------------------------------------------
def _split_cases(split_path: Path, partition: str) -> list[str]:
    from training_wss_min.dataset import load_split_cases  # local import: keeps the view package free of training deps otherwise
    return [f"{c}/{n}" for c, n in load_split_cases(split_path, partition)]


def _zstats(x: np.ndarray) -> dict[str, float]:
    return {"mean": float(x.mean()), "std": float(x.std())}


def build_stats(out_root: Path, scope: str, split_path: Path, partition: str = "train") -> list[Path]:
    ids = _split_cases(split_path, partition)
    tawss, osi = [], []
    for cid in ids:
        with np.load(Path(out_root) / cid / "cycle.npz", allow_pickle=False) as z:
            tawss.append(z["wall_tawss"].astype(np.float64)); osi.append(z["wall_osi"].astype(np.float64))
    tawss = np.concatenate(tawss); osi = np.concatenate(osi)
    log_t = np.log(np.clip(tawss, FLOOR_PA, None) + EPS)
    logit = np.log(np.clip(osi / OSI_SCALE, OSI_LOGIT_CLIP, 1 - OSI_LOGIT_CLIP)) - np.log1p(-np.clip(osi / OSI_SCALE, OSI_LOGIT_CLIP, 1 - OSI_LOGIT_CLIP))
    common = {"schema_version": 1, "generated_at": utc_now(), "pack": PACK_NAME, "pack_version": PACK_VERSION,
              "statistics_scope": f"{partition} partition only ({scope}, valid anatomy wall nodes)", "scope": scope,
              "split_path": str(split_path.relative_to(ROOT)) if split_path.is_relative_to(ROOT) else str(split_path),
              "split_sha256": sha256_file(split_path), "n_cases": len(ids), "n_points": int(tawss.size), "definition": DEFINITION}
    files = {
        f"cycle_stats_tawss_{scope}.json": {**common, "target": "tawss", "method": "log_z", "eps": EPS, "floor": FLOOR_PA,
                                            "timesteps_scope": "cycle_0_79", "zero_frac": float(np.mean(tawss <= 0)),
                                            "linear": _zstats(tawss), "log": _zstats(log_t),
                                            "raw_min": float(tawss.min()), "raw_max": float(tawss.max()),
                                            "below_floor_frac": float(np.mean(tawss < FLOOR_PA))},
        f"cycle_stats_osi_linear_{scope}.json": {**common, "target": "osi", "method": "linear", "eps": 0.0,
                                                 "range": [0.0, OSI_SCALE], "linear": _zstats(osi),
                                                 "raw_min": float(osi.min()), "raw_max": float(osi.max()),
                                                 "gt_0p1_frac": float(np.mean(osi > 0.1)), "gt_0p3_frac": float(np.mean(osi > 0.3))},
        f"cycle_stats_osi_logit_{scope}.json": {**common, "target": "osi", "method": "logit_z", "eps": 0.0,
                                                "range": [0.0, OSI_SCALE], "linear": _zstats(osi),
                                                "logit": {**_zstats(logit), "scale": OSI_SCALE, "clip": OSI_LOGIT_CLIP},
                                                "raw_min": float(osi.min()), "raw_max": float(osi.max())},
    }
    stats_dir = Path(out_root) / "stats"; stats_dir.mkdir(parents=True, exist_ok=True)
    written = []
    for name, payload in files.items():
        p = stats_dir / name; p.write_text(json.dumps(payload, indent=1, ensure_ascii=False), encoding="utf-8"); written.append(p)
    return written


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default=str(OUT_ROOT))
    ap.add_argument("--view-root", default=str(VIEW_ROOT))
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--cases", nargs="*", default=None)
    ap.add_argument("--stats", action="store_true", help="only (re)build train136 + cv3_v51 fold statistics from an existing pack")
    args = ap.parse_args(argv)
    out_root, view_root = Path(args.out), Path(args.view_root)
    if not args.stats:
        manifest = build_all(out_root, view_root, args.workers, args.cases)
        print(f"[{PACK_NAME}] wrote {manifest['n_cases']} cases to {out_root} in {manifest['seconds']}s; "
              f"failed {manifest['n_failed']}; period-wrap flagged {len(manifest['period_wrap_flagged'])}")
        if manifest["n_failed"]:
            for f in manifest["failures"]:
                print("  FAILED", f["case"], f["error"])
            return 1
    if args.cases:
        print(f"[{PACK_NAME}] partial build ({len(args.cases)} cases): statistics skipped")
        return 0
    scopes = {"train136": view_root / "split_V5_train136_test34.json"}
    scopes.update({f"fold{k}": view_root / "cv3_v51" / f"fold{k}.json" for k in range(3)})
    for scope, split in scopes.items():
        for p in build_stats(out_root, scope, split):
            print(f"[{PACK_NAME}] stats {p}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

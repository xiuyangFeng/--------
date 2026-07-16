#!/usr/bin/env python3
"""AG/AAA v4 训练入口数值门禁（不改变几何终签真源）。"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from . import config as C, raw_io
from .v4_cutover import FRAME_VERSION, _write_json_atomic


QUALITY_EXCLUSIONS = {
    "AAA/ruputer/CHEN_FU": {
        "category": "existing_denylist", "reason": "入口量级异常偏低；项目既有 denylist",
    },
    "AAA/ruputer/SU_KAI_LI": {
        "category": "existing_denylist", "reason": "全相位 WSS max=1386.9 Pa；历史 P0 离群",
    },
    "AAA/unruputer/ZHANG_GUI_HUA": {
        "category": "existing_denylist", "reason": "缺关键 Global_conditions 文件；项目既有 denylist",
    },
    "AAA/ruputer/ZHANG_ZAO_SHUAN": {
        "category": "inlet_waveform_outlier", "reason": "入口波形 max≈16661，导致 peak=1154",
    },
    "AAA/unruputer/GUO_YU_YING": {
        "category": "inlet_waveform_outlier", "reason": "入口波形 max≈13976，导致 peak=1154",
    },
    "AAA/ruputer/WANG_SHUN_WEN": {
        "category": "inlet_waveform_outlier", "reason": "入口波形 max≈1693.5，导致 peak=1160",
    },
}

WATCHLIST = {
    "AAA/ruputer/ZHOU_KE_XUN": "稀疏高 WSS 尾部，整场分布未离群",
    "AAA/ruputer/WU_GUANG_CUN": "稀疏高 WSS 尾部，整场分布未离群",
    "AAA/ruputer/DING_JUN_FENG": "稀疏高 WSS 尾部，整场分布未离群",
    "AAA/unruputer/ZHANG_YONG_ZHI": "稀疏高 WSS 尾部，整场分布未离群",
    "AG/fast/LI_ZHI_LIN": "稀疏高 WSS 尾部，整场分布未离群",
}


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def _q(values: np.ndarray, prefix: str) -> dict[str, float]:
    return {
        f"{prefix}_p{q}": float(np.percentile(values, q))
        for q in (0, 50, 90, 95, 99, 99.9, 100)
    }


def audit_case(data_root: Path, unit: str) -> dict:
    case_dir = data_root / unit
    bp, rp = case_dir / "bundle.npz", case_dir / "report.json"
    row: dict = {"unit_id": unit, "bundle_path": str(bp.resolve()), "report_path": str(rp.resolve())}
    hard: list[str] = []
    if not bp.is_file(): hard.append("missing_bundle")
    if not rp.is_file(): hard.append("missing_report")
    if hard:
        return {**row, "hard_failures": hard, "hard_qa_pass": False}
    row.update(bundle_sha256=sha256(bp), report_sha256=sha256(rp))
    report = json.loads(rp.read_text(encoding="utf-8"))
    try:
        with np.load(bp, allow_pickle=False) as z:
            required = {"steps", "peak_step", "wall_nodenumber", "wall_coords_raw",
                        "wall_coords_norm", "wall_wss", "wall_pressure", "transform_frame_version"}
            missing = sorted(required - set(z.files))
            if missing:
                hard.append("missing_bundle_keys:" + ",".join(missing))
                raise ValueError("missing keys")
            steps = np.asarray(z["steps"]); peak = int(np.asarray(z["peak_step"]).item())
            ids = np.asarray(z["wall_nodenumber"]); raw = np.asarray(z["wall_coords_raw"])
            norm = np.asarray(z["wall_coords_norm"]); wss = np.asarray(z["wall_wss"], dtype=np.float64)
            pressure = np.asarray(z["wall_pressure"], dtype=np.float64)
            frame = str(np.asarray(z["transform_frame_version"]).item())
            n_steps, n_nodes = len(steps), len(ids)
            if frame != FRAME_VERSION: hard.append(f"wrong_frame:{frame}")
            if raw.shape != (n_nodes, 3) or norm.shape != (n_nodes, 3): hard.append("coordinate_shape_mismatch")
            if wss.shape != (n_steps, n_nodes): hard.append("wss_shape_mismatch")
            if pressure.shape != (n_steps, n_nodes): hard.append("pressure_shape_mismatch")
            if len(np.unique(ids)) != n_nodes: hard.append("duplicate_node_ids")
            if not np.isfinite(wss).all(): hard.append("wss_nan_or_inf")
            if (wss < 0).any(): hard.append("negative_wss")
            nonpositive_frac = float((wss <= 0).mean())
            if nonpositive_frac > 0.01: hard.append("nonpositive_wss_over_1pct")
            if bool(np.all(wss <= 0)): hard.append("all_case_wss_zero")
            zero_steps = [int(steps[i]) for i in np.flatnonzero(np.all(wss <= 0, axis=1))]
            if zero_steps: hard.append("all_zero_timesteps:" + ",".join(map(str, zero_steps)))
            if peak not in steps.tolist(): hard.append("peak_step_not_exported")
            pi = steps.tolist().index(peak) if peak in steps.tolist() else 0
            peak_wss = wss[pi]
            step_max = np.max(wss, axis=1)
            temporal_spike_ratio = float(step_max.max() / max(np.median(step_max), 1e-12))
            row.update({
                "frame_version": frame, "n_steps": n_steps, "n_nodes": n_nodes,
                "peak_step": peak, "nonpositive_wss_frac": nonpositive_frac,
                "negative_wss_count": int((wss < 0).sum()),
                "nonfinite_wss_count": int((~np.isfinite(wss)).sum()),
                "all_zero_timestep_count": len(zero_steps),
                "temporal_step_max_spike_ratio": temporal_spike_ratio,
                "pressure_mean": float(np.mean(pressure)), "pressure_std": float(np.std(pressure)),
                "pressure_min": float(np.min(pressure)), "pressure_max": float(np.max(pressure)),
                **_q(wss.ravel(), "all_wss"), **_q(peak_wss, "peak_wss"),
            })
    except ValueError as exc:
        if str(exc) != "missing keys": hard.append(f"bundle_read_error:{exc}")

    parts = unit.split("/")
    raw_case = C.raw_case_dir("/".join(parts[:2]), parts[2])
    wave_path = raw_case / C.RAW_LAYOUT["inlet_waveform"]
    row.update(raw_case_dir=str(raw_case.resolve()), inlet_waveform_path=str(wave_path.resolve()),
               inlet_waveform_exists=wave_path.is_file())
    if wave_path.is_file():
        wave = raw_io.read_inlet_waveform(raw_case)
        vals = np.asarray(list(wave.values()), dtype=np.float64)
        row.update(inlet_waveform_sha256=sha256(wave_path), inlet_waveform_min=float(vals.min()),
                   inlet_waveform_max=float(vals.max()), inlet_waveform_abs_max=float(np.abs(vals).max()))
    else:
        row.update(inlet_waveform_sha256=None, inlet_waveform_min=None,
                   inlet_waveform_max=None, inlet_waveform_abs_max=None)
    row["report_status"] = report.get("status")
    row["quality_decision"] = "exclude" if unit in QUALITY_EXCLUSIONS else "include"
    row["quality_reason"] = QUALITY_EXCLUSIONS.get(unit, {}).get("reason", "")
    row["watch_flag"] = WATCHLIST.get(unit, "")
    row["pressure_gauge_offset_policy"] = "record_only_not_auto_excluded"
    row["hard_failures"] = hard
    row["hard_qa_pass"] = not hard
    return row


def run(data_root: Path, report_dir: Path) -> dict:
    truth = report_dir / "v4_final_whitelist.json"
    source = json.loads(truth.read_text(encoding="utf-8"))
    units = list(source["AG"]) + list(source["AAA"])
    if len(source["AG"]) != 76 or len(source["AAA"]) != 63 or len(units) != 139:
        raise RuntimeError("geometry signed-off source must be AG76+AAA63")
    rows = [audit_case(data_root, unit) for unit in units]
    hard = [r for r in rows if not r["hard_qa_pass"]]
    if hard:
        raise RuntimeError("training hard QA failed: " + "; ".join(
            f"{r['unit_id']}={r['hard_failures']}" for r in hard[:10]))
    if not set(QUALITY_EXCLUSIONS).issubset(source["AAA"]):
        raise RuntimeError("quality exclusions are not all in signed-off AAA63")
    aaa_train = [u for u in source["AAA"] if u not in QUALITY_EXCLUSIONS]
    ag_trainable = list(source["AG"])
    if len(aaa_train) != 57:
        raise RuntimeError(f"expected AAA training whitelist57, got {len(aaa_train)}")
    generated = datetime.now(timezone.utc).isoformat()
    payload = {
        "schema_version": 1, "generated_at": generated,
        "required_frame_version": FRAME_VERSION,
        "source_final_whitelist": str(truth.resolve()), "source_final_whitelist_sha256": sha256(truth),
        "counts": {"signedoff_candidates": 139, "AG_signedoff": 76, "AAA_signedoff": 63,
                   "hard_failures": 0, "AAA_quality_excluded": 6, "AAA_training": 57},
        "hard_gate": ["missing bundle/report/key", "wrong frame", "node/shape mismatch",
                      "WSS NaN/Inf", "negative WSS", "case/timestep all-zero", "nonpositive WSS >1%"],
        "rows": rows,
    }
    report_dir.mkdir(parents=True, exist_ok=True)
    _write_json_atomic(report_dir / "v4_training_quality_audit.json", payload)
    fields = sorted({key for row in rows for key in row if key != "hard_failures"}) + ["hard_failures"]
    tmp = report_dir / ".v4_training_quality_audit.csv.tmp"
    with tmp.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields); writer.writeheader()
        for row in rows: writer.writerow({**row, "hard_failures": "|".join(row["hard_failures"])})
    tmp.replace(report_dir / "v4_training_quality_audit.csv")
    exclusions = {
        "schema_version": 1, "generated_at": generated, "scope": "derived training-only whitelist",
        "geometry_final_whitelist_unchanged": True, "excluded_units": [
            {"unit_id": unit, **meta, "numeric_evidence": next(r for r in rows if r["unit_id"] == unit)}
            for unit, meta in QUALITY_EXCLUSIONS.items()],
        "watchlist_retained": [{"unit_id": u, "reason": reason} for u, reason in WATCHLIST.items()],
    }
    _write_json_atomic(report_dir / "v4_training_quality_exclusions.json", exclusions)
    whitelist = {
        "schema_version": 1, "generated_at": generated, "required_frame_version": FRAME_VERSION,
        "source_geometry_whitelist": str(truth.resolve()),
        "quality_exclusions_source": str((report_dir / "v4_training_quality_exclusions.json").resolve()),
        "AG": ag_trainable, "AAA": aaa_train,
        "counts": {"AG": len(ag_trainable), "AAA": len(aaa_train), "total": len(ag_trainable)+len(aaa_train)},
    }
    _write_json_atomic(report_dir / "v4_training_whitelist.json", whitelist)
    return payload["counts"]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", type=Path, default=C.OUT_ROOT)
    ap.add_argument("--report-dir", type=Path, required=True)
    args = ap.parse_args()
    print(json.dumps(run(args.data_root.resolve(), args.report_dir.resolve()), indent=2))


if __name__ == "__main__":
    main()

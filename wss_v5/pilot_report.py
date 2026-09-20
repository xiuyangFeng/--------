"""Pilot deliverables (design v0.3 §10.2): per-case module inventory, I/O benchmark, gate table -> audits/<run>/pilot_report.md."""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import h5py
import numpy as np

from . import contract as C


def io_benchmark(h5_path: Path, seed: int = 0) -> dict[str, Any]:
    rng = np.random.default_rng(seed)
    out: dict[str, Any] = {}
    with h5py.File(h5_path, "r") as h5:
        vel = h5["volume_temporal/velocity_m_s"]
        wss = h5["wall_temporal/wss_scalar_pa"]
        T, N = vel.shape[0], vel.shape[1]
        # (a) one random frame, 20k random points (per-step training query pattern)
        t0 = time.perf_counter()
        for _ in range(10):
            f = int(rng.integers(0, T)); idx = np.sort(rng.choice(N, size=min(20_000, N), replace=False))
            _ = vel[f][idx]
        out["frame_gather_20k_ms"] = (time.perf_counter() - t0) / 10 * 1e3
        # (b) full frame read
        t0 = time.perf_counter(); _ = vel[int(rng.integers(0, T))]; out["full_frame_ms"] = (time.perf_counter() - t0) * 1e3
        # (c) full time series for 2k random points (POD / temporal-basis pattern)
        idx = np.sort(rng.choice(N, size=min(2_000, N), replace=False))
        t0 = time.perf_counter(); _ = vel[:, idx, :]; out["full_T_2k_points_ms"] = (time.perf_counter() - t0) * 1e3
        # (d) wall: all frames all nodes
        t0 = time.perf_counter(); _ = wss[()]; out["wall_all_frames_ms"] = (time.perf_counter() - t0) * 1e3
        out["velocity_chunks"] = list(vel.chunks) if vel.chunks else None
        out["compression"] = vel.compression
        raw_bytes = vel.dtype.itemsize * int(np.prod(vel.shape)) + h5["volume_temporal/pressure_pa"].dtype.itemsize * int(np.prod(h5["volume_temporal/pressure_pa"].shape))
        out["volume_temporal_raw_mb"] = raw_bytes / 1e6
        out["volume_temporal_stored_mb"] = (vel.id.get_storage_size() + h5["volume_temporal/pressure_pa"].id.get_storage_size()) / 1e6
    out["file_mb"] = h5_path.stat().st_size / 1e6
    return out


def module_inventory(h5_path: Path) -> dict[str, Any]:
    groups: dict[str, dict[str, Any]] = {}
    with h5py.File(h5_path, "r") as h5:
        def visit(name, obj):
            if isinstance(obj, h5py.Dataset):
                top = name.split("/")[0]
                g = groups.setdefault(top, {"datasets": 0, "bytes": 0, "tags": set()})
                g["datasets"] += 1
                g["bytes"] += obj.id.get_storage_size()
                g["tags"].add(str(obj.attrs.get("dependency", "")))
        h5.visititems(visit)
    return {k: {"datasets": v["datasets"], "mb": v["bytes"] / 1e6, "tags": sorted(v["tags"])} for k, v in groups.items()}


def main(argv=None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=C.SNAPSHOT_ROOT)
    parser.add_argument("--run-name", required=True)
    args = parser.parse_args(argv)
    audit = args.root / "audits" / args.run_name
    gate = json.loads((audit / "gate_report.json").read_text(encoding="utf-8"))
    regression = {}
    reg_path = audit / "regression_vs_wss_min.json"
    if reg_path.is_file():
        regression = {r["canonical_id"]: r for r in json.loads(reg_path.read_text(encoding="utf-8"))}
    lines = [f"# V5 pilot report — {args.run_name}", "", f"schema {C.SCHEMA_VERSION} · gate {gate['gate_pass']}/{gate['cases']} · overall **{gate['gate_result']}**", ""]
    lines += ["## 1. Gate table", "", "见同目录 `summary.md`（逐例检查与失败项）。", ""]
    lines += ["## 2. Module inventory and I/O benchmark", "",
              "| case | file MB | volume raw→stored MB | frame gather 20k ms | full frame ms | full-T 2k pts ms | wall all ms | modules |",
              "| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |"]
    bench_all = []
    for row in gate["rows"]:
        if row["status"] != "built":
            continue
        h5_path = C.case_dir(row["case"], args.root) / "case.h5"
        inv = module_inventory(h5_path)
        bench = io_benchmark(h5_path)
        bench_all.append(bench)
        mods = ", ".join(f"{k}({v['datasets']})" for k, v in sorted(inv.items()))
        lines.append(f"| {row['case']} | {bench['file_mb']:.0f} | {bench['volume_temporal_raw_mb']:.0f}→{bench['volume_temporal_stored_mb']:.0f} | "
                     f"{bench['frame_gather_20k_ms']:.1f} | {bench['full_frame_ms']:.1f} | {bench['full_T_2k_points_ms']:.0f} | {bench['wall_all_frames_ms']:.0f} | {mods} |")
    if bench_all:
        total_mb = sum(b["file_mb"] for b in bench_all)
        ratio = sum(b["volume_temporal_raw_mb"] for b in bench_all) / max(sum(b["volume_temporal_stored_mb"] for b in bench_all), 1e-9)
        lines += ["", f"pilot cases total {total_mb / 1e3:.2f} GB; lzf ratio on volume temporal {ratio:.2f}×; "
                  f"172-case projection ≈ {total_mb / len(bench_all) * 172 / 1e3:.0f} GB (mean-case extrapolation).", ""]
    lines += ["## 3. Parser regression vs legacy WSS-min bundle", "", "| case | compared nodes | wss exact fraction | max |Δwss| Pa | max |Δp| Pa |", "| --- | ---: | ---: | ---: | ---: |"]
    for cid, r in regression.items():
        if r.get("available") and r.get("steps_equal"):
            lines.append(f"| {cid} | {r['compared_nodes']} | {r['wss_exact_fraction']:.4f} | {r['wss_max_abs_diff']:.3g} | {r['pressure_max_abs_diff']:.3g} |")
        else:
            lines.append(f"| {cid} | — | n/a (no legacy bundle) | — | — |")
    lines += ["", "## 4. Point-cloud domain gate detail", "", "| case | recall | ext ≥2 mm false | ext <2 mm band | shell false | gen leak | cap plane offsets mm (inlet, le, li, re, ri) | area calib |",
              "| --- | ---: | ---: | ---: | ---: | ---: | --- | ---: |"]
    for row in gate["rows"]:
        rep_path = C.case_dir(row["case"], args.root) / "report.json"
        if not rep_path.is_file():
            continue
        rep = json.loads(rep_path.read_text(encoding="utf-8"))
        pc = rep.get("metrics", {}).get("pointcloud", {})
        it = pc.get("inside_test", {})
        offs = it.get("cap_offsets", {})
        off = ", ".join(f"{offs.get(k, {}).get('plane_offset_mm', float('nan')):+.1f}" for k in ("inlet", "out-le", "out-li", "out-re", "out-ri"))
        calib = pc.get("cloud", {}).get("winding_calibration", {})
        lines.append(f"| {row['case']} | {it.get('anatomy_cells', {}).get('inside_fraction_hybrid', float('nan')):.4f} | "
                     f"{it.get('extension_cells_near_interface', {}).get('inside_fraction_hybrid', float('nan')):.4f} | "
                     f"{it.get('extension_cells_within_2mm_of_cut', {}).get('inside_fraction_hybrid', float('nan')):.3f} | "
                     f"{it.get('shell_1_3mm_outside', {}).get('inside_fraction_hybrid', float('nan')):.4f} | "
                     f"{pc.get('generated_queries', {}).get('leak_fraction', float('nan')):.4f} | {off} | {calib.get('area_scale', float('nan')):.3f} |")
    (audit / "pilot_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

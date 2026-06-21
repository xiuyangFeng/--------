#!/usr/bin/env python3
"""Prepare V3P I2-PC intrinsic WSS targets.

This creates an isolated graph target copy for the Q0-selected I2-PC probe:
``processed/graphs_i2pc_intrinsic``. Original ``processed/graphs`` files are
left untouched. The generated graphs keep global ``y_wss`` and add
``y_wss_local = [wss, tau_s, tau_theta, tau_n]`` where the intrinsic vector
components are z-scored with train-split wall statistics.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Sequence, Tuple

import numpy as np
import torch
from tqdm import tqdm

from pipeline.config import NODE_FEATURE_NAMES
from pipeline.dataset import load_graph_data
from pipeline.wall_unwrap.grid import compute_theta
from training.core.splits import SplitSpec
from training.scripts.run_v3_f0_decision import REPO_ROOT, _denorm_zscore

NODE_IDX = {n: i for i, n in enumerate(NODE_FEATURE_NAMES)}
X_TAN = slice(NODE_IDX["Tangent_X"], NODE_IDX["Tangent_Z"] + 1)


def _unit(v: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(v, axis=-1, keepdims=True)
    return v / np.maximum(n, 1e-12)


def _load_norm_stats(path: Path) -> Mapping[str, Mapping[str, float]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    return payload.get("statistics", payload)


def _global_wss_phys(data, stats: Mapping[str, Mapping[str, float]]) -> np.ndarray:
    y = data.y_wss.detach().cpu().numpy()
    return np.stack(
        [
            _denorm_zscore(y[:, 1], stats["wss_x"]),
            _denorm_zscore(y[:, 2], stats["wss_y"]),
            _denorm_zscore(y[:, 3], stats["wss_z"]),
        ],
        axis=1,
    ).astype(np.float64)


def _intrinsic_components(data, stats: Mapping[str, Mapping[str, float]]) -> Tuple[np.ndarray, np.ndarray]:
    x = data.x.detach().cpu().numpy().astype(np.float64)
    wall = x[:, NODE_IDX["is_wall"]] > 0.5
    wss_phys = _global_wss_phys(data, stats)
    tangent = _unit(x[:, X_TAN])

    # Keep the same centerline-derived frame convention as Q0 oracle bundle.
    wall_xyz = x[wall, :3]
    wall_tan = tangent[wall]
    _ = compute_theta(wall_xyz, wall_tan)
    t_mean = wall_tan.mean(axis=0)
    t_mean = t_mean / (np.linalg.norm(t_mean) + 1e-12)
    ref = np.array([0.0, 0.0, 1.0]) if abs(t_mean[2]) < 0.9 else np.array([1.0, 0.0, 0.0])
    e1 = ref - (ref @ t_mean) * t_mean
    e1 = e1 / (np.linalg.norm(e1) + 1e-12)
    e2 = np.cross(t_mean, e1)

    e_s = tangent
    e_theta = _unit(np.tile(e2, (x.shape[0], 1)))
    e_n = _unit(np.cross(e_s, e_theta))
    tau_s = np.sum(wss_phys * e_s, axis=1)
    tau_theta = np.sum(wss_phys * e_theta, axis=1)
    tau_n = np.sum(wss_phys * e_n, axis=1)
    return np.stack([tau_s, tau_theta, tau_n], axis=1), wall


def _iter_graphs(data_root: Path, cases: Sequence[str], graphs_subdir: str) -> Iterable[Tuple[str, Path]]:
    for case in cases:
        graph_dir = data_root / case / graphs_subdir
        if not graph_dir.is_dir():
            continue
        for path in sorted(graph_dir.glob("*.pt")):
            yield case, path


def _collect_train_stats(
    data_root: Path,
    train_cases: Sequence[str],
    graphs_subdir: str,
    stats: Mapping[str, Mapping[str, float]],
) -> Dict[str, Dict[str, float]]:
    chunks: List[np.ndarray] = []
    for _, graph_path in tqdm(
        list(_iter_graphs(data_root, train_cases, graphs_subdir)),
        desc="collect train intrinsic stats",
    ):
        data = load_graph_data(graph_path)
        comp, wall = _intrinsic_components(data, stats)
        if bool(np.any(wall)):
            chunks.append(comp[wall])
    if not chunks:
        raise RuntimeError("No train wall intrinsic components collected")
    arr = np.concatenate(chunks, axis=0)
    names = ("wss_axial", "wss_circ", "wss_rad")
    out: Dict[str, Dict[str, float]] = {}
    for i, name in enumerate(names):
        std = float(np.std(arr[:, i]))
        out[name] = {
            "mean": float(np.mean(arr[:, i])),
            "std": std if std > 1e-12 else 1.0,
        }
    return out


def _write_graph_copy(
    graph_path: Path,
    out_path: Path,
    stats_global: Mapping[str, Mapping[str, float]],
    stats_intrinsic: Mapping[str, Mapping[str, float]],
) -> None:
    data = load_graph_data(graph_path)
    comp, _wall = _intrinsic_components(data, stats_global)
    y_local = torch.zeros_like(data.y_wss)
    y_local[:, 0] = data.y_wss[:, 0]
    for col, name in enumerate(("wss_axial", "wss_circ", "wss_rad"), start=1):
        mean = float(stats_intrinsic[name]["mean"])
        std = max(float(stats_intrinsic[name]["std"]), 1e-12)
        y_local[:, col] = torch.as_tensor((comp[:, col - 1] - mean) / std, dtype=data.y_wss.dtype)
    data.y_wss_global = data.y_wss.clone()
    data.y_wss_local = y_local
    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(data, out_path)


def main() -> None:
    ap = argparse.ArgumentParser(description="Prepare V3P I2-PC intrinsic graph targets")
    ap.add_argument("--data-root", type=Path, default=REPO_ROOT / "data_new" / "AG")
    ap.add_argument("--split-file", type=Path, default=REPO_ROOT / "training/splits/split_AG_v1.json")
    ap.add_argument("--graphs-subdir", type=str, default="processed/graphs")
    ap.add_argument("--output-subdir", type=str, default="processed/graphs_i2pc_intrinsic")
    ap.add_argument("--norm-stats", type=Path, default=None)
    ap.add_argument("--manifest", type=Path, default=None)
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()

    data_root = args.data_root.resolve()
    split = SplitSpec.from_json(args.split_file)
    norm_path = args.norm_stats or (data_root / "normalization_params_global.json")
    stats_global = _load_norm_stats(norm_path)

    stats_intrinsic = _collect_train_stats(
        data_root,
        split.train_cases,
        args.graphs_subdir,
        stats_global,
    )

    cases = list(split.train_cases) + list(split.val_cases) + list(split.test_cases)
    n_written = 0
    for case, graph_path in tqdm(
        list(_iter_graphs(data_root, cases, args.graphs_subdir)),
        desc="write i2pc graphs",
    ):
        rel = graph_path.relative_to(data_root / case / args.graphs_subdir)
        out_path = data_root / case / args.output_subdir / rel
        if out_path.exists() and not args.overwrite:
            n_written += 1
            continue
        _write_graph_copy(graph_path, out_path, stats_global, stats_intrinsic)
        n_written += 1

    manifest = args.manifest or (data_root / "i2pc_intrinsic_manifest.json")
    payload = {
        "label": "V3P-I2-PC-intrinsic-targets",
        "split_file": str(args.split_file),
        "data_root": str(data_root),
        "source_graphs_subdir": args.graphs_subdir,
        "output_graphs_subdir": args.output_subdir,
        "normalization_source": str(norm_path),
        "intrinsic_train_stats": stats_intrinsic,
        "n_cases": len(cases),
        "n_graphs_written_or_present": n_written,
        "target_columns": ["wss", "tau_s", "tau_theta", "tau_n"],
        "note": "Q0 I2-PC point-cloud intrinsic frame target; independent from local_v1/TODO-5.",
    }
    manifest.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")

    # Keep a copy close to the generated graphs for run snapshots and cluster jobs.
    sidecar = data_root / "i2pc_intrinsic_norm_stats.json"
    sidecar.write_text(json.dumps(stats_intrinsic, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({"manifest": str(manifest), "n_graphs": n_written}, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""批量导出 graph.pt → 2D 展开网格（.npz）供 G4 训练缓存。"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from training.core.denylist import filter_case_names
from training.core.splits import SplitSpec

from .grid import UnwrapGridConfig, collect_graph_paths, graph_to_2d_sample

REPO_ROOT = Path(__file__).resolve().parents[2]


def main() -> None:
    ap = argparse.ArgumentParser(description="G4 · 批量构建 2D 壁面展开网格")
    ap.add_argument("--split", type=Path, default=REPO_ROOT / "training/splits/split_AG_v1.json")
    ap.add_argument("--data-root", type=Path, default=REPO_ROOT / "data_new/AG")
    ap.add_argument("--subset", choices=["train", "val", "test", "all"], default="train")
    ap.add_argument("--grid-s", type=int, default=64)
    ap.add_argument("--sectors", type=int, default=4)
    ap.add_argument("--output-dir", type=Path, default=REPO_ROOT / "data_new/AG/processed/unwrap2d")
    ap.add_argument("--max-graphs", type=int, default=0, help="0=不限")
    args = ap.parse_args()

    split = SplitSpec.from_json(args.split.resolve())
    if args.subset == "train":
        cases = filter_case_names(split.train_cases, args.data_root)
    elif args.subset == "val":
        cases = filter_case_names(split.val_cases, args.data_root)
    elif args.subset == "test":
        cases = filter_case_names(split.test_cases, args.data_root)
    else:
        cases = filter_case_names(
            list(split.train_cases) + list(split.val_cases) + list(split.test_cases),
            args.data_root,
        )

    cfg = UnwrapGridConfig(grid_s=args.grid_s, n_sectors=args.sectors)
    paths = collect_graph_paths(cases, args.data_root)
    if args.max_graphs > 0:
        paths = paths[: args.max_graphs]

    out_root = args.output_dir.resolve()
    out_root.mkdir(parents=True, exist_ok=True)
    manifest = []
    n_ok = 0
    for gp in paths:
        sample = graph_to_2d_sample(gp, cfg=cfg)
        if sample is None:
            continue
        rel = gp.relative_to(args.data_root.resolve())
        out_path = out_root / rel.with_suffix(".npz")
        out_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            out_path,
            x_feat=sample["x_feat"],
            y_wss_norm=sample["y_wss_norm"],
            y_wss_phys=sample["y_wss_phys"],
            occupied=sample["occupied"],
            bc_inlet=sample["bc_inlet"],
        )
        manifest.append({"graph": str(gp), "npz": str(out_path), "case": sample["case"]})
        n_ok += 1

    meta_path = out_root / f"manifest_{args.subset}.json"
    meta_path.write_text(json.dumps({"n_exported": n_ok, "cfg": cfg.__dict__, "items": manifest}, indent=2), encoding="utf-8")
    print(f"exported {n_ok}/{len(paths)} -> {out_root}")
    print(meta_path)


if __name__ == "__main__":
    main()

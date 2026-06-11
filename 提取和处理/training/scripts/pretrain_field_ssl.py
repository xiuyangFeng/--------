#!/usr/bin/env python3
"""G3 Phase 1 · 几何 SSL 预训练 / 小子集过拟合探针。

默认：post-denylist train 前 3 case · T1 坐标掩码重建 · 50 epoch。
"""
from __future__ import annotations

import argparse
import json
from datetime import date
from pathlib import Path
from typing import Dict, List

import torch

from ..core.data import FieldGraphDataset, build_dataloader, build_feature_mask
from ..core.denylist import filter_case_names
from ..core.splits import SplitSpec
from ..core.ssl_tasks import PointNeXtCoordSSL
from ..core.utils import ensure_dir, resolve_device, set_seed, timestamp

REPO_ROOT = Path(__file__).resolve().parents[2]

SSL_NODE_FEATURES = [
    "x", "y", "z", "Abscissa", "NormRadius", "Curvature",
    "Tangent_X", "Tangent_Y", "Tangent_Z", "is_wall",
]
SSL_GLOBAL_FEATURES = ["t_norm", "BC_Inlet", "BC_O1", "BC_O2", "BC_O3", "BC_O4"]


def _parse_cases(raw: str, split: SplitSpec, data_root: Path) -> List[str]:
    key = raw.strip().lower()
    if key in ("pretrain", "trainval", "ssl"):
        train = filter_case_names(split.train_cases, data_root)
        val = filter_case_names(split.val_cases, data_root)
        return train + val
    if key in ("auto", "overfit3c"):
        train = filter_case_names(split.train_cases, data_root)
        return train[:3]
    return [c.strip() for c in raw.split(",") if c.strip()]


def main() -> None:
    ap = argparse.ArgumentParser(description="V3P G3 SSL 坐标掩码预训练探针")
    ap.add_argument("--split", type=Path, default=REPO_ROOT / "training/splits/split_AG_v1.json")
    ap.add_argument("--data-root", type=Path, default=REPO_ROOT / "data_new/AG")
    ap.add_argument("--cases", default="auto", help="auto/overfit3c=train前3 · pretrain=train+val · 或逗号分隔")
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--mask-ratio", type=float, default=0.25)
    ap.add_argument("--hidden-dim", type=int, default=256)
    ap.add_argument("--num-layers", type=int, default=4)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--output-root", type=Path, default=REPO_ROOT / "outputs/field")
    ap.add_argument("--exp-id", default="V3P-G-58-SSL-Overfit3c")
    ap.add_argument("--run-tag", default="", help="run 目录后缀；pretrain 时默认 pretrain")
    args = ap.parse_args()

    set_seed(args.seed)
    device = resolve_device(args.device)
    split = SplitSpec.from_json(args.split.resolve())
    data_root = args.data_root.resolve()
    cases = _parse_cases(args.cases, split, data_root)
    if len(cases) < 1:
        raise SystemExit("无可用 case")

    case_set = args.cases.strip().lower()
    is_pretrain = case_set in ("pretrain", "trainval", "ssl")
    run_tag = args.run_tag or ("pretrain" if is_pretrain else "overfit3c")
    phase = "G3-Phase2-pretrain" if is_pretrain else "G3-Phase1-overfit3c"

    feature_mask = build_feature_mask(SSL_NODE_FEATURES, SSL_GLOBAL_FEATURES)
    dataset = FieldGraphDataset(
        root=str(data_root),
        case_names=cases,
        graphs_subdir="processed/graphs",
        augment=False,
        feature_mask=feature_mask,
        required_keys={"x", "global_cond", "edge_index"},
    )
    loader = build_dataloader(dataset, batch_size=1, shuffle=True, num_workers=0, seed=args.seed)

    model = PointNeXtCoordSSL(
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        dropout=0.1,
    ).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    gen = torch.Generator()
    gen.manual_seed(args.seed)

    run_dir = ensure_dir(
        args.output_root
        / f"ssl_v3p_g58_{run_tag}_{split.split_version}_seed{args.seed}_{timestamp()}"
    )
    history: List[Dict[str, float]] = []
    final_loss = None

    for epoch in range(1, args.epochs + 1):
        model.train()
        epoch_loss = 0.0
        n_batches = 0
        for batch in loader:
            batch = batch.to(device)
            opt.zero_grad(set_to_none=True)
            loss, _ = model(batch, mask_ratio=args.mask_ratio, generator=gen)
            loss.backward()
            opt.step()
            epoch_loss += float(loss.detach().cpu())
            n_batches += 1
        mean_loss = epoch_loss / max(n_batches, 1)
        final_loss = mean_loss
        history.append({"epoch": epoch, "loss": mean_loss})
        if epoch == 1 or epoch % 10 == 0 or epoch == args.epochs:
            print(f"[SSL] ep {epoch}/{args.epochs} loss={mean_loss:.6f}", flush=True)

    torch.save(model.state_dict(), run_dir / "ssl_checkpoint.pt")
    torch.save(model.encoder.state_dict(), run_dir / "ssl_encoder.pt")
    summary = {
        "exp_id": args.exp_id,
        "phase": phase,
        "case_set": case_set,
        "cases": cases,
        "n_cases": len(cases),
        "n_graphs": len(dataset),
        "epochs": args.epochs,
        "final_loss": final_loss,
        "initial_loss": history[0]["loss"] if history else None,
        "loss_drop_ratio": (
            (history[0]["loss"] - final_loss) / max(history[0]["loss"], 1e-12)
            if history and final_loss is not None
            else None
        ),
        "go_overfit": bool(final_loss is not None and final_loss < 0.05),
        "mask_ratio": args.mask_ratio,
        "device": str(device),
        "created": str(date.today()),
        "checkpoints": {
            "full": str(run_dir / "ssl_checkpoint.pt"),
            "encoder": str(run_dir / "ssl_encoder.pt"),
        },
    }
    (run_dir / "history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")
    (run_dir / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    print(run_dir)


if __name__ == "__main__":
    main()

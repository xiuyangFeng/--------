"""One-off: compute test-set WSS RMSE/R² for runs listed in experiment_index."""
from __future__ import annotations

import csv
import sys
from pathlib import Path

import torch

from pipeline.config import NODE_FEATURE_NAMES
from pipeline.dataset import load_graph_data

from training.core.config import ExperimentConfig
from training.core.data import (
    FieldGraphDataset,
    build_dataloader,
    build_feature_mask,
    build_required_data_keys,
)
from training.core.io import load_checkpoint
from training.core.metrics import WSSMeter
from training.core.models import build_model, split_model_output
from training.core.splits import SplitSpec
from training.core.utils import resolve_device, set_seed

ROOT = Path(__file__).resolve().parents[2]
want_ids = {
    "A-Base-01-wss-multi",
    "A-Base-02-wss-multi",
    "A-Base-03-wss-multi",
    "A-Main-01-wss-multi",
    "A-Opt-05-wss-multi",
}


def _precompute_wall_masks(data_files: list[Path]) -> dict[str, torch.Tensor]:
    """每个测试图读一次 is_wall，供所有实验复用（避免每个 batch 反复 load_graph_data）。"""
    iw = NODE_FEATURE_NAMES.index("is_wall")
    out: dict[str, torch.Tensor] = {}
    n = len(data_files)
    for i, path in enumerate(data_files):
        if (i + 1) % 300 == 0 or i == 0:
            print(f"  wall masks: {i + 1}/{n}", file=sys.stderr, flush=True)
        data = load_graph_data(path)
        out[str(path.resolve())] = data.x[:, iw].bool().cpu()
    return out


def _batched_wall_mask(
    batch,
    device: torch.device,
    wall_index: dict[str, torch.Tensor],
) -> torch.Tensor:
    """壁面掩码来自磁盘上的完整节点特征，而非 masked 后的 batch.x。"""
    parts: list[torch.Tensor] = []
    for p in batch.graph_path:
        key = str(Path(p).resolve())
        parts.append(wall_index[key])
    wall = torch.cat(parts, dim=0).to(device)
    return wall.unsqueeze(-1).float()


def main() -> None:
    rows = []
    with open(ROOT / "outputs/field/experiment_index.csv", newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if row["exp_id"] in want_ids:
                rows.append(row)

    by_key: dict[tuple[str, str], dict] = {}
    for row in rows:
        by_key[(row["exp_id"], row["seed"])] = row
    rows = list(by_key.values())
    rows.sort(key=lambda r: (r["exp_id"], int(r["seed"])))

    device = resolve_device("auto")
    print("device:", device, file=sys.stderr, flush=True)

    # 各 wss-multitask 实验共用同一 split / data_root / graphs_subdir，测试集文件列表一致。
    probe = ExperimentConfig.from_json(ROOT / rows[0]["run_dir"] / "config.snapshot.json")
    probe.validate()
    split0 = SplitSpec.from_json(ROOT / probe.data.split_file)
    ref_dataset = FieldGraphDataset(
        root=str(ROOT / probe.data.data_root),
        case_names=split0.test_cases,
        graphs_subdir=probe.data.graphs_subdir,
        augment=False,
        preload=False,
        feature_mask=None,
        required_keys={"x", "y", "y_wss", "global_cond"},
    )
    print(
        "precompute wall masks for",
        len(ref_dataset.data_files),
        "test graphs …",
        file=sys.stderr,
        flush=True,
    )
    wall_index = _precompute_wall_masks(ref_dataset.data_files)

    results: list[dict] = []
    for row in rows:
        run_dir = ROOT / row["run_dir"]
        cfg_path = run_dir / "config.snapshot.json"
        ckpt = run_dir / "best_model.pt"
        if not cfg_path.is_file() or not ckpt.is_file():
            print("SKIP missing", run_dir, file=sys.stderr, flush=True)
            continue

        config = ExperimentConfig.from_json(cfg_path)
        config.validate()
        split = SplitSpec.from_json(ROOT / config.data.split_file)
        if split.test_cases != split0.test_cases or config.data.graphs_subdir != probe.data.graphs_subdir:
            print("WARN split mismatch, rebuild wall index for", row["exp_id"], file=sys.stderr, flush=True)
        set_seed(config.system.seed, deterministic=config.system.deterministic)

        feature_mask = build_feature_mask(
            enabled_node_features=config.data.enabled_node_features,
            enabled_global_features=config.data.enabled_global_features,
        )
        required_keys = build_required_data_keys(config.model.name, wss_dim=config.model.wss_dim)
        test_dataset = FieldGraphDataset(
            root=str(ROOT / config.data.data_root),
            case_names=split.test_cases,
            graphs_subdir=config.data.graphs_subdir,
            augment=False,
            preload=config.data.preload,
            feature_mask=feature_mask,
            required_keys=required_keys,
        )
        if [str(p.resolve()) for p in test_dataset.data_files] != [
            str(p.resolve()) for p in ref_dataset.data_files
        ]:
            raise RuntimeError("test set 文件列表与探针实验不一致，请检查配置")
        infer_bs = max(int(config.data.batch_size), 8)
        test_loader = build_dataloader(
            test_dataset,
            batch_size=infer_bs,
            shuffle=False,
            num_workers=config.data.num_workers,
            pin_memory=config.data.pin_memory,
        )

        model = build_model(
            model_name=config.model.name,
            hidden_dim=config.model.hidden_dim,
            num_layers=config.model.num_layers,
            dropout=config.model.dropout,
            heads=config.model.heads,
            use_transformer_prenorm=config.model.use_transformer_prenorm,
            wss_dim=config.model.wss_dim,
        ).to(device)
        load_checkpoint(model, ckpt, device)
        model.eval()

        meter = WSSMeter()
        with torch.no_grad():
            for batch in test_loader:
                batch = batch.to(device)
                _, wss_pred = split_model_output(model(batch))
                if wss_pred is None:
                    continue
                y_wss = batch.y_wss
                if y_wss is None:
                    continue
                is_w = _batched_wall_mask(batch, device, wall_index)
                meter.update(wss_pred, y_wss, is_w)

        m = meter.compute()
        results.append({"exp_id": row["exp_id"], "seed": int(row["seed"]), "model": config.model.name, **m})
        print(row["exp_id"], "seed", row["seed"], m, file=sys.stderr, flush=True)

    keys = [
        "exp_id",
        "seed",
        "model",
        "wss_rmse",
        "wss_rmse_wss",
        "wss_r2_wss",
        "wss_rmse_wss_x",
        "wss_r2_wss_x",
        "wss_rmse_wss_y",
        "wss_r2_wss_y",
        "wss_rmse_wss_z",
        "wss_r2_wss_z",
    ]
    print("\t".join(keys), flush=True)
    for r in results:
        print("\t".join(str(r.get(k, "")) for k in keys), flush=True)


if __name__ == "__main__":
    main()

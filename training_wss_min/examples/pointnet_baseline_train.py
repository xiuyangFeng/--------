#!/usr/bin/env python3
"""250 行内的自包含 PointNet WSS 训练：不调用项目内训练模块。"""
from __future__ import annotations

import argparse, json, logging, math, time
from pathlib import Path
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = ROOT / "training_wss_min/configs/baseline_2x3/pointnet_xyzgeom.json"
DEFAULT_OUTPUT = ROOT / "training_wss_min/runs/teacher_pointnet_xyzgeom"


def path_of(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def normalize_wss(wss: np.ndarray, stats: dict) -> np.ndarray:
    if stats["method"] == "log_z":
        value = np.log(np.clip(wss, 0, None) + stats["eps"])
        return (value - stats["log"]["mean"]) / stats["log"]["std"]
    return (wss - stats["linear"]["mean"]) / stats["linear"]["std"]


def denormalize_wss(value: np.ndarray, stats: dict) -> np.ndarray:
    if stats["method"] == "log_z":
        value = np.exp(value * stats["log"]["std"] + stats["log"]["mean"])
        return np.clip(value - stats["eps"], 0, None)
    return value * stats["linear"]["std"] + stats["linear"]["mean"]


def load_cases(split_path: Path, part: str, stats: dict) -> list[dict]:
    split = json.loads(split_path.read_text())
    cases = []
    for label in split[f"{part}_cases"]:
        subset, name = label.split("/", 1)
        bundle = ROOT / "data_wss_min/AG" / subset / name / "bundle.npz"
        if not bundle.is_file():
            raise FileNotFoundError(bundle)
        with np.load(bundle, allow_pickle=True) as data:
            steps, peak = data["steps"].tolist(), int(data["peak_step"])
            raw = data["wall_wss"][steps.index(peak)].astype(np.float32)
            cases.append({
                "label": label, "pos": data["wall_coords_norm"].astype(np.float32),
                "abscissa_norm": data["wall_abscissa_norm"].astype(np.float32),
                "local_radius": data["wall_local_radius"].astype(np.float32),
                "curvature": data["wall_curvature"].astype(np.float32),
                "y_raw": raw, "y": normalize_wss(raw, stats).astype(np.float32),
            })
    return cases


def transformed(name: str, value: np.ndarray) -> np.ndarray:
    value = np.asarray(value, np.float64)
    return np.sign(value) * np.log1p(np.abs(value)) if name == "curvature" else value


def feature_statistics(cases: list[dict], names: list[str]) -> dict:
    result = {}
    for name in names:
        if name in "xyz" and len(name) == 1:
            continue
        value = transformed(name, np.concatenate([case[name] for case in cases]))
        value = value[np.isfinite(value)]
        clip = float(np.percentile(np.abs(value), 99)) if name == "curvature" else None
        if clip is not None:
            value = np.clip(value, -clip, clip)
        result[name] = {"mean": float(value.mean()), "std": float(value.std() + 1e-6),
                        "clip": clip}
    return result


def make_features(case: dict, indices: np.ndarray, names: list[str], stats: dict) -> np.ndarray:
    columns = []
    for name in names:
        if name in ("x", "y", "z"):
            columns.append(case["pos"][indices, "xyz".index(name)])
        else:
            value, item = transformed(name, case[name][indices]), stats[name]
            if item["clip"] is not None:
                value = np.clip(value, -item["clip"], item["clip"])
            columns.append((value - item["mean"]) / item["std"])
    return np.stack(columns, axis=1).astype(np.float32)


def farthest_point_sample(points: np.ndarray, count: int, seed: int) -> np.ndarray:
    if count <= 0 or count >= len(points):
        return np.arange(len(points))
    rng, selected = np.random.default_rng(seed), np.empty(count, np.int64)
    selected[0] = rng.integers(len(points))
    distance = np.linalg.norm(points - points[selected[0]], axis=1)
    for index in range(1, count):
        selected[index] = int(np.argmax(distance))
        distance = np.minimum(distance, np.linalg.norm(points - points[selected[index]], axis=1))
    return selected


class CaseDataset(Dataset):
    def __init__(self, cases: list[dict], names: list[str], stats: dict,
                 points: int, seed: int):
        self.cases, self.names, self.stats = cases, names, stats
        for index, case in enumerate(cases):
            case["sample"] = farthest_point_sample(case["pos"], points, seed + 7919 * index)

    def __len__(self): return len(self.cases)

    def __getitem__(self, index: int) -> dict:
        case, idx = self.cases[index], self.cases[index]["sample"]
        return {"pos": torch.from_numpy(case["pos"][idx]),
                "x": torch.from_numpy(make_features(case, idx, self.names, self.stats)),
                "y": torch.from_numpy(case["y"][idx])}


def collate(items: list[dict]) -> dict:
    return {"pos": torch.cat([x["pos"] for x in items]),
            "x": torch.cat([x["x"] for x in items]),
            "y": torch.cat([x["y"] for x in items]),
            "batch": torch.cat([torch.full((len(x["y"]),), i, dtype=torch.long)
                                for i, x in enumerate(items)])}


def mlp(channels: list[int], last_act: bool = True) -> nn.Sequential:
    layers = []
    for index, (left, right) in enumerate(zip(channels[:-1], channels[1:])):
        layers.append(nn.Linear(left, right))
        if index < len(channels) - 2 or last_act:
            layers.extend((nn.BatchNorm1d(right), nn.ReLU(inplace=True)))
    return nn.Sequential(*layers)


class PointNetWSS(nn.Module):
    """共享 MLP 编码、病例级最大池化、逐点解码。"""
    def __init__(self, in_dim: int, width: int = 32, head_dim: int = 64, dropout: float = 0):
        super().__init__()
        local = width * 4
        self.local, self.decoder = mlp([in_dim, width, width * 2, local]), mlp([local * 2, local, head_dim])
        self.dropout, self.out = nn.Dropout(dropout) if dropout else nn.Identity(), nn.Linear(head_dim, 1)

    def forward(self, pos: torch.Tensor, x: torch.Tensor, batch: torch.Tensor) -> torch.Tensor:
        del pos
        local, n_case = self.local(x), int(batch.max()) + 1
        global_feature = torch.full((n_case, local.shape[1]), -torch.inf, device=x.device, dtype=local.dtype)
        global_feature.scatter_reduce_(0, batch[:, None].expand_as(local), local, reduce="amax")
        return self.out(self.dropout(self.decoder(torch.cat((local, global_feature[batch]), 1)))).squeeze(1)


def r2(true: np.ndarray, pred: np.ndarray) -> float:
    residual, total = np.sum((true - pred) ** 2), np.sum((true - true.mean()) ** 2)
    return float(1 - residual / total) if total > 1e-12 else float("nan")


@torch.no_grad()
def validate(model: nn.Module, cases: list[dict], names: list[str], feature_stats: dict,
             target_stats: dict, device: str) -> dict:
    model.eval(); true_all, pred_all = [], []
    for case in cases:
        idx = np.arange(len(case["pos"])); x = torch.from_numpy(make_features(case, idx, names, feature_stats)).to(device)
        pos = torch.from_numpy(case["pos"]).to(device); batch = torch.zeros(len(pos), dtype=torch.long, device=device)
        pred = denormalize_wss(model(pos, x, batch).cpu().numpy(), target_stats)
        true_all.append(case["y_raw"].astype(np.float64)); pred_all.append(pred)
        logging.info("验证 %s：R²=%.4f", case["label"], r2(true_all[-1], pred))
    mean = float(np.mean([value.mean() for value in true_all]))
    mse = np.mean([np.mean((a - b) ** 2) for a, b in zip(true_all, pred_all)])
    var = np.mean([np.mean((a - mean) ** 2) for a in true_all])
    return {"r2_field": r2(np.concatenate(true_all), np.concatenate(pred_all)),
            "r2_casebalanced": float(1 - mse / var), "rmse_casebalanced": float(np.sqrt(mse))}


def lr_factor(epoch: int, train: dict) -> float:
    if epoch < train["warmup_epochs"]: return (epoch + 1) / train["warmup_epochs"]
    progress = (epoch - train["warmup_epochs"]) / max(1, train["epochs"] - train["warmup_epochs"])
    return (train["min_lr"] + (train["lr"] - train["min_lr"]) * .5 * (1 + math.cos(math.pi * progress))) / train["lr"]


def main() -> None:
    parser = argparse.ArgumentParser(description="自包含 PointNet WSS 完整训练")
    parser.add_argument("--config", default=str(DEFAULT_CONFIG)); parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT))
    parser.add_argument("--epochs", type=int); args = parser.parse_args()
    cfg = json.loads(path_of(args.config).read_text()); data, model_cfg, train = cfg["data"], cfg["model"], cfg["train"]
    if model_cfg["name"] != "pointnet" or data["target"] != "wss" or train["loss"] != "mse":
        raise ValueError("本文件仅允许 PointNet、峰值 WSS 和无加权 MSE")
    if args.epochs is not None: train["epochs"] = args.epochs
    np.random.seed(train["seed"]); torch.manual_seed(train["seed"]); torch.cuda.manual_seed_all(train["seed"])
    out, device = path_of(args.output_dir), "cuda" if torch.cuda.is_available() else "cpu"; out.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s", handlers=[logging.FileHandler(out / "train.log"), logging.StreamHandler()], force=True)
    target_stats = json.loads(path_of(data["wss_stats_path"]).read_text()); split = path_of(data["split_path"])
    train_cases, val_cases = load_cases(split, "train", target_stats), load_cases(split, "val", target_stats)
    names = list(data["input_features"]); feature_stats = feature_statistics(train_cases, names)
    dataset = CaseDataset(train_cases, names, feature_stats, data["wall_n_points"], train["seed"])
    loader = DataLoader(dataset, train["batch_cases"], shuffle=True, collate_fn=collate,
                        num_workers=data["num_workers"], generator=torch.Generator().manual_seed(train["seed"]))
    model = PointNetWSS(len(names), model_cfg["width"], model_cfg["head_hidden"], model_cfg["dropout"]).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=train["lr"], weight_decay=train["weight_decay"])
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda e: lr_factor(e, train))
    use_amp = train["amp"] and device == "cuda"; scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    (out / "config.json").write_text(json.dumps(cfg, indent=2)); (out / "feature_stats.json").write_text(json.dumps(feature_stats, indent=2))
    history, best, stale, start = out / "history.jsonl", -math.inf, 0, time.time(); history.write_text("")
    for epoch in range(train["epochs"]):
        model.train(); losses = []
        for batch in loader:
            optimizer.zero_grad(set_to_none=True); batch = {k: v.to(device) for k, v in batch.items()}
            with torch.amp.autocast("cuda", enabled=use_amp): loss = (model(batch["pos"], batch["x"], batch["batch"]) - batch["y"]).square().mean()
            scaler.scale(loss).backward(); scaler.unscale_(optimizer); nn.utils.clip_grad_norm_(model.parameters(), train["grad_clip"])
            scaler.step(optimizer); scaler.update(); losses.append(float(loss));
        scheduler.step(); record = {"epoch": epoch, "train_loss": float(np.mean(losses)), "lr": optimizer.param_groups[0]["lr"]}
        do_val = (epoch + 1) % train["eval_every"] == 0 or epoch + 1 == train["epochs"]
        if do_val:
            metrics = validate(model, val_cases, names, feature_stats, target_stats, device); record.update(metrics)
            improved = np.isfinite(metrics["r2_casebalanced"]) and metrics["r2_casebalanced"] > best
            stale = 0 if improved else stale + 1
            if improved: best = metrics["r2_casebalanced"]; torch.save({"model": model.state_dict(), "epoch": epoch, "score": best}, out / "ckpt_best.pt")
            (out / "val_metrics.json").write_text(json.dumps(metrics, indent=2)); logging.info("epoch=%d loss=%.5f val_R²_cb=%.4f", epoch + 1, record["train_loss"], metrics["r2_casebalanced"])
        with history.open("a") as stream: stream.write(json.dumps(record) + "\n")
        torch.save({"model": model.state_dict(), "epoch": epoch}, out / "ckpt_last.pt")
        if do_val and epoch + 1 >= train["min_epoch"] and stale >= train["early_stop_patience"]: break
    if not (out / "ckpt_best.pt").exists(): torch.save({"model": model.state_dict(), "epoch": epoch}, out / "ckpt_best.pt")
    logging.info("完成：best_R²_cb=%.4f，耗时=%.1f 分钟", best, (time.time() - start) / 60)


if __name__ == "__main__": main()

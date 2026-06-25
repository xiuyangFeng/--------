"""Plot CROWN baseline train/val loss curves from history.csv."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Dict, List, Optional

import matplotlib.pyplot as plt
import numpy as np

plt.rcParams["font.sans-serif"] = [
    "Noto Sans CJK SC",
    "Noto Sans CJK JP",
    "DejaVu Sans",
]
plt.rcParams["axes.unicode_minus"] = False


def _read_history(csv_path: Path) -> Dict[str, List[float]]:
    with csv_path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
    if not rows:
        raise ValueError(f"history.csv 为空: {csv_path}")

    history: Dict[str, List[float]] = {}
    for key in reader.fieldnames or []:
        values: List[float] = []
        for row in rows:
            raw = row.get(key, "")
            try:
                values.append(float(raw))
            except (TypeError, ValueError):
                continue
        history[key] = values
    return history


def _load_best_epoch(run_dir: Path, history: Dict[str, List[float]]) -> tuple[int, float]:
    manifest_path = run_dir / "manifest.json"
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        best_epoch = manifest.get("best_epoch")
        best_val = manifest.get("best_val_loss")
        if best_epoch is not None and best_val is not None:
            return int(best_epoch), float(best_val)

    val_arr = np.array(history["val_loss"], dtype=float)
    best_idx = int(np.argmin(val_arr))
    return best_idx + 1, float(val_arr[best_idx])


def plot_train_val_loss(
    history_csv: Path,
    save_path: Path,
    *,
    title: str,
    job_id: str,
    variant_label: str,
    best_epoch: Optional[int] = None,
) -> tuple[int, float]:
    history = _read_history(history_csv)
    if "train_loss" not in history or "val_loss" not in history:
        raise KeyError(f"history.csv 缺少 train_loss/val_loss: {history_csv}")

    run_dir = history_csv.parent
    manifest_best_epoch, manifest_best_val = _load_best_epoch(run_dir, history)
    if best_epoch is None:
        best_epoch = manifest_best_epoch
    best_idx = best_epoch - 1

    train = np.array(history["train_loss"], dtype=float)
    val = np.array(history["val_loss"], dtype=float)
    epochs = np.arange(1, len(train) + 1)

    fig, ax = plt.subplots(figsize=(8.5, 4.8))
    ax.plot(epochs, train, label="训练损失", color="#3182bd", linewidth=1.8, alpha=0.95)
    ax.plot(epochs, val, label="验证损失", color="#e6550d", linewidth=1.8, alpha=0.95)

    best_val = float(val[best_idx]) if 0 <= best_idx < len(val) else manifest_best_val
    ax.axvline(best_epoch, color="gray", linestyle=":", linewidth=1.0, alpha=0.85)
    ax.scatter([best_epoch], [best_val], color="#e6550d", s=45, zorder=6, clip_on=False)
    ax.text(
        best_epoch + len(epochs) * 0.015,
        best_val,
        f"  best={best_val:.5f}\n  @epoch {best_epoch}",
        fontsize=8.5,
        va="center",
        color="gray",
    )

    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.set_yscale("log")
    ax.set_title(f"{title}\n{variant_label} · Job {job_id}")
    ax.legend(loc="upper right")
    ax.grid(True, which="both", linestyle="--", linewidth=0.4, alpha=0.45)

    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(save_path, dpi=300, bbox_inches="tight")
    if save_path.suffix.lower() == ".png":
        svg_path = save_path.with_suffix(".svg")
        fig.savefig(svg_path, bbox_inches="tight")
    plt.close(fig)
    return best_epoch, best_val


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot CROWN train/val loss curves.")
    parser.add_argument("--history-csv", type=Path, required=True)
    parser.add_argument("--save-path", type=Path, required=True)
    parser.add_argument("--title", type=str, required=True)
    parser.add_argument("--job-id", type=str, required=True)
    parser.add_argument("--variant-label", type=str, required=True)
    parser.add_argument("--best-epoch", type=int, default=None)
    args = parser.parse_args()

    best_epoch, best_val = plot_train_val_loss(
        args.history_csv,
        args.save_path,
        title=args.title,
        job_id=args.job_id,
        variant_label=args.variant_label,
        best_epoch=args.best_epoch,
    )
    print(f"saved: {args.save_path}")
    if args.save_path.suffix.lower() == ".png":
        print(f"saved: {args.save_path.with_suffix('.svg')}")
    print(f"best_epoch={best_epoch} best_val_loss={best_val:.8f}")


if __name__ == "__main__":
    main()

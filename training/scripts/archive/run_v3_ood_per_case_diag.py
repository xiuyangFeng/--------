"""
V3 OOD 逐病例诊断脚本
目的：定位哪些测试病例导致 WSS/p/NormRadius 极端 OOD。
输出：outputs/field/diagnostics/v3_ood_per_case/

用法：
  python -m training.scripts.run_v3_ood_per_case_diag
"""

import json
import sys
from pathlib import Path
from collections import defaultdict

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


DATA_ROOT = Path("data_new/AG")
SPLIT_FILE = Path("training/splits/split_AG_v1.json")
OUT_DIR = Path("outputs/field/diagnostics/v3_ood_per_case")

IS_WALL_COL = 9
NORMRADIUS_COL = 4
CURVATURE_COL = 5

Y_CHANNELS = ["u", "v", "w", "p"]
WSS_CHANNELS = ["wss", "wss_x", "wss_y", "wss_z"]


def load_split():
    with open(SPLIT_FILE, encoding="utf-8") as f:
        s = json.load(f)
    return s["train_cases"], s.get("val_cases", []), s["test_cases"]


def get_graph_paths(case_name: str):
    case_dir = DATA_ROOT / case_name / "processed" / "graphs"
    if not case_dir.exists():
        return []
    return sorted(case_dir.glob("*.pt"))


def compute_case_stats(case_name: str, max_timesteps: int = 10):
    paths = get_graph_paths(case_name)
    if not paths:
        return None

    step = max(1, len(paths) // max_timesteps)
    sampled = paths[::step][:max_timesteps]

    wall_wss_all = []
    wall_p_all = []
    int_p_all = []
    int_vel_all = []
    normradius_all = []
    bc_all = []
    n_wall_total = 0
    n_int_total = 0

    for pt_path in sampled:
        g = torch.load(pt_path, weights_only=False)
        is_wall = g.x[:, IS_WALL_COL].bool()
        wall_mask = is_wall
        int_mask = ~is_wall

        n_wall_total += wall_mask.sum().item()
        n_int_total += int_mask.sum().item()

        if wall_mask.any():
            wall_wss_all.append(g.y_wss[wall_mask].numpy())
            wall_p_all.append(g.y[wall_mask, 3:4].numpy())

        if int_mask.any():
            int_p_all.append(g.y[int_mask, 3:4].numpy())
            int_vel_all.append(g.y[int_mask, :3].numpy())

        normradius_all.append(g.x[:, NORMRADIUS_COL].numpy())
        bc_all.append(g.global_cond.numpy().flatten())

    stats = {
        "case": case_name,
        "n_timesteps_sampled": len(sampled),
        "n_timesteps_total": len(paths),
        "n_wall_points": n_wall_total,
        "n_interior_points": n_int_total,
    }

    if wall_wss_all:
        wss = np.concatenate(wall_wss_all, axis=0)
        for i, ch in enumerate(WSS_CHANNELS):
            col = wss[:, i]
            stats[f"wall_{ch}_mean"] = float(np.mean(col))
            stats[f"wall_{ch}_std"] = float(np.std(col))
            stats[f"wall_{ch}_absmax"] = float(np.max(np.abs(col)))
            stats[f"wall_{ch}_q99"] = float(np.quantile(np.abs(col), 0.99))
            stats[f"wall_{ch}_q95"] = float(np.quantile(np.abs(col), 0.95))

    if wall_p_all:
        wp = np.concatenate(wall_p_all, axis=0).flatten()
        stats["wall_p_mean"] = float(np.mean(wp))
        stats["wall_p_std"] = float(np.std(wp))
        stats["wall_p_absmax"] = float(np.max(np.abs(wp)))
        stats["wall_p_q99"] = float(np.quantile(np.abs(wp), 0.99))

    if int_p_all:
        ip = np.concatenate(int_p_all, axis=0).flatten()
        stats["int_p_mean"] = float(np.mean(ip))
        stats["int_p_std"] = float(np.std(ip))
        stats["int_p_absmax"] = float(np.max(np.abs(ip)))
        stats["int_p_q99"] = float(np.quantile(np.abs(ip), 0.99))

    if int_vel_all:
        vel = np.concatenate(int_vel_all, axis=0)
        vel_mag = np.linalg.norm(vel, axis=1)
        stats["int_vel_mag_mean"] = float(np.mean(vel_mag))
        stats["int_vel_mag_std"] = float(np.std(vel_mag))
        stats["int_vel_mag_absmax"] = float(np.max(vel_mag))

    nr = np.concatenate(normradius_all)
    stats["normradius_max"] = float(np.max(nr))
    stats["normradius_q99"] = float(np.quantile(nr, 0.99))
    stats["normradius_q95"] = float(np.quantile(nr, 0.95))
    stats["normradius_mean"] = float(np.mean(nr))

    if bc_all:
        bc = np.stack(bc_all, axis=0)
        bc_labels = ["t_norm", "BC_Inlet", "BC_O1", "BC_O2", "BC_O3", "BC_O4"]
        for i, name in enumerate(bc_labels):
            stats[f"bc_{name}_max"] = float(np.max(np.abs(bc[:, i])))
            stats[f"bc_{name}_mean"] = float(np.mean(bc[:, i]))

    return stats


def make_comparison_plots(train_stats, test_stats, out_dir: Path):
    metrics = [
        ("wall_wss_absmax", "Wall WSS |max| (z-score)"),
        ("wall_wss_q99", "Wall WSS |q99| (z-score)"),
        ("wall_p_absmax", "Wall Pressure |max| (z-score)"),
        ("int_p_absmax", "Interior Pressure |max| (z-score)"),
        ("normradius_max", "NormRadius max"),
        ("bc_BC_O1_max", "BC_O1 |max|"),
    ]

    for metric_key, title in metrics:
        train_vals = [(s["case"], s.get(metric_key, 0)) for s in train_stats if metric_key in s]
        test_vals = [(s["case"], s.get(metric_key, 0)) for s in test_stats if metric_key in s]

        if not train_vals and not test_vals:
            continue

        fig, ax = plt.subplots(figsize=(14, 6))

        train_names = [v[0].split("/")[-1] for v in train_vals]
        train_values = [v[1] for v in train_vals]
        test_names = [v[0].split("/")[-1] for v in test_vals]
        test_values = [v[1] for v in test_vals]

        x_train = np.arange(len(train_names))
        x_test = np.arange(len(train_names), len(train_names) + len(test_names))

        ax.bar(x_train, train_values, color="steelblue", alpha=0.7, label="Train")
        ax.bar(x_test, test_values, color="tomato", alpha=0.7, label="Test")

        all_names = train_names + test_names
        ax.set_xticks(range(len(all_names)))
        ax.set_xticklabels(all_names, rotation=90, fontsize=6)
        ax.set_ylabel(title)
        ax.set_title(f"{title} — Train vs Test (per case)")
        ax.legend()
        ax.axvline(len(train_names) - 0.5, color="gray", linestyle="--", linewidth=0.8)

        fig.tight_layout()
        safe_key = metric_key.replace("|", "")
        fig.savefig(out_dir / f"fig_per_case_{safe_key}.png", dpi=150)
        plt.close(fig)


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    train_cases, val_cases, test_cases = load_split()

    print(f"Train: {len(train_cases)} cases, Val: {len(val_cases)}, Test: {len(test_cases)}")

    fast_slow = {"train": defaultdict(int), "test": defaultdict(int), "val": defaultdict(int)}
    for c in train_cases:
        fast_slow["train"][c.split("/")[0]] += 1
    for c in test_cases:
        fast_slow["test"][c.split("/")[0]] += 1
    for c in val_cases:
        fast_slow["val"][c.split("/")[0]] += 1
    print(f"Split fast/slow breakdown:")
    for split_name, counts in fast_slow.items():
        total = sum(counts.values())
        parts = ", ".join(f"{k}={v} ({v/total*100:.0f}%)" for k, v in sorted(counts.items()))
        print(f"  {split_name}: {parts}")

    print("\n=== Scanning train cases ===")
    train_stats = []
    for i, case in enumerate(train_cases):
        sys.stdout.write(f"\r  [{i+1}/{len(train_cases)}] {case}          ")
        sys.stdout.flush()
        s = compute_case_stats(case)
        if s:
            train_stats.append(s)
    print()

    print("=== Scanning test cases ===")
    test_stats = []
    for i, case in enumerate(test_cases):
        sys.stdout.write(f"\r  [{i+1}/{len(test_cases)}] {case}          ")
        sys.stdout.flush()
        s = compute_case_stats(case)
        if s:
            test_stats.append(s)
    print()

    print("=== Scanning val cases ===")
    val_stats = []
    for i, case in enumerate(val_cases):
        sys.stdout.write(f"\r  [{i+1}/{len(val_cases)}] {case}          ")
        sys.stdout.flush()
        s = compute_case_stats(case)
        if s:
            val_stats.append(s)
    print()

    key_metrics = ["wall_wss_absmax", "wall_wss_q99", "wall_p_absmax", "int_p_absmax",
                   "normradius_max", "bc_BC_O1_max", "bc_BC_O2_max", "bc_BC_O3_max", "bc_BC_O4_max"]

    print("\n" + "=" * 80)
    print("OOD RANKING: Test cases with extreme values (vs train distribution)")
    print("=" * 80)

    for metric in key_metrics:
        train_vals = [s.get(metric, 0) for s in train_stats if metric in s]
        if not train_vals:
            continue
        train_max = max(train_vals)
        train_q95 = np.quantile(train_vals, 0.95) if train_vals else 0

        outliers = []
        for s in test_stats:
            val = s.get(metric, 0)
            if val > train_max:
                outliers.append((s["case"], val, val / train_max if train_max > 0 else float("inf")))

        if outliers:
            outliers.sort(key=lambda x: -x[2])
            print(f"\n  {metric}:")
            print(f"    Train max={train_max:.4f}, q95={train_q95:.4f}")
            for case, val, ratio in outliers[:5]:
                print(f"    ⚠️  {case}: {val:.4f} ({ratio:.1f}× train max)")
        else:
            print(f"\n  {metric}: All test cases within train range ✓")

    with open(OUT_DIR / "train_per_case_stats.json", "w", encoding="utf-8") as f:
        json.dump(train_stats, f, indent=2, ensure_ascii=False)
    with open(OUT_DIR / "test_per_case_stats.json", "w", encoding="utf-8") as f:
        json.dump(test_stats, f, indent=2, ensure_ascii=False)
    with open(OUT_DIR / "val_per_case_stats.json", "w", encoding="utf-8") as f:
        json.dump(val_stats, f, indent=2, ensure_ascii=False)

    split_balance = {
        "train": dict(fast_slow["train"]),
        "val": dict(fast_slow["val"]),
        "test": dict(fast_slow["test"]),
        "train_total": len(train_cases),
        "val_total": len(val_cases),
        "test_total": len(test_cases),
    }
    with open(OUT_DIR / "split_balance.json", "w", encoding="utf-8") as f:
        json.dump(split_balance, f, indent=2, ensure_ascii=False)

    make_comparison_plots(train_stats, test_stats, OUT_DIR)

    print(f"\n产出目录: {OUT_DIR}")
    print("完成。")


if __name__ == "__main__":
    main()

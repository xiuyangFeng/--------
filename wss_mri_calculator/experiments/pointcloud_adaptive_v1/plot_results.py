from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


COLORS = {"AG": "#1f77b4", "AAA": "#d62728", "ILO": "#2ca02c"}
BASELINE = "fixed_k64"
CANDIDATE = "adaptive_cv_tol1p25"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input",
        default="results/stage_c_split173_s1200.json",
    )
    parser.add_argument("--output", default="fig_split173_comparison.png")
    args = parser.parse_args()

    source = Path(args.input)
    payload = json.loads(source.read_text(encoding="utf-8"))
    rows = [row for row in payload["cases"] if row["truth_usable"]]
    baseline = np.asarray([row["methods"][BASELINE]["raw_r2"] for row in rows])
    candidate = np.asarray([row["methods"][CANDIDATE]["raw_r2"] for row in rows])
    delta = candidate - baseline

    fig, axes = plt.subplots(2, 2, figsize=(11.5, 9.2), dpi=160)

    ax = axes[0, 0]
    for cohort in ("AG", "AAA", "ILO"):
        mask = np.asarray([row["cohort"] == cohort for row in rows])
        ax.scatter(
            baseline[mask], candidate[mask], s=22, alpha=0.72,
            color=COLORS[cohort], label=f"{cohort} (n={mask.sum()})",
        )
    lo = min(float(baseline.min()), float(candidate.min())) - 0.02
    hi = max(float(baseline.max()), float(candidate.max())) + 0.02
    ax.plot([lo, hi], [lo, hi], "k--", lw=1.1, label="no change")
    ax.set(xlim=(lo, hi), ylim=(lo, hi), xlabel="fixed K=64 raw R²", ylabel="adaptive-CV raw R²")
    ax.set_title("A. Paired case-level raw R²")
    ax.grid(alpha=0.25)
    ax.legend(fontsize=8)

    ax = axes[0, 1]
    ax.hist(delta, bins=20, color="#9467bd", alpha=0.88)
    ax.axvline(delta.mean(), color="k", ls="--", lw=1.2, label=f"mean={delta.mean():.3f}")
    ax.axvline(delta.min(), color="#d62728", ls=":", lw=1.2, label=f"min={delta.min():.3f}")
    ax.set(xlabel="adaptive − fixed raw R²", ylabel="case count")
    ax.set_title("B. Paired improvement (173/173 > 0)")
    ax.grid(alpha=0.25)
    ax.legend(fontsize=8)

    ax = axes[1, 0]
    positions, values, labels, colors = [], [], [], []
    position = 1
    for cohort in ("AG", "AAA", "ILO"):
        subset = [row for row in rows if row["cohort"] == cohort]
        values.extend(
            [
                [row["methods"][BASELINE]["raw_r2"] for row in subset],
                [row["methods"][CANDIDATE]["raw_r2"] for row in subset],
            ]
        )
        positions.extend([position, position + 0.7])
        labels.extend([f"{cohort}\nfixed", f"{cohort}\nadapt"])
        colors.extend(["#bdbdbd", COLORS[cohort]])
        position += 2.2
    boxes = ax.boxplot(values, positions=positions, widths=0.55, patch_artist=True, showfliers=False)
    for box, color in zip(boxes["boxes"], colors):
        box.set_facecolor(color)
        box.set_alpha(0.82)
    ax.set_xticks(positions, labels, fontsize=8)
    ax.set_ylabel("raw R²")
    ax.set_title("C. Improvement across cohorts")
    ax.grid(axis="y", alpha=0.25)

    ax = axes[1, 1]
    role_names = ("train", "test")
    x = np.arange(len(role_names))
    width = 0.34
    base_means, candidate_means = [], []
    for role in role_names:
        subset = [row for row in rows if row["role"] == role]
        base_means.append(np.mean([row["methods"][BASELINE]["raw_r2"] for row in subset]))
        candidate_means.append(np.mean([row["methods"][CANDIDATE]["raw_r2"] for row in subset]))
    ax.bar(x - width / 2, base_means, width, color="#bdbdbd", label="fixed K=64")
    ax.bar(x + width / 2, candidate_means, width, color="#4c78a8", label="adaptive-CV")
    for xpos, value in zip(x - width / 2, base_means):
        ax.text(xpos, value + 0.008, f"{value:.3f}", ha="center", fontsize=9)
    for xpos, value in zip(x + width / 2, candidate_means):
        ax.text(xpos, value + 0.008, f"{value:.3f}", ha="center", fontsize=9)
    ax.set_xticks(x, ["train138", "blind test35"])
    ax.set_ylim(0.55, 0.91)
    ax.set_ylabel("mean raw R²")
    ax.set_title("D. Frozen train/test generalization")
    ax.grid(axis="y", alpha=0.25)
    ax.legend(fontsize=8)

    fig.suptitle(
        "Point-cloud velocity→WSS: fixed K=64 vs frozen adaptive-CV\n"
        "split173, peak step, 1200 paired wall points per case",
        fontsize=13,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output)
    plt.close(fig)
    print(output)


if __name__ == "__main__":
    main()

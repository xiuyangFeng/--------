#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""第五轮 LC 汇总：读取全部 LC run，产出长表 CSV、mean±std 曲线、配对增量、
链/seed 方差分解、PNG 图与 verdict JSON。

用法：python -m training_wss_min.summarize_round5_lc
输出：training_wss_min/runs/_round5/learning_curve/{lc_points.csv,lc_curve.png,lc_verdict.json}
"""
from __future__ import annotations
import json
import statistics as st
from pathlib import Path

import numpy as np

from . import config as C

RUNS = C.RUNS_ROOT
OUT = RUNS / "_round5" / "learning_curve"
SEEDS = [1234, 7, 2025]
CHAINS = ["chainA", "chainB", "chainC"]
SUB_SIZES = [13, 26, 40]
# LC53 端点：s1234 复用 A0E-ctrl；s7/s2025 用 lc_chainA_53
ENDPOINT = {
    1234: RUNS / "r5_a0e_b1_ctrl_s1234" / "eval" / "metrics.json",
    7: RUNS / "r5_lc_chainA_53_s7" / "eval" / "metrics.json",
    2025: RUNS / "r5_lc_chainA_53_s2025" / "eval" / "metrics.json",
}


def _read(p: Path):
    if not p.is_file():
        return None
    d = json.loads(p.read_text())["val"]
    return {
        "field": d["field"]["r2"],
        "casemean": d["aggregate"]["r2_casemean"],
        "neg": d["aggregate"].get("r2_negative_cases", float("nan")),
    }


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    rows = []
    for seed in SEEDS:
        for chain in CHAINS:
            for size in SUB_SIZES:
                m = _read(RUNS / f"r5_lc_{chain}_{size}_s{seed}" / "eval" / "metrics.json")
                if m:
                    rows.append({"chain": chain, "size": size, "seed": seed, **m})
        e = _read(ENDPOINT[seed])
        if e:
            rows.append({"chain": "shared", "size": 53, "seed": seed, **e})

    # CSV
    hdr = "chain,size,seed,field_r2,casemean,neg_cases"
    lines = [hdr] + [f"{r['chain']},{r['size']},{r['seed']},{r['field']:.6f},{r['casemean']:.6f},{r['neg']}" for r in rows]
    (OUT / "lc_points.csv").write_text("\n".join(lines) + "\n")

    # per-size stats
    def by_size(key):
        out = {}
        for size in SUB_SIZES + [53]:
            vals = [r[key] for r in rows if r["size"] == size]
            out[size] = (st.mean(vals), st.pstdev(vals), len(vals)) if vals else (float("nan"),) * 3
        return out

    fsz, csz = by_size("field"), by_size("casemean")

    # paired increments per chain×seed (40->53 uses shared endpoint per seed)
    incs = {"13_26": [], "26_40": [], "40_53": []}
    endpoint_by_seed = {r["seed"]: r["field"] for r in rows if r["size"] == 53}
    for seed in SEEDS:
        for chain in CHAINS:
            g = {r["size"]: r["field"] for r in rows if r["chain"] == chain and r["seed"] == seed}
            if 13 in g and 26 in g:
                incs["13_26"].append(g[26] - g[13])
            if 26 in g and 40 in g:
                incs["26_40"].append(g[40] - g[26])
            if 40 in g and seed in endpoint_by_seed:
                incs["40_53"].append(endpoint_by_seed[seed] - g[40])

    # variance decomposition at each subset size: chain(case-selection) vs seed
    def decomp(size):
        # seed-var: std across seeds of the chain-mean; chain-var: std across chains of the seed-mean
        chain_means = [st.mean([r["field"] for r in rows if r["size"] == size and r["chain"] == c]) for c in CHAINS
                       if any(r["size"] == size and r["chain"] == c for r in rows)]
        seed_means = [st.mean([r["field"] for r in rows if r["size"] == size and r["seed"] == s]) for s in SEEDS
                      if any(r["size"] == size and r["seed"] == s for r in rows)]
        return {"across_chain_std": st.pstdev(chain_means) if len(chain_means) > 1 else float("nan"),
                "across_seed_std": st.pstdev(seed_means) if len(seed_means) > 1 else float("nan")}

    verdict = {
        "field_by_size": {s: {"mean": fsz[s][0], "std": fsz[s][1], "n": fsz[s][2]} for s in fsz},
        "casemean_by_size": {s: {"mean": csz[s][0], "std": csz[s][1], "n": csz[s][2]} for s in csz},
        "field_paired_increment": {k: {"mean": st.mean(v), "std": st.pstdev(v), "n": len(v)} for k, v in incs.items() if v},
        "field_variance_decomp": {s: decomp(s) for s in SUB_SIZES},
        "endpoint53_field_seeds": endpoint_by_seed,
        "n_runs": len(rows),
    }
    (OUT / "lc_verdict.json").write_text(json.dumps(verdict, indent=2, ensure_ascii=False))

    # plot
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(1, 2, figsize=(11, 4.2))
        xs = SUB_SIZES + [53]
        for i, (key, szstat, ttl) in enumerate([("field", fsz, "R2_field_raw"),
                                                ("casemean", csz, "R2_casemean")]):
            for seed in SEEDS:
                for chain in CHAINS:
                    pts = [(r["size"], r[key]) for r in rows if r["chain"] == chain and r["seed"] == seed]
                    pts += [(53, endpoint_by_seed.get(seed))] if seed in endpoint_by_seed else []
                    pts = sorted([p for p in pts if p[1] is not None])
                    if pts:
                        ax[i].plot([p[0] for p in pts], [p[1] for p in pts],
                                   color="0.7", lw=0.6, alpha=0.5, zorder=1)
            mean = [szstat[s][0] for s in xs]
            std = [szstat[s][1] for s in xs]
            ax[i].errorbar(xs, mean, yerr=std, color="C0", lw=2, marker="o",
                           capsize=3, zorder=3, label="mean±std")
            ax[i].axhline(0.70, color="C3", ls="--", lw=1, label="target 0.70")
            ax[i].set_title(ttl); ax[i].set_xlabel("train cases"); ax[i].set_ylabel(key)
            ax[i].set_xticks(xs); ax[i].grid(alpha=0.3); ax[i].legend(fontsize=8)
        fig.suptitle("Round5 LC — 3 chains × 3 seeds (val, fixed dev1 8-case)")
        fig.tight_layout()
        fig.savefig(OUT / "lc_curve.png", dpi=130)
        print("wrote plot", OUT / "lc_curve.png")
    except Exception as e:
        print("plot skipped:", e)

    print(f"n_runs={len(rows)}")
    for s in xs:
        print(f"size {s:>2}: field {fsz[s][0]:.4f}±{fsz[s][1]:.4f} (n={fsz[s][2]})  "
              f"casemean {csz[s][0]:.4f}±{csz[s][1]:.4f}")
    for k, v in verdict["field_paired_increment"].items():
        print(f"inc {k}: {v['mean']:+.4f}±{v['std']:.4f} (n={v['n']})")


if __name__ == "__main__":
    main()

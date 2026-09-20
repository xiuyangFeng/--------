#!/usr/bin/env python3
"""Dump the learned bottleneck residual scales (gamma) of every BT run to a JSON sidecar.

Run with the GNN conda python (needs torch); the workbook updater runs on /usr/bin/python3 (openpyxl, no torch)
and reads this file. A gamma still sitting at its init value means the attention branch never grew, which is a
different conclusion from "global context does not help".
"""
from __future__ import annotations

import json
from pathlib import Path

import torch

from training_wss_min.tools.report_bt_matrix import ARMS, EXP_DIR, RUNS, SEEDS

out = {}
for _, stem in ARMS:
    for seed in SEEDS:
        p = RUNS / f"{stem}_s{seed}" / "ckpt_best.pt"
        if not p.is_file():
            continue
        sd = torch.load(p, map_location="cpu")["model"]
        g = {k: float(v) for k, v in sd.items() if "bottleneck" in k and "gamma" in k}
        if not g:
            continue
        ga = [v for k, v in sorted(g.items()) if "attention" in k]
        gf = [v for k, v in sorted(g.items()) if "ffn" in k]
        out[f"{stem}_s{seed}"] = f"attn {'/'.join(f'{v:.3f}' for v in ga)} · ffn {'/'.join(f'{v:.3f}' for v in gf)}"
dest = EXP_DIR / "bt_gammas.json"
dest.parent.mkdir(parents=True, exist_ok=True)
dest.write_text(json.dumps(out, indent=1, ensure_ascii=False), encoding="utf-8")
print(f"wrote {len(out)} gamma entries -> {dest}")

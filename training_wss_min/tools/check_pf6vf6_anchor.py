"""Cross-version anchor check for the PF6/VF6 v5.2d retrain preflight (2026-10-02).

There is no earlier PF6/VF6 run on v5.2+ data whose stored metrics could be reproduced, so the anchor is the deployed
v5.0 model (PF6_VF6_peak_3seed_20260920, seed 1234) repointed to the v5.2d volume root and evaluated on recover8:
`<anchor>/eval/ckpt_best/metrics.json` is written by the code that trained the v5.1 PF6/VF6 fold bases
(GNN_voltime_frozen_20260919) on the same GPU earlier in the preflight job, and this check re-evaluates the same weights
on the same inputs with the current (frozen) code, comparing every numeric field (preflight_wss_local_wave1.anchor_reevaluation:
tolerance 5e-4, > 1000 fields, no missing field). The pressure anchor is also the --anchor-run of the GPU preflight; this
tool covers the velocity anchor (and can re-check the pressure one). Preflight 16857 (2026-10-02) showed that GPU float
jitter flips single points across the hotspot quantile threshold (PF6 anchor: FU_GUO_JUN n_high_pred 41610 vs 41611; the
same code run twice differs by up to 5.5e-5 in continuous fields), so both anchors run with --pred-count-tolerance 2.

    python -m training_wss_min.tools.check_pf6vf6_anchor --anchor-run <dir> --out <json>
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import torch

from training_wss_min.tools.preflight_wss_local_wave1 import anchor_reevaluation, save_json


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--anchor-run", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--pred-count-tolerance", type=int, default=None,
                    help="prediction-dependent point counts (n_*pred*) may differ by this many points (see anchor_reevaluation)")
    args = ap.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("GPU required (the stored anchor metrics were produced on a GPU)")
    evidence = anchor_reevaluation(anchor_run=args.anchor_run, pred_count_tolerance=args.pred_count_tolerance)
    evidence.update(slurm_job_id=os.environ.get("SLURM_JOB_ID"), gpu=torch.cuda.get_device_name(0),
                    stored_by="GNN_voltime_frozen_20260919 evaluate (same job, same GPU)")
    save_json(args.out, evidence)
    print(json.dumps(evidence, ensure_ascii=False, indent=1), flush=True)
    return 0 if evidence["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())

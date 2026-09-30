"""Preflight for the OSI tail matrix (wss_osi_tail_20260924) on a local GPU (node04, outside Slurm).

1) anchor re-evaluation: stored M1_f0_s1234 best checkpoint re-evaluated with the current code into a NEW directory
   (never the run's own eval dir); every numeric leaf of metrics.json compared with the stored file (timing keys ignored);
2) paired construction of every arm config (build_paired_model), recording the evidence;
3) 2-epoch CLI smoke per arm type (M1r / K1 / K2 / K5) on a 4-train / 2-test subset of fold 0: train + eval best --save-predictions,
   and the new loss components (osi_mask_bce / osi_tail_fraction) must appear in history.jsonl.

    CUDA_VISIBLE_DEVICES=1 python -u -m training_wss_min.tools.preflight_wss_osi_tail
"""
from __future__ import annotations

import json
import os
import socket
import subprocess
import sys
import time
from pathlib import Path

from training_wss_min import config as C
from training_wss_min.paired_initialization import build_paired_model

ROOT = C.PROJECT_ROOT
NAME = "wss_osi_tail_20260924"
CONFIGS = ROOT / "training_wss_min/configs" / NAME
EXP = ROOT / "training_wss_min/experiments" / NAME
ANCHOR_RUN = ROOT / "training_wss_min/runs/wss_cycle_m1_20260921/M1_f0_s1234"
TOL = 5e-4
SKIP_KEYS = ("seconds", "time", "elapsed", "timestamp", "started", "ended", "host", "device", "path", "dir")


def leaves(obj, prefix=""):
    if isinstance(obj, dict):
        for k, v in obj.items():
            yield from leaves(v, f"{prefix}.{k}" if prefix else str(k))
    elif isinstance(obj, list):
        for i, v in enumerate(obj):
            yield from leaves(v, f"{prefix}[{i}]")
    elif isinstance(obj, (int, float)) and not isinstance(obj, bool):
        yield prefix, float(obj)


def save_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")
    tmp.replace(path)


def run(cmd, log: Path):
    log.parent.mkdir(parents=True, exist_ok=True)
    with log.open("x") as fh:
        rc = subprocess.run(cmd, cwd=str(ROOT), stdout=fh, stderr=subprocess.STDOUT).returncode
    if rc != 0:
        raise RuntimeError(f"command failed ({rc}): {' '.join(map(str, cmd))} — see {log}")


def main():
    tag = time.strftime("%Y%m%d_%H%M%S")
    out = EXP / "preflight" / tag
    evidence = {"host": socket.gethostname(), "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
                "executed_outside_slurm": os.environ.get("SLURM_JOB_ID") is None, "package_root": str(ROOT),
                "started_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "passed": False}
    # 1) anchor
    anchor_out = out / "anchor_eval_M1_f0_best"
    run([sys.executable, "-u", "-m", "training_wss_min.evaluate", "--run-dir", str(ANCHOR_RUN), "--partitions", "test", "--allow-test",
         "--checkpoint", "best", "--no-plots", "--output-dir", str(anchor_out)], out / "logs/anchor_eval.log")
    stored = dict(leaves(json.loads((ANCHOR_RUN / "eval/ckpt_best/metrics.json").read_text())))
    fresh = dict(leaves(json.loads((anchor_out / "metrics.json").read_text())))
    common = [k for k in stored.keys() & fresh.keys() if not any(s in k.lower() for s in SKIP_KEYS)]
    diffs = sorted(((abs(stored[k] - fresh[k]) if stored[k] == stored[k] or fresh[k] == fresh[k] else 0.0, k) for k in common), reverse=True)
    nan_mismatch = [k for k in common if (stored[k] != stored[k]) != (fresh[k] != fresh[k])]
    evidence["anchor_reevaluation"] = {"anchor_run": str(ANCHOR_RUN), "output_dir": str(anchor_out), "compared_fields": len(common),
                                       "only_in_stored": len(stored.keys() - fresh.keys()), "only_in_fresh": len(fresh.keys() - stored.keys()),
                                       "max_abs_diff": diffs[0][0], "max_abs_diff_field": diffs[0][1], "top5": diffs[:5],
                                       "nan_mismatch": nan_mismatch, "tolerance": TOL,
                                       "passed": diffs[0][0] <= TOL and not nan_mismatch}
    print("anchor", {k: v for k, v in evidence["anchor_reevaluation"].items() if k != "top5"}, flush=True)
    if not evidence["anchor_reevaluation"]["passed"]:
        save_json(out / "preflight.json", evidence)
        raise SystemExit("anchor re-evaluation differs")
    # 2) paired construction
    evidence["paired_construction"] = {}
    for path in sorted(CONFIGS.glob("*_s1234.json")):
        cfg = C.ExpConfig.from_json(path)
        model, ev = build_paired_model(cfg)
        evidence["paired_construction"][path.name] = {k: ev.get(k) for k in ("mode", "shared_tensors_exact", "reference_config") } | \
            {"n_params": int(sum(p.numel() for p in model.parameters())), "input_dim": int(C.input_dim(cfg))}
    print("paired construction ok", len(evidence["paired_construction"]), flush=True)
    # 3) smoke
    split = json.loads(Path(json.loads((CONFIGS / "M1r_f0_s1234.json").read_text())["data"]["split_path"]).read_text())
    small = dict(split)
    small["train_cases"] = list(split["train_cases"])[:4]
    small["test_cases"] = list(split["test_cases"])[:2]
    small["counts"] = {"train": 4, "val": 0, "test": 2}
    small["expected_counts"] = {"train": 4, "val": 0, "test": 2}
    for key in list(small):
        if key.endswith("_cases") and key not in ("train_cases", "test_cases"):
            small[key] = []
    save_json(out / "smoke_split.json", small)
    evidence["smoke"] = {}
    for arm in ("M1r", "K1", "K2", "K5"):
        raw = json.loads((CONFIGS / f"{arm}_f0_s1234.json").read_text())
        raw["name"] = f"{NAME}_smoke/{tag}_{arm}"
        raw["data"]["split_path"] = str(out / "smoke_split.json")
        raw["train"].update(epochs=2, min_epoch=2, eval_every=2, warmup_epochs=1)
        cfg_path = out / f"{arm}_smoke_config.json"
        save_json(cfg_path, raw)
        run([sys.executable, "-u", "-m", "training_wss_min.train", "--config", str(cfg_path)], out / f"logs/{arm}_train.log")
        run_dir = ROOT / "training_wss_min/runs" / raw["name"]
        run([sys.executable, "-u", "-m", "training_wss_min.evaluate", "--run-dir", str(run_dir), "--partitions", "test", "--allow-test",
             "--checkpoint", "best", "--no-plots", "--save-predictions"], out / f"logs/{arm}_eval.log")
        hist = [json.loads(l) for l in (run_dir / "history.jsonl").read_text().splitlines() if l.strip()]
        comps = sorted({k for h in hist for k in (h.get("loss_components") or {})})
        need = {"K1": "osi_mask_bce", "K2": "osi_tail_fraction"}.get(arm)
        if need and need not in comps:
            raise RuntimeError(f"{arm}: loss component {need} missing from history ({comps})")
        preds = list((run_dir / "eval/ckpt_best/predictions/test").rglob("predictions.npz"))
        if not preds:
            raise RuntimeError(f"{arm}: no saved predictions")
        evidence["smoke"][arm] = {"run": str(run_dir), "loss_components": comps, "n_prediction_files": len(preds),
                                  "final_loss": hist[-1].get("loss") if hist else None}
        print("smoke", arm, evidence["smoke"][arm], flush=True)
    evidence["passed"] = True
    evidence["finished_at"] = time.strftime("%Y-%m-%dT%H:%M:%S%z")
    save_json(out / "preflight.json", evidence)
    save_json(EXP / "runtime_preflight.json", evidence | {"preflight_dir": str(out)})
    print("PREFLIGHT PASSED", out, flush=True)


if __name__ == "__main__":
    main()

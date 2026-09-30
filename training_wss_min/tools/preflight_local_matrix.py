"""Generic preflight for a local / Slurm matrix run with tools/run_local_train_queue.py (2026-09-25).

1) anchor re-evaluation(s): each stored run's best checkpoint re-evaluated with the current code into a NEW directory under
   experiments/<name>/preflight/<tag>/ (never the run's own eval dir); every numeric leaf of metrics.json compared (timing keys ignored);
2) paired construction of every arm config in configs/<name>/ (build_paired_model when init_reference_config is set);
3) 2-epoch CLI smoke for the chosen arm configs on a 4-train / 2-test subset of their own split: train + eval best --save-predictions;
   optional required loss components per smoke arm must appear in history.jsonl.

    python -u -m training_wss_min.tools.preflight_local_matrix --name <exp> --anchor-run <run> [--anchor-run <run>] \
        --smoke <arm.json> [--smoke ...] [--require <arm.json>=<component>]
"""
from __future__ import annotations

import argparse
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


def anchor(run_dir: Path, out: Path, tol: float) -> dict:
    target = out / f"anchor_{run_dir.parent.name}_{run_dir.name}_best"
    run([sys.executable, "-u", "-m", "training_wss_min.evaluate", "--run-dir", str(run_dir), "--partitions", "test", "--allow-test",
         "--checkpoint", "best", "--no-plots", "--output-dir", str(target)], out / f"logs/anchor_{run_dir.name}.log")
    stored = dict(leaves(json.loads((run_dir / "eval/ckpt_best/metrics.json").read_text())))
    fresh = dict(leaves(json.loads((target / "metrics.json").read_text())))
    common = [k for k in stored.keys() & fresh.keys() if not any(s in k.lower() for s in SKIP_KEYS)]
    diffs = sorted(((abs(stored[k] - fresh[k]) if (stored[k] == stored[k] and fresh[k] == fresh[k]) else 0.0, k) for k in common), reverse=True)
    nan_mismatch = [k for k in common if (stored[k] != stored[k]) != (fresh[k] != fresh[k])]
    return {"anchor_run": str(run_dir), "output_dir": str(target), "compared_fields": len(common),
            "only_in_stored": len(stored.keys() - fresh.keys()), "only_in_fresh": len(fresh.keys() - stored.keys()),
            "max_abs_diff": diffs[0][0], "max_abs_diff_field": diffs[0][1], "top5": diffs[:5], "nan_mismatch": nan_mismatch,
            "tolerance": tol, "passed": diffs[0][0] <= tol and not nan_mismatch}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True)
    ap.add_argument("--anchor-run", action="append", default=[])
    ap.add_argument("--smoke", action="append", default=[])
    ap.add_argument("--require", action="append", default=[], help="<arm config>=<loss component>")
    ap.add_argument("--tolerance", type=float, default=5e-4)
    args = ap.parse_args()
    configs = ROOT / "training_wss_min/configs" / args.name
    exp = ROOT / "training_wss_min/experiments" / args.name
    tag = time.strftime("%Y%m%d_%H%M%S")
    out = exp / "preflight" / tag
    evidence = {"host": socket.gethostname(), "slurm_job_id": os.environ.get("SLURM_JOB_ID"), "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
                "package_root": str(ROOT), "started_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "passed": False, "anchors": []}
    for a in args.anchor_run:
        res = anchor(Path(a), out, args.tolerance)
        evidence["anchors"].append(res)
        print("anchor", {k: v for k, v in res.items() if k != "top5"}, flush=True)
        if not res["passed"]:
            save_json(out / "preflight.json", evidence)
            raise SystemExit(f"anchor re-evaluation differs: {a}")
    evidence["paired_construction"] = {}
    for path in sorted(configs.glob("*_s*.json")):
        cfg = C.ExpConfig.from_json(path)
        if cfg.train.init_reference_config:
            model, ev = build_paired_model(cfg)
            evidence["paired_construction"][path.name] = {k: ev.get(k) for k in ("mode", "shared_tensors_exact", "reference_config")} | \
                {"n_params": int(sum(p.numel() for p in model.parameters())), "input_dim": int(C.input_dim(cfg))}
    print("paired construction ok", len(evidence["paired_construction"]), flush=True)
    required = dict(r.split("=", 1) for r in args.require)
    evidence["smoke"] = {}
    for arm in args.smoke:
        raw = json.loads((configs / arm).read_text())
        split = json.loads(Path(raw["data"]["split_path"]).read_text())
        small = dict(split)
        small["train_cases"] = list(split["train_cases"])[:4]
        small["test_cases"] = list(split["test_cases"])[:2]
        small["counts"] = {"train": 4, "val": 0, "test": 2}
        small["expected_counts"] = {"train": 4, "val": 0, "test": 2}
        for key in list(small):
            if key.endswith("_cases") and key not in ("train_cases", "test_cases"):
                small[key] = []
        stem = Path(arm).stem
        save_json(out / f"{stem}_smoke_split.json", small)
        raw["name"] = f"{args.name}_smoke/{tag}_{stem}"
        raw["data"]["split_path"] = str(out / f"{stem}_smoke_split.json")
        raw["train"].update(epochs=2, min_epoch=2, eval_every=2, warmup_epochs=1)
        cfg_path = out / f"{stem}_smoke_config.json"
        save_json(cfg_path, raw)
        run([sys.executable, "-u", "-m", "training_wss_min.train", "--config", str(cfg_path)], out / f"logs/{stem}_train.log")
        run_dir = ROOT / "training_wss_min/runs" / raw["name"]
        run([sys.executable, "-u", "-m", "training_wss_min.evaluate", "--run-dir", str(run_dir), "--partitions", "test", "--allow-test",
             "--checkpoint", "best", "--no-plots", "--save-predictions"], out / f"logs/{stem}_eval.log")
        hist = [json.loads(l) for l in (run_dir / "history.jsonl").read_text().splitlines() if l.strip()]
        comps = sorted({k for h in hist for k in (h.get("loss_components") or {})})
        need = required.get(arm)
        if need and need not in comps:
            raise RuntimeError(f"{arm}: loss component {need} missing from history ({comps})")
        preds = list((run_dir / "eval/ckpt_best/predictions/test").rglob("predictions.npz"))
        if not preds:
            raise RuntimeError(f"{arm}: no saved predictions")
        evidence["smoke"][arm] = {"run": str(run_dir), "loss_components": comps, "n_prediction_files": len(preds)}
        print("smoke", arm, evidence["smoke"][arm], flush=True)
    evidence["passed"] = True
    evidence["finished_at"] = time.strftime("%Y-%m-%dT%H:%M:%S%z")
    save_json(out / "preflight.json", evidence)
    host = "slurm" + str(evidence["slurm_job_id"]) if evidence["slurm_job_id"] else socket.gethostname()
    save_json(exp / f"runtime_preflight_{host}.json", evidence | {"preflight_dir": str(out)})
    print("PREFLIGHT PASSED", out, flush=True)


if __name__ == "__main__":
    main()

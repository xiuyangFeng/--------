"""Stage-1 sidecar for the T3 hotspot cascade: ln(WSS_pred) of the deployable X5 recipe at every wall node.

* train138: out-of-fold prediction of the cv3 fold model whose held-out fold contains the case (X5_f<k>_s<seed>),
  so stage 2 never sees a stage-1 prediction that was fitted on the same case;
* test34: prediction of one fold model per case (round-robin k = case index mod 3; every fold model is trained on
  91-93 cases), so the stage-1 quality stage 2 sees at test time matches the train-time out-of-fold quality.
  The fold models are evaluated on the frozen test34 split read-only into ``<run>/eval_test34/ckpt_best``.

Writes ``data_wss_v5/views/wss_min_cascade_v1/<case>/features.npz`` (``wall_node_id_cas`` + ``wall_log_wss_base``),
the residual-target statistics ``wss_global_stats_train138_offset_logwssbase.json`` (ln(WSS+eps) − log_wss_base over
train138 out-of-fold points) and a manifest.  ``log_wss_base = ln(max(pred_pa, 0) + eps)`` is stats-independent.

    python -m training_wss_min.tools.make_cascade_sidecar [--seed 1234]
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

from training_wss_min import config as C
from training_wss_min import dataset as D

ROOT = C.PROJECT_ROOT
RUNS = ROOT / "training_wss_min/runs"
WAVE5 = "wss_local_wave5_20260913"
VIEW = ROOT / "data_wss_v5/views/wss_min_view_v1"
CV = VIEW / "cv3_20260913"
FROZEN_SPLIT = VIEW / "split_V5_train138_test34.json"
FROZEN_STATS = VIEW / "wss_global_stats_train138.json"
OUT_ROOT = ROOT / "data_wss_v5/views/wss_min_cascade_v1"
KEY = "log_wss_base"


def fold_runs(seed: int, wave: str = WAVE5, pattern: str = "X5_f{k}_s{seed}") -> dict[int, Path]:
    runs = {k: RUNS / wave / pattern.format(k=k, seed=seed) for k in range(3)}
    for k, run in runs.items():
        if not (run / "ckpt_best.pt").is_file() or not (run / "eval/ckpt_best/predictions/test/manifest.json").is_file():
            raise FileNotFoundError(f"fold {k} model or its held-out predictions missing: {run}")
    return runs


def ensure_test34_predictions(run: Path, frozen_split: Path = FROZEN_SPLIT) -> Path:
    out = run / "eval_test34" / "ckpt_best"
    manifest = out / "predictions" / "manifest.json"
    if manifest.is_file():
        return out
    command = [sys.executable, "-u", "-m", "training_wss_min.evaluate", "--run-dir", str(run), "--checkpoint", "best",
               "--partitions", "test", "--allow-test", "--no-plots", "--save-predictions",
               "--split-path", str(frozen_split), "--output-dir", str(out)]
    print("evaluating fold model on frozen test34:", run.name, flush=True)
    subprocess.run(command, cwd=ROOT, check=True)
    if not manifest.is_file():
        raise RuntimeError(f"test34 evaluation did not write predictions: {out}")
    return out


def log_base_from_prediction(path: Path, n_rows: int, eps: float) -> np.ndarray:
    with np.load(path) as z:
        values = np.full(n_rows, np.nan)
        values[z["row_index"]] = np.log(np.clip(z["pred_pa"].astype(np.float64), 0.0, None) + eps)
    if not np.isfinite(values).all():
        raise ValueError(f"incomplete stage-1 prediction: {path}")
    return values


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--out", default=str(OUT_ROOT))
    # 2026-09-16: v5.1 刷新用（默认 = 旧行为）
    ap.add_argument("--view-root", default=str(VIEW))
    ap.add_argument("--cv-name", default=CV.name)
    ap.add_argument("--split-name", default=FROZEN_SPLIT.name)
    ap.add_argument("--stats-name", default=FROZEN_STATS.name)
    ap.add_argument("--wave", default=WAVE5, help="runs/<wave> holding the fold models")
    ap.add_argument("--fold-run-pattern", default="X5_f{k}_s{seed}")
    args = ap.parse_args(argv)
    started = time.time()
    out_root = Path(args.out)
    view = Path(args.view_root); cv = view / args.cv_name; frozen_split = view / args.split_name; frozen_stats = view / args.stats_name
    runs = fold_runs(args.seed, args.wave, args.fold_run_pattern)
    frozen = json.loads(frozen_split.read_text())
    stats = D.load_wss_stats(frozen_stats)
    eps = float(stats["eps"])
    fold_of = json.load(open(cv / "cv3_summary.json"))["fold_of"]
    test34_dirs = {k: ensure_test34_predictions(run, frozen_split) for k, run in runs.items()}
    records = []
    resid_sum = resid_sq = 0.0
    resid_n = 0
    for part, cases in (("train", frozen["train_cases"]), ("test", frozen["test_cases"])):
        for i, cid in enumerate(sorted(cases)):
            if part == "train":
                k = int(fold_of[cid]); pred_dir = runs[k] / "eval/ckpt_best"; source = "out_of_fold"
            else:
                k = i % 3; pred_dir = test34_dirs[k]; source = "fold_model_round_robin"
            pred_path = pred_dir / "predictions/test" / cid / "predictions.npz"
            with np.load(view / cid / "bundle.npz", allow_pickle=True) as b:
                node_id = b["wall_node_id_cas"].astype(np.int64)
                steps = b["steps"].tolist(); wss = b["wall_wss"][steps.index(int(b["peak_step"]))].astype(np.float64)
            base = log_base_from_prediction(pred_path, len(node_id), eps)
            out_dir = out_root / cid
            out_dir.mkdir(parents=True, exist_ok=True)
            tmp = out_dir / "features.tmp.npz"
            np.savez(tmp, wall_node_id_cas=node_id, **{f"wall_{KEY}": base.astype(np.float32)})
            tmp.replace(out_dir / "features.npz")
            resid = np.log(np.clip(wss, 0.0, None) + eps) - base
            r2 = float(1 - ((resid) ** 2).sum() / ((np.log(wss + eps) - np.log(wss + eps).mean()) ** 2).sum())
            if part == "train":
                resid_sum += resid.sum(); resid_sq += (resid ** 2).sum(); resid_n += len(resid)
            records.append({"case": cid, "partition": part, "fold_model": k, "source": source, "n": int(len(node_id)),
                            "stage1_ln_r2": r2, "prediction": str(pred_path)})
    mean = resid_sum / resid_n
    std = float(np.sqrt(max(resid_sq / resid_n - mean ** 2, 1e-12)))
    offset_stats = {k: v for k, v in stats.items() if k != "train_units"}
    offset_stats["offset"] = {"feature": KEY, "mean": float(mean), "std": std, "n_points": int(resid_n),
                              "note": ("residual target ln(WSS+eps) - log_wss_base over train138 wall points, log_wss_base = "
                                       "out-of-fold cv3 X5 prediction; stats['log'] stays the standard log_z used for reporting")}
    offset_stats["source_stats"] = str(frozen_stats)
    stats_path = out_root / frozen_stats.name.replace(".json", "_offset_logwssbase.json")
    stats_path.write_text(json.dumps(offset_stats, indent=1))
    train_r2 = [r["stage1_ln_r2"] for r in records if r["partition"] == "train"]
    test_r2 = [r["stage1_ln_r2"] for r in records if r["partition"] == "test"]
    manifest = {"pack": "wss_min_cascade_v1", "version": "v1.0", "key": KEY, "seed": args.seed,
                "fold_runs": {k: str(v) for k, v in runs.items()}, "test34_eval_dirs": {k: str(v) for k, v in test34_dirs.items()},
                "cv3_summary": str(cv / "cv3_summary.json"), "frozen_split": str(frozen_split), "eps": eps,
                "offset_stats": str(stats_path), "offset_mean": mean, "offset_std": std,
                "stage1_ln_r2_train_oof_mean": float(np.mean(train_r2)), "stage1_ln_r2_test_mean": float(np.mean(test_r2)),
                "deployment_inputs_only": True, "cases": records, "seconds": round(time.time() - started, 1)}
    (out_root / "cascade_manifest.json").write_text(json.dumps(manifest, indent=1))
    print(f"wrote {len(records)} cases; offset mean {mean:.4f} std {std:.4f}; stage-1 ln R² train(oof) {np.mean(train_r2):.3f} "
          f"test {np.mean(test_r2):.3f}; {time.time() - started:.0f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

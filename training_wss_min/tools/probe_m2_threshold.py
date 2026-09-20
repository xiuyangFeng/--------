"""Read-only repeated inference around the failed discrete log_z threshold."""
import json
import os
import time

import numpy as np
import torch

from training_wss_min import config as C, dataset as D, evaluate as E
from training_wss_min.tools.m2_optimization_common import EXP, RUNS, save_json, sha, stamp


def main():
    assert os.environ.get("SLURM_JOB_ID") and torch.cuda.is_available()
    assert torch.cuda.device_count() == 1
    torch.set_num_threads(2)
    while torch.cuda.mem_get_info()[0] / 2**20 < 7000:
        time.sleep(15)
    run = RUNS / "m2_optimization_20260909/MO-P4_s1234"
    output = EXP / "threshold_probe" / f"job_{os.environ['SLURM_JOB_ID']}"
    output.mkdir(parents=True, exist_ok=True)
    cfg, feature_stats, model, checkpoint = E.load_model_from_run(run, "cuda", "best")
    stats = E.load_wss_stats_for_run(run)
    case = D.load_case("AAA/unruputer", "ZHANG_YONG_ZHI", stats,
        target=cfg.data.target, target_normalization=cfg.data.target_normalization,
        data_root=cfg.data.data_root, required_frame_version=cfg.data.required_frame_version,
        timesteps=cfg.data.timesteps, extra_point_features=C.v6_point_features(cfg),
        point_features_root=cfg.data.point_features_root)
    metric_path = run / "eval/ckpt_best/metrics.json"
    saved = json.loads(metric_path.read_text())["test"]["normalized"]["per_case"][case["unit_id"]]
    record = {"started_at": stamp(), "script_sha256": sha(__file__), "job_id": os.environ["SLURM_JOB_ID"],
        "source_metrics_sha256": sha(metric_path), "reference_n": saved["distribution"]["n"],
        "reference_fraction": saved["distribution"]["pred_above_one_fraction"], "repeats": []}
    first = None
    for index in range(8):
        result = E._evaluate_partition_frame(model, [case], cfg, feature_stats, stats, "cuda", return_predictions=True)
        raw = np.asarray(result["_pred_norm_by_case"][0])
        assert np.isfinite(raw).all() and np.isfinite(case["y_norm"]).all()
        path = output / f"repeat_{index}.npz"
        np.savez_compressed(path, prediction_norm=raw, truth_norm=case["y_norm"])
        if first is None:
            first = raw.copy()
        closest = np.argsort(np.abs(raw - 1.))[:12]
        crossed = np.flatnonzero((raw > 1) != (first > 1))
        item = {"index": index, "dtype": str(raw.dtype), "n": len(raw), "above_one_count": int((raw > 1).sum()),
            "above_one_fraction": float((raw > 1).mean()), "max_abs_vs_first": float(np.max(np.abs(raw - first))),
            "threshold_crossings_vs_first": [{"index": int(i), "first": float(first[i]), "current": float(raw[i])} for i in crossed],
            "nearest_to_one": [{"index": int(i), "value": float(raw[i]), "distance": float(abs(raw[i] - 1)),
                                "float32_ulp": float(np.spacing(np.float32(raw[i])))} for i in closest],
            "array_path": str(path), "array_sha256": sha(path)}
        record["repeats"].append(item)
        print(json.dumps(item, ensure_ascii=False), flush=True)
    record.update(completed=True, completed_at=stamp(), counts=sorted({r["above_one_count"] for r in record["repeats"]}))
    save_json(output / "probe.json", record)
    print(f"Saved {output / 'probe.json'}", flush=True)


if __name__ == "__main__":
    main()

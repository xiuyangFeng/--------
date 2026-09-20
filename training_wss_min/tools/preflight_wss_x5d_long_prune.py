"""GPU preflight for the single-seed longitudinal pruning ladder (control arm configurable)."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import socket
import subprocess
import sys

from training_wss_min.tools import preflight_wss_local_wave1 as base


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config-dir", type=Path, required=True)
    parser.add_argument("--experiment-dir", type=Path, required=True)
    parser.add_argument("--control-arm", default="L8_s1234")
    args = parser.parse_args()
    exp = args.experiment_dir
    prepared = json.loads((exp / "prepared_audit.json").read_text())
    if prepared.get("passed") is not True:
        raise RuntimeError("The complete input audit must pass before GPU preflight")
    gpu = subprocess.check_output([
        "nvidia-smi", "--query-gpu=index,uuid,name,memory.used,memory.total", "--format=csv"], text=True)
    base.save_json(exp / "preflight_runtime.json", {
        "hostname": socket.gethostname(), "job_id": os.environ.get("SLURM_JOB_ID"),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"), "gpu_inventory": gpu,
        "execution_root": str(base.ROOT)})
    base.main(["--config-dir", str(args.config_dir), "--experiment-dir", str(exp),
               "--control-arm", args.control_arm, "--smoke-arms", ""])
    gate = exp / "runtime_preflight.json"
    evidence = json.loads(gate.read_text())
    gate.rename(exp / "model_preflight.json")
    evidence["passed"] = False
    raw = json.loads((args.config_dir / f"{args.control_arm}.json").read_text())
    job = os.environ["SLURM_JOB_ID"]
    raw["name"] = f"{exp.name}_smoke/job_{job}_fulltrain136"
    raw["train"].update(epochs=2, min_epoch=2, eval_every=2)
    smoke = exp / "smoke" / job
    smoke.mkdir(parents=True, exist_ok=True)
    config = smoke / "fulltrain136_config.json"
    base.save_json(config, raw)
    command = [sys.executable, "-u", "-m", "training_wss_min.train", "--config", str(config)]
    print("Full train136 CLI smoke:", " ".join(command), flush=True)
    subprocess.run(command, cwd=base.ROOT, check=True)
    run = base.RUNS / raw["name"]
    history = [json.loads(line) for line in (run / "history.jsonl").read_text().splitlines() if line]
    if len(history) != 2 or not (run / "ckpt_best.pt").is_file():
        raise RuntimeError("CLI smoke did not produce two epochs and its best checkpoint")
    evidence["cli_smoke"] = {args.control_arm: {"run": str(run), "epochs": len(history),
                                                "train_cases": 136, "split_unchanged": True,
                                                "formal_result": False}}
    if base.fingerprints(args.config_dir) != evidence["fingerprints"]:
        raise RuntimeError("Sources/config/protocol changed during CLI smoke")
    evidence["passed"] = True
    evidence["ended_at"] = base.time.strftime("%Y-%m-%dT%H:%M:%S%z")
    base.save_json(gate, evidence)
    print("PRUNE LADDER PREFLIGHT PASSED", flush=True)


if __name__ == "__main__":
    main()

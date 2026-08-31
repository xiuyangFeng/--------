"""Detach-start V4 array indices 46 and 47 on node04. Fresh start only."""
from __future__ import annotations

import os
import subprocess
from pathlib import Path

PROJECT = Path("/public/newhome/cy/Digital_twin/GNN")
PYTHON = "/public/newhome/cy/.conda/envs/GNN/bin/python"
JOBS = (
    {
        "index": 46,
        "gpu": "0",
        "config": PROJECT
        / "wss_pinn/configs/volume_uvwp_bc_rcr_v4/transient_autograd/v4_tr_pnpp_bc_pde_f_s3456.json",
        "log": PROJECT / "outputs/wss_pinn/slurm/v4_train_node04_46.out",
    },
    {
        "index": 47,
        "gpu": "1",
        "config": PROJECT
        / "wss_pinn/configs/volume_uvwp_bc_rcr_v4/transient_autograd/v4_tr_pnpp_bc_pde_ema_s3456.json",
        "log": PROJECT / "outputs/wss_pinn/slurm/v4_train_node04_47.out",
    },
)


def main() -> None:
    os.chdir(PROJECT)
    os.environ["HOME"] = "/public/newhome/cy"
    for job in JOBS:
        log_path = Path(job["log"])
        log_path.parent.mkdir(parents=True, exist_ok=True)
        if log_path.exists() and log_path.stat().st_size > 0:
            raise SystemExit(f"refusing to overwrite existing log: {log_path}")
        log = open(log_path, "ab", buffering=0)
        env = os.environ.copy()
        env.update(
            {
                "HOME": "/public/newhome/cy",
                "CUDA_VISIBLE_DEVICES": str(job["gpu"]),
                "CUBLAS_WORKSPACE_CONFIG": ":4096:8",
                "PYTHONUNBUFFERED": "1",
            }
        )
        argv = [PYTHON, "-m", "wss_pinn.v4.train", "--config", str(job["config"])]
        process = subprocess.Popen(
            argv,
            cwd=str(PROJECT),
            env=env,
            stdin=subprocess.DEVNULL,
            stdout=log,
            stderr=subprocess.STDOUT,
            start_new_session=True,
            close_fds=True,
        )
        print(
            "index={} gpu={} pid={}".format(job["index"], job["gpu"], process.pid),
            flush=True,
        )


if __name__ == "__main__":
    main()

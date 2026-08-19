"""Generate the frozen 16-arm x 3-seed V4 JSON matrix."""

from __future__ import annotations

import json
from pathlib import Path

from wss_pinn.utils import ROOT, atomic_write_json, guard_write_path, utc_now
from wss_pinn.v4.config import CONFIG_ROOT, ROUTE, SEEDS


TEMPORAL = {
    "steady_peak": {"short": "SP", "momentum": "quasi_steady", "time": False},
    "transient_81": {"short": "TR", "momentum": "transient_autograd", "time": True},
}
BACKBONES = {"pointnet": "PN", "pointnetpp": "PNPP"}
MODES = {
    "data": {
        "short": "DATA",
        "bc": False,
        "pde": False,
        "lambda_bc": 0.0,
        "lambda_pde": 0.0,
        "balance": "disabled",
    },
    "data_bc": {
        "short": "BC",
        "bc": True,
        "pde": False,
        "lambda_bc": 1.0,
        "lambda_pde": 0.0,
        "balance": "disabled",
    },
    "bc_pde_fixed": {
        "short": "BC-PDE-F",
        "bc": True,
        "pde": True,
        "lambda_bc": 1.0,
        "lambda_pde": 1.0,
        "balance": "fixed",
    },
    "bc_pde_ema": {
        "short": "BC-PDE-EMA",
        "bc": True,
        "pde": True,
        "lambda_bc": 1.0,
        "lambda_pde": 1.0,
        "balance": "ema_loss_ratio",
    },
}


def generate() -> dict:
    config_names = []
    groups = []
    seed_groups = {str(seed): [] for seed in SEEDS}
    temporal_groups = {key: [] for key in TEMPORAL}
    for seed in SEEDS:
        for temporal_mode, temporal in TEMPORAL.items():
            directory = "steady_peak" if temporal_mode == "steady_peak" else "transient_autograd"
            for backbone, backbone_short in BACKBONES.items():
                paired = []
                for training_mode, mode in MODES.items():
                    experiment_id = (
                        f"V4-{temporal['short']}-{backbone_short}-{mode['short']}-s{seed}"
                    )
                    filename = experiment_id.lower().replace("-", "_") + ".json"
                    relative = f"{directory}/{filename}"
                    payload = {
                        "route": ROUTE,
                        "experiment": {
                            "id": experiment_id,
                            "temporal_mode": temporal_mode,
                            "momentum_mode": temporal["momentum"],
                            "time_input": temporal["time"],
                            "backbone": backbone,
                            "training_mode": training_mode,
                            "seed": seed,
                        },
                        "paths": {
                            "run_dir": (
                                f"outputs/wss_pinn/volume_uvwp_bc_rcr_v4/"
                                f"{directory}/{experiment_id}"
                            )
                        },
                        "sampling": (
                            {"support_points": 2000, "query_points": 2000}
                            if temporal_mode == "transient_81"
                            else {}
                        ),
                        "physics": {
                            "bc_enabled": mode["bc"],
                            "pde_enabled": mode["pde"],
                            "lambda_bc": mode["lambda_bc"],
                            "lambda_pde": mode["lambda_pde"],
                        },
                        "loss_balancing": {"mode": mode["balance"]},
                        "train": {"seed": seed},
                    }
                    atomic_write_json(CONFIG_ROOT / relative, payload)
                    config_names.append(relative)
                    paired.append(relative)
                    seed_groups[str(seed)].append(relative)
                    temporal_groups[temporal_mode].append(relative)
                groups.append(
                    {
                        "temporal_mode": temporal_mode,
                        "backbone": backbone,
                        "seed": seed,
                        "configs": paired,
                    }
                )
    matrix = {
        "schema_version": 1,
        "route": ROUTE,
        "created_at": utc_now(),
        "design": "2 temporal x 2 backbone x 4 training modes x 3 seeds",
        "counts": {"arms": 16, "seeds": 3, "runs": len(config_names)},
        "seeds": list(SEEDS),
        "config_order": config_names,
        "paired_groups": groups,
        "seed_groups": seed_groups,
        "temporal_groups": temporal_groups,
        "submission": {
            "max_concurrent": 4,
            "data_gate": "outputs/wss_pinn/volume_uvwp_bc_rcr_v4/audits/stage0/gate_report.json",
            "preflight_report": "outputs/wss_pinn/volume_uvwp_bc_rcr_v4/audits/preflight/report.json",
            "output": "outputs/wss_pinn/volume_uvwp_bc_rcr_v4/submission.json",
        },
    }
    atomic_write_json(CONFIG_ROOT / "matrix.json", matrix)
    config_list = guard_write_path(CONFIG_ROOT / "matrix_configs.txt")
    config_list.write_text(
        "\n".join(str((CONFIG_ROOT / name).resolve()) for name in config_names) + "\n",
        encoding="utf-8",
    )
    return matrix


def main() -> None:
    print(json.dumps(generate(), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

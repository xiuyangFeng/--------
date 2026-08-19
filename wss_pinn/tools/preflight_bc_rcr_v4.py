"""Static and optional GPU preflight for the complete V4 matrix."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from wss_pinn.utils import atomic_write_json, sha256_file, utc_now
from wss_pinn.volume_utils import tensor_state_sha256
from wss_pinn.v4.config import CONFIG_ROOT, MANIFEST_PATH, STATS_PATH, load_config
from wss_pinn.v4.models import build_model
from wss_pinn.v4.train import run, seed_everything


DEFAULT_OUTPUT = (
    Path("outputs/wss_pinn/volume_uvwp_bc_rcr_v4/audits/preflight/report.json")
)


def preflight(*, gpu_smoke: bool, output: Path) -> dict:
    matrix_path = CONFIG_ROOT / "matrix.json"
    matrix = json.loads(matrix_path.read_text(encoding="utf-8"))
    names = list(matrix["config_order"])
    if len(names) != 48 or len(set(names)) != 48:
        raise ValueError("V4 matrix must contain 48 unique runs")
    configs = [load_config(CONFIG_ROOT / name, require_assets=True) for name in names]
    if any(config["data"]["validation_roles"] for config in configs):
        raise ValueError("validation role leaked into V4")
    if any(config["data"]["train_roles"] != ["train"] for config in configs):
        raise ValueError("formal configs must train on role=train only")

    group_hashes = []
    for group in matrix["paired_groups"]:
        hashes = []
        for name in group["configs"]:
            config = load_config(CONFIG_ROOT / name, require_assets=True)
            seed_everything(int(config["train"]["seed"]))
            hashes.append(tensor_state_sha256(build_model(config).state_dict()))
        if len(set(hashes)) != 1:
            raise ValueError(f"paired initialization mismatch: {group}")
        group_hashes.append({**group, "initialization_state_sha256": hashes[0]})

    representative = {}
    for backbone in ("pointnet", "pointnetpp"):
        config = next(value for value in configs if value.backbone == backbone)
        model = build_model(config)
        representative[backbone] = sum(parameter.numel() for parameter in model.parameters())
    parameter_ratio = max(representative.values()) / min(representative.values())
    if parameter_ratio > 1.10:
        raise ValueError("PointNet/PointNet++ parameter ratio exceeds 1.10")

    smoke_results = []
    if gpu_smoke:
        if not torch.cuda.is_available():
            raise RuntimeError("GPU smoke requested but CUDA is unavailable")
        # Seed 1234 covers all 16 scientific arms; other seeds change only the
        # deterministic initialization/sampling streams already checked above.
        for config in [value for value in configs if value.seed == 1234]:
            result = run(config.source, dry_run=True, device_override="cuda")
            smoke_results.append(result)
    report = {
        "schema_version": 1,
        "created_at": utc_now(),
        "route": matrix["route"],
        "gate_result": "pass",
        "gpu_smoke": gpu_smoke,
        "matrix": {"path": str(matrix_path.resolve()), "sha256": sha256_file(matrix_path)},
        "manifest": {"path": str(MANIFEST_PATH.resolve()), "sha256": sha256_file(MANIFEST_PATH)},
        "stats": {"path": str(STATS_PATH.resolve()), "sha256": sha256_file(STATS_PATH)},
        "configs": [
            {"path": str(config.source), "sha256": sha256_file(config.source)}
            for config in configs
        ],
        "paired_initialization_groups": group_hashes,
        "parameter_counts": representative,
        "parameter_max_min_ratio": parameter_ratio,
        "gpu_smoke_results": smoke_results,
        "test35_read": False,
    }
    atomic_write_json(output, report)
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu-smoke", action="store_true")
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT))
    args = parser.parse_args()
    result = preflight(gpu_smoke=args.gpu_smoke, output=Path(args.output))
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

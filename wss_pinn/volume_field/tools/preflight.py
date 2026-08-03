"""Static preflight for the eight-arm matrix; never submits training."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from wss_pinn.utils import ROOT, atomic_write_json, sha256_file, utc_now

from ..config import VolumeExperimentConfig
from ..models import build_volume_model
from ..utils import tensor_state_sha256


CONFIG_ROOT = ROOT / "wss_pinn/configs/volume_uvwp_peak_v1"
DEFAULT_OUTPUT = ROOT / "outputs/wss_pinn/audits/volume_uvwp_peak_v1_preflight/report.json"
IMPLEMENTATION_FILES = (
    ROOT / "wss_pinn/volume_field/config.py",
    ROOT / "wss_pinn/volume_field/data/audit.py",
    ROOT / "wss_pinn/volume_field/data/builder.py",
    ROOT / "wss_pinn/volume_field/data/dataset.py",
    ROOT / "wss_pinn/volume_field/models/point_models.py",
    ROOT / "wss_pinn/volume_field/physics/rheology.py",
    ROOT / "wss_pinn/volume_field/physics/residuals.py",
    ROOT / "wss_pinn/volume_field/losses.py",
    ROOT / "wss_pinn/volume_field/train.py",
    ROOT / "wss_pinn/volume_field/evaluate.py",
    ROOT / "wss_pinn/volume_field/cluster/preflight.slurm",
    ROOT / "wss_pinn/volume_field/cluster/run_experiment.slurm",
    ROOT / "wss_pinn/volume_field/cluster/submit_matrix.py",
    ROOT / "wss_pinn/volume_field/tools/preflight.py",
)


def _paired_protocol(config: VolumeExperimentConfig) -> dict:
    payload = config.as_dict()
    payload["experiment"].pop("id", None)
    payload["experiment"].pop("description", None)
    payload["experiment"].pop("mode", None)
    payload["paths"].pop("run_dir", None)
    payload["physics"]["enabled"] = "paired_variable"
    for key in ("continuity_weight", "momentum_weight", "no_slip_weight"):
        payload["loss"][key] = "paired_variable"
    return payload


def run(config_root: Path = CONFIG_ROOT) -> dict:
    matrix_path = config_root / "matrix.json"
    matrix = json.loads(matrix_path.read_text(encoding="utf-8"))
    if matrix.get("route") != "volume_uvwp_peak_v1":
        raise ValueError("matrix route drift")
    if matrix.get("warm_start") is not False:
        raise ValueError("matrix must explicitly forbid warm-start")
    split_path = ROOT / VolumeExperimentConfig.from_json(
        config_root / matrix["pairs"][0][0]
    )["paths"]["split"]
    if sha256_file(split_path) != matrix.get("split_sha256"):
        raise ValueError("matrix split hash drift")
    flattened = [name for pair in matrix.get("pairs", []) for name in pair]
    if len(flattened) != 8 or len(set(flattened)) != 8:
        raise ValueError("matrix must contain exactly eight unique experiment configs")
    provenance = matrix.get("architecture_provenance", {})
    workbook = provenance.get("selection_workbook", {})
    workbook_path = ROOT / workbook.get("path", "")
    if not workbook_path.is_file() or sha256_file(workbook_path) != workbook.get("sha256"):
        raise ValueError("architecture-selection workbook provenance drift")
    for architecture in ("pointnet", "pointnetpp"):
        source = provenance.get(architecture, {})
        source_path = ROOT / source.get("config", "")
        if not source_path.is_file() or sha256_file(source_path) != source.get("sha256"):
            raise ValueError(f"{architecture} anchor-config provenance drift")
    pair_reports = []
    all_config_hashes = {}
    for data_name, pinn_name in matrix["pairs"]:
        data_config = VolumeExperimentConfig.from_json(config_root / data_name)
        pinn_config = VolumeExperimentConfig.from_json(config_root / pinn_name)
        if _paired_protocol(data_config) != _paired_protocol(pinn_config):
            raise ValueError(f"paired protocol drift: {data_name} vs {pinn_name}")
        hashes = []
        parameter_counts = []
        for config in (data_config, pinn_config):
            torch.manual_seed(int(config["train"]["seed"]))
            model = build_volume_model(config)
            hashes.append(tensor_state_sha256(model.state_dict()))
            parameter_counts.append(sum(parameter.numel() for parameter in model.parameters()))
            all_config_hashes[config.source.name] = {
                "source_sha256": sha256_file(config.source),
                "resolved_sha256": config.resolved_sha256,
            }
        if hashes[0] != hashes[1]:
            raise ValueError(f"paired initialization mismatch: {data_name} vs {pinn_name}")
        if parameter_counts[0] != parameter_counts[1]:
            raise ValueError(f"paired parameter-count mismatch: {data_name} vs {pinn_name}")
        pair_reports.append(
            {
                "data_only": data_name,
                "pinn": pinn_name,
                "architecture": data_config.architecture,
                "input_variant": data_config.input_variant,
                "initialization_state_sha256": hashes[0],
                "parameters": parameter_counts[0],
                "warm_start": False,
                "protocol_match": True,
            }
        )
    return {
        "schema_version": 1,
        "created_at": utc_now(),
        "status": "completed",
        "gate_result": "pass",
        "route": "volume_uvwp_peak_v1",
        "matrix": {"path": str(matrix_path.resolve()), "sha256": sha256_file(matrix_path)},
        "split_sha256": matrix["split_sha256"],
        "pairs": pair_reports,
        "configs": all_config_hashes,
        "implementation": {
            str(path.relative_to(ROOT)): sha256_file(path)
            for path in IMPLEMENTATION_FILES
        },
        "formal_training_submitted": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config-root", default=str(CONFIG_ROOT))
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT))
    args = parser.parse_args()
    report = run(Path(args.config_root))
    output = atomic_write_json(args.output, report)
    print(json.dumps({"status": "completed", "output": str(output)}, ensure_ascii=False))


if __name__ == "__main__":
    main()

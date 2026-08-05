"""Static preflight for the eight-arm matrix; never submits training."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from wss_pinn.utils import ROOT, atomic_write_json, sha256_file, utc_now

from ..config import ExperimentConfig
from ..models import build_model
from ..volume_utils import tensor_state_sha256


CONFIG_ROOT = ROOT / "wss_pinn/configs/volume_uvwp_peak_v1"
DEFAULT_OUTPUT = ROOT / "outputs/wss_pinn/audits/volume_uvwp_peak_v1_preflight/report.json"
IMPLEMENTATION_FILES = (
    ROOT / "wss_pinn/config.py",
    ROOT / "wss_pinn/data/alignment.py",
    ROOT / "wss_pinn/data/audit.py",
    ROOT / "wss_pinn/data/builder.py",
    ROOT / "wss_pinn/data/dataset.py",
    ROOT / "wss_pinn/data/raw_io.py",
    ROOT / "wss_pinn/models/point_models.py",
    ROOT / "wss_pinn/physics/rheology.py",
    ROOT / "wss_pinn/physics/residuals.py",
    ROOT / "wss_pinn/losses.py",
    ROOT / "wss_pinn/train.py",
    ROOT / "wss_pinn/evaluate.py",
    ROOT / "wss_pinn/volume_utils.py",
    ROOT / "wss_pinn/cluster/preflight.slurm",
    ROOT / "wss_pinn/cluster/run_experiment.slurm",
    ROOT / "wss_pinn/cluster/submit_matrix.py",
    ROOT / "wss_pinn/tools/preflight.py",
    ROOT / "wss_pinn/tools/build_qs_smooth_v3.py",
    ROOT / "wss_pinn/cluster/launch_node04_qs_smooth_v3.py",
    ROOT / "wss_pinn/cluster/monitor_node04_qs_smooth_v3.py",
)


def _paired_protocol(config: ExperimentConfig) -> dict:
    payload = config.as_dict()
    payload["experiment"].pop("id", None)
    payload["experiment"].pop("description", None)
    payload["experiment"].pop("mode", None)
    payload["paths"].pop("run_dir", None)
    payload["physics"]["enabled"] = "paired_variable"
    for key in ("continuity_weight", "momentum_weight", "no_slip_weight"):
        payload["loss"][key] = "paired_variable"
    for key in ("lambda_bc", "lambda_pde"):
        payload["loss"][key] = "paired_variable"
    return payload


def run(config_root: Path = CONFIG_ROOT) -> dict:
    matrix_path = config_root / "matrix.json"
    matrix = json.loads(matrix_path.read_text(encoding="utf-8"))
    route = str(matrix.get("route", ""))
    if route not in {"volume_uvwp_peak_v1", "volume_uvwp_peak_qs_smooth_v3"}:
        raise ValueError("matrix route drift")
    if matrix.get("warm_start") is not False:
        raise ValueError("matrix must explicitly forbid warm-start")
    submission = matrix.get("submission", {})
    if not isinstance(submission.get("enabled"), bool) or any(
        not submission.get(key) for key in ("preflight_report", "data_gate", "output")
    ):
        raise ValueError("matrix submission paths/enabled flag are required")
    groups = matrix.get("pairs") if route == "volume_uvwp_peak_v1" else matrix.get("groups")
    if not groups:
        raise ValueError("matrix experiment groups are missing")
    split_path = Path(
        ExperimentConfig.from_json(config_root / groups[0][0])["paths"]["split"]
    )
    if sha256_file(split_path) != matrix.get("split_sha256"):
        raise ValueError("matrix split hash drift")
    flattened = [name for group in groups for name in group]
    expected_count = 8 if route == "volume_uvwp_peak_v1" else 6
    if len(flattened) != expected_count or len(set(flattened)) != expected_count:
        raise ValueError(f"matrix must contain exactly {expected_count} unique configs")
    if route == "volume_uvwp_peak_v1":
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
    for group in groups:
        configs = [ExperimentConfig.from_json(config_root / name) for name in group]
        if any(config.route != route for config in configs):
            raise ValueError("config route does not match matrix route")
        reference_protocol = _paired_protocol(configs[0])
        if any(_paired_protocol(config) != reference_protocol for config in configs[1:]):
            raise ValueError(f"paired protocol drift: {group}")
        hashes = []
        parameter_counts = []
        for config in configs:
            torch.manual_seed(int(config["train"]["seed"]))
            model = build_model(config)
            hashes.append(tensor_state_sha256(model.state_dict()))
            parameter_counts.append(sum(parameter.numel() for parameter in model.parameters()))
            all_config_hashes[config.source.name] = {
                "source_sha256": sha256_file(config.source),
                "resolved_sha256": config.resolved_sha256,
            }
        if len(set(hashes)) != 1:
            raise ValueError(f"paired initialization mismatch: {group}")
        if len(set(parameter_counts)) != 1:
            raise ValueError(f"paired parameter-count mismatch: {group}")
        pair_reports.append(
            {
                "configs": group,
                "modes": [config.mode for config in configs],
                "architecture": configs[0].architecture,
                "input_variant": configs[0].input_variant,
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
        "route": route,
        "matrix": {"path": str(matrix_path.resolve()), "sha256": sha256_file(matrix_path)},
        "split_sha256": matrix["split_sha256"],
        "submission": submission,
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

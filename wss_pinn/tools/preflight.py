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
    ROOT / "wss_pinn/tools/build_field_v4.py",
    ROOT / "wss_pinn/tools/b0_atlas_v4.py",
    ROOT / "wss_pinn/tools/diagnose_field_v4.py",
    ROOT / "wss_pinn/tools/audit_boundary_field_v4.py",
    ROOT / "wss_pinn/tools/gate_stage0a_field_v4.py",
    ROOT / "wss_pinn/tools/gate_stage1_raw_field_v4.py",
    ROOT / "wss_pinn/tools/gate_stage1_single_seed_field_v4.py",
    ROOT / "wss_pinn/tools/finalize_stage1_field_v4.py",
    ROOT / "wss_pinn/tools/smoke_field_v4.py",
    ROOT / "wss_pinn/validation.py",
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
    if config.route == "volume_uvwp_peak_field_v4":
        payload["model"]["field_v4"]["conditioner"] = "paired_variable"
    return payload


def _confirmation_protocol(
    config: ExperimentConfig, *, normalize_seed: bool, normalize_query_encoding: bool
) -> dict:
    """Normalize only the preregistered axes of the supplemental top-2 matrix."""
    payload = config.as_dict()
    payload["experiment"].pop("id", None)
    payload["experiment"].pop("description", None)
    payload["paths"].pop("run_dir", None)
    if normalize_seed:
        payload["train"]["seed"] = "confirmation_seed"
    if normalize_query_encoding:
        payload["model"]["field_v4"]["query_encoding"] = "selected_query_axis"
    return payload


def run(config_root: Path = CONFIG_ROOT) -> dict:
    matrix_path = config_root / "matrix.json"
    matrix = json.loads(matrix_path.read_text(encoding="utf-8"))
    route = str(matrix.get("route", ""))
    if route not in {
        "volume_uvwp_peak_v1",
        "volume_uvwp_peak_qs_smooth_v3",
        "volume_uvwp_peak_field_v4",
    }:
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
    expected_count = {
        "volume_uvwp_peak_v1": 8,
        "volume_uvwp_peak_qs_smooth_v3": 6,
        "volume_uvwp_peak_field_v4": 4,
    }[route]
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
    all_parameter_counts = []
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
            all_parameter_counts.append(parameter_counts[-1])
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
    extra = {}
    if route == "volume_uvwp_peak_field_v4":
        decision_path = config_root / str(matrix.get("decision_contract", ""))
        if not decision_path.is_file():
            raise ValueError("V4 decision contract is required")
        decision = json.loads(decision_path.read_text(encoding="utf-8"))
        if (
            decision.get("status") != "preregistered_before_formal_training"
            or decision.get("primary_selection") != "validation_field_score_cb"
            or "test35" not in decision.get("forbidden_selection_inputs", [])
        ):
            raise ValueError("V4 decision contract drift")
        ratio = max(all_parameter_counts) / min(all_parameter_counts)
        if ratio > float(decision["equivalence_guard"]["parameter_count_max_to_min"]):
            raise ValueError("V4 parameter-count ratio exceeds the preregistered Gate")
        first = ExperimentConfig.from_json(config_root / flattened[0])
        manifest = json.loads(
            Path(first["paths"]["sidecar_manifest"]).read_text(encoding="utf-8")
        )
        if (
            manifest.get("route") != route
            or manifest.get("counts") != {"train": 123, "val": 15, "total": 138}
            or any(row.get("role") == "test" for row in manifest.get("cases", []))
        ):
            raise ValueError("V4 train/val-only manifest Gate failed")
        query_manifest = json.loads(
            Path(first["paths"]["validation_query_manifest"]).read_text(
                encoding="utf-8"
            )
        )
        query_hashes = {
            row["case_id"]: row["query_sha256"]
            for row in query_manifest.get("cases", [])
        }
        if len(query_hashes) != 15 or len(set(query_hashes.values())) != 15:
            raise ValueError("V4 fixed validation query manifest drift")
        extra = {
            "decision_contract": {
                "path": str(decision_path.resolve()),
                "sha256": sha256_file(decision_path),
            },
            "parameter_count_max_to_min": ratio,
            "fixed_validation_query_hashes": query_hashes,
            "test35_guard": "pass",
        }
        if matrix.get("matrix_kind") == "supplemental_multiseed_top2":
            selection = matrix.get("selection_gate", {})
            selection_path = Path(str(selection.get("path", "")))
            if not selection_path.is_absolute():
                selection_path = ROOT / selection_path
            if (
                not selection_path.is_file()
                or sha256_file(selection_path) != selection.get("sha256")
            ):
                raise ValueError("supplemental selection Gate hash drift")
            selection_report = json.loads(selection_path.read_text(encoding="utf-8"))
            if (
                selection_report.get("gate_result") != "pass"
                or selection_report.get("selected_top2")
                != ["F4-G-PE-s1234", "F4-G-RAW-s1234"]
                or matrix.get("selected_arms") != ["G-PE", "G-Raw"]
                or matrix.get("confirmation_seeds") != [2345, 3456]
            ):
                raise ValueError("supplemental top-2 selection contract drift")
            configs_by_id = {
                ExperimentConfig.from_json(config_root / name)["experiment"]["id"]:
                ExperimentConfig.from_json(config_root / name)
                for name in flattened
            }
            expected_ids = {
                f"F4-G-{encoding}-s{seed}"
                for encoding in ("RAW", "PE")
                for seed in (2345, 3456)
            }
            if set(configs_by_id) != expected_ids:
                raise ValueError("supplemental config membership drift")
            for encoding in ("RAW", "PE"):
                left = configs_by_id[f"F4-G-{encoding}-s2345"]
                right = configs_by_id[f"F4-G-{encoding}-s3456"]
                if _confirmation_protocol(
                    left, normalize_seed=True, normalize_query_encoding=False
                ) != _confirmation_protocol(
                    right, normalize_seed=True, normalize_query_encoding=False
                ):
                    raise ValueError(f"confirmation seed protocol drift: G-{encoding}")
            scientific_pairs = []
            for seed in (2345, 3456):
                raw = configs_by_id[f"F4-G-RAW-s{seed}"]
                pe = configs_by_id[f"F4-G-PE-s{seed}"]
                if _confirmation_protocol(
                    raw, normalize_seed=False, normalize_query_encoding=True
                ) != _confirmation_protocol(
                    pe, normalize_seed=False, normalize_query_encoding=True
                ):
                    raise ValueError(f"confirmation arm protocol drift: seed {seed}")
                if int(raw["sampling"]["seed"]) != 1234 or int(pe["sampling"]["seed"]) != 1234:
                    raise ValueError("fixed evaluation support seed drift")
                scientific_pairs.append(
                    {
                        "seed": seed,
                        "raw": raw["experiment"]["id"],
                        "pe": pe["experiment"]["id"],
                        "protocol_match_except_query_encoding": True,
                        "fixed_evaluation_support_seed": 1234,
                    }
                )
            extra["supplemental_multiseed"] = {
                "selection_gate": {
                    "path": str(selection_path.resolve()),
                    "sha256": sha256_file(selection_path),
                },
                "scientific_pairs": scientific_pairs,
                "test35_guard": "pass",
            }
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
        **extra,
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

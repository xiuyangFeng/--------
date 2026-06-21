from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict

from .data import build_datasets
from .utils import dump_json, load_config


def audit_config(config: Dict[str, Any]) -> Dict[str, Any]:
    datasets = build_datasets(config)
    report: Dict[str, Any] = {
        "experiment_name": config["run"]["experiment_name"],
        "input_features": config["data"]["input_features"],
        "point_filter": config["data"].get("point_filter", "all"),
        "physics_enabled": config.get("physics", {}).get("enabled", False),
        "splits": {},
    }
    for split_name, dataset in datasets.items():
        sample = dataset[0]
        report["splits"][split_name] = {
            "n_samples": len(dataset),
            "input_dim": dataset.input_dim,
            "feature_names": list(dataset.feature_names),
            "first_sample_id": sample["sample_id"],
            "first_n_points": int(sample["features"].shape[1]),
        }
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit CROWN config and pkl compatibility")
    parser.add_argument("--config", required=True)
    parser.add_argument("--out", default=None)
    args = parser.parse_args()
    config = load_config(args.config)
    report = audit_config(config)
    if args.out:
        dump_json(Path(args.out), report)
    else:
        print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

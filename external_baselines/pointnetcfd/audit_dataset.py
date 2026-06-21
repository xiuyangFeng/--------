from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict

from .data import TARGET_MODES, build_datasets
from .utils import dump_json, load_config


def audit_config(config: Dict[str, object]) -> Dict[str, object]:
    datasets = build_datasets(config)
    report: Dict[str, object] = {
        "target_mode": config["target"]["mode"],
        "target_names": TARGET_MODES[config["target"]["mode"]],
        "node_features": config["data"]["node_features"],
        "global_features": config["data"]["global_features"],
        "splits": {},
    }
    for split_name, dataset in datasets.items():
        first = dataset[0]
        report["splits"][split_name] = {
            "n_graph_files": len(dataset),
            "input_dim": dataset.input_dim,
            "global_dim": dataset.global_dim,
            "output_names": dataset.output_names,
            "first_case": first["case_name"],
            "first_sample": first["sample_id"],
            "first_n_points_after_target_filter": int(first["target"].shape[0]),
            "first_node_input_shape": list(first["node_input"].shape),
            "first_target_shape": list(first["target"].shape),
        }
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit PointNetCFD config/data compatibility")
    parser.add_argument("--config", required=True)
    parser.add_argument("--out", default=None, help="Optional JSON output path")
    args = parser.parse_args()
    config = load_config(args.config)
    report = audit_config(config)
    if args.out:
        dump_json(Path(args.out), report)
    else:
        import json

        print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

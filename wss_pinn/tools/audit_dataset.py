"""Audit the new sidecar contract and write JSON/CSV Gate evidence."""

from __future__ import annotations

import argparse
import csv
import json
import os
import tempfile
from pathlib import Path

from wss_pinn.utils import ROOT, atomic_write_json, guard_write_path

from ..data.audit import audit_dataset


DEFAULT_SPLIT = (
    ROOT
    / "wss_pinn/configs/splits/"
    "split_WSS_PINN_AG_AAA_ILO_q2v_pool2025_"
    "train138_test35_exclude_SHI_YUN_XI_v1.json"
)
DEFAULT_ROOT = ROOT / "data_wss_pinn/volume_uvwp_peak_v1_train138_test35"
DEFAULT_OUTPUT = ROOT / "outputs/wss_pinn/audits/volume_uvwp_peak_v1_train138_test35"


def _write_csv(path: Path, rows: list[dict]) -> Path:
    target = guard_write_path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    fields = sorted({key for row in rows for key in row})
    fd, temporary = tempfile.mkstemp(
        prefix=f".{target.name}.", suffix=".tmp", dir=target.parent
    )
    try:
        with os.fdopen(fd, "w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            writer.writerows(rows)
        os.replace(temporary, target)
    finally:
        Path(temporary).unlink(missing_ok=True)
    return target


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", default=str(DEFAULT_SPLIT))
    parser.add_argument("--manifest", default=str(DEFAULT_ROOT / "manifest.json"))
    parser.add_argument("--field-stats", default=str(DEFAULT_ROOT / "field_stats.json"))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT))
    parser.add_argument("--deep-source", action="store_true")
    args = parser.parse_args()
    report = audit_dataset(
        args.manifest,
        args.split,
        args.field_stats,
        deep_source=args.deep_source,
        progress=True,
    )
    output_dir = Path(args.output_dir)
    report_path = atomic_write_json(output_dir / "report.json", report)
    csv_path = _write_csv(output_dir / "cases.csv", report["rows"])
    print(
        json.dumps(
            {
                "status": report["status"],
                "gate_result": report["gate_result"],
                "report": str(report_path),
                "cases_csv": str(csv_path),
            },
            ensure_ascii=False,
        )
    )
    if report["gate_result"] != "pass":
        raise SystemExit(2)


if __name__ == "__main__":
    main()

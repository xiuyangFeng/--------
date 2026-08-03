"""Build all 173 full-volume sidecars; intended for a CPU cluster job."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from wss_pinn.utils import ROOT

from ..data.builder import build_dataset


DEFAULT_SPLIT = (
    ROOT
    / "wss_pinn/configs/splits/"
    "split_WSS_PINN_AG_AAA_ILO_q2v_pool2025_"
    "train138_test35_exclude_SHI_YUN_XI_v1.json"
)
DEFAULT_ROOT = ROOT / "data_wss_pinn/volume_uvwp_peak_v1_train138_test35"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", default=str(DEFAULT_SPLIT))
    parser.add_argument("--sidecar-root", default=str(DEFAULT_ROOT / "cases"))
    parser.add_argument("--manifest", default=str(DEFAULT_ROOT / "manifest.json"))
    parser.add_argument("--field-stats", default=str(DEFAULT_ROOT / "field_stats.json"))
    args = parser.parse_args()
    report = build_dataset(
        split_path=Path(args.split),
        sidecar_root=Path(args.sidecar_root),
        aggregate_path=Path(args.manifest),
        stats_path=Path(args.field_stats),
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

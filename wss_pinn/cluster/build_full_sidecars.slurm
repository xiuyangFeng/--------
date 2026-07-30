#!/bin/bash
# Build WSS-PINN sidecars only after the frozen source-data audit passes.
set -euo pipefail

PROJECT=/public/newhome/cy/Digital_twin/GNN
CONFIG="${WSS_PINN_CONFIG:?WSS_PINN_CONFIG is required}"
AUDIT_REPORT="${WSS_PINN_AUDIT_REPORT:?WSS_PINN_AUDIT_REPORT is required}"
PY="${WSS_PINN_PYTHON:?WSS_PINN_PYTHON is required}"

cd "$PROJECT"
"$PY" - "$AUDIT_REPORT" "$CONFIG" <<'PY'
import json
import sys
from pathlib import Path

from wss_pinn.config import ExperimentConfig
from wss_pinn.utils import sha256_file

path = Path(sys.argv[1])
config = ExperimentConfig.from_json(sys.argv[2])
payload = json.loads(path.read_text(encoding="utf-8"))
if payload.get("gate_result") != "pass":
    raise SystemExit(f"source-data audit did not pass: {path}")
report_split_sha = payload.get("split", {}).get("sha256")
config_split_sha = sha256_file(config["data"]["split_path"])
if report_split_sha != config_split_sha:
    raise SystemExit(
        "source-data audit/config split mismatch: "
        f"{report_split_sha} != {config_split_sha}"
    )
print(
    f"audit_gate=pass total={payload['summary']['total']} "
    f"passed={payload['summary']['passed']} report={path}"
)
PY
echo "sidecar_build_start=$(date --iso-8601=seconds) config=$CONFIG host=$(hostname)"
"$PY" -m wss_pinn.tools.build_physics_sidecar --config "$CONFIG"
echo "sidecar_build_completed=$(date --iso-8601=seconds)"

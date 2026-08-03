#!/usr/bin/env bash
set -euo pipefail

HOST="${NODE04_HOST:-cy@node04}"
RUN_ROOT=/public/newhome/cy/Digital_twin/GNN/wss_mri_calculator/experiments/pointcloud_normal_multiscale_v3/runs

ssh "$HOST" "bash -lc 'if [[ -d $RUN_ROOT ]]; then for d in $RUN_ROOT/*; do [[ -d \"\$d\" ]] || continue; echo ==== \$(basename \"\$d\"); cat \"\$d/status.json\" 2>/dev/null || true; tail -n 4 \"\$d/run.log\" 2>/dev/null || true; done; else echo no_runs; fi'"

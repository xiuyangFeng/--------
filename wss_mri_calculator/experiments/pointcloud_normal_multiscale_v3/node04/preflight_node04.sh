#!/usr/bin/env bash
set -euo pipefail

HOST="${NODE04_HOST:-cy@node04}"

ssh "$HOST" 'bash -s' <<'REMOTE'
set -euo pipefail
expected_uid=1006
expected_gid=1007
project=/public/newhome/cy/Digital_twin/GNN
python=/public/newhome/cy/.conda/envs/GNN/bin/python

echo "host=$(hostname) uid=$(id -u) gid=$(id -g) HOME=$HOME"
if [[ "$(id -u)" != "$expected_uid" || "$(id -g)" != "$expected_gid" ]]; then
  echo "ERROR: node04 UID/GID must be ${expected_uid}/${expected_gid}" >&2
  exit 20
fi
test -r "$project/data_new"
test -x "$python"
"$python" -c 'import numpy, scipy; print("python_ok", numpy.__version__, scipy.__version__)'
nvidia-smi -L
nvidia-smi --query-gpu=index,name,memory.used,memory.total,utilization.gpu --format=csv,noheader
echo "node04_preflight_ok"
REMOTE

#!/bin/bash
#SBATCH --partition=CPU
#SBATCH -w node03
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --job-name=q2vilo_submit
#SBATCH --output=training_wss_min/cluster/logs/%x_%j.out
#SBATCH --error=training_wss_min/cluster/logs/%x_%j.err
#SBATCH --time=01:00:00

set -euo pipefail

PROJECT=/public/newhome/cy/Digital_twin/GNN
PY=/public/newhome/cy/.conda/envs/GNN/bin/python
cd "$PROJECT"

echo "stage=hash_gated_submission started=$(date --iso-8601=seconds) host=$(hostname)"
"$PY" -m training_wss_min.cluster.submit_q2v_ilo_arch_matrix

LO_PROFILE=$(mktemp -d /tmp/q2vilo_lo_XXXXXX)
LO_PORT=2085
/usr/bin/libreoffice --headless --nologo --nodefault --nofirststartwizard \
  "-env:UserInstallation=file://$LO_PROFILE" \
  --accept="socket,host=localhost,port=$LO_PORT;urp;StarOffice.ComponentContext" \
  >"$LO_PROFILE/stdout.log" 2>"$LO_PROFILE/stderr.log" &
LO_PID=$!
cleanup() {
  kill "$LO_PID" 2>/dev/null || true
  wait "$LO_PID" 2>/dev/null || true
  rm -rf "$LO_PROFILE"
}
trap cleanup EXIT
for _ in $(seq 1 30); do
  python3 -c "import uno; c=uno.getComponentContext(); c.ServiceManager.createInstanceWithContext('com.sun.star.bridge.UnoUrlResolver',c).resolve('uno:socket,host=localhost,port=$LO_PORT;urp;StarOffice.ComponentContext')" 2>/dev/null && break
  sleep 1
done
python3 -m training_wss_min.tools.update_q2v_matrix_xlsx_uno \
  --uno-url "uno:socket,host=localhost,port=$LO_PORT;urp;StarOffice.ComponentContext"
unzip -t 'docs/03-汇报材料/WSS_PointNet实验矩阵与结果汇总last.xlsx' >/dev/null
"$PY" -m training_wss_min.tools.finalize_q2v_submission_records
echo "stage=hash_gated_submission status=completed finished=$(date --iso-8601=seconds)"

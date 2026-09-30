#!/bin/bash
#SBATCH --partition=GPU
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --job-name=cycle_reeval
#SBATCH --output=training_wss_min/cluster/logs/%x_%j.out
#SBATCH --error=training_wss_min/cluster/logs/%x_%j.err
# 2026-09-21：evaluate.py 新增周期积分量落地一致性块 result["cycle_agreement"]（仅 tawss/osi 目标）。
# 在冻结副本里重评九臂 × best/last，让 metrics.json 带上该块；旧 metrics.json 先备份为 metrics_pre_agreement_20260921.json，
# 并在重评后逐字段核对旧字段不变（容差 1e-4，GPU 归约噪声）。
set -euo pipefail
FROZEN=/public/newhome/cy/Digital_twin/GNN_cycle_frozen_20260920
MAIN=/public/newhome/cy/Digital_twin/GNN
PY=/public/newhome/cy/.conda/envs/GNN/bin/python
RUNS=$MAIN/training_wss_min/runs/wss_cycle_20260920
cd "$FROZEN"
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
date -Is; hostname; nvidia-smi --query-gpu=index,name,memory.used,memory.total --format=csv
for arm in A1 O1 O2; do for k in 0 1 2; do for ck in best last; do
  run=$RUNS/${arm}_f${k}_s1234
  old=$run/eval/ckpt_$ck/metrics.json
  if [ -f "$old" ] && [ ! -f "$run/eval/ckpt_$ck/metrics_pre_agreement_20260921.json" ]; then
    cp -p "$old" "$run/eval/ckpt_$ck/metrics_pre_agreement_20260921.json"
  fi
  echo "===== ${arm}_f${k} $ck $(date -Is) ====="
  $PY -u -m training_wss_min.evaluate --run-dir "$run" --partitions test --allow-test --checkpoint $ck --no-plots --save-predictions
done; done; done
echo "===== compare old fields $(date -Is) ====="
$PY - <<'EOF'
import json, math, glob
def leaves(d, p=''):
    if isinstance(d, dict):
        for k, v in d.items(): yield from leaves(v, f'{p}.{k}' if p else k)
    elif isinstance(d, list):
        for i, v in enumerate(d): yield from leaves(v, f'{p}[{i}]')
    elif isinstance(d, (int, float)) and not isinstance(d, bool): yield p, float(d)
worst = 0.0; n = 0; bad = []
for old in sorted(glob.glob('/public/newhome/cy/Digital_twin/GNN/training_wss_min/runs/wss_cycle_20260920/*/eval/ckpt_*/metrics_pre_agreement_20260921.json')):
    new = old.replace('metrics_pre_agreement_20260921.json', 'metrics.json')
    a = dict(leaves(json.load(open(old)))); b = dict(leaves(json.load(open(new))))
    missing = [k for k in a if k not in b]
    if missing: bad.append((old, 'missing', missing[:5]))
    for k, v in a.items():
        if k in b and math.isfinite(v) and math.isfinite(b[k]):
            d = abs(v - b[k]); worst = max(worst, d); n += 1
            if d > 1e-4: bad.append((old, k, v, b[k]))
    print(old.split('/')[-4], old.split('/')[-2], 'has cycle_agreement:', 'cycle_agreement' in json.load(open(new))['test'])
print('compared', n, 'fields; max abs diff', worst, '; violations', len(bad))
for x in bad[:20]: print('  ', x)
raise SystemExit(1 if bad else 0)
EOF
echo "===== reeval done $(date -Is) ====="

#!/usr/bin/env bash
# V3P-J5 全链路：build split → GPU 40ep → predict → L3 eval（I6 vs J5）
set -eo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$PROJECT_DIR"
export PATH="/public/slurm/bin:/public/slurm/sbin:${PATH:-}"

J5_RUN="${J5_RUN:-$(cat training/cluster/logs/v3p_j5_run_dir.txt 2>/dev/null || ls -dt outputs/field/field_v3_pointnext_j5activedata_*_split_AG_active_v2_j5_seed1_* 2>/dev/null | head -1)}"
export J5_RUN
DATE_TAG="${DATE_TAG:-20260701}"

mkdir -p training/cluster/logs logs

echo "=== [0] build J5 split ==="
/public/newhome/cy/.conda/envs/GNN/bin/python -m training.scripts.build_v3p_j5_split

echo "=== [1] submit J5 GPU train ==="
JOB_TRAIN=$(/public/slurm/bin/sbatch --parsable training/cluster/run_train_field.slurm "$CONFIG")
echo "J5 train Job: $JOB_TRAIN"

JOB_PRED=$(/public/slurm/bin/sbatch --parsable --dependency=afterok:"$JOB_TRAIN" \
  training/cluster/run_v3p_j5_predict.slurm)
echo "J5 predict Job: $JOB_PRED (afterok $JOB_TRAIN)"

JOB_L3=$(/public/slurm/bin/sbatch --parsable --dependency=afterok:"$JOB_PRED" \
  --export=ALL,DATE_TAG="$DATE_TAG" \
  training/cluster/run_v3p_j5_l3_eval.slurm)
echo "J5 L3 eval Job: $JOB_L3 (afterok $JOB_PRED)"

echo "$JOB_TRAIN" > training/cluster/logs/v3p_j5_train_job_id.txt
echo "$JOB_PRED" > training/cluster/logs/v3p_j5_predict_job_id.txt
echo "$JOB_L3" > training/cluster/logs/v3p_j5_l3_job_id.txt

squeue -u "${USER:-cy}" 2>/dev/null || true

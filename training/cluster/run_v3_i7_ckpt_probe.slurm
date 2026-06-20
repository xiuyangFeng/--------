#!/bin/bash
#SBATCH --partition=CPU
#SBATCH -w node03
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --job-name=v3p_i7_probe
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --time=4:00:00

set -eo pipefail

module purge
TRAINING_ENV=${TRAINING_ENV:-GNN}
TRAINING_PYTHON=${TRAINING_PYTHON:-/public/newhome/cy/.conda/envs/GNN/bin/python}

PROJECT_DIR="${SLURM_SUBMIT_DIR:-$(pwd)}"
cd "$PROJECT_DIR"
mkdir -p "$PROJECT_DIR/logs" "$PROJECT_DIR/outputs/field/f0_decision"

export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_MAX_THREADS=1

echo "=== V3P I7 ckpt probe $(date) ==="
echo "Job: $SLURM_JOB_ID · Node: $SLURM_NODELIST"
echo "Python: $TRAINING_PYTHON"

"$TRAINING_PYTHON" -m training.scripts.run_v3_i7_ckpt_probe \
  --max-train-wall "${I7_MAX_TRAIN_WALL:-120000}" \
  --max-test-wall "${I7_MAX_TEST_WALL:-60000}" \
  --max-wall-per-graph "${I7_MAX_WALL_PER_GRAPH:-2048}"

echo ">>> V3P I7 ckpt probe 完成 $(date)"

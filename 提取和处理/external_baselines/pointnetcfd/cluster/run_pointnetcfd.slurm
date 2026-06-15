#!/usr/bin/env bash
#SBATCH --job-name=pointnetcfd
#SBATCH --output=external_baselines/pointnetcfd/cluster/logs/pointnetcfd_%j.out
#SBATCH --error=external_baselines/pointnetcfd/cluster/logs/pointnetcfd_%j.err
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=24:00:00

set -euo pipefail

CONFIG_PATH="${1:?missing config path}"
mkdir -p external_baselines/pointnetcfd/cluster/logs

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate rag_venv

export PYTHONPATH="$PWD:${PYTHONPATH:-}"
python -m external_baselines.pointnetcfd.train --config "${CONFIG_PATH}"

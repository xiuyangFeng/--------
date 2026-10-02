#!/bin/bash
#SBATCH --partition=GPU
#SBATCH --gres=gpu:4090:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --job-name=vol_v52d_accept
#SBATCH --output=/public/newhome/cy/Digital_twin/GNN/training_wss_min/experiments/wss_deploy_timing_20260917/acceptance_recover8_volume_v52d_20261003/run_%j.out
#SBATCH --time=03:00:00
A=/public/newhome/cy/Digital_twin/GNN/training_wss_min/experiments/wss_deploy_timing_20260917/acceptance_recover8_volume_v52d_20261003
S=/public/newhome/cy/Digital_twin/GNN/outputs/wss_deploy_release_staging_20261003
PY=/public/newhome/cy/.conda/envs/GNN/bin/python
cd /public/newhome/cy/Digital_twin/GNN; export PYTHONPATH=$PWD OMP_NUM_THREADS=8
date -Is; nvidia-smi --query-gpu=name --format=csv,noheader
echo "== model equivalence"; $PY -u $A/model_equivalence_volume.py $S $A/model_equivalence_volume.json
echo "== end to end"; mkdir -p /tmp/vol_accept_$SLURM_JOB_ID; $PY -u $A/e2e_recover8_volume.py $S /tmp/vol_accept_$SLURM_JOB_ID && cp /tmp/vol_accept_$SLURM_JOB_ID/e2e_recover8_volume.json $A/
date -Is

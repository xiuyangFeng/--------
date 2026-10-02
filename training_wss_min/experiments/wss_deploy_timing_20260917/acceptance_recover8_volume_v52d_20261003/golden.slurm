#!/bin/bash
#SBATCH --partition=GPU
#SBATCH --gres=gpu:4090:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=96G
#SBATCH --job-name=golden_v0164
#SBATCH --output=/public/newhome/cy/Digital_twin/GNN/training_wss_min/experiments/wss_deploy_timing_20260917/acceptance_recover8_volume_v52d_20261003/golden_%j.out
#SBATCH --time=01:30:00
A=/public/newhome/cy/Digital_twin/GNN/training_wss_min/experiments/wss_deploy_timing_20260917/acceptance_recover8_volume_v52d_20261003
cd /public/newhome/cy/Digital_twin/GNN; export PYTHONPATH=$PWD TZ=Asia/Shanghai
/public/newhome/cy/.conda/envs/GNN/bin/python -m wss_deploy.regress --jobs-root outputs/wss_deploy_golden/20260920_baseline --out /tmp/golden_v0164_$SLURM_JOB_ID --report $A/golden_regress_retired_fallback.json
echo EXIT $?

#!/bin/bash
#SBATCH --partition=GPU
#SBATCH --gres=gpu:4090:2
#SBATCH --cpus-per-task=16
#SBATCH --mem=96G
#SBATCH --job-name=pf6vf6_bench
#SBATCH --output=/public/newhome/cy/Digital_twin/GNN/training_wss_min/experiments/pf6vf6_v52d_retrain_20261002/bench/bench_4090_%j.out
#SBATCH --time=00:40:00
G=(${CUDA_VISIBLE_DEVICES//,/ })
bash /public/newhome/cy/Digital_twin/GNN/training_wss_min/experiments/pf6vf6_v52d_retrain_20261002/bench/run_bench.sh g4090 ${G[0]} ${G[1]}

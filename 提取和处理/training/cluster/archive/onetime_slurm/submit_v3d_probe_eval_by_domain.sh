#!/usr/bin/env bash
# 【归档 · 2026-05-22】V3D post-4901 探针一次性分域评估提交。
# 通用入口请用: training/cluster/submit_eval_by_domain.sh
#
# 用法（仓库根目录）:
#   bash training/cluster/archive/onetime_slurm/submit_v3d_probe_eval_by_domain.sh
#
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
CLUSTER_DIR="$ROOT/training/cluster"
cd "$ROOT"

PROBE_P_DIR="outputs/field/field_v3d_pointnext_localpool_probe_p01_geom_wall13000_near2000_split_data_new_v3_v3_seed1_20260521_103843"
PROBE_WSS_DIR="outputs/field/field_v3d_pointnext_localpool_probe_wss01_geom_wall13000_near2000_split_data_new_v3_v3_seed1_20260521_101738"

for d in "$PROBE_P_DIR" "$PROBE_WSS_DIR"; do
    if [ ! -f "$d/config.snapshot.json" ]; then
        echo "错误: 缺少 $d/config.snapshot.json"
        exit 1
    fi
done

J1=$(sbatch --parsable \
    --job-name=v3d_eval_p_domain \
    --export=ALL,RUN_DIR_REL="$PROBE_P_DIR",CHECKPOINT=best_model.pt,SUBSET=test \
    "$CLUSTER_DIR/run_eval_field_by_domain.slurm")
echo "Probe-P 分域评估已提交: job $J1"

J2=$(sbatch --parsable \
    --job-name=v3d_eval_wss_domain \
    --export=ALL,RUN_DIR_REL="$PROBE_WSS_DIR",CHECKPOINT=best_wss_model.pt,SUBSET=test \
    "$CLUSTER_DIR/run_eval_field_by_domain.slurm")
echo "Probe-WSS 分域评估已提交: job $J2"

echo ""
echo "监控: squeue -u \$USER"
echo "日志: logs/eval_by_domain_${J1}.out  (Probe-P，仓库根提交时)"
echo "      logs/eval_by_domain_${J2}.out  (Probe-WSS)"
echo "产出: <run_dir>/eval_by_domain_test/metrics_by_domain.json"

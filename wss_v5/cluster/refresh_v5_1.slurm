#!/bin/bash
#SBATCH --job-name=v5_1refresh
#SBATCH --partition=CPU
#SBATCH --time=0
#SBATCH --mem=300G
#SBATCH --cpus-per-task=48
#SBATCH --output=outputs/wss_pinn/slurm/v5_1_refresh_%j.out
#SBATCH --error=outputs/wss_pinn/slurm/v5_1_refresh_%j.err
# v5.1 母库刷新（跟踪文档 §25 第 1–3 步）。前提：contract.py 已切到 v5.1（apply_contract_v5_1.py），26 例重跑完成并瘦身。
set -euo pipefail
ulimit -n 65535
cd /public/newhome/cy/Digital_twin/GNN
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2
PY=/public/newhome/cy/.conda/envs/GNN/bin/python
V=data_wss_v5/views_v5_1
VIEW=$V/wss_min_view_v1
RUN=${V5_RUN_NAME:-full_v5_1_$(date +%Y%m%d)}
echo "== 0 contract"; "$PY" -c "from wss_v5 import contract as C; print(C.SCHEMA_VERSION, C.SNAPSHOT_ROOT, C.SPLIT_PATH.name)"
echo "== 1 build 170 cases -> $("$PY" -c 'from wss_v5 import contract as C; print(C.SNAPSHOT_ROOT)')"
if [ "${V5_SKIP_BUILD:-0}" != "1" ]; then
  if [ -n "${V5_CASES:-}" ]; then SEL="--cases $V5_CASES"; else SEL="--all"; fi   # 注意：${V5_CASES:---all} 在变量非空时会再次展开变量本身（曾导致 19 例被重复构建），不能用
  "$PY" -m wss_v5.build $SEL --resume --workers "${V5_WORKERS:-20}" --run-name "$RUN" --domain-subsample "${V5_DOMAIN_SUBSAMPLE:-6000}" --n-generate "${V5_N_GENERATE:-60000}"
fi
echo "== 2 wss_min_view -> $VIEW"; "$PY" -m wss_v5.views.wss_min_view --out-root "$VIEW"
echo "== 3 wall_geom_v2"; "$PY" -m wss_v5.views.wall_geom_v2 --view-root "$VIEW" --out "$V/wss_min_geom_v2" --workers 24
echo "== 4 capfit rule (train136)"; mkdir -p "$V/wss_min_flowref_v1"; "$PY" -m training_wss_min.tools.fit_flow_split_rule --expect-train 136 --out "$V/wss_min_flowref_v1/flow_split_rule_train136.json"
echo "== 5 wall_flowref_v1"; "$PY" -m wss_v5.views.wall_flowref_v1 --view-root "$VIEW" --split "$VIEW/split_V5_train136_test34.json" --capfit-rule "$V/wss_min_flowref_v1/flow_split_rule_train136.json" --out "$V/wss_min_flowref_v1" --workers 24
echo "== 6 volume_view (+ volume_stats_train136)"; "$PY" -m wss_v5.views.volume_view --out-root "$VIEW" --workers 8
echo "== 7 density sidecars"; "$PY" -m training_wss_min.tools.build_density_sidecars --view-root "$VIEW" --out "$V/wss_min_density_v1" --workers 24
echo "== 8 cv3 (train136)"; "$PY" -m training_wss_min.tools.make_cv3_splits --view-root "$VIEW" --source-name split_V5_train136_test34.json --out-name cv3_v51 --expect-train 136
echo "== done"; ls "$VIEW" | head; date

#!/bin/bash
# 第三批 13 例验收（2026-09-22）：与第二批同口径三条线——
#   1) rcr_runtime_audit.py（最新 transcript：末周期 L/R 0.5±0.05、四出口末步 ≥ 3 kPa（正常 10–16）、Σ|Q_out| = 入口 ±5 %）→ rcr_runtime_audit_batch3.csv
#   2) wall_scan_1162.py（壁面 1162 步压力 p99 13–20 kPa、WSS 分位）→ wall_scan_1162_rerun_batch3_cases.csv
#   3) rcr_protocol_audit_new.py（UDF vs 网格面积；PENG_JI_MING 的 R1 隐含面积项按 ×1e6 解读）→ rcr_protocol_audit_batch3_final.csv
#   外加：每例 ascii/ 与 ascii_in/ 帧数、Slurm 状态与时长。用法：bash accept_batch3.sh
set -u
export PATH=/public/slurm/bin:$PATH
ROOT=/public/newhome/cy/Digital_twin/GNN; R=$ROOT/data_new; HERE=$(cd "$(dirname "$0")" && pwd); PY=/public/newhome/cy/.conda/envs/GNN/bin/python
mapfile -t CASES < <(grep -v '^#' "$HERE/rerun_batch3_cases.txt")
echo "== Slurm 状态 / 帧数"
for c in "${CASES[@]}"; do j=$(grep "\`$c\`（第 3 批" "$HERE/rerun_jobs.md" | tail -1 | awk -F'|' '{gsub(/ /,"",$3); print $3}'); printf "%-28s job %s %-10s %s  ascii=%s ascii_in=%s\n" "$c" "$j" "$(sacct -j $j -X -n -o State | head -1 | tr -d ' ')" "$(sacct -j $j -X -n -o Elapsed | head -1 | tr -d ' ')" "$(ls $R/$c/ascii 2>/dev/null | wc -l)" "$(ls $R/$c/ascii_in 2>/dev/null | wc -l)"; done
echo; echo "== 1 运行态 RCR 审计"
cd "$ROOT" && PYTHONPATH=$ROOT $PY "$HERE/rcr_runtime_audit.py" "${CASES[@]/#/$R/}" && cp "$HERE/rcr_runtime_audit.csv" "$HERE/rcr_runtime_audit_batch3.csv"
echo; echo "== 2 壁面 1162 步扫描"
PYTHONPATH=$ROOT $PY "$HERE/wall_scan_1162.py" --cases-file "$HERE/rerun_batch3_cases.txt"
echo; echo "== 3 协议审计复核"
PYTHONPATH=$ROOT $PY "$HERE/rcr_protocol_audit_new.py" --cases-file "$HERE/rerun_batch3_cases.txt" --out "$HERE/rcr_protocol_audit_batch3_final.csv" 2>&1 | grep -v Warning

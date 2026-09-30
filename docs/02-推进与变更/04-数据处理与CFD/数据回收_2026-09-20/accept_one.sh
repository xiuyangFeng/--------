#!/bin/bash
# 单例即时验收（第三批用）：bash accept_one.sh <canonical_id> <jobid>  → 帧数 / 运行态 RCR / 壁面 1162 / vf-in 峰值，并追加一行到 rerun_jobs.md
set -u
c=$1; j=$2; ROOT=/public/newhome/cy/Digital_twin/GNN; D="$ROOT/docs/02-推进与变更/04-数据处理与CFD/数据回收_2026-09-20"; PY=/public/newhome/cy/.conda/envs/GNN/bin/python; S=$(mktemp -d)
export PATH=/public/slurm/bin:$PATH
na=$(ls $ROOT/data_new/$c/ascii | wc -l); ni=$(ls $ROOT/data_new/$c/ascii_in | wc -l)
cd $ROOT; PYTHONPATH=$ROOT $PY "$D/rcr_runtime_audit.py" $ROOT/data_new/$c >/dev/null 2>&1
rt=$(tail -1 "$D/rcr_runtime_audit.csv" | awk -F, '{printf "运行态 %s，L/R %.2f/%.2f，出口 le/li/re/ri 份额 %.2f/%.2f/%.2f/%.2f，压力 %.1f/%.1f/%.1f/%.1f kPa，质量守恒 %.4f",$2,$4,$5,$6,$7,$8,$9,$10/1e3,$11/1e3,$12/1e3,$13/1e3,$14}')
printf "%s\n" "$c" > $S/one.txt; ws=$(PYTHONPATH=$ROOT $PY "$D/wall_scan_1162.py" --cases-file $S/one.txt --out $S/one.csv | head -1 | sed 's/.*p_p99_1162=\([0-9]*\).*wss_p99_1162=\([0-9.]*\).*wss_max_1162=\([0-9.]*\).*/壁面 1162 步 p99 \1 Pa · WSS p99 \2 · max \3 Pa/')
vf=$($PY -c "
import numpy as np,glob
f=sorted(glob.glob('$ROOT/data_new/$c/Global_conditions/vf-in*rfile*.out'))[-1]; v=np.loadtxt(f,skiprows=3); print(f'vf-in {len(v)} 行 峰值 {v[:,1].max():.4e}')")
el=$(sacct -j $j -X -n -o Elapsed | head -1 | tr -d ' '); st=$(sacct -j $j -X -n -o State | head -1 | tr -d ' ')
line="| \`$c\`（第 3 批结果） | $j | $st $el | ascii $na / ascii_in $ni 帧；$rt；$ws；$vf |"
echo "$line" | tee -a "$D/rerun_jobs.md"; rm -rf $S

#!/bin/bash
#SBATCH --partition=CPU
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=16
#SBATCH --mem=160G
#SBATCH --time=36:00:00
#SBATCH --job-name=v52_ingest
#SBATCH --output=/public/newhome/cy/Digital_twin/GNN/docs/02-推进与变更/04-数据处理与CFD/数据回收_2026-09-20/ingest_wave2_%j.out
#SBATCH --error=/public/newhome/cy/Digital_twin/GNN/docs/02-推进与变更/04-数据处理与CFD/数据回收_2026-09-20/ingest_wave2_%j.err
# 入库链第 2 波（2026-09-22，81 个 RCR 协议审计 ok 的新单元）：与 pilot（pilot_ingest_20260922.slurm）完全相同的六步链，
# 写到独立快照根 anatomy_pointcloud_v5_2_20260922（不碰 v5.1）；三审计 + atlas 并入共享 prep 根（汇总备份 .bak_20260922b）。
set -euo pipefail
ROOT=/public/newhome/cy/Digital_twin/GNN
cd "$ROOT"
PY=/public/newhome/cy/.conda/envs/GNN/bin/python
PYV=/public/newhome/cy/.conda/envs/GNN_vmtk/bin/python
HERE="$ROOT/docs/02-推进与变更/04-数据处理与CFD/数据回收_2026-09-20"
CF="${CASES_FILE:-$HERE/ingest_wave2_cases.txt}"
CASES=$(grep -v '^#' "$CF" | tr '\n' ' ')
CL="${CL_ROOT:-$ROOT/outputs/centerline_v2_meshwall_20260922_new95}"
SNAP="$ROOT/data_wss_v5/anatomy_pointcloud_v5_2_20260922"
SPLIT="${SPLIT_PATH:-$ROOT/wss_pinn/configs/splits/split_WSS_PINN_AG_AAA_ILO_v5_2_new91_train227_test34_s1234.json}"
PREP="$ROOT/outputs/wss_pinn/volume_uvwp_bc_rcr_v4_anatomy_prep_20260903"
START_STEP=${START_STEP:-1}
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONPATH="$ROOT"
date -Is; hostname; echo "n_cases: $(echo $CASES | wc -w)  start_step=$START_STEP"
for f in audits/topology/summary.json audits/wall_wss/summary.json audits/solver_convergence/summary.json atlas/summary.json; do
  [ -f "$PREP/$f" ] && [ ! -f "$PREP/$f.bak_20260922b" ] && cp -p "$PREP/$f" "$PREP/$f.bak_20260922b"
done
if [ $START_STEP -le 1 ]; then echo "== 1 centerline (meshwall, vmtk)"; date -Is
  $PYV tools/centerline_v2_meshwall.py --cases $CASES --output "$CL" || echo "centerline exit code $? (a case without hard_pass makes the tool exit 1; per-case statuses are in run_result.json)"; fi
# 步骤 2–6 只对中心线 hard_pass（pass / pass_review）的单元继续；其余记入 *_cl_failed.txt 另行处理（2026-09-22 起）
CF2="$HERE/ingest_${WAVE_TAG:-wave2}_cl_pass.txt"
$PY - "$CF" "$CL/run_result.json" "$CF2" <<'PYEOF'
import json, sys
from pathlib import Path
cf, rr, out = sys.argv[1:]
want = [l.strip() for l in open(cf) if l.strip() and not l.startswith("#")]
st = {v["canonical_id"]: v for v in json.loads(Path(rr).read_text())["cases"]}
ok = [c for c in want if st.get(c, {}).get("hard_pass")]
bad = [(c, st.get(c, {}).get("status", "not_run"), str(st.get(c, {}).get("error", ""))[:160]) for c in want if c not in ok]
Path(out).write_text("\n".join(ok) + "\n")
Path(out.replace("_cl_pass", "_cl_failed")).write_text("".join(f"{c}\t{s}\t{e}\n" for c, s, e in bad))
print(f"centerline hard_pass {len(ok)} / {len(want)}; review-required: {[c for c in ok if st[c]['status']=='pass_review']}; failed: {bad}")
PYEOF
CF="$CF2"; CASES=$(tr '\n' ' ' < "$CF")
if [ $START_STEP -le 2 ]; then echo "== 2 feature atlas"; date -Is
  $PY -m wss_pinn.v4.build_feature_atlas --cases $CASES --cases-file "$CF" --source-root "$CL" --workers 8; fi
if [ $START_STEP -le 3 ]; then echo "== 3 topology audit"; date -Is
  $PY -m wss_pinn.v4.audit_anatomy_topology --cases-file "$CF" --centerline-root "$CL" --workers 8 || echo "audit exit code $? (historical failures in the cohort summary are expected; per-case gates are read below)"; fi
if [ $START_STEP -le 4 ]; then echo "== 4 wall audit"; date -Is
  $PY -m wss_pinn.v4.audit_wall_wss --cases-file "$CF" --workers 6 || echo "audit exit code $? (historical failures in the cohort summary are expected; per-case gates are read below)"; fi
if [ $START_STEP -le 5 ]; then echo "== 5 solver audit"; date -Is
  $PY -m wss_pinn.v4.audit_solver_convergence --cases-file "$CF" --workers 8 || echo "audit exit code $? (historical failures in the cohort summary are expected; per-case gates are read below)"; fi
if [ $START_STEP -le 6 ]; then echo "== 6 V5 build"; date -Is
  WSS_V5_SPLIT_PATH="$SPLIT" $PY -m wss_v5.build --cases $CASES --root "$SNAP" --workers 6 --run-name "new95_${WAVE_TAG:-wave2}_20260922"; fi
echo "== done"; date -Is

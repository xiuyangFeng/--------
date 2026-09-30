#!/bin/bash
# 2026-09-26: generate the 60 synthetic children (parallel morph writes), sanitise their fluent.slurm, submit one smoke job (read-case + mesh/check for all)
set -u
ROOT=/public/newhome/cy/Digital_twin/GNN; HERE="$ROOT/docs/02-推进与变更/04-数据处理与CFD/数据回收_2026-09-20/synth_20260926"; PY=/public/newhome/cy/.conda/envs/GNN/bin/python
export PATH=/public/slurm/bin:$PATH PYTHONPATH=$ROOT OMP_NUM_THREADS=2
cd $ROOT
$PY - <<'PYEOF'
import json, sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
ROOT=Path("/public/newhome/cy/Digital_twin/GNN"); sys.path.insert(0,str(ROOT)); sys.path.insert(0,str(ROOT/"tools"))
HERE=ROOT/"docs/02-推进与变更/04-数据处理与CFD/数据回收_2026-09-20/synth_20260926"
plan=json.load(open(HERE/"synth_plan_60.json"))
def one(j):
    import synth_morph_case as M
    try:
        r=M.morph(j["parent"], j["tag"], int(j["seed"]), 2, (0.15,0.40), (6.0,14.0), False, ROOT/"data_new"); r.pop("bumps",None); return r
    except Exception as exc:
        return {"parent": j["parent"], "tag": j["tag"], "valid": False, "error": f"{type(exc).__name__}: {str(exc)[:200]}"}
with ProcessPoolExecutor(8) as ex: res=list(ex.map(one, plan))
(HERE/"generate_report.json").write_text(json.dumps(res, indent=1))
ok=[r for r in res if r.get("valid")]; print("generated", len(ok), "/", len(res)); [print("  FAILED", r["parent"], r.get("error")) for r in res if not r.get("valid")]
(HERE/"synth_children.txt").write_text("\n".join(r["child"] for r in ok)+"\n")
PYEOF
mapfile -t KIDS < "$HERE/synth_children.txt"
for c in "${KIDS[@]}"; do s="$ROOT/data_new/$c/fluent.slurm"; sed -i -E "/^#SBATCH -w /d; s/^#SBATCH --ntasks-per-node=.*/#SBATCH --ntasks-per-node=64/; s/fluent 3ddp -g -t[0-9]+/fluent 3ddp -g -t64/" "$s"; done
LIST=$(for c in "${KIDS[@]}"; do printf "%s " "$ROOT/data_new/$c"; done)
cat > "$HERE/smoke_synth_20260926.slurm" <<SL
#!/bin/bash
#SBATCH --partition=CPU
#SBATCH --exclude=node05
#SBATCH --ntasks-per-node=8
#SBATCH --job-name=syn_smoke
#SBATCH --output=$HERE/smoke_synth_%j.out
#SBATCH --error=$HERE/smoke_synth_%j.err
#SBATCH --time=06:00:00
export I_MPI_SHM_LMT=shm
module purge
module load fluent/231
for d in $LIST; do
  cd "\$d" || exit 1
  echo "===== SMOKE \$d ====="
  { grep -v "^/solve\|^/exit" 2.jou; echo "/mesh/check"; echo "/exit y"; } > smoke_readcase.jou
  fluent 3ddp -g -t8 -mpi=intel -platform=intel -i smoke_readcase.jou -ssh 2>&1 | grep -v "^ *\$" | grep -i "Reading\|Error\|error\|warning: \|libudf\|Done\|domain extents\|coordinate\|Checking mesh\|Mesh Check\|quality\|skew" | head -60
  echo "===== fluent exit code \${PIPESTATUS[0]} ====="
  rm -f smoke_readcase.jou *.trn cleanup-fluent-*.sh
  mv -f libudf libudf_smoke_20260926 2>/dev/null
done
SL
J=$(sbatch --parsable "$HERE/smoke_synth_20260926.slurm"); echo "smoke job $J"; echo $J > "$HERE/smoke_job_id"

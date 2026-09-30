#!/bin/bash
#SBATCH --partition=CPU
#SBATCH --exclude=node05
#SBATCH --ntasks-per-node=8
#SBATCH --mem=64G
#SBATCH --time=03:00:00
#SBATCH --job-name=shell_regress
#SBATCH --output=/public/newhome/cy/Digital_twin/GNN/docs/02-推进与变更/04-数据处理与CFD/数据回收_2026-09-20/regress_shell_%j.out
#SBATCH --error=/public/newhome/cy/Digital_twin/GNN/docs/02-推进与变更/04-数据处理与CFD/数据回收_2026-09-20/regress_shell_%j.err
# 2026-09-22 壳层门改为只判"真在壁外"的壳层点后，用两例在库病例（QIN_SI_FU；ZHANG_WEI_XIAN = 在库壳层指标最高 1.87%）重建到临时根，与 v5.1 报告逐项比对。
set -euo pipefail
ROOT=/public/newhome/cy/Digital_twin/GNN; cd "$ROOT"; PY=/public/newhome/cy/.conda/envs/GNN/bin/python
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONPATH="$ROOT"
date -Is; hostname
$PY -m wss_v5.build --cases AG/slow/QIN_SI_FU AG/slow/ZHANG_WEI_XIAN --root "$ROOT/data_wss_v5/_regress_shell_20260922" --workers 2 --run-name regress_shell_20260922
$PY - <<'PYEOF'
import json
for cid in ("AG/slow/QIN_SI_FU","AG/slow/ZHANG_WEI_XIAN"):
    k=cid.replace("/","__")
    a=json.load(open(f"data_wss_v5/anatomy_pointcloud_v5_1_20260916/cases/{k}/report.json")); b=json.load(open(f"data_wss_v5/_regress_shell_20260922/cases/{k}/report.json"))
    ia=a["metrics"]["pointcloud"]["inside_test"]; ib=b["metrics"]["pointcloud"]["inside_test"]
    same={n: ia[n]==ib[n] for n in ia}; print(cid, "inside_test entries identical:", same)
    print("   new true-outside entry:", {kk:vv for kk,vv in ib["shell_1_3mm_outside_true_outside"].items() if kk in ("n","inside_fraction_hybrid","inside_fraction_winding")})
    ga=a["gates"]["checks"]; gb=b["gates"]["checks"]; print("   gates identical:", ga==gb, "pass", a["gates"]["pass"], b["gates"]["pass"])
    ma={k2:v for k2,v in a["metrics"].items() if k2!="pointcloud"}; mb={k2:v for k2,v in b["metrics"].items() if k2!="pointcloud"}
    print("   other metrics identical:", ma==mb, "| counts identical:", a.get("counts")==b.get("counts"), "| dataset digest identical:", a["bundle"].get("dataset_digest_sha256")==b["bundle"].get("dataset_digest_sha256"))
PYEOF
date -Is

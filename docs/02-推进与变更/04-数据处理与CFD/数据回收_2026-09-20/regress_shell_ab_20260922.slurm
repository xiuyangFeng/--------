#!/bin/bash
#SBATCH --partition=CPU
#SBATCH --exclude=node05
#SBATCH --ntasks-per-node=8
#SBATCH --mem=64G
#SBATCH --time=03:00:00
#SBATCH --job-name=shell_ab
#SBATCH --output=/public/newhome/cy/Digital_twin/GNN/docs/02-推进与变更/04-数据处理与CFD/数据回收_2026-09-20/regress_shell_ab_%j.out
#SBATCH --error=/public/newhome/cy/Digital_twin/GNN/docs/02-推进与变更/04-数据处理与CFD/数据回收_2026-09-20/regress_shell_ab_%j.err
# 同日代码 A/B：pilot 快照（09-22 上午、壳层门改前）里的 ZHAO_XIU_XUAN 与 CAO_DIAN_HE 用改后代码重建到临时根，报告逐项比对（只应多出新字段）。
set -euo pipefail
ROOT=/public/newhome/cy/Digital_twin/GNN; cd "$ROOT"; PY=/public/newhome/cy/.conda/envs/GNN/bin/python
SPLIT="$ROOT/wss_pinn/configs/splits/split_WSS_PINN_AG_AAA_ILO_v5_2_new94_train230_test34_s1234.json"
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONPATH="$ROOT"
date -Is; hostname
WSS_V5_SPLIT_PATH="$SPLIT" $PY -m wss_v5.build --cases AG/slow/ZHAO_XIU_XUAN AAA/unruputer/CAO_DIAN_HE --root "$ROOT/data_wss_v5/_regress_shell_ab_20260922" --workers 2 --run-name regress_shell_ab_20260922
$PY - <<'PYEOF'
import json
for cid in ("AG/slow/ZHAO_XIU_XUAN","AAA/unruputer/CAO_DIAN_HE"):
    k=cid.replace("/","__")
    a=json.load(open(f"data_wss_v5/anatomy_pointcloud_v5_2_pilot_20260922/cases/{k}/report.json")); b=json.load(open(f"data_wss_v5/_regress_shell_ab_20260922/cases/{k}/report.json"))
    ia=a["metrics"]["pointcloud"]["inside_test"]; ib=b["metrics"]["pointcloud"]["inside_test"]
    print(cid, "inside_test entries identical:", {n: ia[n]==ib[n] for n in ia}, "| new keys:", sorted(set(ib)-set(ia)))
    print("   true-outside:", {kk:vv for kk,vv in ib["shell_1_3mm_outside_true_outside"].items() if kk in ("n","inside_fraction_hybrid")}, "in-volume fraction:", ib.get("shell_in_cfd_volume_fraction"))
    print("   gates identical:", a["gates"]["checks"]==b["gates"]["checks"], "pass", a["gates"]["pass"], b["gates"]["pass"])
    ma={k2:v for k2,v in a["metrics"].items() if k2!="pointcloud"}; mb={k2:v for k2,v in b["metrics"].items() if k2!="pointcloud"}
    pa={k2:v for k2,v in a["metrics"]["pointcloud"].items() if k2!="inside_test"}; pb={k2:v for k2,v in b["metrics"]["pointcloud"].items() if k2!="inside_test"}
    print("   other metrics identical:", ma==mb, "| pointcloud metrics identical:", pa==pb, "| counts identical:", a.get("counts")==b.get("counts"), "| dataset digest identical:", a["bundle"].get("dataset_digest_sha256")==b["bundle"].get("dataset_digest_sha256"))
PYEOF
date -Is

"""Step 10 of the 2026-10-01 merge: make configs/wss_v52d_retrain_20261001 self-consistent on v5.2d.

    python 10_finish_configs.py

repoint_data_root rewrote every data path. Two fields of the three-head (M1cap) configs live outside the data root and still
pointed at v5.2c products:
  data.wss_stats_path          the merged three-channel statistics -> rebuilt here from the v5.2d statistics files with the
                               v5.2c builder (prepare_wss_v52c_retrain.multi_stats, paths switched to v5.2d)
  train.init_reference_config  the same-seed X5Dcap_asym2 full265 config -> its v5.2d copy
matrix.json described the cancelled v5.2c queue (run names, anchor, post-queue evaluations) and is replaced by a README;
queue / preflight scripts for a v5.2d retrain are not prepared here (nothing is trained).
"""
import json, sys
from pathlib import Path
G = Path("/public/newhome/cy/Digital_twin/GNN"); sys.path.insert(0, str(G))
from training_wss_min.tools import prepare_wss_v52c_retrain as P  # noqa: E402

NAME = "wss_v52d_retrain_20261001"
CD = G / "training_wss_min/configs" / NAME
V = G / "data_wss_v5/views_v5_2d_20261001"
P.NAME, P.EXP, P.V = NAME, G / "training_wss_min/experiments" / NAME, V
P.VIEW, P.CYC = V / "wss_min_view_v1", V / "wss_min_cycle_v1"
P.FULL_SPLIT = P.VIEW / "split_v52p4_full265_train265_test8.json"
stats = P.multi_stats()
d = json.loads(stats.read_text()); d["statistics_scope"] = d["statistics_scope"].replace("v5.2c", "v5.2d").replace("views_v5_2c", "views_v5_2d")
stats.write_text(json.dumps(d, indent=2) + "\n")
old = json.loads((G / "training_wss_min/experiments/wss_v52c_retrain_20260930/stats/multi_stats_full265_train265.json").read_text())
shift = {ch: {k: [old[ch][s][k], d[ch][s][k]] for s in ("log", "logit") if s in d[ch] for k in ("mean", "std")} for ch in ("wss", "tawss", "osi")}
rep = {"multi_stats": str(stats), "old_vs_new": shift, "configs": {}}
for f in sorted(CD.glob("M1cap_full265_s*.json")):
    c = json.loads(f.read_text()); seed = f.stem.rsplit("_s", 1)[1]
    c["data"]["wss_stats_path"] = str(stats)
    ref = CD / f"X5Dcap_asym2_full265_s{seed}.json"; assert ref.is_file()
    c["train"]["init_reference_config"] = str(ref)
    f.write_text(json.dumps(c, indent=2, ensure_ascii=False) + "\n"); rep["configs"][f.name] = {"wss_stats_path": str(stats), "init_reference_config": str(ref)}
left = []
for f in sorted(CD.glob("*.json")):
    if f.name == "matrix.json": continue
    t = f.read_text()
    if "v5_2c" in t or "v52c_retrain" in t or "v52c_labelfix" in t: left.append(f.name)
rep["configs_still_referencing_v52c"] = left
m = CD / "matrix.json"
if m.exists(): m.unlink()
(CD / "README.md").write_text(f"""# {NAME}

21 个配置，全部指向数据版本 v5.2d（`data_wss_v5/views_v5_2d_20261001`）。**未训练，也没有准备队列 / 预检脚本。**

- 来源：`configs/wss_v52c_retrain_20260930/` 的 21 个配置经 `training_wss_min.tools.repoint_data_root` 换根，配方不变；
  run 名前缀改为 `{NAME}/`，不会写进 v5.2c 的半截 run 目录。
- 18 个 X5Dcap_asym2（CV5 5 折 × 3 seed + full265 × 3 seed），3 个三头 M1cap full265。
- 三头配置的合并统计重建为 `experiments/{NAME}/stats/multi_stats_full265_train265.json`（v5.2d 统计），
  `init_reference_config` 指向本目录的同 seed X5Dcap_asym2 full265 配置。
- v5.2c 那次的 `matrix.json`（队列、预检锚点、附加评估）没有带过来。预检锚点「旧模型在 recover8 上复现存档值」
  按理仍成立（recover8 标签没变；输入用纯几何的 `*_murray_cap`，不受分流规则重拟合影响），没有重跑确认。
- 记录：`experiments/wss_v52d_merge_20261001/`（`load_check.json`、`configs_finish.json`），
  文档 `docs/02-推进与变更/04-数据处理与CFD/母库全量审计_2026-10-01.md` §11。
""")
(Path(__file__).resolve().parent / "configs_finish.json").write_text(json.dumps(rep, indent=1))
print(json.dumps({"multi_stats": str(stats), "shift": shift, "left": left, "n_configs": len(list(CD.glob('*.json')))}, indent=1)[:1500])

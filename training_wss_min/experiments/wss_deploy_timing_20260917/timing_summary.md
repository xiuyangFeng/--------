# 部署链路分段计时（2026-09-17，登录节点，Slurm 之外）

| 病例 | 设备 | STL 顶点 | 0.5 mm 点数 | 中心线+atlas (vessel_geom) | 重采样 | 特征 | 五模型推理 | 中心线后合计 | 端到端 | 该例 Pa R² | 预测/真值 p99 (Pa) |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| AAA/ruputer/KANG_YONG | NVIDIA GeForce RTX 4090 | 16448 | 88469 | 10.2 s | 16.0 s | 7.0 s | 6.4 s | 29.4 s | 40 s | 0.887 | 5.31 / 4.89 |
| AG/fast/FAN_JIAN_MING | NVIDIA GeForce RTX 4090 | 10627 | 58031 | 5.3 s | 8.5 s | 4.5 s | 5.6 s | 18.6 s | 24 s | 0.852 | 12.90 / 13.06 |
| ILO/LI_YOU_ZHI-0/before | NVIDIA GeForce RTX 4090 | 101654 | 89503 | 57.8 s | 15.6 s | 6.8 s | 5.5 s | 28.5 s | 86 s | 0.801 | 8.95 / 8.34 |
| AAA/ruputer/KANG_YONG | cpu | 16448 | 88469 | 10.2 s | 15.8 s | 7.0 s | 67.8 s | 90.7 s | 101 s | 0.888 | 5.30 / 4.89 |

脚本 `time_deploy_stages.py`（需 PYTHONPATH=仓库根，capfit 规则已覆盖为 v5.1 的 train136 文件）；`vessel_geom_run_*.json` 是中心线提取的 run.json；逐段原始数值在 `timing_summary.json`。

| 批次 | 病例 | Slurm 作业 | 提交时间 |
|---|---|---|---|
| 1 | `data_new/AAA/unruputer/SUN_SHU_MING` | 14289 | 2026-09-15 01:49 |
| 1 | `data_new/ILO/ZHANG_JIN_CHUN-1/before` | 14290 | 2026-09-15 01:49 |
| 1 | `data_new/AG/fast/WANG_CHUN_MING` | 14291 | 2026-09-15 01:49 |
| 2 | `data_new/ILO/SUN_XU_XIA-1/before` | 14303（node03） | 2026-09-15 02:03 |
| 2 | `data_new/ILO/ZHANG_YONG_SHENG-0/before` | 14304（node03） | 2026-09-15 02:03 |
| 2 | `data_new/AAA/unruputer/GUO_BAO_CHUN` | 14305（node03） | 2026-09-15 02:03 |
| 3 | `data_new/AAA/ruputer/LIU_YONG_LAN` | 14306（node06） | 2026-09-15 02:06 |
| 3 | `data_new/AAA/ruputer/MENG_GUANG_QIN` | 14307（node03） | 2026-09-15 02:06 |
| 3 | `data_new/AAA/ruputer/WU_GUANG_CUN` | 14308（node06） | 2026-09-15 02:06 |
| 3 | `data_new/AAA/ruputer/ZHOU_KE_XUN` | 14309（node03） | 2026-09-15 02:06 |
| 3 | `data_new/AAA/ruputer/ZOU_LI_SHUN` | 14310（node06） | 2026-09-15 02:06 |
| 3 | `data_new/AAA/ruputer/ZUO_DAO_SHENG` | 14311（node03） | 2026-09-15 02:06 |
| 3 | `data_new/AAA/unruputer/LIU_XING_GUO` | 14312（node06） | 2026-09-15 02:06 |
| 3 | `data_new/AAA/unruputer/QIAO_XIU_YUN` | 14313（node03） | 2026-09-15 02:06 |
| 3 | `data_new/AG/fast/LIU_JUN_FENG` | 14314（node06） | 2026-09-15 02:06 |
| 3 | `data_new/AG/fast/LIU_LI_QUN` | 14315（node03） | 2026-09-15 02:06 |
| 3 | `data_new/AG/fast/LIU_YI_BING` | 14316（node06） | 2026-09-15 02:06 |
| 3 | `data_new/AG/fast/ZHANG_QING_WANG` | 14317（node03） | 2026-09-15 02:06 |
| 3 | `data_new/AG/slow/CHENG_GUANG_SEN` | 14318（node06） | 2026-09-15 02:06 |
| 3 | `data_new/ILO/HOU_SHI_GUO-0/before` | 14319（node03） | 2026-09-15 02:06 |
| 3 | `data_new/ILO/LIU_BAO_JUN-0/before` | 14320（node06） | 2026-09-15 02:06 |
| 3 | `data_new/ILO/LIU_LIAN_YOU-0/before` | 14321（node03） | 2026-09-15 02:06 |
| 3 | `data_new/ILO/YANG_WANG_QI-1/before` | 14322（node06） | 2026-09-15 02:06 |
| 3 | `data_new/AG/slow/LIU_ZONG_YANG` | 14323（node03） | 2026-09-15 02:06 |
| 3 | `data_new/AAA/ruputer/WANG_KUI_WU` | 14324（node06） | 2026-09-15 02:06 |
| 3 | `data_new/AAA/unruputer/YANG_BEN_RUI` | 14325（node03） | 2026-09-15 02:06 |
| 刷新 | v5.1 母库 + 视图 | 14365 | 2026-09-15 12:45 |
| 链 | 噪声 sidecar + 配置生成 | 14366（afterok 14365） | 2026-09-15 12:46 |
| 链 | v5.1 波 1 GPU 预检 | 14367（afterok 14366） | |
| 链 | v5.1 波 1 训练队列（14 臂） | 14368（afterok 14367） | |
| 链2 | 刷新重提（19 例 sha 审计更新后重建 + 视图） | 14369 | 2026-09-15 15:14 |
| 链2 | 噪声 sidecar + 配置 | 14370（afterok 14369） | |
| 链2 | GPU 预检 | 14371（afterok 14370） | |
| 链2 | 训练队列 | 14372（afterok 14371） | |
| 链3 | 刷新重提（驱动变量展开 bug 修复，19 例干净重建 + 视图） | 14373 | 2026-09-15 15:34 |
| 链3 | 噪声 sidecar + 配置 / GPU 预检 / 训练队列 | 14374 / 14375 / 14376（afterok 链） | |
| 链4 | 视图重生成（volume_view split 名修复；build 已 170/170 过）/ 噪声+配置 / 预检 / 队列 | 14377 / 14378 / 14379 / 14380 | 2026-09-15 15:50 |
| 链5 | GPU 预检重提（collate 丢 radius 键已修）/ 训练队列 | 14381 / 14382 | 2026-09-15 16:03 |
| 链6 | 波 2a 三折 X5D_v51 预检/队列 → 级联 sidecar + 波 2b 配置 → 波 2b（T3n 三 seed + 三折）预检/队列 | 14383 / 14384 / 14385 / 14386 / 14387（afterok 14382 起） | 2026-09-15 16:12 |

## 旧导出清理（2026-09-16）

用户拍板后删除 26 例的 `ascii_old_20260915` + `ascii_in_old_20260915`（52 目录，248.2 GB）。前置核查：新导出 ascii/ascii_in 各 81 帧、v5.1 母库该例 gate 通过、旧快照 v5.0 的 case.h5 存在（旧标签备份）。清单 `delete_old_exports_plan.json`，逐目录记录 `delete_old_exports_log.json`。保留 `libudf_old_20260915`、`Global_conditions_old_20260915`、`*.orig_20260915`。

| 链7 | 波 1r（11 臂续跑）预检/队列 → 波 2a → 级联 → 波 2b | 14402 / 14403 / 14404 / 14405 / 14407 / 14408 / 14409 | 2026-09-15 20:48 |
| 链8 | 冻结副本 GNN_v51_frozen 提交：波 1r 预检/队列（2 卡×2 槽）→ 波 2a → 级联 → 波 2b | 14415 / 14416 / 14417 / 14418 / 14419 / 14420 / 14421 | 2026-09-15 20:54 |
| 链9 | 用户放开显存：队列改 4 卡×3 槽（预检 14415 保留）→ 波 2a → 级联 → 波 2b | 14422 / 14423 / 14424 / 14425 / 14426 / 14427 | 2026-09-15 20:58 |
| 链10 | 清掉残留 .queue.lock 后重提（4 卡×3 槽） | 14428 / 14429 / 14430 / 14431 / 14432 / 14433 | 2026-09-15 20:58 |
| 链10+ | 波 1r 后的部署鲁棒性复评（抽稀 × X5Ddual / 几何噪声 × X5Dnoise，各配 X5D_v51 三 seed） | 14961（afterok 14428） | 2026-09-16 01:21 |
| 链10+ | STL 重采样部署模拟：X5Ddual vs X5D_v51 三 seed（0/0.5/0.8/1.2 mm） | 14963（afterok 14961） | 2026-09-16 04:54 |

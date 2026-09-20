# 启发式汇报 v3：部署形态与 v5.1 新数据（2026-09-17）

本包补的是 [提纲 v2](../启发式汇报提纲_v2_2026-09-13.md) 之后（跟踪文档 §22–§29）的工作，**v2 与旧图件一字未动**。图 01–16 只读已保存的实验产物与预测缓存生成：**不训练、不做模型推理**；图 17 读的是 2026-09-17 跑完的剪枝阶梯（作业 14991）的产物，本包本身同样不发起训练。讲解入口是 [提纲 v3](启发式汇报提纲_v3_2026-09-17.md)（P19–P36，含可直接复制的指标表）。

## 图件

| 图 | 打开 | 结论 |
| --- | --- | --- |
| P19 部署决策链 | [01](figures/01_deployment_decision.png) | 训练密度不是单值（AG 1.17 / AAA 0.47 / ILO 0.46 mm）；损失定位在 K16 query patch；合同 = 重采样 0.5 mm + 平滑 1 mm + 五 seed 集成 |
| P20 密度扫描 | [02](figures/02_density_curve.png) | 0.5 mm 时 X5 只掉 0.012；1.2 mm 时密度增广把 −0.147 压到 −0.047；代价在 patch 不在 support |
| P21 几何保真敏感性 | [03](figures/03_geometry_fidelity.png) | 平滑几乎免费并能救回七成粗糙损失；噪声 0.4 mm 掉 0.123；切口 ±5 mm 可容忍；TTA 不值得 |
| P22 RCR 录入核查 | [04](figures/04_rcr_audit.png) | 688 个出口里 39 个录错、涉及 23 例（test34 占 4 例）；26 例重算 + 2 例去重 → 170 例 |
| P23 v5.0 → v5.1 | [05](figures/05_data_v50_to_v51.png) | 0.7465 →（只换标签）0.7652 →（重训）0.7749：三分之二的涨幅来自标签修正 |
| P24 新底座 X5D_v51 | [06](figures/06_x5d_v51_baseline.png) | test34 五 seed 集成 0.7749；cv3 合并折外 0.7100；ILO 折外 0.624 最弱 |
| P25 三项后续臂 | [07](figures/07_three_arms_paired.png) | 只有协议版 capfit 归一化 3/3 正（集成 +0.0035）；双尺度 patch、噪声增广不进底座 |
| P26 抽稀复评 | [08](figures/08_thinning_reeval.png) | 双尺度 patch 在 10% 抽稀把冻结特征损失从 −0.146 缩到 −0.056（设计目标达成） |
| P27 STL + 噪声复评 | [09](figures/09_stl_and_noise_reeval.png) | 放回"重采样 0.5 mm + 平滑 1 mm"的部署口径后，两个鲁棒性臂的净收益归零 |
| P28 沿程 32 通道 | [10](figures/10_longitudinal_paired.png) | 尾部一致改善（IoU 3/3 升）；整体 R² 三 seed 均值 +0.0049、sd 0.0143，落在噪声带内 |
| P29 为什么没涨 | [11](figures/11_longitudinal_why.png) | 通道半数互为重复（有效维数 7.16/16）；最强通道 98.4% 是旧信息；对底座残差解释力只有 R² 0.0059 |
| P30 残差结构与尾部口径 | [12](figures/12_residual_structure.png) | 区间内型态占 ≈70%；轨迹 R² 0.93+ 而型态 R² 0.52–0.61；尾部 0.387 / 0.5265 / 0.4022 是三种口径 |
| P31 ILO 与门控混合 | [13](figures/13_ilo_and_gate.png) | 两例占 ILO 平方误差 52.16%；ref 8 门控混合不重训到 0.7844（大头是集成多样性） |
| P33 残差尺度阶梯 | [14](figures/14_residual_scale_ladder.png) | 三代底座 × 两套数据 × test/折外：比例始终 69–70%，幅值始终向均值收缩 |
| P32 杠杆盘点 | [15](figures/15_levers_used_vs_unused.png) | Murray 先验 +0.101 之后每轮加输入都在噪声带内；这类标量的折外天花板 R² 0.00473 |
| P35 后续路线图 | [16](figures/16_future_roadmap.png) | 五个候选（A 轴向×周向分区 token 主推）+ 不投入清单 + 先做只读折外诊断的门槛 |
| P36 通道剪枝阶梯 | [17](figures/17_prune_ladder.png) | 32 列砍到 2 列归一化增益不变（+0.0060 对 +0.0065），覆盖率 0.651 → 0.788；保留集 = 截面圆度 2 列 |

## 口径

- 数据 v5.1，170 例 `train136 / test34`；`ckpt_best` 由 train_loss/top3 选出，不是验证集选模。
- 主指标 Pa R²_cb（病例等权共享均值）；集成一律 Pa 空间均值；方法比较用 test34，泛化确认用 cv3 三折折外，两者不可并排相减。
- 部署类实验的 Δ 一律对「同一 checkpoint 的全云预测限制到同一批点」的同点参照，不用不同点集的绝对 R² 当增益。
- 噪声尺度：同配置抖动 ±0.034（v5.0 波 1 单 seed 口径）；三 seed sd 物理 ±0.0164 / 归一化 ±0.0040；n=3 一律不做显著性。
- 尾部有三种口径：离线 helper 先 pool 再取 P90（0.387）、正式 `evaluate` 每例 P90 再 pool（0.5265）、同口径折外（0.4022）。三者不可交叉比较。
- 图 14/15 跨数据版本取数（v5.0 与 v5.1 并列），只用于看「比例稳不稳」与「效应量的量级」，不作精度排名。
- 图 16 是计划，不是结果：里面没有任何一个候选臂被训练过。
- 图 17 是唯一一张来自新训练的图（作业 14991，四臂单 seed 1234，2026-09-17 跑完）；其余 16 张全部只读已保存产物。L1 若要进底座仍须补 seed 并过 cv3 折外。

## 产物

- `figures/*.png`：17 张，16×9 in @ dpi 300（4800×2700）。
- `source_data/*.json`：每张图实际用到的取值 + 来源文件的仓库相对路径。
- `qa/*.alignment.json`、`qa/*.metadata.json`：面板对齐审计与来源 SHA256（由 `plot_style.save()` 自动生成）。
- `plot_style.py`：与 [09-15 Nature 风格包](../启发式Nature风格优化_20260915/README.md) 逐字节相同的绘图合同。
- `datasrc.py`：只读取数的加载层，所有路径集中在此。

## 重绘

```bash
cd docs/03-汇报材料/启发式实验汇报_2026-09-13/启发式v3_部署与新数据_20260917
P=/public/newhome/cy/.conda/envs/GNN/bin/python
$P build_deploy.py && $P build_rcr.py && $P build_v51.py && $P build_longitudinal.py \
  && $P build_residual.py && $P build_future.py && $P build_prune.py
$P verify_delivery.py    # 图件齐全 / 尺寸 / 对齐审计 / 提纲关键数字回溯，共 145 项
```

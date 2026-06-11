# V3P 路径 G · G4 2D 壁面换轨预研方案（TODO-60）

> **索引**：[路径 G 主规划](../00-当前主线/路径G_下一代架构与精度突破方案_2026-06-05.md) · [S0 交接](V3P_路径G_S0执行计划与交接.md) · [后续优化待办](V3_后续优化待办.md) · [实验跟踪](V3_实验执行跟踪日志.md)
> **创建**：2026-06-11 · **性质**：规划层 / 换轨预研 · **不含已落地 2D 训练代码**
> **前置结论**：G1/G2/G3 GPU 主线 **全部 No-Go 封口**（2026-06-11）；G0-b **Go**（F0 `pod_2d` 20 模态 n_sectors=4 test R² **0.672** > 0.6）

---

## 1. 立项依据

| 证据 | 结论 |
| --- | --- |
| G1-a0 / G2-b / G3 Phase 3 全线 No-Go | 在 **FieldPointNeXt + global xyz + direct head** 表示空间内，单变量已无 ≥+0.03 稳健增益 |
| F0 `oracle_pod_2d`（`v3_f0_oracle_v2.json`） | (Abscissa × θ) 2D 场 **20 模态 R²=0.672**（n_sectors=4）> G0-b 门槛 **0.6** → **G4 立项放行** |
| G3 No-Go（5474/5475 Δ +0.012/−0.002） | 纯几何 SSL 不足以 warm-start 现 backbone → **换轨**而非继续堆预训练 |
| 路径 G §7 / R4 | 边界算子 `WSS=μ·∂u_t/∂n|_wall` 在 **规则 2D 网格**上比非结构点云更易学 |
| MultiViewUNet 2025 | AAA 几何展开 + 2D U-Net 在 ~230 合成例上可行 → **G4 有文献先例** |

**命题（可证伪）**：将壁面 WSS（及可选压力/几何条件）表示为 **(Abscissa s, 周向 θ)** 上的 2D 场，用 **2D U-Net / Geo-FNO** 学习映射，test `wss_r2_wss` 相对 post5463 基线 band（**0.425±0.012**）**Δ≥+0.03**，且 `wss_x/y` 或病例级 Pa 至少一项不退化。

---

## 2. 目标与 Go / No-Go

### 2.1 主指标（微调后 · post-denylist split · best_wss 或 2D 等价选优）

| 层级 | Go | No-Go |
| --- | --- | --- |
| 点级标量 | test `wss_r2_wss` **Δ≥+0.03** vs post5463 band 均值 **0.425** | ±0.005 噪声内 |
| 分量 | `wss_x` 或 `wss_y` **>0.05**（历史从未达到，破线即结构性信号） | 仍 ≈0 |
| 病例级 | Pa p95 Spearman **≥ 5439−0.05（0.347）** | 点级升但 Pa 明显降 |
| 2D→3D | 反映射后 3D 壁面点 R² 与 2D 网格 R² 差距 **<0.02** | 反映射损失大 → 展开方案不可行 |

### 2.2 预研阶段自身（不出 Main 表）

- Phase 0 可行性 JSON **PASS**（Abscissa 无全例退化、平均 occupancy ≥0.20）
- Phase 1 单 case 过拟合：2D U-Net 在 1 train case 上 WSS magnitude R²→**>0.95**
- **禁止** Phase 0 未 PASS 时直接全量 GPU 训练

---

## 3. 方案选型（三轨 · 按风险排序）

| 优先级 | ID | 方案 | 本期 | 说明 |
| ---: | --- | --- | --- | --- |
| **P0** | **G4-a** | **全局 (s, θ) 展开 + 2D U-Net** | **Phase 0–1 首版** | 复用 F0 `oracle_pod_2d` 同一 θ 构造；实现快、与 oracle 口径一致 |
| P1 | **G4-b** | **分支级展开**（按 `branch_id` / 分叉切段） | Phase 1 后 | 规避全局分叉拓扑奇异；预处理更重 |
| P2 | **G4-c** | **Geodesic Patch CNN**（局部测地贴片 + GNN 拼接） | 预研设计 | G0-b 不过或 G4-a 在分叉 case 崩时升级；见路径 G §7 |

**本期范围**：**G4-a Phase 0–2**（可行性审计 → 单 case 过拟合 → 1 seed Probe）；G4-b/c 仅写 checklist，不并行开训。

---

## 4. G4-a · 全局 2D 展开设计

### 4.1 坐标系（与 F0 oracle 对齐）

```text
输入（每 graph · 壁面点）:
  · s ← Abscissa（归一化到 [0,1] per graph）
  · θ ← arctan2(a2, a1)，a1/a2 为壁面点在「切向均值 + 垂直平面」上的投影
  · 与 run_v3_f0_decision._collect_profiles_2d 完全一致（跨病例 θ 仅近似对齐）

输出网格:
  · grid_s = 64（可调）× n_sectors ∈ {4, 8}
  · 每 cell：WSS magnitude / 可选 wss_x,y,z / 几何条件（NormRadius, Curvature）
  · 空 cell：行均值填充（与 oracle 一致；训练时可改 learnable mask）
```

### 4.2 模型与训练

| 项 | 首版 |
| --- | --- |
| 架构 | **2D U-Net**（in_channels=几何+BC 条件，out_channels=1 或 3 矢量） |
| 条件 | BC_Inlet 等全局标量 → FiLM / channel concat |
| 损失 | WSS magnitude Huber + 可选 x/y/z 分量（AsymW 权重） |
| 反映射 | 双线性插值 (s,θ)→最近壁面点；report 3D R² vs 2D R² |
| exp_id | `V3P-G-60-2DUnwrap` · **新表** · 禁与 4957/5439/G3 混表 |

### 4.3 刻意不做（首版）

- 跨病例谱图卷积（H4 仅单病例 oracle；谱漂移不可泛化）
- 直接用 F0 POD 系数当预测（无泛化，仅 oracle 上限）
- 在展开阶段使用 test WSS 标签做网格归一化

---

## 5. Split-safe 与泄漏检查（必过）

| # | 检查项 | 通过标准 |
| ---: | --- | --- |
| 1 | 网格归一化 | Abscissa min/max、θ 参考系 **per graph**，不跨 case 聚合 test 统计 |
| 2 | 2D 增广 | 仅 train；val/test 只做 transform |
| 3 | BC 条件 | 来自 `global_cond`，不用 WSS/压力 GT |
| 4 | 选优 | val 2D/3D R²；test 只报一次 |
| 5 | 对照 | 同 seed · 同 split · 对照 **post5463 band 0.425±0.012** |

产物：`outputs/field/f0_decision/v3p_g4_unwrap_feasibility_<date>.json`（Phase 0）

---

## 6. 技术架构（草案）

```text
pipeline/
└── wall_unwrap/
    ├── build_2d_grid.py          # Phase 1 · graph.pt → (H,W,C) tensor + manifest
    └── remap_to_3d.py            # 预测 2D → 壁面点 WSS

training/
├── scripts/
│   ├── run_v3_g4_unwrap_feasibility.py   # ✅ Phase 0
│   └── train_field_2d_unwrap.py          # Phase 2 新增
├── models/
│   └── unet2d_wss.py                     # Phase 1 新增
└── configs/field/generated/v3_pointcloud/
    └── V3P-G-60-2DUnwrap_seed1.json
```

---

## 7. 执行阶梯（禁止跳步）

| 阶段 | 动作 | 退出条件 |
| --- | --- | --- |
| **Phase 0** | `run_v3_g4_unwrap_feasibility.py` 全例审计 | JSON `phase0_gate=true`；否则改 G4-b/c |
| **Phase 1** | 1 train case 过拟合（2D U-Net · magnitude） | R²>0.95；不过 → 改 grid/θ 或转 Patch |
| **Phase 2** | 全量 1 seed Probe vs post5463 band | Go → seed2/3；No-Go → G4-b 或 G5 兜底 |
| **Phase 3（条件）** | G4-b 分支级展开 ablation | 仅 Phase 2 在分叉 case 系统性失败时 |

---

## 8. 对照与填表口径

| 对照 | run / 指标 | 用途 |
| --- | --- | --- |
| **主基线** | post5463 band **5466/5468/5478 · 0.425±0.012** | 同 split 公平 Δ |
| Oracle 上限 | F0 `pod_2d` 20 模态 **0.672** | 表示天花板参考（非训练对照） |
| 病例叙事 | 5311 Pa p95 **0.559** | G5 第二层门禁 |
| 旧母版 | 4957 **0.399** | **禁止**与 post5463 混表 |

---

## 9. 风险与退出

1. **θ 跨病例不对齐**：F0 已注明单边保守；若 Probe val↑ test↓ → 转 **G4-b 分支级** 或 **G4-c Patch**
2. **分叉处度量扭曲**：全局展开在 bifurcation 邻域 occupancy 低 → Phase 0 `abscissa_gaps` / `sparse_grid` flag 触发 G4-b
3. **2D→3D 反映射损失**：插值平滑抹平高 WSS 峰 → 需 top-k 区域 / 病例级 eval
4. **工程换轨成本**：预处理链 + 新 dataloader；**禁止**与 FieldPointNeXt 训练混在同一 exp_id

---

## 10. 近期交付物

| 交付 | 路径 / 状态 |
| --- | --- |
| G4 预研方案 | ✅ 本文档 |
| Phase 0 可行性审计 | ✅ `v3p_g4_unwrap_feasibility_20260611.json` · **PASS**（occupancy **0.72** · 0 例 Abscissa 退化） |
| F0 pod_2d oracle | ✅ `v3_f0_oracle_v2.json` · 20 模态 **0.672** |
| G4-a 网格构建 | ✅ `pipeline/wall_unwrap/` |
| G4-a 单 case 过拟合 | ✅ `g4_2d_unwrap_overfit1c_s4_seed1_20260611_151301` · grid R² **0.954** |
| G4-a 全量 Probe | 🏃 Job **5507** · seed1 · SLURM `submit_g4_phase2_probe.sh` |

---

## 变更记录

| 日期 | 内容 |
| --- | --- |
| 2026-06-11 | Phase 0 审计 PASS（81 例 occupancy 0.72）；初版方案 |

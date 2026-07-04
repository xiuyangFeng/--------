# V3P 平台期失败病例档案（CHEN / fast 域）

> 日期：2026-06-30  
> 口径：V3P · `split_AG_v1` · I6-diag / G4-c Phase1 同源  
> 用途：平台期典型案例说明；**不**宣称工程可用

---

## 1. 为何单独建档

G4-c Phase1 双 case 门禁暴露：**patch 内 grid R² 高（~0.93）≠ merged 3D 可用**。CHEN（fast）与 GUO（slow）形成对照，是整体精度平台（尤其 fast 域 WSS 弱）的缩影。

---

## 2. 关键指标对照

| 病例 | 域 | I6-diag R²_wss | G4-c merged best | Δ vs I6 | G4-c grid final | 判读 |
| --- | --- | ---: | ---: | ---: | ---: | --- |
| **CHEN_SHI_MING** | fast | **0.363** | **0.406** | +0.043 | 0.932 | ❌ No-Go（差 +0.05 线 **0.007**） |
| **GUO_XI_JIANG** | slow | **0.570** | **0.633** | +0.062 | 0.959 | ✅ 单 case Go |

数据源：`f0_decision/v3p_g4c_phase1_overfit_summary_20260630.json`

---

## 3. CHEN（fast）失败机制

1. **merged 3D 不稳定**：多 patch 拼接方差大；训练后期 merged R² 可跌至负值（note: final merged **0.223**）。
2. **低 occupancy / fast 相位**：壁面采样与 patch coverage 对 fast 病例更敏感。
3. **与 I6-diag 一致**：test 点级 R²_wss **0.363** 为三例汇报包中 fast 域最弱（工程方案 §2.2）。

**不能得出的结论**：2D/patch 路线完全错误（GUO 证明 patch 内可拟合）；**能得出的结论**：当前 patch sampling + overlap merging **不足以**支撑 fast 病例工程化。

---

## 4. 其他 fast 域观察（I6-diag 汇报包）

| 病例 | R²_p | R²_wss | 备注 |
| --- | ---: | ---: | --- |
| CHEN（fast） | 0.951 | **0.363** | 压力好 · WSS 弱 |
| ZHANG（slow） | 0.849 | **0.352** | 跨病例方差大 |
| GUO（slow） | 0.933 | **0.570** | 相对最好 |

**模式**：fast 域 WSS 系统性弱于 slow；hotspot 可视化上 pred 幅值低于 CFD（工程方案 §2.3）。

---

## 5. 后续 oracle / 数据路线指向

- **Idea F**：按 fast/slow、hotspot residual 聚类 → TODO-1 主动选数。
- **Idea G**：检查 V3D 是否存在与 CHEN/fast 失败簇 **near-domain** 的补覆盖病例（V3D-lite，非 full 混训）。
- **Idea E**：对 CHEN 单独报谱重构上限，区分「缝合问题」vs「高频不可重构」。

---

## 6. 可视化索引

- postview：`postview/v3p_i6diag_t016_report/`（含 CHEN 帧）
- G4-c run：`outputs/field/g4c_patch_overfit1c_fast__CHEN_SHI_MING_p12_r3.0_k20_seed1_20260630_111933/`

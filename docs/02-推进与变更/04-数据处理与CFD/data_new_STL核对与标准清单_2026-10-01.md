# data_new STL 核对与标准清单（2026-10-01）

**起因**：出口命名校准有 20 个母库单元没走完部署 A 段，部署工具提示「可能含断裂分支」。用户要求先确认原始数据，再整理。

## 结论

1. **没有断裂的血管，CFD 都是在本例自己的几何上算的。** 有 STL 的单元，CFD 壁面到本例 STL 表面的中位距离都 ≤ 0.6 mm（10-01 母库审计，`extra_checks_v52d.json` / `stl_registered.json`）。
2. **训练不读这些 STL。** 母库的网格、标签都取自 Fluent 网格（`wss_v5/build_case.py`：`load_mesh`、`wall_triangles`）。中心线也不来自 `data_new` 的 STL：
   - 105 个真实单元和全部 59 个合成单元，在 Fluent 网格的血管壁面上提取（`surface_source = fluent_case_anatomy_wall`）。
   - 最早的 168 个单元，在旧数据包登记的权威 STL 上提取。其中 LIU_BAO_JUN-0/before 用的是 `LIU_BAO_JUN_1.stl` 按旧单位系数缩小到 0.9522 倍的版本。
   - 进入 `case.h5` 的 atlas 已换算到真实毫米：全库 332 单元的中心线到壁面距离与半径之比为 0.91–1.00，端点到 CFD 端口中位 1.0 mm、p95 3.8 mm。有 6 例某个端点偏 5–9 mm（LI_BING_JIANG 9.3 mm 最大），与 STL 无关，原因未查。
3. **出问题的是 STL 文件本身**，只影响从表面开始的工具：

| 分类 | 例数 | 说明 |
|---|---|---|
| 装配体 | 11 | 血管 + 直管延长段（占主体面积 28–38%，不在计算域内）+ 切割平板（厚度/宽度 0.0000–0.0001） |
| 坐标系不同 | 7 | 同一几何，需要刚性配准；审计配准后中位 0.21–0.60 mm |
| 缺 STL | 3 | FU_GUO_JUN（09-30 cfd_auto 在已有网格上重算）、SUN_CHUN_PU-0 前后 |
| 封口 | 1 | LIN_SHU_TIAN，另有 22 个碎屑 |
| 端口封住 | 1 | NIE_QUAN_ZHONG，4 个开口 |
| 单位错 | 1 | SUN_XU_XIA-1/after，约 1000 倍 |
| 取错文件 | 1 例 2 个单元 | LIU_BAO_JUN-0：`LIU_BAO_JUN.stl` 是装配体，`LIU_BAO_JUN_1.stl` 才是 CFD 几何 |

## 已做

- **归档**（用户批准）：`data_new/ILO/LIU_BAO_JUN-0/{before,after}/LIU_BAO_JUN.stl` → `data_new/_archive/stl_assembly_20261001/…`。挪动前后哈希一致；两个病例文件夹各留一份 `ARCHIVED_STL_20261001.json`。之后 `find_stl` 取到的是 `LIU_BAO_JUN_1.stl`。
- **标准清单**：`data_wss_v5/stl_canonical_v5_2d_20261001/`（不进 git），生成脚本为 `tools/stl_manifest.py`。
  - 249 例用 `data_new` 的 STL。
  - 24 例用 CFD 壁面导出：去掉重复三角面；有 5 例另外去掉了 1–4 个碎片三角形，它们在三角形共边处形成非流形。
  - 校验：273 个标准输入全部通过部署输入检查；24 个导出全部跑完 A 段，出口命名 24/24 正确。
  - 查询：`tools.stl_manifest.canonical_stl(uid)`。
- **重复单元的标签核查**：同一病人的三组单元（LIN_CHUN_YANG-1 ↔ LIU_CHUN_YANG-1、GUO_AI_JUN 的 AAA ↔ ILO、SHEN_CHUN_WANG 的 AAA ↔ ILO）是各自独立的求解，不是复制的标签：
  - 映射后数值完全相同的比例为 0%；
  - 峰值 WSS 一致性 R² 在 −0.72 到 0.60 之间；
  - 出口分流和出口压力都不同，只有入口波形相同（全库统一协议）。
  - 用户结论：标签不重复就没事。另记：CV5 里只有 LIN/LIU 被分在不同折，IND test91 有 6 个单元在训练集里有同一病人。

## 待办

- v5.2d 重训结束后（预计 10-02 14:00），把 `training_wss_min/tools/deployment_stl_simulation.find_stl` 改为读清单（`canonical_stl`）。队列运行期间改 `training_wss_min/*.py` 会中止队列，所以暂不改。

核对过程、图和脚本：`outputs/stl_integrity_check_20261001/README.md`。

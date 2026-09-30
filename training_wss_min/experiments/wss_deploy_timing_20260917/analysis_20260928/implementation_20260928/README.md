# wss_deploy v0.15.11 算子与缓存提速：实施与验收（2026-09-28）

依据同目录上一级的[审计报告](../README.md)实施。09-28 下午 codex 会话开了 P0-2 / P0-3 / P0-4 三个模块后额度用尽（三个模块都没有测试、`prepared_inference` 未接入推理）；本会话接着完成、验证，并追加四项算子优化。精度两档沿用 09-24 裁定：几何 / 采样 / 特征 / 导出链 **Tier A 逐位相同**，只改浮点归约顺序的 **Tier B 不超过原实现运行间抖动**。模型、采样密度、seed 数、FP32 全部不变。

## 1. 结论

同一组合请求（M1 三头 + PF6/VF6 体场，上传后立即确认出口，GPU 2 空闲，机器 load 13–22）从上传到两份报告都完成：

| 病例 | 旧版 v0.15.10（git HEAD） | 新版 v0.15.11 | 体场伴随任务 B 段 |
|---|---:|---:|---:|
| SHI_YUN_XI（19.2 万面，壁面 72,815 点） | 112.5 / 126.4 s | **80.1 / 82.2 s** | 47.9 / 55.8 → 17.1 / 17.5 s |
| FAN_JIAN_MING（2.1 万面，壁面 57,952 点） | 32.7 / 33.7 s | **25.8 / 24.1 s** | 17.0 / 16.0 → 9.6 / 9.7 s |

另测「确认前等 9 s」（接近真人确认耗时）：SHI 旧 112.9 s → 新 85.3 s（旧版这时父任务预计算要 18 s 还没跑完，伴随任务照样什么都没复用）；FAN 旧 28.8 s → 新 28.3 s（旧版这时预计算已完成、伴随任务能复用，差别只剩算子项）。数字是单次墙钟，机器负载会造成 ±几秒波动；阶段拆分见 `e2e/*.json`。

剩余大头：A 段中心线（VMTK，SHI 约 40 s；09-24 裁定不动）、确认后父任务等待预计算（SHI 16–19 s）、体场特征 9 s、流线 5.7 s。

## 2. 七项改动与验收

| # | 改动 | 文件 | 精度档与证据 | 收益 |
|---|---|---|---|---|
| 1 | **组合任务缓存接力（P0-2）**：伴随 / 重跑任务 B 段开始时，若源任务预计算仍在跑就等它结束（不持管理器锁、不取消源任务、上限 300 s），再把源任务已原子提交的几何缓存条目复制为私有 0600 副本（`O_NOFOLLOW`、跳过硬链接 / 临时文件、不覆盖、逐块复核归属与取消） | `cache_handoff.py`（新）、`jobs.py` | Tier A：复制的是同键条目，读取时仍按完整内容键校验；7 项新测试含「克隆时预计算未完 → B 段等待后继承」真实时序 | SHI 伴随任务重采样 18.0 / 25.5 s、形态 5.9 / 5.8 s → 0 |
| 2 | **体场几何缓存（P0-4）**：确认出口后的 `build_volume_case` 结果存为 `volume_case-*.npz`（类型化树 + 数组，无 pickle，载荷校验和）；同任务重跑 / 换包重跑 / 查重复用命中；过期条目随 B 段清理（补进 `prune_geometry_cache`） | `volume_cache.py`（新）、`volume_pipeline.py`、`pipeline.py` | Tier A：两例真实任务「直接 / 未命中写入 / 命中」三路 case + aux 全部逐位相同（`volume_geometry/check_volume_cache.json`）；9 项新测试 | 命中时 11.0 / 12.9 s → 0.04 / 0.07 s；每例 18–27 MB。组合请求首跑不命中 |
| 3 | **体场采样内判证书**：VTK 判在内部的中心线采样点作锚点，落在锚点安全球（0.999 × 到封闭面距离）内的查询直接判内，其余仍交 VTK（与 v0.12.2 流线同一证书） | `volume_geometry.py` | Tier A：5 个真实几何整例 `build_volume_case` 与 v0.15.10 逐位相同（`volume_geometry/check_sampling_equal.json`）；VTK 调用少 40–60%，内判耗时 2.1–4.4 → 0.7–1.4 s（`proto_cert.json`） | 与 #4 合计：5 例 5.4–13.0 → 4.5–8.4 s |
| 4 | **边去重**：`_boundary_loops` 的 `np.unique(axis=0)` 改为 int64 单键一维去重 | `volume_geometry.py` | Tier A：值 / 顺序 / 计数 / dtype 相同（int64 / int32 / uint32、空、负数回退） | 120 万条边 1.39 → 0.21 s，每例 3 次 |
| 5 | **流线端点复用**：接受的端点已算出的 IDW 方向 / 速度 / 支撑作为下一步起点，不再重复 kNN / IDW；`inside()` 调用序列与 VTK 全局随机位置完全不变；逐点 list 改为数组 | `streamlines.py` | Tier A：两例真实任务 522 / 524 条线、`thin_lines`、info 字节相同，与生产 `streamlines.vtp` 相同（`streamlines/bench_final.txt`）；6 项新测试含冻结的旧循环与变异测试 | 2.09 → 1.73 s、3.79 → 3.22 s |
| 6 | **caps 精简**：`build_case` 只算封口记录（`virtual_caps` + `median_spacing`），不再建完整 oriented cloud；签名守卫回退 | `geometry.py` | Tier A：6 个真实任务 43 个数组字节相同（`caps/realdata_*.json`）；11 项新测试 | `build_case` 每例 −0.17 ~ −0.34 s |
| 7 | **集成设备输入复用（P0-3）**：同一集成内支持点 / 查询块 / 16 邻域 patch 张量每例只上设备一次，各成员只读复用（版本计数防原地改写，预算 1024 MiB 且 ≤ 空闲显存 1/4，OOM 回退原评估器）；块可保留时绕开 `input_memo` 的私有副本，首个成员并行构建其余查询块；输出每成员一次回传 | `prepared_inference.py`（新）、`families.py` | CPU 4 线程：X5D / M1 / PF6_VF6 开关逐位相同；GPU：与原实现差 ≤ 1.9e-5 Pa，与原实现两次运行间差（≤ 3.1e-5 Pa）同一分布（`inference/bench_prepared_final.json`）→ Tier B | 见下 |

**#7 的计时**（GPU 2 空闲、每次新进程、预热后第一次推理，`inference/bench_first2.jsonl`）：M1 首调中位 LV 1.88 → **0.80 s**、SHI 2.60 → **0.60 s**（开启后 SHI 四次都在 0.58–0.61 s，关闭时 1.7–3.4 s）；同例重复调用 X5D 五模型 1.27–1.68 → 0.55 s、M1 1.18–2.28 → 0.52 s；体场 0.22 s 不变。

排查经过（避免重复踩坑）：
- 最初接入后 `reused` 始终为 0——各成员 `cfg.data.feature_stats_path` 路径不同（统计内容相同），块键永不相等。改为键里去掉路径、统计内容另作 `feat_stats` 入键。
- 修好后收益仍被噪声淹没，且体场慢约 70 ms：每个成员对整个 case 做一次 18–24 ms 的哈希。改为每个 case 字典每作用域哈希一次（顶层条目被替换即重算；与 `input_memo` 同一身份前提）。
- 端到端里 M1 推理 0.8–3.5 s 大幅波动，cProfile 定位到 `input_memo` 为每个构建结果另存的约 300 MB 私有副本（18 次 `ndarray.copy` 0.46–2.09 s，新页首次写入在高负载下很慢）+ 单线程 `build_query_patch`（0.65–1.2 s）。设备端复用后这些副本无人读取，于是绕开 `input_memo` 并并行构建查询块；首调由此变快且稳定。

## 3. 测试与回归

- 全量测试：**744 通过 / 3 跳过**（基线 700 + 新增 44：`test_cache_handoff` 7、`test_volume_cache` 9、`test_prepared_inference` 6、`test_volume_sampling_speedups` 5、`test_streamlines_reuse` 6、`test_caps_only` 11）。`test_pipeline_v014::test_memo_and_warm_up_…` 改为在 `WSS_DEPLOY_PREPARED_INPUTS=0` 下验证 `input_memo`（复用路径不再经过它，这是预期变化）。
- 黄金回归（自包含，GPU 2，6 例含 M1 与 PF6/VF6）：**6/6**（最终代码 77.6 s）。
- 真实数据逐位核对：体场缓存 2 例、采样证书 + 边去重 5 例、流线 2 例、caps 6 例（脚本与结果在本目录子文件夹；`streamlines/` 两个脚本读的输入在会话草稿区，重跑需先用 `rebuild_inputs.py` 重建）。

## 4. 没做与原因

- **VTK SMP STDThread 后端**：批量内判快约 15 倍、掩码相同，但后端是进程全局状态，会影响并发的 A 段 VTK / VMTK，未采用。Python 线程并行 VTK 调用无加速（不释放 GIL）。
- **流线里省掉重复的 `inside()`**：可再省 0.35–0.75 s、两例结果相同，但会移动 VTK 全局随机序列、无法证明逐位相同，留待用户裁定。
- **确认出口前预算体场几何**：采样点看起来与命名无关，但封口诊断含出口标签，需要先拆依赖并证明，未做。
- **模型内诊断同步、FPS / radius / 插值计划缓存、FiLM 代数重排、编译 / 混合精度**：需改 `training_wss_min/` 模型代码（配置驱动、冻结副本协议），且空闲 GPU 上集成推理已约 0.55 s，收益有限，未做。
- **P0-1 真实墙钟与等待分项写入 summary**：未做；本次计时用 `e2e/e2e_combo.py` 在沙箱里直接量。
- **上线**：用户裁定提交推送并上线、流线重复 `inside()` 暂不省；提交 `0b975ab` 已推 origin，09-28 19:57 `service upgrade --drain 600` 上线（PID 3496824，GPU 1），三个发布包新代码预热正常。升级后已有任务的几何缓存条目会因源码哈希变化各未命中一次（结果相同，第一次重跑慢几秒）。

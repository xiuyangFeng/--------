# WSS-PINN 下一智能体目标提示词：冻结诊断后执行 Stage 0 至 Stage 1

> 执行状态（2026-08-06）：**已完成**。Stage 0-a/0-b、B0、四臂 seed1234、前二臂
> 三种子确认、导数/support Gate 与 full-volume val15 审计均已通过执行链；最终保留
> G-Raw，G-PE 与 local conditioner No-Go。总报告见
> `outputs/wss_pinn/volume_uvwp_peak_field_v4/stage1_multiseed_and_promotion_report.json`。
> 本次未读取 test35、未运行 WSS，也未启用 BC input/loss 或 PDE loss。

> 创建日期：2026-08-06  
> 归档日期：2026-08-21  
> 状态：🧊冻结 / 已完成执行合同 / 对应 `FROZEN FOR IMPLEMENTATION v1.0`  
> 现行入口：[体域 PINN 路线 README](../README.md) · [V4 大重构设计方案](../WSS_PINN_V4大重构设计方案_2026-08-08.md)  
> 科学真源：[核心代码诊断与下一轮设计建议](../核心代码诊断与下一轮设计建议_2026-08-05.md)  
> 历史提示词：[已完成的准稳态平滑场六臂预注册](./WSS_PINN_下一智能体目标提示词_准稳态平滑场六臂实验_已完成_2026-08-05.md)

## 可直接复制的提示词

```text
你现在接手 `/public/newhome/cy/Digital_twin/GNN` 中峰值体域 `u,v,w,p` 的下一轮实现。

当前科学设计已经冻结为：

`FROZEN FOR IMPLEMENTATION v1.0 / 2026-08-06`

科学真源是：

`docs/02-推进与变更/WSS_PINN/核心代码诊断与下一轮设计建议_2026-08-05.md`

你的任务不是重新讨论路线，也不是继续已完成的准稳态六臂。你要按冻结顺序完成：

1. Stage 0-a：病例等权 validation、唯一选模标量 `S_field^cb`、B0 病例盲 atlas、
   零训练探针工程化；
2. Stage 0-b：outlet 双路径、inlet-wall 交线和 wall-zone 数据合同 Gate；
3. 在 Stage 0-a 通过后，实现并预检 Stage 1 的 B0 + global/local × raw/PE 纯监督矩阵；
4. 不得覆盖 V3 历史配置、数据、checkpoint 或输出。

本提示词不内含正式多 GPU 训练授权。你可以完成代码、数据派生、只读审计、测试、
CPU/GPU smoke、static preflight 和 dry-run submission；只有当前对话中的用户明确授权
正式提交时，才可启动 Stage 1 训练。

一、必须先完整阅读

- `wss_pinn/AGENTS.md`
- `wss_pinn/README.md`
- `docs/02-推进与变更/WSS_PINN/README.md`
- `docs/02-推进与变更/WSS_PINN/核心代码诊断与下一轮设计建议_2026-08-05.md`
- `wss_mri_calculator/README_CFD_ADAPTATION.md`
- 当前 `wss_pinn/config.py`、`train.py`、`evaluate.py`、`losses.py`
- `wss_pinn/models/`、`data/`、`physics/`、`tools/`、`cluster/`
- V3 机器可读结果：
  `outputs/wss_pinn/volume_uvwp_peak_qs_smooth_v3/summary/`

旧六臂提示词已归档，只能用于追溯，不能复制其六臂矩阵、`lambda=1`、2500 epoch、
quasi-steady momentum 或自动正式提交合同。

二、冻结范围与禁止项

1. 唯一主任务是患者级峰值体域 `u,v,w,p` 场重建。
2. 不直接预测 WSS，不新增 WSS loss/WSS 辅助监督，不用 WSS 选择 checkpoint、loss、
   采样、架构或 seed。
3. Profile-Secant V3 velocity→WSS 保持冻结；本阶段不运行新的 WSS 调参或验证闭环。
4. test35 已经 development-exposed：Stage 0/1 不加载、不评估、不调参、不生成新设计结论。
5. Stage 1 不输入病例 BC，不使用 wall/inlet/outlet loss，不使用 continuity/momentum PDE。
6. 不从 V3 checkpoint 热启动新模型；所有训练臂从配对随机初始化开始。
7. 不把 outlet monitor 写成真实 boundary pressure 或可部署输入。
8. 不把 B 级 6 例探针写成全库已确认根因；脚本和逐例产物入库后才能升级证据等级。
9. 不一次性上线局部核、PE、分层采样、分头 decoder、BC、动态权重和 physics。
10. 保留 dirty worktree 中所有用户改动；只修改本任务必要文件。

三、新路线与独立落盘

新实现默认使用独立路线：

- route：`volume_uvwp_peak_field_v4`
- configs：`wss_pinn/configs/volume_uvwp_peak_field_v4/`
- outputs：`outputs/wss_pinn/volume_uvwp_peak_field_v4/`
- audits：`outputs/wss_pinn/audits/volume_uvwp_peak_field_v4_*/`
- 新派生合同：`data_wss_pinn/volume_uvwp_peak_field_v4_train123_val15/`

开发 split 继续引用现有冻结文件：

- `wss_pinn/configs/splits/split_WSS_PINN_AG_AAA_ILO_qs_smooth_v3_train123_val15_test35_s1234.json`
- SHA256：`c80cb65ad95f7d76fadaff82ea02c35c27ad24ab2474e1050bf981eff97493f9`
- Stage 0/1 只允许读取其中 train123/val15 role；test35 role 必须由 guard 拒绝。

四通道 train-only 统计可引用现有
`data_wss_pinn/volume_uvwp_peak_qs_smooth_v3_train123_val15_test35/field_stats.json`，
但必须校验其 scope、split SHA256 和文件 SHA256，不得按 val15 重新拟合。

现有 V3 目录全部只读：

- `wss_pinn/configs/volume_uvwp_peak_qs_smooth_v3/`
- `data_wss_pinn/volume_uvwp_peak_qs_smooth_v3_train123_val15_test35/`
- `outputs/wss_pinn/volume_uvwp_peak_qs_smooth_v3/`

Stage 1 可以通过带 SHA256 的 manifest 引用现有 train123/val15 体域资产，避免复制大文件；
新生成的 selection、atlas、boundary 和 probe 资产必须落到 v4 新路径。新 route/config/output
必须受写路径 guard 保护。

四、Stage 0-a：选模合同和零训练基线

### 0-a.1 病例等权 validation

当前 `train.py` 把 validation batch 等权平均，val15 / `batch_cases=2` 导致最后一例固定
双倍权重。新路线必须改成：

- 每例独立计算四通道标准化 data MSE；
- 先在病例内对点和通道聚合，再对 validation 病例等权平均；
- validation 病例、每例 query index、seed 跨所有臂完全一致；
- 开发阶段固定 5000 query/例；晋级报告再做 full-volume；
- 旧 V3 训练日志和 checkpoint 选择不回写、不重命名。

冻结的唯一 checkpoint/early-stop 标量为：

S_field^cb = mean_case [ (MSE_u + MSE_v + MSE_w + MSE_p) / 4 ]

其中 `u/v/w/p` 已用 train-only component std 标准化，压力沿用病例 strict-volume
mean-centered gauge。

新路线 checkpoint 至少保存：

- `best_validation_field_cb.pt`：唯一主 checkpoint；
- `best_validation_total.pt`：仅为未来含额外 loss 的敏感性入口，Stage 1 与主 checkpoint
  应相同或明确说明；
- `last.pt`；
- `initialization.pt` 和初始化 state SHA256。

日志至少新增：

- `validation_field_score_cb`
- 每例 `data_u/v/w/pressure`
- 病例等权、点等权和当前旧 batch 等权三种口径的对照审计
- 被旧逻辑双倍加权的病例 ID 和修复前后 checkpoint 差异

### 0-a.2 B0 病例盲群体场/atlas

实现不使用目标病例 `u,v,w,p` 的 train-only 零训练基线：

- 输入只允许注册坐标和训练病例资产；
- validation 预测只能使用 train123，不得把 val15/test35 放入 atlas；validation truth 只由
  独立 evaluator 在预测冻结后读取以计算指标；
- K/带宽只在训练内部病例级交叉验证中选择并冻结；
- 禁止在 validation/test truth 上拟合 affine scale、bias 或 calibrator；
- 压力使用与网络一致的病例中心化 gauge；
- 缓存必须记录 train case list、采样规则、K/带宽、坐标/场统计和 SHA256。

B0 至少报告 `u/v/w/speed/p` 的 case-balanced 与 pooled MAE/RMSE/R²，以及每例结果。
Stage 1 所有模型必须同时报告相对 B0 的逐病例 ΔR²/ΔMAE。偏相关或残差相关只能作为
病例级机制诊断，不能替代 `S_field^cb` 选模。

### 0-a.3 将 B 级探针工程化

把冻结诊断中尚未入库的探针变成可复跑工具/测试和 JSON/CSV：

- 制造解：场值、一阶、二阶导数和 Carreau–Yasuda residual；
- current V3 support 重采样稳定性，明确结论只适用于当前 global checkpoint；
- B0 atlas 的全 val15 `u/v/w/p` 结果；
- bulk k-NN 平滑参考，只在 train/val 运行并明确它读取目标真值、不是可部署 baseline；
- 直接测量 `cos(grad data, grad continuity)`、`cos(grad data, grad momentum)`、
  `cos(grad data, grad BC)`，不再由总 PDE 反推 momentum；
- latent 多样性和病例间/病例内方差压缩；
- 所有探针绑定 checkpoint SHA256、split SHA256、case list、随机 seed 和 git state。

以上新探针统一只使用 train/val；不得为了复现附录中的 6 例数字再次读取 test35。

Stage 0-a Gate 至少要求：

1. 新 validation 聚合单元测试通过；
2. `S_field^cb` 与手工病例等权计算一致；
3. B0 的构建/预测路径不读取 val/test target；独立 evaluator 与 predictor 有明确接口隔离；
4. 固定 validation query 跨臂 hash 一致；
5. 旧 V1/V2/V3 配置仍可解析，旧 checkpoint/输出不变化；
6. audit JSON 明确 `gate=pass/fail` 和失败原因。

Stage 0-a 未通过时，不实现或提交 Stage 1 训练。

五、Stage 0-b：最小边界数据合同

Stage 0-b 不阻塞纯监督 Stage 1，但阻塞 Stage 2 和任何 BC loss/input 实验。

### 0-b.1 outlet 双路径

实现可复现解析器：

- 从 `Global_conditions/Fluent_*.out` 提取 `P_ave_out* / Q_ave_out*`；
- 去除 MPI 并行重复打印，恢复唯一时间序列；
- 将日志时刻与 `peak_step`、CFD 导出时刻、zone ID 和 UDF SHA256 对齐；
- 与 CFD outlet face/邻接面静压做 train-only 双路径比较；
- Gate 阈值由 train-only 误差分布冻结，不能沿用单病例的 `20 Pa` 建议；
- 日志缺失病例使用明确的 CFD face-pressure 回退，并写 provenance；
- V3 monitor 只作历史差异对照，不得作为新路线 target/input。

报告至少包含 173 例覆盖、每例四出口、日志/回退来源、绝对差、gauge 转换和 Gate。

### 0-b.2 inlet-wall 与 wall-zone

- 入口和壁面优先在 face interior 做面积采样；
- 若当前阶段无法恢复完整 face connectivity，至少剔除共享坐标并加入一个与网格尺度绑定的
  edge buffer，不能在同一点施加 inlet plug 和 wall no-slip；
- 逐 wall zone 审计是否邻接目标 strict fluid domain；
- 保存 zone ID、点数、面积/法向可用性、transform 和异常病例；
- manifest 明确区分 deployable input、training supervision、solver response 和 monitor。

Stage 0-b 只需要构建合同、审计和未来接口；Stage 1 不启用这些 BC。

六、Stage 1：纯监督表示矩阵

固定使用 `xyz+geom`、train123/val15、相同采样、相同训练预算和相同主选模标量。

矩阵为：

| 臂 | 病例条件 | query coordinate |
| --- | --- | --- |
| B0 | train-only 病例盲 atlas | 零训练 |
| G-Raw | 当前 global max-pool latent | raw xyz |
| L-Raw | 光滑局部多尺度条件 | raw xyz |
| G-PE | 当前 global max-pool latent | 带限 PE |
| L-PE | 光滑局部多尺度条件 | 带限 PE |

执行顺序固定为：

1. B0；
2. G-Raw 与 L-Raw；
3. G/L 的场值与导数 Gate 通过后再运行 G-PE 与 L-PE。

Stage 1 只能回答“在同等缺失 BC 条件下哪种表示更好”，不能把绝对性能写成完整
patient-specific surrogate 上限。

### 1.1 G-Raw

- 复用 V3 global conditional smooth field 的科学结构作为历史对照；
- 使用新 route、新 validation 聚合和新 checkpoint，不加载 V3 权重；
- query 路径为 raw xyz + case latent → smooth decoder。

### 1.2 L-Raw

- support encoder 可以使用离散 FPS/kNN/ball grouping，因为不对 support coordinate 求 PDE；
- query conditioner 必须对 query xyz 连续、稳定可导；禁止 query top-k/kNN 离散切换；
- 优先实现带正带宽的 Gaussian/softmax 光滑局部核或等价连续核；
- 保存带宽、质心数、有效邻域、跨分支泄漏和 support 重采样诊断；
- 通过制造场、有限差分和探针加密的一/二阶导数 Gate；
- 不同时加入 FiLM、attention、Transformer、PointNeXt、分层采样或动态 loss。

### 1.3 PE

- PE 只作用于 query coordinate，不改变病例条件；
- 使用带限频率，配置中同时记录归一化频率和对应物理波长；
- 频率集合必须在正式训练前预注册，不做 test/val 后扫频；
- 检查 PE 对一/二阶导数幅值、流量和压力梯度的影响；
- PE 是低优先级交互轴，不得因为单个点值指标改善而自动晋级。

### 1.4 配对公平

- 四个训练臂统一 data-only；无 BC input、无 BC loss、无 PDE；
- 统一 train/val case、固定 validation query、support/query 数、optimizer、scheduler、
  max epoch、patience、seed 和训练预算；
- 参数量尽量匹配；若最大/最小参数量比超过 `1.10`，preflight 失败，除非有明确结构性
  原因并在矩阵 README 预注册；
- 单 seed 开发筛选后，只按 `S_field^cb` 选前两臂做至少 3 seeds；
- exact max epoch、patience、min_delta、PE 频率和局部核带宽在正式提交前写入
  `decision_contract.json`，不得用 test35 决定。

七、Stage 1 评估和 Go/No-Go

主 checkpoint：`best_validation_field_cb.pt`。

主排序：validation `S_field^cb`，越低越好。

必须同时报告：

- `u/v/w/speed/p` 的 case-balanced 与 pooled MAE/RMSE/R²；
- 速度向量 L2、排除极低速点后的方向误差；
- near-wall/core/分叉和 AG/AAA/ILO 分层；
- 病例内预测/真值方差比和病例间均值方差；
- B0、G-Raw、L-Raw、G-PE、L-PE 的逐病例配对差；
- 多 seed 均值、标准差、病例 bootstrap CI、改善病例数和最差队列；
- 场值、一阶/二阶导数和 support 重采样稳定性。

晋级必须同时满足：

1. 相对 G-Raw 的 `S_field^cb` 改善在多 seed 下方向稳定；
2. 相对 B0 不只是 pooled 指标改善，病例等权和患者特异诊断也有增量；
3. `u/v/w/p`、near-wall 和最差队列没有超过预注册等效界值的退化；
4. 新 query decoder 通过导数和重采样 Gate；
5. 不使用 test35、WSS 或 physics residual 反选。

停止条件：

- L-Raw 多 seed 无增量：停止继续堆局部 decoder，优先进入缺失 BC/条件问题；
- PE 无增量或伤害导数：关闭 PE，不继续扫频；
- 所有模型仅复现 B0：将“病例特异条件不足”升级为主假设，进入 Stage 2；
- overall 改善但某通道/near-wall/最差队列稳定恶化：No-Go；
- 任何 test35 意外读取：该轮结果作废并记录泄漏。

八、Stage 2 及以后只预留接口

本提示词不要求直接实现 Stage 2–4，但 Stage 1 代码要避免堵死后续接口：

- Stage 2：固定最佳表征后比较无/有可部署 `Q_in/phase/R1/R2/C`；
- Stage 3：区域采样、`u/v/w` 与 `p` 分头 decoder、流量/压降辅助，每次只加一个因素；
- Stage 4：先 continuity；瞬态 momentum 分成 CFD `du/dt` 已知源项监督和
  time-conditioned 稀疏多时相两条独立路线；
- WSS 只在最终 `u,v,w,p` checkpoint 冻结后运行一次 downstream audit，不能触发
  WSS 调参或同轮 checkpoint 重选。

九、测试、preflight 和授权边界

至少新增/更新：

- validation 病例等权与固定 query 测试；
- `S_field^cb` 手工对照测试；
- B0 构建/预测路径与 validation truth evaluator 隔离测试；
- 新 route 写路径和旧 V3 只读保护；
- G/L Raw/PE shape、参数量、初始化 hash 和配置合同；
- query 一/二阶 autograd 与有限差分 Gate；
- outlet 日志解析、MPI 去重、时相对齐、回退 provenance；
- inlet-wall overlap 和 wall-zone audit；
- CPU smoke、static preflight、单配置 GPU dry-run；
- 旧 V1/V2/V3 配置兼容性。

未获得当前对话明确训练授权时：

- 可以运行零训练审计、focused tests、CPU smoke、GPU dry-run 和 submission dry-run；
- 不得执行正式 `--submit`，不得后台启动多 GPU 训练。

十、文档与交付

修改实验文档、代码、配置或 Gate 后，按实际状态同步：

- `wss_pinn/README.md`
- `wss_pinn/AGENTS.md`（只放长期硬约束，不写流水账）
- `docs/02-推进与变更/WSS_PINN/README.md`
- `docs/02-推进与变更/代码修改与实验推进记录.md`（文首）
- `docs/02-推进与变更/WSS最小化_代码修改与实验推进记录.md`（文首）
- 只有路线级状态变化时更新 `README.md`、`docs/README.md`、`docs/实验设计总纲.md`

最终交付必须给出：

- 改动文件和独立新路径；
- Stage 0-a/0-b Gate JSON/CSV 与 SHA256；
- `S_field^cb` 修复前后对照；
- B0 atlas 全 validation 结果；
- 测试、CPU/GPU preflight 和 dry-run 证据；
- Stage 1 matrix/decision contract；
- 是否具备正式训练条件、尚缺什么授权或外部资源；
- 没有做的事项，尤其是 test35、WSS、BC/PDE 和正式训练。

不要重新设计路线，不要只输出计划。先机械核对现有实现，再完成授权范围内的实现、测试、
审计和文档收口。遇到 Stage 0 Gate 失败时，保留可复现证据并停止对应后续阶段，不得用
伪数据、monitor proxy 或宽松指标绕过。
```

## 使用说明

复制上面的代码块到新的 Codex 任务即可。若希望下一智能体在 Gate 通过后正式提交
Stage 1 训练，需要在新任务中额外明确授权；仅复制本提示词默认不授权正式训练。

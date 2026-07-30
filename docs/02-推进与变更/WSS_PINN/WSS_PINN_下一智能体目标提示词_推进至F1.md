# WSS-PINN 下一智能体目标提示词：实现并推进至 F1

> 建立日期：2026-07-30  
> 当前状态：**历史执行合同；目标已于 2026-07-30 执行完成。F1 audited /
> No-Go，F2 blocked，不得按本文重复提交。最新事实以
> [阶梯实验矩阵](WSS_PINN_阶梯实验矩阵与进度跟踪.md)为准。**  
> 原用途：整段交给下一智能体执行。  
> 目标边界：完成 P0、P1、F0-U、F0-UP、F1 的代码与实验入口；F2/F3 不在本轮实现范围。  
> 长训练例外：允许提交 Slurm、确认启动正常并完成记录后交接，不要求原地等待完训。

## 可直接复制的目标提示词

你现在位于仓库：

```text
/public/newhome/cy/Digital_twin/GNN
```

请直接实施 WSS-PINN 独立实验线，不要只写方案。你的目标是把该路线从当前
S0/No-Run 推进到 **F1（continuity + no-slip）**，包括可执行代码、配置、测试、
Slurm 提交入口、必要的 P0/P1 产物以及规范记录。

### 1. 先读真源并遵守边界

开始修改前完整阅读：

```text
wss_pinn/AGENTS.md
wss_pinn/README.md
docs/02-推进与变更/WSS_PINN/README.md
docs/02-推进与变更/WSS_PINN/WSS_PINN_阶梯实验矩阵与进度跟踪.md
docs/02-推进与变更/WSS最小化_体域物理约束与PINN训练路线_2026-07-29.md
docs/02-推进与变更/WSS最小化_代码修改与实验推进记录.md
data_new/AAA/ruputer/DING_JUN_FENG/udf-inlet4.c
```

随后只读检查与本任务最相关的既有实现：

```text
pipeline_wss_min/
training_wss_min/
training_wss_min/cluster/
```

只复用经过核对的数据读取、模型、指标和 Slurm 约定，不直接修改旧路线。
工作区可能已有用户未提交修改，必须保留，不得清理、覆盖或回退。

### 2. 冻结的科学口径

- 最终任务始终是预测**峰值时相壁面 WSS 标量分布**。
- \(u,v,w,p\) 是训练期辅助连续场，不把“预测速度/压力”偷换为最终任务。
- 壁面为非滑移刚性壁面：
  \(\mathbf u|_{\Gamma_w}=0\)。
- 血液为不可压缩、\(\rho=1060\ \mathrm{kg/m^3}\)。
- 流变使用与 UDF 一致的 Carreau–Yasuda 型广义牛顿非牛顿模型；参数、量纲和
  流量单位必须从 UDF/现有数据真源审计，不能凭记忆填写。
- 工程可用条件为共享入口 \(Q(t)\) 及每个几何分支的入口/出口面积；真实 RCR
  只允许作为 Oracle，第一阶段不要求重新从 Fluent 导出 zone、connectivity。
- 既有 `random5000` 只保留为 W0 壁面 WSS 锚点，不得把它冒充体域 PDE 点。
- PINN 数据必须分为 wall / near-wall / core 三个独立槽，并复用同一
  sampling manifest 做配对比较。
- `test36` 可以沿用，但所有记录必须注明
  `reused_development_screen`，不得声称为从未使用的独立测试集。

### 3. 硬隔离

只允许在以下位置写本路线的新资产：

```text
代码与配置：wss_pinn/
派生数据：  data_wss_pinn/
运行结果：  outputs/wss_pinn/
路线记录：  docs/02-推进与变更/WSS_PINN/
```

以下均为只读上游：

```text
data_new/
data_wss_min/
pipeline_wss_min/
training_wss_min/
既有 checkpoints 和 runs
```

如需旧模型，只能在 `wss_pinn/` 内写 adapter 读取冻结权重；不得为了方便而修改
旧训练入口或旧 bundle。

### 4. 本轮必须推进的阶梯

按以下顺序实现和验证；不得跳过数据/数值闭环就直接开 physics loss。

#### P0-A｜体点身份与单位审计

- 审计峰值及必要邻近时相的 cell ID、坐标、行序是否稳定。
- 审计坐标、速度、压力、时间、WSS、流量和流变参数的单位。
- 产出逐病例机器可读 JSON/CSV 和汇总报告。
- 原始数据只读；任何修复只能落为显式转换规则和派生 sidecar。

#### P0-B｜CFD velocity→WSS Oracle

- 基于 CFD 真值速度、已有可得壁面/几何法向和
  \(\mu(\dot\gamma)\) 重建 WSS。
- 比较合理的多壳层近壁梯度拟合，报告方向、幅值、稳定性及与 CFD WSS 的误差。
- 如果现有法向或近壁对应关系不足，必须明确指出哪个字段缺失、影响哪些病例；
  不得伪造 Fluent zone/connectivity。

#### P0-C｜CFD residual Oracle

- 为到达 F1，**continuity residual Oracle 是必做项**：在 CFD 真值或可验证解析场
  上确认坐标导数、无量纲化和残差量级稳定。
- momentum residual 属于 F3，本轮可以只保留配置/接口和诊断说明，不得为了
  “看起来完整”提前把 momentum loss 接入 F1。

#### P0-D｜法向与边界可得性审计

- 核验壁面法向的来源、方向、归一化和覆盖率。
- 记录入口/出口 zone、面片法向、面积、connectivity 当前是否可得。
- 缺少可靠 zone 不阻塞 F0/F1，但必须在报告中阻塞 hard flux/RCR residual。

#### P1｜独立 physics sidecar

- 先选 AG、AAA、ILO 各 1 例做 pilot；稳定后再扩到用于严格过拟合的 3–5 例。
- sidecar 至少包含 `physics_manifest.json`、静态几何/采样字段和峰值场标签；
  是否加入峰值前后时相根据实际可用字段决定并记录。
- 每个 manifest 保存原始文件、父 bundle、采样清单的路径与 SHA256。
- wall / near-wall / core 分槽；建议 pilot 约 5k / 8k / 8k，但实际数量必须由
  可用点数和审计结果确定，并通过配置控制。

#### F0-U｜速度 data-only

- 模型保留最终 `WSS_direct` 输出，并新增平滑的 \(u,v,w\) 查询场。
- 只开 WSS/速度数据监督，不开压力、continuity、no-slip、WSS physics 或 momentum。
- 先对 3–5 例严格过拟合，检查有限值、坐标梯度和 checkpoint 重载。

#### F0-UP｜增加压力 data-only

- 在同一数据、采样、容量和协议上增加压力输出及 gauge-invariant 压力监督。
- continuity、no-slip、momentum 仍关闭。
- 压力表示必须显式处理每病例/时相的 gauge，不用绝对压力零点制造伪误差。

#### F1｜continuity + no-slip

- 以 F0-UP 为唯一配对控制，只新增：

  \[
  \mathcal L_\mathrm{cont}=\|\nabla\cdot\mathbf u\|^2,\qquad
  \mathcal L_\mathrm{noslip}=\|\mathbf u|_{\Gamma_w}\|^2.
  \]

- F1 中 momentum weight 必须为 0；若配置为非零应在预检时直接报错。
- 用完全相同的病例、sampling manifest、模型容量、seed 和训练预算比较
  F0-UP/F1。
- 检查 continuity/no-slip 是否改善，同时报告最终 WSS 的
  \(R^2\)、MAE/RMSE、高 WSS 区域指标和病例级配对差；不能只汇报 physics loss。
- 将 F1 的两个物理权重置零时，数值行为必须退化为 F0-UP，至少通过固定
  seed 的单 batch 等价测试。

### 5. 代码必须由配置驱动

不要为 F0-U、F0-UP、F1 复制三套训练代码，也不要通过临时改 Python 常量做实验。
使用一个训练/评估入口和经过校验的配置模式。建议至少实现：

```text
wss_pinn/
├── config.py
├── data/{raw_io.py,sidecar.py,sampling.py,dataset.py}
├── models/{anchor_adapter.py,geometry_encoder.py,field_decoder.py}
├── physics/{rheology.py,residuals.py,wall_shear.py,nondimensionalize.py}
├── tools/{audit_interior_identity.py,audit_udf_units.py,
│          velocity_to_wss_oracle.py,cfd_residual_oracle.py,
│          build_physics_sidecar.py}
├── configs/
├── tests/
├── cluster/
├── train.py
└── evaluate.py
```

可以根据实际代码结构合并模块，但职责和测试边界必须清楚。所有实验差异只通过
JSON/YAML 配置字段表达。配置至少覆盖：

```yaml
experiment:
  id: ...
  stage: p0a|p0b|p0c|p0d|p1|f0u|f0up|f1

paths:
  raw_root: data_new
  wss_bundle_root: data_wss_min
  sidecar_root: data_wss_pinn
  output_root: outputs/wss_pinn

data:
  split_path: ...
  split_label: reused_development_screen
  cohorts: [AG, AAA, ILO]
  timesteps: [peak]

sampling:
  wall_points: ...
  near_wall_points: ...
  core_points: ...
  seed: ...
  manifest_path: ...

model:
  predict_direct_wss: true
  predict_velocity: true|false
  predict_pressure: true|false
  ...

loss:
  direct_wss_weight: ...
  velocity_data_weight: ...
  pressure_data_weight: ...
  continuity_weight: ...
  no_slip_weight: ...
  momentum_weight: 0

physics:
  density_kg_m3: 1060
  rheology:
    model: carreau_yasuda
    # 参数由审计产物/配置真源填写
  characteristic_scales: ...

train:
  seed: ...
  epochs: ...
  precision: ...
  resume: ...
  checkpoint_every: ...

cluster:
  partition: ...
  qos: ...
  time: ...
  memory: ...
  cpus: ...
  gpus: ...
```

字段名可以调整，但不能减少这些能力。必须有 stage-aware 配置校验：

| 阶段 | velocity | pressure | continuity | no-slip | momentum |
| --- | ---: | ---: | ---: | ---: | ---: |
| F0-U | on | off | 0 | 0 | 0 |
| F0-UP | on | on | 0 | 0 | 0 |
| F1 | on | on | \(>0\) | \(>0\) | 0 |

路径、loss 权重、采样数量、seed、训练预算和 Slurm 资源均不得散落硬编码。
后续新实验应只需复制/继承配置、修改字段并提交，不需要改训练源代码。

### 6. 最低测试和门禁

至少覆盖：

- 配置解析、默认值、stage 组合合法性及 F1 禁止 momentum。
- 写路径守卫：拒绝把输出指向四个只读旧根。
- 三槽采样非空、无混槽、固定 seed 可复现、manifest 可重载。
- Carreau–Yasuda 极限、单位、正黏度和有限梯度测试。
- 解析速度场的 divergence 正/反例。
- 非滑移边界损失正/反例。
- 压力加常数后的 gauge-invariant loss 不变。
- F1 两个 physics 权重为零时与 F0-UP 单 batch 等价。
- CPU 小样本 forward/backward；有 GPU 时再做 CUDA smoke。
- 无 NaN/Inf、checkpoint 保存/重载、resolved config 落盘。
- P0/P1 和正式训练前的只读预检。

单元测试和极小 smoke 可以本地执行；不要在登录节点直接跑长训练。

### 7. 集群提交是唯一正式训练方式

正式 F0-U/F0-UP/F1 训练统一走 Slurm，参考现有
`training_wss_min/cluster/` 的可靠约定，在 `wss_pinn/cluster/` 新建独立入口。
建议提供：

```text
preflight.slurm
run_experiment.slurm
submit_experiment.py
```

理想用法类似：

```bash
python wss_pinn/cluster/submit_experiment.py \
  --config wss_pinn/configs/f0u_pilot.json
```

同一个 launcher 必须可以通过配置提交 F0-U、F0-UP、F1，不允许每次手改 Slurm
正文。提交器应：

- 幂等或至少能检测同一 experiment ID 的已有有效作业；
- 先跑 preflight，再提交训练；
- 把 resolved config、config SHA256、代码 commit/dirty 状态、环境、数据/采样
  manifest SHA、Slurm Job ID、stdout/stderr 和 checkpoint 路径写入运行目录；
- 支持 resume，并给出明确的监控与续跑命令；
- 不覆盖已有运行目录。

科学 Gate 与调度依赖必须分开：前一作业 `COMPLETED` 不等于结果达到 Go。
可以提前把 F0-UP/F1 的代码、配置和提交脚本全部准备好，但只有前一阶段结果审计
通过后才提交下一阶段，不能仅靠 `afterok` 自动越过科学 Gate。

### 8. 长训练无需原地等待，但必须可靠交接

本任务的完成口径分为两层：

1. **实现层必须做到 F1**：P0/P1/F0-U/F0-UP/F1 共用代码、配置、测试、评估和
   Slurm 入口全部可用，不能因第一个长训练未结束而只实现到 F0-U。
2. **执行层尽可能推进**：按科学 Gate 运行到当前可解锁的阶段。如果某次模型训练
   很长，可以提交集群后停止等待，不要求在本轮对话里等到完训。

遇到长作业时，在交接前至少做到：

- `sbatch` 成功并记录 Job ID；
- 用 `squeue`/`sacct` 确认作业存在且状态合理；
- 若已启动，检查首段日志，排除配置、路径、导入、CUDA、数据加载等立即错误；
- 若仍排队，记录队列状态；
- 写清实际使用的配置、配置哈希、输出目录、监控命令、恢复命令和下一个科学 Gate；
- 状态必须写成 `submitted`/`running`/`pending_review`，不能写成 `completed`。

若 F0-U 长训练仍在运行，就不要越过其科学 Gate 提交 F0-UP/F1；但 F0-UP/F1 的
代码、配置、测试和 dry-run 必须已经准备完成。

### 9. 记录和最终交付

每完成一个审计、sidecar 或实验，都同步：

```text
docs/02-推进与变更/WSS_PINN/README.md
docs/02-推进与变更/WSS_PINN/WSS_PINN_阶梯实验矩阵与进度跟踪.md
docs/02-推进与变更/WSS最小化_代码修改与实验推进记录.md
docs/02-推进与变更/代码修改与实验推进记录.md
```

记录按时间倒序，至少包含：

1. 本次主要修改；
2. 对应代码/配置/产物；
3. 推进到实验哪一步；
4. 当前状态判断。

每个病例/阶段失败也要记录，不能只保留成功结果。任何 Gate、阈值或数据口径变化
必须先登记，再运行。

最终回复必须明确列出：

- 实际新增/修改的代码和配置；
- 已完成的 P0/P1 产物及审计结论；
- 通过的测试、preflight 和 smoke；
- F0-U/F0-UP/F1 各自是 `code_ready`、`submitted`、`running`、
  `completed`、`pending_review` 还是 `blocked`；
- 所有已提交 Job ID、配置路径、输出路径和监控命令；
- 尚未解锁的科学 Gate 或真实数据阻塞。

### 10. 完成定义与禁止项

本轮达到目标至少要求：

- 实现层完整覆盖到 F1，实验切换只改配置字段；
- P0/P1 pilot 在数据允许范围内产生机器可读报告和 sidecar；
- F0-U/F0-UP/F1 的配置、评估、预检、Slurm 提交和恢复路径可执行；
- 已运行到当前科学上允许的最远阶段，或已按上述规则提交长作业并可靠交接；
- 原 WSS-only 数据、代码、checkpoint 和 run 未被修改；
- 文档状态与真实执行状态一致。

本轮明确禁止：

- 实现或开启 F2 的 \(WSS_\mathrm{phys}\) 训练；
- 实现或开启 F3 的 momentum loss；
- 为做 PINN 重新要求长时间 Fluent 导出；
- 用壁面 `random5000` 代替体域点；
- 用 `test36` 调完参后宣称独立泛化；
- 在未通过前置 Gate 时伪造“已推进到 F1”；
- 只停留在设计讨论而不改代码。

请自主检查数据和现有代码后实施。只有当某个缺失选择会改变冻结物理口径、数据划分
或造成不可逆修改时才询问用户；普通实现细节请做保守、可配置、可测试的决定并记录。

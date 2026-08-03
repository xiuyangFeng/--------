# Point-cloud normal multi-scale WSS v3

## 状态与冻结边界

- V1 `pointcloud_adaptive_v1` 与 V2 `pointcloud_multiscale_v2` 已转为只读比较器；
- 两者的核心源码、冻结配置和正式结果 SHA256 记录在各自
  `freeze_manifest.json`；
- V3 不修改 V1/V2 算法入口，独立实现于
  `src/wss_normal_multiscale_v3.py`；
- V3 已在 train138 上完成消融和唯一配置选择，并冻结为
  `config_frozen_v3.json`；
- 冻结后已完成一次性 blind test35，结果未用于调参，V3 不再接受方法参数修改。

冻结检查：

```bash
cd /public/newhome/cy/Digital_twin/GNN/wss_mri_calculator
/public/newhome/cy/.conda/envs/GNN/bin/python experiments/verify_frozen_methods.py
```

## 方法

V3 针对 V1/V2 的三维欧氏 KNN 横向混合问题，将采样与拟合改为：

1. 对候选内部点计算法向深度 `eta` 和到目标法线的横向距离 `r_perp`；
2. 使用局部半径、边界层外缘速度和局部 Reynolds 数给出最大物理深度；
3. 在目标法线上设置物理深度站点，每个站点用附近内部单元做局部 IDW 插值；
4. 用最近壁面锚点、锚点间距和法向一致性拒绝非局部壁面所属点；
5. 对法线站点速度拟合无滑移二次剖面；实验性真实点模式仍支持各向异性 MLS：

   ```text
   v_t(s1,s2,eta) = eta * (g0 + g_eta*eta + g1*s1 + g2*s2)
   ```

   `g0` 为目标壁面法向速度梯度；
6. 仅在横向混合风险高、归属稳定且 ray 梯度不低于 V1 时接管；保留 V1 方向并把
   幅值修正限制在 `[1.0,1.2]`；
7. 可选地在多个物理深度比例上外推到零深度。

完整诊断包括局部 Re、深度上限、横向/法向距离、归属拒绝数、tube 拒绝数、
fallback 比例、拟合条件数和多尺度修正。

## 配置与阶段

所有病例范围、抽样数、几何来源、方法参数、输出位置和 node04 运行参数均来自 JSON：

| 配置 | 用途 | test35 |
| --- | --- | --- |
| `config_smoke_train3.json` | 三队列各 1 个 train 病例的链路冒烟 | 禁止 |
| `config_stage_a_geometry_train138.json` | 关闭 V3 多尺度，隔离法线管/归属/Re/MLS 收益 | 禁止 |
| `config_stage_b_multiscale_train138.json` | 开启物理深度多尺度的 train138 候选 | 禁止 |
| `config_frozen_v3.json` | 由 train138 唯一选定的冻结方法参数 | 不适用 |
| `config_stage_c_blind_test35.json` | 冻结后一次性 blind test35 | 仅 test |

运行器会校验配置字段并拒绝未知键。`method_frozen=false` 时若配置出现 `test` role，
会立即报错。blind 配置还会检查方法参数与 `config_frozen_v3.json` 完全一致，并核验
冻结配置 SHA256，避免测试阶段发生隐式修改。

仅验证配置，不加载病例：

```bash
PY=/public/newhome/cy/.conda/envs/GNN/bin/python
$PY src/run_v3_experiment.py \
  --config experiments/pointcloud_normal_multiscale_v3/config_smoke_train3.json \
  --validate-only
```

正式执行入口同样只接收配置：

```bash
$PY src/run_v3_experiment.py --config <config.json>
```

## node04：无 Slurm 提交方式

node04 没有可用 Slurm 调度，因此准备了显式 SSH + `nohup` + `flock` GPU 锁方案。
启动前会检查：

- node04 的 UID/GID 必须为 `1006/1007`；
- `/public/newhome/cy/Digital_twin/GNN` 与 GNN 绝对环境可访问；
- 在 node04 本机读取两张 A100 的显存占用；
- `/tmp/cy_wss_v3_gpu{0,1}.lock` 防止本项目两个任务抢同一张卡；
- 日志、PID 和状态写入本实验目录下被忽略的 `runs/`。

默认命令是 dry-run，**不会启动远端进程**：

```bash
experiments/pointcloud_normal_multiscale_v3/node04/submit_node04.sh \
  --config experiments/pointcloud_normal_multiscale_v3/config_smoke_train3.json \
  --gpu auto
```

将来允许提交且 GPU 当前空闲：

```bash
experiments/pointcloud_normal_multiscale_v3/node04/submit_node04.sh \
  --config experiments/pointcloud_normal_multiscale_v3/config_smoke_train3.json \
  --gpu auto --execute
```

若希望在 node04 上用轻量守护进程等待 GPU 释放，再依次启动：

```bash
experiments/pointcloud_normal_multiscale_v3/node04/submit_node04.sh \
  --config <config.json> --gpu auto --execute --wait
```

状态与停止：

```bash
experiments/pointcloud_normal_multiscale_v3/node04/status_node04.sh
experiments/pointcloud_normal_multiscale_v3/node04/stop_node04.sh <run_id>
```

注意：当前 V3 核心是 NumPy/SciPy `cKDTree` 算子，主要消耗 CPU 和内存；启动器仍按
用户要求将任务限定到 node04、检查并锁定指定 GPU、设置 `CUDA_VISIBLE_DEVICES`，
但它不会凭空获得 GPU 加速。后续若 profiling 证明邻域查询是瓶颈，再单独评估
FAISS/CuPy/Torch GPU 化，不能把“运行在 GPU 节点”误写成“核心已经 GPU 加速”。

## 冻结与 blind test 结果

test35 执行前已完成以下门槛：

1. train138 配置和结果完整、失败病例为 0；
2. sampler-only 与 multiscale 候选完成预先声明的消融；
3. 选定唯一 V3 配置，写入 `config_frozen_v3.json`；
4. 对核心源码和冻结配置写 SHA256 manifest；
5. 之后才把 `method_frozen` 改为 `true` 并建立一次性 test35 配置。

冻结配置 SHA256：
`939394978113badeaecebbc2c1131df4dcf5e157b1ae236adaa211d742bdeaae`。

train138 × 600 的 frozen V3 raw R² mean 为 `0.8929`；blind test35 × 1200 为
`0.8957`，相对 V1/V2 分别提升 `+0.0371/+0.0282`。完整指标、成对统计和结论见
`RESULTS.md`。blind 结果 SHA256 已写入 `freeze_manifest.json`，后续不得基于 test35
修改 V3；任何新方法变化必须使用新版本名和新的 train-only 选择流程。

# V5.2 联合周期实验：执行基础设施只读审计

- 审计日期：2026-09-29 17:20–17:29 +0800。
- 授权范围：单seed、INDv52开发split；四种结构 I/J0/J1c/J1，共 **6个实际单GPU模型训练**：Iu、Ip、Iw、J0、J1c、J1，全部 seed=1234。
- 审计员未启动、终止、修改任何作业；只读系统/代码检查。本文件是唯一由审计员写入的实验文件。
- 读取规则：`.cursor/skills/node04-experiments/SKILL.md`、`.cursor/rules/cluster.mdc`、`gpu.mdc`、`wss.mdc`。`.agents/skills`是指向`.cursor/skills`的链接，技能文本的`.agents/rules`相对引用在当前目录不存在，实际规则位于`.cursor/rules`。

## 结论与推荐资源

**master的GPU分区可用，应通过真实Slurm提交；没有采用SSH直启/伪造Slurm环境的必要。node04当前不可用。**

建议两种真实Slurm方式任选：

1. 6个单GPU数组任务，`--array=0-5%4 --partition=GPU --gres=gpu:4090:1 --cpus-per-task=8 --mem=32G`。每任务只训练一个模型；并发上限4，后两臂由调度器补位。优点是资源随单臂结束释放、无需自管GPU子进程。
2. 1个4GPU队列allocation：`--partition=GPU --gres=gpu:4090:4 --cpus-per-task=32 --mem=128G`，内部每卡严格1个训练槽，6臂动态分两批。仅使用Slurm所分配的GPU，不能因为物理编号空闲而使用allocation外设备。

主节点资源足以容纳**4并发×(8CPU、32GB RAM)**。这是初始申请预算，不是已验证的模型峰值内存；运行前短测仍应记录GPU/CPU峰值。

数据处理如需独立CPU批量任务，应使用`--partition=CPU --nodelist=node03 --cpus-per-task=4`或`8 --mem=32G`。node03现场有44个未分配CPU，4/8CPU资源请求可合理提交；是否立即运行由调度器决定，不影响既有CFD或他人任务。

## 1. 身份、共享目录与实际宿主环境

实际命令：

```bash
hostname
id
stat -c '%u:%g %a %n' /public/newhome/cy/Digital_twin/GNN /public/newhome/cy/.conda/envs/GNN/bin/python
ps -p 1 -o pid,user,args
cat /proc/1/cgroup
cat /proc/self/mountinfo
```

结果：

- hostname：`master`。
- `uid=1006(cy) gid=1007(cy) groups=1007(cy)`，通过身份Gate。
- 项目及GNN Python路径均由1006:1007所有，可读/可执行。
- 进程1为`/sbin/init`；根文件系统为`/dev/nvme0n1p2`上的ext4，无overlay根。
- 本机有真实`munge`、`/public/slurm/sbin/slurmctld`、`slurmd`、`slurmdbd`进程。
- 因此先前客户端`command not found`是PATH问题，不是没有Slurm或必须直启。

## 2. Slurm客户端路径与现场调度

**正确绝对路径**：

```text
/public/slurm/bin/sbatch
/public/slurm/bin/sinfo
/public/slurm/bin/squeue
/public/slurm/bin/scontrol
```

上述文件可执行。`/opt/slurm`只有`etc`配置，没有`bin`。`/etc/profile.d/slurm.sh`中记录了`/public/slurm/bin`，但当前exec shell PATH未包含；launcher或提交命令直接用绝对路径即可，无需改系统或用户环境。

执行：

```bash
/public/slurm/bin/sinfo -N -o '%N %P %t %G'
/public/slurm/bin/scontrol show node master
/public/slurm/bin/scontrol show partition GPU
/public/slurm/bin/squeue -o '%.18i %.10u %.24j %.10P %.2t %.10M %.6C %.10m %R'
```

17:24–17:28输出摘要：

| 节点/分区 | 状态 | CPU/GRES | 备注 |
|---|---|---|---|
| master / GPU | IDLE；分区UP | 192CPU，CPUAlloc=0；`gpu:4090:4` | GPU分区只包含master；OverSubscribe=NO |
| node03 / CPU | MIXED | CPUAlloc=148/192；无GPU | 44CPU尚未分配 |
| node04 | DOWN+NOT_RESPONDING | `Gres=(null)`，不在当前GPU/CPU分区列表 | 不能启动 |

全局队列没有GPU任务；仅CPU任务：cy的16141(node01)、16142(node02)、16143(node03)，以及他人15527(node02)、15550(node03)。审计未变更这些作业。

## 3. GPU物理现场

17:20 master首次`nvidia-smi`：4卡均5MiB、0%利用率，无compute PID。

| index | GPU UUID | 型号 | 显存 |
|---:|---|---|---:|
| 0 | GPU-8a88fadc-4ef8-892b-83ee-c0cbc7fa6979 | NVIDIA GeForce RTX4090 | 24564MiB |
| 1 | GPU-99ad58be-5459-6d50-0383-001595b234ab | NVIDIA GeForce RTX4090 | 24564MiB |
| 2 | GPU-550c33a9-610b-c160-ab40-b360e328bd6e | NVIDIA GeForce RTX4090 | 24564MiB |
| 3 | GPU-d0b318d1-2854-e739-dce2-ba709b7f99a8 | NVIDIA GeForce RTX4090 | 24564MiB |

17:28 GPU0出现620MiB占用，已只读核验为本次协作模型测试：PID148857、用户cy、命令`/public/newhome/cy/.conda/envs/GNN/bin/python -m pytest -q training_wss_min/tests/test_joint_cycle_model.py`。其他3卡约8MiB无compute PID。此短测不在Slurm内，已通知主执行者等待其自行结束；未终止。正式提交前重新检查，不能把“Slurm IDLE”当作不存在未调度进程。

## 4. CPU内存与node03资源

master `free -h`：

```text
total 377Gi; used 18Gi; free 103Gi; buff/cache 255Gi; available 356Gi
swap 99Gi; used 0B
```

`MemAvailable`足够128GB并发预算；Slurm的`FreeMem`约116GB是空闲而非可回收后的可用量，不应据此误判128GB无法申请。

node03 `scontrol show node node03`：

```text
CPUTot=192; CPUAlloc=148; CPULoad=111.89
RealMemory=350000; AllocMem=0; FreeMem=233727
State=MIXED; Partitions=CPU
```

node03现有任务16143(cy, Fluent_cfdauto,92CPU)、15550(他人,pysr_joint_potential,56CPU)，两作业`MIN_MEMORY=0`，故`AllocMem=0`不代表它们没用RAM。233727MB现场FreeMem给32GB小数据任务留有充足空间，但仍应通过Slurm请求而非SSH直接跑CPU批处理。

## 5. node04 Gate

```bash
ssh -o BatchMode=yes -o ConnectTimeout=10 cy@node04 'hostname; id; ...'
/public/slurm/bin/scontrol show node node04
```

SSH返回`No route to host`（/etc/hosts将node04解析为192.168.2.5）。Slurm确认：`DOWN+NOT_RESPONDING`、`Gres=(null)`、`Reason=Not responding [slurm@2026-08-02T10:40:02]`，无可用分区。**本轮不能远端核验UID或2×A100物理占用，不能照历史快照宣称其卡空闲。** 即使旧手册有node04无GRES时直启先例，本次DOWN/不可达条件不允许采用它。

## 6. 既有执行习惯及新launcher注意点

参考：

- `training_wss_min/cluster/run_cycle_queue.slurm`：真实4×4090 allocation、冻结源码目录执行、public配置/结果路径。
- `training_wss_min/tools/run_wss_local_wave1_queue.py`：源码/配置/协议指纹，运行前及每次launch复核；独立日志、状态、拒绝覆盖；但它固定调用旧`training_wss_min.train/evaluate`且默认best/last评估，不适合不经适配直接启动新联合周期模型。
- 冻结目录`GNN_cycle_frozen_20260929/training_wss_min/tools/run_cycle_field_queue.py`：轻量周期队列，按`matrix.json`排队、从分配环境取GPU、已有`metrics.json`即跳过。**此runner没有源码/配置指纹Gate，日志以w覆盖，metrics存在也未核验配置一致；新矩阵不能把这些当足够的恢复协议。**
- `node04_run_matrix.sh`有历史手工伪Slurm ID行为，本次有真实Slurm，**不得复用其伪造环境做法**。

本次最小launcher合同：

1. 先准备独立源码冻结目录及SHA256清单，含新模型/训练入口、数据读取、配置和依赖；训练从冻结目录导入，输出仍在public新实验目录。新文件未进入旧冻结树，应显式纳入。
2. manifest必须明确6训练而非4训练：I结构是Iu/Ip/Iw三次独立训练；J0/J1c/J1各一次；seed1234、同INDv52 split、同数据口径、相同预注册选模规则。
3. IND桥接split原路径：`/public/newhome/cy/Digital_twin/GNN/data_wss_v5/views_v5_2_full_20260923/wss_min_view_v1/split_v52_ind_train170_test91.json`；记录其SHA256以及实际患者分组检查。不要随机重新划分，也不要因文件名test自动把开发IND称独立确认集。
4. 单卡数组任务直接使用Slurm给定`CUDA_VISIBLE_DEVICES`，训练内部device=cuda:0；不要硬编码物理GPU0。如果采用4卡queue，父进程保留调度环境，只将已分配设备子集传给各worker，每GPU同时最多1模型。
5. 使用`/public/newhome/cy/.conda/envs/GNN/bin/python -u`；线程默认OMP=2、MKL/OPENBLAS=1，DataLoader worker不超过每模型8CPU预算；不要把全部192CPU交给每worker。
6. 每臂保存started/failed/complete时间、hostname、真实Slurm Job/Array ID、PID、GPU UUID、配置路径/hash、run/log路径；日志覆盖至少epoch与病例读取/缓存进度。
7. 新run目录非空先核验，不以随意覆盖或单个metrics文件存在决定resume；退出码与必需checkpoint/metrics/config指纹共同决定完成。失败单臂保留证据，queue总状态不能宣称全部成功。
8. 脚本`set -euo pipefail`，运行前检查UID/public/env和manifest；GPU预检放进真实allocation；不在launcher里自动扩大seed、CV或额外测试集。候选间比较应使用同一正式评价合同，不直接继承旧runner的best/last双评。

建议提交形式（训练module名称以实际实现为准，不假装当前已存在入口）：

```bash
/public/slurm/bin/sbatch -p GPU --gres=gpu:4090:1 \
  --cpus-per-task=8 --mem=32G --array=0-5%4 <reviewed_array_launcher.slurm>
```

若数据准备需要单独job：

```bash
/public/slurm/bin/sbatch -p CPU -w node03 --cpus-per-task=8 --mem=32G <reviewed_data_launcher.slurm>
```

资源状态为本次快照，正式提交由主执行者最后重查；本审计没有发出任何sbatch/scancel。

# 集群 node04 使用要点

> 更新：2026-08-02；§7 训练矩阵与两机并行补记：2026-10-02
>
> 给人或 Agent：用户说「去 node04 跑实验」时按本文操作。
>
> Agent skill：`.cursor/skills/node04-experiments/SKILL.md`

## 1. 节点角色

| 节点 | 典型用途 |
| --- | --- |
| master / 登录节点 | 提交作业、轻量检查；勿跑重计算 |
| **node03** | CPU 批处理（预处理、审计等），Slurm 常钉 `-w node03` |
| **node04** | GPU 实验 / 冒烟 / 调试；**2× NVIDIA A100-PCIE-40GB（40GB）** |

`/public` 经 NFS 挂载（来自 master 侧导出）。代码、数据、conda 环境以
`/public/newhome/cy/...` 为准。

## 2. 账号与权限（曾踩坑）

NFS 按**数字 UID**鉴权。node04 上 `cy` 必须与 master 一致：

- 期望：`uid=1006(cy) gid=1007(cy)`
- 若为其他 UID（历史上曾出现 1001），即使用户名同为 `cy`，也会无法进入
  `/public/newhome/cy`（目录权限 `700`），从而看不到 `data_new`、项目代码和
  `~/.conda/envs/GNN*`。

上机后先执行：

```bash
id
ls /public/newhome/cy/Digital_twin/GNN/data_new | head
```

不对齐时：**不要**在 node04 本地重装整套环境当正式方案；应请管理员改 UID/GID。

## 3. HOME 与工作目录

SSH 登录 node04 后，默认 `HOME` 往往是**本地** `/home/cy`，不是
`/public/newhome/cy`。

```bash
# ❌ 错误：在本地 home 下找项目
cd ~/Digital_twin/GNN

# ✅ 正确
cd /public/newhome/cy/Digital_twin/GNN
```

实验输出应写到仓库下的 `outputs/` 等 public 路径，避免落到仅本机可见的 `/home/cy`。

## 4. Conda 环境

共享 conda 根：`/public/newapps/anaconda3`

个人环境（在 public home 下）：

| 环境 | 绝对路径 | 用途 |
| --- | --- | --- |
| GNN | `/public/newhome/cy/.conda/envs/GNN` | 训练、验证、CFD WSS 等 |
| GNN_vmtk | `/public/newhome/cy/.conda/envs/GNN_vmtk` | vmtk / 相关预处理 |

```bash
source /public/newapps/anaconda3/etc/profile.d/conda.sh
conda activate /public/newhome/cy/.conda/envs/GNN
```

说明：默认登录 PATH 可能没有 `conda`；`conda env list` 也不一定列出名为 `GNN`
的短名。优先用**绝对路径激活**，或直接调用
`/public/newhome/cy/.conda/envs/GNN/bin/python`。

## 5. 推荐启动序列

```bash
ssh cy@node04
cd /public/newhome/cy/Digital_twin/GNN
source /public/newapps/anaconda3/etc/profile.d/conda.sh
conda activate /public/newhome/cy/.conda/envs/GNN
nvidia-smi
export CUDA_VISIBLE_DEVICES=0   # 按占用选择 0 或 1
```

## 6. 最小冒烟（2026-08-02 已通过）

```bash
PY=/public/newhome/cy/.conda/envs/GNN/bin/python

# GPU
$PY -c "import torch; print(torch.__version__, torch.cuda.is_available()); x=torch.randn(2048,2048,device='cuda'); y=x@x; torch.cuda.synchronize(); print('matmul_ok')"

# 数据 + velocity→WSS 单病例
cd /public/newhome/cy/Digital_twin/GNN/wss_mri_calculator/src
$PY calculate_wss_cfd.py \
  --case-dir /public/newhome/cy/Digital_twin/GNN/data_new/AG/slow/LIU_JIN_LIANG
```

当时结果摘要：`truth_usable=true`，`raw_r2≈0.828`，Spearman≈0.981。

## 7. 跑训练矩阵（2026-09 起的做法）

node04 不在 Slurm 分区里，ssh 上去在 Slurm 之外跑；记录里要如实写「Slurm 之外执行」，不要伪造作业号。

- **用谁的卡**：09-30 用户裁定 node04 的 A100 留给全周期时间建模线，主线默认放 master 4090；其他线要用 node04 先征得用户同意（例：10-02 体场 PF6 / VF6 复验经用户允许用了两张 A100 和 node04 的 CPU，当时 cfd_auto 占满 CPU 分区）。
- **启动方式**：`ssh node04 "cd <仓库或冻结副本> && nohup bash <驱动脚本> > <日志> 2>&1 < /dev/null &"`，本地 ssh 套 `timeout 20`（否则会挂住）。代码在冻结副本里跑，产物写主树的 `runs/` 与 `experiments/`。
- **标准矩阵**：`training_wss_min/cluster/node04_run_matrix.sh`（预检 + 队列一起跑；工具要求纯数字 `SLURM_JOB_ID`，脚本手工设 `9MMDDHHMM`）；或直接 `python -m training_wss_min.tools.run_local_train_queue --config-dir <configs/实验> --gpus 0,1 --slots-per-gpu 1 --only <臂列表> --status-file <状态文件名>`（可续跑，拒绝覆盖未完成的 run）。
- **与 master 共用一个实验目录**（10-02 体场复验的做法）：master 用 Slurm 队列 `run_wss_local_wave1_queue --matrix matrix_master.json`（它独占 `.queue.lock` 与 `queue_status.json`），node04 用 `run_local_train_queue --only <node04 的臂> --status-file queue_status_node04.json`；node04 的驱动先等 Slurm 预检写出 `runtime_preflight.json` 通过，再调用 `run_wss_local_wave1_queue.verify_preflight` 复核同一套指纹；收尾作业轮询 node04 状态文件的 `finished_at` 后再做评估。拆分用的子矩阵文件名里不能有 `_s`（会被 `*_s*.json` 当成臂配置）。示例：`training_wss_min/cluster/node04_pf6vf6_v52d_queue.sh`、`post_pf6vf6_v52d.slurm`。
- **速度与槽数**：node04 空闲时，PF6 每轮一卡一臂 A100 2.2 s、4090 2.25 s（同速）；一卡两臂总吞吐只多 8 %（A100）/ 15 %（4090），单臂慢约 1.8 倍，默认每卡一臂。与他人进程分时时 A100 会明显变慢（09-24 X5D 三槽约 130 s/轮）。A100 与 4090 训练的臂只在 seed 噪声意义上可比，评估差约 1e-4。
- **停机风险**：node04 曾在 09-29 下午停机（下一次开机 22:07），当时在跑的剂量对照 ×1.5 三臂中断在第 392–397 / 400 轮；长队列要能续跑，状态文件里没有 `finished_at` 就要去查。

## 8. 相关规范

- `.cursor/rules/cluster.mdc` — 集群提交与 node03
- `.cursor/rules/gpu.mdc` — GPU 占用检查（在目标节点上执行）
- `.cursor/rules/conda.mdc` — GNN / GNN_vmtk 选型
- `wss_mri_calculator/README_CFD_ADAPTATION.md` — CFD WSS 快速命令

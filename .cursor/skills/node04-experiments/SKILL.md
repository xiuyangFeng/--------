---
name: node04-experiments
description: >-
  在 GNN 项目的 node04 上检查环境、运行已授权的实验或冒烟、恢复与跟踪作业。
  用户指定 node04/04 节点，或任务涉及该节点的 public 路径、UID、conda、GPU/Slurm 时使用。
---

# node04 实验

项目位于 `/public/newhome/cy/Digital_twin/GNN`。node04 的默认 HOME 可能是本地
`/home/cy`；项目、数据、环境和正式产物始终使用 `/public/newhome/cy/...`。
历史故障背景见 [集群 node04 使用要点](../../../docs/00-规范与记录/集群node04使用要点.md)，
该文档的硬件与调度状态是历史快照，运行决策以下面的现场检查为准。

## 接入与资源检查

在 node04 上检查身份、共享路径与 GPU；在有 Slurm 客户端的提交节点检查调度注册：

```bash
ssh cy@node04 '
set -e
id
test -r /public/newhome/cy/Digital_twin/GNN/README.md
test -x /public/newhome/cy/.conda/envs/GNN/bin/python
nvidia-smi --query-gpu=index,uuid,name,memory.used,memory.total --format=csv
nvidia-smi --query-compute-apps=gpu_uuid,pid,process_name,used_memory --format=csv
'
scontrol show node node04
sinfo -N -n node04 -o '%N %P %t %G'
squeue -u cy
```

- **身份 Gate**：期望 `uid=1006(cy) gid=1007(cy)`，与共享目录所有者一致。
  UID/GID 不符或 public 不可访问时，停止依赖该目录的执行并报告具体失败；不要在本地
  HOME 重装环境来绕过 NFS 身份问题。
- **资源 Gate**：GPU 数量、UUID、显存和占用以 node04 当场结果为准；历史为 2×A100
  40GB，不得照抄 master 的卡号或按旧文档假定 node04 当前没有 GPU GRES。
- **调度选择**：核对 node04 所属分区、状态和 GRES，支持 GPU 分配时优先 Slurm。
  用户已指定 node04 执行、而现场确认该节点未注册 GPU GRES 或无对应 GPU 分区时，
  可按既有 node04 直启方式使用空闲物理卡；记录该限制及实际启动方式。可调度但繁忙
  时正常排队，不能以直启绕过维护/停用状态。CPU 批量任务默认仍提交 node03。

## 环境与启动

按任务依赖选环境：训练、评估和普通点云/CFD 工具用 `GNN`；VMTK 几何提取用
`GNN_vmtk`。用户已指定环境时直接沿用；已授权任务不再为激活环境、使用空卡或按计划
排队重复询问。只运行本次任务需要的脚本，不把 WSS 对比附加到普通设备检查中。

```bash
cd /public/newhome/cy/Digital_twin/GNN
source /public/newapps/anaconda3/etc/profile.d/conda.sh
conda activate /public/newhome/cy/.conda/envs/GNN
```

非交互任务可直接调用 `/public/newhome/cy/.conda/envs/GNN/bin/python`，几何任务换成
`GNN_vmtk/bin/python`。需要验证 GPU 计算时，在选定空闲卡上做一个小型 Torch 运算即可；
只有任务涉及 CFD WSS 时才追加相应数据冒烟。

- Slurm 作业使用分配给作业的设备，不覆盖调度器设置的 `CUDA_VISIBLE_DEVICES`。
  SSH 直启时显式设置目标卡，启动后核验实际 GPU UUID、PID 与命令对应。
- 已授权的训练/评估按配置继续执行；无空卡时按计划排队。直启方式须等待满足显存需求
  的空卡，不因显存剩余就侵占已有任务；新矩阵或额外评估不能由节点检查自动扩展出来。
- 重启前检查现有 job/PID、run 状态与 checkpoint，避免重复启动。仅按该 run 的恢复
  协议 resume；需要全新重跑时使用独立目录，保留旧产物，不覆盖失败证据。

## 可追踪交付

沿用该实验族的 submission/queue manifest 与日志格式，至少留下启动时间、主机、
GPU UUID、Job ID 或 PID、配置路径、run 路径和日志路径。后台进程已启动只表示
`submitted/running`；用日志推进、进程/队列状态与预期完成产物共同判断完成或失败。
日志须能看到病例或训练阶段的进度；接续已有任务时从这些证据恢复，不能凭旧 PID 判断仍在跑。

集群通则见 [cluster.mdc](../../rules/cluster.mdc)，GPU 调度见
[gpu.mdc](../../rules/gpu.mdc)。WSS 的执行范围按当前任务与既有授权判断，见
[wss.mdc](../../rules/wss.mdc)；节点技能本身不授权训练、批量评估或取消其他作业。

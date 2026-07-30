# wss_pinn — Agent 指令

本目录是 **WSS-PINN 独立实验线**。最终任务仍是峰值壁面 WSS 预测；
\(u,v,w,p\) 只作为训练期的体域辅助场和物理约束变量。

## 隔离边界

- `data_new/`、`data_wss_min/`、`pipeline_wss_min/`、`training_wss_min/`
  和既有 checkpoint/config/run 一律视为只读上游。
- 不在旧目录中添加 PINN 专用字段、loss、配置或作业脚本。
- PINN 派生数据只写 `data_wss_pinn/`。
- PINN 训练与评估结果只写 `outputs/wss_pinn/`。
- PINN 代码、配置、测试和集群入口只写 `wss_pinn/`。
- 任何复用旧模型的实现都必须经过本目录 adapter，并记录父配置、父 checkpoint、
  代码版本和 SHA256；不得依赖含义不明的 `latest`。

## 实验顺序

必须按以下 Gate 逐级推进：

1. P0 数据身份、单位、velocity→WSS 和 CFD residual Oracle；
2. P1 独立 physics sidecar；
3. F0-U 速度场 data-only；
4. F0-UP 压力场 data-only；
5. F1 continuity + no-slip；
6. F2 \(WSS_{\mathrm{phys}}\) 一致性；
7. F3 非定常 momentum；
8. C1 开发筛选与 C2 确认。

上一级未通过时，不得把后一级批量训练写成主路线。一次只新增一个信息源或物理项。

## 记录责任

- 路线总入口：
  `docs/02-推进与变更/WSS_PINN/README.md`
- 实验族、单次运行和 Go/No-Go：
  `docs/02-推进与变更/WSS_PINN/WSS_PINN_阶梯实验矩阵与进度跟踪.md`
- 第一性原理与物理依据：
  `docs/02-推进与变更/WSS最小化_体域物理约束与PINN训练路线_2026-07-29.md`
- WSS 专用推进记录：
  `docs/02-推进与变更/WSS最小化_代码修改与实验推进记录.md`

修改本目录代码、配置、数据合同、QA、实验文档或作业脚本后，必须更新 WSS 专用推进记录。
每条至少包含：**本次主要修改**、**对应代码/文档**、**推进到实验步骤**、**当前状态判断**。

## 硬门禁

- `random5000` 仍是 W0 的壁面 support，不是 PINN 的全部 collocation 点。
- physics batch 必须显式区分 wall、near-wall 和 core；无 WSS 标签的体点不得进入直接 WSS loss。
- 主物理口径固定为非滑移刚性壁面、不可压缩、\(\rho=1060\ \mathrm{kg/m^3}\)、
  Carreau–Yasuda 型非牛顿流变。
- constant-\(\mu\) 只能作为命名明确的消融。
- 没有通过 velocity→WSS Oracle 时，不得把 \(WSS_{\mathrm{phys}}\) 误差归因于网络。
- 没有通过 CFD residual Oracle 时，不得启用对应 momentum residual。
- `test36` 必须标记为 `reused_development_screen`，不得写成新的独立测试确认。
- 不提交训练作业，除非对应阶段的静态审计、数据 Gate 和单元测试已通过。

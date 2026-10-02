# GNN Digital Twin · Agent 指令（仓库根）

> 2026-10-02 建。只放硬规则与入口，不写实验流水账。路线细节在各块文档；子目录 `AGENTS.md`（`pipeline_wss_min/`、`wss_pinn/`）的科学合同按其路线适用。`CLAUDE.md` 导入本文件；Cursor 规则在 `.cursor/rules/`。

## 先看哪里

- `docs/README.md` → `docs/02-推进与变更/README.md`（V5 各块导航）→ 所属块 README 的「当前状态」与实验跟踪最新节 → 推进记录文首 3–5 条。
- 文档与代码 / 产物冲突时**以产物为准**（run 目录与完成标志、队列状态文件、调度器、进程与服务、git），并把文档改对。

## 多个 AI 会话并行（必须遵守）

本仓库常同时开着多个 AI 会话（Claude Code / Cursor / codex）做不同课题。完整规则在 [`docs/00-规范与记录/项目知识库整理规范.md` §6](docs/00-规范与记录/项目知识库整理规范.md)；做法细节见 skill `lab-docs-engineering`（`parallel-sessions.md`、`sync-audit.md`）。

1. **开工检查**：读所属块 README 与跟踪最新节、推进记录文首；跨文档事实（数据版本、训练底座、部署发布包、运行中的作业……）从规范 §6.5 登记表的**规范位置**读，不从副本读。看一眼 `git status` 和最近改过的文档，知道别人在动什么。
2. **跨线影响告知**：占用共享资源（node04、共享缓存、部署服务等，见规范 §6.6），或新产物依赖另一条线的文件时，在受影响块的 README 追加带日期的告知；释放后改为「已释放」。
3. **收尾同步**（任务完成或停下时必做）：本块跟踪 → 本块 README「当前状态」→ 改变了的跨文档事实（先规范位置，再各副本）→ 推进记录文首一条 → 用户本轮的决定写进状态处 → 撤销告知。
4. **编辑礼仪**：定点替换，不整文件重写；别人的块只追加带日期、带依据的补记；不回滚别人的改动；已结束的小节、用户裁定原文、带日期的报告只在其后补「后续状态（日期补记）」。
5. 用户说「把文档更新到最新 / 全量同步」：按 skill 的 `sync-audit.md` 做（先建事实表，活文档改到现状，快照不改正文，逐条核实，汇报发现的错误）。

## 推进记录

- V5 / WSS 各线（`training_wss_min/`、`wss_v5/`、`wss_deploy/`、`cfd_auto/`、数据）→ `docs/02-推进与变更/WSS最小化_代码修改与实验推进记录.md` 文首；V3 历史 / 通用 → `docs/02-推进与变更/代码修改与实验推进记录.md`。
- 标题 `## YYYY-MM-DD｜主题 · 动作 · 状态`；字段：本次主要修改 / 对应代码/文档 / 推进到实验步骤 / 当前状态判断。状态用绝对词，不写「今天 / 最近」。
- 实验结果写所属块的实验跟踪（节号全局递增，计数见 `docs/02-推进与变更/README.md`「写到哪里」）；推进记录可只写指针条目。

## 环境

- Python：`/public/newhome/cy/.conda/envs/GNN/bin/python`。Slurm 命令先 `export PATH=/public/slurm/bin:$PATH`；GPU 作业必须带 `--gres`。
- master 系统时间是美西时间，文档统一写北京时间。node04 不走 Slurm，本地 ssh 套 `timeout 20`，要点见 `docs/00-规范与记录/集群node04使用要点.md`。
- Markdown 不用 raw HTML 注释和引用式链接定义（渲染器不支持）。

## 默认不做（除非用户在当前对话明确要求）

- `git commit` / `push`；把文档搬进 `_archive/`（归档按规范与 skill 的清单）。
- 移动或删除 `data*`、`outputs/`、模型权重等大数据与证据目录。
- 往 v5.2d 主视图根 `data_wss_v5/views_v5_2d_20261001/wss_min_view_v1/` 补 `volume.npz`（全周期缓存会签名失配）；删除或重建 `training_wss_min/experiments/joint_cycle_v52d_20261001/data_audit_cache/derived_volume/`（体场数据根链接着它）。
- 未经用户同意占用 node04 的 A100（09-30 起留给时间建模线）。
- 跨路线 / 跨数据版本 / 跨评估集裸比指标；编造未跑实验的数字。

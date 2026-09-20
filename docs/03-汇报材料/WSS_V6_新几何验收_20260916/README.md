# 几何特征覆盖优化与确认结果（2026-09-16）

**170 例保留；新增人工待办 0 例，两例首次隔离位置已按用户确认落实。** [交互审查台](index.html) · [病例动作](case_actions.csv) · [逐字段覆盖](field_coverage.csv) · [两例确认记录](remaining_bifurcation_review.html)

本轮以 v3 候选为起点（作业 14401），参考有效截面 38,270 → 38,662；点云 34,884 → 34,988，共 50,470 个站点。
实际新增恢复参考 674 站、点云 340 站；更严格检查同时屏蔽原有效参考 282 站、点云 236 站。恢复与屏蔽分开计数。

算法以连续中心线边与切平面求交，减少离散点接近切面的误判；缺失参考站经重选切面、独立倾角与最终沿程位移复核。点云增加仅由点构建的轮廓候选，再验证参考一致性、所属分支和减半密度稳定性。

所有病例使用相同几何字段，并按每个字段各自的有效 mask 表示可用范围。截面积有效不自动代表坡度或上下游信息有效；派生字段不跨无效区间。保留病例及其全部原有监督，缺少几何特征不删除病例或监督标签。

旧的整例低覆盖禁用建议已撤回。参考覆盖 60% / 点云 50% 只作诊断提示，不决定病例去留，也不要求用户逐站放行。按最终 manifest 重新计算的低覆盖诊断病例为 7 例：AAA/ruputer/FENG_LI_XIN、AAA/ruputer/GAO_FENG_SHAN、AAA/unruputer/SUN_SHU_MING、AAA/unruputer/ZHU_ZI_HAI、AG/fast/CHEN_SHI_MING、AG/fast/FAN_JIAN_MING、AG/slow/ZHANG_JING_SHUN。

两例用户确认仅涉及根部首次隔离事件：LIU_XING_GUO 71.00 mm、SUN_SHU_MING 48.75 mm；s 从旧根图节点沿扩展路径起算，精确坐标、原话及原算法分歧保存在确认侧车。训练许可仍为 false。

位移不一致可能来自真实急变或提取误差；保留缺失不等于认定源解剖违规。独立审计验证机械几何一致性，不代替临床解剖批准。该候选尚未接入训练，保留监督是本轮候选使用约定。

候选：`outputs/wss_v6_geometry_coverage_20260916_final_v2`；基线：`/public/newhome/cy/Digital_twin/GNN/outputs/wss_v6_geometry_candidate_20260915_opt_v3_final`。来源、参数、作业与病例名单见 `coverage_manifest.json`，逐站依据见各例 `coverage.json`；独立审计 `audit_coverage_geometry.json` 已核对相同病例集合。

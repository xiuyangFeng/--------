"""Build a conservative, read-only triage from the V6 geometry screen.

The volume dV/ds and Rmis ratios are retained as diagnostics.  They are not
treated as acceptance gates because atlas s is a projected coordinate and
Rmis is a local radius proxy.  Only invariant/topology/coverage checks can
automatically clear a case here.
"""
from pathlib import Path
import json
import pandas as pd

ROOT = Path(__file__).resolve().parents[5]
BASE = ROOT / "docs/03-汇报材料/WSS_V6_新几何验收_20260909/自动筛查_20260915"
OUT = BASE / "深化审查_20260915"
EXCLUDED = {"AAA/unruputer/LIU_WEN_QI", "AG/slow/HOU_SHEN_QIAN"}

def main():
    OUT.mkdir(parents=True, exist_ok=True)
    d = pd.read_csv(BASE / "screen_per_case.csv")
    d = d[~d.canonical_id.isin(EXCLUDED)].copy()
    # Strict invariants already independently recomputed by screen_geometry_candidate.
    d["strict_l1_pass"] = (
        (d.L1_ref_area_relerr_max <= 1e-4) & (d.L1_pc_area_relerr_max <= 1e-4) &
        (d.L1_ref_valid_reason_mismatch == 0) & (d.L1_pc_valid_reason_mismatch == 0) &
        (d.L1_wallmap_area_relerr_p99 <= 1e-3) &
        (d.openings_count == 5) & (d.openings_valid == 5) &
        (d.L1_opening_vs_fluent_area_relerr_max <= .02) &
        (d.L1_inlet_ref_vs_fluent_relerr <= .02)
    )
    d["structure_pass"] = d.strict_l1_pass & d.topology_valid & d.aortoiliac_pattern
    # Coverage is a review policy, not a claim that an invalid contour is false.
    d["coverage_pass"] = (d.ref_valid_frac >= .60) & (d.pc_valid_frac >= .50)
    d["source_agreement_pass"] = (d.L2_wall_segment_agree_v5 >= 1.0) & (d.L2_wall_segment_disagree_count == 0)
    d["auto_clear"] = d.structure_pass & d.coverage_pass & d.source_agreement_pass
    d["needs_user_geometry"] = ~d.auto_clear
    # Human-only semantics from the independent audit. These are deliberately
    # kept separate from weak numerical diagnostics.
    ind = json.loads((OUT / "independent/geometry_compliance_audit.json").read_text())
    human = set()
    for item in ind["user_determinations_remaining"].values():
        for x in item.get("cases", []) or []:
            human.add(str(x).split(" (")[0])
    # Source-integrity review list from the published 172-case audit.
    human.update({
      "AAA/ruputer/LIU_YONG_LAN", "AAA/ruputer/LI_BING_JIANG", "AAA/ruputer/WANG_KUI_WU",
      "AAA/ruputer/ZHOU_KE_XUN", "AAA/unruputer/GUO_BAO_CHUN", "AAA/unruputer/LIN_JIAN_RONG",
      "AAA/unruputer/LIU_JIE", "AAA/unruputer/MA_JIN_HE", "AAA/unruputer/WANG_MAN_TIAN",
      "AAA/unruputer/YANG_BEN_RUI", "AG/slow/HE_SHU_ZHEN", "AG/slow/LIU_FENG",
      "AG/slow/LI_CHONG_ZENG", "AG/slow/LI_GUI_YING", "ILO/GUO_YU_SHU-0/before",
      "ILO/WANG_XIAO_LIN-0/before", "AAA/ruputer/GAO_FENG_SHAN"})
    d["human_semantic_review"] = d.canonical_id.isin(human)
    d["needs_user_geometry"] |= d.human_semantic_review
    d["auto_clear"] &= ~d.human_semantic_review
    # Diagnostic burden is reported separately; it never changes auto_clear.
    weak = ((d.L2_ref_vs_volume_n_over20pct > 0) | (d.L2_pc_vs_volume_n_over20pct > 0) |
            (d.L2_ref_adjacent_jump_count > 0) | (d.L1_ref_nonstar_from_center > 0) |
            (d.L2_pc_vs_ref_n_over20pct > 0))
    d["diagnostic_flags"] = weak.astype(int)
    cols = ["canonical_id", "auto_clear", "needs_user_geometry", "strict_l1_pass", "structure_pass",
            "coverage_pass", "source_agreement_pass", "human_semantic_review", "diagnostic_flags", "ref_valid_frac", "pc_valid_frac",
            "L2_ref_vs_volume_n_over20pct", "L2_pc_vs_volume_n_over20pct", "L2_ref_adjacent_jump_count",
            "L1_ref_nonstar_from_center", "L2_pc_vs_ref_n_over20pct", "bif_undetermined",
            "cia_left_length_mm", "cia_right_length_mm", "cia_left_tortuosity", "cia_right_tortuosity"]
    d[cols].sort_values(["auto_clear", "diagnostic_flags", "canonical_id"], ascending=[True, False, True]).to_csv(OUT / "case_triage.csv", index=False)
    summary = {"schema":"v6_geometry_deep_triage_v1", "source":"screen_per_case.csv", "excluded":sorted(EXCLUDED),
      "n_input":172, "n_reviewed":len(d), "n_auto_clear":int(d.auto_clear.sum()), "n_user_geometry":int(d.needs_user_geometry.sum()),
      "n_with_weak_diagnostics":int(d.diagnostic_flags.sum()), "n_human_semantic_review":int(d.human_semantic_review.sum()),
      "method":{"auto_clear":"strict L1 invariants + topology + five openings + coverage + V5/V6 wall segment agreement",
                "diagnostic_only":["dV/ds area ratio","A/(pi Rmis^2)","adjacent area jump","nonstar contour","pointcloud/reference area disagreement"],
                "reason":"dV/ds is a projected-atlas volume proxy; Rmis is a local radius proxy; nonstar contours can occur in real aneurysm geometry"},
      "auto_clear_cases":sorted(d.loc[d.auto_clear,"canonical_id"]),
      "user_geometry_cases":sorted(d.loc[d.needs_user_geometry,"canonical_id"]),
      "weak_diagnostic_cases":sorted(d.loc[d.diagnostic_flags,"canonical_id"])}
    (OUT / "case_triage.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2))
    md = ['# V6 几何深化自动分流（2026-09-15）', '',
      '本报告继承 2026-09-15 全队列筛查，排除已解决的两对重复病例中的 `LIU_WEN_QI` 与 `HOU_SHEN_QIAN`，剩余 170 例。',
      '', '## 判定口径', '',
      '- **可自动清除人工几何复核**：L1 独立重算、五个开口、拓扑、有效覆盖和 V5/V6 壁面段归属均通过。',
      '- **仍需你确认**：硬结构/覆盖/来源一致性未通过，或分叉定位、CIA 范围、中心线/源面完整性需要解剖裁定的病例。',
      '- dV/ds、`A/(πRmis²)`、相邻站跳变、非星形和点云/面差异只作沿程诊断排序，不直接判违规。', '',
      f'- 结果：**{int(d.auto_clear.sum())} 例可自动清除，{int(d.needs_user_geometry.sum())} 例需你确认**；其中 {int(d.human_semantic_review.sum())} 例含人工语义/源完整性事项，{int(d.diagnostic_flags.sum())} 例有诊断性弱报警。', '',
      '## 仍需你确认的最小集合', '',
      '`case_triage.csv` 中 `needs_user_geometry=true` 的病例；优先查看 `strict_l1_pass=false`、`structure_pass=false`、`coverage_pass=false` 和 `source_agreement_pass=false` 的原因。', '',
      '## 解释边界', '',
      '“自动清除”只表示机器检查未发现需要人工确认的结构性证据，不等于批准几何进入训练；现有候选的 `training_allowed=false` 保持不变。']
    (OUT / "README.md").write_text('\n'.join(md)+'\n')
    print(summary["n_auto_clear"], summary["n_user_geometry"], summary["n_with_weak_diagnostics"])

if __name__ == '__main__': main()

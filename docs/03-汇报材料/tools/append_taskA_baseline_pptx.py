#!/usr/bin/env python3
"""Append Task A follow-on slides to 任务A基线模型对比汇报_副本.pptx (OOXML, no python-pptx)."""

from __future__ import annotations

import re
import shutil
import zipfile
from pathlib import Path

import pandas as pd

MAT_DIR = Path(__file__).resolve().parents[1]  # .../docs/03-汇报材料
DOCS = MAT_DIR.parent  # .../docs
PPTX = MAT_DIR / "任务A基线模型对比汇报_副本.pptx"
XLSX = DOCS / "00-规范与记录" / "实验记录表.xlsx"


def load_opt_stats():
    df = pd.read_excel(XLSX, sheet_name="taskA_field")
    df["family"] = df["exp_id"].astype(str).str.replace(r"_seed\d+$", "", regex=True)

    def stats(fam: str):
        sub = df[df["family"] == fam]
        if sub.empty:
            return None
        m = sub["RMSE_|v|"].astype(float)
        return m.mean(), m.std(ddof=1) if len(m) > 1 else 0.0

    main_m, main_s = stats("A-Main-01")
    rows = []
    for fam, label in [
        ("A-Main-01", "A-Main-01（基线主线）"),
        ("A-Opt-01", "A-Opt-01 目标权重 tw22205"),
        ("A-Opt-02", "A-Opt-02 Pre-Norm"),
        ("A-Opt-03", "A-Opt-03 tw22205+Pre-Norm"),
        ("A-Opt-04", "A-Opt-04 hidden=256"),
        ("A-Opt-05", "A-Opt-05 +layers=4（战略锚点）"),
        ("A-Opt-07", "A-Opt-07 interior loss boost"),
    ]:
        st = stats(fam)
        if not st:
            continue
        mu, sd = st
        d = 100.0 * (mu - main_m) / main_m if fam != "A-Main-01" else 0.0
        rows.append((label, mu, sd, d))
    return rows, (main_m, main_s)


def strip_pics(slide_xml: str) -> str:
    return re.sub(r"<p:pic>.*?</p:pic>", "", slide_xml, flags=re.DOTALL)


def slide_from_template(template: str, title: str, subtitle: str, footer: str, banner: str) -> str:
    texts = re.findall(r"<a:t>([^<]*)</a:t>", template)
    if len(texts) < 4:
        raise ValueError(f"expected 4 text runs, got {texts}")
    old_t, old_s, old_f, old_b = texts[0], texts[1], texts[2], texts[3]
    xml = template
    xml = xml.replace(f"<a:t>{old_t}</a:t>", f"<a:t>{title}</a:t>", 1)
    xml = xml.replace(f"<a:t>{old_s}</a:t>", f"<a:t>{subtitle}</a:t>", 1)
    xml = xml.replace(f"<a:t>{old_f}</a:t>", f"<a:t>{footer}</a:t>", 1)
    xml = xml.replace(f"<a:t>{old_b}</a:t>", f"<a:t>{banner}</a:t>", 1)
    return xml


def main():
    if not PPTX.is_file():
        raise SystemExit(f"missing {PPTX}")
    if not XLSX.is_file():
        raise SystemExit(f"missing {XLSX}")

    rows, (main_m, main_s) = load_opt_stats()

    def fmt_row(label, mu, sd, d):
        line = f"{label}: {mu:.3f} ± {sd:.3f}"
        if abs(d) > 1e-6:
            line += f"（相对 A-Main-01 {d:+.1f}%）"
        return line

    lines_compact = " · ".join(fmt_row(*r) for r in rows)

    abl_lines = (
        "对照 A-Opt-05 全几何（同类配置 seed=1，outputs/field/experiment_index.csv）"
        " · no Abscissa RMSE|v|=1.062 · R²=0.569"
        " · no NormRadius 1.160 · 0.485"
        " · no Curvature 1.040 · 0.586"
        " · no Tangent 1.064 · 0.567"
        " · 待补 seed2–3 后入账 taskA_field"
    )

    new_slides = [
        (
            "基线之后：主线优化与消融",
            "与《任务A实验清单》§5–7 对齐 · 数据 split_AG_v1 与评估脚本不变 · 数值来自 实验记录表.xlsx",
            "续 1/6",
            "本节覆盖 Line A 已跑优化阶梯与 A-Abl-02 几何分量初探；Line W / V2 见文末",
        ),
        (
            "Line A：RMSE_|v| 优化阶梯（3 seeds）",
            lines_compact[:900] + ("…" if len(lines_compact) > 900 else ""),
            "续 2/6",
            f"A-Main-01 均值 {main_m:.3f} ± {main_s:.3f}；Opt-03/05/07 较主线约 −11.2% / −10.5% / −10.2%（均值）",
        ),
        (
            "战略锚点：A-Opt-05",
            "hidden_dim=256 · num_layers=4 · tw22205 · Pre-Norm · 后续消融与 Line G 默认从此配置派生（见任务A实验状态表）",
            "续 3/6",
            "论文叙事仍保留四组 A-Base/A-Main；控制变量与新增实验以 A-Opt-05 为母版，避免与旧 baseline 混改",
        ),
        (
            "A-Abl-02：几何分量单因子（初报）",
            abl_lines[:900] + ("…" if len(abl_lines) > 900 else ""),
            "续 4/6",
            "Curvature 缺失时 test RMSE|v| 仍接近全几何以示该分量关键；NormRadius 剔除当前伤害最大（待多 seed 确认）",
        ),
        (
            "Route-PhysicsAware-V2（修正路线）",
            "首轮只回答：V2 表示是否优于旧 kNN · geometry 是否仍成立 · MeshGNN vs PointCloud 取舍 · 是否进入第二轮",
            "续 5/6",
            "判定优先级：WSS/TAWSS → near_wall → interior → 全局 u,v,w,p → 效率（详见 任务A V2首轮判定与汇报模板）",
        ),
        (
            "数据出处与待补项",
            "主字段：docs/00-规范与记录/实验记录表.xlsx · taskA_field · experiment_master",
            "续 6/6",
            "待补：A-Abl-02 多 seed；A-Abl-01 输入层级；A-Base-04/05/06 可选 backbone；Line W 与 V2 Gate-0 依冻结卡推进",
        ),
    ]

    backup = PPTX.with_suffix(".pptx.bak_20260404")
    shutil.copy2(PPTX, backup)

    tmp_dir = PPTX.parent / "_pptx_unpack_append"
    if tmp_dir.is_dir():
        shutil.rmtree(tmp_dir)
    with zipfile.ZipFile(PPTX, "r") as z:
        z.extractall(tmp_dir)

    tmpl_path = tmp_dir / "ppt" / "slides" / "slide8.xml"
    template_full = tmpl_path.read_text(encoding="utf-8")
    template = strip_pics(template_full)

    pres_rel = tmp_dir / "ppt" / "_rels" / "presentation.xml.rels"
    rel_text = pres_rel.read_text(encoding="utf-8")
    pres_xml_path = tmp_dir / "ppt" / "presentation.xml"
    pres_xml = pres_xml_path.read_text(encoding="utf-8")

    id_re = re.findall(r'Id="rId(\d+)"', rel_text)
    max_rid = max(int(x) for x in id_re)
    sid_re = re.findall(r'<p:sldId id="(\d+)"', pres_xml)
    max_sid = max(int(x) for x in sid_re)

    start_slide = 15
    for offset, (tit, sub, foot, ban) in enumerate(new_slides):
        sn = start_slide + offset
        xml = slide_from_template(template, tit, sub, foot, ban)
        s_path = tmp_dir / "ppt" / "slides" / f"slide{sn}.xml"
        s_path.write_text(xml, encoding="utf-8")
        r_path = tmp_dir / "ppt" / "slides" / "_rels" / f"slide{sn}.xml.rels"
        r_path.parent.mkdir(parents=True, exist_ok=True)
        if not r_path.exists():
            r_path.write_text(
                '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>\n'
                '<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">'
                '<Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/slideLayout" '
                'Target="../slideLayouts/slideLayout7.xml"/></Relationships>',
                encoding="utf-8",
            )

        max_rid += 1
        rid = max_rid
        rel_line = (
            f'<Relationship Id="rId{rid}" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/slide" '
            f'Target="slides/slide{sn}.xml"/>'
        )
        rel_text = rel_text.replace("</Relationships>", f"{rel_line}</Relationships>")

        max_sid += 1
        sld_id = max_sid
        pres_xml = pres_xml.replace(
            "</p:sldIdLst>",
            f'<p:sldId id="{sld_id}" r:id="rId{rid}"/></p:sldIdLst>',
        )

        ct = tmp_dir / "[Content_Types].xml"
        ct_text = ct.read_text(encoding="utf-8")
        ct_line = (
            f'<Override PartName="/ppt/slides/slide{sn}.xml" '
            f'ContentType="application/vnd.openxmlformats-officedocument.presentationml.slide+xml"/>'
        )
        if f"/ppt/slides/slide{sn}.xml" not in ct_text:
            ct_text = ct_text.replace("</Types>", f"{ct_line}</Types>")
            ct.write_text(ct_text, encoding="utf-8")

    pres_rel.write_text(rel_text, encoding="utf-8")
    pres_xml_path.write_text(pres_xml, encoding="utf-8")

    out = PPTX
    with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as z:
        for p in sorted(tmp_dir.rglob("*")):
            if p.is_file():
                arc = str(p.relative_to(tmp_dir)).replace("\\", "/")
                z.write(p, arc)

    shutil.rmtree(tmp_dir)
    print(f"backup: {backup}")
    print(f"wrote: {out} (+{len(new_slides)} slides)")


if __name__ == "__main__":
    main()

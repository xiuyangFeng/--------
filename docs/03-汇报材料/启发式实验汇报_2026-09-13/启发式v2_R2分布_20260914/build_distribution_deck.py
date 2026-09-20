#!/usr/bin/env python3
"""Build a 16:9 editable slide deck around audited case-R² distribution figures.

This script only presents existing summary.json/PNG assets. It neither evaluates
models nor recomputes metrics or representative-case selections.

Usage:
  python3 build_distribution_deck.py
  python3 build_distribution_deck.py --summary summary.json --render

Expected summary schema:
  {"distributions": [{"key": ..., "title": ..., "subtitle": ...,
    "figure_png": ..., "representatives": {"worst": {"case_id":...,"r2":...},
      "most_frequent": {...}, "best": {...}},
    "stats": {"n_cases":...,"case_mean":...,"case_median":...,"case_p10":...,
      "negative_count":...}, "note":..., "source_paths": [...], "group": ...}]}
Titles, group labels, captions and methodological slides are native PowerPoint
shapes/text. Scientific figure PNGs are embedded whole with no crop or distortion.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import shutil
import subprocess
from pathlib import Path

from PIL import Image
from pptx import Presentation
from pptx.dml.color import RGBColor
from pptx.enum.shapes import MSO_CONNECTOR, MSO_SHAPE
from pptx.enum.text import PP_ALIGN
from pptx.oxml.xmlchemy import OxmlElement
from pptx.util import Inches, Pt

HERE = Path(__file__).resolve().parent
FONT = "Noto Sans CJK SC"
STEM = "启发式v2_病例R2分布补充_20260914"
W, H = 13.333333, 7.5
INK, MUTED, BLUE, TEAL = "17334A", "536875", "286DA0", "187E7D"
PURPLE, ORANGE, LINE, BG = "7952A0", "AA702A", "C9D4DC", "FFFFFF"


def rgb(value):
    return RGBColor.from_string(value)


def txt(slide, x, y, w, h, value, size=14, bold=False, color=INK,
        align=PP_ALIGN.LEFT):
    shape = slide.shapes.add_textbox(Inches(x), Inches(y), Inches(w), Inches(h))
    frame = shape.text_frame
    frame.clear()
    frame.word_wrap = True
    frame.margin_left = frame.margin_right = Inches(.025)
    frame.margin_top = frame.margin_bottom = Inches(.015)
    for i, line in enumerate(str(value).split("\n")):
        p = frame.paragraphs[0] if not i else frame.add_paragraph()
        p.alignment = align
        p.space_after = Pt(2)
        p.line_spacing = 1.05
        r = p.add_run()
        r.text = line
        r.font.name = FONT
        r.font.size = Pt(size)
        r.font.bold = bold
        r.font.color.rgb = rgb(color)
        for tag in ("a:latin", "a:ea", "a:cs"):
            elem = OxmlElement(tag)
            elem.set("typeface", FONT)
            r._r.get_or_add_rPr().append(elem)
    return shape


def panel(slide, x, y, w, h, fill="FFFFFF", stroke=LINE, radius=True):
    kind = MSO_SHAPE.ROUNDED_RECTANGLE if radius else MSO_SHAPE.RECTANGLE
    shape = slide.shapes.add_shape(kind, Inches(x), Inches(y), Inches(w), Inches(h))
    shape.fill.solid()
    shape.fill.fore_color.rgb = rgb(fill)
    shape.line.color.rgb = rgb(stroke)
    shape.line.width = Pt(.9)
    if radius:
        shape.adjustments[0] = .06
    return shape


def arrow(slide, x1, y1, x2, y2, color=TEAL):
    shape = slide.shapes.add_connector(MSO_CONNECTOR.STRAIGHT, Inches(x1), Inches(y1),
                                       Inches(x2), Inches(y2))
    shape.line.color.rgb = rgb(color)
    shape.line.width = Pt(1.7)
    end = OxmlElement("a:tailEnd")
    end.set("type", "triangle")
    shape._element.spPr.get_or_add_ln().append(end)


def card(slide, x, y, w, h, title, body, color=BLUE, fill="EEF5FB"):
    panel(slide, x, y, w, h, fill, color)
    txt(slide, x+.15, y+.14, w-.30, .45, title, 17, True, color)
    txt(slide, x+.15, y+.79, w-.30, h-.85, body, 13.2)


def slide_base(prs, title, number, group=None, subtitle="", compact=False):
    slide = prs.slides.add_slide(prs.slide_layouts[6])
    slide.background.fill.solid()
    slide.background.fill.fore_color.rgb = rgb(BG)
    if compact:
        panel(slide, .40, .28, .07, .35, TEAL, TEAL, False)
        title_size = 22 if len(title) <= 38 else 19.5
        txt(slide, .59, .20, 10.58, .49, title, title_size, True)
        if group:
            label = group["label"]
            panel(slide, 11.32, .23, 1.56, .35, group["fill"], group["color"])
            txt(slide, 11.37, .255, 1.46, .25, label, 9.2, True,
                group["color"], PP_ALIGN.CENTER)
        txt(slide, .62, .71, 12.08, .35, subtitle, 10.5, color=MUTED)
    else:
        panel(slide, .48, .40, .09, .53, TEAL, TEAL, False)
        txt(slide, .76, .34, 11.98, .59, title, 27, True)
        txt(slide, .78, 1.03, 11.95, .50, subtitle, 13, color=MUTED)
    panel(slide, .57, 7.15, 12.2, .008, LINE, LINE, False)
    txt(slide, .63, 7.23, 11.67, .21,
        "病例 R² 分布与代表病例 · 2026-09-14 · 来源与选择规则见备注页", 8.2, color=MUTED)
    txt(slide, 12.28, 7.20, .43, .27, f"{number:02d}", 10.5, True, TEAL, PP_ALIGN.RIGHT)
    return slide


def section_info(item, overrides):
    group = str(item.get("group", ""))
    title_note = f"{group} {item.get('title','')} {item.get('subtitle','')}"
    historical = any(term in title_note.lower() for term in ("历史", "histor", "旧协议", "v4", "r5v"))
    label = overrides.get(group, group or ("历史机制证据" if historical else "当前 V5"))
    # Long original identifiers belong in speaker notes, not a narrow header pill.
    if len(label) > 12:
        label = "历史机制证据" if historical else "当前 V5"
    return {"label": label, "historical": historical,
            "color": ORANGE if historical else TEAL,
            "fill": "FFF4E5" if historical else "EAF5F2"}


def figure_path(value, summary_path):
    path = Path(value).expanduser()
    candidates = [path] if path.is_absolute() else [summary_path.parent/path, Path.cwd()/path]
    for candidate in candidates:
        if candidate.is_file():
            return candidate.resolve()
    raise FileNotFoundError(f"Figure not found: {value}; tried {candidates}")


def insert_uncropped(slide, path, x, y, w, h):
    with Image.open(path) as im:
        pw, ph = im.size
    scale = min(w/pw, h/ph)
    width, height = pw*scale, ph*scale
    picture = slide.shapes.add_picture(str(path), Inches(x+(w-width)/2),
                                       Inches(y+(h-height)/2), Inches(width), Inches(height))
    picture.name = f"完整科学图：{path.name}"
    assert picture.crop_left == picture.crop_top == picture.crop_right == picture.crop_bottom == 0
    return {"path": str(path), "pixels": [pw, ph], "aspect_ratio": pw/ph,
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}


def validate_summary(data):
    items = data.get("distributions")
    if not isinstance(items, list) or not items:
        raise ValueError("summary.json must include a nonempty distributions list")
    keys = set()
    for item in items:
        for key in ("key", "title", "figure_png", "representatives", "stats"):
            if key not in item:
                raise ValueError(f"missing {key} in distribution {item.get('key')}")
        if item["key"] in keys:
            raise ValueError(f"duplicate distribution key: {item['key']}")
        keys.add(item["key"])
        for role in ("worst", "most_frequent", "best"):
            row = item["representatives"][role]
            if not row.get("case_id") or not math.isfinite(float(row["r2"])):
                raise ValueError(f"invalid representative {role} in {item['key']}")
    return items


def introduction(prs):
    slide = slide_base(prs, "从模型到病例：把平均分展开成可讨论的证据", 1,
        subtitle="每条路线采用同一叙事：网络做了什么 → 学得是否稳定 → 哪些病例常见、哪些失分最重。")
    content = [
        ("1  模型与模块", "说明输入、输出与模块连接。\nWSS 是直接预测，还是由预测速度后处理得到？", BLUE, "EEF5FB"),
        ("2  Loss 与 NMAE", "先看训练曲线，再看误差。\n物理路线展开各 loss；\nNMAE 明确变量和分母。", PURPLE, "F4EFF9"),
        ("3  病例 R² 分布", "横向散点：每例落在哪里？\n直方图：常见表现与低分尾部各有多少？", TEAL, "EAF5F2"),
        ("4  三类代表病例", "Best：当前口径最高。\nMost：最高频区间代表。\nWorst：当前口径最低。", ORANGE, "FFF4E5"),
    ]
    for i, (title, body, color, fill) in enumerate(content):
        x = .60+i*3.10
        card(slide, x, 1.93, 2.86, 2.48, title, body, color, fill)
        if i < 3:
            arrow(slide, x+2.86, 3.15, x+3.10, 3.15)
    panel(slide, .63, 4.79, 12.05, 1.73, "FAFBFD", LINE)
    txt(slide, .83, 4.96, 11.64, .33, "如何解释 Most（most frequent）", 16, True, TEAL)
    txt(slide, .83, 5.46, 11.60, .83,
        "Most 按固定 0.10 分箱选频数最高区间的代表病例；不是中位数。三 seed 图展示病例分数均值，并非预测集成。\n"
        "当前 V5 与历史 PINN/派生 WSS 分区展示；最高频区间代表也不是整体最好，不同协议之间不作冠军排名。", 13.3)
    slide.notes_slide.notes_text_frame.text = (
        "展示顺序用于启发讨论，而不是新选模规则。训练loss、NMAE及R²属于不同度量；"
        "病例R²的算术平均不等同于全场 pooled R² 或 R²_cb。Most根据生产summary.json的脚本所冻结分箱选取，"
        "本PPT脚本不会按直方图视觉重新挑选病例。若需要严格跨路线云图比较，再增加一组共同病例。")


def distribution_slide(prs, item, number, summary_path, overrides):
    group = section_info(item, overrides)
    subtitle = str(item.get("subtitle", ""))
    if "last_converged" in subtitle:
        subtitle = subtitle.replace("official last_converged", "epoch 9999；converged:false")
        subtitle = subtitle.replace("last_converged", "epoch 9999；converged:false")
    slide = slide_base(prs, item["title"], number, group=group,
                       subtitle=subtitle, compact=True)
    # A 16:9 figure occupies 10.60×5.96 in, shown entirely without crop.
    asset = insert_uncropped(slide, figure_path(item["figure_png"], summary_path),
                             .64, 1.075, 12.05, 5.87)
    note = str(item.get("note", "")).strip()
    caption = ("历史协议：用于解释该模型的病例失分，不与当前 V5 排名。"
               if group["historical"] else
               "按该模型的完整病例分布选例；Most 为最高频区间代表。")
    if item.get("caption"):
        caption = str(item["caption"])
    txt(slide, .76, 6.96, 11.88, .17, caption, 8.0, color=MUTED)
    slide.notes_slide.notes_text_frame.text = json.dumps({
        "key": item["key"], "group": item.get("group"), "subtitle": item.get("subtitle"),
        "representatives": item["representatives"], "stats": item["stats"],
        "note": note, "source_paths": item.get("source_paths", []),
        "figure": asset, "selection_note": "本脚本直接呈现summary中的选例与统计，没有重新评估或重新选例。"
    }, ensure_ascii=False, indent=2)
    return asset


def closing(prs, number):
    slide = slide_base(prs, "下一步补图：让分布中的三个位置变成可解释的病例", number,
        subtitle="每个模型按已选 Best / Most / Worst 制作；所有图片绑定原 run、checkpoint、seed、病例与指标口径。")
    card(slide, .64, 1.87, 3.85, 2.91, "直接 WSS", "CFD / Pred / signed error 三联\n标出热点位置与局部放大区域\n\n关注：幅值偏差、热点偏移、\n局部条纹是否被平滑。", BLUE, "EEF5FB")
    card(slide, 4.75, 1.87, 3.85, 2.91, "压力 / 速度", "压力与速度幅值的同位置对照\n速度补截面内二次流矢量\n\n速度 → WSS：补近壁剖面，\n对应同一冻结算子的梯度深度。", TEAL, "EAF5F2")
    card(slide, 8.87, 1.87, 3.85, 2.91, "物理约束路线", "病例场图 + 各项 loss 诊断\n有数据时补物理残差空间图\n\n关注：哪一项改善、哪一项冲突；\n历史结果只在原协议内解释。", PURPLE, "F4EFF9")
    panel(slide, .66, 5.16, 12.03, 1.30, "FAFBFD", LINE)
    txt(slide, .84, 5.30, 11.66, .34, "后处理统一口径", 16, True, ORANGE)
    txt(slide, .84, 5.83, 11.62, .48,
        "CFD/Pred 同几何、同视角、同色标；误差 = Pred − CFD，采用对称色标。\n"
        "模型各自三例用于解释分布；横向比较空间误差时，另加同一病例组，避免把病例难度当成模型差异。", 12.4)
    slide.notes_slide.notes_text_frame.text = (
        "图注记录run/checkpoint/seed/case/split/frame/variable/unit/color-range。"
        "如果分布值是三seed病例R²均值，病例图应采用明确说明的单seed或预测集成；"
        "不得把单seed云图的R²直接换成三seed均值。原始同点指标为数值来源，面片插值只作显示。"
        "当前和历史模型的原始病例ID可能相同，但必须保留数据版本、坐标、单位及评估协议。")


def render(pptx_path, outdir):
    for command in ("libreoffice", "pdftoppm"):
        if not shutil.which(command):
            raise RuntimeError(f"required render command unavailable: {command}")
    profile = outdir/".libreoffice_profile"
    subprocess.run(["libreoffice", f"-env:UserInstallation={profile.as_uri()}", "--headless",
                    "--convert-to", "pdf", "--outdir", str(outdir), str(pptx_path)], check=True)
    pdf_path = pptx_path.with_suffix(".pdf")
    if not pdf_path.is_file():
        raise RuntimeError(f"LibreOffice did not create {pdf_path}")
    subprocess.run(["pdftoppm", "-scale-to", "1800", "-png", str(pdf_path),
                    str(outdir/"slide")], check=True)
    return pdf_path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--summary", type=Path, default=HERE/"summary.json")
    parser.add_argument("--output-dir", type=Path, default=HERE)
    parser.add_argument("--render", action="store_true")
    args = parser.parse_args()
    summary_path = args.summary.resolve()
    data = json.loads(summary_path.read_text(encoding="utf-8"))
    items = validate_summary(data)
    overrides = data.get("slide_group_labels", {})
    outdir = args.output_dir.resolve()
    outdir.mkdir(parents=True, exist_ok=True)
    prs = Presentation()
    prs.slide_width, prs.slide_height = Inches(W), Inches(H)
    prs.core_properties.title = "启发式v2：病例R²分布补充"
    prs.core_properties.subject = "模型—Loss/NMAE—病例R²分布—Best/Most/Worst"
    prs.core_properties.author = "GNN research workspace"
    introduction(prs)
    assets = []
    for number, item in enumerate(items, 2):
        assets.append(distribution_slide(prs, item, number, summary_path, overrides))
    closing(prs, len(items)+2)
    for i, slide in enumerate(prs.slides, 1):
        for shape in slide.shapes:
            if not (shape.left >= 0 and shape.top >= 0 and
                    shape.left+shape.width <= prs.slide_width and
                    shape.top+shape.height <= prs.slide_height):
                raise ValueError(f"shape exceeds slide {i}: {shape.name}")
    pptx_path = outdir/f"{STEM}.pptx"
    prs.save(pptx_path)
    manifest = {"pptx": str(pptx_path), "n_slides": len(prs.slides),
                "summary": str(summary_path), "summary_sha256": hashlib.sha256(summary_path.read_bytes()).hexdigest(),
                "figures": assets, "font": FONT, "scientific_figures": "uncropped PNG; labels/captions editable"}
    if args.render:
        manifest["pdf"] = str(render(pptx_path, outdir))
    (outdir/"deck_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2)+"\n", encoding="utf-8")
    print(json.dumps({"pptx": str(pptx_path), "slides": len(prs.slides), "rendered": args.render}, ensure_ascii=False))


if __name__ == "__main__":
    main()

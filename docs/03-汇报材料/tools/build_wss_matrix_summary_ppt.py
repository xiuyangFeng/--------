# -*- coding: utf-8 -*-
"""组装《WSS PointNet/PointNet++ 实验矩阵汇总》组会 PPT。

输入:
  - figures/WSS_PointNet矩阵汇总_20260721/{f1..f10,s1..s6}*.png
    (由 build_wss_matrix_summary_figures.py / build_wss_matrix_schematics.py 生成)
  - 结论文字取自 WSS_PointNet实验矩阵与结果汇总last.xlsx 与三份推进/跟踪文档,
    在本脚本中以常量固化(数据快照 2026-07-20)。
输出:
  - docs/03-汇报材料/WSS_PointNet实验矩阵汇总_组会汇报_20260721.pptx
"""
from pathlib import Path

from PIL import Image
from pptx import Presentation
from pptx.dml.color import RGBColor
from pptx.enum.text import PP_ALIGN, MSO_ANCHOR
from pptx.oxml.ns import qn
from pptx.util import Emu, Inches, Pt

ROOT = Path(__file__).resolve().parents[1]
FIGS = ROOT / "figures" / "WSS_PointNet矩阵汇总_20260721"
OUT = ROOT / "WSS_PointNet实验矩阵汇总_组会汇报_20260721.pptx"

SLIDE_W, SLIDE_H = Inches(13.333), Inches(7.5)

INK = RGBColor(0x0B, 0x0B, 0x0B)
SEC = RGBColor(0x52, 0x51, 0x4E)
MUT = RGBColor(0x89, 0x87, 0x81)
BLUE = RGBColor(0x2A, 0x78, 0xD6)
GREEN = RGBColor(0x0E, 0x8A, 0x5F)
RED = RGBColor(0xC7, 0x33, 0x32)
HDR_FILL = RGBColor(0xF0, 0xEF, 0xEC)
ROW_FILL = RGBColor(0xFB, 0xFB, 0xF9)
FONT = "微软雅黑"


# ---------------------------------------------------------------- helpers
def _style_run(run, size=12, bold=False, color=INK, italic=False):
    f = run.font
    f.size = Pt(size)
    f.bold = bold
    f.italic = italic
    f.color.rgb = color
    f.name = FONT
    rPr = run._r.get_or_add_rPr()
    for tag in ("a:latin", "a:ea"):
        el = rPr.find(qn(tag))
        if el is None:
            el = rPr.makeelement(qn(tag), {})
            rPr.append(el)
        el.set("typeface", FONT)


def add_text(slide, x, y, w, h, paras, valign=MSO_ANCHOR.TOP):
    """paras: list of dicts {text,size,bold,color,align,space_after,line}
    或 list of (text, size, bold, color) 简写; text 内 ' ' 不处理。"""
    box = slide.shapes.add_textbox(x, y, w, h)
    tf = box.text_frame
    tf.word_wrap = True
    tf.vertical_anchor = valign
    first = True
    for p in paras:
        if isinstance(p, tuple):
            p = dict(zip(("text", "size", "bold", "color"), p))
        para = tf.paragraphs[0] if first else tf.add_paragraph()
        first = False
        para.alignment = p.get("align", PP_ALIGN.LEFT)
        if p.get("space_after") is not None:
            para.space_after = Pt(p["space_after"])
        if p.get("space_before") is not None:
            para.space_before = Pt(p["space_before"])
        if p.get("line") is not None:
            para.line_spacing = p["line"]
        runs = p.get("runs")
        if runs:
            for rt, ropt in runs:
                r = para.add_run()
                r.text = rt
                _style_run(r, ropt.get("size", p.get("size", 12)),
                           ropt.get("bold", p.get("bold", False)),
                           ropt.get("color", p.get("color", INK)))
        else:
            r = para.add_run()
            r.text = p.get("text", "")
            _style_run(r, p.get("size", 12), p.get("bold", False),
                       p.get("color", INK))
    return box


def add_title(slide, title, kicker=None):
    add_text(slide, Inches(0.45), Inches(0.22), Inches(12.45), Inches(0.55),
             [dict(text=title, size=22, bold=True, color=INK)])
    if kicker:
        add_text(slide, Inches(0.45), Inches(0.78), Inches(12.45), Inches(0.35),
                 [dict(text=kicker, size=12.5, color=SEC)])
    ln = slide.shapes.add_shape(1, Inches(0.45), Inches(0.74), Inches(1.2), Emu(28575))
    ln.fill.solid()
    ln.fill.fore_color.rgb = BLUE
    ln.line.fill.background()
    ln.shadow.inherit = False


def add_pic_fit(slide, path, x, y, max_w, max_h, align="center"):
    im = Image.open(path)
    iw, ih = im.size
    scale = min(max_w / iw, max_h / ih)
    w, h = int(iw * scale), int(ih * scale)
    if align == "center":
        px = x + int((max_w - w) / 2)
    elif align == "left":
        px = x
    else:
        px = x + (max_w - w)
    py = y + int((max_h - h) / 2)
    return slide.shapes.add_picture(str(path), px, py, width=w, height=h)


def add_table(slide, x, y, w, h, data, col_w=None, font=10.5, hdr_font=None,
              hdr_color=INK, align_map=None, cell_colors=None):
    """data: list of rows(list of str); 第一行为表头。cell_colors: {(r,c):color}"""
    rows, cols = len(data), len(data[0])
    shp = slide.shapes.add_table(rows, cols, x, y, w, h)
    tbl = shp.table
    tbl.first_row = False
    tbl.horz_banding = False
    if col_w:
        total = sum(col_w)
        for i, cw in enumerate(col_w):
            tbl.columns[i].width = int(w * cw / total)
    for r in range(rows):
        for c in range(cols):
            cell = tbl.cell(r, c)
            cell.margin_left = Inches(0.06)
            cell.margin_right = Inches(0.06)
            cell.margin_top = Inches(0.02)
            cell.margin_bottom = Inches(0.02)
            cell.vertical_anchor = MSO_ANCHOR.MIDDLE
            cell.fill.solid()
            cell.fill.fore_color.rgb = HDR_FILL if r == 0 else (
                ROW_FILL if r % 2 == 0 else RGBColor(0xFF, 0xFF, 0xFF))
            tf = cell.text_frame
            tf.word_wrap = True
            para = tf.paragraphs[0]
            if align_map and c in align_map:
                para.alignment = align_map[c]
            run = para.add_run()
            run.text = str(data[r][c])
            color = hdr_color if r == 0 else INK
            if cell_colors and (r, c) in cell_colors:
                color = cell_colors[(r, c)]
            _style_run(run, (hdr_font or font + 0.5) if r == 0 else font,
                       bold=(r == 0), color=color)
    return tbl


def blank(prs):
    return prs.slides.add_slide(prs.slide_layouts[6])


def footer(slide, idx, note=None):
    add_text(slide, Inches(0.45), Inches(7.12), Inches(12.45), Inches(0.3),
             [dict(runs=[
                 (note or "数据源:WSS_PointNet实验矩阵与结果汇总last.xlsx(2026-07-20 快照)",
                  dict(size=9, color=MUT)),
                 (f"    {idx:02d}", dict(size=9, color=MUT)),
             ], align=PP_ALIGN.LEFT)])


# ---------------------------------------------------------------- slides
def s_cover(prs):
    s = blank(prs)
    bar = s.shapes.add_shape(1, 0, Inches(2.0), SLIDE_W, Emu(28575 * 2))
    bar.fill.solid()
    bar.fill.fore_color.rgb = BLUE
    bar.line.fill.background()
    bar.shadow.inherit = False
    add_text(s, Inches(0.9), Inches(2.45), Inches(11.5), Inches(1.5), [
        dict(text="WSS 点云回归 · PointNet/PointNet++ 实验矩阵汇总",
             size=34, bold=True, color=INK),
        dict(text="指标精简建议 · 网络结构与调参图解 · CFD 数据链路与实验取舍",
             size=17, color=SEC, space_before=10),
    ])
    add_text(s, Inches(0.9), Inches(5.7), Inches(11.5), Inches(1.1), [
        dict(text="2026-07-21 组会", size=13, color=SEC),
        dict(text="数据源:WSS_PointNet实验矩阵与结果汇总last.xlsx"
                  "(79 行实验 · 11 个阶段 · 汇总对比/教师汇报视图/指标说明/实验矩阵总览 4 个 sheet)",
             size=11.5, color=MUT, space_before=4),
    ])


def s_summary(prs, idx):
    s = blank(prs)
    add_title(s, "一页结论", "先说结果,再看证据")
    items = [
        ("当前最优配置:Q1V", "PointNet++ SA3 + 顶点随机 5000 点 + SAME + FPS 中心 —— "
         "test27 物理 R²_cb = 0.276(27 例仅 1 例负 R²);PointNet 侧最优 P2V = 0.219。", GREEN),
        ("两次真正的跳变都不是「调网络」",
         "① 容量对齐:E0→E2 宽通道 +0.073;② 输入采样:固定 FPS2000→顶点随机 5000(Q0→Q1V,+0.055)。", INK),
        ("结构旋钮已全面扫过、全部无稳定收益",
         "width/nsample/ball 半径/SA1 分组方式/解码器/中心采样 —— 结构微调线收口(证据见后)。", INK),
        ("指标与实验矩阵都该「减肥」",
         "指标 30 列 → 12 列(删共线与从未参与判定的列);实验线 8 条建议归档、4 条保留待独立确认。", BLUE),
        ("未解决的硬问题:高 WSS 尾部",
         "high-WSS R² 全程为负(最好 −0.51),p99 幅值比 ≤0.73 —— 高 WSS 幅值系统性低估,是下一阶段主攻方向。", RED),
    ]
    y = Inches(1.35)
    for head, body, color in items:
        add_text(s, Inches(0.7), y, Inches(12.0), Inches(1.05), [
            dict(runs=[("■  ", dict(size=14, color=color, bold=True)),
                       (head, dict(size=15.5, bold=True, color=color))]),
            dict(text=body, size=12.5, color=SEC, space_before=2, line=1.05),
        ])
        y += Inches(1.12)
    footer(s, idx)


def s_data(prs, idx):
    s = blank(prs)
    add_title(s, "数据与 CFD 网格", "三个队列 · Fluent 非稳态 CFD · 峰值收缩期壁面 WSS")
    add_table(s, Inches(0.45), Inches(1.3), Inches(12.45), Inches(1.75), [
        ["队列", "病种 / 解剖", "例数(入训)", "壁面网格点/例", "备注"],
        ["AG", "主动脉-髂动脉分叉", "76(61)", "约 0.9–2.1 万", "v4 重建 84→76;历史主队列与全部锚点所在"],
        ["AAA", "腹主动脉瘤(破裂/未破裂)", "63(入训 57)", "约 5–7 万", "分层 split 纳入 train/test;网格密度约为 AG 4–6 倍"],
        ["ILO", "髂动脉闭塞·术前(仅 before)", "41", "约 8 万", "仅用于扩容实验(D1/D2/pool2025),未获稳定增益"],
    ], col_w=[1.0, 2.6, 1.5, 1.8, 5.2], font=11)
    add_text(s, Inches(0.45), Inches(3.25), Inches(12.45), Inches(3.8), [
        dict(text="CFD 与网格口径", size=14, bold=True, color=INK, space_after=4),
        dict(text="• 求解:ANSYS Fluent 非稳态,每例约 81 个时间步;目标取入口流量峰值步(peak systole)"
                  "的壁面 WSS 标量,单位 Pa(全队列 p99 ≈ 20 Pa,AG/AAA 幅值同量级可比)。",
             size=12, color=SEC, line=1.12),
        dict(text="• 原始导出:Fluent ASCII 壁面节点(nodenumber·x/y/z·pressure·wall-shear 标量+3 矢量分量)"
                  " + 原始 STL 曲面 + VMTK 中心线(弧长/内切球半径/曲率);坐标 SI 米→mm,单位因子逐例反推(≈1000)。",
             size=12, color=SEC, line=1.12, space_before=3),
        dict(text="• 边界条件:入口为共享 Fourier 波形模板(逐例唯一差异=入口面积);出口 4 髂支各自 RCR "
                  "Windkessel、逐例不同 —— 属 oracle 信息,模型拿不到,是精度天花板的主要来源。",
             size=12, color=SEC, line=1.12, space_before=3),
        dict(text="• 可信度:CFD 自复现误差 floor≈2%,对应信息上限 R²_cap≈0.92–0.96(不是标签噪声问题)。",
             size=12, color=SEC, line=1.12, space_before=3),
        dict(text="• QA 硬门:peak/all 零值率≤1%;n_wall≤5 万(AG 护栏);裁剪/居中偏移≤5%;刚性配准 det(R)=+1、"
                  "+Z 主干/−Z 双髂支;单位异常即拒。新队列(AAA/ILO)审计:达标 171/190,排除 19。",
             size=12, color=SEC, line=1.12, space_before=3),
    ])
    footer(s, idx)


def s_fullfig(prs, idx, title, kicker, fig, note=None, max_h=5.7):
    s = blank(prs)
    add_title(s, title, kicker)
    add_pic_fit(s, FIGS / fig, Inches(0.45), Inches(1.25), Inches(12.45), Inches(max_h))
    footer(s, idx, note)
    return s


def s_norm_post(prs, idx):
    s = blank(prs)
    add_title(s, "归一化 · 训练协议 · 后处理", "从训练目标到 ParaView 交付物")
    add_pic_fit(s, FIGS / "s6_normalization.png", Inches(0.45), Inches(1.3),
                Inches(6.4), Inches(5.6))
    add_text(s, Inches(7.15), Inches(1.3), Inches(5.75), Inches(5.8), [
        dict(text="训练目标归一化", size=13.5, bold=True, color=INK),
        dict(text="• log1p → z-score,统计只用 train + 峰值步(log μ/σ=1.16/1.09);评估反归一化回 Pa 再算全部物理指标。",
             size=11.5, color=SEC, line=1.1),
        dict(text="• 已试口径:WSS/WSSmax(No-Go)、逐病例等权(不晋级);仅「逐父系等权 A1b」保留为确认候选。",
             size=11.5, color=SEC, line=1.1, space_before=2),
        dict(text="模型输入(6 维)", size=13.5, bold=True, color=INK, space_before=8),
        dict(text="• xyz(病例内 [−1,1] 保形缩放) + 中心线弧长 abscissa_norm + 局部半径 local_radius + "
                  "曲率 curvature(signed_log1p);几何列 train-only z-score。",
             size=11.5, color=SEC, line=1.1),
        dict(text="训练协议(矩阵口径)", size=13.5, bold=True, color=INK, space_before=8),
        dict(text="• AdamW lr=1e-3 · warmup10+cosine · batch=8 病例 · AMP;矩阵协议 400ep 无验证集,"
                  "一律取 ckpt_best(train_loss),不按 test 反选;单 seed=1234 只作工程比较,确认须 3 seed。",
             size=11.5, color=SEC, line=1.1),
        dict(text="后处理 PostView(每病例)", size=13.5, bold=True, color=INK, space_before=8),
        dict(text="• 完整壁面推理 → 同点 wall CSV(正式指标唯一来源) → Gaussian(r=3mm,sharpness=2)回插 STL "
                  "→ VTP + CFD|Pred|Error 三联图 + self-max 视图(导师口径:各除自身最大值,只看空间分布)。",
             size=11.5, color=SEC, line=1.1),
    ])
    footer(s, idx)


def s_metric_system(prs, idx):
    s = blank(prs)
    add_title(s, "指标体系:按「四句话」汇报", "同一模型可能平均误差更小、但空间解释更差 —— 四类问题分开回答")
    add_table(s, Inches(0.45), Inches(1.35), Inches(12.45), Inches(3.6), [
        ["汇报顺序", "指标", "回答的问题"],
        ["① 总体拟合", "物理 R²_cb(主指标) · case mean R² · 负例数",
         "病例等权下解释了多少总体变化;是否对每个病例都稳定;几例连均值基线都不如"],
        ["② 平均误差", "RMSE(Pa) · NMAE(%)",
         "平均差多少 Pa;误差占真实动态范围的百分比"],
        ["③ 高尾幅值", "high-WSS R² · top10 幅值比 · p99 比",
         "临床最关心的高 WSS 区学得如何;热点/极端尾部幅值是否被系统压缩"],
        ["④ 热点定位", "top10 IoU · Spearman",
         "预测热点是否出现在正确位置;整体高低排序是否一致"],
    ], col_w=[1.6, 4.6, 6.2], font=12, hdr_font=12.5)
    add_text(s, Inches(0.45), Inches(5.3), Inches(12.45), Inches(1.7), [
        dict(text="三个必须口径隔离的维度", size=13.5, bold=True, color=INK, space_after=3),
        dict(text="• pooled vs case-mean:pooled 被大病例/高 WSS 病例主导;case-mean 病例等权 → 主指标用 case-balanced。",
             size=12, color=SEC, line=1.12),
        dict(text="• 物理空间 vs 归一化空间:log-z 反变换非线性,归一化变好≠物理变好 → 汇报以物理 Pa 空间为主。",
             size=12, color=SEC, line=1.12),
        dict(text="• 不同测试集(test16/test15/test27/val21/test36)禁止横比;test16/test27 已被多轮选择复用,只作同协议工程比较。",
             size=12, color=RED, line=1.12),
    ])
    footer(s, idx)


def s_stage_table(prs, idx):
    s = blank(prs)
    add_title(s, "实验矩阵总览:11 个阶段、79 行实验", "每阶段一句话结论(物理 R²_cb 为主指标)")
    rows = [
        ["阶段", "测试集", "代表配置(物理 R²_cb)", "一句话结论"],
        ["Ⅰ 基线 2×3+深度", "test16", "E2 宽通道 0.214", "容量是主增量(+0.073);加点数/加深/WSS/WSSmax 均无收益"],
        ["Ⅱ v4 数据切换", "test15", "AG+AAA 0.205", "v4 单独重跑回退 No-Go;AAA 扩容小幅非均匀(+0.023)"],
        ["Ⅲ 分层协议锚点", "test27", "SA3 0.133 vs E2 0.053", "换 stratified 划分;SA3 在新协议反超 E2(+0.080)"],
        ["Ⅳ Phase-V 采样矩阵", "test27", "Q1V 0.276★ / P2V 0.219★", "顶点随机 5000 是主要增益;两候选晋级"],
        ["Ⅴ 数据扩容(ILO)", "test27/36", "D1/D2/pool2025", "三个主配对全为负差 →「扩容有效」不成立"],
        ["Ⅵ 架构搜索 n×w", "val21", "n32_w64 0.260", "第一名仅 +0.0014、参数 4 倍 → 无稳定容量增益"],
        ["Ⅶ 半径探索", "test27", "1.2× 0.285", "响应非单调;缩小全负,仅 1.2× 微弱正向留复核"],
        ["Ⅷ QAD-Lite 解码器", "val21+test27", "3 种子配对", "val +0.0143<门;精确复核 −0.0048 → No-Go"],
        ["Ⅸ-Ⅹ SA1 覆盖/重叠矩阵", "test27", "17 臂(分组×support)", "无一超过 random5000+ball16 基准;覆盖率↑不带来精度↑"],
        ["Ⅺ 归一化×尾部损失", "test36", "A1b 0.275", "仅 A1b(逐父系等权)全面正向(+0.030),保留确认候选"],
    ]
    add_table(s, Inches(0.45), Inches(1.3), Inches(12.45), Inches(5.55), rows,
              col_w=[2.5, 1.5, 3.0, 6.3], font=11, hdr_font=11.5)
    footer(s, idx)


def s_two_figs(prs, idx, title, kicker, fig_l, fig_r, note=None, split=0.5):
    s = blank(prs)
    add_title(s, title, kicker)
    wl = Inches(12.45 * split - 0.1)
    wr = Inches(12.45 * (1 - split) - 0.1)
    add_pic_fit(s, FIGS / fig_l, Inches(0.45), Inches(1.25), wl, Inches(5.7))
    add_pic_fit(s, FIGS / fig_r, Inches(0.45) + wl + Inches(0.2), Inches(1.25),
                wr, Inches(5.7))
    footer(s, idx, note)
    return s


def s_metric_evidence(prs, idx):
    s = blank(prs)
    add_title(s, "指标取舍(1/2):冗余证据", "全矩阵 79 行实验上的指标间相关性")
    add_pic_fit(s, FIGS / "f10_metric_redundancy.png", Inches(0.45), Inches(1.25),
                Inches(12.45), Inches(4.1))
    add_text(s, Inches(0.45), Inches(5.5), Inches(12.45), Inches(1.55), [
        dict(text="• 高度共线可删:R²_raw≈R²_cb(r=0.85)、range-NRMSE≈NMAE(r=0.98/0.89)、"
                  "case med/P10≈case mean(r=0.89/0.73)。", size=12.5, color=SEC, line=1.15),
        dict(text="• 从未参与判定可删:high-WSS Spearman 全矩阵 ≈0(−0.067~0.054,高尾样本太少)。",
             size=12.5, color=SEC, line=1.15, space_before=3),
        dict(text="• 不共线必须保留:top10 幅值比 vs p99 比 r=−0.18、top10 IoU vs Spearman r=0.24 —— "
                  "幅值校准与定位各携带独立信息。", size=12.5, color=GREEN, line=1.15, space_before=3),
    ])
    footer(s, idx)


def s_metric_verdict(prs, idx):
    s = blank(prs)
    add_title(s, "指标取舍(2/2):30 列 → 12 列", "保留 = 物理 9 列 + 归一化 3 列;其余 18 列删除(数据归档,不再汇报)")
    keep = [
        ["保留(12)", "理由"],
        ["物理 R²_cb + case mean R² + 负例数", "预注册主指标;跨病例稳定性;失败病例计数(三者语义不同)"],
        ["RMSE(Pa) · NMAE pooled/case-mean", "绝对误差(Pa)与相对误差(占动态范围)各留一族"],
        ["high-WSS R² · top10 幅值比 · p99 比", "高尾三件套互不共线(r=−0.18~−0.32),且是核心开放问题"],
        ["top10 IoU · Spearman", "热点定位与全局排序,相互独立(r=0.24)"],
        ["归一化 R²_cb", "训练目标诊断;历史锚点 Q2V=0.621 需延续"],
    ]
    drop = [
        ["删除(18)", "理由"],
        ["R² raw(物理+归一化)", "与 R²_cb r=0.85 共线,且受大病例点数隐式加权"],
        ["case med / case P10(物理+归一化)", "与 case mean r=0.89/0.73;尾部风险已由负例数覆盖"],
        ["range-NRMSE 4 列", "与 NMAE 一族 r=0.98/0.89,完全同信息"],
        ["物理 MAE", "NMAE 分子就是 MAE;绝对误差已有 RMSE"],
        ["归一化 MAE/RMSE/NMAE/NRMSE(6 列)", "无物理单位;不同归一化协议之间数值不可比"],
        ["high-WSS Spearman", "全矩阵 ≈0,从未参与任何 Go/No-Go 判定"],
        ["归一化 top10 幅值比", "正比例缩放不改变幅值比,与物理列重复"],
    ]
    add_table(s, Inches(0.45), Inches(1.35), Inches(6.05), Inches(3.9), keep,
              col_w=[3.1, 3.6], font=10.5, hdr_color=GREEN)
    add_table(s, Inches(6.85), Inches(1.35), Inches(6.05), Inches(4.4), drop,
              col_w=[3.1, 3.6], font=10.5, hdr_color=RED)
    add_text(s, Inches(0.45), Inches(6.15), Inches(12.45), Inches(0.85), [
        dict(text="落地方式:「实验矩阵总览」sheet 保留全部列作档案;「教师汇报视图」与今后 PPT 只呈现上面 12 列,"
                  "按四句话顺序排列。", size=12.5, color=INK, bold=True),
    ])
    footer(s, idx)


def s_exp_verdict(prs, idx):
    s = blank(prs)
    add_title(s, "实验线取舍:8 条归档 · 4 条保留", "归档=结论已定,从汇报表移出(证据与产物保留);保留=待独立确认")
    arch = [
        ["建议归档(已收口)", "原因(证据)"],
        ["E0/E3/E23/E4-DEEP + 旧协议加宽", "容量/深度/点数扫描收口:E2 定锚;加深 test −0.051 过拟合"],
        ["E2-CASE / E3-CASE(WSS/WSSmax)", "归一化口径 No-Go:R² 0.46→0.17,排序/尾部全面退化"],
        ["AG-v4 单独重跑", "相对 v3 锚点回退(pooled R² 0.26→0.19)"],
        ["Q3V(随机 SA 中心)", "相对 Q1V 全面回退(−0.024)"],
        ["半径探索 5/6 臂", "响应非单调、缩小方向全负;只留 1.2× 复核"],
        ["QAD-Lite 全系", "精确 Q2V 3 种子 −0.0048,p99 恶化(p=0.001) → No-Go"],
        ["SA1 矩阵 17 臂(ball32/KNN/cover/adaptive…)", "无一超过 random5000+ball16;覆盖率↑≠精度↑"],
        ["扩容 D1/D2/pool2025 + NormLoss A1/A2/A3/A4", "扩容三主配对全负;cohort one-hot、raw-Huber 关闭"],
    ]
    keep = [
        ["保留(待独立确认)", "下一步"],
        ["Q1V / Q2V / P2V", "独立 holdout + 3 种子重复(当前最优候选)"],
        ["Q1V 半径 1.2×", "唯一多指标同向正向(+0.0085),随候选一起复核"],
        ["n32_w64", "架构主指标第一但仅 +0.0014,须确认是否真实"],
        ["NormLoss A1b(逐父系等权)", "+0.030 全面正向,唯一归一化候选"],
    ]
    add_table(s, Inches(0.45), Inches(1.3), Inches(7.35), Inches(5.3), arch,
              col_w=[3.4, 3.9], font=10, hdr_color=RED)
    add_table(s, Inches(8.05), Inches(1.3), Inches(4.85), Inches(2.9), keep,
              col_w=[2.3, 2.5], font=10, hdr_color=GREEN)
    add_text(s, Inches(8.05), Inches(4.5), Inches(4.85), Inches(2.2), [
        dict(text="取舍原则", size=13, bold=True, color=INK, space_after=3),
        dict(text="• 单 seed + 已复用测试集 → 只能排除方向,不能定稿。", size=11, color=SEC, line=1.12),
        dict(text="• 归档≠删除数据:xlsx 总览与 runs/ 产物全部保留,只从汇报口径移出。",
             size=11, color=SEC, line=1.12, space_before=2),
        dict(text="• 新增实验先问:它改变的是「信息」还是「结构」?结构线已收口。",
             size=11, color=SEC, line=1.12, space_before=2),
    ])
    footer(s, idx)


def s_open(prs, idx):
    s = blank(prs)
    add_title(s, "开放问题与下一步", "留给讨论")
    add_text(s, Inches(0.7), Inches(1.4), Inches(12.0), Inches(5.6), [
        dict(runs=[("① 高 WSS 尾部是硬问题:", dict(size=15, bold=True, color=RED)),
                   ("high-WSS R² 全程为负(最好 −0.51),top10 幅值比 ≤0.41、p99 比 ≤0.73 —— "
                    "平均误差的改进从未转化为峰值恢复。", dict(size=13.5, color=SEC))], line=1.15),
        dict(runs=[("② 信息天花板:", dict(size=15, bold=True, color=INK)),
                   ("学习曲线约 40 例后进入 ~0.31 平台;出口 RCR 流量分配逐例不同但模型拿不到"
                    "(oracle) —— 提精度要补信息,不是继续调结构。", dict(size=13.5, color=SEC))],
             space_before=10, line=1.15),
        dict(runs=[("③ 评估纪律:", dict(size=15, bold=True, color=INK)),
                   ("test16/test27 已被多轮选择复用,失去独立确认资格;候选(Q1V/P2V/1.2×/n32_w64/A1b)"
                    "的定稿必须用独立 holdout/前瞻数据 + 3 种子。", dict(size=13.5, color=SEC))],
             space_before=10, line=1.15),
        dict(runs=[("④ 待批准的下一阶段(均 No-Run):", dict(size=15, bold=True, color=BLUE)),
                   ("P0 无泄漏开发协议 / P0b PointNeXt 主干对齐 / P1 新队列 learning curve / "
                    "P2 出口流量分配几何代理 / P3 目标改写(self-max p99 + Spearman + top-k IoU 作主指标) / "
                    "P5 合成数据。", dict(size=13.5, color=SEC))], space_before=10, line=1.15),
        dict(runs=[("⑤ 工程尾巴:", dict(size=15, bold=True, color=INK)),
                   ("Phase-V 5 组 PostView export-only 补齐;Phase-A 面积口径因严格面积映射 127/133 "
                    "未全通而冻结,修复后才能比较 AreaRandom。", dict(size=13.5, color=SEC))],
             space_before=10, line=1.15),
    ])
    footer(s, idx)


def s_appendix(prs, idx):
    s = blank(prs)
    add_title(s, "附录:保留 12 列指标速查", "定义与汇报话术(摘自 xlsx「指标说明」sheet)")
    rows = [
        ["指标", "定义", "怎么说"],
        ["物理 R²_cb", "每病例总权重=1,病例等权中心的 1−MSE/Var(Pa 空间)", "病例等权下解释了 X×100% 的总体变化"],
        ["case mean R²", "逐病例 R² 的算术平均", "平均每个病例解释多少局部变化"],
        ["负例数", "R²<0 的病例数/总数", "n/N 个病例不如直接预测均值,是稳定性风险项"],
        ["RMSE (Pa)", "pooled 均方根误差,物理单位", "RMSE 为 X Pa,对大误差敏感"],
        ["NMAE (%)", "pooled/case-mean MAE ÷ 真实极差", "平均偏差约占真实动态范围的 X%"],
        ["high-WSS R²", "每病例真值 top10% 点上 pooled R²", "高 WSS 区变化是否学好(当前全负)"],
        ["top10 幅值比", "真值 top10% 点上 均值(pred)/均值(true)", "热点幅值恢复到真实的 X×100%(<1 低估)"],
        ["p99 比", "pred p99 ÷ true p99", "极端尾部幅值恢复程度(<1 低估)"],
        ["top10 IoU", "真实/预测 top10% 点集交并比", "热点位置重叠程度(0–1)"],
        ["Spearman", "逐病例秩相关再平均", "高低排序是否一致(不看绝对幅值)"],
        ["归一化 R²_cb", "训练目标(log-z)空间的 R²_cb", "训练目标是否易拟合;不替代物理结论"],
    ]
    add_table(s, Inches(0.45), Inches(1.3), Inches(12.45), Inches(5.55), rows,
              col_w=[2.0, 5.3, 5.2], font=10.5, hdr_font=11)
    footer(s, idx)


def main():
    prs = Presentation()
    prs.slide_width = SLIDE_W
    prs.slide_height = SLIDE_H

    s_cover(prs)
    i = 2
    s_summary(prs, i); i += 1
    s_data(prs, i); i += 1
    s_fullfig(prs, i, "数据链路:从 CFD 原始结果到训练样本",
              "四阶段预处理管线(preprocess → qa-gate → global-stats → build-samples)",
              "s1_data_pipeline.png"); i += 1
    s_norm_post(prs, i); i += 1
    s_fullfig(prs, i, "模型①:PointNet 基线(E2 宽通道锚点)",
              "逐点 MLP + 病例级全局 max-pool;E2=0.79M 参数,是全部 PointNet 结论的锚点",
              "s2_pointnet_e2.png"); i += 1
    s_fullfig(prs, i, "模型②:PointNet++ SA3 —— 本轮调整的旋钮都在这张图上",
              "width(通道基数) · nsample(邻域点数) · radius(ball 半径) · SA1 分组方式 · 中心采样",
              "s3_pointnetpp_sa3.png"); i += 1
    s_fullfig(prs, i, "输入采样机制:support/query 与四种采样器",
              "vertex random5000 / fixed FPS / FPS-multistart;SAME(query=support) vs SEP(独立重采)",
              "s5_sampling.png"); i += 1
    s_fullfig(prs, i, "SA1 邻域分组机制:ball / KNN / 覆盖修复 / adaptive",
              "核心权衡:邻域覆盖率 vs 组间重叠 —— 但实验显示提高覆盖并不改善精度",
              "s4_grouping_mechanism.png"); i += 1
    s_metric_system(prs, i); i += 1
    s_stage_table(prs, i); i += 1
    s_two_figs(prs, i, "结果①:test16 基线矩阵 与 v4 数据切换",
               "容量是唯一主增量;v4 重跑回退、AAA 扩容仅小幅补偿",
               "f1_test16_baseline.png", "f2_v4_anchors.png"); i += 1
    s_fullfig(prs, i, "结果②:Phase-V 采样矩阵(test27)—— 本项目最大单项增益",
              "固定 FPS2000 → 顶点随机 5000:PointNet++ +0.055、PointNet +0.060;Q1V/P2V 晋级候选",
              "f3_phasev_test27.png"); i += 1
    s_two_figs(prs, i, "结果③:架构搜索 nsample×width 与 ball 半径缩放",
               "两条结构线均无稳定收益:第一名领先 0.0014;半径响应非单调",
               "f5_nsample_width.png", "f4_radius_scan.png"); i += 1
    s_two_figs(prs, i, "结果④:SA1 分组方式 与 support×分组矩阵(17 臂)",
               "ball16+random5000 基准未被任何替代方案超过 → 邻域机制线收口",
               "f6_sa1_grouping.png", "f7_support_matrix.png"); i += 1
    s_fullfig(prs, i, "结果⑤:数据扩容 与 归一化×尾部损失",
              "扩容(加 ILO)三主配对全负;归一化仅 A1b(逐父系等权)全面正向",
              "f8_expansion_normloss.png"); i += 1
    s_fullfig(prs, i, "结果⑥:QAD-Lite 解码器三种子配对复核",
              "val21 +0.0143 未过预注册门;精确 Q2V 协议 −0.0048 → No-Go(方差换偏差,不是精度)",
              "f9_qad_paired.png"); i += 1
    s_metric_evidence(prs, i); i += 1
    s_metric_verdict(prs, i); i += 1
    s_exp_verdict(prs, i); i += 1
    s_open(prs, i); i += 1
    s_appendix(prs, i); i += 1

    prs.save(OUT)
    print("saved ->", OUT, f"({i - 1} slides)")


if __name__ == "__main__":
    main()

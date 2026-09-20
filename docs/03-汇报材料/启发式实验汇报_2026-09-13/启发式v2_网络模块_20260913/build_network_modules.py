#!/usr/bin/env python3
"""Three editable architecture slides, verified against frozen X5/PF6/VF6 configs."""
from pathlib import Path
from pptx import Presentation
from pptx.util import Inches, Pt
from pptx.dml.color import RGBColor
from pptx.enum.shapes import MSO_SHAPE, MSO_CONNECTOR
from pptx.enum.text import PP_ALIGN, MSO_ANCHOR
from pptx.oxml.xmlchemy import OxmlElement

OUT = Path(__file__).resolve().parent
FONT = 'Noto Sans CJK SC'
W,H = 13.333333,7.5
BG='F7F9FC'; INK='16334A'; MUTED='516778'; BLUE='286DA0'; TEAL='167E7D'; PURPLE='7952A0'; ORANGE='C07B28'; LINE='B6C5D1'
prs=Presentation(); prs.slide_width=Inches(W); prs.slide_height=Inches(H)

def color(s): return RGBColor.from_string(s)
def text(sl,x,y,w,h,txt,size=14,bold=False,fill=INK,align=PP_ALIGN.LEFT):
    sh=sl.shapes.add_textbox(Inches(x),Inches(y),Inches(w),Inches(h))
    tf=sh.text_frame; tf.clear(); tf.word_wrap=True
    tf.margin_left=tf.margin_right=Inches(.02); tf.margin_top=tf.margin_bottom=Inches(.015)
    for i,line in enumerate(txt.split('\n')):
        p=tf.paragraphs[0] if i==0 else tf.add_paragraph(); p.alignment=align; p.space_after=Pt(2)
        p.line_spacing=1.02
        r=p.add_run();r.text=line;r.font.name=FONT;r.font.size=Pt(size);r.font.bold=bold;r.font.color.rgb=color(fill)
        for tag in ['a:latin','a:ea','a:cs']:
            el=OxmlElement(tag);el.set('typeface',FONT);r._r.get_or_add_rPr().append(el)
    return sh
def box(sl,x,y,w,h,title,body='',accent=BLUE,shade='E8F1F9',ts=15,bs=11.5):
    sh=sl.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE,Inches(x),Inches(y),Inches(w),Inches(h))
    sh.fill.solid();sh.fill.fore_color.rgb=color(shade);sh.line.color.rgb=color(accent);sh.line.width=Pt(1)
    sh.adjustments[0]=.08
    text(sl,x+.10,y+.09,w-.20,.37,title,ts,True,accent)
    if body:text(sl,x+.10,y+.51,w-.20,h-.56,body,bs,False,INK)
    return sh
def rect(sl,x,y,w,h,fill='FFFFFF',stroke=None):
    sh=sl.shapes.add_shape(MSO_SHAPE.RECTANGLE,Inches(x),Inches(y),Inches(w),Inches(h));sh.fill.solid();sh.fill.fore_color.rgb=color(fill)
    if stroke:sh.line.color.rgb=color(stroke)
    else:sh.line.fill.background()
    return sh
def arrow(sl,x1,y1,x2,y2,c=BLUE,dash=False,end=True,width=1.5):
    sh=sl.shapes.add_connector(MSO_CONNECTOR.STRAIGHT,Inches(x1),Inches(y1),Inches(x2),Inches(y2));sh.line.color.rgb=color(c);sh.line.width=Pt(width)
    ln=sh._element.spPr.get_or_add_ln()
    if dash:
        el=OxmlElement('a:prstDash');el.set('val','dash');ln.append(el)
    if end:
        el=OxmlElement('a:tailEnd');el.set('type','triangle');el.set('w','sm');el.set('len','sm');ln.append(el)
    return sh
def route(sl,pts,c=BLUE,dash=False):
    for i in range(len(pts)-1):arrow(sl,*pts[i],*pts[i+1],c,dash,i==len(pts)-2)
def plus(sl,x,y,d=.34,c=TEAL):
    sh=sl.shapes.add_shape(MSO_SHAPE.OVAL,Inches(x),Inches(y),Inches(d),Inches(d));sh.fill.solid();sh.fill.fore_color.rgb=color('FFFFFF');sh.line.color.rgb=color(c)
    text(sl,x,y-.005,d,d,'+',17,True,c,PP_ALIGN.CENTER)
def new(title,sub,n):
    sl=prs.slides.add_slide(prs.slide_layouts[6]);sl.background.fill.solid();sl.background.fill.fore_color.rgb=color(BG)
    rect(sl,.45,.42,.10,.53,TEAL)
    text(sl,.70,.36,12.05,.50,title,26,True)
    text(sl,.72,1.00,12,.46,sub,12.8,False,MUTED)
    rect(sl,.55,7.09,12.2,.014,LINE)
    text(sl,.62,7.17,11.8,.20,'V5 峰值帧 1162 · 原生可编辑模块图 · 配置/实现核验：2026-09-13',8.4,False,MUTED)
    text(sl,12.18,7.12,.5,.3,f'{n:02d}',11,True,TEAL,PP_ALIGN.RIGHT)
    return sl

# 1: only R4 backbone, not an assumed full X5/PF6/VF6 implementation.
s=new('共用主干：先读几何，再逐点还原血流相关特征','R4 骨架 = PointNet++ 层级采样 + PointNeXt-R 残差 + LocalGeoPE + SA2 邻域注意力。后两页标出各路线新增模块。',1)
xs=[.57,2.07,3.57,5.07,6.57,8.07,9.57,11.07]
titles=['17D 几何','Stem','SA1 + 残差','SA2 + 注意力','SA3','三级 FP','Query','WSS 头']
bodies=['5000 壁面点\n位置、尺度、\n管道坐标、法向','17 → 32 → 32\n逐点组合输入\n形成初始特征','5000 → 125 点\n64D；KNN-cover64\n保留局部几何','125 → 125 点\n128D；ball 0.10\n4 头邻域交互','125 → 32 点\n256D；ball 0.20\n压缩空间上下文','32 → 125 →\n125 → 5000 点\n回到 support 32D','Nq × 32D\n3-NN 特征插值\n查询全壁面','32 → 64 → 1\n预测 log_z\n反变换为 Pa']
for i,x in enumerate(xs):
    a=PURPLE if i==3 else TEAL if 5<=i<=6 else ORANGE if i==7 else BLUE
    sh='EEE7F5' if i==3 else 'E6F2F0' if 5<=i<=6 else 'FFF2DF' if i==7 else 'E8F1F9'
    box(s,x,1.90,1.35,1.69,titles[i],bodies[i],a,sh,13,10.7)
    if i<7:arrow(s,x+1.35,2.67,x+1.50,2.67)
route(s,[(2.74,1.90),(2.74,1.68),(8.74,1.68),(8.74,1.90)],TEAL,True)
text(s,4.24,1.46,3.60,.25,'编码器各层 skip → 对应 FP 层',10.5,False,TEAL,PP_ALIGN.CENTER)
box(s,.58,4.00,3.92,1.87,'LocalGeoPE｜邻居“在哪里”','相对位置 + 距离 + 弧长/半径/曲率差\n边 MLP 输出 → 加入 SA 消息，再聚合。\n作用：让邻域几何关系参与特征提取。',BLUE,'FFFFFF',16,13)
box(s,4.71,4.00,3.93,1.87,'PointNeXt-R｜保留信息再修正','SA1/SA2 各 1 个 InvRes：\n局部聚合残差 + C → 2C → C 逐点残差。\n作用：加强局部表达并保留原特征。',TEAL,'FFFFFF',16,13)
box(s,8.84,4.00,3.91,1.87,'SA2 注意力｜邻域内怎么关联','边 MLP + GeoPE → 4 头注意力\n→ FFN → max → 独立 InvRes 块。\n作用：聚合前让同邻域 token 交互。',PURPLE,'FFFFFF',16,13)
text(s,.65,6.08,12.1,.64,'读图边界：SA2 注意力是局部交互；R4 未启用瓶颈全局 Transformer。半径均为逐例归一化坐标，不能写成 mm。\n“结构作用”说明设计意图；“实验有用”需另看同条件消融。FP 插值的是特征，之后才回归 WSS。',12,False,MUTED)
s.notes_slide.notes_text_frame.text='来源：docs/03-汇报材料/启发式实验汇报_2026-09-13/WSS_V5_R4_模型结构_20260909/README.md 与 R4 冻结配置；实际类 PointNetPlusPlusRegressor，而非 pointnext.py 的默认网络。SA1 knn_cover=64可因补边超过64。FP输出通道依次128/64/32，3-NN为逆平方距离权重。R4 SAME训练，推理固定support+完整wall query。本页只画R4骨架。'

# 2: support residual path and output residual path deliberately separated.
s=new('直接 WSS：X5 把分支尺度信息交给局部修正网络','27D = R4 几何 17D + 壁面微分几何 8D + Murray 先验 2D。X5X11 在 X5 上再加入截面 token 上下文。',2)
box(s,.56,1.65,1.69,1.27,'27D → Stem','5000 support\n32D 初始特征',BLUE,'E8F1F9',14,12)
box(s,2.63,1.65,2.03,1.27,'三级编码器','SA1 → SA2 → SA3\n125 → 125 → 32 点',BLUE,'E8F1F9',14,11.7)
box(s,5.07,1.65,1.55,1.27,'三级 FP','回到 support\n5000 × 32D',TEAL,'E6F2F0',14,11.7)
plus(s,7.09,2.10,.42)
box(s,7.93,1.65,1.63,1.27,'3-NN 查询','独立 wall query\nNq × 32D',TEAL,'E6F2F0',14,11.5)
box(s,10.00,1.65,2.11,1.27,'基础回归头','32 → 64 → 1\n逐点 log_z 预测',ORANGE,'FFF2DF',14,12)
for a,b in [(2.25,2.63),(4.66,5.07),(6.62,7.09),(7.51,7.93),(9.56,10.00)]:arrow(s,a,2.31,b,2.31)
box(s,2.64,3.25,4.00,1.22,'方向多尺度局部分支','读取 Stem 32D；r = .015/.03/.06，各 16 邻居\n沿轴向/周向保留邻居，抑制跨壁面错误连接。',TEAL,'E6F2F0',14,11.2)
route(s,[(1.40,2.92),(1.40,3.87),(2.64,3.87)],TEAL)
route(s,[(6.64,3.87),(7.30,3.87),(7.30,2.52)],TEAL)
text(s,7.60,3.35,1.16,1.00,'32D 增量\n加到 FP\nsupport\n特征。',11.3,False,MUTED)

box(s,.56,4.93,2.42,1.22,'完整壁面 patch','每个 query 的 K=16 邻点\n27D 特征 + 相对 mm 坐标',BLUE,'E8F1F9',13.5,11.3)
box(s,3.28,4.93,2.24,1.22,'FiLM 上下文','query 32D ⊕\nSA3 max-pool 256D',PURPLE,'EEE7F5',13.5,11.4)
box(s,5.80,4.93,3.40,1.22,'Patch → FiLM → 残差','58 → 64 → 64；γ/β 调制\n均值池化 → 输出 Δlog_z',PURPLE,'EEE7F5',13.3,11.3)
box(s,9.03,3.25,3.06,1.22,'X5X11 才开启','support → 4 mm 截面 token\n2 层 Transformer → Δ256D',PURPLE,'F5F0FA',13,11.2)
plus(s,10.05,5.31,.42,ORANGE)
box(s,10.83,4.93,1.95,1.22,'WSS（Pa）','log_z 相加后\n反变换为 Pa',ORANGE,'FFF2DF',14,11.5)
route(s,[(1.77,6.15),(1.77,6.41),(6.69,6.41),(6.69,6.15)],BLUE)
arrow(s,5.52,5.52,5.80,5.52,PURPLE)
arrow(s,9.20,5.52,10.05,5.52,ORANGE)
route(s,[(12.11,2.31),(12.48,2.31),(12.48,4.61),(10.26,4.61),(10.26,5.31)],ORANGE)
arrow(s,10.47,5.52,10.83,5.52,ORANGE)
route(s,[(9.03,4.17),(8.79,4.17),(8.79,4.72),(4.43,4.72),(4.43,4.93)],PURPLE,True)
text(s,5.83,4.47,2.95,.23,'Δ256D 加入粗层上下文',9.2,False,PURPLE,PP_ALIGN.CENTER)
text(s,.61,6.62,12.15,.30,'第二处相加：基础 log_z + patch 残差，再反变换为 Pa。截面 token 只改变 FiLM 条件，不直接加到 WSS。',11.5,False,MUTED)
s.notes_slide.notes_text_frame.text='精确连线：Stem→SA→FP；LocalWallBranch读取Stem features，附法向属性14–16与raw geometry，在FP后加到support 32D。SA3输出max pool得到病例256D。X5X11的SectionContext从修正后的support32D、原输入27D及segment/s_local池化，在每例token之间做2层4头128D Transformer，投影256D，按query截面lookup并加到病例256D。query32D与这一256D拼接为288D，映射γ/β调制patch。Patch输入为neighbor27D+query27D+relative4D=58D，mean pool64D，out64→1输出残差。默认patch_frame=atlas、pool=mean；方向邻域仅用于support局部分支，不应把X5画成tangent-frame query patch。Murray log_q/log_tau0为几何估计输入，非CFD分流输入或PDE。底部布局用标签引用32/256D来源，避免跨图连线混乱。'

# 3: distinct models, single radius inside VF6.
s=new('体场：PF6 与 VF6 共用骨架，但采用不同的上下文模块','两个独立训练的模型；输入均为 20D = R4 几何 17D + 到壁距离 1D + Murray 先验 2D。速度分量由 VF6 直接输出。',3)
text(s,.60,1.61,1.20,.50,'PF6 压力',17,True,BLUE)
text(s,.60,3.29,1.20,.50,'VF6 速度',17,True,TEAL)
rows=[(1.93,BLUE,['Stem + SA1/2','普通 SA3','瓶颈全局 BT','三级 FP','QAD-lite','p 相对压力'],['壁面 support 5000\n含 SA2 局部注意力','32 token × 256D\nball r = .20','SA3 后追加\n1 层 / 8 头','恢复 support 32D\n无局部分支','任意 query 定位\n插值 + 门控残差','独立 1 通道头\n壁面 + 内部 query']),
      (3.61,TEAL,['Stem + SA1/2','SA3 内置 BT','三级 FP + 局部','QAD-lite','u / v / w','速度 → WSS'],['壁面 support 5000\n含 SA2 局部注意力','单半径 r = .20\n32 token × 256D','普通三尺度分支\n加回 support 32D','任意 query 定位\n插值 + 门控残差','独立 3 通道头\n内部 query','另外运行冻结算子\n当前闭环待补'])]
for y,a,tt,bb in rows:
    for i in range(6):
        x=.60+i*2.045
        shade='FFF2DF' if i==5 else 'E8F1F9' if a==BLUE else 'E6F2F0'
        box(s,x,y,1.90,1.16,tt[i],bb[i],ORANGE if i==5 else a,shade,12.8,10.5)
        if i<5:arrow(s,x+1.90,y+.55,x+2.045,y+.55,a)
text(s,.65,4.92,12.10,.34,'VF6 局部分支从 Stem 读取 32D 特征，r = .015/.03/.06，各 16 邻居；未开启方向邻域，也未开启 X5 的 patch FiLM。',11.5,False,MUTED)
rect(s,.59,5.43,12.18,1.28,'FFFFFF',LINE)
text(s,.80,5.58,2.26,.38,'QAD-lite 补偿什么？',15,True,PURPLE)
text(s,.82,6.05,2.19,.46,'query 不在 support 上，\n需补偿插值遗漏的位置差。',11.5,False,INK)
box(s,3.23,5.62,3.17,.85,'相对输入 24D','Δ20D 特征 + Δxyz + 距离',PURPLE,'EEE7F5',13,10.7)
box(s,6.74,5.62,2.91,.85,'MLP × 特征门控','24 → 32 → 32；门控来自 query32D',PURPLE,'EEE7F5',13,10.0)
box(s,10.00,5.62,2.55,.85,'输出残差相加','基础头预测 + Δp 或 Δu,v,w',ORANGE,'FFF2DF',13,10.7)
arrow(s,6.40,6.05,6.74,6.05,PURPLE);arrow(s,9.65,6.05,10.00,6.05,PURPLE)
s.notes_slide.notes_text_frame.text='PF6/VF6配置：configs/wss_local_wave2_20260912/{PF6,VF6}_s1234.json，三seed后续同结构。PF6 bottleneck_transformer=true，local_branch=false，multiradius_values=[]；VF6 bottleneck_transformer=false，multiradius_values=[0.2]，multiradius_bottleneck=true，local_branch=true，directional默认false。VF6因此存在SA3内置单半径branch BT，不能据multiradius类名画三半径SA3。两者QAD: x_query - interp(input_support20D), pos_query-interp(pos_support3D), norm(delta_pos)合为24D，经24→32→32，与Sigmoid(linear(query_feature32D))相乘，线性输出1或3通道残差加到基础头。QAD不是PDE模块，也不是已证实提升近壁梯度的结构。压力和速度各自训练，非共享权重多任务头。'

# 4: finished v5.1 X5D; density augmentation is training-only.
s=new('X5D：推理结构与 X5 相同，训练时加入多密度视图','v5.1 数据更新后重训完成的部署底座。27D 输入与 X5 一致；密度增广只替换训练点云视图，不增加推理模块。不是纵向几何 X5D_long。',4)
box(s,.56,1.65,1.69,1.27,'27D → Stem','5000 support\n32D 初始特征',BLUE,'E8F1F9',14,12)
box(s,2.63,1.65,2.03,1.27,'三级编码器','SA1 → SA2 → SA3\n125 → 125 → 32 点',BLUE,'E8F1F9',14,11.7)
box(s,5.07,1.65,1.55,1.27,'三级 FP','回到 support\n5000 × 32D',TEAL,'E6F2F0',14,11.7)
plus(s,7.09,2.10,.42)
box(s,7.93,1.65,1.63,1.27,'3-NN 查询','独立 wall query\nNq × 32D',TEAL,'E6F2F0',14,11.5)
box(s,10.00,1.65,2.11,1.27,'基础回归头','32 → 64 → 1\n逐点 log_z 预测',ORANGE,'FFF2DF',14,12)
for a,b in [(2.25,2.63),(4.66,5.07),(6.62,7.09),(7.51,7.93),(9.56,10.00)]:arrow(s,a,2.31,b,2.31)
box(s,.56,3.22,12.22,1.18,'训练期密度增广（X5D 新增，推理时关闭）','以 p=0.6 抽取 70 / 50 / 35 / 25% 密度视图替换 support/query 点云；测试与部署仍用全密度。这是训练采样策略，不是第四条残差通路。',ORANGE,'FFF2DF',15,12.2)
box(s,2.64,4.68,4.00,1.22,'方向多尺度局部分支','读取 Stem 32D；r = .015/.03/.06\n与 X5 相同，加到 FP 后的 support。',TEAL,'E6F2F0',14,11.2)
route(s,[(1.40,2.92),(1.40,5.30),(2.64,5.30)],TEAL)
route(s,[(6.64,5.30),(7.30,5.30),(7.30,2.52)],TEAL)
box(s,.56,6.08,2.42,0.90,'完整壁面 K16 patch','27D + 相对 mm 坐标',BLUE,'E8F1F9',13,11.0)
box(s,3.28,6.08,3.40,0.90,'Patch → FiLM → Δlog_z','query32D ⊕ SA3 256D 调制',PURPLE,'EEE7F5',13,11.0)
plus(s,7.10,6.32,.38,ORANGE)
box(s,7.70,6.08,2.20,0.90,'WSS（Pa）','残差相加后反变换',ORANGE,'FFF2DF',13,11.0)
arrow(s,2.98,6.53,3.28,6.53,BLUE)
arrow(s,6.68,6.53,7.10,6.53,PURPLE)
arrow(s,7.48,6.53,7.70,6.53,ORANGE)
box(s,10.10,4.68,2.65,1.22,'默认关闭','X11 截面 token\n不在 X5D 底座里',PURPLE,'F5F0FA',13,11.0)
s.notes_slide.notes_text_frame.text='配置：training_wss_min/configs/wss_v51_wave1_20260916/X5D_v51_s1234.json。input_features 27D 与 X5 相同；density_aug_levels=[70,50,35,25]，density_aug_prob=0.6。query_patch_film=true，local_branch_directional=true。没有 section context。纵向几何 32 通道属于未完成的 X5D_long，本页不画。'

OUT.mkdir(parents=True,exist_ok=True)
for i,slide in enumerate(prs.slides,1):
    for sh in slide.shapes:
        assert sh.left >= 0 and sh.top >= 0 and sh.left+sh.width <= prs.slide_width and sh.top+sh.height <= prs.slide_height, (i,sh.name)
path=OUT/'启发式v2_网络模块补充_20260913.pptx';prs.save(path)
print(path)

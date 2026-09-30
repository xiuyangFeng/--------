"""生成讲解网页（docs/03-汇报材料/）：HTML 几十 KB，三张图复制到同名「_图」文件夹按相对路径引用；用浏览器打开 HTML。改文字改这里再重跑。"""
import base64
from pathlib import Path
A = Path(__file__).resolve().parent
OUT = Path('/public/newhome/cy/Digital_twin/GNN/docs/03-汇报材料/WSS_全周期_减速期与谷底为何难预测_导师讲解材料_2026-09-22.html')
import shutil
IMGDIR = OUT.with_name(OUT.stem + '_图')   # 图放旁边文件夹，HTML 只引用相对路径（base64 单文件 2.2 MB 会超出 Orca 的读取限制）
IMGDIR.mkdir(exist_ok=True)
def img(name):
    shutil.copyfile(A / name, IMGDIR / name)
    return f'{IMGDIR.name}/{name}'
figA, figB, figC = img('fig_teacher_A_phenomenon.png'), img('fig_why_decel_trough.png'), img('fig_teacher_C_womersley.png')
html = f'''<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>减速期与谷底为何难预测</title>
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=Noto+Serif+SC:wght@600;700&family=Noto+Sans+SC:wght@400;500;700&family=IBM+Plex+Mono:wght@400;500&display=swap">
<style>
:root {{
  color-scheme: light;
  --bg:#f7f7f4; --ink:#1a1d21; --ink-2:#545b63; --rule:#dcdfe2; --accent:#1c5cab; --accent-soft:#e4edf9; --callout:#eef1f5; --figbg:#fcfcfb;
  --s-blue:#2a78d6; --s-orange:#eb6834; --s-green:#1baf7a; --s-yellow:#eda100;
  --serif:"Noto Serif SC","Songti SC","SimSun",serif; --sans:"Noto Sans SC","PingFang SC","Microsoft YaHei",system-ui,sans-serif; --mono:"IBM Plex Mono","SFMono-Regular",Menlo,monospace;
}}
@media (prefers-color-scheme: dark) {{ :root:not([data-theme="light"]) {{ color-scheme: dark; --bg:#16191d; --ink:#eceef1; --ink-2:#aab2bc; --rule:#2d3239; --accent:#6da7ec; --accent-soft:#1f2c3d; --callout:#1d2229; --figbg:#fcfcfb; }} }}
:root[data-theme="dark"] {{ color-scheme: dark; --bg:#16191d; --ink:#eceef1; --ink-2:#aab2bc; --rule:#2d3239; --accent:#6da7ec; --accent-soft:#1f2c3d; --callout:#1d2229; --figbg:#fcfcfb; }}
body {{ margin:0; background:var(--bg); color:var(--ink); font-family:var(--sans); font-size:15.5px; line-height:1.75; }}
.page {{ max-width:1120px; margin:0 auto; padding:40px 28px 80px; }}
.eyebrow {{ font-family:var(--mono); font-size:12px; letter-spacing:.12em; text-transform:uppercase; color:var(--ink-2); }}
h1 {{ font-family:var(--serif); font-weight:700; font-size:34px; line-height:1.3; margin:8px 0 6px; text-wrap:balance; }}
.sub {{ color:var(--ink-2); max-width:70ch; margin:0 0 28px; }}
.lead {{ border-left:4px solid var(--accent); background:var(--accent-soft); padding:16px 22px; margin:0 0 40px; font-family:var(--serif); font-size:19px; line-height:1.7; font-weight:600; max-width:60em; text-wrap:balance; }}
section {{ margin:0 0 56px; }}
h2 {{ font-family:var(--serif); font-weight:600; font-size:24px; margin:0 0 6px; padding-top:18px; border-top:1px solid var(--rule); text-wrap:balance; }}
h2 .n {{ font-family:var(--mono); font-size:13px; color:var(--accent); letter-spacing:.1em; display:block; margin-bottom:4px; font-weight:500; }}
p, li {{ max-width:70ch; }}
figure {{ margin:22px 0 10px; }}
figure img {{ display:block; width:100%; height:auto; background:var(--figbg); border:1px solid var(--rule); border-radius:4px; }}
figcaption {{ font-size:13px; color:var(--ink-2); margin-top:8px; max-width:90ch; }}
.talk {{ background:var(--callout); border-radius:6px; padding:16px 22px 6px; margin:18px 0 22px; }}
.talk h3 {{ margin:0 0 6px; font-size:13px; font-family:var(--mono); letter-spacing:.1em; text-transform:uppercase; color:var(--accent); font-weight:500; }}
.talk ol {{ margin:0 0 10px 1.2em; padding:0; }}
.talk li {{ margin:0 0 10px; }}
.tbl {{ overflow-x:auto; margin:14px 0 8px; }}
table {{ border-collapse:collapse; font-size:13.5px; font-variant-numeric:tabular-nums; white-space:nowrap; }}
th, td {{ padding:6px 12px; border-bottom:1px solid var(--rule); text-align:right; }}
th {{ font-weight:500; color:var(--ink-2); font-family:var(--mono); font-size:12px; letter-spacing:.04em; border-bottom:2px solid var(--rule); }}
td:first-child, th:first-child, td:nth-child(2), th:nth-child(2) {{ text-align:left; }}
tr.hl td {{ background:var(--accent-soft); }}
.note {{ font-size:13px; color:var(--ink-2); max-width:90ch; }}
.qa {{ display:grid; grid-template-columns:1fr; gap:14px; max-width:80ch; }}
.qa .q {{ font-weight:700; }} .qa .q::before {{ content:"问 "; font-family:var(--mono); color:var(--accent); font-weight:500; }}
.qa .a::before {{ content:"答 "; font-family:var(--mono); color:var(--ink-2); }}
.qa div {{ padding:12px 16px; border-left:3px solid var(--rule); }}
.sw {{ display:inline-block; width:10px; height:10px; border-radius:2px; vertical-align:-1px; margin-right:6px; }}
strong {{ font-weight:700; }}
a {{ color:var(--accent); }} a:focus-visible {{ outline:2px solid var(--accent); outline-offset:2px; }}
@media (prefers-reduced-motion: reduce) {{ * {{ scroll-behavior:auto; }} }}
</style>
</head>
<body>
<div class="page">
<div class="eyebrow">WSS · 全周期时间线 · 导师讲解材料 · 2026-09-22</div>
<h1>为什么全周期 WSS 在减速期和谷底难预测</h1>
<p class="sub">数据：cv3 三折留出（136 例）、Pa 空间、ckpt best；真值结构量用 136 例训练集 CFD 真值算。全部只读已有产物，未训练、未改代码。产物目录 <code>training_wss_min/experiments/wss_time_ecc_20260918/analysis_20260921/</code>。</p>
<div class="lead">掉的不是幅值，是型态。减速期和谷底的 WSS 场不再由「当下流量 × 几何」决定，而由压力梯度历史（惯性、回流、出口间再分配）决定。几何输入能解释的份额从峰值的 0.77 掉到谷底的 0.41，模型的逐帧 r² 贴着这个天花板走（秩相关 0.96）。</div>

<section>
<h2><span class="n">01 · 现象</span>三种口径在减速期和谷底一起掉</h2>
<figure><img src="{figA}" alt="Figure A. Inlet waveform and frame-wise R²_cb, Pearson r², and Spearman for four arms."><figcaption>Figure A. <span class="sw" style="background:var(--s-blue)"></span>X5D_v51 peak model, frozen (no time input). <span class="sw" style="background:var(--s-orange)"></span>T-null: peak prediction × cohort time shape (untrained). <span class="sw" style="background:var(--s-green)"></span>T0: phase input, trained on all frames. <span class="sw" style="background:var(--s-yellow)"></span>TB8: temporal-basis head. Grey band = peak window (10 frames); light orange = trough quartile (20 frames).</figcaption></figure>
<div class="tbl"><table>
<tr><th>臂</th><th>相位</th><th>MAE (Pa)</th><th>真值均值 (Pa)</th><th>R²_cb</th><th>Pearson r²</th><th>Spearman</th><th>CCC</th><th>approx. disparity</th></tr>
<tr><td>X5D_v51 峰值冻结</td><td>峰值窗</td><td>1.53</td><td>4.31</td><td>0.699</td><td>0.710</td><td>0.891</td><td>0.828</td><td>0.483</td></tr>
<tr><td>X5D_v51 峰值冻结</td><td>减速</td><td>3.54</td><td>0.47</td><td>−131</td><td>0.496</td><td>0.604</td><td>0.098</td><td>8.68</td></tr>
<tr><td>X5D_v51 峰值冻结</td><td>谷底</td><td>3.49</td><td>0.55</td><td>−113</td><td>0.290</td><td>0.405</td><td>0.085</td><td>7.71</td></tr>
<tr><td>T-null</td><td>峰值窗</td><td>1.45</td><td>4.31</td><td>0.700</td><td>0.707</td><td>0.891</td><td>0.816</td><td>0.484</td></tr>
<tr><td>T-null</td><td>减速</td><td>0.22</td><td>0.47</td><td>0.465</td><td>0.477</td><td>0.604</td><td>0.631</td><td>0.596</td></tr>
<tr><td>T-null</td><td>谷底</td><td>0.31</td><td>0.55</td><td>0.222</td><td>0.263</td><td>0.405</td><td>0.429</td><td>0.706</td></tr>
<tr><td>T0</td><td>峰值窗</td><td>1.46</td><td>4.31</td><td>0.684</td><td>0.697</td><td>0.897</td><td>0.816</td><td>0.495</td></tr>
<tr><td>T0</td><td>减速</td><td>0.18</td><td>0.47</td><td>0.525</td><td>0.547</td><td>0.743</td><td>0.663</td><td>0.562</td></tr>
<tr class="hl"><td>T0</td><td>谷底</td><td>0.24</td><td>0.55</td><td>0.322</td><td>0.372</td><td>0.613</td><td>0.429</td><td>0.661</td></tr>
</table></div>
<p class="note">TB8 / TB16 与 T0 同形态，谷底 Spearman 0.61；全表含加速段、top10 IoU、最差帧，见 <code>frame_metric_suite_best.md</code>。</p>
<div class="talk"><h3>怎么讲</h3><ol>
<li>先看蓝线：峰值模型不带时间输入，直接拿它的峰值预测去对非峰值帧，R²_cb 掉到 −100 以下。这不是模型错，是峰值场的幅值在低流量帧大 6 到 10 倍。</li>
<li>把幅值按群体时间形状缩回去（T-null，不训练），R²_cb 回到 0.46；但 Spearman 与冻结版逐位相同（谷底 0.405），因为单调缩放不改排序。这一对说明：<strong>冻结底座的问题全是幅值</strong>，剩下的 0.405 才是型态本身的差距。</li>
<li>T0 加相位输入、全帧训练，谷底 Spearman 从 0.405 提到 0.613，r² 从 0.26 到 0.37，<strong>增益全在型态</strong>；但仍只解释不到四成的空间方差。三个口径一起掉，所以谷底的问题是型态，不是幅值。</li>
<li>MAE 列看起来谷底更小（0.24 Pa 对峰值 1.46 Pa），那只是场本身从 4.3 Pa 缩到 0.55 Pa；除以均值后 0.34 → 0.44，方向与 r² 一致。单看 MAE 或 NMAE 会误判。</li>
</ol></div>
</section>

<section>
<h2><span class="n">02 · 先排除三个常见解释</span>幅值、地板噪声、模型容量都不是主因</h2>
<div class="tbl"><table>
<tr><th>假设</th><th>证据</th><th>结论</th></tr>
<tr><td>幅值小把 R² 拉下去</td><td style="text-align:left;white-space:normal;max-width:60ch">Gupta 分解：R² − r² 差距峰值 0.013、谷底 0.05；Spearman 同样从 0.90 掉到 0.61</td><td style="text-align:left">否，幅值只贡献 0.05</td></tr>
<tr><td>低 WSS 落到地板 / CFD 噪声</td><td style="text-align:left;white-space:normal;max-width:60ch">|τ| &lt; 0.05 Pa 份额全周期 ≤ 4%，与逐帧 r² 相关 −0.08；&lt; 0.4 Pa 份额剔峰后相关 −0.23</td><td style="text-align:left">否，不是主因</td></tr>
<tr><td>模型容量 / 归一化 / 选模</td><td style="text-align:left;white-space:normal;max-width:60ch">帧条件归一化的 T0、容量更大的 TB8/TB16、不训练的 T-null 三条曲线同形态（相关 &gt; 0.95），谷底 Spearman 都停在 0.61；D1：K=8 时间基对真值谷底可重建到 0.97，时间维本身低秩</td><td style="text-align:left">否，天花板是信息性的</td></tr>
</table></div>
</section>

<section>
<h2><span class="n">03 · 机理 · 数据</span>模型能学到的，就是几何还能解释的那部分</h2>
<figure><img src="{figB}" alt="Figure B. Ground-truth c_peak, c_murray, reversed fraction, and small-scale share versus T0 r², with a scatter plot and a matched-flow pair."><figcaption>Figure B. CFD truth from 136 training cases. c_peak = spatial correlation with the peak field; c_murray = correlation with the Murray geometric/flow-split prior; rev_frac = wall fraction whose WSS is reversed relative to the peak; within4mm = share of ln|τ| variance inside 4 mm segments (small scale).</figcaption></figure>
<div class="tbl"><table>
<tr><th>相位</th><th>c_peak</th><th>c_murray</th><th>反向份额</th><th>小尺度份额</th><th>地板以下</th><th>T0 r²</th><th>T0 Spearman</th></tr>
<tr><td>加速 22 帧</td><td>0.726</td><td>0.682</td><td>0.228</td><td>0.258</td><td>0.005</td><td>0.637</td><td>0.824</td></tr>
<tr><td>峰值窗 10 帧</td><td>0.950</td><td>0.771</td><td>0.030</td><td>0.240</td><td>0.002</td><td>0.697</td><td>0.897</td></tr>
<tr><td>减速 29 帧</td><td>0.612</td><td>0.564</td><td>0.179</td><td>0.274</td><td>0.043</td><td>0.547</td><td>0.743</td></tr>
<tr class="hl"><td>谷底 20 帧</td><td>0.418</td><td>0.409</td><td>0.404</td><td>0.374</td><td>0.030</td><td>0.372</td><td>0.613</td></tr>
</table></div>
<p class="note">81 帧跨帧秩相关：T0 r² 对 c_peak 0.96、对 c_murray 0.94、对反向份额 −0.87、对小尺度份额 −0.72、对地板份额 −0.29。T0 Spearman 对 c_murray 的线性相关 0.99。</p>
<div class="tbl"><table>
<tr><th>同流量配对（Q = 0.16 Q_max）</th><th>加速帧 10</th><th>减速帧 42</th></tr>
<tr><td>c_peak</td><td>0.82</td><td>0.36</td></tr>
<tr><td>c_murray</td><td>0.81</td><td>0.34</td></tr>
<tr><td>反向份额</td><td>0.11</td><td>0.56</td></tr>
<tr><td>小尺度份额</td><td>0.20</td><td>0.47</td></tr>
<tr><td>T0 r²</td><td>0.745</td><td>0.343</td></tr>
<tr><td>T0 Spearman</td><td>0.876</td><td>0.554</td></tr>
</table></div>
<div class="talk"><h3>怎么讲</h3><ol>
<li>图 B ① 三条线几乎重合：模型的逐帧 r²（绿）跟着真值「还有多少能被峰值场 / 几何先验解释」（蓝、橙）走。几何解释力从峰值 0.77 掉到谷底 0.41，模型就跟着掉。</li>
<li>图 B ④ 是最干净的一组：同一个入口流量 0.16，加速时场与峰值场相关 0.82、只有 11% 壁面反向；减速时相关掉到 0.36、56% 壁面反向。<strong>流量一样，结果完全不同，差的不是流量小，是流动的方向和历史。</strong></li>
<li>图 B ② 谷底 40% 的壁面 WSS 方向反了，小尺度份额从 0.24 升到 0.37：场变成斑块状，反向边界处 |τ| 趋近 0，在 log 空间是一条条随时间移动的「沟」，点几何回归器放不准这些沟的位置。</li>
<li>地板份额（绿虚线）全程贴着 0，所以不是数据落到下限的问题。</li>
</ol></div>
</section>

<section>
<h2><span class="n">04 · 机理 · 理论示意</span>脉动流的惯性让壁面剪切领先于流量反向</h2>
<figure><img src="{figC}" alt="Figure C. Womersley velocity profiles, wall shear versus flow, and the τ–Q hysteresis loop at R = 8 mm and 4 mm."><figcaption>Figure C. Rigid straight-tube Womersley solution driven by 12 Fourier harmonics of the shared protocol inlet waveform (172 cases); ν = 3.5e-6 m²/s, T = 0.8 s. Schematic only, not a CFD result. Real bifurcations and the aneurysm lumen also add separation and secondary flow.</figcaption></figure>
<div class="tbl"><table>
<tr><th>直管解析解</th><th>8 mm（α ≈ 12，腹主动脉量级）</th><th>4 mm（α ≈ 6，髂动脉量级）</th></tr>
<tr><td>壁面 τ 首次转负的帧 / 该帧 Q/Q_max</td><td>帧 4 / 0.13</td><td>帧 35 / 0.43</td></tr>
<tr><td>一个周期内 τ &lt; 0 的帧数</td><td>33</td><td>18</td></tr>
</table></div>
<div class="talk"><h3>怎么讲</h3><ol>
<li>这不是 CFD，是解析解，只为说明机理。脉动流里近壁 Stokes 层薄，对压力梯度的响应比核心流快；流量一开始下降，逆压梯度先把近壁流反过来，核心还在向前。所以壁面剪切<strong>领先</strong>于流量，α 越大越接近跟着 dQ/dt 走。</li>
<li>图 C ③ / ⑥ 的 τ–Q 环：上支是加速、下支是减速，同一个 Q 对应两个 τ，减速时更低甚至为负。这正是图 B ④ 在 CFD 真值里看到的滞后。</li>
<li>4 mm 管在 Q 还有 0.43 时壁面剪切已经反向。我们中位病例的减速帧 35（Q = 0.43）正好有 41% 壁面反向，数据和理论对上了。</li>
<li>真实几何里再叠加分叉分离、瘤腔涡、出口间通过 RCR 电容放电的再分配（内 / 外髂参数逐例不同，不在输入里）。谷底入口净流量只有峰值的 1.7%，壁面剪切几乎全是这些「历史项」贡献的。</li>
</ol></div>
</section>

<section>
<h2><span class="n">05 · 中位病例解剖</span>误差不集中在反向区，是整张图的型态变了</h2>
<div class="tbl"><table>
<tr><th>SHEN_FANG_JIN（fold2 留出）</th><th>真值小尺度份额</th><th>壁面反向份额</th><th>T0 ln 相关</th><th>反向区 SSE 份额 / 点份额</th><th>反向区内 / 外相关</th></tr>
<tr><td>加速 帧 13</td><td>0.09</td><td>0.07</td><td>0.964</td><td>0.08 / 0.07</td><td>0.87 / 0.97</td></tr>
<tr><td>峰值 帧 21</td><td>0.17</td><td>0.00</td><td>0.945</td><td>—</td><td>— / 0.95</td></tr>
<tr><td>减速 帧 35</td><td>0.21</td><td>0.41</td><td>0.874</td><td>0.38 / 0.41</td><td>0.82 / 0.86</td></tr>
<tr class="hl"><td>谷底 帧 48</td><td>0.35</td><td>0.63</td><td>0.750</td><td>0.57 / 0.63</td><td>0.77 / 0.71</td></tr>
</table></div>
<div class="talk"><h3>怎么讲</h3><ol><li>反向区的误差份额和它的点份额差不多，区内相关也不比区外差。难的不是「反向区里预测不准」，而是整张图的型态在低流量时不再由几何决定，反向份额只是这个状态的标志。</li></ol></div>
</section>

<section>
<h2><span class="n">06 · 老师可能追问</span></h2>
<div class="qa">
<div><div class="q">谷底型态差，是不是因为 X5D 只用峰值帧训练？</div><div class="a">部分是，主要不是。只用峰值帧训练的 X5D 冻结后谷底 Spearman 0.405，恰好等于峰值场与谷底场本来的相关 0.42：峰值模型只能给你峰值型态。在全部 81 帧上训练的 T0 / TB8 / TB16 抬到 0.61，这 0.2 是训练帧带来的。但三个全帧模型无论架构都停在 0.61，离峰值的 0.90 还差 0.3：模型输入只有几何和 172 例共享的时间特征，它在谷底帧能做的最好的事是输出「这种几何下谷底场的平均样子」，即几何能解释的那 0.41。峰值场 ≈ 当下流量 × 局部几何，Murray 先验就能解释 0.77；谷底场 = 惯性反向 + 瘤腔残余涡 + 出口间 RCR 再分配，是全局几何和流动历史的复杂函数，且髂内外分流不在输入里。</div></div>
<div><div class="q">能不能改网络结构来提高谷底帧？</div><div class="a">不建议。三种结构（T0 相位输入、TB8 / TB16 时间基头）谷底 Spearman 全停在 0.61，与不训练的 T-null 同形态，卡的是输入信息不是容量；时间维本身低秩（K=8 重建真值谷底 0.97），难点在非峰值模态的空间系数图；峰值帧上全局注意力、感受野、扇区 token 等结构线已验证增益在噪声内。有机理的只有切向向量输出和更大全局上下文两条，但谷底帧本身不是临床量，没有部署去向。周期量已给出答案：TAWSS 三 seed 集成归一化 0.88 与峰值持平，OSI 0.57、滞留区 0.56，M1 三头单模型已部署。若还要动结构，目标应是 OSI，判据用 OSI &gt; 0.1 IoU 和滞留区 IoU，不用谷底 R²。</div></div>
<div><div class="q">那是不是数据太少？</div><div class="a">学习曲线每翻倍 +0.03，那是峰值帧口径。谷底的问题是输入里没有决定它的量（流动历史、逐例出口动力学），加数据不会把几何解释力从 0.41 抬起来。</div></div>
<div><div class="q">换个更大的模型，或加时间基呢？</div><div class="a">TB8 / TB16 容量更大、带 K=8 时间基头，谷底 Spearman 与 T0 一样 0.61。时间维本身低秩（K=8 重建 0.97），难在非峰值时间模态的空间系数图。</div></div>
<div><div class="q">输入里加流量历史行不行？</div><div class="a">部署只有点云和中心线，拿不到压力史和逐例 RCR 参数。入口波形又是 172 例共享的协议波形，4 个时间特征对所有病例相同，模型只能学群体平均形变。</div></div>
<div><div class="q">那临床量怎么办？</div><div class="a">走周期积分量。TAWSS 把过渡态积掉，归一化 R² 0.86 到 0.88 与峰值帧持平；OSI 的空间结构恰恰由「哪里流弱、哪里反向」决定（与 ln TAWSS 逐点 Spearman −0.74），三 seed 集成 0.57，比逐帧谷底场好学。</div></div>
<div><div class="q">逐帧谷底还要不要做？</div><div class="a">若一定要，只剩方向 / 符号输出（向量头）这条路。不建议再加容量、加 K、补 seed。</div></div>
</div>
</section>

<section>
<h2><span class="n">07 · 汇报口径</span></h2>
<p>主指标 R²_cb 并列 Pearson r²；型态单独说用 Spearman；approximation disparity 只为与 Suk / Rygiel 对表（峰值 0.48 到 0.50、全周期 0.57，对 Rygiel 2025 AAA 的 0.51 到 0.57）；CCC 可不报；MAE / NMAE 写明分母，÷ 周期 max 的版本不要单独出现。把 c_peak 或 c_murray 当「可预测性参照曲线」画在模型旁边，说明模型贴着天花板走。</p>
<p class="note">脚本：frame_metric_suite.py（指标套件）、why_decel_trough.py（真值结构）、case_error_anatomy.py（单例解剖）、plot_teacher_phenomenon.py / plot_why_decel_trough.py / plot_teacher_womersley.py（三图）。训练跟踪文档 §30.9 / §30.10。</p>
</section>
</div>
</body>
</html>
'''
OUT.write_text(html, encoding='utf-8'); print('wrote', OUT, f'{OUT.stat().st_size/1e3:.0f} KB', '+', IMGDIR.name, sorted(p.name for p in IMGDIR.iterdir()))

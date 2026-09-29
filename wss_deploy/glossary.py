"""Single source of the terminology shown in both reports, the one-page summary and the workbench (C17).

``python -m wss_deploy.glossary`` writes ``static/glossary.json``; ``tests/test_glossary.py`` keeps the
JSON, this dictionary and the ``data-gloss`` keys used by the report templates in step.
"""
from __future__ import annotations

import json
from pathlib import Path

from .paths import STATIC_DIR

SCHEMA_VERSION = "wss-deploy.glossary/v1"

GLOSSARY: dict[str, dict[str, str]] = {
    "p99": {"zh": "空间 p99", "zh_desc": "预测点云上 99% 分位的 WSS，作为主峰值指标；比最大值稳定，不受个别离群点影响。",
            "en": "Spatial p99", "en_desc": "99th percentile of WSS over the prediction point cloud; the primary peak metric, more stable than the maximum."},
    "max": {"zh": "全场最大值", "zh_desc": "预测点云上的单点最大 WSS，仅作参考并标出位置；与 p99 不是同一统计量。",
            "en": "Field maximum", "en_desc": "Largest single-point WSS in the prediction cloud, shown as a reference with its location; not the same statistic as p99."},
    "mean": {"zh": "均值", "zh_desc": "预测点云上的等权平均 WSS。",
             "en": "Mean", "en_desc": "Equal-weight average WSS over the prediction points."},
    "area_fraction": {"zh": "面积占比（估计）", "zh_desc": "低 / 高 / 极高 WSS 的点数占比乘以输入壁面面积，是估计值，不是三角面精确积分。",
                      "en": "Area fraction (estimate)", "en_desc": "Share of prediction points below/above a threshold multiplied by the input wall area; an estimate, not an exact surface integral."},
    "area_weighted_p99": {"zh": "面积加权 p99", "zh_desc": "把插值到 STL 三角面上的 WSS 按三角面面积加权后取的 99% 分位；只在插值完全覆盖的三角面上统计。",
                          "en": "Area-weighted p99", "en_desc": "99th percentile of WSS interpolated onto the STL triangles, weighted by triangle area; computed only on fully covered triangles."},
    "thresholds": {"zh": "阈值 0.4 / 4 / 7 Pa", "zh_desc": "低 WSS < 0.4 Pa、高 WSS > 4 Pa、极高 WSS > 7 Pa；报告里可改，面积占比即时重算，不改 summary.json。",
                   "en": "Thresholds 0.4 / 4 / 7 Pa", "en_desc": "Low WSS < 0.4 Pa, high > 4 Pa, very high > 7 Pa; editable in the report, area fractions are recomputed on the fly."},
    "units_wss": {"zh": "WSS 单位", "zh_desc": "1 Pa = 10 dyn/cm²；换算只在显示层。",
                  "en": "WSS units", "en_desc": "1 Pa = 10 dyn/cm²; conversion is display-only."},
    "relative_pressure": {"zh": "相对压力", "zh_desc": "预测压力减去当前峰值帧体积平均压力；不能恢复绝对血压，只用于比较压差。",
                          "en": "Relative pressure", "en_desc": "Predicted pressure minus the volume-mean pressure of the peak frame; absolute blood pressure cannot be recovered, use it for differences only."},
    "delta_p": {"zh": "ΔP 分支压差", "zh_desc": "分支近端 10% 与远端 10% 弧长段内部点的平均压力之差（近端 − 远端）。",
                "en": "ΔP branch pressure drop", "en_desc": "Mean pressure of interior points in the proximal 10 % arc-length segment minus that of the distal 10 % segment."},
    "pressure_units": {"zh": "压力单位", "zh_desc": "1 mmHg = 133.322 Pa；换算只在显示层。",
                       "en": "Pressure units", "en_desc": "1 mmHg = 133.322 Pa; conversion is display-only."},
    "speed": {"zh": "速度大小", "zh_desc": "三 seed 集成速度向量的模长，只在内部点上预测；壁面速度为零不展示。",
              "en": "Speed", "en_desc": "Magnitude of the ensemble velocity vector, predicted on interior points only; the zero wall velocity is not displayed."},
    "trust_interpolation_uncovered": {"zh": "可信位：插值无支撑", "zh_desc": "Gaussian 插值 1.5 mm 内没有预测点的顶点，显示为灰色，不外推。",
                                      "en": "Trust bit: uncovered by interpolation", "en_desc": "Vertices with no prediction point within 1.5 mm of the Gaussian kernel; shown grey, never extrapolated."},
    "trust_rough_surface": {"zh": "可信位：表面粗糙", "zh_desc": "局部 PCA 表面变化率 > 0.02 的区域，分割噪声可能影响预测。",
                            "en": "Trust bit: rough surface", "en_desc": "Regions whose local PCA surface variation exceeds 0.02; segmentation noise may affect the prediction."},
    "trust_geometry_out_of_range": {"zh": "可信位：几何越界", "zh_desc": "分支半径或长度超出训练队列范围（train136 atlas 的 0.9×min–1.1×max）。",
                                    "en": "Trust bit: geometry out of range", "en_desc": "Branch radius or length outside the training cohort range (0.9×min–1.1×max of the train136 atlas)."},
    "trust_low_sample_support": {"zh": "可信位：采样支撑弱", "zh_desc": "内部点 8 近邻半径超过全局中位数的两倍，该处体场采样稀疏。",
                                 "en": "Trust bit: low sample support", "en_desc": "Interior points whose 8-neighbour radius exceeds twice the global median; the volume is sparsely sampled there."},
    "trust_near_opening": {"zh": "可信位：近切口", "zh_desc": "距最近切口中心小于两倍切口半径；入口 / 出口边界条件影响较大。",
                           "en": "Trust bit: near opening", "en_desc": "Closer than two opening radii to the nearest inlet/outlet cut; boundary conditions dominate there."},
    "confidence_proxy": {"zh": "置信度代理", "zh_desc": "出口自动命名的综合分数（左右、内外四项加权），不是预测误差；≥ 0.95 且门控校准通过才自动放行。",
                         "en": "Confidence proxy", "en_desc": "Combined score of the automatic outlet naming (left/right and internal/external weights); not a prediction error. Automatic routing needs ≥ 0.95 and a validated gate."},
    "population_percentile": {"zh": "人群分位", "zh_desc": "本例 p99 在 136 例训练队列折外预测 p99 分布中的位置；参照来自模型折外预测，不是 CFD。",
                              "en": "Population percentile", "en_desc": "Position of this case's p99 within the out-of-fold p99 distribution of the 136-case training cohort; the reference comes from model predictions, not CFD."},
    "quality_grade": {"zh": "多模型一致性", "zh_desc": "按集成中各模型预测之间的离散度给出三档：多模型一致 / 存在不确定性 / 不稳定；只提示可信程度，不改数值。一致只说明几个模型的预测彼此接近，不代表与 CFD 相符：一致不代表准确。",
                      "en": "Model agreement", "en_desc": "Three levels (models agree / some uncertainty / unstable) from the spread between the ensemble's models; a confidence hint, not a correction. Agreement only means the models give similar predictions, not that they match CFD: agreement does not mean accuracy."},
    "release": {"zh": "发布包", "zh_desc": "冻结的权重与推理合同目录（如 X5D_v51_5seed_20260916）；任务固定绑定，重跑不覆盖历史结果。",
                "en": "Release package", "en_desc": "Frozen weights plus inference contract (e.g. X5D_v51_5seed_20260916); a job is bound to one release and reruns never overwrite history."},
    "feature_contract": {"zh": "特征合同哈希", "zh_desc": "冻结特征库 wss_features 的源码哈希；不一致的发布包会被拒绝加载。",
                         "en": "Feature contract hash", "en_desc": "Source hash of the frozen feature library wss_features; a release declaring a different hash is refused."},
    "run_identity": {"zh": "运行身份 run_identity", "zh_desc": "由输入 SHA256、发布包指纹、出口映射与推理参数决定的稳定哈希，用于同病例比较与复现。",
                     "en": "Run identity", "en_desc": "Stable hash of the input SHA256, release fingerprint, outlet mapping and inference parameters; used for same-case comparison and reproduction."},
    "standard_views": {"zh": "标准视角来源", "zh_desc": "前 / 后 / 左 / 右 / 上 / 下按解剖坐标架推断（x = 患者左，z = 指向入口）；STL 无患者方向时请核对左右。",
                       "en": "Standard views", "en_desc": "Front/back/left/right/top/bottom follow the inferred anatomical frame (x = patient left, z = towards the inlet); verify left/right when the STL carries no patient orientation."},
    "gaussian_interpolation": {"zh": "壁面着色插值", "zh_desc": "三维报告的顶点颜色与鼠标读值是 σ = 0.5 mm 的 Gaussian 插值；统计仍在预测点云上计算。",
                               "en": "Wall colour interpolation", "en_desc": "Vertex colours and hover values are a σ = 0.5 mm Gaussian interpolation; statistics stay on the prediction cloud."},
    "local_diameter": {"zh": "局部管径", "zh_desc": "中心线内切半径的两倍，不是该点截面的管腔最大直径。",
                       "en": "Local diameter", "en_desc": "Twice the centreline inscribed radius, not the largest lumen diameter of the cross-section at that point."},
    "arc_distance": {"zh": "沿中心线弧长", "zh_desc": "两点投影到中心线后的弧长距离；跨分支时沿分支树经公共祖先求和。",
                     "en": "Arc distance", "en_desc": "Arc length between the two points projected onto the centreline; across branches it is summed along the tree via the common ancestor."},
    "review_status": {"zh": "审阅状态", "zh_desc": "未审阅 / 已审阅（锁定：不能重算、改出口或删除）/ 已重新打开。",
                      "en": "Review status", "en_desc": "Unreviewed / reviewed (locked: no recomputation, outlet change or deletion) / reopened."},
    "finding_decision": {"zh": "发现判定", "zh_desc": "审阅人对每条自动发现标记确认或驳回并备注；驳回项不进一页纸正文，只列附录。",
                         "en": "Finding decision", "en_desc": "The reviewer confirms or rejects each automatic finding with a note; rejected items leave the one-page body and go to the appendix."},
    "streamlines": {"zh": "流线", "zh_desc": "固定收缩期帧速度场的稳态流线，由局部 IDW 插值双向积分；不是随时间演化的粒子轨迹。",
                    "en": "Streamlines", "en_desc": "Steady streamlines of the fixed systolic-frame velocity, integrated both ways through local IDW interpolation; not time-evolving particle paths."},
    "slice": {"zh": "截面", "zh_desc": "有限厚度的预测点云切片，二维图用局部 IDW 着色，覆盖不足处留灰。",
              "en": "Slice", "en_desc": "A finite-thickness slab of prediction points; the 2-D map uses local IDW colouring and stays grey where coverage is insufficient."},
    "max_diameter": {"zh": "管腔最大直径", "zh_desc": "中心线每 1 mm 取一站，用局部切线为法向切壁面网格得到管腔轮廓，取轮廓上最远两点的距离（最大 Feret 直径）；不是中心线内切直径。输入是管腔面，不含附壁血栓与管壁，通常小于 CT 报告的瘤体直径。",
                     "en": "Maximum lumen diameter", "en_desc": "At stations 1 mm apart the wall mesh is cut perpendicular to the centreline; the value is the largest distance between two points of that contour (maximum Feret diameter), not the inscribed diameter. The input is the lumen surface, excluding mural thrombus and the wall, so the value is usually smaller than the aneurysm diameter in a CT report."},
    "equivalent_diameter": {"zh": "等效直径", "zh_desc": "与截面面积相同的圆的直径，2·sqrt(面积/π)；比最大 Feret 直径对斜切和轮廓形状更稳健。",
                            "en": "Equivalent diameter", "en_desc": "Diameter of the circle with the same cross-sectional area, 2·sqrt(area/π); more robust to oblique cuts and contour shape than the maximum Feret diameter."},
    "reference_diameter": {"zh": "参考直径", "zh_desc": "主动脉全部闭合截面等效直径的第 10 百分位，作为该病例「正常管径」的稳健估计；瘤体与瘤颈判据都以它为基准。",
                           "en": "Reference diameter", "en_desc": "10th percentile of the equivalent diameter over all closed aortic stations, a robust estimate of this case's normal calibre; both the sac and the neck criteria are relative to it."},
    "aneurysm_sac": {"zh": "瘤体（瘤囊）", "zh_desc": "等效直径 ≥ 1.5 × 参考直径的连续区段（AAA 常用定义）；体积按截面面积沿弧长梯形积分，斜切站会偏大。按管腔判定：附壁血栓不计入直径和体积，血栓占据大部分瘤腔时可能判不出瘤体。",
                     "en": "Aneurysm sac", "en_desc": "The contiguous run of stations whose equivalent diameter is at least 1.5 × the reference diameter (the usual AAA definition); its volume is the trapezoidal integral of section area along the arc length and is overestimated where a section is oblique. Judged on the lumen: mural thrombus is not counted in the diameter or the volume, and a sac largely filled by thrombus may not be detected."},
    "aneurysm_neck": {"zh": "瘤颈", "zh_desc": "瘤体近端、等效直径 < 1.2 × 参考直径的连续区段；瘤体与瘤颈之间常有几毫米过渡肩部，不计入两者。按管腔判定。",
                      "en": "Aneurysm neck", "en_desc": "The contiguous run proximal to the sac whose equivalent diameter stays below 1.2 × the reference diameter; the few-millimetre shoulder between neck and sac belongs to neither. Judged on the lumen."},
    "lumen_volume": {"zh": "全腔体积", "zh_desc": "壁面网格把每个开口用扇形封盖后按散度定理求得的封闭体积；随输入 STL 的切口位置变化，不是解剖学血管容积。",
                     "en": "Lumen volume", "en_desc": "Volume enclosed by the wall mesh after each opening is fan-capped at its centroid (divergence theorem); it depends on where the input STL was cut and is not an anatomical vessel volume."},
    "tortuosity": {"zh": "扭曲度", "zh_desc": "分支中心线弧长除以两端直线距离；1.0 为笔直，越大越迂曲。",
                   "en": "Tortuosity", "en_desc": "Centreline arc length of a branch divided by the straight distance between its ends; 1.0 is straight and larger values are more tortuous."},
    "narrative": {"zh": "自动结论（参考）", "zh_desc": "由本页已有数字自动组织的中文描述，只陈述存在的量，不含诊断判断；审阅人可改写，改写后标「审阅人已修改」。",
                  "en": "Automatic conclusion (reference)", "en_desc": "A sentence-level restatement of the numbers already on this page; it states only quantities that exist and contains no diagnosis. A reviewer may replace it, in which case the text is marked as edited."},
    "fixed_frame": {"zh": "固定收缩期帧", "zh_desc": "峰值 WSS、压力与速度都对应 step 1162（约 0.21 s）的单一时相，不是全周期结果；三头发布包的 TAWSS / OSI 另见「单周期积分」。",
                    "en": "Fixed systolic frame", "en_desc": "Peak WSS, pressure and velocity refer to the single phase at step 1162 (≈ 0.21 s), not a full-cycle result; for the TAWSS / OSI of the three-head release see “single-cycle integral”."},
    "tawss": {"zh": "TAWSS 周期平均壁面切应力", "zh_desc": "一个心动周期内壁面切应力模长的时间平均（Pa）；低 TAWSS 阈值 0.4 Pa，与峰值 WSS 同口径。由三头模型直接预测，不是由逐帧结果积分得到。",
              "en": "TAWSS (time-averaged WSS)", "en_desc": "Time average of the wall shear stress magnitude over one cardiac cycle (Pa); low TAWSS means < 0.4 Pa, the same threshold as peak WSS. Predicted directly by the three-head model, not integrated from per-frame predictions."},
    "osi": {"zh": "OSI 振荡剪切指数", "zh_desc": "0.5 ×（1 − |周期平均切应力向量| / 周期平均切应力模长），无量纲；0 表示方向始终不变，0.5 表示完全往复。阈值 0.1 / 0.2 / 0.3。",
            "en": "OSI (oscillatory shear index)", "en_desc": "0.5 × (1 − |time-averaged shear vector| / time-averaged shear magnitude), dimensionless; 0 = unidirectional, 0.5 = fully oscillatory. Thresholds 0.1 / 0.2 / 0.3."},
    "rrt": {"zh": "RRT 相对滞留时间", "zh_desc": "RRT = 1 / [(1 − 2·OSI)·TAWSS]，单位 Pa⁻¹；切应力低且方向往复处数值大，表示血液在壁面附近停留更久。由预测的 TAWSS 与 OSI 逐点计算（TAWSS 下限 0.01 Pa、1 − 2·OSI 下限 0.01），不是模型单独输出；没有公认的绝对阈值，默认 5 / 10 / 20 Pa⁻¹ 只作分级显示，可在报告中修改。误差继承 TAWSS / OSI，在 TAWSS 很低处被放大，尚未单独对照 CFD 验证。",
            "en": "RRT (relative residence time)", "en_desc": "RRT = 1 / [(1 − 2·OSI)·TAWSS] in 1/Pa; large where shear is low and oscillating. Computed point by point from the predicted TAWSS and OSI (floors TAWSS ≥ 0.01 Pa, 1 − 2·OSI ≥ 0.01), not a separate model output. No agreed absolute threshold: the default 5 / 10 / 20 1/Pa is a display grading and is editable. Its error inherits TAWSS / OSI, amplified where TAWSS is small; not separately validated against CFD."},
    "ecap": {"zh": "ECAP 内皮细胞激活势", "zh_desc": "ECAP = OSI / TAWSS，单位 Pa⁻¹；低切应力与往复剪切叠加处升高，腹主动脉瘤文献常以 ECAP > 1.4 Pa⁻¹ 作为易形成附壁血栓的参考水平。由预测的 TAWSS 与 OSI 逐点计算（TAWSS 下限 0.01 Pa），默认阈值 1.4 / 2.8 / 4.2 Pa⁻¹，可在报告中修改；误差继承 TAWSS / OSI，尚未单独对照 CFD 验证。",
             "en": "ECAP (endothelial cell activation potential)", "en_desc": "ECAP = OSI / TAWSS in 1/Pa; high where low and oscillating shear coincide. AAA studies commonly use ECAP > 1.4 1/Pa as a thrombus-prone reference level. Computed point by point from the predicted TAWSS and OSI (floor TAWSS ≥ 0.01 Pa); default thresholds 1.4 / 2.8 / 4.2 1/Pa, editable in the report. Error inherits TAWSS / OSI; not separately validated against CFD."},
    "stagnation": {"zh": "滞留区", "zh_desc": "TAWSS < 0.4 Pa 且 OSI > 0.1 的壁面区域（切应力低且方向往复）；面积按点占比 × 输入壁面面积估计，发现列表按连通簇列出面积最大的 3 处。",
                   "en": "Stagnation region", "en_desc": "Wall where TAWSS < 0.4 Pa and OSI > 0.1 (low and oscillating shear); area = point fraction × input wall area, and the findings list the three largest connected clusters."},
    "high_osi": {"zh": "高 OSI 区", "zh_desc": "OSI > 0.3 的连通壁面簇；数值为簇内最大 OSI，位置取该点。",
                 "en": "High-OSI region", "en_desc": "Connected wall cluster with OSI > 0.3; the value is the cluster's largest OSI, located at that point."},
    "cycle_period": {"zh": "单周期积分", "zh_desc": "TAWSS / OSI 的训练标签由 81 帧 CFD 壁面切应力向量在一个 0.8 s 周期内（帧 0–79，每帧权重 1/80）积分得到；部署时模型一次前向直接给出，不做逐帧推演。",
                     "en": "Single-cycle integral", "en_desc": "The TAWSS / OSI training labels integrate the 81-frame CFD wall shear vector over one 0.8 s cycle (frames 0–79, weight 1/80 each); at deployment the model returns them in one forward pass without per-frame prediction."},
    "growth_rate": {"zh": "年增长率", "zh_desc": "同一患者两次扫描之间的变化量除以相隔年数；只在两次都填写了扫描日期且相隔 ≥ 30 天时计算，分别给出首末两次与最近两次。",
                    "en": "Annual growth rate", "en_desc": "Change between two scans of the same patient divided by the years between them; computed only when both scans carry a scan date at least 30 days apart, for the first-to-last and the latest pair."},
    "follow_up_timeline": {"zh": "随访时间线", "zh_desc": "同一匿名患者编号下按扫描（输入几何）合并的多次结果；几何量跨发布包可比，模型量只在同一发布包内比较。",
                           "en": "Follow-up timeline", "en_desc": "Results of one anonymised patient grouped per scan (input geometry); geometry is comparable across releases, model quantities only within one release."},
    "institution_template": {"zh": "机构报告模板", "zh_desc": "一页纸页眉的机构 / 科室 / 标题、页脚附加声明、签字栏以及术语表和附录开关；按任务根目录统一配置，只影响呈现，不改任何数值。",
                             "en": "Institution report template", "en_desc": "Institution / department / title in the one-page header, an extra footer note, signature lines and the glossary / appendix switches; configured once per jobs root and affecting presentation only."},
}


def glossary_document() -> dict:
    return {"schema_version": SCHEMA_VERSION, "terms": GLOSSARY}


def glossary_json() -> str:
    return json.dumps(glossary_document(), ensure_ascii=False, indent=1, sort_keys=True) + "\n"


def write_static(path: Path | None = None) -> Path:
    path = Path(path) if path else STATIC_DIR / "glossary.json"
    path.write_text(glossary_json(), encoding="utf-8")
    return path


if __name__ == "__main__":
    print(write_static())

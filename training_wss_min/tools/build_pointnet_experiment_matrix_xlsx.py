#!/usr/bin/env python3
"""生成 PointNet baseline 与后续变量实验的一页式精装 Excel 汇总。"""

from __future__ import annotations

from pathlib import Path

from openpyxl import Workbook, load_workbook
from openpyxl.comments import Comment
from openpyxl.formatting.rule import CellIsRule
from openpyxl.styles import Alignment, Border, Font, PatternFill, Side
from openpyxl.utils import get_column_letter
from openpyxl.worksheet.dimensions import ColumnDimension


ROOT = Path(__file__).resolve().parents[2]
OUTPUT = ROOT / "docs/03-汇报材料/WSS_PointNet实验矩阵与结果汇总.xlsx"

COLORS = {
    "navy": "17365D",
    "blue": "2F75B5",
    "green": "548235",
    "purple": "7030A0",
    "orange": "C65911",
    "teal": "0F6B78",
    "gold": "BF9000",
    "white": "FFFFFF",
    "ink": "263238",
    "muted": "667085",
    "grid": "D0D5DD",
    "baseline": "FFF2CC",
    "protocol": "DDEBF7",
    "probe": "FCE4D6",
    "global": "E2F0D9",
    "case": "E4DFEC",
    "na": "F2F4F7",
    "positive": "E2F0D9",
    "negative": "FCE4D6",
}

HEADERS = [
    "阶段", "实验 ID", "Job / Eval", "评价分区", "Split", "模型", "输入", "容量结构",
    "采样策略", "每例点数", "每 epoch 重采", "目标归一化", "Epoch", "Val / 早停", "选模",
    "唯一变量 / 区别", "主对照", "Δ主R²",
    "R² cb", "R² raw", "case mean", "case med", "case P10", "负例", "RMSE (Pa)", "MAE (Pa)",
    "high-WSS R²", "top10 幅值比", "top10 IoU",
    "R² cb", "R² raw", "case mean", "case med", "case P10", "负例", "MAE", "RMSE",
    "Spearman", "high-WSS Spearman", "top10 幅值比", "top10 IoU", "峰值距/bbox", "质心距/bbox",
    "p99 比", "动态范围比", "best epoch", "判读", "证据路径",
]


def row(**kwargs):
    data = {h: None for h in HEADERS}
    data.update(kwargs)
    return data


ROWS = [
    row(**{
        "阶段": "① 最早 2×3 baseline", "实验 ID": "B0-MLP-XYZ", "Job / Eval": "8700_0",
        "评价分区": "val8", "Split": "53/8/16", "模型": "MLP", "输入": "xyz",
        "容量结构": "MLP 64→64→64", "采样策略": "固定 FPS", "每例点数": 2000,
        "每 epoch 重采": "否", "目标归一化": "train-only 全局 log-z", "Epoch": 160,
        "Val / 早停": "有；patience=6", "选模": "val R² cb", "唯一变量 / 区别": "2×3：架构×输入起点",
        "主对照": "—", "R² cb": 0.1980, "R² raw": 0.2043, "case mean": 0.0080,
        "case med": 0.1114, "case P10": -0.3414, "负例": "1/8", "RMSE (Pa)": 4.424,
        "MAE (Pa)": 2.615, "high-WSS R²": -1.3788, "top10 幅值比": 0.361,
        "top10 IoU": 0.189, "判读": "MLP·xyz 起点；热点明显低估",
        "证据路径": "training_wss_min/runs/baseline_2x3_simple/outputs/mlp_xyz",
    }),
    row(**{
        "阶段": "① 最早 2×3 baseline", "实验 ID": "B0-MLP-XYZGEOM", "Job / Eval": "8700_1",
        "评价分区": "val8", "Split": "53/8/16", "模型": "MLP", "输入": "xyz+geom",
        "容量结构": "MLP 64→64→64", "采样策略": "固定 FPS", "每例点数": 2000,
        "每 epoch 重采": "否", "目标归一化": "train-only 全局 log-z", "Epoch": 160,
        "Val / 早停": "有；patience=6", "选模": "val R² cb", "唯一变量 / 区别": "仅增加 geom",
        "主对照": "B0-MLP-XYZ", "Δ主R²": 0.0462, "R² cb": 0.2442, "R² raw": 0.2632,
        "case mean": 0.0669, "case med": 0.0496, "case P10": -0.2142, "负例": "3/8",
        "RMSE (Pa)": 4.257, "MAE (Pa)": 2.476, "high-WSS R²": -1.0972,
        "top10 幅值比": 0.447, "top10 IoU": 0.182, "判读": "总体 R² 提升；负例数反而增加",
        "证据路径": "training_wss_min/runs/baseline_2x3_simple/outputs/mlp_xyzgeom",
    }),
    row(**{
        "阶段": "① 最早 2×3 baseline", "实验 ID": "B0-PN-XYZ", "Job / Eval": "8700_2",
        "评价分区": "val8", "Split": "53/8/16", "模型": "PointNet", "输入": "xyz",
        "容量结构": "width=32；head=64", "采样策略": "固定 FPS", "每例点数": 2000,
        "每 epoch 重采": "否", "目标归一化": "train-only 全局 log-z", "Epoch": 160,
        "Val / 早停": "有；patience=6", "选模": "val R² cb", "唯一变量 / 区别": "2×3：PointNet·xyz",
        "主对照": "—", "R² cb": 0.1727, "R² raw": 0.1704, "case mean": 0.1194,
        "case med": 0.0699, "case P10": -0.0554, "负例": "2/8", "RMSE (Pa)": 4.517,
        "MAE (Pa)": 2.454, "high-WSS R²": -1.6736, "top10 幅值比": 0.313,
        "top10 IoU": 0.150, "判读": "纯 xyz 下 PointNet 不占优",
        "证据路径": "training_wss_min/runs/baseline_2x3_simple/outputs/pointnet_xyz",
    }),
    row(**{
        "阶段": "① 最早 2×3 baseline", "实验 ID": "B0-PN-XYZGEOM", "Job / Eval": "8700_3",
        "评价分区": "val8", "Split": "53/8/16", "模型": "PointNet", "输入": "xyz+geom",
        "容量结构": "width=32；head=64", "采样策略": "固定 FPS", "每例点数": 2000,
        "每 epoch 重采": "否", "目标归一化": "train-only 全局 log-z", "Epoch": 160,
        "Val / 早停": "有；patience=6", "选模": "val R² cb", "唯一变量 / 区别": "仅增加 geom",
        "主对照": "B0-PN-XYZ", "Δ主R²": 0.1288, "R² cb": 0.3015, "R² raw": 0.3122,
        "case mean": 0.1842, "case med": 0.1404, "case P10": 0.0143, "负例": "1/8",
        "RMSE (Pa)": 4.113, "MAE (Pa)": 2.291, "high-WSS R²": -1.1690,
        "top10 幅值比": 0.457, "top10 IoU": 0.235, "判读": "2×3 最佳格；后续主线锚点",
        "证据路径": "training_wss_min/runs/baseline_2x3_simple/outputs/pointnet_xyzgeom",
    }),
    row(**{
        "阶段": "① 最早 2×3 baseline", "实验 ID": "B0-PNPP-XYZ", "Job / Eval": "8700_4",
        "评价分区": "val8", "Split": "53/8/16", "模型": "PointNet++", "输入": "xyz",
        "容量结构": "经典 SA+FP", "采样策略": "固定 FPS", "每例点数": 2000,
        "每 epoch 重采": "否", "目标归一化": "train-only 全局 log-z", "Epoch": 160,
        "Val / 早停": "有；patience=6", "选模": "val R² cb", "唯一变量 / 区别": "2×3：PointNet++·xyz",
        "主对照": "—", "R² cb": 0.2169, "R² raw": 0.2367, "case mean": 0.0156,
        "case med": 0.0694, "case P10": -0.3146, "负例": "2/8", "RMSE (Pa)": 4.333,
        "MAE (Pa)": 2.583, "high-WSS R²": -1.0329, "top10 幅值比": 0.461,
        "top10 IoU": 0.207, "判读": "纯 xyz 中主 R² 最高；病例稳定性差",
        "证据路径": "training_wss_min/runs/baseline_2x3_simple/outputs/pointnetpp_xyz",
    }),
    row(**{
        "阶段": "① 最早 2×3 baseline", "实验 ID": "B0-PNPP-XYZGEOM", "Job / Eval": "8700_5",
        "评价分区": "val8", "Split": "53/8/16", "模型": "PointNet++", "输入": "xyz+geom",
        "容量结构": "经典 SA+FP", "采样策略": "固定 FPS", "每例点数": 2000,
        "每 epoch 重采": "否", "目标归一化": "train-only 全局 log-z", "Epoch": 160,
        "Val / 早停": "有；patience=6", "选模": "val R² cb", "唯一变量 / 区别": "仅增加 geom",
        "主对照": "B0-PNPP-XYZ", "Δ主R²": 0.0418, "R² cb": 0.2587, "R² raw": 0.2668,
        "case mean": 0.1287, "case med": 0.1427, "case P10": -0.0950, "负例": "1/8",
        "RMSE (Pa)": 4.247, "MAE (Pa)": 2.281, "high-WSS R²": -1.4213,
        "top10 幅值比": 0.387, "top10 IoU": 0.226, "判读": "MAE 最低；主 R² 仍低于 PointNet+geom",
        "证据路径": "training_wss_min/runs/baseline_2x3_simple/outputs/pointnetpp_xyzgeom",
    }),
    row(**{
        "阶段": "② 协议调整", "实验 ID": "P0-PN-XYZ", "Job / Eval": "8968_0 / 8974",
        "评价分区": "test16", "Split": "61/0/16", "模型": "PointNet", "输入": "xyz",
        "容量结构": "width=32；head=64", "采样策略": "固定 FPS", "每例点数": 2000,
        "每 epoch 重采": "否", "目标归一化": "train61 全局 log-z", "Epoch": 400,
        "Val / 早停": "无 / 无", "选模": "train loss", "唯一变量 / 区别": "val8 并入 train；400ep；测试锁定",
        "主对照": "协议变化，不能与 val8 直接比", "R² cb": 0.0825, "R² raw": 0.0832,
        "case mean": 0.071, "case med": 0.033, "case P10": -0.143, "负例": "5/16",
        "RMSE (Pa)": 5.60, "MAE (Pa)": 3.10, "high-WSS R²": -1.88,
        "top10 幅值比": 0.284, "top10 IoU": 0.095, "判读": "test 泛化弱；纯 xyz 淘汰",
        "证据路径": "training_wss_min/runs/pointnet_trainloss_e400/outputs/pointnet_xyz",
    }),
    row(**{
        "阶段": "② 协议调整", "实验 ID": "E0-GLOBAL", "Job / Eval": "8968_1 / 8975",
        "评价分区": "test16", "Split": "61/0/16", "模型": "PointNet", "输入": "xyz+geom",
        "容量结构": "width=32；head=64", "采样策略": "固定 FPS", "每例点数": 2000,
        "每 epoch 重采": "否", "目标归一化": "train61 全局 log-z", "Epoch": 400,
        "Val / 早停": "无 / 无", "选模": "train loss", "唯一变量 / 区别": "同协议仅增加 geom；正式共同对照",
        "主对照": "P0-PN-XYZ", "Δ主R²": 0.0589, "R² cb": 0.1414, "R² raw": 0.1433,
        "case mean": 0.106, "case med": 0.074, "case P10": -0.062, "负例": "5/16",
        "RMSE (Pa)": 5.42, "MAE (Pa)": 2.95, "high-WSS R²": -1.76,
        "top10 幅值比": 0.328, "top10 IoU": 0.174,
        "R² cb__norm": 0.396355, "case med__norm": 0.31854, "case P10__norm": 0.11756,
        "负例__norm": "0/16", "Spearman": 0.635214, "top10 幅值比__norm": 0.305713,
        "top10 IoU__norm": 0.1279995, "峰值距/bbox": 0.174543, "质心距/bbox": 0.090052,
        "p99 比": 0.65899, "动态范围比": 0.56906, "best epoch": 383,
        "判读": "几何优于 xyz；仍过拟合且热点 No-Go",
        "证据路径": "training_wss_min/runs/pointnet_trainloss_e400/outputs/pointnet_xyzgeom",
    }),
    row(**{
        "阶段": "③ 容量探针", "实验 ID": "WIDE128-PROBE", "Job / Eval": "8970",
        "评价分区": "val8", "Split": "53/8/16", "模型": "PointNet", "输入": "xyz+geom",
        "容量结构": "6→128→256→512；1024→512→256→1", "采样策略": "固定 FPS", "每例点数": 2000,
        "每 epoch 重采": "否", "目标归一化": "train-only 全局 log-z", "Epoch": 160,
        "Val / 早停": "有；patience=6", "选模": "val R² cb", "唯一变量 / 区别": "只加宽；但多一层 128，不是导师精确结构",
        "主对照": "B0-PN-XYZGEOM", "Δ主R²": 0.0056, "R² cb": 0.3071, "R² raw": 0.3167,
        "case mean": 0.173, "case med": 0.169, "case P10": -0.007, "负例": "1/8",
        "RMSE (Pa)": 4.10, "MAE (Pa)": 2.30, "high-WSS R²": -1.17,
        "top10 幅值比": 0.406, "top10 IoU": 0.224, "判读": "旧协议增益仅 +0.006；补充 No-Go",
        "证据路径": "training_wss_min/runs/pointnet_wide128/outputs/pointnet_xyzgeom",
    }),
    row(**{
        "阶段": "④ 正式 GLOBAL 矩阵", "实验 ID": "E2-GLOBAL", "Job / Eval": "8976",
        "评价分区": "test16", "Split": "61/0/16", "模型": "PointNet", "输入": "xyz+geom",
        "容量结构": "导师精确：6→256→512；1024→512→256→1", "采样策略": "固定 FPS", "每例点数": 2000,
        "每 epoch 重采": "否", "目标归一化": "train61 全局 log-z", "Epoch": 400,
        "Val / 早停": "无 / 无", "选模": "train loss", "唯一变量 / 区别": "相对 E0 只改容量",
        "主对照": "E0-GLOBAL", "Δ主R²": 0.0725, "R² cb": 0.213966, "R² raw": 0.217416,
        "case mean": 0.171651, "case med": 0.201046, "case P10": -0.007453, "负例": "2/16",
        "RMSE (Pa)": 5.114140, "MAE (Pa)": 2.800541, "high-WSS R²": -1.486146,
        "top10 幅值比": 0.378074, "top10 IoU": 0.145746,
        "R² cb__norm": 0.460600, "R² raw__norm": 0.464047, "case mean__norm": 0.409957,
        "case med__norm": 0.431405, "case P10__norm": 0.201957, "负例__norm": "1/16",
        "MAE__norm": 0.573218, "RMSE__norm": 0.735370, "Spearman": 0.661274,
        "high-WSS Spearman": -0.006642, "top10 幅值比__norm": 0.393876,
        "top10 IoU__norm": 0.145746, "峰值距/bbox": 0.171848, "质心距/bbox": 0.084555,
        "p99 比": 0.683619, "动态范围比": 0.576923, "best epoch": 364,
        "判读": "本轮最强物理 R²；容量有效，但热点仍 No-Go",
        "证据路径": "training_wss_min/runs/pointnet_distribution_matrix/outputs/e2_global",
    }),
    row(**{
        "阶段": "④ 正式 GLOBAL 矩阵", "实验 ID": "E3-GLOBAL", "Job / Eval": "8977",
        "评价分区": "test16", "Split": "61/0/16", "模型": "PointNet", "输入": "xyz+geom",
        "容量结构": "baseline width=32；head=64", "采样策略": "random；不放回", "每例点数": 5000,
        "每 epoch 重采": "是", "目标归一化": "train61 全局 log-z", "Epoch": 400,
        "Val / 早停": "无 / 无", "选模": "train loss", "唯一变量 / 区别": "相对 E0 只改点数与重采样",
        "主对照": "E0-GLOBAL", "Δ主R²": 0.0223, "R² cb": 0.163705, "R² raw": 0.164095,
        "case mean": 0.138061, "case med": 0.177428, "case P10": -0.093088, "负例": "4/16",
        "RMSE (Pa)": 5.275115, "MAE (Pa)": 2.876399, "high-WSS R²": -1.664569,
        "top10 幅值比": 0.342550, "top10 IoU": 0.123991,
        "R² cb__norm": 0.421763, "R² raw__norm": 0.421581, "case mean__norm": 0.384811,
        "case med__norm": 0.372048, "case P10__norm": 0.233925, "负例__norm": "0/16",
        "MAE__norm": 0.592841, "RMSE__norm": 0.761383, "Spearman": 0.645480,
        "high-WSS Spearman": 0.020101, "top10 幅值比__norm": 0.338513,
        "top10 IoU__norm": 0.123991, "峰值距/bbox": 0.178374, "质心距/bbox": 0.072971,
        "p99 比": 0.641665, "动态范围比": 0.581238, "best epoch": 354,
        "判读": "单独增加点数增益弱；热点未改善",
        "证据路径": "training_wss_min/runs/pointnet_distribution_matrix/outputs/e3_global",
    }),
    row(**{
        "阶段": "④ 正式 GLOBAL 矩阵", "实验 ID": "E23-GLOBAL", "Job / Eval": "8978",
        "评价分区": "test16", "Split": "61/0/16", "模型": "PointNet", "输入": "xyz+geom",
        "容量结构": "导师精确：6→256→512；1024→512→256→1", "采样策略": "random；不放回", "每例点数": 5000,
        "每 epoch 重采": "是", "目标归一化": "train61 全局 log-z", "Epoch": 400,
        "Val / 早停": "无 / 无", "选模": "train loss", "唯一变量 / 区别": "容量+5000 点交互；比 E3 可隔离容量",
        "主对照": "E2-GLOBAL", "Δ主R²": -0.0152, "R² cb": 0.198763, "R² raw": 0.199869,
        "case mean": 0.171776, "case med": 0.138580, "case P10": -0.020351, "负例": "2/16",
        "RMSE (Pa)": 5.163361, "MAE (Pa)": 2.805594, "high-WSS R²": -1.524527,
        "top10 幅值比": 0.372459, "top10 IoU": 0.147831,
        "R² cb__norm": 0.459778, "R² raw__norm": 0.458426, "case mean__norm": 0.417554,
        "case med__norm": 0.393815, "case P10__norm": 0.282006, "负例__norm": "0/16",
        "MAE__norm": 0.574011, "RMSE__norm": 0.735930, "Spearman": 0.673999,
        "high-WSS Spearman": -0.010559, "top10 幅值比__norm": 0.392909,
        "top10 IoU__norm": 0.147831, "峰值距/bbox": 0.158465, "质心距/bbox": 0.089579,
        "p99 比": 0.692811, "动态范围比": 0.604686, "best epoch": 354,
        "判读": "未超过 E2；相对 E3 有容量增益，无明确协同",
        "证据路径": "training_wss_min/runs/pointnet_distribution_matrix/outputs/e23_global",
    }),
    row(**{
        "阶段": "⑤ 逐病例归一化", "实验 ID": "E2-CASE", "Job / Eval": "8979",
        "评价分区": "test16", "Split": "61/0/16", "模型": "PointNet", "输入": "xyz+geom",
        "容量结构": "导师精确：6→256→512；1024→512→256→1", "采样策略": "固定 FPS", "每例点数": 2000,
        "每 epoch 重采": "否", "目标归一化": "逐病例 WSS/WSSmax", "Epoch": 400,
        "Val / 早停": "无 / 无", "选模": "train loss", "唯一变量 / 区别": "相对 E2-GLOBAL 只改目标归一化",
        "主对照": "E2-GLOBAL（normalized）", "Δ主R²": -0.2882,
        "R² cb__norm": 0.172354, "R² raw__norm": 0.167098, "case mean__norm": 0.070722,
        "case med__norm": 0.122336, "case P10__norm": -0.327905, "负例__norm": "4/16",
        "MAE__norm": 0.057018, "RMSE__norm": 0.092842, "Spearman": 0.573918,
        "high-WSS Spearman": 0.061577, "top10 幅值比__norm": 0.371904,
        "top10 IoU__norm": 0.139577, "峰值距/bbox": 0.172488, "质心距/bbox": 0.093650,
        "p99 比": 0.485733, "动态范围比": 0.456266, "best epoch": 374,
        "判读": "CASE No-Go：分布、排序、动态范围全面退化",
        "证据路径": "training_wss_min/runs/pointnet_distribution_matrix/outputs/e2_case",
    }),
    row(**{
        "阶段": "⑤ 逐病例归一化", "实验 ID": "E3-CASE", "Job / Eval": "8980",
        "评价分区": "test16", "Split": "61/0/16", "模型": "PointNet", "输入": "xyz+geom",
        "容量结构": "baseline width=32；head=64", "采样策略": "random；不放回", "每例点数": 5000,
        "每 epoch 重采": "是", "目标归一化": "逐病例 WSS/WSSmax", "Epoch": 400,
        "Val / 早停": "无 / 无", "选模": "train loss", "唯一变量 / 区别": "相对 E3-GLOBAL 只改目标归一化",
        "主对照": "E3-GLOBAL（normalized）", "Δ主R²": -0.2666,
        "R² cb__norm": 0.155149, "R² raw__norm": 0.157479, "case mean__norm": 0.029819,
        "case med__norm": 0.076732, "case P10__norm": -0.405929, "负例__norm": "5/16",
        "MAE__norm": 0.058646, "RMSE__norm": 0.093802, "Spearman": 0.558968,
        "high-WSS Spearman": 0.046589, "top10 幅值比__norm": 0.398187,
        "top10 IoU__norm": 0.159401, "峰值距/bbox": 0.161766, "质心距/bbox": 0.077526,
        "p99 比": 0.547106, "动态范围比": 0.454380, "best epoch": 364,
        "判读": "局部 IoU 提升，但整体分布明显退化；CASE No-Go",
        "证据路径": "training_wss_min/runs/pointnet_distribution_matrix/outputs/e3_case",
    }),
    row(**{
        "阶段": "⑥ 导师追加深度探针", "实验 ID": "E4-DEEP-GLOBAL", "Job / Eval": "8999",
        "评价分区": "test16", "Split": "61/0/16", "模型": "PointNet", "输入": "xyz+geom",
        "容量结构": "6→64→128→256→512；1024→512→256→128→64→1",
        "采样策略": "固定 FPS", "每例点数": 2000, "每 epoch 重采": "否",
        "目标归一化": "train61 全局 log-z", "Epoch": 400, "Val / 早停": "无 / 无",
        "选模": "train loss", "唯一变量 / 区别": "相对 E2 仅增加编码/解码深度",
        "主对照": "E2-GLOBAL", "Δ主R²": -0.051116,
        "R² cb": 0.162851, "R² raw": 0.163594, "case mean": 0.138983,
        "case med": 0.096228, "case P10": -0.028810, "负例": "4/16",
        "RMSE (Pa)": 5.277807, "MAE (Pa)": 2.851001, "high-WSS R²": -1.679826,
        "top10 幅值比": 0.332838, "top10 IoU": 0.153077,
        "R² cb__norm": 0.447638, "R² raw__norm": 0.448934,
        "case mean__norm": 0.399451, "case med__norm": 0.384871,
        "case P10__norm": 0.227187, "负例__norm": "0/16",
        "MAE__norm": 0.579897, "RMSE__norm": 0.744153, "Spearman": 0.657024,
        "high-WSS Spearman": -0.066911, "top10 幅值比__norm": 0.336297,
        "top10 IoU__norm": 0.153077, "峰值距/bbox": 0.214879,
        "质心距/bbox": 0.077346, "p99 比": 0.642825, "动态范围比": 0.585638,
        "best epoch": 379,
        "判读": "train-fit 增强，test R²/误差/high-WSS 退化；深度探针 No-Go",
        "证据路径": "training_wss_min/runs/pointnet_deeper/outputs/e4_deep_global",
    }),
]


def normalized_value(item: dict, header: str, col_index: int):
    """处理物理/normalized 两组重名表头。"""
    if col_index >= 30:
        aliases = {
            "R² cb": "R² cb__norm", "R² raw": "R² raw__norm", "case mean": "case mean__norm",
            "case med": "case med__norm", "case P10": "case P10__norm", "负例": "负例__norm",
            "MAE": "MAE__norm", "RMSE": "RMSE__norm", "top10 幅值比": "top10 幅值比__norm",
            "top10 IoU": "top10 IoU__norm",
        }
        return item.get(aliases.get(header, header))
    return item.get(header)


def build_workbook() -> None:
    wb = Workbook()
    ws = wb.active
    ws.title = "实验矩阵总览"
    ws.sheet_view.showGridLines = False
    ws.freeze_panes = "F8"
    ws.sheet_properties.pageSetUpPr.fitToPage = True
    ws.page_setup.orientation = "landscape"
    ws.page_setup.paperSize = ws.PAPERSIZE_A3
    ws.page_setup.fitToWidth = 1
    ws.page_setup.fitToHeight = 0
    ws.print_title_rows = "1:7"
    ws.sheet_view.zoomScale = 75

    last_col = len(HEADERS)
    last_letter = get_column_letter(last_col)

    ws.merge_cells(start_row=1, start_column=1, end_row=1, end_column=last_col)
    title = ws.cell(1, 1, "WSS PointNet 实验矩阵与结果总览")
    title.font = Font(name="Noto Sans CJK SC", size=20, bold=True, color=COLORS["white"])
    title.fill = PatternFill("solid", fgColor=COLORS["navy"])
    title.alignment = Alignment(horizontal="left", vertical="center")
    ws.row_dimensions[1].height = 38

    ws.merge_cells(start_row=2, start_column=1, end_row=2, end_column=last_col)
    subtitle = ws.cell(2, 1, "从最早 2×3 baseline 到容量、点数与 WSS/WSSmax 归一化：单 seed 探索性证据链｜更新 2026-07-15")
    subtitle.font = Font(name="Noto Sans CJK SC", size=10, color=COLORS["muted"])
    subtitle.alignment = Alignment(horizontal="left", vertical="center")
    ws.row_dimensions[2].height = 24

    badges = [
        (1, 8, "15 条实验记录", COLORS["gold"]),
        (9, 20, "E2-GLOBAL 最强物理 R² cb = 0.2140", COLORS["green"]),
        (21, 34, "E4 deeper：train-fit↑，test R²↓", COLORS["teal"]),
        (35, last_col, "CASE / deeper：No-Go", COLORS["purple"]),
    ]
    for start, end, text, color in badges:
        ws.merge_cells(start_row=3, start_column=start, end_row=3, end_column=end)
        c = ws.cell(3, start, text)
        c.fill = PatternFill("solid", fgColor=color)
        c.font = Font(name="Noto Sans CJK SC", size=10, bold=True, color=COLORS["white"])
        c.alignment = Alignment(horizontal="center", vertical="center")
    ws.row_dimensions[3].height = 27

    ws.merge_cells(start_row=4, start_column=1, end_row=4, end_column=last_col)
    note = ws.cell(4, 1, "阅读口径：val8 与 test16 不可直接横比；GLOBAL 的物理 Pa 与 normalized log-z 同时报出，CASE 只在 WSS/WSSmax 空间评价、不能恢复 Pa；空白表示该轮未计算或不适用。")
    note.font = Font(name="Noto Sans CJK SC", size=9, italic=True, color=COLORS["ink"])
    note.fill = PatternFill("solid", fgColor="EAF2F8")
    note.alignment = Alignment(horizontal="left", vertical="center", wrap_text=True)
    ws.row_dimensions[4].height = 30

    ws.merge_cells(start_row=5, start_column=1, end_row=5, end_column=last_col)
    legend = ws.cell(5, 1, "阶段色带：  黄色=最早 2×3   蓝色=协议调整   橙色=容量探针   绿色=正式 GLOBAL   紫色=逐病例 CASE   青色=deeper 追加探针    ｜    Δ主R² 始终相对“主对照”列所列对象。")
    legend.font = Font(name="Noto Sans CJK SC", size=9, color=COLORS["muted"])
    legend.alignment = Alignment(horizontal="left", vertical="center")
    ws.row_dimensions[5].height = 22

    groups = [
        (1, 5, "实验身份", COLORS["navy"]),
        (6, 15, "模型与训练协议", COLORS["blue"]),
        (16, 18, "变量与配对", COLORS["gold"]),
        (19, 29, "物理 WSS 指标（Pa 空间）", COLORS["green"]),
        (30, 45, "归一化空间分布与 high-risk 指标", COLORS["purple"]),
        (46, 48, "选模与结论", COLORS["orange"]),
    ]
    for start, end, text, color in groups:
        ws.merge_cells(start_row=6, start_column=start, end_row=6, end_column=end)
        c = ws.cell(6, start, text)
        c.fill = PatternFill("solid", fgColor=color)
        c.font = Font(name="Noto Sans CJK SC", size=10, bold=True, color=COLORS["white"])
        c.alignment = Alignment(horizontal="center", vertical="center")
    ws.row_dimensions[6].height = 25

    header_colors = {}
    for start, end, _, color in groups:
        for idx in range(start, end + 1):
            header_colors[idx] = color
    for idx, header in enumerate(HEADERS, 1):
        c = ws.cell(7, idx, header)
        c.fill = PatternFill("solid", fgColor=header_colors[idx])
        c.font = Font(name="Noto Sans CJK SC", size=9, bold=True, color=COLORS["white"])
        c.alignment = Alignment(horizontal="center", vertical="center", wrap_text=True)
    ws.row_dimensions[7].height = 43

    thin = Side(style="thin", color=COLORS["grid"])
    stage_fills = {
        "①": COLORS["baseline"], "②": COLORS["protocol"], "③": COLORS["probe"],
        "④": COLORS["global"], "⑤": COLORS["case"], "⑥": "DDEBF7",
    }
    start_data = 8
    for ridx, item in enumerate(ROWS, start_data):
        stage_color = next(color for prefix, color in stage_fills.items() if item["阶段"].startswith(prefix))
        for cidx, header in enumerate(HEADERS, 1):
            value = normalized_value(item, header, cidx)
            c = ws.cell(ridx, cidx, value)
            c.font = Font(name="Noto Sans CJK SC", size=8.5, color=COLORS["ink"])
            c.alignment = Alignment(horizontal="center", vertical="center", wrap_text=True)
            c.border = Border(bottom=thin)
            if cidx <= 5:
                c.fill = PatternFill("solid", fgColor=stage_color)
                c.font = Font(name="Noto Sans CJK SC", size=8.5, bold=cidx in (1, 2), color=COLORS["ink"])
            elif value is None:
                c.fill = PatternFill("solid", fgColor=COLORS["na"])
            elif ridx % 2 == 0:
                c.fill = PatternFill("solid", fgColor="FAFBFC")
            if cidx in (8, 12, 16, 17, 47, 48):
                c.alignment = Alignment(horizontal="left", vertical="center", wrap_text=True)
            if isinstance(value, float):
                c.number_format = "0.0000;[Red]-0.0000;–"
        ws.row_dimensions[ridx].height = 55

        evidence = ws.cell(ridx, 48)
        if evidence.value:
            target = (ROOT / str(evidence.value)).resolve()
            evidence.value = "打开产物"
            evidence.hyperlink = target.as_uri()
            evidence.style = "Hyperlink"
            evidence.font = Font(name="Noto Sans CJK SC", size=8.5, color="0563C1", underline="single")
            evidence.comment = Comment(str(target), "Codex")

    end_data = start_data + len(ROWS) - 1
    ws.auto_filter.ref = f"A7:{last_letter}{end_data}"

    # 视觉强调：Δ、关键 R² 和结论中的 No-Go。
    ws.conditional_formatting.add(
        f"R{start_data}:R{end_data}", CellIsRule(operator="greaterThan", formula=["0"], fill=PatternFill("solid", fgColor=COLORS["positive"]))
    )
    ws.conditional_formatting.add(
        f"R{start_data}:R{end_data}", CellIsRule(operator="lessThan", formula=["0"], fill=PatternFill("solid", fgColor=COLORS["negative"]))
    )
    for coord in ("S11", "S17", "AD17"):
        ws[coord].fill = PatternFill("solid", fgColor="C6E0B4")
        ws[coord].font = Font(name="Noto Sans CJK SC", size=8.5, bold=True, color="1F4E2C")
    for coord in ("AD20", "AD21"):
        ws[coord].fill = PatternFill("solid", fgColor="F4CCCC")
        ws[coord].font = Font(name="Noto Sans CJK SC", size=8.5, bold=True, color="9C0006")

    widths = {
        "A": 21, "B": 19, "C": 17, "D": 10, "E": 11, "F": 13, "G": 13, "H": 28,
        "I": 16, "J": 10, "K": 12, "L": 23, "M": 9, "N": 15, "O": 13,
        "P": 30, "Q": 24, "R": 11, "AT": 11, "AU": 35, "AV": 14,
    }
    for idx in range(19, 46):
        widths[get_column_letter(idx)] = 12
    for col, width in widths.items():
        ws.column_dimensions[col].width = width

    # 次指标默认不隐藏，但建立列分组，便于在 Excel 中折叠精简视图。
    for idx in (20, 21, 31, 32, 39):
        letter = get_column_letter(idx)
        dim = ws.column_dimensions.get(letter, ColumnDimension(ws, index=letter))
        dim.outlineLevel = 1
        ws.column_dimensions[letter] = dim
    ws.sheet_properties.outlinePr.summaryRight = True

    foot = end_data + 2
    notes = [
        "注 1｜本表以 ckpt_best 为主；正式矩阵 ckpt_last 仅作敏感性审计，不依据 test16 反选模型。",
        "注 2｜2×3 与 wide128 probe 报 val8；E0/E2/E3/E23/CASE/E4 报 test16。test16 已用于连续探索性比较，不能表述为未经复用的最终无偏泛化结果。",
        "注 3｜GLOBAL 可逆回物理 Pa；CASE 的真实 WSSmax 只用于构造评价真值，不进入网络，也不在部署时借用来恢复 Pa。",
        "结论｜导师精确宽网 E2 仍是 PointNet 锚点；random-5000 单独增益弱且无明确协同；逐病例 WSS/WSSmax 与 E4 deeper 均 No-Go。E4 训练拟合增强，但 test R²、误差与 high-WSS 泛化均退化。",
    ]
    for offset, text in enumerate(notes):
        rr = foot + offset
        ws.merge_cells(start_row=rr, start_column=1, end_row=rr, end_column=last_col)
        c = ws.cell(rr, 1, text)
        c.font = Font(name="Noto Sans CJK SC", size=9, bold=offset == 3, color=COLORS["ink"])
        c.fill = PatternFill("solid", fgColor="EAF2F8" if offset < 3 else "D9EAD3")
        c.alignment = Alignment(horizontal="left", vertical="center", wrap_text=True)
        ws.row_dimensions[rr].height = 27 if offset < 3 else 34

    ws.print_area = f"A1:{last_letter}{foot + len(notes) - 1}"
    ws.auto_filter.ref = f"A7:{last_letter}{end_data}"
    ws.sheet_properties.tabColor = COLORS["navy"]
    ws.oddHeader.center.text = "&B WSS PointNet 实验矩阵"
    ws.oddFooter.center.text = "第 &P / &N 页"
    ws.oddFooter.right.text = "2026-07-15"

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    wb.save(OUTPUT)

    # 保存后重新加载，防止生成不可读的 OOXML 包。
    check = load_workbook(OUTPUT, read_only=False, data_only=False)
    assert check.sheetnames == ["实验矩阵总览"]
    sheet = check["实验矩阵总览"]
    assert sheet.max_column == len(HEADERS)
    assert sheet["B8"].value == "B0-MLP-XYZ"
    assert sheet[f"B{end_data}"].value == "E4-DEEP-GLOBAL"
    assert sheet.auto_filter.ref == f"A7:{last_letter}{end_data}"
    check.close()
    print(OUTPUT)


if __name__ == "__main__":
    build_workbook()

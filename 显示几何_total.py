#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
单脚本集成：
1) 读取 STL 血管表面，自动抽取中心线并计算弧长/曲率/扭率
2) 将径向距离、弧长、归一化弧长、曲率、扭率映射到表面点并导出 CSV
3) 可选：为任意 N×3 点云计算同样的几何特征并保存为 .npz
"""

# 导入所需的库
import csv       # 用于CSV文件操作
import vtk       # Visualization Toolkit，用于三维计算机图形学
import numpy as np  # 用于数值计算
from vmtk import vmtkscripts  # VMTK库，专门用于血管建模和分析


def read_surface(stl_path):
    """
    读取STL格式的血管表面模型文件

    参数:
        stl_path (str): STL文件的路径

    返回:
        vtkPolyData: 表示血管表面的VTK多边形数据对象
    """
    # 创建VMTK表面读取器实例
    reader = vmtkscripts.vmtkSurfaceReader()
    # 设置输入文件路径
    reader.InputFileName = stl_path
    # 执行读取操作
    reader.Execute()
    # 返回读取到的表面数据
    return reader.Surface


def extract_centerlines(surface):
    """
    从血管表面提取中心线并计算相关属性

    参数:
        surface (vtkPolyData): 血管表面数据

    返回:
        vtkPolyData: 包含弧长、曲率、扭率等属性的中心线数据

    处理步骤:
        1) 自动检测入口/出口端点
        2) 计算中心线
        3) 为中心线附加弧长、曲率、扭率
    """
    # 创建端点检测器实例
    endpoints = vmtkscripts.vmtkEndPoints()
    # 设置要处理的表面数据
    endpoints.Surface = surface
    # 执行端点检测
    endpoints.Execute()
    # 获取检测到的端点数据
    endpoints_pd = endpoints.EndPoints

    # 检查是否检测到至少两个端点（入口和出口）
    if endpoints_pd.GetNumberOfPoints() < 2:
        raise RuntimeError("检测到的端点少于 2 个，无法抽取中心线")

    # 获取第一个端点作为源点（通常为入口）
    source_point = endpoints_pd.GetPoint(0)
    # 获取第二个端点作为目标点（通常为出口）
    target_point = endpoints_pd.GetPoint(1)

    # 创建中心线计算实例
    centerline_filter = vmtkscripts.vmtkCenterlines()
    # 设置表面数据
    centerline_filter.Surface = surface
    # 设置种子选择方法为点列表方式
    centerline_filter.SeedSelectorName = "pointlist"
    # 设置源点坐标
    centerline_filter.SourcePoints = [source_point]
    # 设置目标点坐标
    centerline_filter.TargetPoints = [target_point]
    # 设置追加端点标志
    centerline_filter.AppendEndPoints = 1
    # 执行中心线计算
    centerline_filter.Execute()
    # 获取计算得到的中心线数据
    centerlines = centerline_filter.Centerlines

    # 创建中心线属性计算实例
    attrs = vmtkscripts.vmtkCenterlineAttributes()
    # 设置中心线数据
    attrs.Centerlines = centerlines
    # 设置弧长数组名称
    attrs.AbscissasArrayName = "Abscissas"
    # 设置曲率数组名称
    attrs.CurvatureArrayName = "Curvature"
    # 设置扭率数组名称
    attrs.TorsionArrayName = "Torsion"
    # 执行属性计算
    attrs.Execute()
    # 返回包含所有属性的中心线数据
    return attrs.Centerlines


def map_features_to_surface(surface, centerlines):
    """
    将中心线的几何特征映射回血管表面的每个点

    参数:
        surface (vtkPolyData): 血管表面数据
        centerlines (vtkPolyData): 已计算好属性的中心线数据

    返回:
        list: 包含每个表面点及其几何特征的字典列表
    """
    # 创建距离计算实例，用于计算表面点到中心线的距离及相关属性
    dist = vmtkscripts.vmtkDistanceToCenterlines()
    # 设置表面数据
    dist.Surface = surface
    # 设置中心线数据
    dist.Centerlines = centerlines
    # 设置距离数组名称
    dist.DistanceToCenterlinesArrayName = "DistanceToCenterlines"
    # 设置弧长数组名称
    dist.AbscissasArrayName = "Abscissas"
    # 执行距离及属性计算
    dist.Execute()
    # 获取包含新属性的表面数据
    surface_with_attr = dist.Surface

    # 获取中心线点数据中的各种属性数组
    cl_point_data = centerlines.GetPointData()
    # 获取曲率数组
    curvature_arr = cl_point_data.GetArray("Curvature")
    # 获取扭率数组
    torsion_arr = cl_point_data.GetArray("Torsion")
    # 获取弧长数组
    abscissas_cl = cl_point_data.GetArray("Abscissas")

    # 检查必要的属性数组是否存在
    if curvature_arr is None or torsion_arr is None or abscissas_cl is None:
        raise RuntimeError("中心线上缺少 Curvature/Torsion/Abscissas 数组，请检查 vmtk 版本")

    # 计算最大弧长值，用于后续归一化
    max_s = max(abscissas_cl.GetTuple1(i) for i in range(abscissas_cl.GetNumberOfTuples()))

    # 获取表面点数据中的各种属性数组
    pd = surface_with_attr.GetPointData()
    # 获取中心线点ID数组（用于查找最近的中心线点）
    centerline_ids_arr = pd.GetArray("CenterlinePointIds")
    # 获取弧长数组
    abscissas_arr = pd.GetArray("Abscissas")
    # 获取到中心线的距离数组
    dist_arr = pd.GetArray("DistanceToCenterlines")

    # 检查必要的映射数组是否存在
    if centerline_ids_arr is None or abscissas_arr is None or dist_arr is None:
        raise RuntimeError("未找到映射数组 CenterlinePointIds/Abscissas/DistanceToCenterlines")

    # 初始化存储所有表面点特征的列表
    points = []
    # 遍历表面上的每一个点
    for i in range(surface_with_attr.GetNumberOfPoints()):
        # 获取当前点的三维坐标
        x, y, z = surface_with_attr.GetPoint(i)
        # 获取该点到中心线的径向距离
        radial_dist = dist_arr.GetTuple1(i)
        # 获取该点对应的弧长值
        s = abscissas_arr.GetTuple1(i)
        # 计算归一化弧长（0到1之间）
        s_norm = s / max_s if max_s > 0 else 0.0

        # 获取最近的中心线点的ID
        cl_pt_id = int(centerline_ids_arr.GetTuple1(i))
        # 根据ID获取对应的曲率值
        curvature = curvature_arr.GetTuple1(cl_pt_id)
        # 根据ID获取对应的扭率值
        torsion = torsion_arr.GetTuple1(cl_pt_id)

        # 将当前点的所有特征信息添加到列表中
        points.append(
            {
                "x": x,              # X坐标
                "y": y,              # Y坐标
                "z": z,              # Z坐标
                "radial_dist": radial_dist,   # 径向距离
                "s": s,              # 弧长
                "s_norm": s_norm,    # 归一化弧长
                "curvature": curvature,       # 曲率
                "torsion": torsion,           # 扭率
            }
        )

    # 返回包含所有点特征的列表
    return points


def save_features_to_csv(points, out_csv):
    """
    将表面点的几何特征保存为CSV文件

    参数:
        points (list): 包含点特征的字典列表
        out_csv (str): 输出CSV文件的路径
    """
    # 定义CSV文件的列标题
    fieldnames = ["x", "y", "z", "radial_dist", "s", "s_norm", "curvature", "torsion"]
    # 打开输出文件准备写入
    with open(out_csv, "w", newline="") as f:
        # 创建CSV字典写入器
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        # 写入表头
        writer.writeheader()
        # 遍历所有点，逐行写入特征数据
        for p in points:
            writer.writerow(p)


def numpy_points_to_polydata(points_np):
    """
    将N×3的numpy点云数组转换为vtkPolyData格式（仅有点，无面）

    参数:
        points_np (numpy.ndarray): N×3形状的点云数组

    返回:
        vtkPolyData: 只包含点信息的VTK多边形数据对象
    """
    # 创建VTK点集合
    vtk_points = vtk.vtkPoints()
    # 遍历numpy数组中的每个点，将其添加到VTK点集合中
    for p in points_np:
        vtk_points.InsertNextPoint(float(p[0]), float(p[1]), float(p[2]))

    # 创建VTK多边形数据对象
    polydata = vtk.vtkPolyData()
    # 设置点集合
    polydata.SetPoints(vtk_points)

    # 为了让polydata在VTK中被识别为"点云"，需要添加顶点单元
    vertices = vtk.vtkCellArray()
    # 为每个点创建一个顶点单元
    for i in range(points_np.shape[0]):
        vertices.InsertNextCell(1)
        vertices.InsertCellPoint(i)
    # 设置顶点单元数组
    polydata.SetVerts(vertices)
    # 返回构建好的polydata对象
    return polydata


def add_geometric_features_to_points(points_np, centerlines):
    """
    为任意N×3点云计算相同的几何特征并返回

    输入：
      - points_np: N×3 numpy 数组（内部点 + 壁面点都可以）
      - centerlines: vmtk 计算好的中心线（已有 Abscissas/Curvature/Torsion）
    输出：
      - 一个 dict，每个键是特征名，对应一个长度 N 的 numpy 数组
    """
    # 将numpy点云转换为VTK PolyData格式
    points_pd = numpy_points_to_polydata(points_np)

    # 创建距离计算实例
    dist = vmtkscripts.vmtkDistanceToCenterlines()
    # 设置点云数据（这里Surface可以是一团点云）
    dist.Surface = points_pd
    # 设置中心线数据
    dist.Centerlines = centerlines
    # 设置距离数组名称
    dist.DistanceToCenterlinesArrayName = "DistanceToCenterlines"
    # 设置弧长数组名称
    dist.AbscissasArrayName = "Abscissas"
    # 执行距离及属性计算
    dist.Execute()
    # 获取包含新属性的点云数据
    pts_with_attr = dist.Surface

    # 获取中心线点数据中的各种属性数组
    cl_point_data = centerlines.GetPointData()
    # 获取曲率数组
    curvature_arr = cl_point_data.GetArray("Curvature")
    # 获取扭率数组
    torsion_arr = cl_point_data.GetArray("Torsion")
    # 获取弧长数组
    abscissas_cl = cl_point_data.GetArray("Abscissas")

    # 检查必要的属性数组是否存在
    if curvature_arr is None or torsion_arr is None or abscissas_cl is None:
        raise RuntimeError("中心线上缺少 Curvature/Torsion/Abscissas 数组，请检查 vmtk 版本")

    # 计算最大弧长值，用于后续归一化
    max_s = max(abscissas_cl.GetTuple1(i) for i in range(abscissas_cl.GetNumberOfTuples()))

    # 获取点云数据中的各种属性数组
    pd = pts_with_attr.GetPointData()
    # 获取距离数组
    dist_arr = pd.GetArray("DistanceToCenterlines")
    # 获取弧长数组
    abscissas_arr = pd.GetArray("Abscissas")
    # 获取中心线点ID数组
    cl_ids_arr = pd.GetArray("CenterlinePointIds")

    # 检查必要的映射数组是否存在
    if dist_arr is None or abscissas_arr is None or cl_ids_arr is None:
        raise RuntimeError("未找到映射数组 CenterlinePointIds/Abscissas/DistanceToCenterlines")

    # 获取点的数量
    N = pts_with_attr.GetNumberOfPoints()
    # 初始化各个特征数组
    radial_dist = np.zeros(N)   # 径向距离数组
    s = np.zeros(N)             # 弧长数组
    s_norm = np.zeros(N)        # 归一化弧长数组
    curvature = np.zeros(N)     # 曲率数组
    torsion = np.zeros(N)       # 扭率数组

    # 遍历每个点，计算其各项特征值
    for i in range(N):
        # 获取径向距离
        radial_dist[i] = dist_arr.GetTuple1(i)
        # 获取弧长
        s[i] = abscissas_arr.GetTuple1(i)
        # 计算归一化弧长
        s_norm[i] = s[i] / max_s if max_s > 0 else 0.0

        # 获取最近的中心线点ID
        cl_id = int(cl_ids_arr.GetTuple1(i))
        # 根据ID获取曲率值
        curvature[i] = curvature_arr.GetTuple1(cl_id)
        # 根据ID获取扭率值
        torsion[i] = torsion_arr.GetTuple1(cl_id)

    # 返回包含所有特征的字典
    return {
        "radial_dist": radial_dist,   # 径向距离数组
        "s": s,                       # 弧长数组
        "s_norm": s_norm,             # 归一化弧长数组
        "curvature": curvature,       # 曲率数组
        "torsion": torsion,           # 扭率数组
    }


def main(stl_path, out_csv, points_npy=None, out_points_npy=None):
    """
    主函数：执行完整的几何特征提取流程

    参数:
        stl_path (str): 输入的血管表面STL文件路径
        out_csv (str): 表面点特征CSV输出路径
        points_npy (str, optional): 可选的N×3 numpy点云文件路径
        out_points_npy (str, optional): 点云特征输出为.npz文件的路径
    """
    # 步骤1: 读取STL表面文件
    surface = read_surface(stl_path)
    # 步骤2: 提取中心线并计算属性
    centerlines = extract_centerlines(surface)

    # 步骤3: 将几何特征映射到表面点
    surface_points = map_features_to_surface(surface, centerlines)
    # 步骤4: 保存表面点特征到CSV文件
    save_features_to_csv(surface_points, out_csv)
    # 打印完成信息
    print(f"表面特征已保存到: {out_csv}")

    # 如果提供了额外的点云文件，则为其计算特征
    if points_npy:
        # 加载numpy点云文件
        pts_np = np.load(points_npy)
        # 为这些点计算几何特征
        feats = add_geometric_features_to_points(pts_np, centerlines)
        # 确定输出文件路径（如果没有指定则使用默认路径）
        out_path = out_points_npy or "points_geometric_features.npz"
        # 将特征保存为NPZ格式文件
        np.savez(out_path, **feats)
        # 打印完成信息
        print(f"点云特征已保存到: {out_path}")


# 当脚本作为主程序运行时执行以下代码
if __name__ == "__main__":
    # 导入命令行参数解析模块
    import argparse

    # 创建参数解析器
    parser = argparse.ArgumentParser(
        description="单脚本：从 STL 抽取中心线，输出表面/点云的几何特征"
    )
    # 添加必需的STL文件路径参数
    parser.add_argument("--stl", required=True, help="输入的血管表面 STL 路径")
    # 添加表面点特征CSV输出路径参数（有默认值）
    parser.add_argument(
        "--out_csv",
        default="vessel_geometric_features.csv",
        help="表面点特征 CSV 输出路径",
    )
    # 添加可选的numpy点云文件参数
    parser.add_argument(
        "--points_npy",
        help="可选：N×3 numpy 点云文件（.npy），将为这些点计算特征",
    )
    # 添加可选的点云特征输出路径参数
    parser.add_argument(
        "--out_points_npy",
        help="可选：点云特征保存为 .npz 的路径（未指定则默认 points_geometric_features.npz）",
    )
    # 解析命令行参数
    args = parser.parse_args()

    # 调用主函数执行处理流程
    main(args.stl, args.out_csv, args.points_npy, args.out_points_npy)
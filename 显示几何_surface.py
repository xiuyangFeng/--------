#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
用 VMTK 从血管 STL 生成中心线，并为每个表面点计算：
- radial_dist: 点到中心线的最短距离（局部半径）
- s: 测地线弧长（沿中心线的坐标）
- s_norm: 归一化测地线坐标（0=入口, 1=出口）
- curvature: 中心线曲率
- torsion: 中心线扭率

输出一个 CSV: x,y,z,radial_dist,s,s_norm,curvature,torsion
"""

import vtk
import numpy as np
import csv
from vmtk import vmtkscripts


def read_surface(stl_path):
    reader = vmtkscripts.vmtkSurfaceReader()
    reader.InputFileName = stl_path
    reader.Execute()
    return reader.Surface


def extract_centerlines(surface):
    """
    1) 自动检测端点（inlet/outlet）
    2) 计算中心线
    3) 计算中心线属性（弧长、曲率、扭率）
    """
    # 1. 端点提取
    endpoints = vmtkscripts.vmtkEndPoints()
    endpoints.Surface = surface
    endpoints.Execute()
    endpoints_pd = endpoints.EndPoints

    # 提取 source / target 点（这里假设只有一条主中心线）
    n_endpoints = endpoints_pd.GetNumberOfPoints()
    if n_endpoints < 2:
        raise RuntimeError("检测到的端点少于 2 个，无法提取中心线！")

    # 默认第一个点为 source，第二个点为 target
    source_id = 0
    target_id = 1

    # 2. 计算中心线
    centerline_filter = vmtkscripts.vmtkCenterlines()
    centerline_filter.Surface = surface
    centerline_filter.SeedSelectorName = 'pointlist'
    centerline_filter.SourcePoints = [endpoints_pd.GetPoint(source_id)]
    centerline_filter.TargetPoints = [endpoints_pd.GetPoint(target_id)]
    centerline_filter.AppendEndPoints = 1
    centerline_filter.Execute()
    centerlines = centerline_filter.Centerlines

    # 3. 计算中心线属性（弧长 / 曲率 / 扭率）
    attrs = vmtkscripts.vmtkCenterlineAttributes()
    attrs.Centerlines = centerlines
    # 这些名字是 vmtk 的默认数组名，也可以自定义
    attrs.AbscissasArrayName = 'Abscissas'
    attrs.CurvatureArrayName = 'Curvature'
    attrs.TorsionArrayName = 'Torsion'
    attrs.Execute()
    centerlines_attr = attrs.Centerlines

    return centerlines_attr


def map_features_to_surface(surface, centerlines):
    """
    用 vmtkDistanceToCenterlines 把：
    - 点到中心线距离（radial_dist）
    - 弧长 s
    - 对应中心线点的 curvature, torsion
    映射回表面点。
    """

    dist = vmtkscripts.vmtkDistanceToCenterlines()
    dist.Surface = surface
    dist.Centerlines = centerlines
    # 数组名保持默认即可
    dist.DistanceToCenterlinesArrayName = 'DistanceToCenterlines'
    dist.AbscissasArrayName = 'Abscissas'
    dist.Execute()
    surface_with_attr = dist.Surface

    # 从中心线 polydata 中取出曲率 / 扭率数组
    cl_point_data = centerlines.GetPointData()
    curvature_arr = cl_point_data.GetArray('Curvature')
    torsion_arr = cl_point_data.GetArray('Torsion')

    if curvature_arr is None or torsion_arr is None:
        raise RuntimeError("中心线上没有 Curvature / Torsion 数组，请检查 vmtkCenterlineAttributes 设置。")

    # 注意：vmtkDistanceToCenterlines 会在表面上附加一些映射信息数组
    # 一般会有 'CenterlinePointIds'，指每个表面点对应的最近中心线点索引
    surf_point_data = surface_with_attr.GetPointData()
    centerline_ids_arr = surf_point_data.GetArray('CenterlinePointIds')
    abscissas_arr = surf_point_data.GetArray('Abscissas')
    dist_arr = surf_point_data.GetArray('DistanceToCenterlines')

    if centerline_ids_arr is None:
        raise RuntimeError("找不到 'CenterlinePointIds' 数组，可能 vmtk 版本不同，请打印所有数组名检查。")

    n_points = surface_with_attr.GetNumberOfPoints()

    # 计算中心线最大弧长，用于归一化 s
    # 这里直接取中心线上 Abscissas 的最大值
    cl_abscissas = cl_point_data.GetArray('Abscissas')
    max_s = max(cl_abscissas.GetTuple1(i) for i in range(cl_abscissas.GetNumberOfTuples()))

    # 为每个表面点构造特征
    points = []
    for i in range(n_points):
        x, y, z = surface_with_attr.GetPoint(i)

        radial_dist = dist_arr.GetTuple1(i)  # 点到中心线的距离
        s = abscissas_arr.GetTuple1(i)       # 对应的弧长坐标
        s_norm = s / max_s if max_s > 0 else 0.0

        cl_pt_id = int(centerline_ids_arr.GetTuple1(i))
        curvature = curvature_arr.GetTuple1(cl_pt_id)
        torsion = torsion_arr.GetTuple1(cl_pt_id)

        points.append({
            "x": x,
            "y": y,
            "z": z,
            "radial_dist": radial_dist,
            "s": s,
            "s_norm": s_norm,
            "curvature": curvature,
            "torsion": torsion
        })

    return points


def save_features_to_csv(points, out_csv):
    fieldnames = ["x", "y", "z", "radial_dist", "s", "s_norm", "curvature", "torsion"]
    with open(out_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for p in points:
            writer.writerow(p)


def main(stl_path, out_csv):
    # 1. 读 STL 血管表面
    surface = read_surface(stl_path)

    # 2. 提取中心线并计算属性
    centerlines = extract_centerlines(surface)

    # 3. 把径向距离 / 弧长 / 曲率 / 扭率映射回表面点
    points_with_features = map_features_to_surface(surface, centerlines)

    # 4. 保存为 csv
    save_features_to_csv(points_with_features, out_csv)
    print(f"几何特征已保存到: {out_csv}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="从血管 STL 提取中心线几何特征（径向距离、测地线坐标、曲率、扭率）")
    parser.add_argument("--stl", required=True, help="输入的血管表面 STL 文件路径")
    parser.add_argument("--out_csv", default="vessel_geometric_features.csv", help="输出 CSV 文件路径")
    args = parser.parse_args()

    main(args.stl, args.out_csv)

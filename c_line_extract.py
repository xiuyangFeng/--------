import os
import numpy as np
import pandas as pd
import vtk
from vmtk import vmtkscripts
from vtkmodules.util.numpy_support import vtk_to_numpy, numpy_to_vtk

# ==========================================
# 1. 基础文件读取与写入模块
# ==========================================

def read_surface(path):
    """读取 .vtp 或 .stl 表面文件"""
    if path.endswith(".vtp"):
        reader = vtk.vtkXMLPolyDataReader()
    else:
        reader = vtk.vtkSTLReader()
    reader.SetFileName(path)
    reader.Update()
    return reader.GetOutput()

def write_vtp(polydata, path):
    """保存为 .vtp 文件，用于 Paraview 可视化"""
    writer = vtk.vtkXMLPolyDataWriter()
    writer.SetFileName(path)
    writer.SetInputData(polydata)
    writer.Write()

# ==========================================
# 2. 几何处理核心模块
# ==========================================

def detect_all_openings(surface):
    """
    使用几何拓扑检测所有开口，并区分入口(Inlet)和出口(Outlets)。
    策略：Z轴最高的为Inlet，其余为Outlets。
    """
    print("   [1.1] 正在扫描网格边界 (Feature Edges)...")

    # 1. 提取边界边
    feature_edges = vtk.vtkFeatureEdges()
    feature_edges.SetInputData(surface)
    feature_edges.BoundaryEdgesOn()
    feature_edges.FeatureEdgesOff()
    feature_edges.ManifoldEdgesOff()
    feature_edges.NonManifoldEdgesOff()
    feature_edges.Update()

    edges = feature_edges.GetOutput()

    # 2. 连通域分析（区分不同的开口）
    conn = vtk.vtkConnectivityFilter()
    conn.SetInputData(edges)
    conn.SetExtractionModeToAllRegions()
    conn.ColorRegionsOn()
    conn.Update()

    num_regions = conn.GetNumberOfExtractedRegions()
    print(f"   ...初步检测到 {num_regions} 个潜在开口。")

    opening_centers = []
    valid_count = 0

    # 3. 遍历每个区域，计算中心点
    for i in range(num_regions):
        thresh = vtk.vtkThreshold()
        thresh.SetInputData(conn.GetOutput())
        thresh.SetInputArrayToProcess(0, 0, 0, vtk.vtkDataObject.FIELD_ASSOCIATION_CELLS, "RegionId")
        thresh.SetLowerThreshold(i)
        thresh.SetUpperThreshold(i)
        thresh.Update()

        loop = thresh.GetOutput()

        # [过滤] 如果边界点太少（噪点），则忽略
        if loop.GetNumberOfPoints() < 10:
            continue

        points = vtk_to_numpy(loop.GetPoints().GetData())
        center = np.mean(points, axis=0)

        opening_centers.append(center)
        valid_count += 1

    if valid_count < 2:
        print(f"[Warning] 有效开口不足 2 个 (仅 {valid_count})，无法生成中心线。")
        return None, None

    opening_centers = np.array(opening_centers)

    # 4. 区分入口出口 (假设 Zmax 为入口)
    z_coords = opening_centers[:, 2]
    inlet_idx = np.argmax(z_coords)

    inlet_pt = opening_centers[inlet_idx]
    outlet_pts = np.delete(opening_centers, inlet_idx, axis=0)

    print(f"   ...确认 1 个入口，{len(outlet_pts)} 个出口。")
    return list(inlet_pt), outlet_pts.tolist()


def clean_centerline(centerline, surface):
    """
    [重要优化] 清洗中心线：
    1. 剪裁掉跑出血管包围盒的部分。
    2. 只保留最大的连通部分（去除剪裁产生的悬浮碎片）。
    """
    print("   [2.1] 正在清洗中心线 (Clipping & Connectivity)...")

    bounds = surface.GetBounds()
    # 容差设为 0.5mm，只允许极小的溢出
    buffer = 0.5

    xmin, xmax, ymin, ymax, zmin, zmax = bounds

    # 1. 物理剪裁 (Box Clip)
    box = vtk.vtkBox()
    box.SetBounds(xmin - buffer, xmax + buffer,
                  ymin - buffer, ymax + buffer,
                  zmin - buffer, zmax + buffer)

    clipper = vtk.vtkClipPolyData()
    clipper.SetInputData(centerline)
    clipper.SetClipFunction(box)
    clipper.InsideOutOn() # 保留盒子内部
    clipper.Update()

    clipped_cl = clipper.GetOutput()

    if clipped_cl.GetNumberOfPoints() == 0:
        print("   [Warning] 剪裁后中心线为空！回退到原始数据。")
        return centerline

    # 2. 保留最大连通域 (去除断裂的短线)
    conn = vtk.vtkConnectivityFilter()
    conn.SetInputData(clipped_cl)
    conn.SetExtractionModeToLargestRegion() # 关键：只留主干
    conn.Update()

    final_cl = conn.GetOutput()

    print(f"   ...清洗完成。点数优化: {centerline.GetNumberOfPoints()} -> {final_cl.GetNumberOfPoints()}")
    return final_cl

# ==========================================
# 3. 主处理逻辑
# ==========================================

def extract_features_robust(surface):
    # --- 1. 检测端口 ---
    print("1. [Detection] 检测进出口...")
    inlet_pt, outlet_pts_list = detect_all_openings(surface)

    # Fallback: 如果检测失败，使用包围盒兜底
    if inlet_pt is None:
        print("   [Fallback] 使用包围盒极值作为端口。")
        bounds = surface.GetBounds()
        inlet_pt = [(bounds[0]+bounds[1])/2, (bounds[2]+bounds[3])/2, bounds[5]]
        outlet_pts_list = [[(bounds[0]+bounds[1])/2, (bounds[2]+bounds[3])/2, bounds[4]]]

    # VMTK 需要扁平化的 TargetPoints 列表
    target_points_flat = []
    for pt in outlet_pts_list:
        target_points_flat.extend(pt)

    # --- 2. 提取中心线 ---
    print("2. [VMTK] 生成中心线...")
    cl = vmtkscripts.vmtkCenterlines()
    cl.Surface = surface
    cl.SeedSelectorName = 'pointlist'
    cl.SourcePoints = list(inlet_pt)
    cl.TargetPoints = target_points_flat
    # [关键] 设为0，防止强制延伸到错误的外部点
    cl.AppendEndPoints = 0
    cl.Execute()

    raw_centerline = cl.Centerlines

    # --- 3. 清洗中心线 ---
    clean_cl = clean_centerline(raw_centerline, surface)

    # --- 4. 计算几何属性 ---
    print("4. [Geometry] 计算几何特征 (曲率/挠率/坐标)...")
    geom = vmtkscripts.vmtkCenterlineGeometry()
    geom.Centerlines = clean_cl
    geom.LineSmoothing = 1
    geom.SmoothingFactor = 0.1
    geom.NumberOfSmoothingIterations = 10
    geom.Execute()

    attr = vmtkscripts.vmtkCenterlineAttributes()
    attr.Centerlines = geom.Centerlines
    attr.Execute()

    centerline = attr.Centerlines

    # --- 5. 特征映射到表面 ---
    print("5. [Mapping] 将中心线特征映射到血管壁面...")

    cl_points = vtk_to_numpy(centerline.GetPoints().GetData())
    cl_pd = centerline.GetPointData()

    # 获取中心线数据
    arr_abscissa = vtk_to_numpy(cl_pd.GetArray("Abscissas"))
    arr_radius = vtk_to_numpy(cl_pd.GetArray("MaximumInscribedSphereRadius"))
    arr_curv = vtk_to_numpy(cl_pd.GetArray("Curvature"))
    arr_tangent = vtk_to_numpy(cl_pd.GetArray("FrenetTangent"))

    # 构建 KDTree 加速查找
    locator = vtk.vtkKdTreePointLocator()
    locator.SetDataSet(centerline)
    locator.BuildLocator()

    surf_points = vtk_to_numpy(surface.GetPoints().GetData())
    n_pts = surf_points.shape[0]

    # 初始化结果数组
    res_abscissa = np.zeros(n_pts)
    res_norm_dist = np.zeros(n_pts)
    res_curv = np.zeros(n_pts)
    res_tangent = np.zeros((n_pts, 3))

    # 遍历所有表面点
    for i in range(n_pts):
        pt = surf_points[i]
        closest_id = locator.FindClosestPoint(pt)

        dist = np.linalg.norm(pt - cl_points[closest_id])
        local_r = arr_radius[closest_id]

        # 映射特征
        res_abscissa[i] = arr_abscissa[closest_id]
        res_curv[i] = arr_curv[closest_id]
        res_tangent[i] = arr_tangent[closest_id]

        # 计算归一化半径 (dist / R)，防止除零
        safe_r = local_r if local_r > 1e-6 else 1.0
        res_norm_dist[i] = dist / safe_r

    # --- 6. 后处理与归一化 (数值安全) ---

    # 归一化 Abscissa 到 0-1
    ab_min = np.nanmin(res_abscissa)
    ab_max = np.nanmax(res_abscissa)
    denom = ab_max - ab_min

    if denom > 1e-8:
        res_abscissa = (res_abscissa - ab_min) / denom
    else:
        res_abscissa[:] = 0.0 # 避免除以零

    # 清除可能产生的 NaN
    res_abscissa = np.nan_to_num(res_abscissa)
    res_norm_dist = np.nan_to_num(res_norm_dist)
    res_curv = np.nan_to_num(res_curv)
    res_tangent = np.nan_to_num(res_tangent)

    # --- 7. 将特征写入 PolyData ---
    def add_array(name, data):
        arr = numpy_to_vtk(data, deep=1)
        arr.SetName(name)
        surface.GetPointData().AddArray(arr)

    add_array("Feature_Abscissa", res_abscissa)
    add_array("Feature_NormRadius", res_norm_dist)
    add_array("Feature_Curvature", res_curv)
    add_array("Feature_Tangent", res_tangent)

    return surface, centerline

# ==========================================
# 4. 程序入口
# ==========================================

if __name__ == "__main__":
    # --- 配置部分 ---
    input_stl = "MA+XIAO+DONG-new.stl"  # 输入文件

    # 输出文件名
    output_surface_vtp = "result_surface.vtp"
    output_centerline_vtp = "result_centerline.vtp"
    output_csv = "result_features.csv"

    # --- 执行部分 ---
    if not os.path.exists(input_stl):
        print(f"❌ 错误: 找不到文件 {input_stl}")
    else:
        try:
            print(f"🚀 开始处理: {input_stl}")

            # 读取表面
            surf = read_surface(input_stl)

            # 提取特征 (核心函数)
            final_surface, final_centerline = extract_features_robust(surf)

            # 保存可视化文件
            write_vtp(final_surface, output_surface_vtp)
            write_vtp(final_centerline, output_centerline_vtp)

            # 导出 CSV 数据 (供 PyG 使用)
            pd_surf = final_surface.GetPointData()

            # 获取所有点数据
            points = vtk_to_numpy(final_surface.GetPoints().GetData())
            tangents = vtk_to_numpy(pd_surf.GetArray("Feature_Tangent"))

            df = pd.DataFrame({
                "x": points[:, 0],
                "y": points[:, 1],
                "z": points[:, 2],
                "Abscissa": vtk_to_numpy(pd_surf.GetArray("Feature_Abscissa")),
                "NormRadius": vtk_to_numpy(pd_surf.GetArray("Feature_NormRadius")),
                "Curvature": vtk_to_numpy(pd_surf.GetArray("Feature_Curvature")),
                "Tangent_X": tangents[:, 0],
                "Tangent_Y": tangents[:, 1],
                "Tangent_Z": tangents[:, 2]
            })

            df.to_csv(output_csv, index=False)

            print("\n✅ 处理成功!")
            print(f"   - 表面模型: {output_surface_vtp}")
            print(f"   - 中心线:   {output_centerline_vtp}")
            print(f"   - 特征数据: {output_csv}")

        except Exception as e:
            print(f"\n❌ 发生未知错误: {e}")
            import traceback
            traceback.print_exc()
import os
import numpy as np
import pandas as pd
import vtk
from vtkmodules.util.numpy_support import vtk_to_numpy, numpy_to_vtk
import vmtk_core  # 导入我们写好的核心模块

def process_surface_dataset(stl_path, output_csv_path):
    print(f"🚀 [Scenario A] Processing Surface: {stl_path}")
    
    # 1. 读取表面
    surface = vmtk_core.read_surface(stl_path)
    
    # 2. 调用核心模块提取增强版中心线 (包含手动计算的所有特征)
    centerline = vmtk_core.extract_rich_centerline(surface)
    
    # 3. 准备映射 (Mapping)
    # 获取中心线上的数据数组
    cl_points = vtk_to_numpy(centerline.GetPoints().GetData())
    cl_pd = centerline.GetPointData()
    
    # 注意：这里的名字必须和 vmtk_core 里 add_array 的名字一致
    arr_abscissa = vtk_to_numpy(cl_pd.GetArray("Abscissas"))
    arr_radius   = vtk_to_numpy(cl_pd.GetArray("MaximumInscribedSphereRadius"))
    arr_curv     = vtk_to_numpy(cl_pd.GetArray("Curvature"))
    arr_tangent  = vtk_to_numpy(cl_pd.GetArray("FrenetTangent"))
    
    # 构建 KDTree 加速查找
    locator = vtk.vtkKdTreePointLocator()
    locator.SetDataSet(centerline)
    locator.BuildLocator()
    
    # 获取表面点
    surf_points = vtk_to_numpy(surface.GetPoints().GetData())
    n_pts = surf_points.shape[0]
    
    # 初始化结果容器
    res_abscissa = np.zeros(n_pts)
    res_norm_dist = np.zeros(n_pts)
    res_curv = np.zeros(n_pts)
    res_tangent = np.zeros((n_pts, 3))
    
    print(f"   ...Mapping features to {n_pts} surface nodes...")
    
    # 4. 遍历映射
    for i in range(n_pts):
        pt = surf_points[i]
        closest_id = locator.FindClosestPoint(pt)
        
        # 物理距离
        dist = np.linalg.norm(pt - cl_points[closest_id])
        local_r = arr_radius[closest_id]
        
        # 拷贝特征
        res_abscissa[i] = arr_abscissa[closest_id]
        res_curv[i] = arr_curv[closest_id]
        res_tangent[i] = arr_tangent[closest_id]
        
        # 计算归一化径向距离 (dist / R)
        # 对于壁面点，这个值理论上应该接近 1.0
        safe_r = local_r if local_r > 1e-6 else 1.0
        res_norm_dist[i] = dist / safe_r

    # 5. 后处理 (归一化 Abscissa 到 0-1)
    ab_min, ab_max = np.nanmin(res_abscissa), np.nanmax(res_abscissa)
    denom = ab_max - ab_min
    
    if denom > 1e-8:
        res_abscissa = (res_abscissa - ab_min) / denom
    else:
        res_abscissa[:] = 0.0
        
    # 清洗 NaN
    res_abscissa = np.nan_to_num(res_abscissa)
    res_norm_dist = np.nan_to_num(res_norm_dist)
    res_curv = np.nan_to_num(res_curv)
    res_tangent = np.nan_to_num(res_tangent)

    # 6. 保存 CSV
    df = pd.DataFrame({
        "x": surf_points[:, 0],
        "y": surf_points[:, 1],
        "z": surf_points[:, 2],
        "Abscissa": res_abscissa,
        "NormRadius": res_norm_dist,
        "Curvature": res_curv,
        "Tangent_X": res_tangent[:, 0],
        "Tangent_Y": res_tangent[:, 1],
        "Tangent_Z": res_tangent[:, 2]
    })
    
    df.to_csv(output_csv_path, index=False)
    print(f"✅ [Scenario A] Saved: {output_csv_path}")

if __name__ == "__main__":
    # 测试用例
    stl_file = "MA+XIAO+DONG-new.stl"
    if os.path.exists(stl_file):
        process_surface_dataset(stl_file, "train_data_surface.csv")
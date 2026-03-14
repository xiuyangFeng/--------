import os
import numpy as np
import pandas as pd


def _load_geometry_dependencies():
    import vtk
    from vtkmodules.util.numpy_support import vtk_to_numpy, numpy_to_vtk

    try:
        from pipeline import vmtk_core  # type: ignore
    except ImportError:
        from . import vmtk_core  # type: ignore

    return vtk, vtk_to_numpy, numpy_to_vtk, vmtk_core

def process_surface_dataset(stl_path, output_csv_path):
    """
    处理表面数据集，提取几何特征并保存为CSV格式
    参数:
        stl_path: STL文件路径
        output_csv_path: 输出CSV文件路径
    """
    vtk, vtk_to_numpy, _, vmtk_core = _load_geometry_dependencies()
    print(f"🚀 [Scenario A] Processing Surface: {stl_path}")
    
    # 1. 读取表面几何模型
    surface = vmtk_core.read_surface(stl_path)
    
    # 2. 调用核心模块提取增强版中心线 (包含手动计算的所有特征)
    centerline = vmtk_core.extract_rich_centerline(surface)
    
    # 3. 准备映射 (Mapping)，获取中心线上的数据数组
    cl_points = vtk_to_numpy(centerline.GetPoints().GetData())  # 获取中心线点坐标
    cl_pd = centerline.GetPointData()  # 获取中心线点数据
    
    # 注意：这里的名字必须和 vmtk_core 里 add_array 的名字一致
    arr_abscissa = vtk_to_numpy(cl_pd.GetArray("Abscissas"))  # 获取弧长参数
    arr_radius   = vtk_to_numpy(cl_pd.GetArray("MaximumInscribedSphereRadius"))  # 获取最大内切球半径
    arr_curv     = vtk_to_numpy(cl_pd.GetArray("Curvature"))  # 获取曲率
    arr_tangent  = vtk_to_numpy(cl_pd.GetArray("FrenetTangent"))  # 获取弗雷奈切向量
    
    # 构建 KDTree 加速查找最近邻点
    locator = vtk.vtkKdTreePointLocator()
    locator.SetDataSet(centerline)
    locator.BuildLocator()
    
    # 获取表面点坐标
    surf_points = vtk_to_numpy(surface.GetPoints().GetData())
    n_pts = surf_points.shape[0]  # 表面点的数量
    
    # 初始化结果容器，用于存储映射后的特征
    res_abscissa = np.zeros(n_pts)  # 弧长参数
    res_norm_dist = np.zeros(n_pts)  # 归一化距离
    res_curv = np.zeros(n_pts)  # 曲率
    res_tangent = np.zeros((n_pts, 3))  # 切向量
    
    print(f"   ...Mapping features to {n_pts} surface nodes...")
    
    # 4. 遍历映射，将中心线特征映射到表面节点
    for i in range(n_pts):
        pt = surf_points[i]  # 当前表面点
        closest_id = locator.FindClosestPoint(pt)  # 找到最近的中心线点ID
        
        # 计算物理距离（表面点到中心线点的距离）
        dist = np.linalg.norm(pt - cl_points[closest_id])
        local_r = arr_radius[closest_id]  # 获取该中心线点的局部半径
        
        # 拷贝对应中心线点的特征值
        res_abscissa[i] = arr_abscissa[closest_id]
        res_curv[i] = arr_curv[closest_id]
        res_tangent[i] = arr_tangent[closest_id]
        
        # 计算归一化径向距离 (dist / R)
        # 对于壁面点，这个值理论上应该接近 1.0
        safe_r = local_r if local_r > 1e-6 else 1.0  # 防止除零错误
        res_norm_dist[i] = dist / safe_r

    # 5. 后处理 (归一化 Abscissa 到 0-1范围)
    ab_min, ab_max = np.nanmin(res_abscissa), np.nanmax(res_abscissa)  # 获取最小最大值
    denom = ab_max - ab_min  # 计算差值
    
    # 如果差值足够大，则进行归一化处理
    if denom > 1e-8:
        res_abscissa = (res_abscissa - ab_min) / denom
    else:
        res_abscissa[:] = 0.0  # 否则全部设置为0
        
    # 清洗 NaN 值，将其替换为0
    res_abscissa = np.nan_to_num(res_abscissa)
    res_norm_dist = np.nan_to_num(res_norm_dist)
    res_curv = np.nan_to_num(res_curv)
    res_tangent = np.nan_to_num(res_tangent)

    # 6. 保存为 CSV 文件
    df = pd.DataFrame({
        "x": surf_points[:, 0],  # X坐标
        "y": surf_points[:, 1],  # Y坐标
        "z": surf_points[:, 2],  # Z坐标
        "Abscissa": res_abscissa,  # 弧长参数（已归一化）
        "NormRadius": res_norm_dist,  # 归一化半径
        "Curvature": res_curv,  # 曲率
        "Tangent_X": res_tangent[:, 0],  # 切向量X分量
        "Tangent_Y": res_tangent[:, 1],  # 切向量Y分量
        "Tangent_Z": res_tangent[:, 2]   # 切向量Z分量
    })
    
    df.to_csv(output_csv_path, index=False)  # 保存到CSV文件，不包含行索引
    print(f"✅ [Scenario A] Saved: {output_csv_path}")

if __name__ == "__main__":
    # 测试用例，处理指定的STL文件
    stl_file = "MA+XIAO+DONG-new.stl"
    if os.path.exists(stl_file):  # 检查文件是否存在
        process_surface_dataset(stl_file, "train_data_surface.csv")

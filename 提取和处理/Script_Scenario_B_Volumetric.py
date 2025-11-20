import os
import numpy as np
import pandas as pd
import vtk
from vtkmodules.util.numpy_support import vtk_to_numpy
import vmtk_core  # 导入核心模块

def process_volumetric_dataset(stl_path, cfd_cloud_path, output_csv_path):
    print(f"🚀 [Scenario B] Processing Volume: STL={stl_path}, Cloud={cfd_cloud_path}")
    
    # 1. 准备中心线 (骨架)
    # 即使是对内部点云处理，我们也需要 STL 来生成准确的中心线作为参照系
    surface = vmtk_core.read_surface(stl_path)
    centerline = vmtk_core.extract_rich_centerline(surface)
    
    # 获取骨架数据
    cl_points = vtk_to_numpy(centerline.GetPoints().GetData())
    cl_pd = centerline.GetPointData()
    
    arr_abscissa = vtk_to_numpy(cl_pd.GetArray("Abscissas"))
    arr_radius   = vtk_to_numpy(cl_pd.GetArray("MaximumInscribedSphereRadius"))
    arr_curv     = vtk_to_numpy(cl_pd.GetArray("Curvature"))
    arr_tangent  = vtk_to_numpy(cl_pd.GetArray("FrenetTangent"))
    
    locator = vtk.vtkKdTreePointLocator()
    locator.SetDataSet(centerline)
    locator.BuildLocator()
    
    # 2. 加载 CFD 点云数据
    # 假设数据包含坐标 (x,y,z) 和 物理标签 (u,v,w,p)
    if cfd_cloud_path.endswith('.npy'):
        data = np.load(cfd_cloud_path)
        # 假设前3列是坐标，后面是标签
        cloud_xyz = data[:, 0:3] 
        cloud_others = data[:, 3:] 
    elif cfd_cloud_path.endswith('.csv'):
        df_in = pd.read_csv(cfd_cloud_path)
        cloud_xyz = df_in[['x', 'y', 'z']].values
        # 保留除了坐标以外的所有列 (如 u, v, w, p, is_wall)
        cloud_others_df = df_in.drop(columns=['x', 'y', 'z'])
    else:
        raise ValueError("只支持 .npy 或 .csv 格式的点云文件")
        
    n_pts = cloud_xyz.shape[0]
    print(f"   ...Mapping features to {n_pts} volumetric points...")
    
    # 3. 映射特征
    geo_abscissa = np.zeros(n_pts)
    geo_norm_dist = np.zeros(n_pts)
    geo_curv = np.zeros(n_pts)
    geo_tangent = np.zeros((n_pts, 3))
    
    for i in range(n_pts):
        pt = cloud_xyz[i]
        closest_id = locator.FindClosestPoint(pt)
        
        dist = np.linalg.norm(pt - cl_points[closest_id])
        local_r = arr_radius[closest_id]
        
        geo_abscissa[i] = arr_abscissa[closest_id]
        geo_curv[i] = arr_curv[closest_id]
        geo_tangent[i] = arr_tangent[closest_id]
        
        # 这里的 NormRadius 非常重要！
        # 0.0 = 血管中心, 1.0 = 血管壁
        safe_r = local_r if local_r > 1e-6 else 1.0
        geo_norm_dist[i] = dist / safe_r

    # 4. 归一化 Abscissa
    ab_min, ab_max = np.nanmin(geo_abscissa), np.nanmax(geo_abscissa)
    denom = ab_max - ab_min
    if denom > 1e-8:
        geo_abscissa = (geo_abscissa - ab_min) / denom
    else:
        geo_abscissa[:] = 0.0
        
    # 清洗
    geo_abscissa = np.nan_to_num(geo_abscissa)
    geo_norm_dist = np.nan_to_num(geo_norm_dist)
    geo_curv = np.nan_to_num(geo_curv)
    geo_tangent = np.nan_to_num(geo_tangent)

    # 5. 整合并保存
    # 我们构建一个新的 DataFrame，包含：坐标 + 几何特征 + 原始CFD标签
    df_out = pd.DataFrame({
        "x": cloud_xyz[:, 0],
        "y": cloud_xyz[:, 1],
        "z": cloud_xyz[:, 2],
        "Abscissa": geo_abscissa,
        "NormRadius": geo_norm_dist,
        "Curvature": geo_curv,
        "Tangent_X": geo_tangent[:, 0],
        "Tangent_Y": geo_tangent[:, 1],
        "Tangent_Z": geo_tangent[:, 2]
    })
    
    # 如果原始文件是 CSV，把标签列拼回来
    if cfd_cloud_path.endswith('.csv'):
        df_out = pd.concat([df_out, cloud_others_df], axis=1)
    elif cfd_cloud_path.endswith('.npy'):
        # 如果是 npy，假设我们知道标签列的含义，这里简单起见存为 f_0, f_1...
        for k in range(cloud_others.shape[1]):
            df_out[f"Feature_{k}"] = cloud_others[:, k]
            
    df_out.to_csv(output_csv_path, index=False)
    print(f"✅ [Scenario B] Saved: {output_csv_path}")

if __name__ == "__main__":
    # 测试配置
    stl_file = "MA+XIAO+DONG-new.stl"   # 提供几何骨架
    
    # 假设你有一个已经存在的点云文件 (这里为了测试，你可以暂时用上面的 result_features.csv 假装是 CFD 点云)
    # 实际使用时，替换成组内的真实数据路径
    cfd_file = "result_features.csv" 
    
    if os.path.exists(stl_file) and os.path.exists(cfd_file):
        process_volumetric_dataset(stl_file, cfd_file, "train_data_volumetric.csv")
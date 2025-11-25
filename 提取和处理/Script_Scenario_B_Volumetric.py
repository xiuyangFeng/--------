import os
import numpy as np
import pandas as pd
import vtk
from vtkmodules.util.numpy_support import vtk_to_numpy
import vmtk_core  # 导入核心模块

def process_volumetric_dataset(stl_path, cfd_cloud_path, output_csv_path):
    """
    处理体积数据集，将几何特征映射到CFD点云数据
    参数:
        stl_path: STL几何模型文件路径
        cfd_cloud_path: CFD点云数据文件路径
        output_csv_path: 输出CSV文件路径
    """
    print(f"🚀 [Scenario B] Processing Volume: STL={stl_path}, Cloud={cfd_cloud_path}")
    
    # 1. 准备中心线 (骨架)
    # 即使是对内部点云处理，我们也需要 STL 来生成准确的中心线作为参照系
    surface = vmtk_core.read_surface(stl_path)  # 读取表面几何模型
    centerline = vmtk_core.extract_rich_centerline(surface)  # 提取丰富的中心线特征
    
    # 获取骨架数据
    cl_points = vtk_to_numpy(centerline.GetPoints().GetData())  # 中心线点坐标
    cl_pd = centerline.GetPointData()  # 中心线点数据
    
    # 获取中心线的各种属性数组
    arr_abscissa = vtk_to_numpy(cl_pd.GetArray("Abscissas"))  # 弧长参数
    arr_radius   = vtk_to_numpy(cl_pd.GetArray("MaximumInscribedSphereRadius"))  # 最大内切球半径
    arr_curv     = vtk_to_numpy(cl_pd.GetArray("Curvature"))  # 曲率
    arr_tangent  = vtk_to_numpy(cl_pd.GetArray("FrenetTangent"))  # 弗雷奈切向量
    
    # 创建KDTree定位器，用于快速查找最近点
    locator = vtk.vtkKdTreePointLocator()
    locator.SetDataSet(centerline)
    locator.BuildLocator()
    
    # 2. 加载 CFD 点云数据
    # 假设数据包含坐标 (x,y,z) 和 物理标签 (u,v,w,p)
    if cfd_cloud_path.endswith('.npy'):
        # 如果是.npy格式，直接加载numpy数组
        data = np.load(cfd_cloud_path)
        # 假设前3列是坐标，后面是标签
        cloud_xyz = data[:, 0:3] 
        cloud_others = data[:, 3:] 
    elif cfd_cloud_path.endswith('.csv'):
        # 如果是.csv格式，使用pandas读取
        df_in = pd.read_csv(cfd_cloud_path)
        cloud_xyz = df_in[['x', 'y', 'z']].values  # 提取坐标列
        # 保留除了坐标以外的所有列 (如 u, v, w, p, is_wall)
        cloud_others_df = df_in.drop(columns=['x', 'y', 'z'])
    else:
        # 不支持的格式抛出异常
        raise ValueError("只支持 .npy 或 .csv 格式的点云文件")
        
    n_pts = cloud_xyz.shape[0]  # 获取点云数量
    print(f"   ...Mapping features to {n_pts} volumetric points...")
    
    # 3. 映射特征，初始化几何特征数组
    geo_abscissa = np.zeros(n_pts)  # 弧长参数
    geo_norm_dist = np.zeros(n_pts)  # 归一化距离
    geo_curv = np.zeros(n_pts)  # 曲率
    geo_tangent = np.zeros((n_pts, 3))  # 切向量
    
    # 对每个点云点进行特征映射
    for i in range(n_pts):
        pt = cloud_xyz[i]  # 当前点云点
        closest_id = locator.FindClosestPoint(pt)  # 找到最近的中心线点
        
        # 计算物理距离和局部半径
        dist = np.linalg.norm(pt - cl_points[closest_id])  # 点到中心线的距离
        local_r = arr_radius[closest_id]  # 对应中心线点的局部半径
        
        # 将中心线特征复制到当前点
        geo_abscissa[i] = arr_abscissa[closest_id]
        geo_curv[i] = arr_curv[closest_id]
        geo_tangent[i] = arr_tangent[closest_id]
        
        # 这里的 NormRadius 非常重要！
        # 0.0 = 血管中心, 1.0 = 血管壁
        safe_r = local_r if local_r > 1e-6 else 1.0  # 防止除零错误
        geo_norm_dist[i] = dist / safe_r  # 计算归一化径向距离

    # 4. 归一化 Abscissa 弧长参数
    ab_min, ab_max = np.nanmin(geo_abscissa), np.nanmax(geo_abscissa)  # 获取最值
    denom = ab_max - ab_min  # 计算差值
    # 如果差值足够大，则进行归一化
    if denom > 1e-8:
        geo_abscissa = (geo_abscissa - ab_min) / denom
    else:
        geo_abscissa[:] = 0.0  # 否则全部置零
        
    # 清洗NaN值，将其替换为0
    geo_abscissa = np.nan_to_num(geo_abscissa)
    geo_norm_dist = np.nan_to_num(geo_norm_dist)
    geo_curv = np.nan_to_num(geo_curv)
    geo_tangent = np.nan_to_num(geo_tangent)

    # 5. 整合并保存
    # 我们构建一个新的 DataFrame，包含：坐标 + 几何特征 + 原始CFD标签
    df_out = pd.DataFrame({
        "x": cloud_xyz[:, 0],  # X坐标
        "y": cloud_xyz[:, 1],  # Y坐标
        "z": cloud_xyz[:, 2],  # Z坐标
        "Abscissa": geo_abscissa,  # 归一化后的弧长参数
        "NormRadius": geo_norm_dist,  # 归一化半径（0为中心，1为壁面）
        "Curvature": geo_curv,  # 曲率
        "Tangent_X": geo_tangent[:, 0],  # 切向量X分量
        "Tangent_Y": geo_tangent[:, 1],  # 切向量Y分量
        "Tangent_Z": geo_tangent[:, 2]   # 切向量Z分量
    })
    
    # 如果原始文件是 CSV，把标签列拼回来
    if cfd_cloud_path.endswith('.csv'):
        df_out = pd.concat([df_out, cloud_others_df], axis=1)  # 按列拼接
    elif cfd_cloud_path.endswith('.npy'):
        # 如果是 npy，假设我们知道标签列的含义，这里简单起见存为 f_0, f_1...
        for k in range(cloud_others.shape[1]):
            df_out[f"Feature_{k}"] = cloud_others[:, k]  # 添加其他特征列
            
    df_out.to_csv(output_csv_path, index=False)  # 保存为CSV文件
    print(f"✅ [Scenario B] Saved: {output_csv_path}")

if __name__ == "__main__":
    # 测试配置
    stl_file = "MA+XIAO+DONG-new.stl"   # 提供几何骨架
    cfd_cloud_file = "cfd_volume_points.npy"  # 提供CFD点云数据
import os  # 导入操作系统相关模块，用于处理文件路径等操作
import numpy as np  # 导入numpy库，用于数值计算
import pandas as pd  # 导入pandas库，用于数据处理和分析
import vtk  # 导入VTK库，用于三维计算机图形学处理
from vtkmodules.util.numpy_support import vtk_to_numpy  # 从VTK模块导入vtk_to_numpy函数，用于VTK数据与numpy数组之间的转换
import vmtk_core  # 导入核心模块，用于血管建模工具包的核心功能

def prepare_geometry_data(stl_path):  # 定义预处理几何数据的函数，接收STL文件路径作为参数
    """
    预处理几何数据：读取STL，提取中心线，构建KDTree。
    返回一个包含所有必要几何对象的字典/元组，供后续重复使用。
    """
    print(f"   [Geometry] Pre-processing surface: {stl_path}")  # 打印正在预处理的表面文件路径
    # 1. 准备中心线 (骨架)
    surface = vmtk_core.read_surface(stl_path)  # 使用vmtk_core读取STL表面文件
    centerline = vmtk_core.extract_rich_centerline(surface)  # 从表面提取丰富的中心线数据
    
    # 获取骨架数据
    cl_points = vtk_to_numpy(centerline.GetPoints().GetData())  # 将中心线点数据转换为numpy数组
    cl_pd = centerline.GetPointData()  # 获取中心线点数据对象
    
    # 获取中心线的各种属性数组
    geo_data = {  # 创建包含各种几何数据的字典
        "cl_points": cl_points,  # 存储中心线点坐标
        "arr_abscissa": vtk_to_numpy(cl_pd.GetArray("Abscissas")),  # 存储弧长坐标数据
        "arr_radius": vtk_to_numpy(cl_pd.GetArray("MaximumInscribedSphereRadius")),  # 存储最大内切球半径数据
        "arr_curv": vtk_to_numpy(cl_pd.GetArray("Curvature")),  # 存储曲率数据
        "arr_tangent": vtk_to_numpy(cl_pd.GetArray("FrenetTangent")),  # 存储弗勒内标架切线数据
        "locator": None  # 初始化定位器为空
    }
    
    # 创建KDTree定位器
    locator = vtk.vtkKdTreePointLocator()  # 创建VTK的KD树点定位器实例
    locator.SetDataSet(centerline)  # 设置定位器的数据集为中心线数据
    locator.BuildLocator()  # 构建定位器
    geo_data["locator"] = locator  # 将构建好的定位器存储到几何数据字典中
    
    return geo_data  # 返回包含所有几何数据的字典

def process_single_cloud(geo_data, cfd_cloud_path, output_csv_path):  # 定义处理单个CFD点云文件的函数
    """
    使用预处理好的几何数据(geo_data)处理单个CFD点云文件。
    """
    # 解包几何数据
    cl_points = geo_data["cl_points"]  # 从几何数据中获取中心线点坐标
    arr_abscissa = geo_data["arr_abscissa"]  # 从几何数据中获取弧长坐标数组
    arr_radius = geo_data["arr_radius"]  # 从几何数据中获取半径数组
    arr_curv = geo_data["arr_curv"]  # 从几何数据中获取曲率数组
    arr_tangent = geo_data["arr_tangent"]  # 从几何数据中获取切线数组
    locator = geo_data["locator"]  # 从几何数据中获取定位器

    # 2. 加载 CFD 点云数据
    if cfd_cloud_path.endswith('.npy'):  # 判断点云文件是否为.npy格式
        data = np.load(cfd_cloud_path)  # 加载.npy格式的点云数据
        cloud_xyz = data[:, 0:3]  # 提取前3列作为XYZ坐标
        cloud_others = data[:, 3:]  # 提取第4列及之后的数据作为其他特征
    elif cfd_cloud_path.endswith('.csv'):  # 判断点云文件是否为.csv格式
        df_in = pd.read_csv(cfd_cloud_path)  # 读取CSV格式的点云数据
        cloud_xyz = df_in[['x', 'y', 'z']].values  # 提取x,y,z列作为坐标值
        cloud_others_df = df_in.drop(columns=['x', 'y', 'z'])  # 删除x,y,z列，保留其他列作为其他特征
    else:  # 如果不是支持的格式
        raise ValueError("只支持 .npy 或 .csv 格式的点云文件")  # 抛出异常提示仅支持.npy或.csv格式
        
    n_pts = cloud_xyz.shape[0]  # 获取点云中点的数量
    # print(f"   ...Mapping features to {n_pts} volumetric points...")  # 注释掉的打印语句，显示正在映射特征到体素点
    
    # 3. 映射特征
    geo_abscissa = np.zeros(n_pts)  # 初始化弧长坐标的数组
    geo_norm_dist = np.zeros(n_pts)  # 初始化归一化距离的数组
    geo_curv = np.zeros(n_pts)  # 初始化曲率数组
    geo_tangent = np.zeros((n_pts, 3))  # 初始化切线数组，每个点有3个分量(x,y,z)
    
    for i in range(n_pts):  # 遍历每一个点云中的点
        pt = cloud_xyz[i]  # 获取当前点的坐标
        closest_id = locator.FindClosestPoint(pt)  # 使用定位器找到中心线上最近的点的ID
        
        dist = np.linalg.norm(pt - cl_points[closest_id])  # 计算当前点到中心线上最近点的距离
        local_r = arr_radius[closest_id]  # 获取中心线上最近点的局部半径
        
        geo_abscissa[i] = arr_abscissa[closest_id]  # 将中心线上最近点的弧长坐标赋给当前点
        geo_curv[i] = arr_curv[closest_id]  # 将中心线上最近点的曲率赋给当前点
        geo_tangent[i] = arr_tangent[closest_id]  # 将中心线上最近点的切线向量赋给当前点
        
        safe_r = local_r if local_r > 1e-6 else 1.0  # 确保半径不为0，避免除法运算错误
        geo_norm_dist[i] = dist / safe_r  # 计算并存储归一化距离（距离/半径）

    # 4. 归一化 Abscissa
    ab_min, ab_max = np.nanmin(geo_abscissa), np.nanmax(geo_abscissa)  # 计算弧长坐标的最小值和最大值
    denom = ab_max - ab_min  # 计算差值作为分母
    if denom > 1e-8:  # 如果分母足够大
        geo_abscissa = (geo_abscissa - ab_min) / denom  # 对弧长坐标进行归一化处理
    else:  # 如果分母过小
        geo_abscissa[:] = 0.0  # 将所有弧长坐标设为0
        
    geo_abscissa = np.nan_to_num(geo_abscissa)  # 将NaN值替换为0
    geo_norm_dist = np.nan_to_num(geo_norm_dist)  # 将NaN值替换为0
    geo_curv = np.nan_to_num(geo_curv)  # 将NaN值替换为0
    geo_tangent = np.nan_to_num(geo_tangent)  # 将NaN值替换为0

    # 5. 整合并保存
    df_out = pd.DataFrame({  # 创建输出的DataFrame对象
        "x": cloud_xyz[:, 0],  # 存储x坐标
        "y": cloud_xyz[:, 1],  # 存储y坐标
        "z": cloud_xyz[:, 2],  # 存储z坐标
        "Abscissa": geo_abscissa,  # 存储弧长坐标
        "NormRadius": geo_norm_dist,  # 存储归一化半径
        "Curvature": geo_curv,  # 存储曲率
        "Tangent_X": geo_tangent[:, 0],  # 存储切线X分量
        "Tangent_Y": geo_tangent[:, 1],  # 存储切线Y分量
        "Tangent_Z": geo_tangent[:, 2]  # 存储切线Z分量
    })
    
    if cfd_cloud_path.endswith('.csv'):  # 如果原始文件是CSV格式
        df_out = pd.concat([df_out, cloud_others_df], axis=1)  # 将其他特征数据连接到输出DataFrame
    elif cfd_cloud_path.endswith('.npy'):  # 如果原始文件是NPY格式
        for k in range(cloud_others.shape[1]):  # 遍历所有其他特征
            df_out[f"Feature_{k}"] = cloud_others[:, k]  # 将每个特征作为一个新列添加到DataFrame
            
    df_out.to_csv(output_csv_path, index=False)  # 将结果保存为CSV文件，不包含行索引
    # print(f"✅ Saved: {output_csv_path}")  # 注释掉的打印语句，显示保存成功的信息

def process_volumetric_dataset(stl_path, cfd_cloud_path, output_csv_path):  # 定义处理体数据集的函数
    """
    [Legacy Wrapper] 保持原有接口不变，但内部调用拆分后的函数。
    """
    print(f"🚀 [Scenario B] Processing Volume: STL={stl_path}, Cloud={cfd_cloud_path}")  # 打印处理信息
    geo_data = prepare_geometry_data(stl_path)  # 预处理几何数据
    process_single_cloud(geo_data, cfd_cloud_path, output_csv_path)  # 处理单个点云文件
    print(f"✅ [Scenario B] Saved: {output_csv_path}")  # 打印保存成功的消息

if __name__ == "__main__":  # 当脚本作为主程序运行时执行以下代码
    # 测试配置
    stl_file = "MA+XIAO+DONG-new.stl"   # 提供几何骨架
    cfd_cloud_file = "cfd_volume_points.npy"  # 提供CFD点云数据
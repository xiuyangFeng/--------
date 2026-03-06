import os  # 导入操作系统相关模块，用于处理文件路径等操作
import json  # 导入json模块，用于读取归一化参数文件
import numpy as np  # 导入numpy库，用于数值计算
import pandas as pd  # 导入pandas库，用于数据处理和分析


def _load_geometry_dependencies():
    import vtk  # 导入VTK库，用于三维计算机图形学处理
    from vtkmodules.util.numpy_support import vtk_to_numpy

    try:
        from . import vmtk_core  # type: ignore
    except ImportError:
        import vmtk_core  # type: ignore

    return vtk, vtk_to_numpy, vmtk_core


def restore_coordinates(coords: np.ndarray, norm_params: dict) -> np.ndarray:
    """
    根据归一化参数将坐标还原到原始坐标系。
    
    参数:
        coords: 归一化后的坐标数组 (N, 3)
        norm_params: 归一化参数字典，包含 method, centroid/min_values, scale_factor/ranges
    
    返回:
        还原后的坐标数组 (N, 3)
    """
    method = norm_params.get('method', 'center_scale')
    
    if method == 'center_scale':
        centroid = np.array(norm_params['centroid'])
        scale_factor = norm_params['scale_factor']
        # original = normalized * scale_factor + centroid
        restored = coords * scale_factor + centroid
    elif method == 'min_max':
        min_vals = np.array(norm_params['min_values'])
        ranges = np.array(norm_params['ranges'])
        # original = normalized * range + min
        restored = coords * ranges + min_vals
    else:
        print(f"⚠️  未知的归一化方法: {method}，返回原始坐标")
        restored = coords
    
    return restored


def load_normalization_params(norm_params_path: str) -> dict:
    """
    从 JSON 文件加载归一化参数。
    
    参数:
        norm_params_path: 归一化参数文件路径
    
    返回:
        归一化参数字典，如果文件不存在则返回 None
    """
    if not os.path.exists(norm_params_path):
        return None
    
    try:
        with open(norm_params_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception as e:
        print(f"⚠️  读取归一化参数文件失败: {e}")
        return None

def prepare_geometry_data(stl_path):  # 定义预处理几何数据的函数，接收STL文件路径作为参数
    """
    预处理几何数据：读取STL，提取中心线，构建KDTree。
    返回一个包含所有必要几何对象的字典/元组，供后续重复使用。
    """
    vtk, vtk_to_numpy, vmtk_core = _load_geometry_dependencies()
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

def process_single_cloud(geo_data, cfd_cloud_path, output_csv_path, global_bcs=None, norm_params=None):  # 定义处理单个CFD点云文件的函数
    """
    使用预处理好的几何数据(geo_data)处理单个CFD点云文件。
    
    参数:
        geo_data: 预处理的几何数据字典
        cfd_cloud_path: CFD点云文件路径
        output_csv_path: 输出CSV文件路径
        global_bcs: 全局边界条件
        norm_params: 归一化参数字典（如果点云坐标已归一化，需要提供此参数以还原坐标进行特征映射）
    """
    _, _, _ = _load_geometry_dependencies()
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
    
    # 如果提供了归一化参数，将坐标还原到原始坐标系进行特征映射
    if norm_params is not None:
        cloud_xyz_for_mapping = restore_coordinates(cloud_xyz, norm_params)  # 还原坐标用于特征映射
    else:
        cloud_xyz_for_mapping = cloud_xyz  # 直接使用原始坐标
        
    n_pts = cloud_xyz.shape[0]  # 获取点云中点的数量
    # print(f"   ...Mapping features to {n_pts} volumetric points...")  # 注释掉的打印语句，显示正在映射特征到体素点
    
    # 3. 映射特征
    geo_abscissa = np.zeros(n_pts)  # 初始化弧长坐标的数组
    geo_norm_dist = np.zeros(n_pts)  # 初始化归一化距离的数组
    geo_curv = np.zeros(n_pts)  # 初始化曲率数组
    geo_tangent = np.zeros((n_pts, 3))  # 初始化切线数组，每个点有3个分量(x,y,z)
    
    for i in range(n_pts):  # 遍历每一个点云中的点
        pt = cloud_xyz_for_mapping[i]  # 获取当前点的坐标（使用还原后的坐标进行特征映射）
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
    
    if cfd_cloud_path.endswith('.csv'):
        df_out = pd.concat([df_out, cloud_others_df], axis=1)
    elif cfd_cloud_path.endswith('.npy'):
        for k in range(cloud_others.shape[1]):
            df_out[f"Feature_{k}"] = cloud_others[:, k]

    # 6. 添加全局边界条件 (Global Boundary Conditions)
    # 支持两种格式:
    #   - 列表格式: [BC_Flag, Inlet, O1, O2, O3, O4]
    #   - 字典格式: {bc_name: bc_value, ...} (向后兼容)
    if global_bcs:
        if isinstance(global_bcs, list) and len(global_bcs) == 6:
            # 新格式: [BC_Flag, Inlet, O1, O2, O3, O4]
            bc_names = ["BC_Flag", "BC_Inlet", "BC_O1", "BC_O2", "BC_O3", "BC_O4"]
            for bc_name, bc_value in zip(bc_names, global_bcs):
                df_out[bc_name] = bc_value
        elif isinstance(global_bcs, dict):
            # 旧格式: 字典形式 (向后兼容)
            for bc_name, bc_value in global_bcs.items():
                df_out[bc_name] = bc_value
        else:
            print(f"⚠️  [警告] 无法识别的边界条件格式: {type(global_bcs)}")
            
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

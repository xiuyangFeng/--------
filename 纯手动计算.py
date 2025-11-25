"""
纯手动计算几何特征程序
功能：基于VTK库读取血管模型表面数据，提取中心线，并手动计算各种几何特征
"""

# 导入所需的标准库和第三方库
import os                         # 用于操作系统相关功能，如文件路径操作
import numpy as np                # 数值计算库，用于高效的数组运算
import pandas as pd               # 数据分析库，用于数据处理和CSV导出
import vtk                        # Visualization Toolkit，用于3D计算机图形学
from vmtk import vmtkscripts      # 血管建模工具包，专门用于血管图像处理
from vtkmodules.util.numpy_support import vtk_to_numpy, numpy_to_vtk  # VTK与NumPy数组转换工具

# ==========================================
# 1. 基础工具函数部分
# ==========================================

def read_surface(path):
    """
    读取表面模型文件（支持VTP和STL格式）
    
    参数:
        path (str): 文件路径
        
    返回:
        vtkPolyData: 表面模型数据
    """
    # 根据文件扩展名选择合适的读取器
    if path.endswith(".vtp"):
        reader = vtk.vtkXMLPolyDataReader()  # VTP格式读取器
    else:
        reader = vtk.vtkSTLReader()          # STL格式读取器
    reader.SetFileName(path)                 # 设置文件路径
    reader.Update()                          # 执行读取操作
    return reader.GetOutput()                # 返回读取的数据

def write_vtp(polydata, path):
    """
    将polydata数据写入VTP文件
    
    参数:
        polydata (vtkPolyData): 要写入的数据
        path (str): 输出文件路径
    """
    writer = vtk.vtkXMLPolyDataWriter()     # 创建VTP写入器
    writer.SetFileName(path)                # 设置输出文件名
    writer.SetInputData(polydata)           # 设置输入数据
    writer.Write()                          # 执行写入操作

# ==========================================
# 2. 核心算法模块 (全部手动计算)
# ==========================================

def detect_all_openings(surface):
    """
    检测血管模型的所有开口（入口和出口）
    
    参数:
        surface (vtkPolyData): 血管表面模型数据
        
    返回:
        tuple: (入口点坐标, 出口点坐标列表)
    """
    print("   [1.1] 扫描网格边界...")       # 打印进度信息
    
    # 提取边界边（用于识别开口）
    feature_edges = vtk.vtkFeatureEdges()
    feature_edges.SetInputData(surface)              # 设置输入数据
    feature_edges.BoundaryEdgesOn()                  # 开启边界边提取
    feature_edges.FeatureEdgesOff()                  # 关闭特征边提取
    feature_edges.ManifoldEdgesOff()                 # 关闭流形边提取
    feature_edges.NonManifoldEdgesOff()              # 关闭非流形边提取
    feature_edges.Update()                           # 执行操作
    
    # 连通区域分析，用于分离不同的开口
    conn = vtk.vtkConnectivityFilter()
    conn.SetInputData(feature_edges.GetOutput())     # 设置输入数据
    conn.SetExtractionModeToAllRegions()             # 提取所有连通区域
    conn.ColorRegionsOn()                            # 为不同区域着色以便区分
    conn.Update()                                    # 执行操作
    
    # 获取连通区域数量
    num_regions = conn.GetNumberOfExtractedRegions()
    opening_centers = []                             # 存储开口中心点
    valid_count = 0                                  # 有效开口计数
    
    # 遍历每个连通区域
    for i in range(num_regions):
        # 使用阈值过滤器提取特定区域
        thresh = vtk.vtkThreshold()
        thresh.SetInputData(conn.GetOutput())                                # 设置输入数据
        thresh.SetInputArrayToProcess(0, 0, 0, vtk.vtkDataObject.FIELD_ASSOCIATION_CELLS, "RegionId")  # 设置处理数组
        thresh.SetLowerThreshold(i)                                          # 设置下限阈值
        thresh.SetUpperThreshold(i)                                          # 设置上限阈值
        thresh.Update()                                                      # 执行操作
        
        loop = thresh.GetOutput()                                            # 获取当前区域数据
        if loop.GetNumberOfPoints() < 10: continue                           # 忽略过小的区域（少于10个点）
            
        # 计算当前开口的中心点
        points = vtk_to_numpy(loop.GetPoints().GetData())                    # 将VTK点数据转为numpy数组
        opening_centers.append(np.mean(points, axis=0))                      # 计算平均坐标作为中心点
        valid_count += 1                                                     # 有效开口计数加1
        
    # 如果有效开口少于2个，则返回None
    if valid_count < 2: return None, None

    # 将开口中心点转换为numpy数组
    opening_centers = np.array(opening_centers)
    # 将Z坐标最大的开口作为入口
    inlet_idx = np.argmax(opening_centers[:, 2])                             # 找到Z坐标最大的点索引
    # 返回入口点和其他所有出口点
    return list(opening_centers[inlet_idx]), np.delete(opening_centers, inlet_idx, axis=0).tolist()

def clean_centerline(centerline, surface):
    """
    清洗中心线数据，去除超出表面边界的无效部分
    
    参数:
        centerline (vtkPolyData): 原始中心线数据
        surface (vtkPolyData): 血管表面模型
        
    返回:
        vtkPolyData: 清洗后的中心线数据
    """
    print("   [2.1] 清洗中心线...")          # 打印进度信息
    
    # 获取表面边界并添加缓冲区
    bounds = surface.GetBounds()             # 获取表面边界框
    buffer = 0.5                             # 设置缓冲距离
    
    # 创建边界框函数
    box = vtk.vtkBox()
    box.SetBounds(bounds[0]-buffer, bounds[1]+buffer,        # 设置X轴范围（带缓冲）
                  bounds[2]-buffer, bounds[3]+buffer,        # 设置Y轴范围（带缓冲）
                  bounds[4]-buffer, bounds[5]+buffer)        # 设置Z轴范围（带缓冲）
    
    # 使用边界框裁剪中心线
    clipper = vtk.vtkClipPolyData()
    clipper.SetInputData(centerline)         # 设置要裁剪的数据
    clipper.SetClipFunction(box)             # 设置裁剪函数
    clipper.InsideOutOn()                    # 反转裁剪方向（保留内部）
    clipper.Update()                         # 执行裁剪
    
    clipped = clipper.GetOutput()            # 获取裁剪结果
    # 如果裁剪后没有点数据，则返回原始中心线
    if clipped.GetNumberOfPoints() == 0: return centerline

    # 只保留最大的连通区域
    conn = vtk.vtkConnectivityFilter()
    conn.SetInputData(clipped)               # 设置输入数据
    conn.SetExtractionModeToLargestRegion()  # 只提取最大连通区域
    conn.Update()                            # 执行操作
    
    # 返回清洗后的中心线
    return conn.GetOutput()

# --- 手动计算几何特征函数 ---

def compute_tangents_manually(points):
    """
    手动计算曲线上的切线向量（一阶导数归一化）
    
    参数:
        points (np.ndarray): 曲线上各点的坐标数组，形状为(N, 3)
        
    返回:
        np.ndarray: 每个点的单位切线向量，形状为(N, 3)
    """
    # 计算一阶导数（速度向量）
    grads = np.gradient(points, axis=0)                    # 沿点序列轴计算梯度
    norms = np.linalg.norm(grads, axis=1, keepdims=True)   # 计算每个向量的模长
    # T = r' / |r'| （单位化切线向量）
    tangents = np.divide(grads, norms, out=np.zeros_like(grads), where=norms!=0)  # 归一化，避免除零错误
    return tangents

def compute_curvature_manually(points):
    """
    手动计算曲线的曲率
    公式: k = |r' x r''| / |r'|^3
    
    参数:
        points (np.ndarray): 曲线上各点的坐标数组，形状为(N, 3)
        
    返回:
        np.ndarray: 每个点的曲率值，形状为(N,)
    """
    # 1. 计算一阶导数 r' (速度向量)
    vel = np.gradient(points, axis=0)
    
    # 2. 计算二阶导数 r'' (加速度向量)
    acc = np.gradient(vel, axis=0)
    
    # 3. 计算叉乘 |r' x r''|
    cross_prod = np.cross(vel, acc)                       # 计算速度和加速度的叉积
    num = np.linalg.norm(cross_prod, axis=1)              # 计算叉积向量的模长
    
    # 4. 计算分母 |r'|^3
    vel_norm = np.linalg.norm(vel, axis=1)                # 计算速度向量的模长
    denom = np.power(vel_norm, 3)                         # 计算模长的三次方
    
    # 5. 计算曲率 k = |r' x r''| / |r'|^3
    curvature = np.divide(num, denom, out=np.zeros_like(num), where=denom!=0)  # 避免除零错误
    
    return curvature

def compute_abscissas_manually(centerline_polydata):
    """
    手动计算沿中心线的路径长度（使用广度优先搜索算法）
    
    参数:
        centerline_polydata (vtkPolyData): 中心线数据
        
    返回:
        np.ndarray: 每个点到入口点的路径长度，形状为(N,)
    """
    # 获取中心线上的点坐标
    points = vtk_to_numpy(centerline_polydata.GetPoints().GetData())
    n_points = points.shape[0]                            # 点的数量
    
    # 构建邻接表表示的图结构
    adj = {i: [] for i in range(n_points)}                # 初始化邻接表
    lines = centerline_polydata.GetLines()                # 获取线段连接关系
    lines.InitTraversal()                                 # 初始化遍历
    id_list = vtk.vtkIdList()                             # 创建ID列表
    
    # 遍历所有线段，构建图的邻接关系
    while lines.GetNextCell(id_list):
        n_ids = id_list.GetNumberOfIds()                  # 当前线段包含的点数
        for i in range(n_ids - 1):
            u = id_list.GetId(i)                          # 起始点ID
            v = id_list.GetId(i+1)                        # 终止点ID
            dist = np.linalg.norm(points[u] - points[v])  # 计算两点间距离
            adj[u].append((v, dist))                      # 添加邻接关系
            adj[v].append((u, dist))                      # 添加反向邻接关系
            
    # 确定入口点（Z坐标最大的点）
    start_node = np.argmax(points[:, 2])                  # 找到入口点索引
    
    # 初始化距离数组
    distances = np.full(n_points, -1.0)                   # 初始化距离数组，-1表示未访问
    distances[start_node] = 0.0                           # 入口点距离设为0
    queue = [start_node]                                  # 初始化BFS队列
    
    # 广度优先搜索计算各点到入口的距离
    while len(queue) > 0:
        u = queue.pop(0)                                  # 取出队首节点
        current_dist = distances[u]                       # 当前节点距离
        for v, edge_len in adj[u]:                        # 遍历相邻节点
            if distances[v] == -1.0:                      # 如果未访问过
                distances[v] = current_dist + edge_len    # 更新距离
                queue.append(v)                           # 加入队列
                
    # 处理无法到达的点（理论上不应该出现）
    distances[distances == -1.0] = 0.0
    return distances

# ==========================================
# 3. 主逻辑处理函数
# ==========================================

def extract_features_robust(surface):
    """
    提取血管表面的各种几何特征
    
    参数:
        surface (vtkPolyData): 血管表面模型数据
        
    返回:
        tuple: (带有特征数据的表面模型, 中心线数据)
    """
    
    # --- 1. 检测血管的入口和出口 ---
    print("1. [Detection] 检测进出口...")
    inlet_pt, outlet_pts_list = detect_all_openings(surface)
    
    # 如果未能成功检测到开口，则使用边界框的顶部和底部作为默认开口
    if inlet_pt is None:
        bounds = surface.GetBounds()                     # 获取表面边界
        # 默认入口为边界框顶部中心
        inlet_pt = [(bounds[0]+bounds[1])/2, (bounds[2]+bounds[3])/2, bounds[5]]
        # 默认出口为边界框底部中心
        outlet_pts_list = [[(bounds[0]+bounds[1])/2, (bounds[2]+bounds[3])/2, bounds[4]]]

    # 将多个出口点展平为一维列表
    target_flat = []
    for pt in outlet_pts_list: target_flat.extend(pt)

    # --- 2. 使用VMTK生成中心线 ---
    print("2. [VMTK] 生成中心线...")
    cl = vmtkscripts.vmtkCenterlines()                   # 创建中心线生成器
    cl.Surface = surface                                 # 设置输入表面
    cl.SeedSelectorName = 'pointlist'                    # 设置种子点选择方式
    cl.SourcePoints = list(inlet_pt)                     # 设置源点（入口）
    cl.TargetPoints = target_flat                        # 设置目标点（出口）
    cl.AppendEndPoints = 0                               # 不追加端点
    cl.Execute()                                         # 执行中心线生成
    
    # --- 3. 清洗中心线数据 ---
    clean_cl = clean_centerline(cl.Centerlines, surface)
    
    # 注意：MaximumInscribedSphereRadius 是在 Centerlines 步骤生成的
    # clean_centerline (vtkClipPolyData) 会保留这个属性
    
    centerline = clean_cl
    
    # --- 4. 手动计算所有几何特征 ---
    print("3. [Calculation] 手动计算几何特征 ...")
    
    # 获取中心线上的坐标点 (N, 3)
    cl_points = vtk_to_numpy(centerline.GetPoints().GetData())
    
    # A. 计算切线向量
    arr_tangent = compute_tangents_manually(cl_points)
    # B. 计算曲率（新增）
    arr_curv = compute_curvature_manually(cl_points)
    # C. 计算路径长度
    arr_abscissa_raw = compute_abscissas_manually(centerline)
    
    # --- 5. 将特征映射到表面 ---
    print("4. [Mapping] 特征投影到表面...")
    
    cl_pd = centerline.GetPointData()                    # 获取中心线点数据
    # 半径是唯一必须从 VMTK 获取的，因为它依赖 Voronoi 图算法
    arr_radius = vtk_to_numpy(cl_pd.GetArray("MaximumInscribedSphereRadius"))
    
    # 创建KD树用于快速查找最近点
    locator = vtk.vtkKdTreePointLocator()
    locator.SetDataSet(centerline)                       # 设置数据集
    locator.BuildLocator()                               # 构建KD树
    
    # 获取表面点坐标
    surf_points = vtk_to_numpy(surface.GetPoints().GetData())
    n_pts = surf_points.shape[0]                         # 表面点数量
    
    # 初始化结果数组
    res_abscissa = np.zeros(n_pts)                       # 路径长度特征
    res_norm_dist = np.zeros(n_pts)                      # 归一化距离特征
    res_curv = np.zeros(n_pts)                           # 曲率特征
    res_tangent = np.zeros((n_pts, 3))                   # 切线向量特征
    
    # 将中心线特征映射到表面每个点
    for i in range(n_pts):
        pt = surf_points[i]                              # 当前表面点
        closest_id = locator.FindClosestPoint(pt)        # 找到中心线上最近点的ID
        
        # 计算点到中心线的距离
        dist = np.linalg.norm(pt - cl_points[closest_id])
        local_r = arr_radius[closest_id]                 # 获取最近点的半径
        
        # 将特征值赋给当前表面点
        res_abscissa[i] = arr_abscissa_raw[closest_id]   # 路径长度
        res_curv[i] = arr_curv[closest_id]               # 曲率（使用手动计算值）
        res_tangent[i] = arr_tangent[closest_id]         # 切线向量
        
        # 计算归一化距离（距离/半径）
        safe_r = local_r if local_r > 1e-6 else 1.0      # 避免除零错误
        res_norm_dist[i] = dist / safe_r                 # 归一化距离

    # --- 6. 后处理特征数据 ---
    # 对路径长度特征进行归一化处理
    ab_min, ab_max = np.nanmin(res_abscissa), np.nanmax(res_abscissa)  # 获取最小最大值
    denom = ab_max - ab_min                                              # 计算差值
    if denom > 1e-8:                                                     # 避免除零错误
        res_abscissa = (res_abscissa - ab_min) / denom                   # 归一化到[0,1]区间
    else:
        res_abscissa[:] = 0.0                                            # 如果差值过小则全置为0
        
    # 清洗NaN值
    res_abscissa = np.nan_to_num(res_abscissa)           # 将NaN替换为0
    res_norm_dist = np.nan_to_num(res_norm_dist)         # 将NaN替换为0
    res_curv = np.nan_to_num(res_curv)                   # 将NaN替换为0
    res_tangent = np.nan_to_num(res_tangent)             # 将NaN替换为0

    # --- 7. 将特征数据写入表面模型 ---
    def add_array(name, data):
        """
        辅助函数：将numpy数组添加到VTK数据中
        
        参数:
            name (str): 数组名称
            data (np.ndarray): 要添加的数据
        """
        arr = numpy_to_vtk(data, deep=1)                 # 将numpy数组转换为VTK数组
        arr.SetName(name)                                # 设置数组名称
        surface.GetPointData().AddArray(arr)             # 添加到表面点数据中
        
    # 添加各种特征数组
    add_array("Feature_Abscissa", res_abscissa)          # 添加路径长度特征
    add_array("Feature_NormRadius", res_norm_dist)       # 添加归一化距离特征
    add_array("Feature_Curvature", res_curv)             # 添加曲率特征
    add_array("Feature_Tangent", res_tangent)            # 添加切线向量特征
    
    # 返回处理后的表面模型和中心线数据
    return surface, centerline

# ==========================================
# 4. 程序入口点
# ==========================================
if __name__ == "__main__":
    """
    程序主入口，执行完整的特征提取流程
    """
    input_stl = "MA+XIAO+DONG-new.stl"                   # 输入STL文件名
    
    # 检查输入文件是否存在
    if os.path.exists(input_stl):
        try:
            print(f"🚀 开始处理: {input_stl}")              # 打印开始处理信息
            surf = read_surface(input_stl)                 # 读取表面模型
            final_surf, final_cl = extract_features_robust(surf)  # 提取特征
            
            # 保存处理结果
            write_vtp(final_surf, "result_surface.vtp")    # 保存带特征的表面模型
            write_vtp(final_cl, "result_centerline.vtp")   # 保存中心线
            
            # 导出CSV格式数据
            pd_surf = final_surf.GetPointData()            # 获取表面点数据
            points = vtk_to_numpy(final_surf.GetPoints().GetData())  # 获取点坐标
            tangents = vtk_to_numpy(pd_surf.GetArray("Feature_Tangent")) # 获取切线数据
            
            # 创建DataFrame并保存为CSV
            df = pd.DataFrame({
                "x": points[:, 0], "y": points[:, 1], "z": points[:, 2],          # 空间坐标
                "Abscissa": vtk_to_numpy(pd_surf.GetArray("Feature_Abscissa")),   # 路径长度
                "NormRadius": vtk_to_numpy(pd_surf.GetArray("Feature_NormRadius")), # 归一化距离
                "Curvature": vtk_to_numpy(pd_surf.GetArray("Feature_Curvature")),  # 曲率
                "Tangent_X": tangents[:, 0],                                      # 切线X分量
                "Tangent_Y": tangents[:, 1],                                      # 切线Y分量
                "Tangent_Z": tangents[:, 2]                                       # 切线Z分量
            })
            df.to_csv("result_features.csv", index=False)                         # 保存为CSV文件
            print("\n✅ 成功! 曲率 (Curvature) 和切线 (Tangent) 均已恢复正常。")   # 打印成功信息
            
        except Exception as e:
            print(f"\n❌ 错误: {e}")                                               # 打印错误信息
            import traceback
            traceback.print_exc()                                                 # 打印详细错误堆栈
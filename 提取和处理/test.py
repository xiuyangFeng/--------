import os
import numpy as np
import pandas as pd
import vtk
from vmtk import vmtkscripts
from vtkmodules.util.numpy_support import vtk_to_numpy, numpy_to_vtk

# ==========================================
# 1. 基础工具
# ==========================================

def read_surface(path):
    """
    读取表面几何模型文件
    参数:
        path: 文件路径，支持STL和VTP格式
    返回:
        polydata: VTK多边形数据对象
    """
    # 根据文件扩展名选择合适的读取器
    if path.endswith(".vtp"):
        reader = vtk.vtkXMLPolyDataReader()  # VTP格式读取器
    else:
        reader = vtk.vtkSTLReader()  # STL格式读取器
    reader.SetFileName(path)  # 设置文件名
    reader.Update()  # 更新读取器
    return reader.GetOutput()  # 返回读取的数据

def write_vtp(polydata, path):
    """
    将polydata数据写入VTP文件
    参数:
        polydata: VTK多边形数据对象
        path: 输出文件路径
    """
    writer = vtk.vtkXMLPolyDataWriter()  # 创建VTP写入器
    writer.SetFileName(path)  # 设置输出文件名
    writer.SetInputData(polydata)  # 设置输入数据
    writer.Write()  # 执行写入操作

# ==========================================
# 2. 核心算法模块 (全部手动计算)
# ==========================================

def detect_all_openings(surface):
    """
    检测表面模型的所有开口区域
    参数:
        surface: VTK表面模型数据
    返回:
        inlet_pt: 入口点坐标列表
        outlet_pts_list: 出口点坐标列表
    """
    print("   [1.1] 扫描网格边界...")
    # 提取边界边
    feature_edges = vtk.vtkFeatureEdges()
    feature_edges.SetInputData(surface)
    feature_edges.BoundaryEdgesOn()  # 开启边界边提取
    feature_edges.FeatureEdgesOff()  # 关闭特征边提取
    feature_edges.ManifoldEdgesOff()  # 关闭流形边提取
    feature_edges.NonManifoldEdgesOff()  # 关闭非流形边提取
    feature_edges.Update()  # 更新执行
    
    # 连通区域分析
    conn = vtk.vtkConnectivityFilter()
    conn.SetInputData(feature_edges.GetOutput())
    conn.SetExtractionModeToAllRegions()  # 提取所有连通区域
    conn.ColorRegionsOn()  # 为不同区域着色
    conn.Update()
    
    # 获取区域数量
    num_regions = conn.GetNumberOfExtractedRegions()
    opening_centers = []  # 存储开口中心点
    valid_count = 0  # 有效开口计数
    
    # 遍历所有连通区域
    for i in range(num_regions):
        # 使用阈值过滤器提取特定区域
        thresh = vtk.vtkThreshold()
        thresh.SetInputData(conn.GetOutput())
        thresh.SetInputArrayToProcess(0, 0, 0, vtk.vtkDataObject.FIELD_ASSOCIATION_CELLS, "RegionId")
        thresh.SetLowerThreshold(i)  # 设置下限阈值
        thresh.SetUpperThreshold(i)  # 设置上限阈值
        thresh.Update()
        
        loop = thresh.GetOutput()
        # 忽略点数过少的区域（小于10个点）
        if loop.GetNumberOfPoints() < 10: 
            continue 
            
        # 计算区域点的平均坐标作为中心点
        points = vtk_to_numpy(loop.GetPoints().GetData())
        opening_centers.append(np.mean(points, axis=0))
        valid_count += 1
        
    # 如果有效开口少于2个，返回None
    if valid_count < 2: 
        return None, None

    # 将开口中心点转换为NumPy数组
    opening_centers = np.array(opening_centers)
    # 将Z坐标最大的开口作为入口
    inlet_idx = np.argmax(opening_centers[:, 2])
    # 返回入口点和其他开口点
    return list(opening_centers[inlet_idx]), np.delete(opening_centers, inlet_idx, axis=0).tolist()

def clean_centerline(centerline, surface):
    """
    清洗中心线数据，去除超出表面边界的中心线部分
    参数:
        centerline: 原始中心线数据
        surface: 表面模型数据
    返回:
        cleaned_centerline: 清洗后的中心线数据
    """
    print("   [2.1] 清洗中心线...")
    bounds = surface.GetBounds()  # 获取表面边界
    buffer = 0.5  # 边界缓冲区大小
    
    # 创建包围盒
    box = vtk.vtkBox()
    box.SetBounds(bounds[0]-buffer, bounds[1]+buffer, 
                  bounds[2]-buffer, bounds[3]+buffer, 
                  bounds[4]-buffer, bounds[5]+buffer)
    
    # 使用包围盒裁剪中心线
    clipper = vtk.vtkClipPolyData()
    clipper.SetInputData(centerline)
    clipper.SetClipFunction(box)
    clipper.InsideOutOn()  # 反转裁剪方向，保留内部
    clipper.Update()
    
    clipped = clipper.GetOutput()
    # 如果裁剪后没有点，返回原始中心线
    if clipped.GetNumberOfPoints() == 0: 
        return centerline

    # 提取最大的连通区域
    conn = vtk.vtkConnectivityFilter()
    conn.SetInputData(clipped)
    conn.SetExtractionModeToLargestRegion()  # 只提取最大区域
    conn.Update()
    
    return conn.GetOutput()

# --- 手动计算几何特征函数 ---

def compute_tangents_manually(points):
    """
    手动计算点列的切线向量
    参数:
        points: 点坐标数组，形状为(N, 3)
    返回:
        tangents: 切线向量数组，形状为(N, 3)
    """
    # r' (velocity) - 计算一阶导数（速度）
    grads = np.gradient(points, axis=0)
    norms = np.linalg.norm(grads, axis=1, keepdims=True)
    # T = r' / |r'| - 归一化得到单位切向量
    tangents = np.divide(grads, norms, out=np.zeros_like(grads), where=norms!=0)
    return tangents

def compute_curvature_manually(points):
    """
    [新增] 手动计算曲率
    公式: k = |r' x r''| / |r'|^3
    参数:
        points: 点坐标数组，形状为(N, 3)
    返回:
        curvature: 曲率数组，形状为(N,)
    """
    # 1. 一阶导数 r' (Velocity)
    vel = np.gradient(points, axis=0)
    
    # 2. 二阶导数 r'' (Acceleration)
    acc = np.gradient(vel, axis=0)
    
    # 3. 叉乘 |r' x r''|
    cross_prod = np.cross(vel, acc)
    num = np.linalg.norm(cross_prod, axis=1)
    
    # 4. 分母 |r'|^3
    vel_norm = np.linalg.norm(vel, axis=1)
    denom = np.power(vel_norm, 3)
    
    # 5. 计算曲率
    curvature = np.divide(num, denom, out=np.zeros_like(num), where=denom!=0)
    
    return curvature

def compute_abscissas_manually(centerline_polydata):
    """
    手动计算路径长度（弧长参数）
    使用广度优先搜索(BFS)算法计算
    参数:
        centerline_polydata: 中心线polydata对象
    返回:
        distances: 各点到起点的路径长度数组
    """
    points = vtk_to_numpy(centerline_polydata.GetPoints().GetData())  # 获取点坐标
    n_points = points.shape[0]  # 点数量
    
    # 构建邻接表表示的图
    adj = {i: [] for i in range(n_points)}
    lines = centerline_polydata.GetLines()  # 获取线段连接关系
    lines.InitTraversal()
    id_list = vtk.vtkIdList()
    
    # 遍历所有线段，构建邻接表
    while lines.GetNextCell(id_list):
        n_ids = id_list.GetNumberOfIds()
        for i in range(n_ids - 1):
            u = id_list.GetId(i)
            v = id_list.GetId(i+1)
            dist = np.linalg.norm(points[u] - points[v])  # 计算两点间距离
            adj[u].append((v, dist))  # 添加邻接关系
            adj[v].append((u, dist))
            
    # 确定起点（Z坐标最大的点作为入口）
    start_node = np.argmax(points[:, 2])
    distances = np.full(n_points, -1.0)  # 初始化距离数组，-1表示未访问
    distances[start_node] = 0.0  # 起点距离为0
    queue = [start_node]  # BFS队列
    
    # 广度优先搜索计算各点到起点的距离
    while len(queue) > 0:
        u = queue.pop(0)  # 取出队首元素
        current_dist = distances[u]  # 当前点的距离
        for v, edge_len in adj[u]:  # 遍历相邻点
            if distances[v] == -1.0:  # 如果未访问过
                distances[v] = current_dist + edge_len  # 更新距离
                queue.append(v)  # 加入队列
                
    distances[distances == -1.0] = 0.0  # 处理未访问的点
    return distances

# ==========================================
# 3. 主逻辑
# ==========================================

def extract_features_robust(surface):
    """
    提取稳健的几何特征
    参数:
        surface: 表面模型数据
    返回:
        surface: 添加了几何特征的表面模型
        centerline: 中心线数据
    """
    # --- 1. 检测进出口 ---
    print("1. [Detection] 检测进出口...")
    inlet_pt, outlet_pts_list = detect_all_openings(surface)
    
    # 如果未检测到足够的开口，使用默认方法确定进出口
    if inlet_pt is None:
        bounds = surface.GetBounds()
        inlet_pt = [(bounds[0]+bounds[1])/2, (bounds[2]+bounds[3])/2, bounds[5]]  # 默认入口
        outlet_pts_list = [[(bounds[0]+bounds[1])/2, (bounds[2]+bounds[3])/2, bounds[4]]]  # 默认出口

    # 展平目标点列表
    target_flat = []
    for pt in outlet_pts_list: 
        target_flat.extend(pt)

    # --- 2. 生成中心线 ---
    print("2. [VMTK] 生成中心线...")
    cl = vmtkscripts.vmtkCenterlines()  # 创建中心线生成器
    cl.Surface = surface  # 设置表面模型
    cl.SeedSelectorName = 'pointlist'  # 使用点列表种子选择器
    cl.SourcePoints = list(inlet_pt)  # 设置源点（入口）
    cl.TargetPoints = target_flat  # 设置目标点（出口）
    cl.AppendEndPoints = 0  # 不追加端点
    cl.Execute()  # 执行中心线生成
    
    # --- 3. 清洗中心线 ---
    clean_cl = clean_centerline(cl.Centerlines, surface)
    
    # 注意：MaximumInscribedSphereRadius 是在 Centerlines 步骤生成的
    # clean_centerline (vtkClipPolyData) 会保留这个属性
    
    centerline = clean_cl
    
    # --- 4. 手动计算所有特征 ---
    print("3. [Calculation] 手动计算几何特征 (摆脱VMTK依赖)...")
    
    # 获取坐标点 (N, 3)
    cl_points = vtk_to_numpy(centerline.GetPoints().GetData())
    
    # A. 切线 - 手动计算切向量
    arr_tangent = compute_tangents_manually(cl_points)
    # B. 曲率 (新增) - 手动计算曲率
    arr_curv = compute_curvature_manually(cl_points)
    # C. 路径长度 - 手动计算弧长参数
    arr_abscissa_raw = compute_abscissas_manually(centerline)
    
    # --- 5. 映射特征到表面 ---
    print("4. [Mapping] 特征投影到表面...")
    
    cl_pd = centerline.GetPointData()
    # 半径是唯一必须从 VMTK 获取的，因为它依赖 Voronoi 图算法
    arr_radius = vtk_to_numpy(cl_pd.GetArray("MaximumInscribedSphereRadius"))
    
    # 创建KDTree定位器用于快速查找最近点
    locator = vtk.vtkKdTreePointLocator()
    locator.SetDataSet(centerline)
    locator.BuildLocator()
    
    # 获取表面点坐标
    surf_points = vtk_to_numpy(surface.GetPoints().GetData())
    n_pts = surf_points.shape[0]  # 表面点数量
    
    # 初始化结果数组
    res_abscissa = np.zeros(n_pts)
    res_norm_dist = np.zeros(n_pts)
    res_curv = np.zeros(n_pts)
    res_tangent = np.zeros((n_pts, 3))
    
    # 将中心线特征映射到表面点
    for i in range(n_pts):
        pt = surf_points[i]  # 当前表面点
        closest_id = locator.FindClosestPoint(pt)  # 查找最近的中心线点
        
        # 计算距离和局部半径
        dist = np.linalg.norm(pt - cl_points[closest_id])
        local_r = arr_radius[closest_id]
        
        # 映射各种特征
        res_abscissa[i] = arr_abscissa_raw[closest_id]
        res_curv[i] = arr_curv[closest_id] # 使用手动计算值
        res_tangent[i] = arr_tangent[closest_id]
        
        # 安全计算归一化距离
        safe_r = local_r if local_r > 1e-6 else 1.0
        res_norm_dist[i] = dist / safe_r

    # --- 6. 后处理特征数据 ---
    # 归一化弧长参数到[0,1]区间
    ab_min, ab_max = np.nanmin(res_abscissa), np.nanmax(res_abscissa)
    denom = ab_max - ab_min
    if denom > 1e-8:
        res_abscissa = (res_abscissa - ab_min) / denom
    else:
        res_abscissa[:] = 0.0
        
    # 清洗NaN值
    res_abscissa = np.nan_to_num(res_abscissa)
    res_norm_dist = np.nan_to_num(res_norm_dist)
    res_curv = np.nan_to_num(res_curv)
    res_tangent = np.nan_to_num(res_tangent)

    # --- 7. 将特征添加到表面模型 ---
    def add_array(name, data):
        """
        将数组数据添加到表面模型的点数据中
        参数:
            name: 数组名称
            data: 数据数组
        """
        arr = numpy_to_vtk(data, deep=1)
        arr.SetName(name)
        surface.GetPointData().AddArray(arr)
        
    # 添加各种几何特征数组
    add_array("Feature_Abscissa", res_abscissa)
    add_array("Feature_NormRadius", res_norm_dist)
    add_array("Feature_Curvature", res_curv)
    add_array("Feature_Tangent", res_tangent)
    
    return surface, centerline

# ==========================================
# 4. 入口函数
# ==========================================
if __name__ == "__main__":
    input_stl = "MA+XIAO+DONG-new.stl"  # 输入STL文件名
    
    # 检查输入文件是否存在
    if os.path.exists(input_stl):
        try:
            print(f"🚀 开始处理: {input_stl}")
            surf = read_surface(input_stl)  # 读取表面模型
            final_surf, final_cl = extract_features_robust(surf)  # 提取特征
            
            # 保存处理后的表面和中心线
            write_vtp(final_surf, "result_surface.vtp")
            write_vtp(final_cl, "result_centerline.vtp")
            
            # 导出CSV格式特征数据
            pd_surf = final_surf.GetPointData()
            points = vtk_to_numpy(final_surf.GetPoints().GetData())
            tangents = vtk_to_numpy(pd_surf.GetArray("Feature_Tangent"))
            
            # 构建DataFrame并保存为CSV
            df = pd.DataFrame({
                "x": points[:, 0], "y": points[:, 1], "z": points[:, 2],
                "Abscissa": vtk_to_numpy(pd_surf.GetArray("Feature_Abscissa")),
                "NormRadius": vtk_to_numpy(pd_surf.GetArray("Feature_NormRadius")),
                "Curvature": vtk_to_numpy(pd_surf.GetArray("Feature_Curvature")),
                "Tangent_X": tangents[:, 0],
                "Tangent_Y": tangents[:, 1],
                "Tangent_Z": tangents[:, 2]
            })
            df.to_csv("result_features.csv", index=False)
            print("\n✅ 成功! 曲率 (Curvature) 和切线 (Tangent) 均已恢复正常。")
            
        except Exception as e:
            print(f"\n❌ 错误: {e}")
            import traceback
            traceback.print_exc()
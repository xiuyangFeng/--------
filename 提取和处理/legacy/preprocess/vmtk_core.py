# vmtk_core.py
import numpy as np
import vtk
from vmtk import vmtkscripts
from vtkmodules.util.numpy_support import vtk_to_numpy, numpy_to_vtk

# ==========================================
# 1. 基础 I/O
# ==========================================

def read_surface(path):
    """
    读取 STL 或 VTP 表面文件
    参数:
        path: 表面文件路径，支持STL或VTP格式
    返回:
        surface: VTK表面数据对象
    """
    # 根据文件扩展名选择合适的读取器
    if path.endswith(".vtp"):
        reader = vtk.vtkXMLPolyDataReader()  # VTP格式读取器
    else:
        reader = vtk.vtkSTLReader()  # STL格式读取器
    reader.SetFileName(path)  # 设置文件路径
    reader.Update()  # 更新读取器
    return reader.GetOutput()  # 返回读取的表面数据

# ==========================================
# 2. 鲁棒性处理算法 (检测与清洗)
# ==========================================

def detect_all_openings(surface):
    """
    使用 vtkFeatureEdges 拓扑分析检测所有开口。
    返回: (入口坐标, [出口坐标列表])
    参数:
        surface: VTK表面数据对象
    返回:
        inlet_pt: 入口点坐标列表
        outlet_pts_list: 出口点坐标列表
    """
    print("[Core] Scanning mesh boundaries (Feature Edges)...")
    
    # 提取边界边，用于检测模型的开口
    feature_edges = vtk.vtkFeatureEdges()
    feature_edges.SetInputData(surface)
    feature_edges.BoundaryEdgesOn()  # 开启边界边提取
    feature_edges.FeatureEdgesOff()  # 关闭特征边提取
    feature_edges.ManifoldEdgesOff()  # 关闭流形边提取
    feature_edges.NonManifoldEdgesOff()  # 关闭非流形边提取
    feature_edges.Update()  # 执行边缘提取
    
    # 连通域分析，识别不同的边界区域
    conn = vtk.vtkConnectivityFilter()
    conn.SetInputData(feature_edges.GetOutput())
    conn.SetExtractionModeToAllRegions()  # 提取所有连通区域
    conn.ColorRegionsOn()  # 为不同区域着色以便区分
    conn.Update()  # 执行连通域分析
    
    num_regions = conn.GetNumberOfExtractedRegions()  # 获取区域数量
    opening_centers = []  # 存储开口中心点
    valid_count = 0  # 有效开口计数
    
    # 遍历每个开口区域，计算其中心点
    for i in range(num_regions):
        # 使用阈值过滤器提取特定区域
        thresh = vtk.vtkThreshold()
        thresh.SetInputData(conn.GetOutput())
        thresh.SetInputArrayToProcess(0, 0, 0, vtk.vtkDataObject.FIELD_ASSOCIATION_CELLS, "RegionId")
        thresh.SetLowerThreshold(i)  # 设置下限阈值
        thresh.SetUpperThreshold(i)  # 设置上限阈值
        thresh.Update()  # 执行阈值过滤
        
        loop = thresh.GetOutput()
        # 过滤噪点 (点数过少的边界，认为是噪声)
        if loop.GetNumberOfPoints() < 10:
            continue
            
        # 计算该区域所有点的平均坐标作为中心点
        points = vtk_to_numpy(loop.GetPoints().GetData())
        opening_centers.append(np.mean(points, axis=0))
        valid_count += 1
        
    # 如果有效开口少于2个，无法确定进出口
    if valid_count < 2:
        print(f"[Core Warning] Found only {valid_count} valid openings.")
        return None, None

    opening_centers = np.array(opening_centers)
    
    # 策略：Z轴最高点为入口 (Inlet)，其余为出口
    inlet_idx = np.argmax(opening_centers[:, 2])
    
    inlet_pt = opening_centers[inlet_idx]
    outlet_pts = np.delete(opening_centers, inlet_idx, axis=0)
    
    return list(inlet_pt), outlet_pts.tolist()

def clean_centerline(centerline, surface):
    """
    清洗中心线：
    1. 物理剪裁 (BoxClip) 去除跑出血管外的线段。
    2. 连通性过滤 (Connectivity) 去除断裂的悬浮碎片。
    参数:
        centerline: 原始中心线数据
        surface: 表面对象，用于确定边界
    返回:
        cleaned_centerline: 清洗后的中心线
    """
    print("[Core] Cleaning centerline artifacts...")
    bounds = surface.GetBounds()  # 获取表面边界
    buffer = 0.5 # 容差 mm，增加一些缓冲空间
    
    # 1. 物理剪裁，创建包围盒用于裁剪
    box = vtk.vtkBox()
    box.SetBounds(bounds[0]-buffer, bounds[1]+buffer, 
                  bounds[2]-buffer, bounds[3]+buffer, 
                  bounds[4]-buffer, bounds[5]+buffer)
    
    # 使用包围盒裁剪中心线
    clipper = vtk.vtkClipPolyData()
    clipper.SetInputData(centerline)
    clipper.SetClipFunction(box)
    clipper.InsideOutOn()  # 反转裁剪，保留内部
    clipper.Update()  # 执行裁剪
    
    clipped = clipper.GetOutput()
    # 如果裁剪后没有点，恢复原始中心线
    if clipped.GetNumberOfPoints() == 0:
        print("[Core Warning] Clipped centerline is empty, reverting.")
        return centerline

    # 2. 保留最大连通域，去除小的碎片
    conn = vtk.vtkConnectivityFilter()
    conn.SetInputData(clipped)
    conn.SetExtractionModeToLargestRegion()  # 只提取最大的连通区域
    conn.Update()  # 执行连通域过滤
    
    return conn.GetOutput()

# ==========================================
# 3. 手动几何计算算法 (Numpy加速)
# ==========================================

def compute_tangents_manually(points):
    """
    计算单位切线向量 (Gradient)
    参数:
        points: 点坐标数组，形状为(N, 3)
    返回:
        tangents: 单位切线向量数组，形状为(N, 3)
    """
    # 计算点序列的一阶导数（梯度）
    grads = np.gradient(points, axis=0)
    norms = np.linalg.norm(grads, axis=1, keepdims=True)
    # 避免除零，计算单位切向量
    tangents = np.divide(grads, norms, out=np.zeros_like(grads), where=norms!=0)
    return tangents

def compute_curvature_manually(points):
    """
    计算曲率 k = |r' x r''| / |r'|^3
    参数:
        points: 点坐标数组，形状为(N, 3)
    返回:
        curvature: 曲率数组，形状为(N,)
    """
    # 一阶导 (速度)，描述曲线的变化率
    vel = np.gradient(points, axis=0)
    # 二阶导 (加速度)，描述速度的变化率
    acc = np.gradient(vel, axis=0)
    
    # 叉乘计算曲率分子部分
    cross_prod = np.cross(vel, acc)
    num = np.linalg.norm(cross_prod, axis=1)
    
    # 分母计算
    vel_norm = np.linalg.norm(vel, axis=1)
    denom = np.power(vel_norm, 3)
    
    # 计算最终曲率值，避免除零
    curvature = np.divide(num, denom, out=np.zeros_like(num), where=denom!=0)
    return curvature

def compute_abscissas_manually(centerline_polydata):
    """
    使用 BFS 图遍历算法计算沿程距离 (Abscissas)
    参数:
        centerline_polydata: 中心线polydata对象
    返回:
        distances: 各点到起点的路径距离数组
    """
    # 获取中心线点坐标
    points = vtk_to_numpy(centerline_polydata.GetPoints().GetData())
    n_points = points.shape[0]  # 点的数量
    
    # 构建邻接表表示点之间的连接关系
    adj = {i: [] for i in range(n_points)}
    lines = centerline_polydata.GetLines()  # 获取线段信息
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
            
    # 从 Z 最高点 (入口) 开始遍历计算距离
    start_node = np.argmax(points[:, 2])
    distances = np.full(n_points, -1.0)  # 初始化距离数组，-1表示未访问
    distances[start_node] = 0.0  # 起点距离为0
    queue = [start_node]  # BFS队列
    
    # 广度优先搜索计算各点到起点的距离
    while len(queue) > 0:
        u = queue.pop(0)  # 取出队首节点
        current_dist = distances[u]  # 当前节点距离
        for v, edge_len in adj[u]:  # 遍历相邻节点
            if distances[v] == -1.0:  # 如果未访问过
                distances[v] = current_dist + edge_len  # 更新距离
                queue.append(v)  # 加入队列
    
    # 未连通点距离归零
    distances[distances == -1.0] = 0.0
    return distances

# ==========================================
# 4. 核心接口函数
# ==========================================

def extract_rich_centerline(surface):
    """
    [主流程] 输入血管表面，输出包含全套几何特征的中心线 PolyData。
    所有特征 (Abscissa, Curvature, Tangent) 均为手动计算，确保数值稳定。
    Radius 来自 VMTK。
    参数:
        surface: 血管表面模型数据
    返回:
        centerline: 包含丰富几何特征的中心线数据
    """
    # 1. 检测进出口点
    inlet_pt, outlet_pts_list = detect_all_openings(surface)
    
    # Fallback 机制，如果智能检测失败则使用边界框方法
    if inlet_pt is None:
        print("[Core Warning] Intelligent detection failed, using BoundingBox fallback.")
        bounds = surface.GetBounds()
        inlet_pt = [(bounds[0]+bounds[1])/2, (bounds[2]+bounds[3])/2, bounds[5]]  # 上表面中心作为入口
        outlet_pts_list = [[(bounds[0]+bounds[1])/2, (bounds[2]+bounds[3])/2, bounds[4]]]  # 下表面中心作为出口

    # 展平目标点列表
    target_flat = []
    for pt in outlet_pts_list: 
        target_flat.extend(pt)

    # 2. 使用VMTK提取中心线拓扑结构
    print("[Core] Extracting centerline topology...")
    cl = vmtkscripts.vmtkCenterlines()  # 创建中心线生成器
    cl.Surface = surface  # 设置表面模型
    cl.SeedSelectorName = 'pointlist'  # 使用点列表种子选择器
    cl.SourcePoints = list(inlet_pt)  # 设置源点（入口）
    cl.TargetPoints = target_flat  # 设置目标点（出口）
    cl.AppendEndPoints = 0 # 关键：防止强制延伸错误
    cl.Execute()  # 执行中心线生成
    
    # 3. 清洗中心线数据
    clean_cl = clean_centerline(cl.Centerlines, surface)
    
    # 4. 使用VMTK计算几何特征（主要用于获取最大内切球半径）
    # 虽然我们会覆盖曲率等特征，但半径必须依赖 Voronoi 图，这步不能省
    print("[Core] Computing VMTK geometry (for Radius)...")
    geom = vmtkscripts.vmtkCenterlineGeometry()  # 创建几何计算器
    geom.Centerlines = clean_cl  # 设置中心线数据
    geom.LineSmoothing = 1  # 开启线平滑
    geom.SmoothingFactor = 0.1  # 平滑因子
    geom.NumberOfSmoothingIterations = 10  # 平滑迭代次数
    geom.Execute()  # 执行几何计算
    
    centerline = geom.Centerlines
    cl_points = vtk_to_numpy(centerline.GetPoints().GetData())  # 获取中心线点坐标
    
    # 5. 手动计算高精度特征
    print("[Core] Computing robust features manually...")
    
    # A. 计算各种几何特征
    arr_tangent = compute_tangents_manually(cl_points)  # 计算切线
    arr_curv = compute_curvature_manually(cl_points)  # 计算曲率
    arr_abscissa = compute_abscissas_manually(centerline)  # 计算弧长参数
    
    # B. 注入回 PolyData (覆盖原有数组或新建)
    pd = centerline.GetPointData()  # 获取点数据对象
    
    def add_array(name, data):
        """
        将数组添加到PolyData中
        参数:
            name: 数组名称
            data: 要添加的数据
        """
        arr = numpy_to_vtk(data, deep=1)  # 转换为VTK数组
        arr.SetName(name)  # 设置数组名称
        pd.AddArray(arr) # 如果存在同名数组，通常会覆盖或添加成新的
        
    # VMTK 生成的半径通常叫 "MaximumInscribedSphereRadius"
    # 我们把手动算的特征也加进去，保持命名统一方便后续调用
    add_array("Abscissas", arr_abscissa)        # 覆盖 VMTK 的 Abscissas
    add_array("Curvature", arr_curv)            # 覆盖 VMTK 的 Curvature
    add_array("FrenetTangent", arr_tangent)     # 覆盖 VMTK 的 FrenetTangent
    
    print("[Core] Centerline processing complete.")
    return centerline
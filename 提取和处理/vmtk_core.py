# vmtk_core.py
import numpy as np
import vtk
from vmtk import vmtkscripts
from vtkmodules.util.numpy_support import vtk_to_numpy, numpy_to_vtk

# ==========================================
# 1. 基础 I/O
# ==========================================

def read_surface(path):
    """读取 STL 或 VTP 表面"""
    if path.endswith(".vtp"):
        reader = vtk.vtkXMLPolyDataReader()
    else:
        reader = vtk.vtkSTLReader()
    reader.SetFileName(path)
    reader.Update()
    return reader.GetOutput()

# ==========================================
# 2. 鲁棒性处理算法 (检测与清洗)
# ==========================================

def detect_all_openings(surface):
    """
    使用 vtkFeatureEdges 拓扑分析检测所有开口。
    返回: (入口坐标, [出口坐标列表])
    """
    print("[Core] Scanning mesh boundaries (Feature Edges)...")
    
    # 提取边界边
    feature_edges = vtk.vtkFeatureEdges()
    feature_edges.SetInputData(surface)
    feature_edges.BoundaryEdgesOn()
    feature_edges.FeatureEdgesOff()
    feature_edges.ManifoldEdgesOff()
    feature_edges.NonManifoldEdgesOff()
    feature_edges.Update()
    
    # 连通域分析
    conn = vtk.vtkConnectivityFilter()
    conn.SetInputData(feature_edges.GetOutput())
    conn.SetExtractionModeToAllRegions()
    conn.ColorRegionsOn()
    conn.Update()
    
    num_regions = conn.GetNumberOfExtractedRegions()
    opening_centers = []
    valid_count = 0
    
    # 遍历每个开口区域
    for i in range(num_regions):
        thresh = vtk.vtkThreshold()
        thresh.SetInputData(conn.GetOutput())
        thresh.SetInputArrayToProcess(0, 0, 0, vtk.vtkDataObject.FIELD_ASSOCIATION_CELLS, "RegionId")
        thresh.SetLowerThreshold(i)
        thresh.SetUpperThreshold(i)
        thresh.Update()
        
        loop = thresh.GetOutput()
        # 过滤噪点 (点数过少的边界)
        if loop.GetNumberOfPoints() < 10:
            continue
            
        points = vtk_to_numpy(loop.GetPoints().GetData())
        opening_centers.append(np.mean(points, axis=0))
        valid_count += 1
        
    if valid_count < 2:
        print(f"[Core Warning] Found only {valid_count} valid openings.")
        return None, None

    opening_centers = np.array(opening_centers)
    
    # 策略：Z轴最高点为入口 (Inlet)
    inlet_idx = np.argmax(opening_centers[:, 2])
    
    inlet_pt = opening_centers[inlet_idx]
    outlet_pts = np.delete(opening_centers, inlet_idx, axis=0)
    
    return list(inlet_pt), outlet_pts.tolist()

def clean_centerline(centerline, surface):
    """
    清洗中心线：
    1. 物理剪裁 (BoxClip) 去除跑出血管外的线段。
    2. 连通性过滤 (Connectivity) 去除断裂的悬浮碎片。
    """
    print("[Core] Cleaning centerline artifacts...")
    bounds = surface.GetBounds()
    buffer = 0.5 # 容差 mm
    
    # 1. 物理剪裁
    box = vtk.vtkBox()
    box.SetBounds(bounds[0]-buffer, bounds[1]+buffer, 
                  bounds[2]-buffer, bounds[3]+buffer, 
                  bounds[4]-buffer, bounds[5]+buffer)
    
    clipper = vtk.vtkClipPolyData()
    clipper.SetInputData(centerline)
    clipper.SetClipFunction(box)
    clipper.InsideOutOn()
    clipper.Update()
    
    clipped = clipper.GetOutput()
    if clipped.GetNumberOfPoints() == 0:
        print("[Core Warning] Clipped centerline is empty, reverting.")
        return centerline

    # 2. 保留最大连通域
    conn = vtk.vtkConnectivityFilter()
    conn.SetInputData(clipped)
    conn.SetExtractionModeToLargestRegion()
    conn.Update()
    
    return conn.GetOutput()

# ==========================================
# 3. 手动几何计算算法 (Numpy加速)
# ==========================================

def compute_tangents_manually(points):
    """计算单位切线向量 (Gradient)"""
    grads = np.gradient(points, axis=0)
    norms = np.linalg.norm(grads, axis=1, keepdims=True)
    # 避免除零
    tangents = np.divide(grads, norms, out=np.zeros_like(grads), where=norms!=0)
    return tangents

def compute_curvature_manually(points):
    """计算曲率 k = |r' x r''| / |r'|^3"""
    # 一阶导 (速度)
    vel = np.gradient(points, axis=0)
    # 二阶导 (加速度)
    acc = np.gradient(vel, axis=0)
    
    # 叉乘
    cross_prod = np.cross(vel, acc)
    num = np.linalg.norm(cross_prod, axis=1)
    
    # 分母
    vel_norm = np.linalg.norm(vel, axis=1)
    denom = np.power(vel_norm, 3)
    
    curvature = np.divide(num, denom, out=np.zeros_like(num), where=denom!=0)
    return curvature

def compute_abscissas_manually(centerline_polydata):
    """使用 BFS 图遍历算法计算沿程距离 (Abscissas)"""
    points = vtk_to_numpy(centerline_polydata.GetPoints().GetData())
    n_points = points.shape[0]
    
    # 构建邻接表
    adj = {i: [] for i in range(n_points)}
    lines = centerline_polydata.GetLines()
    lines.InitTraversal()
    id_list = vtk.vtkIdList()
    
    while lines.GetNextCell(id_list):
        n_ids = id_list.GetNumberOfIds()
        for i in range(n_ids - 1):
            u = id_list.GetId(i)
            v = id_list.GetId(i+1)
            dist = np.linalg.norm(points[u] - points[v])
            adj[u].append((v, dist))
            adj[v].append((u, dist))
            
    # 从 Z 最高点 (入口) 开始遍历
    start_node = np.argmax(points[:, 2])
    distances = np.full(n_points, -1.0)
    distances[start_node] = 0.0
    queue = [start_node]
    
    while len(queue) > 0:
        u = queue.pop(0)
        current_dist = distances[u]
        for v, edge_len in adj[u]:
            if distances[v] == -1.0:
                distances[v] = current_dist + edge_len
                queue.append(v)
    
    # 未连通点归零
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
    """
    # 1. 检测
    inlet_pt, outlet_pts_list = detect_all_openings(surface)
    
    # Fallback 机制
    if inlet_pt is None:
        print("[Core Warning] Intelligent detection failed, using BoundingBox fallback.")
        bounds = surface.GetBounds()
        inlet_pt = [(bounds[0]+bounds[1])/2, (bounds[2]+bounds[3])/2, bounds[5]]
        outlet_pts_list = [[(bounds[0]+bounds[1])/2, (bounds[2]+bounds[3])/2, bounds[4]]]

    target_flat = []
    for pt in outlet_pts_list: target_flat.extend(pt)

    # 2. VMTK 提取拓扑
    print("[Core] Extracting centerline topology...")
    cl = vmtkscripts.vmtkCenterlines()
    cl.Surface = surface
    cl.SeedSelectorName = 'pointlist'
    cl.SourcePoints = list(inlet_pt)
    cl.TargetPoints = target_flat
    cl.AppendEndPoints = 0 # 关键：防止强制延伸错误
    cl.Execute()
    
    # 3. 清洗
    clean_cl = clean_centerline(cl.Centerlines, surface)
    
    # 4. VMTK 计算几何 (主要为了获取 MaximumInscribedSphereRadius)
    # 虽然我们会覆盖曲率，但半径必须依赖 Voronoi 图，这步不能省
    print("[Core] Computing VMTK geometry (for Radius)...")
    geom = vmtkscripts.vmtkCenterlineGeometry()
    geom.Centerlines = clean_cl
    geom.LineSmoothing = 1
    geom.SmoothingFactor = 0.1
    geom.NumberOfSmoothingIterations = 10
    geom.Execute()
    
    centerline = geom.Centerlines
    cl_points = vtk_to_numpy(centerline.GetPoints().GetData())
    
    # 5. 手动计算高精度特征
    print("[Core] Computing robust features manually...")
    
    # A. 计算
    arr_tangent = compute_tangents_manually(cl_points)
    arr_curv = compute_curvature_manually(cl_points)
    arr_abscissa = compute_abscissas_manually(centerline)
    
    # B. 注入回 PolyData (覆盖原有数组或新建)
    pd = centerline.GetPointData()
    
    def add_array(name, data):
        arr = numpy_to_vtk(data, deep=1)
        arr.SetName(name)
        pd.AddArray(arr) # 如果存在同名数组，通常会覆盖或添加成新的
        
    # VMTK 生成的半径通常叫 "MaximumInscribedSphereRadius"
    # 我们把手动算的特征也加进去，保持命名统一方便后续调用
    add_array("Abscissas", arr_abscissa)        # 覆盖 VMTK 的 Abscissas
    add_array("Curvature", arr_curv)            # 覆盖 VMTK 的 Curvature
    add_array("FrenetTangent", arr_tangent)     # 覆盖 VMTK 的 FrenetTangent
    
    print("[Core] Centerline processing complete.")
    return centerline
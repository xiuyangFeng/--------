# vmtk_core.py
"""
Pipeline 主线使用的 VMTK 几何核心模块。

该模块从 legacy 归档目录中提升为正式主线依赖，负责：
- 读取 STL / VTP 表面
- 检测血管开口
- 提取并清洗中心线
- 计算沿程距离、曲率、切线等几何特征
"""

import numpy as np
import vtk
from vmtk import vmtkscripts
from vtkmodules.util.numpy_support import vtk_to_numpy, numpy_to_vtk


def read_surface(path):
    """
    读取 STL 或 VTP 表面文件
    参数:
        path: 表面文件路径，支持STL或VTP格式
    返回:
        surface: VTK表面数据对象
    """
    if path.endswith(".vtp"):
        reader = vtk.vtkXMLPolyDataReader()
    else:
        reader = vtk.vtkSTLReader()
    reader.SetFileName(path)
    reader.Update()
    return reader.GetOutput()


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

    feature_edges = vtk.vtkFeatureEdges()
    feature_edges.SetInputData(surface)
    feature_edges.BoundaryEdgesOn()
    feature_edges.FeatureEdgesOff()
    feature_edges.ManifoldEdgesOff()
    feature_edges.NonManifoldEdgesOff()
    feature_edges.Update()

    conn = vtk.vtkConnectivityFilter()
    conn.SetInputData(feature_edges.GetOutput())
    conn.SetExtractionModeToAllRegions()
    conn.ColorRegionsOn()
    conn.Update()

    num_regions = conn.GetNumberOfExtractedRegions()
    opening_centers = []
    valid_count = 0

    for i in range(num_regions):
        thresh = vtk.vtkThreshold()
        thresh.SetInputData(conn.GetOutput())
        thresh.SetInputArrayToProcess(
            0, 0, 0, vtk.vtkDataObject.FIELD_ASSOCIATION_CELLS, "RegionId"
        )
        thresh.SetLowerThreshold(i)
        thresh.SetUpperThreshold(i)
        thresh.Update()

        loop = thresh.GetOutput()
        if loop.GetNumberOfPoints() < 10:
            continue

        points = vtk_to_numpy(loop.GetPoints().GetData())
        opening_centers.append(np.mean(points, axis=0))
        valid_count += 1

    if valid_count < 2:
        print(f"[Core Warning] Found only {valid_count} valid openings.")
        return None, None

    opening_centers = np.array(opening_centers)
    inlet_idx = np.argmax(opening_centers[:, 2])

    inlet_pt = opening_centers[inlet_idx]
    outlet_pts = np.delete(opening_centers, inlet_idx, axis=0)
    return list(inlet_pt), outlet_pts.tolist()


def clean_centerline(centerline, surface):
    """
    清洗中心线：
    1. 物理剪裁去除跑出血管外的线段。
    2. 连通性过滤去除断裂的悬浮碎片。
    """
    print("[Core] Cleaning centerline artifacts...")
    bounds = surface.GetBounds()
    buffer = 0.5

    box = vtk.vtkBox()
    box.SetBounds(
        bounds[0] - buffer,
        bounds[1] + buffer,
        bounds[2] - buffer,
        bounds[3] + buffer,
        bounds[4] - buffer,
        bounds[5] + buffer,
    )

    clipper = vtk.vtkClipPolyData()
    clipper.SetInputData(centerline)
    clipper.SetClipFunction(box)
    clipper.InsideOutOn()
    clipper.Update()

    clipped = clipper.GetOutput()
    if clipped.GetNumberOfPoints() == 0:
        print("[Core Warning] Clipped centerline is empty, reverting.")
        return centerline

    conn = vtk.vtkConnectivityFilter()
    conn.SetInputData(clipped)
    conn.SetExtractionModeToLargestRegion()
    conn.Update()
    return conn.GetOutput()


def compute_tangents_manually(points):
    """计算单位切线向量。"""
    grads = np.gradient(points, axis=0)
    norms = np.linalg.norm(grads, axis=1, keepdims=True)
    tangents = np.divide(grads, norms, out=np.zeros_like(grads), where=norms != 0)
    return tangents


def compute_curvature_manually(points):
    """计算曲率 k = |r' x r''| / |r'|^3。"""
    vel = np.gradient(points, axis=0)
    acc = np.gradient(vel, axis=0)

    cross_prod = np.cross(vel, acc)
    num = np.linalg.norm(cross_prod, axis=1)

    vel_norm = np.linalg.norm(vel, axis=1)
    denom = np.power(vel_norm, 3)

    curvature = np.divide(num, denom, out=np.zeros_like(num), where=denom != 0)
    return curvature


def compute_abscissas_manually(centerline_polydata):
    """使用 BFS 图遍历算法计算沿程距离。"""
    points = vtk_to_numpy(centerline_polydata.GetPoints().GetData())
    n_points = points.shape[0]

    adj = {i: [] for i in range(n_points)}
    lines = centerline_polydata.GetLines()
    lines.InitTraversal()
    id_list = vtk.vtkIdList()

    while lines.GetNextCell(id_list):
        n_ids = id_list.GetNumberOfIds()
        for i in range(n_ids - 1):
            u = id_list.GetId(i)
            v = id_list.GetId(i + 1)
            dist = np.linalg.norm(points[u] - points[v])
            adj[u].append((v, dist))
            adj[v].append((u, dist))

    start_node = np.argmax(points[:, 2])
    distances = np.full(n_points, -1.0)
    distances[start_node] = 0.0
    queue = [start_node]

    while queue:
        u = queue.pop(0)
        current_dist = distances[u]
        for v, edge_len in adj[u]:
            if distances[v] == -1.0:
                distances[v] = current_dist + edge_len
                queue.append(v)

    distances[distances == -1.0] = 0.0
    return distances


def compute_torsion_manually(points):
    """计算 Frenet 扭率 τ = (r' × r'') · r''' / |r' × r''|²。

    扭率描述中心线偏离其密切平面的速率，补充曲率无法表达的三维弯扭信息。
    """
    vel = np.gradient(points, axis=0)
    acc = np.gradient(vel, axis=0)
    jerk = np.gradient(acc, axis=0)

    cross_va = np.cross(vel, acc)
    cross_sq = np.sum(cross_va ** 2, axis=1)

    dot = np.sum(cross_va * jerk, axis=1)
    torsion = np.divide(dot, cross_sq, out=np.zeros_like(dot), where=cross_sq > 1e-20)
    return torsion


def compute_tangent_change_rate(tangent_arr, abscissa_arr):
    """计算切向变化率 |dT/ds|。

    比单点 Tangent 更强调局部流向转折的强度。对于直线段该值为 0，
    急转弯处该值很大。本质上是曲率的另一种度量视角。
    """
    n = len(tangent_arr)
    dtds_mag = np.zeros(n)
    if n < 2:
        return dtds_mag
    for i in range(n):
        if i == 0:
            ds = abscissa_arr[1] - abscissa_arr[0]
            dt = tangent_arr[1] - tangent_arr[0]
        elif i == n - 1:
            ds = abscissa_arr[n - 1] - abscissa_arr[n - 2]
            dt = tangent_arr[n - 1] - tangent_arr[n - 2]
        else:
            ds = abscissa_arr[i + 1] - abscissa_arr[i - 1]
            dt = tangent_arr[i + 1] - tangent_arr[i - 1]
        dtds_mag[i] = np.linalg.norm(dt) / abs(ds) if abs(ds) > 1e-10 else 0.0
    return dtds_mag


def compute_radius_change_rate(radius_arr, abscissa_arr):
    """计算半径沿弧长的变化率 dR/ds。

    使用中心差分，端点用前向/后向差分。结果表示管腔的局部扩张（正）或收缩（负）。
    """
    n = len(radius_arr)
    dR_ds = np.zeros(n)
    if n < 2:
        return dR_ds
    for i in range(n):
        if i == 0:
            ds = abscissa_arr[1] - abscissa_arr[0]
            dR_ds[i] = (radius_arr[1] - radius_arr[0]) / ds if abs(ds) > 1e-10 else 0.0
        elif i == n - 1:
            ds = abscissa_arr[n - 1] - abscissa_arr[n - 2]
            dR_ds[i] = (radius_arr[n - 1] - radius_arr[n - 2]) / ds if abs(ds) > 1e-10 else 0.0
        else:
            ds = abscissa_arr[i + 1] - abscissa_arr[i - 1]
            dR_ds[i] = (radius_arr[i + 1] - radius_arr[i - 1]) / ds if abs(ds) > 1e-10 else 0.0
    return dR_ds


def compute_bifurcation_features(centerline_polydata, abscissa_arr):
    """计算距分叉点距离和分叉拓扑标记。

    分叉点定义为中心线上度 >= 3 的节点（即连接 3 条以上线段的交汇点）。
    dist_to_bifurcation: 沿弧长到最近分叉点的距离。
    branch_id: 0 = 主干（含分叉点自身及其上游），1 = 分支。
    """
    points = vtk_to_numpy(centerline_polydata.GetPoints().GetData())
    n_points = points.shape[0]

    adj = {i: [] for i in range(n_points)}
    lines = centerline_polydata.GetLines()
    lines.InitTraversal()
    id_list = vtk.vtkIdList()

    while lines.GetNextCell(id_list):
        n_ids = id_list.GetNumberOfIds()
        for i in range(n_ids - 1):
            u = id_list.GetId(i)
            v = id_list.GetId(i + 1)
            edge_len = abs(abscissa_arr[v] - abscissa_arr[u])
            adj[u].append((v, edge_len))
            adj[v].append((u, edge_len))

    degree = {i: len({v for v, _ in adj[i]}) for i in range(n_points)}
    bifurcation_ids = [i for i in range(n_points) if degree[i] >= 3]

    dist_to_bif = np.full(n_points, np.inf)
    branch_id = np.ones(n_points, dtype=np.float64)

    if not bifurcation_ids:
        dist_to_bif[:] = 0.0
        branch_id[:] = 0.0
        return dist_to_bif, branch_id

    for bif_id in bifurcation_ids:
        dist_to_bif[bif_id] = 0.0

    import heapq
    heap = [(0.0, bif_id) for bif_id in bifurcation_ids]
    heapq.heapify(heap)

    while heap:
        current_dist, u = heapq.heappop(heap)
        if current_dist > dist_to_bif[u]:
            continue
        for v, edge_len in adj[u]:
            candidate = current_dist + edge_len
            if candidate < dist_to_bif[v]:
                dist_to_bif[v] = candidate
                heapq.heappush(heap, (candidate, v))

    dist_to_bif[np.isinf(dist_to_bif)] = 0.0

    from collections import deque
    start_node = np.argmax(points[:, 2])
    trunk_set = set()
    trunk_queue = deque([start_node])
    trunk_set.add(start_node)
    while trunk_queue:
        u = trunk_queue.popleft()
        for v, _ in adj[u]:
            if v not in trunk_set:
                if degree[v] >= 3 or degree[u] < 3:
                    trunk_set.add(v)
                    trunk_queue.append(v)

    for i in trunk_set:
        branch_id[i] = 0.0

    return dist_to_bif, branch_id


def extract_rich_centerline(surface):
    """
    输入血管表面，输出包含全套几何特征的中心线 PolyData。
    所有特征 (Abscissa, Curvature, Tangent) 均为手动计算，确保数值稳定。
    Radius 来自 VMTK。新增 G01/G02 特征。
    """
    inlet_pt, outlet_pts_list = detect_all_openings(surface)

    if inlet_pt is None:
        print("[Core Warning] Intelligent detection failed, using BoundingBox fallback.")
        bounds = surface.GetBounds()
        inlet_pt = [(bounds[0] + bounds[1]) / 2, (bounds[2] + bounds[3]) / 2, bounds[5]]
        outlet_pts_list = [[(bounds[0] + bounds[1]) / 2, (bounds[2] + bounds[3]) / 2, bounds[4]]]

    target_flat = []
    for pt in outlet_pts_list:
        target_flat.extend(pt)

    print("[Core] Extracting centerline topology...")
    cl = vmtkscripts.vmtkCenterlines()
    cl.Surface = surface
    cl.SeedSelectorName = "pointlist"
    cl.SourcePoints = list(inlet_pt)
    cl.TargetPoints = target_flat
    cl.AppendEndPoints = 0
    cl.Execute()

    clean_cl = clean_centerline(cl.Centerlines, surface)

    print("[Core] Computing VMTK geometry (for Radius)...")
    geom = vmtkscripts.vmtkCenterlineGeometry()
    geom.Centerlines = clean_cl
    geom.LineSmoothing = 1
    geom.SmoothingFactor = 0.1
    geom.NumberOfSmoothingIterations = 10
    geom.Execute()

    centerline = geom.Centerlines
    cl_points = vtk_to_numpy(centerline.GetPoints().GetData())

    print("[Core] Computing robust features manually...")
    arr_tangent = compute_tangents_manually(cl_points)
    arr_curv = compute_curvature_manually(cl_points)
    arr_abscissa = compute_abscissas_manually(centerline)

    pd = centerline.GetPointData()

    def add_array(name, data):
        arr = numpy_to_vtk(data, deep=1)
        arr.SetName(name)
        pd.AddArray(arr)

    add_array("Abscissas", arr_abscissa)
    add_array("Curvature", arr_curv)
    add_array("FrenetTangent", arr_tangent)

    print("[Core] Computing G01 bifurcation features...")
    arr_dist_to_bif, arr_branch_id = compute_bifurcation_features(centerline, arr_abscissa)
    add_array("DistToBifurcation", arr_dist_to_bif)
    add_array("BranchId", arr_branch_id)

    print("[Core] Computing G02 radius change rate...")
    radius_arr_vtk = pd.GetArray("MaximumInscribedSphereRadius")
    if radius_arr_vtk is not None:
        arr_radius = vtk_to_numpy(radius_arr_vtk)
        arr_dR_ds = compute_radius_change_rate(arr_radius, arr_abscissa)
    else:
        arr_dR_ds = np.zeros(cl_points.shape[0])
        print("[Core Warning] MaximumInscribedSphereRadius not found, dR_ds set to 0.")
    add_array("dR_ds", arr_dR_ds)

    print("[Core] Computing G03 torsion...")
    arr_torsion = compute_torsion_manually(cl_points)
    add_array("Torsion", arr_torsion)

    print("[Core] Computing G05 tangent change rate...")
    arr_dtds = compute_tangent_change_rate(arr_tangent, arr_abscissa)
    add_array("TangentChangeRate", arr_dtds)

    print("[Core] Centerline processing complete.")
    return centerline

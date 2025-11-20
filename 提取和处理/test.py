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
    if path.endswith(".vtp"):
        reader = vtk.vtkXMLPolyDataReader()
    else:
        reader = vtk.vtkSTLReader()
    reader.SetFileName(path)
    reader.Update()
    return reader.GetOutput()

def write_vtp(polydata, path):
    writer = vtk.vtkXMLPolyDataWriter()
    writer.SetFileName(path)
    writer.SetInputData(polydata)
    writer.Write()

# ==========================================
# 2. 核心算法模块 (全部手动计算)
# ==========================================

def detect_all_openings(surface):
    """检测所有开口"""
    print("   [1.1] 扫描网格边界...")
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
        thresh.SetInputArrayToProcess(0, 0, 0, vtk.vtkDataObject.FIELD_ASSOCIATION_CELLS, "RegionId")
        thresh.SetLowerThreshold(i)
        thresh.SetUpperThreshold(i)
        thresh.Update()
        
        loop = thresh.GetOutput()
        if loop.GetNumberOfPoints() < 10: continue 
            
        points = vtk_to_numpy(loop.GetPoints().GetData())
        opening_centers.append(np.mean(points, axis=0))
        valid_count += 1
        
    if valid_count < 2: return None, None

    opening_centers = np.array(opening_centers)
    inlet_idx = np.argmax(opening_centers[:, 2])
    return list(opening_centers[inlet_idx]), np.delete(opening_centers, inlet_idx, axis=0).tolist()

def clean_centerline(centerline, surface):
    """清洗中心线"""
    print("   [2.1] 清洗中心线...")
    bounds = surface.GetBounds()
    buffer = 0.5
    
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
    if clipped.GetNumberOfPoints() == 0: return centerline

    conn = vtk.vtkConnectivityFilter()
    conn.SetInputData(clipped)
    conn.SetExtractionModeToLargestRegion()
    conn.Update()
    
    return conn.GetOutput()

# --- 手动计算几何特征函数 ---

def compute_tangents_manually(points):
    """计算切线 (一阶导数归一化)"""
    # r' (velocity)
    grads = np.gradient(points, axis=0)
    norms = np.linalg.norm(grads, axis=1, keepdims=True)
    # T = r' / |r'|
    tangents = np.divide(grads, norms, out=np.zeros_like(grads), where=norms!=0)
    return tangents

def compute_curvature_manually(points):
    """
    [新增] 手动计算曲率
    公式: k = |r' x r''| / |r'|^3
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
    """手动计算路径长度 (BFS)"""
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
            v = id_list.GetId(i+1)
            dist = np.linalg.norm(points[u] - points[v])
            adj[u].append((v, dist))
            adj[v].append((u, dist))
            
    start_node = np.argmax(points[:, 2]) # Inlet
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
                
    distances[distances == -1.0] = 0.0
    return distances

# ==========================================
# 3. 主逻辑
# ==========================================

def extract_features_robust(surface):
    # --- 1. 检测 ---
    print("1. [Detection] 检测进出口...")
    inlet_pt, outlet_pts_list = detect_all_openings(surface)
    
    if inlet_pt is None:
        bounds = surface.GetBounds()
        inlet_pt = [(bounds[0]+bounds[1])/2, (bounds[2]+bounds[3])/2, bounds[5]]
        outlet_pts_list = [[(bounds[0]+bounds[1])/2, (bounds[2]+bounds[3])/2, bounds[4]]]

    target_flat = []
    for pt in outlet_pts_list: target_flat.extend(pt)

    # --- 2. 生成 ---
    print("2. [VMTK] 生成中心线...")
    cl = vmtkscripts.vmtkCenterlines()
    cl.Surface = surface
    cl.SeedSelectorName = 'pointlist'
    cl.SourcePoints = list(inlet_pt)
    cl.TargetPoints = target_flat
    cl.AppendEndPoints = 0 
    cl.Execute()
    
    # --- 3. 清洗 ---
    clean_cl = clean_centerline(cl.Centerlines, surface)
    
    # 注意：MaximumInscribedSphereRadius 是在 Centerlines 步骤生成的
    # clean_centerline (vtkClipPolyData) 会保留这个属性
    
    centerline = clean_cl
    
    # --- 4. 手动计算所有特征 ---
    print("3. [Calculation] 手动计算几何特征 (摆脱VMTK依赖)...")
    
    # 获取坐标点 (N, 3)
    cl_points = vtk_to_numpy(centerline.GetPoints().GetData())
    
    # A. 切线
    arr_tangent = compute_tangents_manually(cl_points)
    # B. 曲率 (新增)
    arr_curv = compute_curvature_manually(cl_points)
    # C. 路径长度
    arr_abscissa_raw = compute_abscissas_manually(centerline)
    
    # --- 5. 映射 ---
    print("4. [Mapping] 特征投影到表面...")
    
    cl_pd = centerline.GetPointData()
    # 半径是唯一必须从 VMTK 获取的，因为它依赖 Voronoi 图算法
    arr_radius = vtk_to_numpy(cl_pd.GetArray("MaximumInscribedSphereRadius"))
    
    # KDTree
    locator = vtk.vtkKdTreePointLocator()
    locator.SetDataSet(centerline)
    locator.BuildLocator()
    
    surf_points = vtk_to_numpy(surface.GetPoints().GetData())
    n_pts = surf_points.shape[0]
    
    res_abscissa = np.zeros(n_pts)
    res_norm_dist = np.zeros(n_pts)
    res_curv = np.zeros(n_pts)
    res_tangent = np.zeros((n_pts, 3))
    
    for i in range(n_pts):
        pt = surf_points[i]
        closest_id = locator.FindClosestPoint(pt)
        
        dist = np.linalg.norm(pt - cl_points[closest_id])
        local_r = arr_radius[closest_id]
        
        res_abscissa[i] = arr_abscissa_raw[closest_id]
        res_curv[i] = arr_curv[closest_id] # 现在使用手动计算值
        res_tangent[i] = arr_tangent[closest_id]
        
        safe_r = local_r if local_r > 1e-6 else 1.0
        res_norm_dist[i] = dist / safe_r

    # --- 6. 后处理 ---
    ab_min, ab_max = np.nanmin(res_abscissa), np.nanmax(res_abscissa)
    denom = ab_max - ab_min
    if denom > 1e-8:
        res_abscissa = (res_abscissa - ab_min) / denom
    else:
        res_abscissa[:] = 0.0
        
    # NaN 清洗
    res_abscissa = np.nan_to_num(res_abscissa)
    res_norm_dist = np.nan_to_num(res_norm_dist)
    res_curv = np.nan_to_num(res_curv)
    res_tangent = np.nan_to_num(res_tangent)

    # --- 7. 写入 ---
    def add_array(name, data):
        arr = numpy_to_vtk(data, deep=1)
        arr.SetName(name)
        surface.GetPointData().AddArray(arr)
        
    add_array("Feature_Abscissa", res_abscissa)
    add_array("Feature_NormRadius", res_norm_dist)
    add_array("Feature_Curvature", res_curv)
    add_array("Feature_Tangent", res_tangent)
    
    return surface, centerline

# ==========================================
# 4. 入口
# ==========================================
if __name__ == "__main__":
    input_stl = "MA+XIAO+DONG-new.stl"
    
    if os.path.exists(input_stl):
        try:
            print(f"🚀 开始处理: {input_stl}")
            surf = read_surface(input_stl)
            final_surf, final_cl = extract_features_robust(surf)
            
            write_vtp(final_surf, "result_surface.vtp")
            write_vtp(final_cl, "result_centerline.vtp")
            
            # 导出CSV
            pd_surf = final_surf.GetPointData()
            points = vtk_to_numpy(final_surf.GetPoints().GetData())
            tangents = vtk_to_numpy(pd_surf.GetArray("Feature_Tangent"))
            
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
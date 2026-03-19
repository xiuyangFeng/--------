#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
vmtk_aorta_centerline.py
从表面网格提取腹主动脉中心线并导出逐点几何特征。

依赖：
  conda create -n vmtk -c conda-forge vmtk vtk itk python=3.10
  conda activate vmtk
"""

import argparse
import sys
import os
import numpy as np
import pandas as pd

try:
    from vmtk import vmtkscripts as vmtk
    import vtk
except Exception as e:
    print("❌ 无法导入 vmtk/vtk，请先用 conda 安装：conda install -c conda-forge vmtk vtk itk")
    raise

def read_surface(path):
    reader = vmtk.vmtkSurfaceReader()
    reader.InputFileName = path
    reader.Execute()
    return reader.Surface

def write_surface(surface, path):
    writer = vmtk.vmtkSurfaceWriter()
    writer.Surface = surface
    writer.OutputFileName = path
    writer.Execute()

def clean_surface(surface, clean=True, smooth=False, iterations=20, relaxation=0.1):
    surf = surface

    if clean:
        cleaner = vmtk.vmtkSurfaceCleaner()
        cleaner.Surface = surf
        cleaner.CleanOutput = 1
        cleaner.Execute()
        surf = cleaner.Surface

    if smooth:
        smoother = vmtk.vmtkSurfaceSmoothing()
        smoother.Surface = surf
        smoother.NumberOfIterations = iterations
        smoother.RelaxationFactor = relaxation
        smoother.Execute()
        surf = smoother.Surface

    return surf

def cap_surface(surface):
    capper = vmtk.vmtkSurfaceCapper()
    capper.Surface = surface
    capper.Execute()
    return capper.Surface

def compute_centerlines(surface, seed_mode="openprofiles",
                        sources=None, targets=None,
                        resample_spacing=None,
                        keep_longest=False):
    """
    seed_mode: 'openprofiles'（默认）或 'pointlist'
    sources/targets: [(x,y,z), ...]，仅在 pointlist 时使用
    resample_spacing: float，若给出则对中心线按弧长重采样
    keep_longest: True 时，仅保留最长一条中心线（主干）
    """
    cl = vmtk.vmtkCenterlines()
    cl.Surface = surface

    if seed_mode == "openprofiles":
        cl.SeedSelectorName = 'openprofiles'
    elif seed_mode == "pointlist":
        cl.SeedSelectorName = 'pointlist'
        if not sources or not targets:
            raise ValueError("pointlist 模式需要 --sources 与 --targets")
        # 设置源点与目标点
        srcPts = vtk.vtkPoints()
        for p in sources: srcPts.InsertNextPoint(p)
        tgtPts = vtk.vtkPoints()
        for p in targets: tgtPts.InsertNextPoint(p)
        cl.SourcePoints = srcPts
        cl.TargetPoints = tgtPts
    else:
        raise ValueError("seed_mode 只能为 'openprofiles' 或 'pointlist'")

    cl.AppendEndPointsToCenterlines = 1  # 更完整的端点
    cl.Execute()
    centerlines = cl.Centerlines

    # 计算属性：沿程、法线等
    attr = vmtk.vmtkCenterlineAttributes()
    attr.Centerlines = centerlines
    attr.Execute()
    centerlines = attr.Centerlines

    # 几何量：曲率/扭率/Frenet 框架
    geo = vmtk.vmtkCenterlineGeometry()
    geo.Centerlines = centerlines
    geo.Execute()
    centerlines = geo.Centerlines

    # 可选：只保留最长中心线主干
    if keep_longest:
        stripper = vmtk.vmtkCenterlineSmoothing()  # 只为获得 PolyData 拆分
        stripper.Centerlines = centerlines
        stripper.Execute()  # 不改变拓扑，这里只是触发更新
        # 手动按 cell 计算长度并筛最长
        lengths = []
        cell_ids = []
        for i in range(centerlines.GetNumberOfCells()):
            cell = centerlines.GetCell(i)
            ids = cell.GetPointIds()
            length = 0.0
            for k in range(ids.GetNumberOfIds()-1):
                p0 = np.array(centerlines.GetPoint(ids.GetId(k)))
                p1 = np.array(centerlines.GetPoint(ids.GetId(k+1)))
                length += np.linalg.norm(p1 - p0)
            lengths.append(length); cell_ids.append(i)
        if lengths:
            max_id = int(np.argmax(lengths))
            extract = vtk.vtkExtractCells()
            extract.SetInputData(centerlines)
            id_list = vtk.vtkIdList(); id_list.InsertNextId(max_id)
            extract.SetCellList(id_list)
            extract.Update()
            gf = vtk.vtkGeometryFilter()
            gf.SetInputData(extract.GetOutput())
            gf.Update()
            centerlines = gf.GetOutput()

    # 可选：按弧长重采样（更均匀的点间距，曲率更稳）
    if resample_spacing and resample_spacing > 0:
        resampler = vmtk.vmtkCenterlineResampling()
        resampler.Centerlines = centerlines
        resampler.Length = 0.0  # 用 spacing 而非总长
        resampler.Resampling = 1
        resampler.ResamplingStepLength = float(resample_spacing)
        resampler.Execute()
        centerlines = resampler.Centerlines

        # 重采样后，属性与几何建议再算一次
        attr = vmtk.vmtkCenterlineAttributes()
        attr.Centerlines = centerlines
        attr.Execute()
        centerlines = attr.Centerlines

        geo = vmtk.vmtkCenterlineGeometry()
        geo.Centerlines = centerlines
        geo.Execute()
        centerlines = geo.Centerlines

    return centerlines

def vtk_array_to_np(pddata, name, npts, ncomp=1):
    arr = pddata.GetArray(name)
    if arr is None:
        return None
    if ncomp == 1:
        return np.array([arr.GetTuple1(i) for i in range(npts)])
    elif ncomp == 3:
        out = np.zeros((npts,3))
        for i in range(npts):
            out[i,:] = np.array(arr.GetTuple3(i))
        return out
    else:
        # 通用 ncomp
        out = np.zeros((npts, ncomp))
        for i in range(npts):
            out[i,:] = np.array(arr.GetTuple(i))
        return out

def export_point_features(centerlines, csv_path):
    npts = centerlines.GetNumberOfPoints()
    pts = np.array([centerlines.GetPoint(i) for i in range(npts)])
    pddata = centerlines.GetPointData()

    # 常见字段名（VMTK 输出）
    s   = vtk_array_to_np(pddata, 'Abscissas', npts, 1)
    r   = vtk_array_to_np(pddata, 'MaximumInscribedSphereRadius', npts, 1)
    k   = vtk_array_to_np(pddata, 'Curvature', npts, 1)
    tau = vtk_array_to_np(pddata, 'Torsion', npts, 1)
    T   = vtk_array_to_np(pddata, 'FrenetTangent', npts, 3)
    N   = vtk_array_to_np(pddata, 'FrenetNormal', npts, 3)
    B   = vtk_array_to_np(pddata, 'FrenetBinormal', npts, 3)

    # 分支/段信息（如存在）
    group_ids = vtk_array_to_np(pddata, 'GroupIds', npts, 1)
    tract_ids = vtk_array_to_np(pddata, 'TractIds', npts, 1)
    centerline_ids = vtk_array_to_np(pddata, 'CenterlineIds', npts, 1)

    df = pd.DataFrame({
        'x': pts[:,0], 'y': pts[:,1], 'z': pts[:,2],
        's': s, 'radius': r, 'curvature': k, 'torsion': tau,
        'Tx': T[:,0] if T is not None else None,
        'Ty': T[:,1] if T is not None else None,
        'Tz': T[:,2] if T is not None else None,
        'Nx': N[:,0] if N is not None else None,
        'Ny': N[:,1] if N is not None else None,
        'Nz': N[:,2] if N is not None else None,
        'Bx': B[:,0] if B is not None else None,
        'By': B[:,1] if B is not None else None,
        'Bz': B[:,2] if B is not None else None,
        'GroupId': group_ids[:,0] if group_ids is not None else None,
        'TractId': tract_ids[:,0] if tract_ids is not None else None,
        'CenterlineId': centerline_ids[:,0] if centerline_ids is not None else None,
    })
    df.to_csv(csv_path, index=False)
    return df

def parse_point_list(s):
    # "x,y,z;x,y,z" -> [(x,y,z),...]
    triples = []
    for seg in s.split(';'):
        if not seg.strip():
            continue
        xyz = [float(v) for v in seg.strip().split(',')]
        if len(xyz) != 3:
            raise ValueError("坐标必须是 x,y,z 三元组，用分号分隔多个点")
        triples.append(tuple(xyz))
    return triples

def main():
    ap = argparse.ArgumentParser(description="从表面网格提取腹主动脉中心线并导出几何特征")
    ap.add_argument("--ifile", required=True, help="输入表面文件（.stl/.vtp/.vtk/.ply 等）")
    ap.add_argument("--ocl", default="centerlines.vtp", help="输出中心线 VTP（默认 centerlines.vtp）")
    ap.add_argument("--ocsv", default="centerline_features.csv", help="输出逐点特征 CSV（默认 centerline_features.csv）")

    ap.add_argument("--seed", choices=["openprofiles","pointlist"], default="openprofiles",
                    help="中心线种子模式：openprofiles（自动）或 pointlist（手动）")
    ap.add_argument("--sources", type=str, help='pointlist 模式下的源点："x,y,z;x,y,z"')
    ap.add_argument("--targets", type=str, help='pointlist 模式下的目标点："x,y,z;x,y,z"')

    ap.add_argument("--clean", action="store_true", help="清理表面（去孤立片/重复点等）")
    ap.add_argument("--smooth", action="store_true", help="平滑表面")
    ap.add_argument("--smooth-iters", type=int, default=20, help="平滑迭代次数（默认20）")
    ap.add_argument("--smooth-relax", type=float, default=0.1, help="平滑松弛系数（默认0.1）")

    ap.add_argument("--resample-spacing", type=float, default=None, help="中心线重采样步长（按弧长，单位 mm 等）")
    ap.add_argument("--keep-longest", action="store_true", help="仅保留最长主干中心线")

    ap.add_argument("--save-cleaned", default=None, help="可选：保存清理/封口后的表面到该路径")

    args = ap.parse_args()

    if not os.path.isfile(args.ifile):
        print(f"❌ 找不到输入文件：{args.ifile}")
        sys.exit(1)

    print("▶ 读取表面…")
    surface = read_surface(args.ifile)

    print("▶ 清理/平滑…")
    surface = clean_surface(surface, clean=args.clean, smooth=args.smooth,
                            iterations=args.smooth_iters, relaxation=args.smooth_relax)

    print("▶ 封口（cap）…")
    surface = cap_surface(surface)

    if args.save_cleaned:
        print(f"▶ 保存清理后的表面：{args.save_cleaned}")
        write_surface(surface, args.save_cleaned)

    seed_mode = args.seed
    sources = targets = None
    if seed_mode == "pointlist":
        if not args.sources or not args.targets:
            print("❌ pointlist 模式需要 --sources 与 --targets")
            sys.exit(2)
        sources = parse_point_list(args.sources)
        targets = parse_point_list(args.targets)
        print(f"▶ 手动种子：源点 {sources}；目标点 {targets}")
    else:
        print("▶ 自动种子模式：openprofiles")

    print("▶ 计算中心线…（可能需要几秒到几十秒，视网格复杂度而定）")
    centerlines = compute_centerlines(
        surface,
        seed_mode=seed_mode,
        sources=sources, targets=targets,
        resample_spacing=args.resample_spacing,
        keep_longest=args.keep_longest
    )

    print(f"▶ 导出中心线：{args.ocl}")
    write_surface(centerlines, args.ocl)

    print(f"▶ 导出逐点特征：{args.ocsv}")
    df = export_point_features(centerlines, args.ocsv)
    print(f"✅ 完成！共 {len(df)} 个中心线点。")

if __name__ == "__main__":
    main()

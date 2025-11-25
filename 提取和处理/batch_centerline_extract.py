import os
import argparse
import traceback
from typing import Optional

import pandas as pd
import vtk
from vtkmodules.util.numpy_support import vtk_to_numpy

import vmtk_core


def find_stl(case_dir: str) -> Optional[str]:
    """
    在病例目录下寻找可用的 STL 文件。
    优先匹配与目录同名的文件（如 004/004.stl），否则取第一个 .stl。
    """
    case_name = os.path.basename(case_dir)
    prefer_path = os.path.join(case_dir, f"{case_name}.stl")
    if os.path.exists(prefer_path):
        return prefer_path

    for fname in os.listdir(case_dir):
        if fname.lower().endswith(".stl"):
            return os.path.join(case_dir, fname)
    return None


def export_centerline(centerline, vtp_path: str, csv_path: Optional[str]):
    """
    将中心线同时保存为 VTP 与 CSV（便于快速查看）。
    """
    os.makedirs(os.path.dirname(vtp_path), exist_ok=True)

    # 保存 VTP
    writer = vtk.vtkXMLPolyDataWriter()
    writer.SetFileName(vtp_path)
    writer.SetInputData(centerline)
    if writer.Write() == 0:
        raise RuntimeError(f"写入 VTP 失败: {vtp_path}")

    if not csv_path:
        return

    pts = vtk_to_numpy(centerline.GetPoints().GetData())
    pd_obj = centerline.GetPointData()

    def get_arr(name: str):
        arr = pd_obj.GetArray(name)
        return vtk_to_numpy(arr) if arr is not None else None

    data = {
        "x": pts[:, 0],
        "y": pts[:, 1],
        "z": pts[:, 2],
    }

    # 尝试导出常用几何属性，若不存在则跳过
    for key in [
        "Abscissas",
        "MaximumInscribedSphereRadius",
        "Curvature",
    ]:
        arr = get_arr(key)
        if arr is not None:
            data[key] = arr

    tangent = get_arr("FrenetTangent")
    if tangent is not None and tangent.shape[1] == 3:
        data["Tangent_X"] = tangent[:, 0]
        data["Tangent_Y"] = tangent[:, 1]
        data["Tangent_Z"] = tangent[:, 2]

    pd.DataFrame(data).to_csv(csv_path, index=False)


def process_case(case_dir: str, output_root: str, overwrite: bool, save_csv: bool):
    """
    针对单个病例执行中心线提取。
    - case_dir: 病例目录（内含 STL + ascii 点云）
    - output_root: 输出根目录，会在其中生成 case/centerline 目录
    """
    case_name = os.path.basename(case_dir)
    stl_path = find_stl(case_dir)
    if not stl_path:
        print(f"⚠️  跳过 {case_name}: 未找到 STL。")
        return False

    case_out_dir = os.path.join(output_root, case_name, "centerline")
    vtp_path = os.path.join(case_out_dir, "centerline.vtp")
    csv_path = os.path.join(case_out_dir, "centerline_points.csv") if save_csv else None

    if (os.path.exists(vtp_path) or (csv_path and os.path.exists(csv_path))) and not overwrite:
        print(f"⏩ {case_name} 已存在输出，使用 --overwrite 可重算。")
        return True

    try:
        print(f"🚀 处理 {case_name} ...")
        surface = vmtk_core.read_surface(stl_path)
        centerline = vmtk_core.extract_rich_centerline(surface)
        export_centerline(centerline, vtp_path, csv_path)
        print(f"✅ {case_name} 完成，保存至 {case_out_dir}")
        return True
    except Exception as e:
        print(f"❌ {case_name} 失败: {e}")
        traceback.print_exc()
        return False


def main():
    parser = argparse.ArgumentParser(description="批量提取中心线 (基于点云目录结构)")
    parser.add_argument(
        "--input_dir",
        default="点云",
        help="输入根目录，形如 点云/004、点云/010 ...",
    )
    parser.add_argument(
        "--output_dir",
        default="点云",
        help="输出根目录，默认写回点云目录下的每个 case/centerline",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="若输出已存在则强制重算",
    )
    parser.add_argument(
        "--no_csv",
        action="store_true",
        help="仅生成 VTP，不导出 CSV",
    )
    args = parser.parse_args()

    input_root = args.input_dir
    output_root = args.output_dir
    save_csv = not args.no_csv

    if not os.path.isdir(input_root):
        print(f"❌ 输入目录不存在: {input_root}")
        return

    subdirs = [
        os.path.join(input_root, d)
        for d in os.listdir(input_root)
        if os.path.isdir(os.path.join(input_root, d)) and not d.startswith(".")
    ]
    subdirs.sort()

    print(f"📂 待处理病例数: {len(subdirs)} (根目录: {input_root})")

    ok = 0
    for idx, case_dir in enumerate(subdirs, start=1):
        print(f"[{idx}/{len(subdirs)}] -> {os.path.basename(case_dir)}")
        if process_case(case_dir, output_root, args.overwrite, save_csv):
            ok += 1

    print(f"🎉 完成: {ok}/{len(subdirs)} 个病例生成中心线。输出根目录: {output_root}")


if __name__ == "__main__":
    main()

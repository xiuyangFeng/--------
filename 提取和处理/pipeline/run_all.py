#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Pipeline 一键运行脚本

按顺序执行完整的数据处理流程:
1. preprocess.py: 数据清洗 + 合并 + 降采样
2. extract_features.py: 几何特征提取 + 边界条件
3. coord_normalize.py: 坐标系归一化（中心化 + PCA对齐 + 缩放）【新增】
4. normalize.py: 特征归一化
5. convert_to_graph.py: 转换为图数据

使用示例:
  # 处理单个病例
  python -m pipeline.run_all --case ZHANG_CHUN
  
  # 处理所有病例
  python -m pipeline.run_all
  
  # 跳过已完成的步骤
  python -m pipeline.run_all --start-step 3
  
  # 使用随机采样（速度快）
  python -m pipeline.run_all --sampling-method random
"""

import argparse
import shutil
import sys
import time
from pathlib import Path
from typing import Optional, List

# 导入各处理模块
if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from pipeline.config import (
        DATA_ROOT,
        MERGED_DIR,
        FEATURES_DIR,
        COORD_NORMALIZED_DIR,
        NORMALIZED_DIR,
        GRAPHS_DIR,
        SAMPLING_CONFIG,
        GRAPH_CONFIG,
        MODE,
        get_case_dirs,
    )
    from pipeline.validation import build_batch_issue_report, save_batch_issue_report
else:
    from .config import (
        DATA_ROOT,
        MERGED_DIR,
        FEATURES_DIR,
        COORD_NORMALIZED_DIR,
        NORMALIZED_DIR,
        GRAPHS_DIR,
        SAMPLING_CONFIG,
        GRAPH_CONFIG,
        MODE,
        get_case_dirs,
    )
    from .validation import build_batch_issue_report, save_batch_issue_report


def run_pipeline(
    data_root: Path = None,
    target_case: Optional[str] = None,
    start_step: int = 1,
    end_step: int = 5,
    target_points: int = None,
    sampling_method: str = None,
    fps_ratio: float = None,
    mode: str = None,
    k_neighbors: int = None,
    strict_bc_match: bool = True,
) -> None:
    """
    运行完整的数据处理流程。
    
    参数:
        data_root: 数据根目录
        target_case: 指定处理的病例名称
        start_step: 开始步骤 (1-5)
        end_step: 结束步骤 (1-5)
        target_points: 目标点数
        sampling_method: 采样方法
        mode: 处理模式
        k_neighbors: KNN 邻居数
    """
    if data_root is None:
        data_root = DATA_ROOT
    else:
        data_root = Path(data_root)
    
    if target_points is None:
        target_points = SAMPLING_CONFIG["target_total_points"]
    if sampling_method is None:
        sampling_method = SAMPLING_CONFIG["sampling_method"]
    if fps_ratio is None:
        fps_ratio = SAMPLING_CONFIG["hybrid_fps_ratio"]
    if mode is None:
        mode = MODE
    if k_neighbors is None:
        k_neighbors = GRAPH_CONFIG["k_neighbors"]
    
    print("=" * 60)
    print("🚀 Pipeline - 完整处理流程（含坐标系归一化）")
    print("=" * 60)
    print(f"📁 数据根目录: {data_root}")
    print(f"📊 目标点数: {target_points}")
    print(f"📊 采样方法: {sampling_method}")
    if sampling_method == "hybrid":
        print(f"📊 FPS 占比: {fps_ratio}")
    print(f"📊 处理模式: {mode}")
    print(f"📊 KNN 邻居数: {k_neighbors}")
    print(f"📊 BC 严格匹配: {'是' if strict_bc_match else '否'}")
    
    case_dirs = get_case_dirs(data_root)
    if target_case:
        target_std = target_case.replace(' ', '_').replace('-', '_').upper()
        case_dirs = [
            d for d in case_dirs
            if d.name.replace(' ', '_').replace('-', '_').upper() == target_std
        ]
        print(f"🎯 指定病例: {target_case}")
    else:
        print(f"📊 待处理病例数: {len(case_dirs)}")

    if not case_dirs:
        if target_case:
            print(f"❌ 未找到病例: {target_case}")
        else:
            print("❌ 未找到任何病例")
        return
    
    print(f"📋 执行步骤: {start_step} → {end_step}")
    audit_report = build_batch_issue_report(case_dirs)
    audit_dir = data_root / "pipeline_reports"
    report_name = (
        f"pipeline_input_audit_step{start_step}_{end_step}"
        if target_case is None
        else f"pipeline_input_audit_{case_dirs[0].name}_step{start_step}_{end_step}"
    )
    audit_json, audit_csv = save_batch_issue_report(audit_report, audit_dir, report_name)
    print(f"📝 输入检查报告: {audit_json}")
    print(f"📝 输入检查表格: {audit_csv}")
    if start_step <= 1:
        print(f"📊 可直接跑步骤1的病例: {audit_report['preprocess_ready_count']}/{len(case_dirs)}")
    if start_step <= 2 <= end_step:
        print(f"📊 具备几何+BC输入的病例: {audit_report['feature_ready_count']}/{len(case_dirs)}")
    print()
    
    total_start = time.time()

    if __package__ in {None, ""}:
        from pipeline.preprocess import process_all_cases as preprocess
        from pipeline.extract_features import process_all_cases as extract_features
        from pipeline.coord_normalize import process_all_cases as coord_normalize
        from pipeline.normalize import process_all_cases as normalize
        from pipeline.convert_to_graph import process_all_cases as convert_to_graph
    else:
        from .preprocess import process_all_cases as preprocess
        from .extract_features import process_all_cases as extract_features
        from .coord_normalize import process_all_cases as coord_normalize
        from .normalize import process_all_cases as normalize
        from .convert_to_graph import process_all_cases as convert_to_graph
    
    # 步骤1: 数据预处理
    if start_step <= 1 <= end_step:
        print("\n" + "=" * 60)
        print("📌 步骤1/5: 数据预处理（清洗 + 合并 + 降采样）")
        print("=" * 60)
        
        preprocess(
            data_root=data_root,
            target_case=target_case,
            target_total=target_points,
            sampling_method=sampling_method,
            boundary_threshold=SAMPLING_CONFIG["boundary_threshold"],
            boundary_core_ratio=SAMPLING_CONFIG["boundary_core_ratio"],
            seed=SAMPLING_CONFIG["seed"],
            fps_ratio=fps_ratio,
            mode=mode,
        )
    
    # 步骤2: 几何特征提取
    if start_step <= 2 <= end_step:
        print("\n" + "=" * 60)
        print("📌 步骤2/5: 几何特征提取 + 边界条件")
        print("=" * 60)
        
        extract_features(
            data_root=data_root,
            target_case=target_case,
            input_subdir=MERGED_DIR,
            output_subdir=FEATURES_DIR,
            strict_bc_match=strict_bc_match,
        )
    
    # 步骤3: 坐标系归一化【新增】
    if start_step <= 3 <= end_step:
        print("\n" + "=" * 60)
        print("📌 步骤3/5: 坐标系归一化（中心化 + PCA对齐 + 缩放）")
        print("=" * 60)
        
        coord_normalize(
            data_root=data_root,
            target_case=target_case,
            input_subdir=FEATURES_DIR,
            output_subdir=COORD_NORMALIZED_DIR,
        )
    
    # 步骤4: 特征归一化
    if start_step <= 4 <= end_step:
        print("\n" + "=" * 60)
        print("📌 步骤4/5: 特征归一化")
        print("=" * 60)
        
        normalize(
            data_root=data_root,
            target_case=target_case,
            input_subdir=COORD_NORMALIZED_DIR,
            output_subdir=NORMALIZED_DIR,
        )
    
    # 步骤5: 图数据转换
    if start_step <= 5 <= end_step:
        print("\n" + "=" * 60)
        print("📌 步骤5/5: 图数据转换")
        print("=" * 60)
        
        convert_to_graph(
            data_root=data_root,
            target_case=target_case,
            input_subdir=NORMALIZED_DIR,
            output_subdir=GRAPHS_DIR,
            k=k_neighbors,
        )

    if mode == "production":
        cleanup_intermediate_outputs(
            case_dirs=case_dirs,
            start_step=start_step,
            end_step=end_step,
        )
    
    total_time = time.time() - total_start
    
    print("\n" + "=" * 60)
    print("🎉 Pipeline 执行完成!")
    print("=" * 60)
    print(f"⏱️  总耗时: {total_time:.1f}s ({total_time/60:.1f} 分钟)")
    
    print("\n📂 输出目录结构:")
    print(f"  病例目录/")
    print(f"  └── processed/")
    print(f"      ├── merged/           # 步骤1: 合并降采样后的数据")
    print(f"      ├── features/         # 步骤2: 添加几何特征和边界条件")
    print(f"      ├── coord_normalized/ # 步骤3: 坐标系归一化后的数据")
    print(f"      │   └── transform_params.json  # 变换参数（用于逆变换）")
    print(f"      ├── normalized/       # 步骤4: 特征归一化后的数据")
    print(f"      └── graphs/           # 步骤5: PyG 图数据 (.pt)")
    print(f"          └── transform_params.json  # 变换参数副本")
    
    print("\n💡 提示:")
    print("  - 训练时使用 pipeline.dataset.CFDAugmentedDataset 加载数据")
    print("  - 启用 augment=True 进行在线数据增强（旋转、平移）")
    print("  - 推理时可使用 transform_params.json 还原到原始坐标系")
    if mode == "production":
        print("  - production 模式已自动清理本次运行上游步骤的中间目录")


def _step_output_dir(case_dir: Path, step: int) -> Path:
    step_dirs = {
        1: MERGED_DIR,
        2: FEATURES_DIR,
        3: COORD_NORMALIZED_DIR,
        4: NORMALIZED_DIR,
        5: GRAPHS_DIR,
    }
    return case_dir / step_dirs[step]


def _case_has_final_output(case_dir: Path, final_step: int) -> bool:
    final_dir = _step_output_dir(case_dir, final_step)
    if not final_dir.exists():
        return False

    if final_step == 5:
        return any(final_dir.glob("*.pt"))
    if final_step == 4:
        return any(final_dir.glob("result_features_*.csv"))
    if final_step == 3:
        return (final_dir / "transform_params.json").exists()
    if final_step == 2:
        return any(final_dir.glob("result_features_*.csv"))
    if final_step == 1:
        return any(final_dir.glob("merged-*.csv"))
    return False


def cleanup_intermediate_outputs(
    case_dirs: List[Path],
    start_step: int,
    end_step: int,
) -> None:
    """
    在 production 模式下清理本次运行生成的上游中间目录。

    仅当最终步骤的目标输出存在时才清理，避免误删失败病例的数据。
    """
    if start_step >= end_step:
        return

    print("\n" + "=" * 60)
    print("🧹 Production 清理中间目录")
    print("=" * 60)

    deleted = 0
    skipped = 0

    for case_dir in case_dirs:
        if not _case_has_final_output(case_dir, end_step):
            print(f"  ⚠️ 跳过 {case_dir.name}: 未检测到步骤{end_step}的有效输出")
            skipped += 1
            continue

        for step in range(start_step, end_step):
            output_dir = _step_output_dir(case_dir, step)
            if output_dir.exists():
                shutil.rmtree(output_dir)
                print(f"  🗑️ 已删除 {case_dir.name}/{output_dir.relative_to(case_dir)}")
                deleted += 1

    print(f"✅ 清理完成: 删除 {deleted} 个目录, 跳过 {skipped} 个病例")


def main():
    parser = argparse.ArgumentParser(
        description="Pipeline - 一键运行完整处理流程",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
处理流程:
  步骤1: preprocess.py       - 数据清洗 + 合并 + 降采样
  步骤2: extract_features.py - 几何特征提取 + 边界条件
  步骤3: coord_normalize.py  - 坐标系归一化（中心化+PCA对齐+缩放）【新增】
  步骤4: normalize.py        - 特征归一化
  步骤5: convert_to_graph.py - 转换为图数据

坐标系归一化说明:
  - 消除不同病例间血管位置和朝向的差异
  - 将坐标归一化到 [-1, 1] 范围
  - 速度、切线等矢量特征同步旋转

示例:
  # 处理单个病例（完整流程）
  python -m pipeline.run_all --case ZHANG_CHUN
  
  # 处理所有病例
  python -m pipeline.run_all
  
  # 从步骤3开始（跳过预处理和特征提取）
  python -m pipeline.run_all --start-step 3
  
  # 只执行步骤1和2
  python -m pipeline.run_all --end-step 2
  
  # 使用随机采样（速度快）
  python -m pipeline.run_all --sampling-method random
        """
    )
    parser.add_argument(
        "--data-root",
        type=str,
        default=None,
        help="数据根目录",
    )
    parser.add_argument(
        "--case",
        type=str,
        default=None,
        help="指定处理的病例名称",
    )
    parser.add_argument(
        "--start-step",
        type=int,
        default=1,
        choices=[1, 2, 3, 4, 5],
        help="开始步骤 (1-5)，默认 1",
    )
    parser.add_argument(
        "--end-step",
        type=int,
        default=5,
        choices=[1, 2, 3, 4, 5],
        help="结束步骤 (1-5)，默认 5",
    )
    parser.add_argument(
        "--target-points",
        type=int,
        default=None,
        help=f"目标总点数，默认 {SAMPLING_CONFIG['target_total_points']}",
    )
    parser.add_argument(
        "--sampling-method",
        type=str,
        choices=["fps", "random", "hybrid"],
        default=None,
        help=f"采样方法，默认 {SAMPLING_CONFIG['sampling_method']}",
    )
    parser.add_argument(
        "--mode",
        type=str,
        choices=["debug", "production"],
        default=None,
        help=f"处理模式，默认 {MODE}",
    )
    parser.add_argument(
        "--fps-ratio",
        type=float,
        default=None,
        help=f"混合采样时 FPS 占比，默认 {SAMPLING_CONFIG['hybrid_fps_ratio']}",
    )
    parser.add_argument(
        "--k",
        type=int,
        default=None,
        help=f"KNN 邻居数，默认 {GRAPH_CONFIG['k_neighbors']}",
    )
    parser.add_argument(
        "--allow-nearest-bc",
        action="store_true",
        help="允许使用最近时间步 BC 作为兜底；默认严格匹配",
    )
    
    args = parser.parse_args()
    
    if args.start_step > args.end_step:
        print("❌ 错误: start-step 不能大于 end-step")
        return
    
    run_pipeline(
        data_root=args.data_root,
        target_case=args.case,
        start_step=args.start_step,
        end_step=args.end_step,
        target_points=args.target_points,
        sampling_method=args.sampling_method,
        fps_ratio=args.fps_ratio,
        mode=args.mode,
        k_neighbors=args.k,
        strict_bc_match=not args.allow_nearest_bc,
    )


if __name__ == "__main__":
    main()

import argparse
from pathlib import Path
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def load_cloud(path: Path) -> pd.DataFrame:
    """读取单个点云并去掉 node_id 列。"""
    df = pd.read_csv(path)
    if "node_id" in df.columns:
        df = df.drop(columns=["node_id"])
    return df


def merge_and_shuffle(surface_df: pd.DataFrame, inner_df: pd.DataFrame, seed: int | None = None) -> pd.DataFrame:
    """对齐列后合并，并随机打乱行。"""
    all_cols = list(surface_df.columns)
    for col in inner_df.columns:
        if col not in all_cols:
            all_cols.append(col)

    surface_df = surface_df.reindex(columns=all_cols)
    inner_df = inner_df.reindex(columns=all_cols)

    merged = pd.concat([surface_df, inner_df], ignore_index=True)
    if seed is not None:
        merged = merged.sample(frac=1, random_state=seed).reset_index(drop=True)
    else:
        merged = merged.sample(frac=1).reset_index(drop=True)
    return merged


def process_case(
    case_dir: Path,
    surface_dir: str = "ascii_clean",
    inner_dir: str = "ascii_in_clean",
    output_dir: str = "ascii_merged",
    seed: int | None = None,
) -> None:
    """处理单个病例目录，将同名的壁面点与内部点合并打乱。"""
    surface_path = case_dir / surface_dir
    inner_path = case_dir / inner_dir
    if not surface_path.is_dir() or not inner_path.is_dir():
        print(f"⚠️  跳过 {case_dir.name}: 缺少 {surface_dir} 或 {inner_dir}")
        return

    surface_files = {p.stem: p for p in surface_path.glob("*.csv")}
    inner_files = {p.stem: p for p in inner_path.glob("*.csv")}
    common_keys = sorted(set(surface_files) & set(inner_files))
    if not common_keys:
        print(f"⚠️  跳过 {case_dir.name}: 未找到同名的清洗与内部点文件")
        return

    out_dir = case_dir / output_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    for key in common_keys:
        try:
            surf_df = load_cloud(surface_files[key])
            inner_df = load_cloud(inner_files[key])
            merged = merge_and_shuffle(surf_df, inner_df, seed=seed)
            out_path = out_dir / f"{key}.csv"
            merged.to_csv(out_path, index=False)
            print(f"✅ {case_dir.name} 合并完成: {key}.csv -> {out_path}")
        except Exception as e:
            print(f"❌ 合并 {case_dir.name}/{key} 失败: {e}")


def batch_merge(
    root_dir: Path,
    surface_dir: str = "ascii_clean",
    inner_dir: str = "ascii_in_clean",
    output_dir: str = "ascii_merged",
    seed: int | None = None,
) -> None:
    """批量遍历根目录下的病例文件夹，合并对应点云。"""
    root_dir = Path(root_dir)
    if not root_dir.is_dir():
        print(f"❌ 根目录不存在: {root_dir}")
        return

    case_dirs = [p for p in root_dir.iterdir() if p.is_dir()]
    if not case_dirs:
        print(f"❌ {root_dir} 下未找到病例目录")
        return

    for case_dir in sorted(case_dirs):
        process_case(case_dir, surface_dir, inner_dir, output_dir, seed)


def main() -> None:
    parser = argparse.ArgumentParser(description="合并 ascii_clean 与 ascii_in_clean 点云并随机打乱")
    parser.add_argument(
        "--root_dir",
        type=Path,
        default=PROJECT_ROOT / "点云",
        help="病例根目录，默认项目根目录下的 点云",
    )
    parser.add_argument("--surface_dir", type=str, default="ascii_clean", help="壁面点清洗目录名，默认 ascii_clean")
    parser.add_argument("--inner_dir", type=str, default="ascii_in_clean", help="内部点清洗目录名，默认 ascii_in_clean")
    parser.add_argument("--output_dir", type=str, default="ascii_merged", help="合并后输出目录名，默认 ascii_merged")
    parser.add_argument("--seed", type=int, default=1234, help="随机种子，可选，便于复现打乱结果")
    args = parser.parse_args()

    batch_merge(
        root_dir=args.root_dir,
        surface_dir=args.surface_dir,
        inner_dir=args.inner_dir,
        output_dir=args.output_dir,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()

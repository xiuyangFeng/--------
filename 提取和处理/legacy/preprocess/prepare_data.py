"""
一体化数据整理脚本：
- 可选：按姓名重命名 stl_data 内的 STL，生成/更新 id_mapping.csv（保留或移除后缀）。
- 按映射重命名点云目录（点云/*），包括：主目录改为编号、ascii 内文件改为 “编号-时间步”。
- 将 stl_data 内的 STL 拷贝或移动到点云对应编号目录下，形成完整数据集。
"""

from pathlib import Path
import shutil
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]

# -------------------- 配置区 --------------------
# STL 存放目录
STL_DIR = PROJECT_ROOT / "stl_data"
# 点云根目录
POINT_DIR = PROJECT_ROOT / "点云"
# 映射文件路径
MAPPING_PATH = STL_DIR / "id_mapping.csv"

# 预演模式：True 时只打印计划，不改动任何文件
DRY_RUN = False

# 是否执行 STL 重命名与映射生成（通常初始原始姓名时打开，当前数据已编号可保持 False）
RUN_STL_RENAME = False
# STL 重命名时是否保留原有后缀（例如 "-all"）
KEEP_STL_SUFFIX = False

# 是否执行点云目录与 ascii 文件重命名
RUN_POINT_RENAME = False

# 是否执行 STL 合并到点云目录
RUN_MERGE_STL = True
# 合并时是复制还是移动 STL；True = 移动，False = 复制
MOVE_STL = False
# ------------------------------------------------


def norm_name(name: str) -> str:
    """统一姓名格式：去首尾空格，空格转下划线，小写化"""
    return name.strip().replace(" ", "_").lower()


# ============ STL 重命名与映射生成 ============
def collect_stl_files(stl_dir: Path):
    files = sorted(stl_dir.glob("*.stl"), key=lambda p: p.name.lower())
    if not files:
        raise FileNotFoundError(f"{stl_dir} 中未找到 .stl 文件")
    return files


def build_stl_mapping(files):
    records = []
    for idx, path in enumerate(files, start=1):
        base = path.stem
        name_part, suffix_part = (base.split("-", 1) + [""])[:2]
        suffix = f"-{suffix_part}" if suffix_part else ""

        new_id = f"{idx:03d}"
        new_base = f"{new_id}{suffix if KEEP_STL_SUFFIX else ''}"
        new_filename = f"{new_base}{path.suffix}"
        new_path = path.with_name(new_filename)

        records.append(
            {
                "Original_Name": name_part,
                "New_ID": new_id,
                "old_path": path,
                "new_path": new_path,
            }
        )
    return records


def write_mapping_csv(records):
    df = pd.DataFrame(
        [{"Original_Name": r["Original_Name"], "New_ID": r["New_ID"]} for r in records]
    )
    df.to_csv(MAPPING_PATH, index=False)
    print(f"已生成映射文件: {MAPPING_PATH}")


def rename_stl_files():
    files = collect_stl_files(STL_DIR)
    records = build_stl_mapping(files)
    write_mapping_csv(records)

    for r in records:
        old_path = r["old_path"]
        new_path = r["new_path"]
        if new_path.exists() and old_path.resolve() != new_path.resolve():
            raise FileExistsError(f"目标已存在，终止: {new_path}")
        print(f"STL: {old_path.name} -> {new_path.name}")
        if not DRY_RUN:
            old_path.rename(new_path)

    if DRY_RUN:
        print("DRY_RUN 为 True，STL 未重命名。")
    else:
        print("STL 重命名完成。")

    return {norm_name(r["Original_Name"]): r["New_ID"] for r in records}


# ============ 映射加载 ============
def load_mapping():
    """优先读取已有映射；若不存在且允许，则基于 STL 生成"""
    if MAPPING_PATH.exists():
        df = pd.read_csv(MAPPING_PATH, dtype=str)
        mapping = {norm_name(r["Original_Name"]): str(r["New_ID"]) for _, r in df.iterrows()}
        print(f"已读取映射: {MAPPING_PATH}（共 {len(mapping)} 条）")
        return mapping

    if RUN_STL_RENAME:
        print("映射不存在，将基于 STL 生成。")
        return rename_stl_files()

    raise FileNotFoundError(f"缺少映射文件: {MAPPING_PATH}。可设置 RUN_STL_RENAME=True 生成。")


# ============ 点云目录与 ascii 文件重命名 ============
def collect_person_dirs(base_dir: Path):
    if not base_dir.is_dir():
        raise NotADirectoryError(f"未找到点云目录: {base_dir}")
    return sorted([p for p in base_dir.iterdir() if p.is_dir()], key=lambda p: p.name)


def build_point_operations(mapping: dict):
    file_ops = []
    dir_ops = []
    skipped = {"无映射": [], "无ascii目录": [], "无时间步后缀": []}

    for person_dir in collect_person_dirs(POINT_DIR):
        person_key = norm_name(person_dir.name)
        if person_key not in mapping:
            skipped["无映射"].append(person_dir)
            continue

        person_id = mapping[person_key]
        target_dir = person_dir.with_name(person_id)
        if target_dir.exists() and target_dir.resolve() != person_dir.resolve():
            raise FileExistsError(f"目标目录已存在，终止: {target_dir}")
        dir_ops.append((person_dir, target_dir))

        ascii_dir = person_dir / "ascii"
        if not ascii_dir.is_dir():
            skipped["无ascii目录"].append(person_dir)
            continue

        for file in sorted(ascii_dir.iterdir(), key=lambda p: p.name):
            if not file.is_file():
                continue
            name_no_ext = file.stem
            ext = file.suffix
            head, sep, tail = name_no_ext.rpartition("-")
            if sep == "":
                skipped["无时间步后缀"].append(file)
                continue
            new_name = f"{person_id}-{tail}{ext}"
            new_path = file.with_name(new_name)
            if new_path.exists() and file.resolve() != new_path.resolve():
                raise FileExistsError(f"目标已存在，终止: {new_path}")
            file_ops.append((file, new_path))

    return file_ops, dir_ops, skipped


def apply_point_operations(file_ops, dir_ops):
    if file_ops:
        print("文件重命名计划:")
        for old, new in file_ops:
            print(f"{old} -> {new.name}")
        if not DRY_RUN:
            for old, new in file_ops:
                old.rename(new)

    if dir_ops:
        print("\n目录重命名计划:")
        for old, new in dir_ops:
            print(f"{old} -> {new.name}")
        if not DRY_RUN:
            for old, new in dir_ops:
                old.rename(new)

    if DRY_RUN:
        print("\nDRY_RUN 为 True，点云未改动。")
    else:
        print("\n点云重命名完成。")


# ============ 合并 STL 到点云 ============
def build_merge_operations():
    """将 STL 拷贝/移动到点云对应编号目录下"""
    ops = []
    for stl_file in sorted(STL_DIR.glob("*.stl"), key=lambda p: p.name):
        base = stl_file.stem
        id_part = base.split("-", 1)[0]
        target_dir = POINT_DIR / id_part
        target_dir.mkdir(parents=True, exist_ok=True)
        target_file = target_dir / stl_file.name

        if target_file.exists():
            # 已存在同名文件则跳过
            print(f"已存在，跳过: {target_file}")
            continue

        ops.append((stl_file, target_file))
    return ops


def apply_merge_operations(ops):
    action = "移动" if MOVE_STL else "复制"
    if ops:
        print(f"STL {action}计划:")
        for src, dst in ops:
            print(f"{src} -> {dst}")
        if not DRY_RUN:
            for src, dst in ops:
                if MOVE_STL:
                    shutil.move(src, dst)
                else:
                    shutil.copy2(src, dst)
    if DRY_RUN:
        print("\nDRY_RUN 为 True，STL 未合并。")
    else:
        print("\nSTL 合并完成。")


# ============ 主流程 ============
def main():
    mapping = load_mapping()

    if RUN_STL_RENAME and not MAPPING_PATH.exists():
        # 如果刚生成映射，重命名已在 rename_stl_files 内做了
        pass
    elif RUN_STL_RENAME:
        # 已有映射但仍需 STL 重命名的情况
        rename_stl_files()

    if RUN_POINT_RENAME:
        file_ops, dir_ops, skipped = build_point_operations(mapping)
        reported = False
        for reason, items in skipped.items():
            if not items:
                continue
            if not reported:
                print("以下未处理项：")
                reported = True
            print(f"- {reason}:")
            for item in items:
                print(f"    {item}")
        if reported:
            print("")

        if not file_ops and not dir_ops:
            print("点云重命名：没有可执行任务。")
        else:
            apply_point_operations(file_ops, dir_ops)

    if RUN_MERGE_STL:
        ops = build_merge_operations()
        if not ops:
            print("STL 合并：没有可执行任务。")
        else:
            apply_merge_operations(ops)


if __name__ == "__main__":
    main()

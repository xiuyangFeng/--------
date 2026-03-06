"""
Pipeline 工具模块。

保持惰性加载，避免仅查看 CLI 帮助时提前触发重依赖导入。
"""

__all__ = [
    "load_ascii_df",
    "save_csv",
    "load_bc_file",
    "load_boundary_conditions",
    "farthest_point_sampling",
    "random_sampling",
    "stratified_sampling_by_distance",
]


def __getattr__(name):
    if name in {
        "load_ascii_df",
        "save_csv",
        "load_bc_file",
        "load_boundary_conditions",
    }:
        from . import io as _io

        return getattr(_io, name)

    if name in {
        "farthest_point_sampling",
        "random_sampling",
        "stratified_sampling_by_distance",
    }:
        from . import sampling as _sampling

        return getattr(_sampling, name)

    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

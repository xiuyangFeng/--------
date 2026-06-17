# -*- coding: utf-8 -*-
"""单个病例过滤（--case）：支持叶目录名或相对 data_root 的路径。"""
from __future__ import annotations

from pathlib import Path


def case_dir_matches_query(case_dir: Path, data_root: Path, query: str) -> bool:
    """
    与 preprocess/run_all 等批量入口的 ``target_case`` 语义一致。

    - 若 ``query`` 含 ``/``（或 ``\\``）：按相对 ``data_root`` 的 POSIX 路径匹配（忽略大小写，
      空格规范为下划线）。用于 ILO（叶目录名为 before/after）及多数据源区分同名病例。
    - 否则：仅匹配病例目录叶节点名的规范化形式（连字符与下划线等价），兼容旧用法。
    """
    q = query.strip().replace("\\", "/")
    if not q:
        return False
    dr = data_root.resolve()
    cd = case_dir.resolve()
    try:
        rel = cd.relative_to(dr).as_posix()
    except ValueError:
        rel = cd.as_posix()
    rel_cmp = rel.replace(" ", "_").upper()

    if "/" in q:
        q_cmp = q.replace(" ", "_").strip().upper()
        return rel_cmp == q_cmp

    leaf_cmp = cd.name.replace(" ", "_").replace("-", "_").upper()
    q_leaf = q.replace(" ", "_").replace("-", "_").upper()
    return leaf_cmp == q_leaf

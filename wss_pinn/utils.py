"""WSS-PINN 通用工具：路径隔离、哈希校验、原子写盘、Git 元数据。

学习要点
--------
1. **写路径护栏** ``guard_write_path``：防止误写只读上游（CFD 原始数据 /
   WSS-min bundle），所有产物只能落在 ``wss_pinn`` / ``data_wss_pinn`` /
   ``outputs/wss_pinn``。
2. **SHA256**：复现实验时用文件/JSON 哈希锁定依赖，避免「悄悄改了上游」。
3. **原子写** ``atomic_write_json``：先写临时文件再 ``os.replace``，避免进程
   中断留下半截 JSON。
"""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


# 仓库根目录（wss_pinn/ 的上一级，即 Digital_twin/GNN）
ROOT = Path(__file__).resolve().parents[1]

# 只读上游：任何写操作若落在这些树下都会被拒绝
READ_ONLY_ROOTS = tuple(
    (ROOT / name).resolve()
    for name in ("data_new", "data_wss_min", "pipeline_wss_min", "training_wss_min")
)

# 允许写入的 PINN 专属根目录
ALLOWED_WRITE_ROOTS = tuple(
    (ROOT / name).resolve()
    for name in ("wss_pinn", "data_wss_pinn", "outputs/wss_pinn")
)


def utc_now() -> str:
    """返回 UTC 时间的 ISO-8601 字符串，用于 manifest / 日志时间戳。"""
    return datetime.now(timezone.utc).isoformat()


def sha256_file(path: str | Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    """分块计算文件 SHA256，避免大 CFD/npz 一次性读入内存。"""
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_json(value: Any) -> str:
    """对任意 JSON 可序列化对象做稳定哈希（键排序、紧凑分隔符）。

    配置 ``resolved_sha256`` 用此函数：同一内容无论 Python dict 插入顺序如何，
    哈希一致，便于对照实验是否「配置真的相同」。
    """
    payload = json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def is_relative_to(path: Path, parent: Path) -> bool:
    """判断 ``path`` 是否位于 ``parent`` 目录树内（兼容旧 Python 的 relative_to）。"""
    try:
        path.relative_to(parent)
        return True
    except ValueError:
        return False


def guard_write_path(path: str | Path) -> Path:
    """校验并返回规范化后的可写绝对路径。

    规则：
    - 相对路径相对仓库根解析；
    - 禁止写进只读上游；
    - 必须落在 ALLOWED_WRITE_ROOTS 之一。
    """
    resolved = Path(path).expanduser()
    if not resolved.is_absolute():
        resolved = ROOT / resolved
    resolved = resolved.resolve()
    if any(is_relative_to(resolved, root) for root in READ_ONLY_ROOTS):
        raise ValueError(f"refusing write inside read-only upstream: {resolved}")
    if not any(is_relative_to(resolved, root) for root in ALLOWED_WRITE_ROOTS):
        raise ValueError(f"write path is outside WSS-PINN roots: {resolved}")
    return resolved


def atomic_write_json(path: str | Path, payload: Any) -> Path:
    """原子写入 JSON：同目录临时文件 → ``os.replace`` 覆盖目标。

    这样即使写到一半崩溃，目标文件要么是旧完整版，要么还不存在，不会半截损坏。
    """
    target = guard_write_path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(
        prefix=f".{target.name}.", suffix=".tmp", dir=target.parent
    )
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, ensure_ascii=False, indent=2, sort_keys=True)
            handle.write("\n")
        os.replace(tmp_name, target)
    finally:
        # replace 成功后临时文件已不存在；失败时也尽量清理
        Path(tmp_name).unlink(missing_ok=True)
    return target


def git_state() -> dict[str, Any]:
    """采集当前仓库 commit / 分支 / 是否 dirty，写入 environment.json 便于复现。"""

    def run(*args: str) -> str:
        process = subprocess.run(
            ["git", *args], cwd=ROOT, text=True, capture_output=True, check=True
        )
        return process.stdout.strip()

    try:
        commit = run("rev-parse", "HEAD")
        branch = run("rev-parse", "--abbrev-ref", "HEAD")
        status = run("status", "--short")
        return {
            "commit": commit,
            "branch": branch,
            "dirty": bool(status),  # 有未提交改动则 dirty=True
            "status_short": status.splitlines(),
        }
    except Exception as exc:  # pragma: no cover - 防御性元数据路径
        return {"commit": "unknown", "branch": "unknown", "dirty": True, "error": str(exc)}

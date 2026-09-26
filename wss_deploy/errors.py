"""Typed pipeline errors (v0.14): what the user sees, whether a retry makes sense, what only admins see.

Before v0.14 only ``ValueError`` text reached the user; a VMTK failure, the 900 s VMTK timeout and a CUDA
out-of-memory all collapsed into "计算未完成，请重试". Stage code raises one of the classes below (or lets
``classify()`` map a foreign exception); the job manager stores ``to_record()`` in ``job["error"]`` and the
HTTP layer strips ``admin_detail`` for non-admin sessions.

    raise InputGeometryError("STL 有 3 个连通片，主体占比不足 …")          # user fixes the input, retry is pointless
    raise ToolchainError("中心线提取失败。", admin_detail=stderr_tail)     # VMTK / vessel_geom broke, admins see why
    raise ResourceError("GPU 显存不足。", retry_hint="cpu")                # transient; manager may retry on CPU

Plain ``ValueError`` keeps its old behaviour (message shown verbatim, not retryable) so existing raises still work.
"""
from __future__ import annotations

CATEGORIES = ("input_geometry", "toolchain", "resource", "internal")


class PipelineError(Exception):
    """Base class. ``user_message`` is safe to show; ``admin_detail`` may contain paths or tool stderr."""

    category = "internal"
    retryable = False

    def __init__(self, user_message: str, *, admin_detail: str | None = None, retryable: bool | None = None,
                 category: str | None = None, retry_hint: str | None = None):
        super().__init__(user_message)
        self.user_message = str(user_message)[:800]
        self.admin_detail = None if admin_detail is None else str(admin_detail)[:4000]
        if retryable is not None:
            self.retryable = bool(retryable)
        if category is not None:
            if category not in CATEGORIES:
                raise ValueError(f"unknown error category {category!r}")
            self.category = category
        self.retry_hint = retry_hint  # e.g. "cpu": the manager may retry once on the CPU

    def to_record(self, *, diagnostic_id: str) -> dict:
        """Serializable ``job["error"]`` entry. ``admin_detail`` is included; the HTTP layer strips it for non-admins."""
        record = {"message": self.user_message, "diagnostic_id": diagnostic_id,
                  "category": self.category, "retryable": self.retryable}
        if self.admin_detail:
            record["admin_detail"] = self.admin_detail
        if self.retry_hint:
            record["retry_hint"] = self.retry_hint
        return record


class InputGeometryError(PipelineError):
    """The input surface itself is the problem (degenerate STL, no openings, unit check failed …)."""
    category = "input_geometry"
    retryable = False


class ToolchainError(PipelineError):
    """An external tool or the installed environment failed (VMTK, vessel_geom, missing module, timeout)."""
    category = "toolchain"
    retryable = False


class ResourceError(PipelineError):
    """Transient resource exhaustion (CUDA OOM, disk full, worker limit). Worth retrying, possibly on CPU."""
    category = "resource"
    retryable = True


def classify(error: BaseException) -> PipelineError | None:
    """Map a foreign exception to a typed error, or return None when it should keep the generic path.

    Recognised: torch CUDA out-of-memory (→ ResourceError with retry_hint "cpu"), ``subprocess.TimeoutExpired``
    (→ ToolchainError), ``MemoryError`` / ``OSError(ENOSPC)`` (→ ResourceError), ``ImportError`` (→ ToolchainError).
    """
    if isinstance(error, PipelineError):
        return error
    name = type(error).__name__
    text = str(error)
    if name == "OutOfMemoryError" or ("CUDA" in text and "out of memory" in text.lower()):
        return ResourceError("GPU 显存不足，已改用 CPU 重试。", admin_detail=text[:2000], retry_hint="cpu")
    if isinstance(error, MemoryError):
        return ResourceError("内存不足，请稍后重试。", admin_detail=text[:2000])
    if isinstance(error, OSError) and getattr(error, "errno", None) == 28:  # ENOSPC
        return ResourceError("磁盘空间不足，请联系维护者清理后重试。", admin_detail=text[:2000])
    if name == "TimeoutExpired":
        return ToolchainError("中心线提取超时。请检查网格是否异常大或含大量碎片。", admin_detail=text[:2000])
    if isinstance(error, ImportError):
        return ToolchainError("计算环境缺少必要模块，请联系维护者。", admin_detail=text[:2000])
    return None

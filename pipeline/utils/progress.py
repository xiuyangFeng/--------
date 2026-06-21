#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Pipeline 进度日志工具。

支持两类日志：
1. 病例级日志: <case_dir>/processed/logs/progress.log
2. 批量/总流程日志: <data_root>/pipeline_reports/logs/*.log
"""

from __future__ import annotations

import io
import sys
from contextlib import contextmanager, redirect_stderr, redirect_stdout
from datetime import datetime
from pathlib import Path
from typing import Iterator


def _timestamp() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


class _TimestampedTee(io.TextIOBase):
    """将输出同时写到终端和日志文件；日志文件按行补时间戳。"""

    def __init__(self, terminal, log_fp, prefix: str):
        self._terminal = terminal
        self._log_fp = log_fp
        self._prefix = prefix
        self._buffer = ""

    def write(self, text: str) -> int:
        if not text:
            return 0

        self._terminal.write(text)

        self._buffer += text
        while "\n" in self._buffer:
            line, self._buffer = self._buffer.split("\n", 1)
            self._log_fp.write(f"[{_timestamp()}] {self._prefix}{line}\n")
        self._log_fp.flush()
        return len(text)

    def flush(self) -> None:
        self._terminal.flush()
        if self._buffer:
            self._log_fp.write(f"[{_timestamp()}] {self._prefix}{self._buffer}\n")
            self._buffer = ""
        self._log_fp.flush()

    def isatty(self) -> bool:
        return bool(getattr(self._terminal, "isatty", lambda: False)())


@contextmanager
def case_progress_logging(case_dir: Path, step_name: str) -> Iterator[Path]:
    """
    将当前代码块内的 stdout/stderr 追加写入病例级 progress.log。

    日志位置:
      <case_dir>/processed/logs/progress.log
    """
    case_dir = Path(case_dir)
    log_dir = case_dir / "processed" / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / "progress.log"

    with open(log_path, "a", encoding="utf-8") as log_fp:
        header = f"\n{'=' * 24} {_timestamp()} | {step_name} {'=' * 24}\n"
        log_fp.write(header)
        log_fp.flush()

        tee_out = _TimestampedTee(sys.stdout, log_fp, f"[{step_name}] ")
        tee_err = _TimestampedTee(sys.stderr, log_fp, f"[{step_name}][stderr] ")
        with redirect_stdout(tee_out), redirect_stderr(tee_err):
            try:
                yield log_path
            finally:
                tee_out.flush()
                tee_err.flush()


@contextmanager
def batch_progress_logging(data_root: Path, log_name: str, stage_name: str) -> Iterator[Path]:
    """
    将批量处理/总流程输出写入 data_root 下的 pipeline_reports/logs。

    参数:
      data_root: 数据根目录
      log_name: 日志文件名，例如 run_all.log / step1_preprocess_batch.log
      stage_name: 日志前缀，例如 run_all / step1_preprocess_batch
    """
    data_root = Path(data_root)
    log_dir = data_root / "pipeline_reports" / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / log_name

    with open(log_path, "a", encoding="utf-8") as log_fp:
        header = f"\n{'=' * 24} {_timestamp()} | {stage_name} {'=' * 24}\n"
        log_fp.write(header)
        log_fp.flush()

        tee_out = _TimestampedTee(sys.stdout, log_fp, f"[{stage_name}] ")
        tee_err = _TimestampedTee(sys.stderr, log_fp, f"[{stage_name}][stderr] ")
        with redirect_stdout(tee_out), redirect_stderr(tee_err):
            try:
                yield log_path
            finally:
                tee_out.flush()
                tee_err.flush()

from __future__ import annotations

import json
import random
from datetime import datetime
from pathlib import Path
from typing import Any, Dict

import numpy as np
import torch


def set_seed(seed: int, deterministic: bool = True) -> None:
    # 所有入口统一调这个函数，保证训练、评估、预测使用同一随机性策略。
    # 固定 Python 标准库随机数种子。
    random.seed(seed)
    # 固定 NumPy 随机数种子。
    np.random.seed(seed)
    # 固定 PyTorch CPU 侧随机数种子。
    torch.manual_seed(seed)
    # 固定 PyTorch 所有 CUDA 设备上的随机数种子。
    torch.cuda.manual_seed_all(seed)
    # 如果要求严格可复现，则关闭 cuDNN 的非确定性优化。
    if deterministic:
        # 强制 cuDNN 走确定性实现。
        torch.backends.cudnn.deterministic = True
        # 关闭 benchmark，避免根据输入动态选算子导致结果漂移。
        torch.backends.cudnn.benchmark = False


def resolve_device(device: str) -> torch.device:
    # auto 是本地开发和服务器共用的最稳默认值。
    # auto 时优先选择 CUDA，否则回退到 CPU。
    if device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # 其他情况直接按配置字符串构造 torch.device。
    return torch.device(device)


def timestamp() -> str:
    # 统一输出“年月日_时分秒”格式，便于拼接到实验目录名中。
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def ensure_dir(path: str | Path) -> Path:
    # 先把传入路径统一转成 Path 对象。
    path = Path(path)
    # 递归创建目录；已存在时不报错。
    path.mkdir(parents=True, exist_ok=True)
    # 返回 Path，方便调用方继续拼路径。
    return path


def dump_json(data: Dict[str, Any], path: str | Path) -> None:
    # 项目里多数中间产物都会被人直接打开看，因此统一使用 UTF-8 + pretty json。
    # 以写入模式打开目标 JSON 文件。
    with open(path, "w", encoding="utf-8") as f:
        # 保留中文、使用 2 空格缩进，便于人工检查。
        json.dump(data, f, ensure_ascii=False, indent=2)

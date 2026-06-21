from __future__ import annotations

import csv
from pathlib import Path
from typing import Dict, Iterable, List

import torch

from .utils import ensure_dir


def save_checkpoint(model: torch.nn.Module, path: str | Path) -> None:
    # 只保存参数字典，不保存整个模型对象，便于跨环境加载。
    torch.save(model.state_dict(), path)


def load_checkpoint(model: torch.nn.Module, path: str | Path, device: torch.device) -> None:
    # 按当前设备把 checkpoint 读到内存里。
    state = torch.load(path, map_location=device)
    # 将参数加载回模型。
    model.load_state_dict(state)


_ENCODER_PREFIXES = ("in_proj.", "blocks.", "shared_decoder.")


def load_pretrained_encoder(model: torch.nn.Module, path: str | Path, device: torch.device) -> Dict[str, int]:
    """加载 SSL encoder ckpt 至 FieldPointNeXt backbone；field_head / wss_head 保持随机初始化。"""
    state = torch.load(path, map_location=device)
    model_state = model.state_dict()
    filtered = {
        k: v for k, v in state.items()
        if k.startswith(_ENCODER_PREFIXES) and k in model_state and model_state[k].shape == v.shape
    }
    if not filtered:
        raise ValueError(f"pretrained_encoder 无匹配 backbone 参数: {path}")
    model_state.update(filtered)
    model.load_state_dict(model_state)
    return {"loaded": len(filtered), "skipped": len(state) - len(filtered)}


def append_experiment_index(
    output_root: str | Path,
    row: Dict[str, object],
    fieldnames: Iterable[str],
) -> None:
    # 确保输出根目录存在。
    output_root = ensure_dir(output_root)
    # 实验索引统一记录到 output_root 下的 experiment_index.csv。
    csv_path = Path(output_root) / "experiment_index.csv"
    # fieldnames 可能是生成器，这里先实体化成 list。
    requested_fields = list(fieldnames)

    # 如果索引文件还不存在，就创建并直接写入第一行。
    if not csv_path.exists():
        with open(csv_path, "w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=requested_fields)
            writer.writeheader()
            writer.writerow(row)
        return

    # 旧索引已存在时，先整表读出来。
    with open(csv_path, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        existing_fields = reader.fieldnames or []
        existing_rows = list(reader)

    # 实验索引字段会随着项目推进逐渐变多。
    # 这里选择“读旧表 -> 合并列 -> 重写整表”，保证新增列不会把旧索引写坏。
    # 先从旧字段开始，保持原有列顺序。
    merged_fields = existing_fields.copy()
    # 再把本次新增字段追加到表头末尾。
    for field in requested_fields:
        if field not in merged_fields:
            merged_fields.append(field)

    # 把当前实验结果按完整列集合补齐后追加到末尾。
    existing_rows.append({key: row.get(key, "") for key in merged_fields})
    normalized_rows = []
    # 所有旧行也按新列集合补齐，避免重写后缺列。
    for old_row in existing_rows:
        normalized_rows.append({key: old_row.get(key, "") for key in merged_fields})

    # 用合并后的列定义重写整个 CSV。
    with open(csv_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=merged_fields)
        writer.writeheader()
        writer.writerows(normalized_rows)


def sanitize_batch_metadata(values) -> List[str]:
    # PyG batch 后，字符串元信息有时会保留成单值，有时会变成 list/tuple。
    # 导出脚本统一走这个函数，把两种情况收口成 List[str]。
    # 如果已经是 list / tuple，就逐个转成字符串。
    if isinstance(values, (list, tuple)):
        return [str(v) for v in values]
    # 单值情况则包装成只含一个元素的列表。
    return [str(values)]

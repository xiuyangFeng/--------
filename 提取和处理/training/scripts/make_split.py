from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Dict, List, Tuple


def load_cases(path: str | Path) -> List[str]:
    path = Path(path)
    if path.suffix.lower() == ".json":
        with open(path, "r", encoding="utf-8") as f:
            raw = json.load(f)
        if isinstance(raw, dict) and "cases" in raw:
            return [str(x) for x in raw["cases"]]
        if isinstance(raw, list):
            return [str(x) for x in raw]
        raise ValueError("JSON 文件必须是病例列表，或包含 'cases' 字段。")

    with open(path, "r", encoding="utf-8") as f:
        return [line.strip() for line in f if line.strip()]


def _domain_prefix(case_rel: str) -> str:
    parts = tuple(Path(case_rel.replace("\\", "/")).parts)
    if not parts:
        raise ValueError(f"空的病例路径: {case_rel!r}")
    seg = parts[0].upper()
    if seg not in {"AAA", "AG", "ILO"}:
        raise ValueError(
            f"病例路径首段必须为 AAA/AG/ILO（data_new 顶层域），不符: {case_rel!r}"
        )
    return seg


def _sizes_for_shard(
    n_total: int,
    train_ratio: float,
    val_ratio: float,
) -> Tuple[int, int, int]:
    """单域内与原 make_split 一致的 train/val/test 计数口径。"""
    n_train = max(1, int(n_total * train_ratio))
    n_val = max(1, int(n_total * val_ratio))
    if n_train + n_val >= n_total:
        n_val = max(1, n_total - n_train - 1)
    n_test = n_total - n_train - n_val
    if n_test < 1:
        raise ValueError(f"无法在 n={n_total} 下切分 train/val/test，请减小 val 占比")
    return n_train, n_val, n_test


def _shard_train_val_test(
    domain: str,
    cases: List[str],
    rng: random.Random,
    train_ratio: float,
    val_ratio: float,
) -> Tuple[List[str], List[str], List[str], List[str]]:
    ws: List[str] = []
    c = sorted(cases)
    rng.shuffle(c)
    n = len(c)
    if n == 0:
        return [], [], [], ws
    if n == 1:
        ws.append(f"域 {domain}：仅 1 例，全部归入 train（该域无比 val/test）")
        return c, [], [], ws
    if n == 2:
        ws.append(f"域 {domain}：仅 2 例，切为 1 train / 1 val（该域无 test）")
        return [c[0]], [c[1]], [], ws

    nt, nv, ntst = _sizes_for_shard(n, train_ratio, val_ratio)
    ws.append(
        "域 %s：n=%d → train=%d val=%d test=%d（套全局 train/val 比例推导 test）"
        % (domain, n, nt, nv, ntst)
    )
    return c[:nt], c[nt : nt + nv], c[nt + nv : nt + nv + ntst], ws


def _stratify_domain_splits(
    cases: List[str],
    seed: int,
    train_ratio: float,
    val_ratio: float,
    test_ratio: float,
) -> Tuple[List[str], List[str], List[str], dict, List[str]]:
    """AAA / AG / ILO 分层：逐域按比例切分，再以 AAA→AG→ILO 合并（各子表字典序）。"""
    abs_total_ratio = train_ratio + val_ratio + test_ratio
    if abs(abs_total_ratio - 1.0) > 1e-6:
        raise ValueError("train/val/test 比例之和必须为 1.0")

    uniq = sorted(set(cases))
    buckets = {"AAA": [], "AG": [], "ILO": []}
    for rel in uniq:
        buckets[_domain_prefix(rel)].append(rel)

    trains: List[str] = []
    vals: List[str] = []
    tests: List[str] = []
    slog: List[str] = []
    break_down: Dict[str, Dict[str, int]] = {}

    for i, dom in enumerate(("AAA", "AG", "ILO")):
        g = buckets[dom]
        if not g:
            slog.append(f"域 `{dom}` 无病例（已跳过）。")
            continue
        rng_d = random.Random(seed * 1103515245 + i * 9973 + 246934587)
        tr, va, te, lw = _shard_train_val_test(
            dom,
            list(g),
            rng_d,
            train_ratio,
            val_ratio,
        )
        slog.extend(lw)
        trains.extend(sorted(tr))
        vals.extend(sorted(va))
        tests.extend(sorted(te))
        break_down[dom] = {
            "n_total": len(g),
            "train": len(tr),
            "val": len(va),
            "test": len(te),
        }

    merged = trains + vals + tests
    if len(merged) != len(uniq):
        raise RuntimeError("分层合并后病例数与原集合不符")

    if not vals or not tests:
        raise RuntimeError(
            "分层后 val 或 test 为空。常为某域仅 1～2 例；请改用非分层或调整名单。"
        )
    return trains, vals, tests, break_down, slog


def main() -> None:
    parser = argparse.ArgumentParser(description="根据病例名单生成患者级 split 文件")
    parser.add_argument("--cases-file", required=True, help="病例名单 txt/json 文件")
    parser.add_argument("--output", required=True, help="输出 split.json 路径")
    parser.add_argument("--split-version", required=True, help="划分版本号，如 split_v1")
    parser.add_argument("--source", default="", help="数据源标识，如 AG/fast")
    parser.add_argument("--seed", type=int, default=1, help="随机种子")
    parser.add_argument("--train-ratio", type=float, default=0.7, help="训练集比例")
    parser.add_argument("--val-ratio", type=float, default=0.1, help="验证集比例")
    parser.add_argument(
        "--test-ratio",
        type=float,
        default=0.2,
        help="测试集比例，默认与前两者互补",
    )
    parser.add_argument(
        "--stratify-by-domain",
        action="store_true",
        help="按路径首域 AAA / AG / ILO 分层：每域套用相同 train/val（test 由剩余推导）后再合并",
    )
    args = parser.parse_args()

    total_ratio = args.train_ratio + args.val_ratio + args.test_ratio
    if abs(total_ratio - 1.0) > 1e-6:
        raise ValueError("train/val/test 比例之和必须为 1.0")

    cases = load_cases(args.cases_file)
    if len(set(cases)) < 3:
        raise ValueError("去重后病例数至少需要 3 个")

    rng = random.Random(args.seed)

    if args.stratify_by_domain:
        trains, vals, tests, bk, slog = _stratify_domain_splits(
            cases,
            args.seed,
            args.train_ratio,
            args.val_ratio,
            args.test_ratio,
        )
        meta = {"stratify_by_domain": True, "counts_per_domain": bk, "logs": slog}
        split_notes = (
            "由 training/scripts/make_split.py 自动生成。"
            "**按 AAA/AG/ILO 域分层**：每域套用相同全局 train/val 比例；随机种子="
            + str(args.seed)
            + "。"
        )
    else:
        c = sorted(set(cases))
        rng.shuffle(c)
        n_total = len(c)
        n_train = max(1, int(n_total * args.train_ratio))
        n_val = max(1, int(n_total * args.val_ratio))
        if n_train + n_val >= n_total:
            n_val = max(1, n_total - n_train - 1)
        n_test = n_total - n_train - n_val
        if n_test < 1:
            raise ValueError("测试集病例数不足，请调整比例。")
        trains = c[:n_train]
        vals = c[n_train : n_train + n_val]
        tests = c[n_train + n_val :]
        meta = {"stratify_by_domain": False}
        split_notes = (
            "由 training/scripts/make_split.py 自动生成。全局随机.shuffle；请核对是否满足分层需求。"
        )

    split_dict = {
        "split_version": args.split_version,
        "source": args.source or "data_new:AAA+AG+ILO",
        "train_cases": trains,
        "val_cases": vals,
        "test_cases": tests,
        "notes": split_notes,
        "split_meta": meta,
    }

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with open(output, "w", encoding="utf-8") as f:
        json.dump(split_dict, f, ensure_ascii=False, indent=2)

    print(f"已生成 split 文件: {output}")
    print(
        f"train={len(trains)}, val={len(vals)}, test={len(tests)} "
        f"（ stratify-by-domain={'on' if args.stratify_by_domain else 'off'}）"
    )
    if args.stratify_by_domain:
        for line in meta.get("logs", []):
            print(line)


if __name__ == "__main__":
    main()

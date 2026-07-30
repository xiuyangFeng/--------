"""
批量验证：velocity → WSS 算子（默认绑定 WSS-PINN 已通过审计的 173 例）
======================================================================
**默认不再扫描整个 data_new。** 全库导出有大量损坏/缺列/拓扑异常病例；
正式对比只允许使用 WSS-PINN 冻结 split：

  wss_pinn/configs/splits/
    split_WSS_PINN_AG_AAA_ILO_q2v_pool2025_train138_test35_exclude_SHI_YUN_XI_v1.json
  SHA256: 964d7021f2d12baadd630e7b936456a4e62294fa72c5ada4fc70abe4d9361f2b
  计数：train138 + test35 = 173（已排除 SHI_YUN_XI）

用法
----
  # 正式：PINN 173 例全量（峰值步）
  python batch_validate_cfd.py --json-out ../../outputs/wss_from_velocity_pinn173.json

  # 快速：每角色抽几例（仍来自该 split，不是 data_new 全库）
  python batch_validate_cfd.py --limit-per-cohort 3

  # 仅 test35
  python batch_validate_cfd.py --roles test

  # 仅当明确需要“扫描全库找坏导出”时才开（非正式精度结论）
  python batch_validate_cfd.py --discover-all --limit-per-cohort 5
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import traceback
from pathlib import Path

import numpy as np

from calculate_wss_cfd import run_case

ROOT = Path("/public/newhome/cy/Digital_twin/GNN")
DATA_ROOT = ROOT / "data_new"
DEFAULT_SPLIT = (
    ROOT
    / "wss_pinn/configs/splits"
    / "split_WSS_PINN_AG_AAA_ILO_q2v_pool2025_train138_test35_exclude_SHI_YUN_XI_v1.json"
)
EXPECTED_SPLIT_SHA256 = (
    "964d7021f2d12baadd630e7b936456a4e62294fa72c5ada4fc70abe4d9361f2b"
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_split_cases(
    split_path: Path,
    data_root: Path,
    roles: set[str],
) -> list[dict]:
    """
    读取 PINN split，返回可跑病例列表。

    每项含：canonical_id / cohort / role / case_dir。
    case_dir 指向 data_new/<canonical_id>。
    """
    payload = json.loads(split_path.read_text(encoding="utf-8"))
    cases: list[dict] = []
    for role, key in (("train", "train_cases"), ("test", "test_cases"), ("val", "val_cases")):
        if role not in roles:
            continue
        for canonical_id in payload.get(key, []):
            parts = str(canonical_id).split("/")
            if len(parts) < 3 or parts[0] not in {"AG", "AAA", "ILO"}:
                raise ValueError(f"invalid canonical case ID: {canonical_id}")
            case_dir = data_root / canonical_id
            cases.append(
                {
                    "canonical_id": str(canonical_id),
                    "cohort": parts[0],
                    "role": role,
                    "case_dir": case_dir,
                    "label": str(canonical_id),
                }
            )
    if not cases:
        raise ValueError(f"split 在 roles={sorted(roles)} 下为空：{split_path}")
    return cases


def discover(root: Path) -> list[dict]:
    """
    扫描 data_new 全部可算目录（仅 --discover-all 使用）。

    这会混入审计未通过的坏导出，**不能**作为正式 raw R² 结论的病例池。
    """
    cases = []
    for cohort in ("AG", "AAA", "ILO"):
        base = root / cohort
        if not base.is_dir():
            continue
        for depth in ("*/*", "*"):
            for candidate in sorted(base.glob(depth)):
                wall, interior = candidate / "ascii", candidate / "ascii_in"
                if not (wall.is_dir() and interior.is_dir()):
                    continue
                if not any(wall.iterdir()) or not any(interior.iterdir()):
                    continue
                cases.append(
                    {
                        "canonical_id": str(candidate.relative_to(root)),
                        "cohort": cohort,
                        "role": "unspecified",
                        "case_dir": candidate,
                        "label": str(candidate.relative_to(root)),
                    }
                )
    seen, unique = set(), []
    for case in cases:
        if case["canonical_id"] not in seen:
            seen.add(case["canonical_id"])
            unique.append(case)
    return unique


def summarize(values: list[float]) -> dict:
    array = np.asarray([v for v in values if np.isfinite(v)], dtype=float)
    if not len(array):
        return {"n": 0}
    return {
        "n": int(len(array)),
        "mean": float(array.mean()),
        "p05": float(np.percentile(array, 5)),
        "p50": float(np.median(array)),
        "p95": float(np.percentile(array, 95)),
        "min": float(array.min()),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    parser.add_argument("--data-root", type=str, default=str(DATA_ROOT))
    parser.add_argument(
        "--split",
        type=str,
        default=str(DEFAULT_SPLIT),
        help="WSS-PINN 冻结 split JSON（默认 train138/test35 exclude SHI）",
    )
    parser.add_argument(
        "--roles",
        type=str,
        default="train,test",
        help="使用 split 中的哪些角色，逗号分隔：train,test,val",
    )
    parser.add_argument(
        "--discover-all",
        action="store_true",
        help="忽略 split，扫描整个 data_new（非正式；仅用于找坏导出）",
    )
    parser.add_argument(
        "--limit-per-cohort",
        type=int,
        default=0,
        help="每队列最多几例；0=不截断（正式跑应用 0）",
    )
    parser.add_argument("--sample-count", type=int, default=1200)
    parser.add_argument("--neighbors", type=int, default=64)
    parser.add_argument("--degree", type=int, default=2)
    parser.add_argument("--viscosity", choices=["carreau", "newton"], default="carreau")
    parser.add_argument(
        "--normals",
        choices=["pca", "stl"],
        default="pca",
        help="法向来源：pca=壁面点云局部PCA（默认）；stl=最近STL面片法向",
    )
    parser.add_argument(
        "--no-peak",
        action="store_true",
        help="禁用 bundle peak_step，改用 ascii 第一帧",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--json-out", type=str, default=None)
    args = parser.parse_args()

    root = Path(args.data_root)
    roles = {item.strip() for item in args.roles.split(",") if item.strip()}
    split_meta = None

    if args.discover_all:
        print(
            "⚠ --discover-all：正在扫描整个 data_new，结果不能当作正式 PINN 口径精度。",
            file=sys.stderr,
        )
        cases = discover(root)
        source = "discover_all_data_new"
    else:
        split_path = Path(args.split).resolve()
        if not split_path.is_file():
            raise FileNotFoundError(f"split 不存在：{split_path}")
        split_sha = sha256_file(split_path)
        if split_path == DEFAULT_SPLIT.resolve() and split_sha != EXPECTED_SPLIT_SHA256:
            raise RuntimeError(
                "默认 PINN split 的 SHA256 与冻结值不一致："
                f"got {split_sha}, expected {EXPECTED_SPLIT_SHA256}"
            )
        cases = load_split_cases(split_path, root, roles)
        split_meta = {
            "path": str(split_path),
            "sha256": split_sha,
            "roles": sorted(roles),
            "n_cases": len(cases),
            "expected_sha256_default": EXPECTED_SPLIT_SHA256,
        }
        source = "wss_pinn_split"

    if args.limit_per_cohort > 0:
        kept, counts = [], {}
        for case in cases:
            cohort = case["cohort"]
            if counts.get(cohort, 0) < args.limit_per_cohort:
                kept.append(case)
                counts[cohort] = counts.get(cohort, 0) + 1
        cases = kept

    print(
        f"病例来源={source}  n={len(cases)}  "
        f"K={args.neighbors} deg={args.degree} visc={args.viscosity} "
        f"peak={not args.no_peak}\n"
    )
    if split_meta:
        print(f"split: {split_meta['path']}")
        print(f"sha256: {split_meta['sha256']}\n")

    header = f"{'case':<44} {'role':>5} {'cover':>6} {'rawR2':>8} {'sclR2':>7} {'spear':>7} {'alpha':>6} {'cos':>6}"
    print(header)
    print("-" * len(header))

    rows, failures = [], []
    for case in cases:
        label = case["label"]
        try:
            report = run_case(
                case["case_dir"],
                None,
                args.sample_count,
                args.neighbors,
                args.degree,
                args.viscosity,
                False,
                args.seed,
                args.normals,
                prefer_peak=not args.no_peak,
            )
        except Exception as error:  # 单例失败不中断全批
            failures.append({"case": label, "error": f"{type(error).__name__}: {error}"})
            print(f"{label:<44} FAILED {type(error).__name__}")
            traceback.print_exc(limit=1)
            continue
        report["cohort"] = case["cohort"]
        report["role"] = case["role"]
        report["label"] = label
        report["canonical_id"] = case["canonical_id"]
        rows.append(report)
        flag = "  ⚠ 真值损坏" if not report["truth_usable"] else ""
        print(
            f"{label:<44} {case['role']:>5} {report['coverage']:>6.2f} {report['raw_r2']:>8.3f} "
            f"{report['scaled_r2']:>7.3f} {report['spearman']:>7.3f} "
            f"{report['alpha']:>6.2f} {report['direction_cosine_p50']:>6.3f}{flag}"
        )

    if not rows:
        print("\n无成功病例。")
        return

    usable = [r for r in rows if r["truth_usable"]]
    excluded = [r for r in rows if not r["truth_usable"]]

    print(f"\n=== 汇总（{len(usable)} 例可用，排除 {len(excluded)} 例真值损坏）===")
    overall = {}
    for metric in ("raw_r2", "scaled_r2", "spearman", "alpha", "direction_cosine_p50", "coverage"):
        stats = summarize([r[metric] for r in usable])
        overall[metric] = stats
        print(
            f"{metric:<22} mean={stats['mean']:>7.3f}  p05={stats['p05']:>7.3f}  "
            f"p50={stats['p50']:>7.3f}  p95={stats['p95']:>7.3f}  min={stats['min']:>7.3f}"
        )

    per_cohort = {}
    print()
    for cohort in sorted({r["cohort"] for r in usable}):
        subset = [r for r in usable if r["cohort"] == cohort]
        stats = summarize([r["raw_r2"] for r in subset])
        per_cohort[cohort] = {"n_cases": len(subset), "raw_r2": stats}
        print(
            f"{cohort:<6} n={len(subset):<4} raw_R² mean={stats['mean']:.3f} "
            f"p05={stats['p05']:.3f} min={stats['min']:.3f}"
        )

    if excluded:
        print(f"\n{len(excluded)} 例真值损坏（已排除，非算子问题）：")
        for row in excluded:
            print(f"  {row['label']}: {'; '.join(row['data_warnings'])}")

    if failures:
        print(f"\n{len(failures)} 例读取失败：")
        for failure in failures[:10]:
            print(f"  {failure['case']}: {failure['error']}")

    if args.json_out:
        out = Path(args.json_out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(
            json.dumps(
                {
                    "source": source,
                    "split": split_meta,
                    "config": {
                        "neighbors": args.neighbors,
                        "degree": args.degree,
                        "viscosity": args.viscosity,
                        "normals": args.normals,
                        "sample_count": args.sample_count,
                        "prefer_peak": not args.no_peak,
                        "seed": args.seed,
                        "roles": sorted(roles),
                    },
                    "n_cases": len(rows),
                    "n_usable": len(usable),
                    "overall": overall,
                    "per_cohort": per_cohort,
                    "cases": rows,
                    "excluded_corrupt_truth": [r["label"] for r in excluded],
                    "failures": failures,
                },
                indent=2,
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )
        print(f"\n写入 {out}")


if __name__ == "__main__":
    main()

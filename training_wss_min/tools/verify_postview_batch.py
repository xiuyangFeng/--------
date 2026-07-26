#!/usr/bin/env python3
"""Verify a complete canonical PostView batch without changing run artifacts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from training_wss_min import dataset as D
from training_wss_min.tools.export_wss_postview import _existing_complete_manifest


def verify(split_path: Path, partition: str, output_dir: Path) -> dict:
    expected = [f"{cohort}/{case}" for cohort, case in D.load_split_cases(split_path, partition)]
    rows = []
    errors = []
    seen = set()
    for label in expected:
        canonical = D.canonical_unit_id(label)
        found = _existing_complete_manifest(output_dir, canonical)
        if found is None:
            errors.append(f"missing_or_incomplete_manifest:{canonical}")
            continue
        manifest, case_dir = found
        actual = D.canonical_unit_id(manifest.get("case", canonical))
        if actual != canonical:
            errors.append(f"case_mismatch:{canonical}:{actual}")
        if actual in seen:
            errors.append(f"duplicate_case:{actual}")
        seen.add(actual)
        counts = {
            "vtp": len(list(case_dir.rglob("*.vtp"))),
            "png": len(list(case_dir.rglob("*.png"))),
            "csv": len(list(case_dir.rglob("*.csv"))),
        }
        required = {"vtp": 4, "png": 3, "csv": 3}
        for kind, minimum in required.items():
            if counts[kind] < minimum:
                errors.append(f"too_few_{kind}:{canonical}:{counts[kind]}<{minimum}")
        coverage = (manifest.get("mapping_coverage") or {}).get("valid_ratio")
        if coverage is None or float(coverage) < 0.999999:
            errors.append(f"mapping_coverage:{canonical}:{coverage}")
        rows.append({"case": canonical, "case_dir": str(case_dir), **counts,
                     "mapping_valid_ratio": coverage})

    batch_path = output_dir / "batch_manifest.json"
    if not batch_path.is_file():
        errors.append("missing_batch_manifest")
    else:
        batch = json.loads(batch_path.read_text(encoding="utf-8"))
        batch_cases = [D.canonical_unit_id(x["case"]) for x in batch.get("cases", [])]
        if batch_cases != [D.canonical_unit_id(x) for x in expected]:
            errors.append("batch_manifest_case_order_or_membership_mismatch")

    report = {
        "status": "passed" if not errors else "failed",
        "partition": partition,
        "expected_cases": len(expected),
        "verified_cases": len(rows),
        "errors": errors,
        "cases": rows,
    }
    if errors:
        raise RuntimeError(json.dumps(report, ensure_ascii=False, indent=2))
    return report


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", required=True, type=Path)
    ap.add_argument("--partition", default="test", choices=("train", "val", "test"))
    ap.add_argument("--output-dir", required=True, type=Path)
    ap.add_argument("--report", type=Path)
    args = ap.parse_args()
    report = verify(args.split, args.partition, args.output_dir)
    text = json.dumps(report, ensure_ascii=False, indent=2) + "\n"
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(text, encoding="utf-8")
    print(text, end="")


if __name__ == "__main__":
    main()

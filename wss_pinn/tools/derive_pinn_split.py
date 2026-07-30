from __future__ import annotations

import argparse
import json
from copy import deepcopy
from pathlib import Path
from typing import Any

from wss_pinn.utils import (
    ROOT,
    atomic_write_json,
    guard_write_path,
    sha256_file,
    utc_now,
)


PINN_SPLIT_ROOT = (ROOT / "wss_pinn/configs/splits").resolve()


def _stratum(case_id: str) -> str:
    parts = case_id.split("/")
    if parts[0] == "AG":
        return "AG"
    if parts[0] == "AAA" and len(parts) >= 2:
        return f"AAA_{parts[1]}"
    if parts[0] == "ILO" and len(parts) >= 2:
        phase = parts[1].rsplit("-", 1)[-1]
        return f"ILO_{phase}"
    raise ValueError(f"unsupported canonical case ID: {case_id}")


def _counts(payload: dict[str, Any]) -> dict[str, int]:
    output = {
        "train": len(payload["train_cases"]),
        "val": len(payload.get("val_cases", [])),
        "test": len(payload["test_cases"]),
    }
    for role in ("train", "test"):
        for case_id in payload[f"{role}_cases"]:
            key = f"{role}_{_stratum(case_id)}"
            output[key] = output.get(key, 0) + 1
    return output


def derive_split(
    parent: dict[str, Any],
    *,
    parent_path: Path,
    remove_test_case: str,
    split_version: str,
    reason: str,
) -> dict[str, Any]:
    if remove_test_case in parent.get("train_cases", []):
        raise ValueError("refusing to remove a training case")
    if remove_test_case not in parent.get("test_cases", []):
        raise ValueError("requested removal is not present in parent test_cases")
    result = deepcopy(parent)
    result["created_at"] = utc_now()
    result["pipeline"] = "wss_pinn"
    result["split_version"] = split_version
    result["protocol"] = (
        "PINN-specific derived development screen; preserve parent train138 "
        "and remove one test case with unusable volume-field supervision"
    )
    result["algorithm"] = (
        f"{parent.get('algorithm', '')}; user-approved PINN-only exclusion"
    )
    result["source_parent_split"] = str(parent_path.resolve())
    result["source_parent_split_sha256"] = sha256_file(parent_path)
    result["baseline_parent_immutable"] = True
    result["approval"] = {
        "approved_by_user": True,
        "approved_at": utc_now(),
        "scope": "WSS-PINN route only; baseline_wss split must not change",
    }
    result["test_cases"] = [
        case_id
        for case_id in parent["test_cases"]
        if case_id != remove_test_case
    ]
    result["unused_cases"] = sorted(
        set(parent.get("unused_cases", [])) | {remove_test_case}
    )
    result["excluded_cases"] = [
        *parent.get("excluded_cases", []),
        {
            "canonical_id": remove_test_case,
            "previous_role": "test",
            "reason": reason,
            "scope": "wss_pinn",
        },
    ]
    if "test_quotas" in result:
        quotas = deepcopy(result["test_quotas"])
        quota_key = _stratum(remove_test_case).replace("_", "/", 1)
        if quota_key not in quotas:
            raise KeyError(f"missing parent test quota: {quota_key}")
        quotas[quota_key] = int(quotas[quota_key]) - 1
        result["test_quotas"] = quotas
    result["counts"] = _counts(result)
    result["expected_counts"] = {
        "train": result["counts"]["train"],
        "val": result["counts"]["val"],
        "test": result["counts"]["test"],
    }
    if result["counts"]["train"] != len(parent["train_cases"]):
        raise AssertionError("training membership changed")
    if result["counts"]["test"] != len(parent["test_cases"]) - 1:
        raise AssertionError("test removal count is not exactly one")
    if set(result["train_cases"]).intersection(result["test_cases"]):
        raise AssertionError("train/test overlap introduced")
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--parent", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--remove-test-case", required=True)
    parser.add_argument("--split-version", required=True)
    parser.add_argument("--reason", required=True)
    args = parser.parse_args()

    parent_path = Path(args.parent).resolve()
    output_path = guard_write_path(args.output)
    if output_path.parent.resolve() != PINN_SPLIT_ROOT:
        raise ValueError(f"PINN split output must be directly under {PINN_SPLIT_ROOT}")
    if output_path.exists():
        raise FileExistsError(output_path)
    parent = json.loads(parent_path.read_text(encoding="utf-8"))
    result = derive_split(
        parent,
        parent_path=parent_path,
        remove_test_case=args.remove_test_case,
        split_version=args.split_version,
        reason=args.reason,
    )
    written = atomic_write_json(output_path, result)
    print(
        json.dumps(
            {
                "status": "completed",
                "output": str(written),
                "sha256": sha256_file(written),
                "parent": str(parent_path),
                "parent_sha256": sha256_file(parent_path),
                "counts": result["counts"],
                "removed_test_case": args.remove_test_case,
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()

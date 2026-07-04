#!/usr/bin/env python3
"""汇总 V3P 平台期 Ideas A–G oracle JSON，判 Go/No-Go 并写 GPU 优先级。"""
from __future__ import annotations

import argparse
import json
from datetime import date
from pathlib import Path
from typing import Any, Dict, List, Mapping

from ._figure_utils import load_json, save_json
from .run_v3_f0_decision import REPO_ROOT

GPU_MAP = {
    "B": "V3P-J1-StreamCrossRay-WSS",
    "C/D": "V3P-J2-BranchScaleMode-WSS",
    "E": "V3P-J3-SpectralRegularizedWall-WSS",
    "G": "V3P-J4-V3Dlite-Pretrain-AGFinetune",
}


def _layers_pass(block: Mapping[str, Any]) -> int:
    layers = block.get("layers") or {}
    n = 0
    for _k, v in layers.items():
        if not isinstance(v, dict):
            continue
        val = v.get("value")
        if val is True or (isinstance(val, (int, float)) and val > 0):
            n += 1
        elif isinstance(val, float) and val > 0.05:
            n += 1
    return n


def summarize(bundle_path: Path) -> Dict[str, Any]:
    bundle = load_json(bundle_path)
    ideas = bundle.get("ideas", {})
    go_list: List[Dict[str, Any]] = []
    weak_list: List[Dict[str, Any]] = []
    no_go_list: List[Dict[str, Any]] = []

    for idea_id, block in ideas.items():
        v = block.get("verdict", "unknown")
        entry = {
            "idea": idea_id,
            "verdict": v,
            "layers_hit": _layers_pass(block),
            "gpu_mapping": block.get("gpu_mapping") or GPU_MAP.get(idea_id),
        }
        if v == "go":
            go_list.append(entry)
        elif v == "weak_no_go":
            weak_list.append(entry)
        else:
            no_go_list.append(entry)

    gpu_candidates = [
        e for e in go_list + weak_list
        if e.get("gpu_mapping") and e["layers_hit"] >= 2
    ][:2]

    all_no_go = len(go_list) == 0 and len(weak_list) == 0
    return {
        "label": "v3p_platform_oracle_summary",
        "date": date.today().isoformat(),
        "source_bundle": str(bundle_path),
        "n_go": len(go_list),
        "n_weak_no_go": len(weak_list),
        "n_no_go": len(no_go_list),
        "go_list": go_list,
        "weak_no_go_list": weak_list,
        "no_go_list": no_go_list,
        "gpu_priority": gpu_candidates,
        "gate_rule": "oracle 须解释 L1/L2/L3 至少两层才开 GPU；最多 1–2 个 J* seed1 短训",
        "next_hop": (
            "G5 叙事收口 + 平台期阶段结论（横向 WSS 可能不可学）"
            if all_no_go
            else f"立项 GPU: {[g['gpu_mapping'] for g in gpu_candidates]}"
        ),
        "m_e_gpu_line": "封口 unless gpu_priority non-empty and seed1 Go",
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--bundle",
        type=Path,
        default=REPO_ROOT / "outputs/field/f0_decision" / f"v3p_platform_oracle_bundle_{date.today():%Y%m%d}.json",
    )
    ap.add_argument("--output", type=Path, default=None)
    args = ap.parse_args()
    bundle_path = args.bundle.resolve()
    summary = summarize(bundle_path)
    out = (
        args.output.resolve()
        if args.output is not None
        else bundle_path.parent / f"v3p_platform_oracle_summary_{date.today():%Y%m%d}.json"
    )
    save_json(out, summary)
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()

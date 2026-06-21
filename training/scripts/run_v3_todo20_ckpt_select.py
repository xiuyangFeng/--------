#!/usr/bin/env python3
"""V3P TODO-20 · 权重 EMA / 多 checkpoint 平均 离线选优（0 重训）。

对一个 save_best_only=false 的诊断 run（含 checkpoint_epoch_*.pt）：
  - 构造候选权重：单点参考（best_wss / best_model / last）、权重 EMA（多档 α）、尾部均值（多档 K）
  - 在 **val** 上用 wss_r2_wss 选最优候选
  - 选中的候选在 **test** 上只报一次，与 best_wss（母版口径）对照求 Δ

口径纪律：所有候选用同一 _eval_wss_r2 重算（含 best_wss / best_model），保证 Δ 内部一致；
该值可能与 summary.json 官方 wss_r2_wss 略有差异，仅用候选间相对比较，不替代母版。

产物：``outputs/field/f0_decision/v3p_todo20_ckpt_select_<date>.json``
"""
from __future__ import annotations

import argparse
import json
from datetime import date
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import torch

from ..core.io import load_checkpoint
from .run_v3_f0_decision import REPO_ROOT, _r2_score, _safe_json_float
from .run_v3_i7_ckpt_probe import (
    _build_eval_context,
    _discover_periodic_ckpts,
    _read_history,
    _history_summary,
)

_IS_WALL_IDX = 9
_GO_DELTA = 0.005  # 相对 best_wss 的 test wss_r2_wss 提升门槛
_MAG_NOGO_DELTA = -0.005  # magnitude 退化硬防线（严于通用 ±0.005）


@torch.no_grad()
def _eval_wss_r2(model: torch.nn.Module, loader, device: torch.device) -> float:
    """wss_r2_wss（壁面节点 dim0 |WSS| 的聚合 R²），所有候选同口径重算。"""
    model.eval()
    preds: List[torch.Tensor] = []
    tgts: List[torch.Tensor] = []
    for batch in loader:
        batch = batch.to(device)
        out = model(batch)
        wss_pred = out[1] if isinstance(out, tuple) else None
        if wss_pred is None:
            continue
        wss_target = getattr(batch, "y_wss", None)
        if wss_target is None:
            continue
        wall = batch.x[:, _IS_WALL_IDX] > 0.5
        if wall.sum() == 0:
            continue
        preds.append(wss_pred[wall, 0].detach().float().cpu())
        tgts.append(wss_target[wall, 0].detach().float().cpu())
    if not preds:
        return float("nan")
    y_pred = torch.cat(preds).numpy()
    y_true = torch.cat(tgts).numpy()
    return _r2_score(y_true, y_pred)


def _avg_state(paths: Sequence[Path], device: torch.device, ema_alpha: Optional[float] = None) -> Dict[str, torch.Tensor]:
    """对一组 checkpoint（按传入顺序 = epoch 升序）做权重平均或 EMA。

    ema_alpha=None：等权平均；否则 w_ema = a*w_ema + (1-a)*w（a=ema_alpha，越大越偏向晚期）。
    仅平均浮点张量；整型 buffer 取最后一个。
    """
    states = [torch.load(p, map_location=device) for p in paths]
    keys = states[0].keys()
    out: Dict[str, torch.Tensor] = {}
    for k in keys:
        if not torch.is_floating_point(states[0][k]):
            out[k] = states[-1][k].clone()
            continue
        if ema_alpha is None:
            acc = torch.zeros_like(states[0][k], dtype=torch.float32)
            for s in states:
                acc += s[k].float()
            out[k] = (acc / len(states)).to(states[0][k].dtype)
        else:
            acc = states[0][k].float().clone()
            for s in states[1:]:
                acc = ema_alpha * acc + (1.0 - ema_alpha) * s[k].float()
            out[k] = acc.to(states[0][k].dtype)
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description="V3P TODO-20 ckpt EMA/averaging selection")
    ap.add_argument("--run-dir", type=Path, required=True, help="含 checkpoint_epoch_*.pt 的诊断 run 目录")
    ap.add_argument("--output", type=Path, default=None)
    ap.add_argument("--ema-alphas", type=float, nargs="*", default=[0.5, 0.7, 0.8, 0.9])
    ap.add_argument("--tail-ks", type=int, nargs="*", default=[2, 3, 5])
    args = ap.parse_args()

    run_dir = args.run_dir.resolve()
    config, model, device, _loader = _build_eval_context(run_dir)
    val_loader = _loader("val")
    test_loader = _loader("test")

    hist = _history_summary(_read_history(run_dir))
    periodic = _discover_periodic_ckpts(run_dir)
    periodic_paths = [p for _, p in periodic]

    # 候选集合：单点参考 + EMA + 尾部均值。
    candidates: List[Dict[str, Any]] = []

    def _add_single(name: str, path: Path) -> None:
        if path.is_file():
            candidates.append({"name": name, "kind": "single", "paths": [path], "ema_alpha": None})

    _add_single("best_wss_model", run_dir / "best_wss_model.pt")
    _add_single("best_model", run_dir / "best_model.pt")
    _add_single("last_model", run_dir / "last_model.pt")

    if len(periodic_paths) >= 2:
        for a in args.ema_alphas:
            candidates.append(
                {"name": f"ema_a{a}", "kind": "ema", "paths": periodic_paths, "ema_alpha": float(a)}
            )
        for k in args.tail_ks:
            if k <= len(periodic_paths):
                candidates.append(
                    {"name": f"tail_avg_k{k}", "kind": "tail_avg", "paths": periodic_paths[-k:], "ema_alpha": None}
                )

    results: List[Dict[str, Any]] = []
    for cand in candidates:
        if cand["kind"] == "single":
            load_checkpoint(model, cand["paths"][0], device)
        else:
            state = _avg_state(cand["paths"], device, ema_alpha=cand["ema_alpha"])
            model.load_state_dict(state)
        val_r2 = _eval_wss_r2(model, val_loader, device)
        results.append(
            {
                "name": cand["name"],
                "kind": cand["kind"],
                "n_ckpts": len(cand["paths"]),
                "ema_alpha": cand["ema_alpha"],
                "val_wss_r2_wss": _safe_json_float(val_r2),
                "test_wss_r2_wss": None,
            }
        )

    # 基线（母版口径）：best_wss 的 test（同口径重算）。
    baseline = next((r for r in results if r["name"] == "best_wss_model"), None)
    if baseline is not None:
        load_checkpoint(model, run_dir / "best_wss_model.pt", device)
        baseline["test_wss_r2_wss"] = _safe_json_float(_eval_wss_r2(model, test_loader, device))

    # 选优：只在「平均类」候选（ema / tail_avg）里按 val 选最优；single 仅作参考。
    avg_results = [r for r in results if r["kind"] in {"ema", "tail_avg"} and r["val_wss_r2_wss"] == r["val_wss_r2_wss"]]
    selected = max(avg_results, key=lambda r: r["val_wss_r2_wss"]) if avg_results else None

    if selected is not None:
        cand = next(c for c in candidates if c["name"] == selected["name"])
        state = _avg_state(cand["paths"], device, ema_alpha=cand["ema_alpha"])
        model.load_state_dict(state)
        selected["test_wss_r2_wss"] = _safe_json_float(_eval_wss_r2(model, test_loader, device))

    # 判读。
    judge: Dict[str, Any] = {"go": False, "reason": "无平均类候选（周期 checkpoint 不足）"}
    if selected is not None and baseline is not None:
        base_test = baseline["test_wss_r2_wss"]
        sel_test = selected["test_wss_r2_wss"]
        if isinstance(base_test, (int, float)) and isinstance(sel_test, (int, float)):
            delta = sel_test - base_test
            if delta <= _MAG_NOGO_DELTA:
                go, reason = False, f"magnitude 退化 {delta:+.4f} ≤ {_MAG_NOGO_DELTA}：硬防线 No-Go"
            elif delta >= _GO_DELTA:
                go, reason = True, f"test Δ {delta:+.4f} ≥ {_GO_DELTA}：选优有救（观察 Go，需补 seed）"
            else:
                go, reason = False, f"test Δ {delta:+.4f} 在 ±{_GO_DELTA} 内：选优无显著收益 No-Go"
            judge = {
                "go": go,
                "reason": reason,
                "selected": selected["name"],
                "selected_test_wss_r2_wss": sel_test,
                "baseline_best_wss_test_wss_r2_wss": base_test,
                "delta_test": _safe_json_float(delta),
            }

    report = {
        "label": "V3P-TODO20-ckpt-select",
        "date": date.today().isoformat(),
        "motivation": "Q0 收尾：权重 EMA / 多 ckpt 平均能否在 best_wss 上抬升 test wss_r2_wss",
        "run_dir": str(run_dir),
        "exp_id": config.meta.exp_id,
        "n_periodic_ckpts": len(periodic_paths),
        "history": hist,
        "metric_note": "wss_r2_wss = 壁面 dim0 |WSS| 聚合 R²，所有候选同口径重算；仅候选间相对比较。",
        "candidates": results,
        "judge": judge,
    }

    out = args.output or (
        REPO_ROOT / "outputs/field/f0_decision" / f"v3p_todo20_ckpt_select_{date.today().strftime('%Y%m%d')}.json"
    )
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(judge, indent=2, ensure_ascii=False))
    print(out)


if __name__ == "__main__":
    main()

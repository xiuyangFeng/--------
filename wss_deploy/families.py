"""Model families: everything that differs between kinds of deployable models lives here.

A *family* is the adapter between a frozen model release and the orchestration in
``pipeline``/``registry``/``infer``.  Those modules know nothing about WSS versus
pressure/velocity; they ask the family for

* ``contract(info)``       – validate ``release.json`` and return the narrow deployment contract;
* ``model_specs(...)``      – resolve which checkpoints to load for a seed subset;
* ``verify_model(...)``     – extra per-checkpoint configuration checks;
* ``predict(release, case)`` – run the ensemble and restore physical units;
* ``stage_b(...)``          – build inputs, predict, summarise and export for one job.

Adding a family (multi-frame, new fields) means registering one more ``ModelFamily``;
no ``if target == ...`` branch is needed elsewhere.
"""
from __future__ import annotations

from dataclasses import dataclass
import json
import math
from pathlib import Path
import time
from typing import Any, Callable

import numpy as np

from .input_memo import shared_inputs
from .knn_memo import shared_knn


class ReleaseError(ValueError):
    """A release is missing, unsafe, or outside the supported contract."""


PEAK_FRAME = {"index": 0, "step": 1162, "time_s": 0.21, "label": "peak_systole"}
LEGACY_WSS_RELEASE = "X5D_v51_5seed_20260916"
LEGACY_WSS_SEEDS = (1234, 7, 2025, 11, 2026)

VOLUME_FEATURES = [
    "x", "y", "z", "abscissa_norm", "local_radius", "curvature", "log_local_radius",
    "rho", "theta_sin", "theta_cos", "dr_ds", "dist_to_junction_mm",
    "dist_to_endpoint_mm", "end_zone", "nx_aligned", "ny_aligned", "nz_aligned",
    "dist_to_wall_mm", "log_q_branch_murray", "log_tau0_murray",
]
VOLUME_FIELDS = {
    "pressure": {"units": "Pa", "location": "wall_and_interior", "kind": "scalar",
                 "components": 1, "reference": "volume_mean_relative"},
    "velocity": {"units": "m/s", "location": "interior", "kind": "vector",
                 "components": 3, "frame": "world"},
}


# ----------------------------------------------------------------------------- shared helpers
def _declared_models(info: dict[str, Any]) -> list[dict[str, Any]] | None:
    """Normalise ``release.json`` ``models``/``model_runs`` to a list of ``{"seed", "path", ["field"]}``."""
    declared = info.get("models") or info.get("model_runs")
    if not declared:
        return None
    if isinstance(declared, dict):
        declared = [{"seed": seed, "path": path} for seed, path in declared.items()]
    if not isinstance(declared, list):
        raise ValueError("release.json 的 models 必须是列表或映射。")
    specs = []
    for index, item in enumerate(declared):
        if isinstance(item, str):
            item = {"path": item, "seed": index}
        if not isinstance(item, dict) or not item.get("path"):
            raise ValueError(f"release.json 的 models[{index}] 缺少 path。")
        rel = Path(str(item["path"]))
        if rel.is_absolute() or ".." in rel.parts:
            raise ValueError(f"release.json 的模型路径必须位于发布包内：{rel}")
        specs.append({"seed": item.get("seed", index), "path": rel.as_posix(),
                      **({"field": item["field"]} if "field" in item else {})})
    return specs


def _check_seed_count(seed_count, available: int) -> None:
    if type(seed_count) is not int or not 1 <= seed_count <= available:
        raise ValueError("seed_count 超出发布包模型数量。")


# ----------------------------------------------------------------------------- wall WSS family
def _wss_contract(info: dict[str, Any]) -> dict[str, Any]:
    target = str(info.get("target", "")).lower()
    # A legacy single-frame WSS description mentions the source cardiac
    # ``cycle``; that word alone does not make the output multi-frame.  Reject
    # only unambiguous target contracts here.
    forbidden = ("velocity", "pressure", "multi-frame", "multiframe", "time series")
    if any(word in target for word in forbidden):
        raise ReleaseError("发布包合同不支持：当前仅支持单帧 WSS 推理。")
    models = info.get("models") or info.get("model_runs")
    if models is not None and not isinstance(models, (list, dict)):
        raise ReleaseError("发布包合同不支持：models 必须是列表或映射。")
    # A release can omit an explicit axis for legacy WSS packages; infer the
    # existing fixed peak frame.  Explicit multi-frame declarations fail closed.
    axis = info.get("time_axis")
    if axis is None and info.get("release") == LEGACY_WSS_RELEASE:
        axis = [dict(PEAK_FRAME)]
    if not isinstance(axis, list) or len(axis) != 1 or not isinstance(axis[0], dict):
        raise ReleaseError("发布包必须声明一个峰值 time_axis 帧。")
    frame = axis[0]
    if frame.get("step") != PEAK_FRAME["step"] or frame.get("time_s") != PEAK_FRAME["time_s"] or frame.get("index", 0) != 0:
        raise ReleaseError("当前仅支持 step 1162 / 0.21 s 峰值帧。")
    fields = info.get("fields") or {"wss": {"units": "Pa", "location": "wall", "kind": "scalar"}}
    if not isinstance(fields, dict) or set(fields) != {"wss"} or not isinstance(fields["wss"], dict):
        raise ReleaseError("当前仅支持壁面标量 WSS 字段。")
    field = fields["wss"]
    if field.get("units") != "Pa" or field.get("location") != "wall" or field.get("kind") != "scalar" or field.get("components", 1) != 1:
        raise ReleaseError("WSS 字段必须为 wall / scalar / Pa。")
    return {"target": info.get("target"), "time_axis": axis, "fields": fields, "protocol": "single_frame_wss",
            "family": "wall_wss_v1"}


def _wss_model_specs(info: dict[str, Any], contract: dict[str, Any], seeds, seed_count) -> list[dict[str, Any]]:
    specs = _declared_models(info)
    if specs is None:
        specs = [{"seed": seed, "path": f"models/X5D_v51_s{seed}"} for seed in seeds]
    if seed_count is not None:
        _check_seed_count(seed_count, len(specs))
        specs = specs[:seed_count]
    return specs


def _wss_model_paths(info: dict[str, Any]) -> list[str]:
    specs = _declared_models(info)
    return [s["path"] for s in specs] if specs else [f"models/X5D_v51_s{s}" for s in LEGACY_WSS_SEEDS]


def _wss_verify_model(run: Path, cfg: dict, spec: dict | None) -> None:
    data = cfg.get("data", {})
    if data.get("target", "wss") != "wss" or data.get("timesteps", "peak") != "peak":
        raise ReleaseError("模型配置不是当前支持的峰值单帧 WSS。")


def _wss_predict(release, case: dict) -> dict:
    import torch
    from training_wss_min import dataset as D, evaluate as E
    preds, per = [], []
    # Identical kNN graphs (knn_memo.py) and identical support / query / patch inputs (input_memo.py)
    # are built once per case; every member still runs its own forward pass.
    with shared_knn() as knn_stats, shared_inputs() as input_stats:
        for m in release.models:
            t = time.perf_counter()
            with torch.no_grad():
                p = E.predict_case_norm(m["model"], case, m["cfg"].data.input_features, m["feat_stats"], release.device, cfg=m["cfg"])
            if release.device == "cuda":
                torch.cuda.synchronize()
            per.append(time.perf_counter() - t)
            preds.append(D.denormalize_wss(np.asarray(p, dtype=np.float64), m["stats"]))
    P = np.stack(preds)
    return {"wss_pa": P.mean(axis=0), "seed_pred_pa": P, "seed_sd_pa": P.std(axis=0), "seconds_per_model": per, "knn_reuse": dict(knn_stats),
            "input_reuse": dict(input_stats), "device": release.device,
            "gpu": torch.cuda.get_device_name(0) if release.device == "cuda" else "cpu"}


def _wss_stage_b(job_dir, mapping, release, **kwargs) -> dict:
    from .pipeline import stage_b_wall
    return stage_b_wall(job_dir, mapping, release, **kwargs)


# ----------------------------------------------------------------------------- PF6/VF6 volume family
def _volume_contract(info: dict[str, Any]) -> dict[str, Any]:
    """Explicit PF6/VF6 peak contracts; never infer units or a pressure gauge."""
    expected = ({"pressure", "velocity"} if info["target"] == "pressure_velocity"
                else {"pressure" if info["target"] == "pressure_mixed" else "velocity"})
    family = {"pressure_velocity": "PF6_VF6", "pressure_mixed": "PF6", "velocity": "VF6"}[info["target"]]
    if info.get("model_family") != family:
        raise ReleaseError("体场发布包必须明确声明 PF6/VF6 模型族。")
    fields = info.get("fields")
    if not isinstance(fields, dict) or set(fields) != expected:
        raise ReleaseError("体场 fields 与目标合同不一致。")
    for name in expected:
        if not isinstance(fields[name], dict) or any(fields[name].get(k) != v for k, v in VOLUME_FIELDS[name].items()):
            raise ReleaseError(f"{name} 字段缺少正确单位、位置、参考压力或向量坐标架。")
    axis = info.get("time_axis")
    if (not isinstance(axis, list) or len(axis) != 1 or not isinstance(axis[0], dict)
            or axis[0].get("step") != PEAK_FRAME["step"] or axis[0].get("time_s") != PEAK_FRAME["time_s"] or axis[0].get("index") != 0):
        raise ReleaseError("体场仅支持 step 1162 / 0.21 s 峰值帧。")
    models = info.get("models")
    if not isinstance(models, list) or not models:
        raise ReleaseError("体场发布包必须显式列出 models、field 和 seed。")
    seeds = {name: [] for name in expected}
    paths = set()
    for spec in models:
        if (not isinstance(spec, dict) or spec.get("field") not in expected
                or type(spec.get("seed")) is not int or not isinstance(spec.get("path"), str)):
            raise ReleaseError("体场模型必须显式声明 field、整数 seed 和 path。")
        path = Path(spec["path"])
        if (path.is_absolute() or ".." in path.parts or not path.parts or path.parts[0] != "models"
                or spec["path"] in paths or spec["seed"] in seeds[spec["field"]]):
            raise ReleaseError("体场模型路径或 seed 重复/无效。")
        paths.add(spec["path"])
        seeds[spec["field"]].append(spec["seed"])
    first = next(iter(seeds.values()))
    if not first or any(s != first for s in seeds.values()):
        raise ReleaseError("压力与速度集成必须声明相同顺序的 seed。")
    return {"target": info["target"], "time_axis": axis, "fields": fields,
            "protocol": "single_frame_volume", "model_family": family,
            "seeds": first, "pressure_reference": "p - volume-weighted interior mean at peak",
            "family": "pf6_vf6_volume_v1"}


def _validate_volume_model(run: Path, cfg: dict, spec: dict) -> None:
    data, model = cfg.get("data", {}), cfg.get("model", {})
    target = "pressure_mixed" if spec["field"] == "pressure" else "velocity"
    if (data.get("target") != target or data.get("timesteps") != "peak"
            or data.get("target_normalization") != "global_stats"
            or data.get("required_frame_version") != "v5_atlas_frame_v1"
            or data.get("input_features") != VOLUME_FEATURES
            or not cfg.get("eval", {}).get("fixed_support")
            or model.get("out_dim") != (1 if target == "pressure_mixed" else 3)
            or cfg.get("train", {}).get("seed") != spec["seed"]):
        raise ReleaseError("模型配置不满足 PF6/VF6 峰值体场、特征和归一化合同。")
    normalization = json.loads((run / "target_normalization.json").read_text(encoding="utf-8"))
    if normalization.get("mode") != "global_stats" or normalization.get("physical_recovery_enabled") is not True:
        raise ReleaseError("体场发布包不能恢复物理单位。")
    stats = json.loads((run / "wss_global_stats.json").read_text(encoding="utf-8"))
    if (stats.get("method") != "linear" or stats.get("required_frame_version") != "v5_atlas_frame_v1"
            or stats.get("peak_step") != 1162 or stats.get("timesteps_scope") != "peak"):
        raise ReleaseError("体场统计文件不是已核验的单帧线性物理量合同。")
    key = "linear" if target == "pressure_mixed" else "velocity"
    block = stats.get(key, {})
    means = [block.get("mean")] if key == "linear" else block.get("mean")
    stds = [block.get("std")] if key == "linear" else block.get("std")
    count = 1 if key == "linear" else 3
    if (not isinstance(means, list) or not isinstance(stds, list) or len(means) != count or len(stds) != count
            or any(not isinstance(v, (float, int)) or not math.isfinite(v) for v in means + stds)
            or any(v <= 0 for v in stds)):
        raise ReleaseError("体场反归一化统计无效。")


def _volume_model_specs(info: dict[str, Any], contract: dict[str, Any], seeds, seed_count) -> list[dict[str, Any]]:
    specs = _declared_models(info)
    if not specs:
        raise ValueError("体场发布包必须显式列出 models。")
    if seed_count is not None:
        _check_seed_count(seed_count, len(contract["seeds"]))
        selected = contract["seeds"][:seed_count]
        specs = [s for s in specs if s["seed"] in selected]
    return specs


def _volume_model_paths(info: dict[str, Any]) -> list[str]:
    return [s["path"] for s in (_declared_models(info) or [])]


def _volume_verify_model(run: Path, cfg: dict, spec: dict | None) -> None:
    if spec is None:
        raise ReleaseError("体场模型必须在 release.json 中显式声明。")
    _validate_volume_model(run, cfg, spec)


def _volume_predict(release, case: dict) -> dict:
    """Wall-supported pressure on all rows, velocity on interior rows only.

    Pressure retains the training gauge (relative to the unknown peak
    volume mean); no reference pressure is read, guessed or added.
    Velocity is rotated back to the uploaded STL's world axes, in m/s.
    """
    import torch
    from training_wss_min import evaluate as E
    n = len(case["pos"])
    n_wall = case.get("n_wall")
    if not isinstance(n_wall, (int, np.integer)) or not 0 < n_wall < n:
        raise ValueError("体场 case 必须包含壁面在前、内部在后的 n_wall。")
    support = np.asarray(case.get("support_pool"))
    if not np.array_equal(support, np.arange(n_wall)):
        raise ValueError("体场支持点必须严格来自壁面。")
    rotation = np.asarray(case.get("frame_rotation"), dtype=np.float64)
    if (rotation.shape != (3, 3) or not np.isfinite(rotation).all()
            or not np.allclose(rotation @ rotation.T, np.eye(3), atol=1e-6)
            or not np.isclose(np.linalg.det(rotation), 1.0, atol=1e-6)):
        raise ValueError("体场需要有效的 anatomical-to-world 正交坐标变换。")
    predictions = {field: [] for field in release.contract["fields"]}
    per = []
    # One query view per field, shared by that field's members, so the input memo can recognise the case.
    views = {True: dict(case, query_pool=np.arange(n)), False: dict(case, query_pool=np.arange(n_wall, n))}
    with shared_knn() as knn_stats, shared_inputs() as input_stats:   # knn_memo.py / input_memo.py
        for m in release.models:
            pressure = m["field"] == "pressure"
            view = views[pressure]
            query = view["query_pool"]
            start = time.perf_counter()
            with torch.no_grad():
                normalized = E.predict_case_norm(m["model"], view, m["cfg"].data.input_features,
                                                m["feat_stats"], release.device, cfg=m["cfg"],
                                                return_all_channels=not pressure)
            if release.device == "cuda":
                torch.cuda.synchronize()
            per.append(time.perf_counter() - start)
            normalized = np.asarray(normalized, dtype=np.float64)
            expected = (len(query),) if pressure else (len(query), 3)
            if normalized.shape != expected or not np.isfinite(normalized).all():
                raise ValueError(f"{m['field']} 模型输出形状或数值不符合发布合同。")
            physical = E.denormalize_volume_prediction(normalized, m["cfg"].data.target, m["stats"])
            if not pressure:
                physical = physical @ rotation
            if not np.isfinite(physical).all():
                raise ValueError("体场反归一化产生非有限物理量。")
            predictions[m["field"]].append(physical)
    result = {"seconds_per_model": per, "knn_reuse": dict(knn_stats), "input_reuse": dict(input_stats), "device": release.device,
              "gpu": torch.cuda.get_device_name(0) if release.device == "cuda" else "cpu",
              "n_wall": int(n_wall), "n_interior": int(n - n_wall),
              "pressure_reference": "volume_mean_relative", "velocity_frame": "world"}
    if "pressure" in predictions:
        values = np.stack(predictions["pressure"])
        result.update(pressure_pa=values.mean(axis=0), seed_pressure_pa=values,
                      pressure_seed_sd_pa=values.std(axis=0))
    if "velocity" in predictions:
        values = np.stack(predictions["velocity"])
        mean = values.mean(axis=0)
        result.update(velocity_m_s=mean, speed_m_s=np.linalg.norm(mean, axis=1),
                      seed_velocity_m_s=values, velocity_seed_sd_m_s=values.std(axis=0))
    return result


def _volume_stage_b(job_dir, mapping, release, **kwargs) -> dict:
    from .volume_pipeline import stage_b_volume
    return stage_b_volume(job_dir, mapping, release, **kwargs)


# ----------------------------------------------------------------------------- three-head wall family: peak WSS + TAWSS + OSI (M1)
CYCLE_MULTI_TARGET = "wss_cycle_multi"
CYCLE_MULTI_FAMILY = "M1_3head"
CYCLE_CHANNELS = ("wss", "tawss", "osi")
CYCLE_FIELDS = {
    "wss": {"units": "Pa", "location": "wall", "kind": "scalar", "components": 1, "frame": "peak_systole"},
    "tawss": {"units": "Pa", "location": "wall", "kind": "scalar", "components": 1, "frame": "cycle_average"},
    "osi": {"units": "1", "location": "wall", "kind": "scalar", "components": 1, "frame": "cycle_average", "range": [0.0, 0.5]},
}
CYCLE_STATS_METHODS = {"wss": "log_z", "tawss": "log_z", "osi": "logit_z"}


def _cycle_contract(info: dict[str, Any]) -> dict[str, Any]:
    """Explicit three-head contract: one forward pass yields peak WSS, TAWSS and OSI on the wall."""
    if str(info.get("target", "")).lower() != CYCLE_MULTI_TARGET:
        raise ReleaseError("三头发布包 target 必须为 wss_cycle_multi。")
    if info.get("model_family") != CYCLE_MULTI_FAMILY:
        raise ReleaseError("三头发布包必须明确声明 model_family = M1_3head。")
    fields = info.get("fields")
    if not isinstance(fields, dict) or set(fields) != set(CYCLE_CHANNELS):
        raise ReleaseError("三头发布包 fields 必须恰为 wss / tawss / osi。")
    for name, expected in CYCLE_FIELDS.items():
        if not isinstance(fields[name], dict) or any(fields[name].get(k) != v for k, v in expected.items()):
            raise ReleaseError(f"{name} 字段缺少正确单位、位置、类型或时间口径。")
    axis = info.get("time_axis")
    if (not isinstance(axis, list) or len(axis) != 1 or not isinstance(axis[0], dict)
            or axis[0].get("step") != PEAK_FRAME["step"] or axis[0].get("time_s") != PEAK_FRAME["time_s"] or axis[0].get("index", 0) != 0):
        raise ReleaseError("三头发布包的峰值通道仅支持 step 1162 / 0.21 s 峰值帧。")
    cycle = info.get("cycle")
    if (not isinstance(cycle, dict) or not str(cycle.get("frames", "")).startswith("0-79")
            or cycle.get("period_s") != 0.8):
        raise ReleaseError("三头发布包必须声明周期积分定义（frames 0-79，period_s 0.8）。")
    models = info.get("models")
    if not isinstance(models, list) or not models:
        raise ReleaseError("三头发布包必须显式列出 models（seed 与 path）。")
    seeds, paths = [], set()
    for spec in models:
        if not isinstance(spec, dict) or type(spec.get("seed")) is not int or not isinstance(spec.get("path"), str):
            raise ReleaseError("三头模型必须显式声明整数 seed 和 path。")
        path = Path(spec["path"])
        if (path.is_absolute() or ".." in path.parts or not path.parts or path.parts[0] != "models"
                or spec["path"] in paths or spec["seed"] in seeds):
            raise ReleaseError("三头模型路径或 seed 重复/无效。")
        paths.add(spec["path"]); seeds.append(spec["seed"])
    return {"target": CYCLE_MULTI_TARGET, "time_axis": axis, "fields": fields, "protocol": "single_frame_wss_cycle_multi",
            "model_family": CYCLE_MULTI_FAMILY, "seeds": seeds, "channels": list(CYCLE_CHANNELS), "cycle": cycle,
            "family": "wall_cycle_multi_v1"}


def _cycle_model_specs(info: dict[str, Any], contract: dict[str, Any], seeds, seed_count) -> list[dict[str, Any]]:
    specs = _declared_models(info)
    if not specs:
        raise ValueError("三头发布包必须显式列出 models。")
    if seed_count is not None:
        _check_seed_count(seed_count, len(contract["seeds"]))
        selected = contract["seeds"][:seed_count]
        specs = [s for s in specs if s["seed"] in selected]
    # Every three-head model carries all three channels; tagging the spec makes the loader pass it to verify_model.
    return [{**s, "field": s.get("field", CYCLE_MULTI_TARGET)} for s in specs]


def _cycle_model_paths(info: dict[str, Any]) -> list[str]:
    return [s["path"] for s in (_declared_models(info) or [])]


def _cycle_verify_model(run: Path, cfg: dict, spec: dict | None) -> None:
    if spec is None:
        raise ReleaseError("三头模型必须在 release.json 中显式声明。")
    data, model = cfg.get("data", {}), cfg.get("model", {})
    if (data.get("target") != CYCLE_MULTI_TARGET or data.get("timesteps", "peak") != "peak"
            or data.get("target_normalization", "global_stats") != "global_stats"
            or data.get("required_frame_version") != "v5_atlas_frame_v1"
            or model.get("out_dim") != len(CYCLE_CHANNELS)
            or cfg.get("train", {}).get("seed") != spec["seed"]):
        raise ReleaseError("模型配置不满足三头峰值/周期壁面标量合同（target、out_dim 3、峰值帧、global_stats、seed）。")
    normalization = json.loads((run / "target_normalization.json").read_text(encoding="utf-8"))
    if normalization.get("mode") != "global_stats" or normalization.get("physical_recovery_enabled") is not True:
        raise ReleaseError("三头发布包不能恢复物理单位。")
    stats = json.loads((run / "wss_global_stats.json").read_text(encoding="utf-8"))
    if stats.get("method") != "multi" or list(stats.get("channels", [])) != list(CYCLE_CHANNELS):
        raise ReleaseError("三头统计文件必须是 method='multi' 且通道为 wss / tawss / osi。")
    for name in CYCLE_CHANNELS:
        sub = stats.get(name)
        if not isinstance(sub, dict) or sub.get("method") != CYCLE_STATS_METHODS[name]:
            raise ReleaseError(f"{name} 通道统计方法必须为 {CYCLE_STATS_METHODS[name]}。")
        block = sub.get("logit" if name == "osi" else "log", {})
        mean, std = block.get("mean"), block.get("std")
        if (not isinstance(mean, (int, float)) or not isinstance(std, (int, float)) or not math.isfinite(mean)
                or not math.isfinite(std) or std <= 0):
            raise ReleaseError(f"{name} 通道反归一化统计无效。")
        if name == "osi" and (block.get("scale") != 0.5 or not 0 < float(block.get("clip", 0)) < 0.5):
            raise ReleaseError("OSI 通道 logit 统计必须声明 scale 0.5 与裁剪常数。")


def _cycle_predict(release, case: dict) -> dict:
    """One forward pass per seed → (N, 3) normalised channels → physical units per channel → seed mean.

    Channel 0 keeps the legacy peak-WSS keys (``wss_pa`` / ``seed_pred_pa`` / ``seed_sd_pa``) so the wall
    stage runs unchanged; TAWSS and OSI travel in ``extra_fields`` for the generic extra-field exports.
    """
    import torch
    from training_wss_min import dataset as D, evaluate as E
    from .cycle_fields import CYCLE_DEFINITION, OSI_MAX
    n = len(case["pos"])
    channels = list(CYCLE_CHANNELS)
    preds, per = [], []
    with shared_knn() as knn_stats, shared_inputs() as input_stats:   # knn_memo.py / input_memo.py
        for m in release.models:
            stats = m["stats"]
            if stats.get("method") != "multi" or list(stats.get("channels", [])) != channels:
                raise ValueError("三头模型的统计文件不是 wss/tawss/osi 多通道合同。")
            t = time.perf_counter()
            with torch.no_grad():
                z = E.predict_case_norm(m["model"], case, m["cfg"].data.input_features, m["feat_stats"], release.device,
                                        cfg=m["cfg"], return_all_channels=True)
            if release.device == "cuda":
                torch.cuda.synchronize()
            per.append(time.perf_counter() - t)
            z = np.asarray(z, dtype=np.float64)
            if z.shape != (n, len(channels)) or not np.isfinite(z).all():
                raise ValueError("三头模型输出形状或数值不符合发布合同。")
            physical = np.asarray(D.denormalize_wss(z, stats), dtype=np.float64)
            physical[:, 0] = np.clip(physical[:, 0], 0.0, None)
            physical[:, 1] = np.clip(physical[:, 1], 0.0, None)
            physical[:, 2] = np.clip(physical[:, 2], 0.0, OSI_MAX)
            if not np.isfinite(physical).all():
                raise ValueError("三头反归一化产生非有限物理量。")
            preds.append(physical)
    P = np.stack(preds)                                    # (seeds, N, 3)
    wss, tawss, osi = P[:, :, 0], P[:, :, 1], P[:, :, 2]
    osi_mean = np.clip(osi.mean(axis=0), 0.0, OSI_MAX)
    result = {"wss_pa": wss.mean(axis=0), "seed_pred_pa": wss, "seed_sd_pa": wss.std(axis=0),
              "tawss_pa": tawss.mean(axis=0), "seed_tawss_pa": tawss, "tawss_seed_sd_pa": tawss.std(axis=0),
              "osi": osi_mean, "seed_osi": osi, "osi_seed_sd": osi.std(axis=0),
              "seconds_per_model": per, "knn_reuse": dict(knn_stats), "input_reuse": dict(input_stats),
              "device": release.device,
              "gpu": torch.cuda.get_device_name(0) if release.device == "cuda" else "cpu",
              "channels": channels, "cycle": dict(CYCLE_DEFINITION)}
    result["extra_fields"] = {"tawss": {"values": result["tawss_pa"], "seed_pred": tawss, "seed_sd": result["tawss_seed_sd_pa"]},
                              "osi": {"values": osi_mean, "seed_pred": osi, "seed_sd": result["osi_seed_sd"]}}
    return result


def _cycle_stage_b(job_dir, mapping, release, **kwargs) -> dict:
    from .pipeline import stage_b_wall
    return stage_b_wall(job_dir, mapping, release, **kwargs)


# ----------------------------------------------------------------------------- registry
@dataclass(frozen=True)
class ModelFamily:
    id: str
    protocol: str
    matches: Callable[[dict[str, Any]], bool]
    contract: Callable[[dict[str, Any]], dict[str, Any]]
    model_specs: Callable[[dict[str, Any], dict[str, Any], tuple, Any], list[dict[str, Any]]]
    model_paths: Callable[[dict[str, Any]], list[str]]
    verify_model: Callable[[Path, dict, dict | None], None]
    predict: Callable[[Any, dict], dict]
    stage_b: Callable[..., dict]


def _is_wss_target(info: dict[str, Any]) -> bool:
    target = str(info.get("target", "")).lower()
    return target in {"wss", "wss_magnitude"} or all(word in target for word in ("wall", "shear", "stress"))


WALL_WSS = ModelFamily(
    id="wall_wss_v1", protocol="single_frame_wss", matches=_is_wss_target, contract=_wss_contract,
    model_specs=_wss_model_specs, model_paths=_wss_model_paths, verify_model=_wss_verify_model,
    predict=_wss_predict, stage_b=_wss_stage_b,
)
PF6_VF6_VOLUME = ModelFamily(
    id="pf6_vf6_volume_v1", protocol="single_frame_volume",
    matches=lambda info: str(info.get("target", "")).lower() in {"pressure_velocity", "pressure_mixed", "velocity"},
    contract=_volume_contract, model_specs=_volume_model_specs, model_paths=_volume_model_paths,
    verify_model=_volume_verify_model, predict=_volume_predict, stage_b=_volume_stage_b,
)
WALL_CYCLE_MULTI = ModelFamily(
    id="wall_cycle_multi_v1", protocol="single_frame_wss_cycle_multi",
    matches=lambda info: str(info.get("target", "")).lower() == CYCLE_MULTI_TARGET,
    contract=_cycle_contract, model_specs=_cycle_model_specs, model_paths=_cycle_model_paths,
    verify_model=_cycle_verify_model, predict=_cycle_predict, stage_b=_cycle_stage_b,
)
FAMILIES: dict[str, ModelFamily] = {WALL_WSS.id: WALL_WSS, PF6_VF6_VOLUME.id: PF6_VF6_VOLUME, WALL_CYCLE_MULTI.id: WALL_CYCLE_MULTI}


def family_for_info(info: dict[str, Any]) -> ModelFamily:
    """Select the family from ``release.json``; an unrecognised target fails closed."""
    for family in FAMILIES.values():
        if family.matches(info):
            return family
    # A release that does not identify its target is not safe to run.  This is
    # preferable to accidentally running a velocity/pressure checkpoint as WSS.
    raise ReleaseError("发布包合同不支持：当前仅支持单帧 wall shear stress (WSS) 推理。")


def family_for_protocol(protocol: str) -> ModelFamily:
    for family in FAMILIES.values():
        if family.protocol == protocol:
            return family
    raise ReleaseError(f"未注册的模型族协议：{protocol}")


def family_for_release(release) -> ModelFamily:
    """Family of a loaded ``Release`` (or any object carrying ``contract``/``info``)."""
    contract = getattr(release, "contract", None) or getattr(release, "registry_contract", None)
    if isinstance(contract, dict) and contract.get("family") in FAMILIES:
        return FAMILIES[contract["family"]]
    if isinstance(contract, dict) and contract.get("protocol"):
        return family_for_protocol(contract["protocol"])
    info = getattr(release, "info", None)
    if isinstance(info, dict):
        return family_for_info(info)
    return WALL_WSS


__all__ = ["FAMILIES", "ModelFamily", "PF6_VF6_VOLUME", "ReleaseError", "VOLUME_FEATURES", "VOLUME_FIELDS", "WALL_WSS",
           "family_for_info", "family_for_protocol", "family_for_release"]

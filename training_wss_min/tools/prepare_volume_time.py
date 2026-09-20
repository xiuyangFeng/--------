"""Pre-register the volume_time_20260919 matrix: pressure / velocity full-cycle arms on cv3_v51 (seed 1234).

用户 2026-09-19 要求「把融入 t 的实验在压力和速度上也各做一遍，每个只做一个 seed，用显卡并行」。
臂结构与 WSS 线 `wss_time_ecc_20260918` 逐条对应，只把目标从壁面 WSS 换成体场压力 / 速度：

  PF6_v51_f{k} / VF6_v51_f{k}   峰值帧折底座：老 v5 的 PF6 / VF6 配方（18D 体场特征 + Murray F6）
                                原样搬到 v5.1 数据与 cv3_v51 折，做本批时间臂的配对父臂与读数基准
  PT0_f{k}  / VT0_f{k}          相位查询：+ q_norm/dq_norm/t_sin/t_cos，timesteps=random_frame，
                                逐帧线性 z（frame_stats），EMA(train_loss) 选模
  PTB8/PTB16, VTB4/VTB8         时间基头 K* 与 2K*（A2：K*压力=8、K*速度=4）：压力 out_dim = K+1，
                                速度 out_dim = 3(K+1)（三分量共用一套基向量），系数 MSE，
                                无时间输入，一次推理重建 81 帧

阶段 0 冻结产物（`experiments/volume_time_20260919/offline`，作业 15112 / 15113）：
`volume_stats_{scope}.json`（逐帧线性统计量）、`time_basis_{target}_{scope}.npz`（训练折 PCA 基）、
以及 `data_wss_v5/views_v5_1/wss_min_volume_time_v1/<cid>/sidecar.npz`（81 帧标签访问层）。

    python -u -m training_wss_min.tools.prepare_volume_time    （CPU 节点：要加载折训练集算特征统计量）
"""
from __future__ import annotations

import json
from pathlib import Path

from training_wss_min import config as C
from training_wss_min import dataset as D
from training_wss_min.tools import prepare_wss_local_wave1 as W1

ROOT = C.PROJECT_ROOT
NAME = "volume_time_20260919"
CONFIGS = ROOT / "training_wss_min/configs" / NAME
EXP = ROOT / "training_wss_min/experiments" / NAME
OFF = EXP / "offline"
VIEW51 = ROOT / "data_wss_v5/views_v5_1/wss_min_view_v1"
CV3 = VIEW51 / "cv3_v51"
FLOW51 = ROOT / "data_wss_v5/views_v5_1/wss_min_flowref_v1"
SIDECAR51 = ROOT / "data_wss_v5/views_v5_1/wss_min_volume_time_v1"
H5_ROOT = ROOT / "data_wss_v5/anatomy_pointcloud_v5_1_20260916/cases"
LEGACY = ROOT / "training_wss_min/configs/wss_local_wave2_20260912"
TIME_KEYS = ["q_norm", "dq_norm", "t_sin", "t_cos"]
SEED = 1234
# 目标 → (老 v5 参考配置, 训练目标键, 离线基文件里的目标名, 输出分量数)
TARGETS = {"P": ("PF6_s1234", "pressure_mixed", "pressure", 1),
           "V": ("VF6_s1234", "velocity", "velocity", 3)}


def kinds(k_star: int) -> dict[str, int | None]:
    """相位查询 + 两个时间基截断：K* 与 2K*，与 WSS 线 `wss_time_ecc_20260918` 同一条规则。

    A2 给出 K*(压力)=8、K*(速度)=4，所以压力跑 TB8/TB16、速度跑 TB4/TB8。
    """
    return {"T0": None, f"TB{k_star}": k_star, f"TB{2 * k_star}": 2 * k_star}


def base_id(prefix: str, fold: int) -> str:
    return f"{'PF6' if prefix == 'P' else 'VF6'}_v51_f{fold}_s{SEED}"


def feature_stats(prefix: str, fold: int, data: dict) -> Path:
    """折训练集（两折）上的逐特征 z 统计量。

    体场目标的特征统计量必须在**体场点集**（wall∪interior）上算：内部单元的 dist_to_wall_mm、
    rho、法向等分布与壁面完全不同，不能借用 WSS 线的壁面统计量。
    """
    path = EXP / "feature_stats" / f"{prefix}_fold{fold}_train.json"
    if path.is_file():
        return path
    features = tuple(data["input_features"])
    stats = D.load_wss_stats(data["wss_stats_path"])
    extra = tuple(f for f in features if f in C.SIDECAR_FEATURE_KEYS)
    cases = D.load_partition(data["split_path"], "train", stats, strict=True, target=data["target"],
                             data_root=data["data_root"], required_frame_version=data["required_frame_version"],
                             extra_point_features=extra, point_features_root=data["point_features_root"])
    expected = json.loads(Path(data["split_path"]).read_text())["counts"]["train"]
    if len(cases) != expected:
        raise RuntimeError(f"{prefix} fold{fold}: expected {expected} train cases, loaded {len(cases)}")
    out = D.compute_feature_stats(cases, features, data["curvature_transform"])
    out["_provenance"] = {"train_split": data["split_path"], "n_cases": len(cases),
                          "volume_stats": data["wss_stats_path"], "target": data["target"],
                          "scope": "volume point set (wall nodes ∪ interior cells) of the two training folds"}
    W1.save(path, out)
    return path


def time_feature_stats(prefix: str, fold: int, peak_path: Path) -> Path:
    """T0 的特征统计量 = 折底座的 20 维统计量逐位不变 + 4 个相位特征的恒等统计。

    相位特征是 172 例共享的有界协议常数（`compute_feature_stats` 对它们也走恒等），
    所以不必为 T0 再遍历一次折训练集。
    """
    path = EXP / "feature_stats" / f"{prefix}_fold{fold}_train_time.json"
    if path.is_file():
        return path
    out = json.loads(peak_path.read_text())
    provenance = out.pop("_provenance")
    for key in TIME_KEYS:
        out[key] = {"mean": 0.0, "std": 1.0, "transform": "none"}
    out["_provenance"] = {**provenance, "peak_reference": str(peak_path), "time_features": TIME_KEYS,
                          "note": "the 20 volume statistics are bit-identical to the fold base file; "
                                  "phase features are bounded protocol constants (identity)"}
    W1.save(path, out)
    return path


def fold_base(prefix: str, fold: int) -> dict:
    """峰值帧折底座：老 v5 的 PF6 / VF6 配方 → v5.1 数据 + cv3_v51 fold，其余逐字段不动。"""
    legacy_name, target, _, _ = TARGETS[prefix]
    cfg = json.loads((LEGACY / f"{legacy_name}.json").read_text())
    aid = base_id(prefix, fold)
    cfg["name"] = f"{NAME}/{aid}"
    d = cfg["data"]
    d["data_root"] = str(VIEW51)
    d["split_path"] = str(CV3 / f"fold{fold}.json")
    d["wss_stats_path"] = str(OFF / f"volume_stats_fold{fold}.json")
    d["point_features_root"] = [str(FLOW51)]
    d["feature_stats_path"] = str(feature_stats(prefix, fold, d))
    # 折底座本身就是这批的父臂：与老 v5 的同 seed PF6/VF6 配对初始化（同特征、同结构）
    cfg["train"]["init_reference_config"] = str(LEGACY / f"{legacy_name}.json")
    cfg["notes"] = (f"volume-time peak-frame fold base {aid}: legacy {legacy_name} recipe (18D volume features + Murray F6) "
                    f"ported to the v5.1 view and cv3_v51 fold-{fold} (train = other two folds, held-out fold read as 'test'; "
                    f"test34 unused); peak step 1162, global_stats normalisation; parent/paired init = legacy {legacy_name}.")
    return cfg


def arm(prefix: str, kind: str, K: int | None, fold: int) -> dict:
    base = fold_base(prefix, fold)
    legacy_name, target, basis_target, channels = TARGETS[prefix]
    aid = f"{prefix}{kind}_f{fold}_s{SEED}"
    cfg = json.loads(json.dumps(base))
    cfg["name"] = f"{NAME}/{aid}"
    d = cfg["data"]
    d["target_normalization"] = "frame_stats"
    d["waveform_path"] = str(ROOT / "training_wss_min/experiments/wss_time_ecc_20260918/offline"
                             / "protocol_inlet_waveform_v51.json")
    d["volume_time_sidecar_root"] = str(SIDECAR51)
    d["volume_h5_root"] = str(H5_ROOT)
    cfg["eval"]["eval_frames"] = "all"
    cfg["eval"]["time_metrics"] = True
    cfg["train"]["init_reference_config"] = str(CONFIGS / f"{base_id(prefix, fold)}.json")
    name = "pressure" if prefix == "P" else "velocity"
    if kind == "T0":
        d["timesteps"] = "random_frame"
        d["input_features"] = list(base["data"]["input_features"]) + TIME_KEYS
        d["feature_stats_path"] = str(time_feature_stats(prefix, fold, Path(base["data"]["feature_stats_path"])))
        cfg["train"]["selection_rule"] = "train_loss_ema"
        cfg["train"]["selection_ema_alpha"] = 0.2
        cfg["notes"] = (f"volume-time stage-2 {aid}: {base_id(prefix, fold)} recipe + phase condition {TIME_KEYS} "
                        f"({len(d['input_features'])} inputs, appended columns zero-initialised), timesteps=random_frame "
                        f"(uniform frames, peak guarantee 1/9), per-frame linear z on the {name} target, "
                        f"checkpoint by EMA(train_loss, alpha 0.2); held-out fold on all 81 frames; test34 unused.")
    else:
        d["timesteps"] = "time_basis"
        d["time_basis_path"] = str(OFF / f"time_basis_{basis_target}_fold{fold}.npz")
        cfg["model"]["time_basis_k"] = int(K)
        cfg["model"]["out_dim"] = channels * (int(K) + 1)
        cfg["train"]["loss_pinball_lambda"] = 0.0
        shared = " (the three components share one set of basis vectors)" if channels == 3 else ""
        cfg["notes"] = (f"volume-time stage-2 {aid}: {base_id(prefix, fold)} recipe with a time-basis output head K={K} "
                        f"(out_dim {channels * (int(K) + 1)} = {channels} component(s) × [b0, a_1..a_K] on the train-fold PCA "
                        f"basis{shared}; coefficient MSE); no time inputs; one inference reconstructs all 81 frames; "
                        f"backbone/decoder paired with the fold base, head fresh; held-out fold on all 81 frames; test34 unused.")
    return cfg


def check(aid: str, cfg: dict) -> None:
    classes = {"data": C.DataConfig, "model": C.ModelConfig, "train": C.TrainConfig, "eval": C.EvalConfig}
    for section, cls in classes.items():
        unknown = set(cfg[section]) - set(cls.__dataclass_fields__)
        if unknown:
            raise ValueError(f"{aid}: unknown {section} fields: {unknown}")
    parsed = C.ExpConfig.from_dict(cfg)
    C.validate_features(parsed)
    for key in ("wss_stats_path", "feature_stats_path", "split_path"):
        if not Path(cfg["data"][key]).is_file():
            raise FileNotFoundError(f"{aid}: {key} missing: {cfg['data'][key]}")
    for key in ("waveform_path", "time_basis_path"):
        value = cfg["data"].get(key)
        if value and not Path(value).is_file():
            raise FileNotFoundError(f"{aid}: {key} missing: {value}")


def main() -> None:
    a2 = json.loads((OFF / "a2_time_basis.json").read_text())
    k_star = {t: a2[t]["k_star"] for t in ("pressure", "velocity")}
    arms = []
    for prefix in TARGETS:
        for fold in range(3):
            aid = base_id(prefix, fold)
            cfg = fold_base(prefix, fold)
            check(aid, cfg)
            W1.save(CONFIGS / f"{aid}.json", cfg)
            arms.append(dict(id=aid, title=f"{'pressure' if prefix == 'P' else 'velocity'} peak-frame fold base",
                             config=f"{aid}.json", phase=1, modules=["F6"], seed=SEED, fold=fold,
                             target=TARGETS[prefix][1],
                             parent=f"legacy {TARGETS[prefix][0]} (old v5 view, train138)", depends_on=[],
                             evaluate=["best"], single_change=False))
    for prefix in TARGETS:
        for kind, K in kinds(int(k_star[TARGETS[prefix][2]])).items():
            for fold in range(3):
                aid = f"{prefix}{kind}_f{fold}_s{SEED}"
                cfg = arm(prefix, kind, K, fold)
                check(aid, cfg)
                W1.save(CONFIGS / f"{aid}.json", cfg)
                arms.append(dict(id=aid, title=cfg["notes"].split(":", 1)[1].split(";")[0].strip(),
                                 config=f"{aid}.json", phase=2, seed=SEED, fold=fold,
                                 target=TARGETS[prefix][1],
                                 modules=["F6", "frame_stats"] + (["phase_condition", "train_loss_ema"] if kind == "T0"
                                                                  else [f"time_basis_K{K}", "pinball_off"]),
                                 parent=base_id(prefix, fold), depends_on=[base_id(prefix, fold)],
                                 evaluate=["best", "last"], single_change=kind == "T0",
                                 config_diff_vs_anchor=W1.declared_diff(fold_base(prefix, fold), cfg)))
    cv3 = json.loads((CV3 / "cv3_summary.json").read_text())
    a0 = json.loads((OFF / "a0_scan.json").read_text())
    a3 = json.loads((OFF / "a3_tnull.json").read_text())
    W1.save(CONFIGS / "matrix.json", dict(
        schema_version=1, experiment=NAME, anchor=str(LEGACY / "PF6_s1234.json"),
        anchor_run=str(ROOT / "training_wss_min/runs/wss_local_wave2_20260912/PF6_s1234"), control_id=None,
        external_reference_runs={
            "PF6_s1234 (legacy old-v5 pressure base)": str(ROOT / "training_wss_min/runs/wss_local_wave2_20260912/PF6_s1234/eval/ckpt_best/metrics.json"),
            "VF6_s1234 (legacy old-v5 velocity base)": str(ROOT / "training_wss_min/runs/wss_local_wave2_20260912/VF6_s1234/eval/ckpt_best/metrics.json")},
        cv3=cv3, stage0_jobs=[15112, 15113], k_star=k_star,
        a0_contract={k: a0[k] for k in ("n_cases", "n_errors", "all_steps_match", "all_peak_index_21",
                                        "max_peak_pressure_diff", "max_peak_velocity_diff",
                                        "p_volmean_ptp_pa", "p_rel_spatial_std_peak_pa_median")},
        a2_fold_mean_ceiling={t: {str(k): a2[t]["fold_mean"][str(k)]
                                  for k in (a2[t]["k_star"], 2 * a2[t]["k_star"])}
                              for t in ("pressure", "velocity")},
        t_null={t: a3[t]["fold_mean"] for t in ("pressure", "velocity")},
        cycle_metric_protocol=("peak frame full cloud; cycle metrics on a fixed per-case subsample "
                               "(20000 interior cells + 4000 wall nodes for pressure, interior only for velocity, "
                               "seed 20260919 keyed by unit_id) shared by the offline ceiling / T-null and every arm"),
        expected_training_runs=len(arms),
        expected_evaluations=sum(len(a["evaluate"]) for a in arms), arms=arms,
        reading_rule="fold arms are read only on their held-out fold (all 81 frames); gates use the three-fold mean and "
                     "fold sign agreement; every time arm must beat both its peak-frame fold base (peak frame) and the "
                     "A3 T-null (cycle); test34 is read once only after a winner is locked"))
    print(CONFIGS / "matrix.json", len(arms), "arms; K* =", k_star)


if __name__ == "__main__":
    main()

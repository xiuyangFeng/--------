"""Pre-register the wave-1 local-information screening matrix (one seed per arm).

Every arm is derived from the C1 anchor (E2 directional neighbourhood + E3 full-wall
query patch/FiLM, ``wss_direct_recovery_20260912/C1_s1234``) and changes exactly one
thing, except the two declared combinations.  All arms share C1's initial weights for
every inherited tensor through ``train.init_reference_config`` (paired fresh
initialisation); appended point features get zero-initialised input columns.

    python -m training_wss_min.tools.prepare_wss_local_wave1
"""
from __future__ import annotations

import copy
import json
from pathlib import Path

import numpy as np

from training_wss_min import config as C
from training_wss_min import dataset as D

ROOT = C.PROJECT_ROOT
NAME = "wss_local_wave1_20260912"
CONFIGS = ROOT / "training_wss_min/configs" / NAME
EXP = ROOT / "training_wss_min/experiments" / NAME
RUNS = ROOT / "training_wss_min/runs"
ANCHOR = ROOT / "training_wss_min/configs/wss_direct_recovery_20260912/C1_s1234.json"
ANCHOR_RUN = RUNS / "wss_direct_recovery_20260912/C1_s1234"
GEOM_ROOT = ROOT / "data_wss_v5/views/wss_min_geom_v2"
FLOW_ROOT = ROOT / "data_wss_v5/views/wss_min_flowref_v1"
VIEW = ROOT / "data_wss_v5/views/wss_min_view_v1"
ALLFRAME_STATS = VIEW / "wss_global_stats_train138_allframes_floor005.json"
WAVEFORM = VIEW / "protocol_inlet_waveform.json"
SEED = 1234

F1 = ["bend_cos", "bend_signed", "torsion_rr", "bend_cos_up2", "bend_cos_up5"]
F2 = ["carina_cos", "bif_plane_cos", "bif_angle_cos", "sibling_radius_ratio"]
F3 = ["s_over_d", "up_min_r_ratio_2d", "up_min_r_ratio_5d", "up_max_r_ratio_5d", "up_kappa_5d"]
F6 = ["log_q_branch_murray", "log_tau0_murray"]
P5 = ["atlas_prior_logwss"]
TIME = ["q_norm", "dq_norm", "t_sin", "t_cos"]
NEW_POINT_KEYS = F1 + F2 + F3 + F6 + P5

ARMS = {
    "X0": ("C1 contemporaneous control (identical config, fresh run)", []),
    "X1": ("P4 weight EMA 0.999 (extra ckpt_ema evaluation)", ["ema"]),
    "X2": ("F1 bend-referenced circumferential angle (+ upstream 2D/5D lags, torsion)", ["F1"]),
    "X3": ("F2 bifurcation-referenced angles (carina side, plane, angle, sibling radius)", ["F2"]),
    "X4": ("F3 upstream history (s/D, upstream radius extrema, upstream max kappa*R)", ["F3"]),
    "X5": ("F6 Murray flow-share prior (log Q share, log tau0)", ["F6"]),
    "X6": ("P5 population prior ln WSS (train138, leave-one-out)", ["P5"]),
    "X7": ("S1a query patch offsets in the query tangent frame", ["frame"]),
    "X8": ("S1b query-conditioned attention pooling over the patch", ["attn"]),
    "X9": ("S1c tangent frame + attention + K=32 patch", ["frame", "attn", "k32"]),
    "X10": ("T1 tangent-plane WSS direction auxiliary head (cosine loss 0.1)", ["direction"]),
    "X11": ("S3 centreline section-token context replacing the per-case FiLM vector", ["section"]),
    "X12": ("S5 mixture-of-experts head gated by query input features (4 experts)", ["moe"]),
    "X13a": ("P1 stage 1: all-frame pretraining with phase features (peak prob 1/9)", ["allframe"]),
    "X13b": ("P1 stage 2: peak-frame fine-tuning from X13a (150 epochs, lr 5e-4)", ["peak_finetune"]),
    "X15": ("F-all: F1 + F2 + F3 + F6 appended together", ["F1", "F2", "F3", "F6"]),
    "X16": ("Combo: S1c + F-all + T1 direction head", ["frame", "attn", "k32", "F1", "F2", "F3", "F6", "direction"]),
}
SINGLE_CHANGE = {aid for aid, (_, modules) in ARMS.items() if len(modules) <= 1 or aid == "X9"}


def save(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(value, ensure_ascii=False, indent=2) + "\n"
    if path.exists() and path.read_text() != text:
        raise FileExistsError(f"refusing to change preregistered file: {path}")
    path.write_text(text)


def frozen_feature_stats(extra_keys: list[str]) -> dict:
    """C1's frozen 25-D stats plus train138 statistics of the appended keys (identity for phase features)."""
    base = json.loads(Path(json.loads(ANCHOR.read_text())["data"]["feature_stats_path"]).read_text())
    union_path = EXP / "feature_stats" / "union_new_keys_train138.json"
    if not union_path.is_file():
        anchor = C.ExpConfig.from_json(ANCHOR)
        stats = D.load_wss_stats(anchor.data.wss_stats_path)
        cases = D.load_partition(anchor.data.split_path, "train", stats, strict=True, target="wss",
                                 data_root=anchor.data.data_root,
                                 required_frame_version=anchor.data.required_frame_version,
                                 extra_point_features=tuple(NEW_POINT_KEYS),
                                 point_features_root=[str(GEOM_ROOT), str(FLOW_ROOT)])
        if len(cases) != 138:
            raise RuntimeError(f"expected train138, loaded {len(cases)}")
        union = D.compute_feature_stats(cases, tuple(NEW_POINT_KEYS), anchor.data.curvature_transform)
        for key in TIME:
            union[key] = {"mean": 0.0, "std": 1.0, "transform": "none"}
        union["_provenance"] = {"train_split": anchor.data.split_path, "n_cases": len(cases),
                                "sidecar_roots": [str(GEOM_ROOT), str(FLOW_ROOT)],
                                "curvature_transform": anchor.data.curvature_transform}
        save(union_path, union)
    union = json.loads(union_path.read_text())
    out = dict(base)
    for key in extra_keys:
        out[key] = union[key]
    return out


def configure(aid: str, modules: list[str]) -> dict:
    cfg = json.loads(ANCHOR.read_text())
    cfg["name"] = f"{NAME}/{aid}_s{SEED}"
    cfg["notes"] = f"wave-1 local-information screening {aid}: {ARMS[aid][0]}; modules={modules}; single seed."
    cfg["train"]["init_reference_config"] = str(ANCHOR)
    cfg["train"]["log_loss_components"] = True
    extra_keys: list[str] = []
    for module in modules:
        if module in {"F1", "F2", "F3", "F6", "P5"}:
            extra_keys += {"F1": F1, "F2": F2, "F3": F3, "F6": F6, "P5": P5}[module]
        elif module == "ema":
            cfg["train"]["ema_decay"] = 0.999
        elif module == "frame":
            cfg["model"]["query_patch_frame"] = "tangent"
        elif module == "attn":
            cfg["model"]["query_patch_pool"] = "attention"
        elif module == "k32":
            cfg["model"]["query_patch_k"] = 32
            cfg["data"]["query_patch_nsample"] = 32
        elif module == "direction":
            cfg["model"].update(direction_head=True, direction_head_hidden=32)
            cfg["train"]["loss_direction_lambda"] = 0.1
            cfg["data"]["direction_target"] = True
        elif module == "section":
            cfg["model"].update(section_context=True, section_context_bin_mm=4.0, section_context_max_bins=64,
                                section_context_layers=2, section_context_heads=4, section_context_hidden=128)
            cfg["data"]["section_tokens"] = True
        elif module == "moe":
            cfg["model"].update(head_moe_experts=4, head_moe_hidden=16)
        elif module in {"allframe", "peak_finetune"}:
            extra_keys += TIME
            cfg["data"].update(timesteps="random_frame", waveform_path=str(WAVEFORM))
            cfg["eval"]["eval_frames"] = "peak"
            if module == "allframe":
                cfg["data"]["peak_frame_prob"] = 1.0 / 9.0
                cfg["data"]["wss_stats_path"] = str(ALLFRAME_STATS)
            else:
                cfg["data"]["peak_frame_prob"] = 1.0
                cfg["train"].update(epochs=150, min_epoch=150, eval_every=150, lr=5e-4, warmup_epochs=5,
                                    init_checkpoint_path=str(RUNS / NAME / f"X13a_s{SEED}" / "ckpt_last.pt"),
                                    init_checkpoint_strict=True)
                cfg["train"]["init_reference_config"] = None
        else:
            raise ValueError(module)
    if extra_keys:
        cfg["data"]["input_features"] = list(cfg["data"]["input_features"]) + extra_keys
        point_keys = [k for k in extra_keys if k in NEW_POINT_KEYS]
        if point_keys:
            cfg["data"]["point_features_root"] = [str(GEOM_ROOT), str(FLOW_ROOT)]
        stats_path = EXP / "feature_stats" / f"{aid}_feature_stats.json"
        save(stats_path, frozen_feature_stats(extra_keys))
        cfg["data"]["feature_stats_path"] = str(stats_path)
    classes = {"data": C.DataConfig, "model": C.ModelConfig, "train": C.TrainConfig, "eval": C.EvalConfig}
    for section, cls in classes.items():
        unknown = set(cfg[section]) - set(cls.__dataclass_fields__)
        if unknown:
            raise ValueError(f"unknown {section} fields: {unknown}")
    C.ExpConfig.from_dict(cfg)
    return cfg


def declared_diff(anchor: dict, cfg: dict) -> dict:
    """Mechanical field-level diff against the anchor (name/notes excluded)."""
    diff = {}
    for section in ("data", "model", "train", "eval"):
        for key in sorted(set(anchor[section]) | set(cfg[section])):
            before, after = anchor[section].get(key), cfg[section].get(key)
            if before != after:
                diff[f"{section}.{key}"] = {"anchor": before, "arm": after}
    return diff


def main() -> None:
    anchor = json.loads(ANCHOR.read_text())
    if not ANCHOR_RUN.joinpath("eval/ckpt_best/metrics.json").is_file():
        raise FileNotFoundError("the C1 anchor run must exist for the historical reference row")
    arms = []
    for aid, (title, modules) in ARMS.items():
        cfg = configure(aid, modules)
        filename = f"{aid}_s{SEED}.json"
        save(CONFIGS / filename, cfg)
        diff = declared_diff(anchor, cfg)
        allowed_prefixes = {"data.input_features", "data.point_features_root", "data.feature_stats_path",
                            "train.init_reference_config", "train.log_loss_components"}
        substantive = {k: v for k, v in diff.items() if k not in allowed_prefixes}
        arm = dict(id=aid, title=title, config=filename, phase=0, modules=modules, parent="X0",
                   depends_on=(["X13a"] if aid == "X13b" else []),
                   evaluate=(["best", "last", "ema"] if "ema" in modules else ["best", "last"]),
                   single_change=aid in SINGLE_CHANGE, config_diff_vs_anchor=diff,
                   substantive_diff_keys=sorted(substantive))
        arms.append(arm)
    save(CONFIGS / "matrix.json", dict(
        schema_version=1, experiment=NAME, anchor=str(ANCHOR), anchor_run=str(ANCHOR_RUN), seed=SEED,
        epochs={aid: (150 if aid == "X13b" else 400) for aid in ARMS},
        expected_training_runs=len(ARMS),
        expected_evaluations=sum(len(a["evaluate"]) for a in arms), arms=arms,
        controls={"X0": "same config as C1, fresh run (contemporaneous control)",
                  "C1_historical": str(ANCHOR_RUN / "eval/ckpt_best/metrics.json")},
        reading_rule=("single seed 1234 on the exposed test34; deltas vs X0 and vs historical C1 are "
                      "screening evidence only (single-seed 95% band ~ +-0.034 Pa R2, +-0.009 normalized); "
                      "no significance or generalisation claims"),
        skipped=["T4 anisotropic local-difference loss (E6 interaction unresolved)",
                 "F4 geodesic/HKS descriptors", "F5 section-shape features (geometry candidate still training_allowed=false)",
                 "F7 predicted wall-pressure gradient cascade", "S2 windowed attention", "S4 (s,theta) chart U-Net",
                 "S6 DiffusionNet branch", "P2/P3 pretraining data", "T3 hotspot cascade"],
    ))
    print(CONFIGS / "matrix.json", len(arms), "arms")


if __name__ == "__main__":
    main()

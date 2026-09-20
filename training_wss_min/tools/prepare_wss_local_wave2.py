"""Wave 2 (2026-09-12, user request): X5+X11 at three seeds, F6 variants, F6 into the volume targets.

Arms (all single runs):
  X5X11_s1234 / _s7 / _s2025   Murray prior + centreline section-token context (seed-matched references)
  X5e233_s1234                 Murray exponent 2.33 (log_q_branch_murray233 only)
  X5cap_s1234                  Murray split from virtual-cap radii (log_q_branch_murray_cap only)
  X5res_s1234                  X5 inputs + residual target ln(WSS) - log_tau0_murray (metrics in standard log_z)
  PF6_s1234                    P02 (best pressure arm) + F6            VF6_s1234   V07 (best velocity arm) + F6
  P02r_s1234 / V07r_s1234      contemporaneous reruns of P02 / V07 (phase 1, same-config controls)

    python -m training_wss_min.tools.prepare_wss_local_wave2
"""
from __future__ import annotations

import json
from pathlib import Path

from training_wss_min import config as C
from training_wss_min.tools import prepare_wss_local_wave1 as W1
from training_wss_min.tools import prepare_wss_local_wave1b as W1B

ROOT = C.PROJECT_ROOT
NAME = "wss_local_wave2_20260912"
CONFIGS = ROOT / "training_wss_min/configs" / NAME
EXP = ROOT / "training_wss_min/experiments" / NAME
RUNS = ROOT / "training_wss_min/runs"
VOLUME_CONFIGS = ROOT / "training_wss_min/configs/volume_attention_20260910"
OFFSET_STATS = EXP / "wss_global_stats_train138_offset_logtau0.json"
VOLUME_UNION = EXP / "feature_stats" / "union_murray_volume_train138.json"
WAVE1_UNION = W1.EXP / "feature_stats" / "union_new_keys_train138.json"
F6 = W1.F6

ARMS = {
    "X5X11_s1234": ("F6 Murray prior + S3 section-token context, seed 1234", ["F6", "section"], 1234, "wss"),
    "X5X11_s7": ("F6 + S3 section-token context, seed 7", ["F6", "section"], 7, "wss"),
    "X5X11_s2025": ("F6 + S3 section-token context, seed 2025", ["F6", "section"], 2025, "wss"),
    "X5e233_s1234": ("F6 variant: Murray exponent 2.33 (log_q only)", ["F6e233"], 1234, "wss"),
    "X5cap_s1234": ("F6 variant: virtual-cap leaf radii (log_q only)", ["F6cap"], 1234, "wss"),
    "X5res_s1234": ("F6 variant: residual target ln(WSS) - log_tau0 (X5 inputs)", ["F6", "residual"], 1234, "wss"),
    "PF6_s1234": ("P02 pressure (mixed) + F6 Murray prior", ["F6"], 1234, "P02"),
    "VF6_s1234": ("V07 velocity + F6 Murray prior", ["F6"], 1234, "V07"),
    "P02r_s1234": ("P02 contemporaneous rerun (control)", [], 1234, "P02"),
    "V07r_s1234": ("V07 contemporaneous rerun (control)", [], 1234, "V07"),
}
PHASE = {"P02r_s1234": 1, "V07r_s1234": 1}


def wss_reference(seed: int) -> Path:
    return W1.ANCHOR if seed == 1234 else W1B.CONFIGS / "refs" / f"C1_s{seed}.json"


def new_key_stats(keys: list[str], population: str) -> dict:
    """Frozen base stats of the anchor + train138 stats of the appended keys (wall or wall+interior population)."""
    if population == "wss":
        union = json.loads(WAVE1_UNION.read_text())
        extra_path = EXP / "feature_stats" / "union_flowref_v11_wall_train138.json"
        if not extra_path.is_file():
            from training_wss_min import dataset as D
            anchor = C.ExpConfig.from_json(W1.ANCHOR)
            stats = D.load_wss_stats(anchor.data.wss_stats_path)
            cases = D.load_partition(anchor.data.split_path, "train", stats, strict=True, target="wss",
                                     data_root=anchor.data.data_root,
                                     required_frame_version=anchor.data.required_frame_version,
                                     extra_point_features=("log_q_branch_murray233", "log_q_branch_murray_cap"),
                                     point_features_root=[str(W1.GEOM_ROOT), str(W1.FLOW_ROOT)])
            extra = D.compute_feature_stats(cases, ("log_q_branch_murray233", "log_q_branch_murray_cap"),
                                            anchor.data.curvature_transform)
            W1.save(extra_path, extra)
        union.update(json.loads(extra_path.read_text()))
    else:
        union = json.loads(VOLUME_UNION.read_text())
    return {key: union[key] for key in keys}


def configure(aid: str) -> dict:
    title, modules, seed, base = ARMS[aid]
    if base == "wss":
        cfg = json.loads(W1.ANCHOR.read_text())
        cfg["train"]["init_reference_config"] = str(wss_reference(seed))
        cfg["train"]["seed"] = seed
        base_stats = json.loads(Path(cfg["data"]["feature_stats_path"]).read_text())
    else:
        source = VOLUME_CONFIGS / f"{base}_s1234.json"
        cfg = json.loads(source.read_text())
        # pair to the historical volume arm itself (its own reference chain is reconstructed by paired init)
        cfg["train"]["init_reference_config"] = str(source)
        base_stats = json.loads(Path(cfg["data"]["feature_stats_path"]).read_text())
    cfg["name"] = f"{NAME}/{aid}"
    cfg["notes"] = f"wave-2 {aid}: {title}; modules={modules}; seed {seed}; single run."
    cfg["train"]["log_loss_components"] = True
    extra_keys: list[str] = []
    for module in modules:
        if module == "F6":
            extra_keys += F6
        elif module == "F6e233":
            extra_keys += ["log_q_branch_murray233"]
        elif module == "F6cap":
            extra_keys += ["log_q_branch_murray_cap"]
        elif module == "section":
            cfg["model"].update(section_context=True, section_context_bin_mm=4.0, section_context_max_bins=64,
                                section_context_layers=2, section_context_heads=4, section_context_hidden=128)
            cfg["data"]["section_tokens"] = True
        elif module == "residual":
            cfg["data"]["target_log_offset_feature"] = "log_tau0_murray"
            cfg["data"]["wss_stats_path"] = str(OFFSET_STATS)
        else:
            raise ValueError(module)
    if extra_keys:
        cfg["data"]["input_features"] = list(cfg["data"]["input_features"]) + extra_keys
        cfg["data"]["point_features_root"] = ([str(W1.GEOM_ROOT), str(W1.FLOW_ROOT)] if base == "wss"
                                              else [str(W1.FLOW_ROOT)])
        stats = dict(base_stats)
        stats.update(new_key_stats(extra_keys, "wss" if base == "wss" else "volume"))
        stats_path = EXP / "feature_stats" / f"{aid}_feature_stats.json"
        W1.save(stats_path, stats)
        cfg["data"]["feature_stats_path"] = str(stats_path)
    classes = {"data": C.DataConfig, "model": C.ModelConfig, "train": C.TrainConfig, "eval": C.EvalConfig}
    for section, cls in classes.items():
        unknown = set(cfg[section]) - set(cls.__dataclass_fields__)
        if unknown:
            raise ValueError(f"unknown {section} fields: {unknown}")
    C.ExpConfig.from_dict(cfg)
    return cfg


def main() -> None:
    if not OFFSET_STATS.is_file() or not VOLUME_UNION.is_file():
        raise FileNotFoundError("build the residual-target stats and the volume Murray feature stats first")
    arms = []
    for aid, (title, modules, seed, base) in ARMS.items():
        cfg = configure(aid)
        filename = f"{aid}.json"
        W1.save(CONFIGS / filename, cfg)
        anchor_path = W1.ANCHOR if base == "wss" else VOLUME_CONFIGS / f"{base}_s1234.json"
        arms.append(dict(id=aid, title=title, config=filename, phase=PHASE.get(aid, 0), modules=modules, seed=seed,
                         base=base, depends_on=[], evaluate=["best", "last"], single_change=len(modules) <= 1,
                         config_diff_vs_anchor=W1.declared_diff(json.loads(anchor_path.read_text()), cfg)))
    W1.save(CONFIGS / "matrix.json", dict(
        schema_version=1, experiment=NAME, anchor=str(W1.ANCHOR), anchor_run=str(W1.ANCHOR_RUN),
        control_id=None,
        external_reference_runs={
            "X5_s1234 (wave-1)": str(RUNS / W1.NAME / "X5_s1234" / "eval/ckpt_best/metrics.json"),
            "X5_s7 (wave-1b)": str(RUNS / W1B.NAME / "X5_s7" / "eval/ckpt_best/metrics.json"),
            "X5_s2025 (wave-1b)": str(RUNS / W1B.NAME / "X5_s2025" / "eval/ckpt_best/metrics.json"),
            "X11_s1234 (wave-1)": str(RUNS / W1.NAME / "X11_s1234" / "eval/ckpt_best/metrics.json"),
            "X5q_s1234 (wave-1b)": str(RUNS / W1B.NAME / "X5q_s1234" / "eval/ckpt_best/metrics.json"),
            "P02_s1234 (historical)": str(RUNS / "volume_attention_20260910/P02_s1234/eval/ckpt_best/metrics.json"),
            "V07_s1234 (historical)": str(RUNS / "volume_attention_20260910/V07_s1234/eval/ckpt_best/metrics.json"),
        },
        expected_training_runs=len(ARMS), expected_evaluations=2 * len(ARMS), arms=arms,
        reading_rule=("WSS arms: compare X5X11_s<seed> with X5_s<seed> and X0_s<seed> (same init); variants vs X5/X5q at seed 1234; "
                      "volume arms: compare PF6/VF6 with the historical P02/V07 and with the contemporaneous reruns P02r/V07r; "
                      "single runs on the exposed test34, no significance claims"),
        volume_note=("interior points receive the Murray share of their segment (vol_segment_id) and "
                     "log_tau0 = log_q - 3 log(atlas radius at the cell); wall-only sidecar keys are not allowed"),
    ))
    print(CONFIGS / "matrix.json", len(arms), "arms")


if __name__ == "__main__":
    main()

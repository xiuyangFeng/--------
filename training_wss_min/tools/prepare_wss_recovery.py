"""Pre-register the original fourteen direct-WSS experiments, one seed each."""
from __future__ import annotations

import copy
import json
from pathlib import Path

from training_wss_min import config as C

ROOT = C.PROJECT_ROOT
NAME = "wss_direct_recovery_20260912"
CONFIGS = ROOT / "training_wss_min/configs" / NAME
EXP = ROOT / "training_wss_min/experiments" / NAME
BASE = ROOT / "training_wss_min/configs/v6_followup_20260909/M2_a5_independent_k3_s1234.json"
A5 = ROOT / "training_wss_min/configs/v6_singleframe_20260909/A5_r4_localbranch_diffgeom_s1234.json"
FEATURES = ROOT / "training_wss_min/runs/v6_followup_20260909/M2_a5_independent_k3_s1234/feature_stats.json"


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(value, ensure_ascii=False, indent=2) + "\n"
    if path.exists() and path.read_text() != text:
        raise FileExistsError(f"refusing to change preregistered file: {path}")
    path.write_text(text)


def configure(aid, modules, *, a5=False):
    cfg = json.loads((A5 if a5 else BASE).read_text())
    cfg["name"] = f"{NAME}/{aid}_s1234"
    cfg["notes"] = f"Corrected original matrix {aid}; modules={modules}; single seed exploratory; fresh paired initialization."
    cfg["data"]["feature_stats_path"] = str(FEATURES)
    cfg["train"].update(log_loss_components=True, init_reference_config=str(BASE))
    for module in modules:
        if module == "E2":
            cfg["model"]["local_branch_directional"] = True
            cfg["data"]["local_geometry"] = True
        elif module == "E3":
            cfg["model"].update(query_patch_film=True, query_patch_k=16, query_patch_hidden=64)
            cfg["data"]["query_patch_nsample"] = 16
            cfg["data"]["local_geometry"] = True
        elif module == "E4":
            cfg["model"]["local_branch_pool"] = "multistat"
        elif module == "E5":
            cfg["model"].update(local_branch_radius_mode="physical_dual",
                                   local_branch_radii_mm=[2.5, 5.0], local_branch_radius_ratio=0.5)
            cfg["data"]["local_geometry"] = True
        elif module == "E6":
            cfg["data"]["local_geometry"] = True
            cfg["train"].update(loss_local_diff_lambda=0.05, local_diff_min_distance_mm=2.,
                local_diff_max_distance_mm=5., local_diff_max_pairs=256,
                local_diff_normal_cos_min=0.5, local_diff_normal_displacement_max=0.5)
        elif module == "E7":
            cfg["train"].update(loss_pairwise_rank_lambda=0.05,
                pairwise_rank_margin=0.05, pairwise_rank_max_pairs=256)
        elif module == "E8":
            cfg["data"]["support_sampling"] = "geom_stratified"
        elif module == "E9":
            cfg["data"]["input_features"].append("inlet_velocity_nominal_m_s")
            cfg["data"]["case_features_path"] = str(EXP / "inlet_audit/case_features.json")
            cfg["data"]["feature_stats_path"] = str(EXP / "inlet_audit/feature_stats_26d.json")
        else:
            raise ValueError(module)
    # Existing parser keeps historical forward compatibility. New configs must
    # additionally prove that no experimental flag is silently ignored.
    classes = {"data": C.DataConfig, "model": C.ModelConfig, "train": C.TrainConfig, "eval": C.EvalConfig}
    for section, cls in classes.items():
        unknown = set(cfg[section]) - set(cls.__dataclass_fields__)
        if unknown:
            raise ValueError(f"unknown {section} fields: {unknown}")
    C.ExpConfig.from_dict(cfg)
    return cfg


def main():
    titles = {
        "E0": "M2 independent-query contemporary control", "E1": "A5 SAME-query contemporary control",
        "E2": "Tangent-plane axial/circumferential directional quotas",
        "E3": "Full-wall query patch with coarse-context FiLM residual",
        "E4": "Local max + weighted mean + std pooling, unchanged width",
        "E5": "Two fixed-mm scales plus 0.5 local-radius scale",
        "E6": "Physical local surface-pair WSS difference loss",
        "E7": "Within-case pairwise ranking loss",
        "E8": "70% spatial coverage + 15% high-curvature + 15% junction support",
        "E9": "Audited nominal inlet Q/A feature",
        "C1": "E2 + E3", "C2": "E2 + E6", "C3": "E3 + E7", "C4": "Selected E2-E5 structure + E9",
    }
    arms = []
    for aid, title in titles.items():
        modules = ([] if aid in {"E0", "E1"} else [aid]) if aid.startswith("E") else {
            "C1": ["E2", "E3"], "C2": ["E2", "E6"], "C3": ["E3", "E7"], "C4": ["E2", "E9"]}[aid]
        filename = f"{aid}_s1234.json"
        cfg = configure(aid, modules, a5=aid == "E1")
        save(CONFIGS / filename, cfg)
        arm = dict(id=aid, title=title, config=filename, phase=int(aid.startswith("C")),
                   modules=modules if aid != "C4" else "selected_before_phase1",
                   parent="E0" if aid != "E1" else "historical_A5_same")
        if aid == "C4":
            arm["candidate_configs"] = {}
            for structural in ("E2", "E3", "E4", "E5"):
                candidate = f"C4_{structural}_s1234.json"
                save(CONFIGS / candidate, configure("C4", [structural, "E9"]))
                arm["candidate_configs"][structural] = candidate
        arms.append(arm)
    save(CONFIGS / "matrix.json", dict(schema_version=1, experiment=NAME, arms=arms,
        seed=1234, epochs=400, expected_training_runs=14, expected_evaluations=28,
        controls={"E0": str(BASE), "E1": str(A5)},
        combination_policy="Run all combinations as requested; record qualification. C4 selects eligible structure, or best structure labelled forced_exploratory if none qualifies.",
        selection=dict(checkpoint="best", partition="test", candidates=["E2", "E3", "E4", "E5"],
            normalized_casebalanced_r2_gain_min=0.005, pa_casebalanced_r2_gain_min_exclusive=0,
            pa_pooled_mae_ratio_max=1.02, iou_drop_max=0.010, pa_r2_p10_drop_max=0.020,
            no_additional_negative_r2_cases=True,
            ordering=["normalized_casebalanced_r2_desc", "pa_casebalanced_r2_desc", "id_asc"],
            disclosure="Adaptive selection on test34: exploratory evidence, not an untouched confirmation set; no multi-seed significance claims.")))
    print(CONFIGS / "matrix.json")


if __name__ == "__main__":
    main()

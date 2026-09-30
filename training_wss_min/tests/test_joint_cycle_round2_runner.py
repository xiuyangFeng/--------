import json
from pathlib import Path

import pytest

from training_wss_min.joint_cycle_round2 import validate_config, build, run

ROOT = Path(__file__).resolve().parents[2]
CONFIGS = ROOT / "training_wss_min/configs/joint_cycle_round2_v52_20260929"


def read(arm):
    return json.loads((CONFIGS / f"{arm}_f0_s1234.json").read_text())


def test_all_registered_configs_validate_and_pcgrad_is_exclusive_to_j2():
    configs = [read(a) for a in ["UT0", "UT1", "WT0", "WT1", "U0", "U1", "W0", "W1", "P1", "J2"]]
    for cfg in configs:
        validate_config(cfg)
    assert sum(c["train"]["gradient_method"] == "pcgrad" for c in configs) == 1
    for field, wrong in [("seed", 7), ("tasks", ["pressure"])]:
        cfg = read("UT1")
        cfg[field] = wrong
        with pytest.raises(ValueError):
            validate_config(cfg)
    for field, wrong in [("gradient_method", "mean"), ("epochs", 151), ("selection", "best")]:
        cfg = read("J2")
        cfg["train"][field] = wrong
        with pytest.raises(ValueError):
            validate_config(cfg)


def test_p1_build_reuses_original_j1_pressure_parameters_and_j2_original_joint():
    from training_wss_min.joint_cycle_model import JointCycleModel
    for arm in ("P1", "J2"):
        cfg = read(arm)
        stats = json.loads(Path(cfg["data"]["stats_path"]).read_text())
        model = build(cfg, stats)
        assert type(model) is JointCycleModel
        assert list(model.decoders) == cfg["tasks"]
        assert model.interaction is not None


def test_in_place_resume_is_rejected_before_mutating_prior_evidence():
    with pytest.raises(ValueError, match="forbids in-place resume"):
        run("does_not_exist.json", resume=True)

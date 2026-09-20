import json
from pathlib import Path
import numpy as np
import pytest
import torch

from training_wss_min import config as C
from training_wss_min.models import build_model
from training_wss_min.paired_initialization import build_paired_model
from training_wss_min.evaluate import _save_wss_prediction
from training_wss_min.runtime import seed_all
from training_wss_min.tools.prepare_wss_recovery import CONFIGS, BASE, configure


def test_matrix_is_original_fourteen_and_candidates_are_registered():
    matrix = json.loads((CONFIGS / "matrix.json").read_text())
    assert [a["id"] for a in matrix["arms"]] == [*[f"E{i}" for i in range(10)], *[f"C{i}" for i in range(1,5)]]
    for p in CONFIGS.glob("*s1234.json"):
        raw = json.loads(p.read_text())
        cfg = C.ExpConfig.from_dict(raw)
        assert cfg.train.seed == 1234 and cfg.train.epochs == 400
        assert cfg.data.support_n_points == cfg.data.query_n_points == 5000
        assert cfg.train.init_checkpoint_path is None
        for name, cls in (("data",C.DataConfig),("model",C.ModelConfig),("train",C.TrainConfig),("eval",C.EvalConfig)):
            assert set(raw[name]) <= set(cls.__dataclass_fields__)
    assert C.ExpConfig.from_json(CONFIGS / "E1_s1234.json").data.query_mode == "same"
    assert C.ExpConfig.from_json(CONFIGS / "E3_s1234.json").data.query_mode == "independent"


def test_e9_preserves_shared_weights_and_zero_extends_only_new_input():
    torch.set_num_threads(2)
    cfg = C.ExpConfig.from_json(CONFIGS / "E9_s1234.json")
    reference_cfg = C.ExpConfig.from_json(BASE)
    seed_all(1234)
    ref = build_model(reference_cfg.model, 25)
    candidate, evidence = build_paired_model(cfg)
    assert evidence["zero_extended_input_keys"] == ["stem.0.weight"]
    for key, tensor in ref.state_dict().items():
        if key == "stem.0.weight":
            assert torch.equal(tensor, candidate.state_dict()[key][:,:25])
            assert torch.count_nonzero(candidate.state_dict()[key][:,25]) == 0
        else:
            assert torch.equal(tensor, candidate.state_dict()[key]), key
    # New physical column is trainable at initialization, not permanently masked.
    x = torch.randn(12, 26)
    candidate.stem[0](x).square().mean().backward()
    assert candidate.stem[0].weight.grad[:,25].abs().sum() > 0


@pytest.mark.parametrize("arm,edit", [("E2", lambda c: c["data"].update(local_geometry=False)),
    ("E3", lambda c: c["data"].update(query_patch_nsample=8)),
    ("E6", lambda c: c["train"].update(local_diff_max_distance_mm=1.))])
def test_enabled_features_fail_loud_on_invalid_contract(arm, edit):
    cfg = configure(arm, [arm])
    edit(cfg)
    with pytest.raises(ValueError):
        C.ExpConfig.from_dict(cfg)


def test_saved_wss_preserves_rows_and_records_metric_clipping(tmp_path):
    case = dict(unit_id="AG/fast/example", y_norm=np.array([-2., 0., 2.], np.float32),
                y_raw=np.array([0., .9, 7.289], np.float32))
    pred = np.array([-10., 0., 2.])
    stats = {"method":"log_z", "log":{"mean":0., "std":1.}, "eps":.1}
    entry = _save_wss_prediction(case, pred, "wss", stats, tmp_path)
    with np.load(tmp_path / entry["file"]) as z:
        np.testing.assert_array_equal(z["row_index"], np.arange(3))
        np.testing.assert_allclose(z["pred_pa_unclipped"], np.exp(pred)-.1)
        np.testing.assert_allclose(z["pred_pa"], np.maximum(np.exp(pred)-.1, 0))
        np.testing.assert_array_equal(z["true_pa"], case["y_raw"])

"""Boundary-noise augmentation (2026-09-16): config contract, noisy-cloud views, dataset hook, exact no-op when off."""
from __future__ import annotations

import copy
import json

import numpy as np
import pytest
import torch

from training_wss_min import config as C
from training_wss_min import dataset as D
from training_wss_min.tests.test_density_augmentation import synthetic_case, stats, FEATURES

C1 = C.PROJECT_ROOT / "training_wss_min/configs/wss_direct_recovery_20260912/C1_s1234.json"


def noise_data(case, seed=3):
    rng = np.random.default_rng(seed)
    n = len(case["pos"])
    normals = rng.normal(size=(n, 3)).astype(np.float32)
    return {"rows": np.arange(n, dtype=np.int64), "pos": (case["pos"] + 0.01 * rng.normal(size=(n, 3))).astype(np.float32), "normals": normals,
            "overrides": {"nx_aligned": normals[:, 0], "ny_aligned": normals[:, 1], "nz_aligned": normals[:, 2],
                          "curv_k1": rng.normal(size=n).astype(np.float32), "tn_dot": rng.normal(size=n).astype(np.float32)}}


def make_cfg(prob):
    return C.DataConfig(input_features=FEATURES, wall_n_points=16, sampling="random", support_n_points=16, support_sampling="random",
                        query_mode="independent", query_n_points=12, query_sampling="random", local_geometry=True, query_patch_nsample=4,
                        noise_aug_prob=prob, noise_aug_root="/unused" if prob else None, noise_aug_levels=("0.2",) if prob else ())


def test_config_contract_default_off_and_validation():
    assert C.DataConfig().noise_aug_prob == 0.0 and C.DataConfig().noise_aug_root is None and C.DataConfig().noise_aug_levels == ()
    raw = json.loads(C1.read_text())
    C.ExpConfig.from_dict(raw)
    bad = copy.deepcopy(raw); bad["data"]["noise_aug_prob"] = 0.5
    with pytest.raises(ValueError):
        C.ExpConfig.from_dict(bad)
    bad = copy.deepcopy(raw); bad["data"].update(noise_aug_prob=0.5, noise_aug_root="/x", noise_aug_levels=["5"])
    with pytest.raises(ValueError):
        C.ExpConfig.from_dict(bad)
    ok = copy.deepcopy(raw); ok["data"].update(noise_aug_prob=0.5, noise_aug_root="/x", noise_aug_levels=["0.2", "0.3"])
    assert C.ExpConfig.from_dict(ok).data.noise_aug_levels == ("0.2", "0.3")


def test_noise_view_replaces_positions_and_cloud_inputs():
    case = synthetic_case()
    case["_full_wall_tree"] = object(); case["_full_wall_features"] = {}
    data = noise_data(case)
    sub = D.noise_view(case, data, "0.2")
    assert len(sub["pos"]) == len(case["pos"]) and np.array_equal(sub["pos"], data["pos"]) and not np.array_equal(sub["pos"], case["pos"])
    np.testing.assert_array_equal(sub["y_raw"], case["y_raw"])
    np.testing.assert_array_equal(sub["curv_k1"], data["overrides"]["curv_k1"])
    np.testing.assert_array_equal(sub["local_geometry"][:, :3], data["normals"])
    np.testing.assert_array_equal(sub["local_geometry"][:, 3:], case["local_geometry"][:, 3:])
    np.testing.assert_allclose(sub["wall_coords_raw"], data["pos"].astype(np.float64) * 100.0)
    assert "_full_wall_tree" not in sub and sub["_noise_level"] == "0.2" and np.array_equal(case["pos"], synthetic_case()["pos"])


def test_dataset_uses_the_noisy_view_when_drawn_and_is_a_noop_when_off():
    case_on = synthetic_case(); data = noise_data(case_on); case_on["_noise"] = {"0.2": data}
    ds_on = D.WSSMinDataset([case_on], make_cfg(1.0), stats(), training=True, base_seed=7)
    sample = ds_on[0]
    q = sample["query_idx"].numpy()
    np.testing.assert_allclose(sample["query_geometry"][:, :3].numpy(), data["normals"][q], rtol=1e-6)
    np.testing.assert_allclose(sample["pos"].numpy(), data["pos"][q], rtol=1e-6)
    case_off_a, case_off_b = synthetic_case(), synthetic_case()
    a = D.WSSMinDataset([case_off_a], make_cfg(0.0), stats(), training=True, base_seed=7)[0]
    b = D.WSSMinDataset([case_off_b], make_cfg(0.0), stats(), training=True, base_seed=7)[0]
    for key in ("support_pos", "support_x", "pos", "x", "y", "query_geometry"):
        torch.testing.assert_close(a[key], b[key])
    np.testing.assert_allclose(a["pos"].numpy(), case_off_a["pos"][a["query_idx"].numpy()], rtol=1e-6)

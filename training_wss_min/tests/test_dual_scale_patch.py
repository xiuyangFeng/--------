"""Dual-scale query patch (2026-09-16): config contract, fixed-mm ball sampling with mask, spacing-normalised K offsets,
zero-initialised model branch (exact no-op at init), and bit-identical default behaviour."""
from __future__ import annotations

import copy
import json

import numpy as np
import pytest
import torch

from training_wss_min import config as C
from training_wss_min import dataset as D
from training_wss_min import next_geometry as NG
from training_wss_min.local_refinement import LocalPatchFiLMRefinement
from training_wss_min.tests.test_density_augmentation import synthetic_case, stats, FEATURES

C1 = C.PROJECT_ROOT / "training_wss_min/configs/wss_direct_recovery_20260912/C1_s1234.json"


def test_config_default_off_and_validation():
    d = C.DataConfig()
    assert d.query_patch_radius_mm == 0.0 and d.query_patch_radius_k == 0 and d.query_patch_offset_norm == "mm"
    assert C.ModelConfig().query_patch_radius_k == 0
    raw = json.loads(C1.read_text())
    assert raw["model"]["query_patch_film"] and raw["data"]["query_patch_nsample"] == 16
    C.ExpConfig.from_dict(raw)
    bad = copy.deepcopy(raw); bad["data"]["query_patch_radius_k"] = 8  # no radius / no model side
    with pytest.raises(ValueError):
        C.ExpConfig.from_dict(bad)
    bad = copy.deepcopy(raw); bad["data"].update(query_patch_radius_k=8, query_patch_radius_mm=4.0); bad["model"]["query_patch_radius_k"] = 4
    with pytest.raises(ValueError):
        C.ExpConfig.from_dict(bad)
    bad = copy.deepcopy(raw); bad["data"]["query_patch_offset_norm"] = "voxel"
    with pytest.raises(ValueError):
        C.ExpConfig.from_dict(bad)
    ok = copy.deepcopy(raw); ok["data"].update(query_patch_radius_k=8, query_patch_radius_mm=4.0, query_patch_offset_norm="spacing"); ok["model"]["query_patch_radius_k"] = 8
    cfg = C.ExpConfig.from_dict(ok)
    assert cfg.model.query_patch_radius_k == 8 and D._patch_scale_kwargs(cfg.data) == {"radius_mm": 4.0, "radius_k": 8, "offset_norm": "spacing"}


def test_build_query_patch_defaults_are_bit_identical_and_ball_sampling_is_masked():
    case = synthetic_case(n=80, seed=1)
    idx = np.array([0, 5, 17, 40])
    base = NG.build_query_patch(case, idx, FEATURES, stats(), nsample=8)
    same = NG.build_query_patch(case, idx, FEATURES, stats(), nsample=8, radius_mm=0.0, radius_k=0, offset_norm="mm")
    assert set(base) == {"features", "relative_mm"} and set(same) == set(base)
    np.testing.assert_array_equal(base["features"], same["features"]); np.testing.assert_array_equal(base["relative_mm"], same["relative_mm"])
    dual = NG.build_query_patch(case, idx, FEATURES, stats(), nsample=8, radius_mm=60.0, radius_k=6, offset_norm="spacing")
    assert dual["radius_features"].shape == (4, 6, base["features"].shape[-1]) and dual["radius_relative_mm"].shape == (4, 6, 3)
    assert dual["radius_mask"].shape == (4, 6) and dual["radius_mask"].dtype == np.float32
    dist = np.linalg.norm(dual["radius_relative_mm"], axis=-1)
    assert (dist[dual["radius_mask"] > 0] <= 60.0 + 1e-4).all()
    assert (dist[dual["radius_mask"] == 0] == 0).all()  # padded with the query itself
    # spacing normalisation: per query, the median of the non-zero K offsets is exactly 1
    k_dist = np.linalg.norm(dual["relative_mm"], axis=-1)
    med = np.array([np.median(row[row > 0]) for row in k_dist])
    np.testing.assert_allclose(med, 1.0, rtol=1e-5)
    np.testing.assert_array_equal(dual["features"], base["features"])
    with pytest.raises(ValueError):
        NG.build_query_patch(case, idx, FEATURES, stats(), nsample=8, radius_mm=5.0, radius_k=0)


def test_radius_branch_is_exact_noop_at_init_and_consumes_mask():
    torch.manual_seed(0)
    nq, k, k2, d = 5, 4, 3, 7
    ref = LocalPatchFiLMRefinement(d, d, context_channels=6, out_dim=1, k=k, hidden=8)
    torch.manual_seed(0)
    dual = LocalPatchFiLMRefinement(d, d, context_channels=6, out_dim=1, k=k, hidden=8, radius_k=k2)
    dual.load_state_dict({**dual.state_dict(), **ref.state_dict()})  # shared tensors identical
    patch = {"features": torch.randn(nq, k, d), "relative_mm": torch.randn(nq, k, 3),
             "radius_features": torch.randn(nq, k2, d), "radius_relative_mm": torch.randn(nq, k2, 3),
             "radius_mask": torch.tensor([[1, 1, 1], [1, 0, 0], [1, 1, 0], [1, 1, 1], [1, 0, 0]], dtype=torch.float32)}
    qx, ctx = torch.randn(nq, d), torch.randn(nq, 6)
    torch.testing.assert_close(dual.forward_patch(patch, qx, ctx), ref.forward_patch({k_: patch[k_] for k_ in ("features", "relative_mm")}, qx, ctx))
    with torch.no_grad():
        dual.out_r.weight.fill_(0.3); dual.out_r.bias.fill_(0.1)
    out = dual.forward_patch(patch, qx, ctx)
    # masked entries must not influence the output: perturb them and compare
    patch2 = {**patch, "radius_features": patch["radius_features"].clone()}
    patch2["radius_features"][1, 1:] += 10.0
    torch.testing.assert_close(dual.forward_patch(patch2, qx, ctx), out)
    with pytest.raises(ValueError):
        dual.forward_patch({k_: patch[k_] for k_ in ("features", "relative_mm")}, qx, ctx)



def test_collate_keeps_dual_scale_patch_keys():
    cfg = C.DataConfig(input_features=FEATURES, wall_n_points=16, sampling="random", support_n_points=16, support_sampling="random",
                       query_mode="independent", query_n_points=12, query_sampling="random", local_geometry=True, query_patch_nsample=4,
                       query_patch_radius_mm=60.0, query_patch_radius_k=6, query_patch_offset_norm="spacing")
    ds = D.WSSMinDataset([synthetic_case(seed=1), synthetic_case(seed=2)], cfg, stats(), training=True, base_seed=7)
    out = D.collate([ds[0], ds[1]])
    assert set(out["query_patch"]) == {"features", "relative_mm", "radius_features", "radius_relative_mm", "radius_mask"}
    assert out["query_patch"]["radius_mask"].shape == (24, 6) and out["query_patch"]["features"].shape[:2] == (24, 4)
    plain = D.WSSMinDataset([synthetic_case(seed=1)], C.DataConfig(**{**{f.name: getattr(cfg, f.name) for f in C.DataConfig.__dataclass_fields__.values()},
                                                                        "query_patch_radius_mm": 0.0, "query_patch_radius_k": 0, "query_patch_offset_norm": "mm"}), stats(), training=True, base_seed=7)
    assert set(D.collate([plain[0]])["query_patch"]) == {"features", "relative_mm"}

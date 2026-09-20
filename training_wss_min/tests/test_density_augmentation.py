"""Density augmentation (2026-09-15): config contract, decimated-cloud views, dataset hook, exact no-op when off."""
from __future__ import annotations

import copy
import json

import numpy as np
import pytest
import torch

from training_wss_min import config as C
from training_wss_min import dataset as D

C1 = C.PROJECT_ROOT / "training_wss_min/configs/wss_direct_recovery_20260912/C1_s1234.json"
FEATURES = ("x", "y", "z", "abscissa_norm", "local_radius", "curvature", "log_local_radius", "nx_aligned", "ny_aligned",
            "nz_aligned", "curv_k1", "tn_dot")


def synthetic_case(n=60, seed=0):
    rng = np.random.default_rng(seed)
    pos = rng.normal(size=(n, 3)).astype(np.float32)
    case = dict(unit_id="AG/fast/SYN", case="SYN", cohort="AG/fast", pos=pos, wall_coords_raw=(pos * 100).astype(np.float64),
                y_raw=rng.lognormal(size=n).astype(np.float32), y_norm=rng.normal(size=n).astype(np.float32),
                abscissa_norm=rng.random(n).astype(np.float32), local_radius=(rng.random(n) + 5).astype(np.float32),
                curvature=rng.normal(size=n).astype(np.float32), coord_scale=np.full(n, 100.0, dtype=np.float32),
                coord_scale_scalar=100.0, radius_gradient=rng.normal(size=n).astype(np.float32),
                nx_aligned=rng.normal(size=n).astype(np.float32), ny_aligned=rng.normal(size=n).astype(np.float32),
                nz_aligned=rng.normal(size=n).astype(np.float32), curv_k1=rng.normal(size=n).astype(np.float32),
                tn_dot=rng.normal(size=n).astype(np.float32), bundle_path="/nonexistent/bundle.npz", peak_step=1162)
    case["log_local_radius"] = np.log(case["local_radius"]).astype(np.float32)
    geometry = np.zeros((n, 8), dtype=np.float32); geometry[:, 2] = 1.0; geometry[:, 3] = 1.0; geometry[:, 6] = 6.0; geometry[:, 7] = 100.0
    case["local_geometry"] = geometry
    return case


def level_data(case, rows, seed=1):
    rng = np.random.default_rng(seed)
    normals = rng.normal(size=(len(rows), 3)).astype(np.float32)
    return {"rows": np.asarray(rows, dtype=np.int64), "normals": normals,
            "overrides": {"nx_aligned": normals[:, 0], "ny_aligned": normals[:, 1], "nz_aligned": normals[:, 2],
                          "curv_k1": rng.normal(size=len(rows)).astype(np.float32), "tn_dot": rng.normal(size=len(rows)).astype(np.float32)}}


def test_config_contract_default_off_and_validation():
    assert C.DataConfig().density_aug_prob == 0.0 and C.DataConfig().density_aug_root is None
    raw = json.loads(C1.read_text())
    C.ExpConfig.from_dict(raw)
    bad = copy.deepcopy(raw); bad["data"]["density_aug_prob"] = 0.5
    with pytest.raises(ValueError):  # needs root + levels
        C.ExpConfig.from_dict(bad)
    bad = copy.deepcopy(raw); bad["data"].update(density_aug_prob=0.5, density_aug_root="/x", density_aug_levels=[100])
    with pytest.raises(ValueError):
        C.ExpConfig.from_dict(bad)
    ok = copy.deepcopy(raw); ok["data"].update(density_aug_prob=0.6, density_aug_root="/x", density_aug_levels=[70, 50, 35, 25])
    cfg = C.ExpConfig.from_dict(ok)
    assert cfg.data.density_aug_levels == (70, 50, 35, 25)


def test_density_view_subsets_and_overrides():
    case = synthetic_case()
    case["_full_wall_tree"] = object(); case["_full_wall_features"] = {}
    rows = np.array([0, 3, 7, 11, 20, 59])
    data = level_data(case, rows)
    sub = D.density_view(case, data, 35)
    assert len(sub["pos"]) == len(rows) and np.array_equal(sub["pos"], case["pos"][rows])
    np.testing.assert_array_equal(sub["y_raw"], case["y_raw"][rows])
    np.testing.assert_array_equal(sub["curv_k1"], data["overrides"]["curv_k1"])
    np.testing.assert_array_equal(sub["local_geometry"][:, :3], data["normals"])
    np.testing.assert_array_equal(sub["local_geometry"][:, 3:], case["local_geometry"][rows, 3:])
    assert "_full_wall_tree" not in sub and "_full_wall_features" not in sub and sub["_density_level"] == 35
    assert sub["coord_scale_scalar"] == 100.0 and len(case["pos"]) == 60  # original untouched


def make_cfg(prob):
    cfg = C.DataConfig(input_features=FEATURES, wall_n_points=16, sampling="random", support_n_points=16, support_sampling="random",
                       query_mode="independent", query_n_points=12, query_sampling="random", local_geometry=True,
                       query_patch_nsample=4, density_aug_prob=prob, density_aug_root="/unused" if prob else None,
                       density_aug_levels=(35,) if prob else ())
    return cfg


def stats():
    return {f: {"mean": 0.0, "std": 1.0, "transform": "none"} for f in FEATURES if f not in ("x", "y", "z")}


def test_dataset_uses_the_decimated_view_when_drawn_and_is_a_noop_when_off():
    rows = np.arange(0, 60, 2)
    case_on = synthetic_case(); case_on["_density"] = {35: level_data(case_on, rows)}
    ds_on = D.WSSMinDataset([case_on], make_cfg(1.0), stats(), training=True, base_seed=7)
    sample = ds_on[0]
    assert int(sample["support_idx"].max()) < len(rows) and int(sample["query_idx"].max()) < len(rows)
    # geometry normals of the query points come from the decimated-cloud recomputation
    q = sample["query_idx"].numpy()
    np.testing.assert_allclose(sample["query_geometry"][:, :3].numpy(), case_on["_density"][35]["normals"][q], rtol=1e-6)
    case_off_a, case_off_b = synthetic_case(), synthetic_case()
    ds_off = D.WSSMinDataset([case_off_a], make_cfg(0.0), stats(), training=True, base_seed=7)
    ds_ref = D.WSSMinDataset([case_off_b], make_cfg(0.0), stats(), training=True, base_seed=7)
    a, b = ds_off[0], ds_ref[0]
    for key in ("support_pos", "support_x", "pos", "x", "y", "query_geometry"):
        torch.testing.assert_close(a[key], b[key])
    assert int(a["support_idx"].max()) >= len(rows)  # full cloud when augmentation is off

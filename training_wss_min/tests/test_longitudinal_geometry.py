from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from training_wss_min import longitudinal_geometry as L
from training_wss_min import config as C
from training_wss_min import dataset as D


def _case(unit="AG/fast/A"):
    return {
        "unit_id": unit,
        "pos": np.zeros((3, 3), np.float32),
        "geom_ref_log_area": np.array([1., np.nan, 3.], np.float32),
        "geom_ref_log_area_valid": np.array([1, 0, 1], bool),
        "_longitudinal_partition": "train",
    }


def test_value_and_mask_are_required_and_missing_is_zero_after_normalization():
    with pytest.raises(ValueError):
        L.validate_feature_pairs(("geom_ref_log_area",))
    case = _case()
    stats = L.compute_stats([case], ("geom_ref_log_area", "geom_ref_log_area_valid"))
    out = L.feature_column(case, np.arange(3), "geom_ref_log_area", stats)
    assert np.isfinite(out).all() and out[1] == 0.0
    assert L.feature_column(case, np.arange(3), "geom_ref_log_area_valid", stats).tolist() == [1., 0., 1.]


def test_stats_reject_all_invalid_and_non_train_leakage():
    case = _case()
    case["geom_ref_log_area_valid"][:] = False
    with pytest.raises(ValueError, match="all training values"):
        L.compute_stats([case], ("geom_ref_log_area", "geom_ref_log_area_valid"))
    case = _case(); case["_longitudinal_partition"] = "test"
    with pytest.raises(ValueError, match="only explicit train"):
        L.compute_stats([case], ("geom_ref_log_area", "geom_ref_log_area_valid"))


def test_sidecar_export_contract_keeps_all_view_rows(tmp_path):
    from wss_v5.views.wall_longitudinal_v1 import export_case
    candidate = Path("outputs/wss_v6_geometry_candidate_20260915_opt_v3_final")
    view = Path("data_wss_v5/views_v5_1/wss_min_view_v1")
    result = export_case("AAA/ruputer/FENG_LI_XIN", candidate, view, tmp_path)
    assert result["exported_wall_points"] == result["view_wall_points"]
    with np.load(tmp_path / "AAA/ruputer/FENG_LI_XIN/features.npz") as z:
        assert np.isfinite(z["wall_geom_ref_log_area"][z["wall_geom_ref_log_area_valid"]]).all()
        assert np.isnan(z["wall_geom_ref_log_area"][~z["wall_geom_ref_log_area_valid"]]).all()
        assert np.array_equal(z["wall_geom_ref_log_area_valid"], z["wall_geom_ref_log_area_valid"].astype(bool))


def test_config_requires_longitudinal_pairs_and_defaults_are_unchanged():
    assert C.ExpConfig().data.input_features == ("x", "y", "z")
    with pytest.raises(ValueError, match="requires both"):
        C.ExpConfig.from_dict({"data": {"input_features": ["x", "geom_ref_roundness"]}})


def test_frozen_stats_require_matching_train_units(tmp_path):
    case = _case()
    stats = L.compute_stats([case], ("geom_ref_log_area", "geom_ref_log_area_valid"))
    p = tmp_path / "stats.json"; p.write_text(json.dumps(stats))
    L.validate_frozen_stats(stats, ("geom_ref_log_area", "geom_ref_log_area_valid"), [case["unit_id"]])
    with pytest.raises(ValueError, match="do not match"):
        L.validate_frozen_stats(stats, ("geom_ref_log_area", "geom_ref_log_area_valid"), ["other"])


def test_valid_true_zero_is_distinguished_from_absent_value_and_labels_retained():
    c=_case();names=('x','geom_ref_log_area','geom_ref_log_area_valid')
    c['geom_ref_log_area']=np.array([0.,np.nan,2.])
    c['y_raw']=np.array([10.,20.,30.]);c['y_norm']=c['y_raw'].copy()
    before=c['y_raw'].copy();stats=D.compute_feature_stats([c],names)
    x=D.build_features(c,np.arange(3),names,stats)
    assert x[0,1]<0 and x[0,2]==1  # A real log-area zero remains a measurement.
    assert x[1,1]==0 and x[1,2]==0
    assert np.array_equal(c['y_raw'],before) and len(x)==len(before)


def test_optional_longitudinal_loader_keys_match_config_without_changing_defaults(tmp_path):
    cfg=C.ExpConfig.from_dict({'data':{'input_features':['x','geom_ref_roundness','geom_ref_roundness_valid'],
                                     'point_features_root':str(tmp_path)}})
    assert C.v6_point_features(cfg)==('geom_ref_roundness','geom_ref_roundness_valid')
    c=_case();old=D.compute_feature_stats([c],('x','y','z'))
    assert old=={} and np.array_equal(D.build_features(c,np.arange(3),('x','y','z'),old),c['pos'])


def test_geometry_sidecar_identity_and_raw_coordinates_are_checked(tmp_path):
    from wss_v5.views.wall_longitudinal_v1 import align_rows
    ids=np.array([11,12,13]);xyz=np.arange(9,dtype=np.float32).reshape(3,3)
    assert align_rows(ids,xyz,ids[::-1],xyz[::-1]).tolist()==[2,1,0]
    bad=xyz.copy();bad[0,0]+=.01
    with pytest.raises(ValueError,match='different original'):
        align_rows(ids,xyz,ids,bad)
    with pytest.raises(ValueError,match='duplicate'):
        align_rows(np.array([11,11,13]),xyz,ids,xyz)

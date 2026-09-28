"""build_case reads only the caps of the oriented cloud (2026-09-28).  ``geometry.cap_records`` must return exactly
``build_oriented_cloud(pts, atlas, normals_out=normals, calibrate=False)[1]["caps"]`` — same keys, order, types and
values (Tier A) — and ``build_case`` must be byte-identical to the full-cloud path it replaced."""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
from test_morphology import _atlas, _tube  # noqa: E402  (synthetic tube + atlas shared with the morphology tests)

import wss_features.cloud as cloud  # noqa: E402
from wss_deploy import geometry as G  # noqa: E402
from wss_features.cloud import build_oriented_cloud, median_spacing, patch_atlas_end_radius  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
RULE = {"a": 1.1504325542818956, "b": 0.14732857530580903}
FEATURES = ["x", "y", "z", "abscissa_norm", "local_radius", "curvature", "log_local_radius", "rho", "theta_sin", "theta_cos",
            "dr_ds", "dist_to_junction_mm", "dist_to_endpoint_mm", "end_zone", "nx_aligned", "ny_aligned", "nz_aligned",
            "curv_k1", "curv_k2", "curv_gauss", "curvedness", "curv_k1_c", "curv_k2_c", "curv_gauss_c", "tn_dot",
            "log_q_branch_murray", "log_tau0_murray"]


def _identical(a, b, path="value"):
    """Exact structural equality: arrays by dtype, shape and bytes; floats by repr; containers by order."""
    if isinstance(a, np.ndarray) or isinstance(b, np.ndarray):
        assert isinstance(a, np.ndarray) and isinstance(b, np.ndarray), path
        assert a.dtype == b.dtype and a.shape == b.shape and a.tobytes() == b.tobytes(), path
        return
    assert type(a) is type(b), f"{path}: {type(a).__name__} vs {type(b).__name__}"
    if isinstance(a, dict):
        assert list(a) == list(b), f"{path}: keys {list(a)} vs {list(b)}"
        for key in a:
            _identical(a[key], b[key], f"{path}.{key}")
    elif isinstance(a, (list, tuple)):
        assert len(a) == len(b), f"{path}: length"
        for i, (x, y) in enumerate(zip(a, b)):
            _identical(x, y, f"{path}[{i}]")
    elif isinstance(a, float):
        assert repr(a) == repr(b), f"{path}: {a!r} vs {b!r}"
    else:
        assert a == b, f"{path}: {a!r} vs {b!r}"


def _wall(branches=(0, 1, 2), seed=3):
    """Noisy wall points of the root tube (with its sac) and the two child tubes of ``_atlas``."""
    tubes = {0: (120.0, 8.0, 0.0, 0.0), 1: (30.0, 5.0, -14.0, 120.0), 2: (30.0, 5.0, 14.0, 120.0)}
    pts = np.concatenate([_tube(sid, *tubes[sid])[0] for sid in branches])
    return pts + np.random.default_rng(seed).normal(scale=0.05, size=pts.shape)


def _inflated_atlas():
    """Left outlet endpoint radius tripled (escaped inscribed sphere): the end-radius patch and band rule engage."""
    atlas = _atlas()
    seg = atlas.col("segment_id").astype(int)
    atlas.table[np.flatnonzero(seg == 1)[-1], atlas.columns.index("radius_mm")] = 15.0
    return atlas


CASES = {"three_branches": (lambda: _wall(), _atlas), "right_branch_missing": (lambda: _wall((0, 1)), _atlas),
         "inflated_end": (lambda: _wall(), _inflated_atlas), "float32_points": (lambda: _wall().astype(np.float32), _atlas)}


@pytest.mark.parametrize("name", sorted(CASES))
def test_cap_records_equal_the_full_oriented_cloud_caps(name):
    make_pts, make_atlas = CASES[name]
    pts, atlas = make_pts(), make_atlas()
    patch_atlas_end_radius(atlas, pts)
    normals = G.point_geometry(pts, atlas)[0]
    expected = build_oriented_cloud(pts, atlas, normals_out=normals, calibrate=False)[1]["caps"]
    got, spacing = G.cap_records(pts, atlas, normals)
    _identical(expected, got, "caps")
    if pts.dtype == np.float64:   # the spacing is only handed back when it is median_spacing(pts) itself
        assert spacing is not None and repr(spacing) == repr(median_spacing(pts))
    else:
        assert spacing is None
    if name == "right_branch_missing":
        assert [c["cap_source"] for c in got][-1] != "rim_plane_fit"


@pytest.mark.parametrize("name", sorted(CASES))
def test_build_case_is_identical_to_the_full_cloud_path(name, monkeypatch):
    make_pts, make_atlas = CASES[name]
    pts = make_pts()
    new = G.build_case(pts, make_atlas(), FEATURES, capfit_rule=RULE)
    monkeypatch.setattr(G, "_VIRTUAL_CAPS", None)   # the original path: full oriented cloud + median_spacing(pts)
    calls = []
    monkeypatch.setattr(G, "build_oriented_cloud", lambda *a, **k: calls.append(1) or build_oriented_cloud(*a, **k))
    old = G.build_case(pts, make_atlas(), FEATURES, capfit_rule=RULE)
    assert calls == [1]
    _identical(old, new, "build_case")


def test_guard_falls_back_when_the_cap_helper_is_missing_or_different(monkeypatch):
    assert G._caps_helper() is cloud.virtual_caps
    monkeypatch.setattr(cloud, "virtual_caps", lambda atlas, wall_points_mm, spacing: [])
    assert G._caps_helper() is None
    monkeypatch.delattr(cloud, "virtual_caps")
    assert G._caps_helper() is None


def test_no_opening_raises_like_the_full_cloud():
    atlas = _atlas()
    atlas.segments[:] = [{**s, "starts_at_root": False, "ends_at_leaf": False} for s in atlas.segments]
    pts = _wall()
    normals = np.tile([1.0, 0.0, 0.0], (len(pts), 1))
    with pytest.raises(ValueError) as full:
        build_oriented_cloud(pts, atlas, normals_out=normals, calibrate=False)
    with pytest.raises(ValueError) as caps_only:
        G.cap_records(pts, atlas, normals)
    assert str(full.value) == str(caps_only.value)


# ----------------------------------------------------------------------------- a real finished job (read-only)
def _real_job() -> Path | None:
    configured = os.environ.get("WSS_DEPLOY_CAPS_JOB")
    candidates = [Path(configured)] if configured else sorted((ROOT / "outputs" / "wss_deploy_jobs").glob("*/"))
    for job in candidates:
        try:
            ok = json.loads((job / "job.json").read_text(encoding="utf-8")).get("status") == "done"
        except (OSError, ValueError):
            continue
        if ok and (job / "centerline" / "atlas.npz").is_file() and (job / "summary.json").is_file() \
                and any((job / "geometry_cache").glob("resample-*.npz")):
            return job
    return None


def test_real_job_caps_and_build_case(monkeypatch):
    job = _real_job()
    if job is None:
        pytest.skip("no finished deployment job with a centreline and a cached resampled cloud")
    from wss_deploy import centerline as CL
    mapping = json.loads((job / "summary.json").read_text(encoding="utf-8"))["mapping"]
    with np.load(next((job / "geometry_cache").glob("resample-*.npz")), allow_pickle=False) as data:
        pts = data["points"]
    atlas_of = lambda: CL.apply_mapping(CL.load_vessel_geom_atlas(job / "centerline"), mapping)
    atlas = atlas_of()
    patch_atlas_end_radius(atlas, pts)
    geometry = G.point_geometry(pts, atlas)
    expected = build_oriented_cloud(pts, atlas, normals_out=geometry[0], calibrate=False)[1]["caps"]
    got, spacing = G.cap_records(pts, atlas, geometry[0])
    _identical(expected, got, "caps")
    assert repr(spacing) == repr(median_spacing(pts))
    provider = lambda *_: geometry
    new = G.build_case(pts, atlas_of(), FEATURES, capfit_rule=RULE, point_geometry_provider=provider)
    monkeypatch.setattr(G, "_VIRTUAL_CAPS", None)
    old = G.build_case(pts, atlas_of(), FEATURES, capfit_rule=RULE, point_geometry_provider=provider)
    _identical(old, new, "build_case")

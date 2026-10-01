"""Analysis layer on a synthetic tube with a spherical sac: profiles, findings, trust, frame."""
from __future__ import annotations

import json

import numpy as np
import pytest
from scipy.spatial import cKDTree

from wss_deploy import analysis as A
from wss_features.atlas import Atlas

COLUMNS = ["segment_id", "sample_index", "x_mm", "y_mm", "z_mm", "tangent_x", "tangent_y", "tangent_z", "radius_mm",
           "s_local_mm", "s_from_root_mm", "curvature_per_mm", "dr_ds", "dist_to_junction_mm", "dist_to_endpoint_mm",
           "junction_mask", "endpoint_mask", "end_zone", "is_child_duplicate"]
NAMES = {"0": "主动脉", "1": "左髂总", "2": "右髂总"}


def _atlas(lengths=(40.0, 20.0, 20.0), step=0.5, radii=(8.0, 4.0, 3.0)):
    """Root along +z from the origin, two children continuing along +z from its end (offset in x)."""
    rows, segments, offset = [], [], 0.0
    ends = {}
    for sid, (length, radius) in enumerate(zip(lengths, radii)):
        n = int(round(length / step)) + 1
        parent = -1 if sid == 0 else 0
        x0 = 0.0 if sid == 0 else (-10.0 if sid == 1 else 10.0)
        z0 = 0.0 if sid == 0 else lengths[0]
        s_offset = 0.0 if sid == 0 else lengths[0]
        for k in range(n):
            s = k * step
            r = radius
            if sid == 0 and 15 <= s <= 25:      # sac in the middle of the root
                r = radius + 6.0 * np.sin(np.pi * (s - 15) / 10)
            if sid == 2 and 8 <= s <= 12:       # stenosis in the right child
                r = radius * 0.5
            rows.append([sid, k, x0, 0.0, z0 + s, 0.0, 0.0, 1.0, r, s, s_offset + s, 0.0, 0.0,
                         min(s, length - s), min(s, length - s), 0.0, float(k == 0 or k == n - 1), 0.0, 0.0])
        segments.append({"segment_id": sid, "parent_id": parent, "starts_at_root": sid == 0, "ends_at_leaf": sid != 0,
                         "outlet_name": {1: "out-le", 2: "out-re"}.get(sid, ""), "length_mm": length, "s_offset_mm": s_offset,
                         "descendant_outlets": []})
    table = np.asarray(rows, dtype=np.float64)
    xyz = table[:, 2:5]
    n_rows = len(table)
    frame_n = np.tile([1.0, 0.0, 0.0], (n_rows, 1)); frame_b = np.tile([0.0, 1.0, 0.0], (n_rows, 1))
    return Atlas(table=table, columns=COLUMNS, segments=segments, semantic_of_segment={0: 0, 1: 1, 2: 2},
                 frame_n=frame_n, frame_b=frame_b, tree_rows=np.arange(n_rows), tree=cKDTree(xyz), provenance={})


def _wall_points(atlas, per_ring=48):
    """Points on the tube wall at every atlas sample; feats mirror the atlas row."""
    seg = atlas.col("segment_id").astype(int); s_local = atlas.col("s_local_mm"); s_root = atlas.col("s_from_root_mm")
    radius = atlas.col("radius_mm"); xyz = atlas.xyz
    pts, feats = [], {"segment_id": [], "s_local_mm": [], "s_from_root_mm": [], "radius_mm": [], "dist_to_junction_mm": []}
    angles = np.linspace(0, 2 * np.pi, per_ring, endpoint=False)
    for i in range(len(xyz)):
        for a in angles:
            pts.append(xyz[i] + radius[i] * np.array([np.cos(a), np.sin(a), 0.0]))
            feats["segment_id"].append(seg[i]); feats["s_local_mm"].append(s_local[i]); feats["s_from_root_mm"].append(s_root[i])
            feats["radius_mm"].append(radius[i]); feats["dist_to_junction_mm"].append(atlas.col("dist_to_junction_mm")[i])
    return np.asarray(pts), {k: np.asarray(v) for k, v in feats.items()}


@pytest.fixture(scope="module")
def tube():
    atlas = _atlas()
    pts, feats = _wall_points(atlas)
    return atlas, pts, feats


def test_profiles_bins_follow_arc_length_and_keep_empty_bins_null(tube):
    atlas, pts, feats = tube
    wss = 1.0 + feats["s_from_root_mm"] / 10.0          # increases along the tree
    keep = ~((feats["segment_id"] == 0) & (feats["s_local_mm"] >= 30) & (feats["s_local_mm"] < 34))  # gap -> empty bins
    out = A.profiles(atlas, feats, {"wss": wss}, point_mask=keep, bin_mm=2.0, branch_names=NAMES)
    assert out["schema_version"] == "wss-deploy.profiles/v1" and out["kind"] == "wss"
    assert [b["name"] for b in out["branches"]] == ["主动脉", "左髂总", "右髂总"]
    root = out["branches"][0]
    assert root["n_bins"] == 20 and len(root["s_local_mm"]) == 20 and root["parent_id"] == -1
    assert root["radius_mm"][9] > root["radius_mm"][0]       # sac radius from the atlas
    means = root["wss"]["mean_pa"]
    assert means[15] is None and means[16] is None and root["wss"]["n"][15] == 0
    finite = [m for m in means if m is not None]
    assert finite == sorted(finite)
    assert root["s_from_root_mm"][0] == pytest.approx(0.75, abs=0.5)
    child = out["branches"][1]
    assert child["s_from_root_mm"][0] > 40.0 and child["n_bins"] == 10
    json.dumps(out, allow_nan=False)                          # JSON safe


def test_profiles_volume_kind(tube):
    atlas, pts, feats = tube
    speed = np.full(len(pts), 0.5); pressure = 100.0 - feats["s_from_root_mm"]
    out = A.profiles(atlas, feats, {"speed": speed, "pressure": pressure}, branch_names=NAMES)
    assert out["kind"] == "volume"
    block = out["branches"][0]["volume"]
    assert block["speed_max_m_s"][0] == pytest.approx(0.5) and block["pressure_mean_pa"][0] > block["pressure_mean_pa"][-1]


def test_findings_wall_clusters_geometry_and_severity(tube):
    atlas, pts, feats = tube
    n = len(pts)
    wss = np.full(n, 2.0)
    hot = (feats["segment_id"] == 1) & (feats["s_local_mm"] >= 5) & (feats["s_local_mm"] <= 7)   # 5 rings × 48 = 240 points
    wss[hot] = 12.0
    low = (feats["segment_id"] == 0) & (feats["s_local_mm"] >= 15) & (feats["s_local_mm"] <= 25)  # the sac, 21 rings
    wss[low] = 0.1
    spacing = 1.0   # linking radius 2 mm: ring neighbours (≤ 1.9 mm at the sac) and ring steps (0.5 mm) connect
    geometry = {"主动脉": {"stenosis_index": 0.05}, "左髂总": {"stenosis_index": 0.02}, "右髂总": {"stenosis_index": 0.5}}
    out = A.findings_wall(pts, wss, feats, atlas, geometry, thresholds=[0.4, 4, 7], total_area_mm2=1000.0,
                          spacing_mm=spacing, branch_names=NAMES, min_cluster_points=20)
    kinds = [i["kind"] for i in out["items"]]
    assert kinds.count("high_wss_cluster") == 1 and kinds.count("low_wss_cluster") == 1
    # 2026-09-30 (U12): the global maximum lies in the hot cluster, so it is merged into it, not listed on its own.
    assert kinds.count("max_wss") == 0 and kinds.count("max_diameter") == 1 and kinds.count("min_radius") == 1
    high = next(i for i in out["items"] if i["kind"] == "high_wss_cluster")
    assert high["branch"] == "左髂总" and high["n_points"] == 240 and high["value"] == pytest.approx(12.0)
    assert high["contains_global_max"] is True and high["global_max_pa"] == pytest.approx(12.0) and "含全场最大值" in high["label"]
    # U11: no same-protocol cohort reference → 提示, marked ungraded (the 7 Pa level no longer grades).
    assert high["area_mm2"] == pytest.approx(1000.0 * 240 / n) and high["severity"] == "note" and high["grading"] == "no_reference"
    assert out["high_grading"]["status"] == "no_reference" and out["listing_rules"]["global_max_merged"] is True
    assert len(high["point_indices"]) == 240 and high["extent_mm"] > 0
    lo = next(i for i in out["items"] if i["kind"] == "low_wss_cluster")
    assert lo["branch"] == "主动脉" and lo["n_points"] == 21 * 48 and lo["severity"] == ("attention" if out["low_wss_total_fraction"] > 0.2 else "note")
    assert out["low_wss_total_fraction"] == pytest.approx(21 * 48 / n)
    md = next(i for i in out["items"] if i["kind"] == "max_diameter")
    assert md["branch"] == "主动脉" and md["value"] == pytest.approx(28.0) and md["severity"] == "info" and "管腔" in md["label"]
    assert "中心线内切半径" in md["definition"] and "source" not in md
    mr = next(i for i in out["items"] if i["kind"] == "min_radius")
    assert mr["branch"] == "右髂总" and mr["value"] == pytest.approx(1.5) and mr["stenosis_index"] == pytest.approx(0.5)
    assert [i["id"] for i in out["items"]] == [f"F{k}" for k in range(1, len(out["items"]) + 1)]
    assert all(i["definition"] for i in out["items"])
    json.dumps(out, allow_nan=False)


MORPHOLOGY_BLOCK = {"schema_version": "wss-deploy.morphology/v1",
                    "aorta": {"segment_id": 0, "name": "主动脉",
                              "max": {"max_diameter_mm": 31.5, "equivalent_diameter_mm": 29.0, "area_mm2": 660.5,
                                      "s_from_root_mm": 20.0, "xyz_mm": [0.0, 0.0, 20.0], "oblique": True}}}


def test_max_diameter_finding_switches_to_the_wall_mesh_when_morphology_is_given(tube):
    """§17.1: with a morphology block the diameter comes from the mesh sections, ids and order unchanged."""
    atlas, pts, feats = tube
    wss = np.full(len(pts), 2.0)
    kwargs = dict(thresholds=[0.4, 4, 7], total_area_mm2=1000.0, spacing_mm=1.0, branch_names=NAMES)
    base = A.findings_wall(pts, wss, feats, atlas, {}, **kwargs)
    with_morphology = A.findings_wall(pts, wss, feats, atlas, {}, morphology=MORPHOLOGY_BLOCK, **kwargs)
    assert [i["id"] for i in base["items"]] == [i["id"] for i in with_morphology["items"]]
    assert [i["kind"] for i in base["items"]] == [i["kind"] for i in with_morphology["items"]]
    item = next(i for i in with_morphology["items"] if i["kind"] == "max_diameter")
    assert item["value"] == pytest.approx(31.5) and item["xyz_mm"] == [0.0, 0.0, 20.0]
    assert item["s_from_root_mm"] == pytest.approx(20.0) and item["branch"] == "主动脉" and item["segment_id"] == 0
    assert item["definition"] == A.MAX_DIAMETER_DEFINITION and item["source"] == "morphology"
    assert item["equivalent_diameter_mm"] == pytest.approx(29.0) and item["oblique"] is True
    assert item["severity"] == "info" and item["extent_mm"] > 0
    json.dumps(with_morphology, allow_nan=False)
    # apply_morphology patches an existing list in place, keeping the id and rank of the item it replaces.
    patched = A.apply_morphology(A.findings_wall(pts, wss, feats, atlas, {}, **kwargs), MORPHOLOGY_BLOCK, NAMES)
    replaced = next(i for i in patched["items"] if i["kind"] == "max_diameter")
    original = next(i for i in base["items"] if i["kind"] == "max_diameter")
    assert replaced["id"] == original["id"] and replaced["rank"] == original["rank"] and replaced["value"] == pytest.approx(31.5)
    # A malformed or absent block leaves the centreline implementation in place.
    for bad in (None, {}, {"aorta": {}}, {"aorta": {"max": {"max_diameter_mm": None, "xyz_mm": [0, 0, 0]}}}):
        item = next(i for i in A.findings_wall(pts, wss, feats, atlas, {}, morphology=bad, **kwargs)["items"]
                    if i["kind"] == "max_diameter")
        assert "source" not in item and "中心线内切半径" in item["definition"]


def _interior_grid(atlas):
    """Regular 1 mm grid inside the root tube and both children; feats from the tube axis."""
    pts, feats = [], {"segment_id": [], "s_local_mm": [], "s_from_root_mm": [], "radius_mm": []}
    for sid, x0, z0, length in ((0, 0.0, 0.0, 40.0), (1, -10.0, 40.0, 20.0), (2, 10.0, 40.0, 20.0)):
        for z in np.arange(0.0, length + 0.5, 1.0):
            for x in np.arange(-2.0, 2.5, 1.0):
                for y in np.arange(-2.0, 2.5, 1.0):
                    pts.append([x0 + x, y, z0 + z]); feats["segment_id"].append(sid); feats["s_local_mm"].append(z)
                    feats["s_from_root_mm"].append(z0 + z); feats["radius_mm"].append(3.0)
    return np.asarray(pts), {k: np.asarray(v) for k, v in feats.items()}


def test_findings_volume_pressure_drop_sign_and_low_speed(tube):
    atlas, _, _ = tube
    pts, feats = _interior_grid(atlas)
    n = len(pts)
    pressure = 200.0 - 2.0 * feats["s_from_root_mm"]                 # drops along the flow
    speed = np.full(n, 1.0)
    slow = (feats["segment_id"] == 0) & (feats["s_local_mm"] >= 17) & (feats["s_local_mm"] <= 23)   # 7 planes × 25 < 10 % of points
    speed[slow] = 0.01
    speed[0] = 3.0
    out = A.findings_volume(pts, speed, pressure, feats, atlas, {"主动脉": {"stenosis_index": 0.0}}, branch_names=NAMES, min_cluster_points=20)
    kinds = [i["kind"] for i in out["items"]]
    assert kinds[:2] == ["max_speed", "min_pressure"]
    drops = [i for i in out["items"] if i["kind"] == "pressure_drop"]
    assert {d["branch"] for d in drops} == {"主动脉", "左髂总", "右髂总"}
    assert all(d["value"] > 0 for d in drops)
    root_drop = next(d for d in drops if d["branch"] == "主动脉")
    assert root_drop["value"] == pytest.approx(2.0 * (40 - 4), abs=2.0 * 1.0)
    assert root_drop["proximal_n"] == 5 * 25 and root_drop["distal_n"] == 5 * 25
    low = [i for i in out["items"] if i["kind"] == "low_speed_region"]
    assert len(low) == 1 and low[0]["branch"] == "主动脉" and low[0]["n_points"] == 7 * 25
    assert next(i for i in out["items"] if i["kind"] == "max_speed")["value"] == pytest.approx(3.0)
    json.dumps(out, allow_nan=False)


def test_trust_wall_bits_and_fractions(tube):
    atlas, pts, feats = tube
    vertices = pts[::3].copy()
    values = np.ones(len(vertices)); values[:5] = np.nan
    variation = np.zeros(len(pts)); variation[30:60] = 0.05          # rough points -> nearest vertices
    vseg = feats["segment_id"][::3]
    reference = {"checks": [{"path": "geometry.右髂总.radius_min_mm", "status": "review"},
                            {"path": "geometry.主动脉.length_mm", "status": "pass"}]}
    bits, block = A.trust_wall(vertices, values, pts, variation, vseg, reference, NAMES)
    assert bits.dtype == np.uint8 and bits.shape == (len(vertices),)
    assert np.all(bits[:5] & 1) and not np.any(bits[5:] & 1)
    rough_vertices = np.flatnonzero(bits & 2)
    np.testing.assert_array_equal(rough_vertices, np.arange(10, 20))   # vertices whose own point (index 3i) is rough
    assert np.all((bits[vseg == 2] & 4) > 0) and not np.any(bits[vseg != 2] & 4)
    assert block["schema_version"] == "wss-deploy.trust/v1" and block["bits"] == {"1": "interpolation_uncovered", "2": "rough_surface", "4": "geometry_out_of_range"}
    assert all(0.0 <= f <= 1.0 for f in block["fractions"].values())
    assert block["fractions"]["interpolation_uncovered"] == pytest.approx(5 / len(vertices))
    assert block["review_segments"] == [2] and {s["bit"] for s in block["sources"]} == {1, 2, 4}
    assert A.review_segments(None, NAMES) == set()


def test_trust_volume_bits(tube):
    atlas, pts, feats = tube
    vertices = pts[::4]
    vp = np.ones(len(vertices)); vp[-1] = np.nan
    interior = pts[feats["segment_id"] == 0][::2] * np.array([0.5, 0.5, 1.0])      # inside the root tube
    interior = np.vstack([interior, [[0.0, 0.0, 200.0]]])                           # isolated far point
    iseg = np.zeros(len(interior), dtype=int)
    caps = [{"center_mm": [0.0, 0.0, 0.0], "radius_mm": 8.0}]
    mesh_bits, cloud_bits, block = A.trust_volume(vertices, vp, np.zeros(len(vertices), int), interior, iseg, caps, None, NAMES)
    assert mesh_bits[-1] & 1 and not np.any(mesh_bits[:-1] & 1)
    assert cloud_bits[-1] & 8 and (cloud_bits[:-1] & 8).mean() < 0.05
    near = np.linalg.norm(interior - np.array([0.0, 0.0, 0.0]), axis=1) < 16.0
    assert np.all(cloud_bits[near] & 16) and not np.any(cloud_bits[~near] & 16)
    assert not np.any(cloud_bits & 4)
    assert block["family"] == "volume" and 0 < block["fractions"]["near_opening"] < 1 and block["n_interior"] == len(interior)


def test_frame_transform_block():
    R = np.eye(3); block = A.frame_transform(R, [1, 2, 3], source="test", direction_source=None)
    assert block["rotation"] == np.eye(3).tolist() and block["origin_mm"] == [1.0, 2.0, 3.0] and block["direction_source"] == "unknown_stl"
    with pytest.raises(ValueError):
        A.frame_transform(np.full((3, 3), np.nan), [0, 0, 0], source="x", direction_source="s")

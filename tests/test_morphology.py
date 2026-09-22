"""§17.1 morphology: cross-sections on a synthetic tube with a sac, sac/neck detection, volumes, trimming."""
from __future__ import annotations

import json
import math

import numpy as np
import pytest
from scipy.spatial import cKDTree

from wss_deploy import morphology as MO
from wss_features.atlas import Atlas

COLUMNS = ["segment_id", "sample_index", "x_mm", "y_mm", "z_mm", "tangent_x", "tangent_y", "tangent_z", "radius_mm",
           "s_local_mm", "s_from_root_mm", "curvature_per_mm", "dr_ds", "dist_to_junction_mm", "dist_to_endpoint_mm",
           "junction_mask", "endpoint_mask", "end_zone", "is_child_duplicate"]
NAMES = {"0": "主动脉", "1": "左髂总", "2": "右髂总"}
SAC_START, SAC_END, SAC_BULGE = 40.0, 80.0, 9.0     # radius rises from 8 mm to 17 mm in the middle of the root


def _radius(sid: int, s: float, base: float) -> float:
    if sid == 0 and SAC_START <= s <= SAC_END:
        return base + SAC_BULGE * math.sin(math.pi * (s - SAC_START) / (SAC_END - SAC_START))
    return base


def _atlas(lengths=(120.0, 30.0, 30.0), step=0.5, radii=(8.0, 5.0, 5.0)):
    """Root along +z from the origin; two children continue along +z from its end, offset in x."""
    rows, segments = [], []
    for sid, (length, base) in enumerate(zip(lengths, radii)):
        n = int(round(length / step)) + 1
        x0 = 0.0 if sid == 0 else (-14.0 if sid == 1 else 14.0)
        z0 = 0.0 if sid == 0 else lengths[0]
        offset = 0.0 if sid == 0 else lengths[0]
        for k in range(n):
            s = k * step
            rows.append([sid, k, x0, 0.0, z0 + s, 0.0, 0.0, 1.0, _radius(sid, s, base), s, offset + s,
                         0.0, 0.0, min(s, length - s), min(s, length - s), 0.0, float(k in (0, n - 1)), 0.0, 0.0])
        segments.append({"segment_id": sid, "parent_id": -1 if sid == 0 else 0, "starts_at_root": sid == 0,
                         "ends_at_leaf": sid != 0, "outlet_name": {1: "out-le", 2: "out-re"}.get(sid, ""),
                         "length_mm": length, "s_offset_mm": offset, "descendant_outlets": []})
    table = np.asarray(rows, dtype=np.float64)
    rows_n = len(table)
    return Atlas(table=table, columns=COLUMNS, segments=segments, semantic_of_segment={0: 0, 1: 1, 2: 2},
                 frame_n=np.tile([1.0, 0.0, 0.0], (rows_n, 1)), frame_b=np.tile([0.0, 1.0, 0.0], (rows_n, 1)),
                 tree_rows=np.arange(rows_n), tree=cKDTree(table[:, 2:5]), provenance={})


def _tube(sid: int, length: float, base: float, x0: float, z0: float, *, nring=48, step=1.0):
    """Open-ended tube of revolution around the (straight) branch axis."""
    levels = np.arange(0.0, length + 0.5 * step, step)
    angles = np.linspace(0.0, 2.0 * np.pi, nring, endpoint=False)
    vertices, faces = [], []
    for i, s in enumerate(levels):
        r = _radius(sid, float(s), base)
        for a in angles:
            vertices.append([x0 + r * np.cos(a), r * np.sin(a), z0 + s])
        if i:
            for j in range(nring):
                k = (j + 1) % nring
                low, high = (i - 1) * nring, i * nring
                faces.append([low + j, low + k, high + k])
                faces.append([low + j, high + k, high + j])
    return np.asarray(vertices, dtype=np.float64), np.asarray(faces, dtype=np.int64)


@pytest.fixture(scope="module")
def case():
    atlas = _atlas()
    vertices, faces = _tube(0, 120.0, 8.0, 0.0, 0.0)
    return atlas, vertices, faces


# ------------------------------------------------------------------ cross-section primitives
def test_section_of_a_straight_tube_matches_the_analytic_circle(case):
    _, vertices, faces = case
    mesh = MO.MeshSections(vertices, faces)
    section = mesh.section([0.0, 0.0, 20.0], [0.0, 0.0, 1.0], max_dist=32.0)
    assert section["found"] and section["closed"] and not section["open"]
    metrics = section["metrics"]
    # A 48-gon inscribed in r = 8: area = 0.5 n r² sin(2π/n) = 199.0 mm², equivalent diameter 15.92 mm.
    assert metrics["area_mm2"] == pytest.approx(0.5 * 48 * 64 * math.sin(2 * math.pi / 48), rel=1e-9)
    assert metrics["max_diameter_mm"] == pytest.approx(16.0, rel=1e-9)
    assert metrics["equivalent_diameter_mm"] == pytest.approx(2 * math.sqrt(metrics["area_mm2"] / math.pi), rel=1e-12)
    assert metrics["min_diameter_mm"] < metrics["max_diameter_mm"]
    world = np.asarray(section["polygon_world"])
    assert np.allclose(world[:, 2], 20.0, atol=1e-9) and len(world) >= 40


def test_section_off_axis_and_outside_the_search_radius(case):
    _, vertices, faces = case
    mesh = MO.MeshSections(vertices, faces)
    # The plane origin is outside the lumen but well inside maxDist: the nearest loop is still found.
    near = mesh.section([12.0, 0.0, 20.0], [0.0, 0.0, 1.0], max_dist=32.0)
    assert near["found"] and near["metrics"]["max_diameter_mm"] == pytest.approx(16.0, rel=1e-9)
    # Far outside maxDist: no loop qualifies.
    far = mesh.section([400.0, 0.0, 20.0], [0.0, 0.0, 1.0], max_dist=8.0)
    assert not far["found"] and far["metrics"] is None
    # A plane that misses the mesh entirely has no crossing segments at all.
    empty = mesh.section([0.0, 0.0, 400.0], [0.0, 0.0, 1.0], max_dist=32.0)
    assert not empty["found"] and empty["n_segments"] == 0


def test_open_chain_is_closed_with_one_synthetic_segment_when_the_gap_is_small(case):
    _, vertices, faces = case
    # Remove a wedge of the wall: the contour becomes an open chain the closer must bridge.
    keep = ~((vertices[faces][:, :, 0].mean(axis=1) > 5.0) & (np.abs(vertices[faces][:, :, 1].mean(axis=1)) < 3.0))
    mesh = MO.MeshSections(vertices, faces[keep])
    section = mesh.section([0.0, 0.0, 20.0], [0.0, 0.0, 1.0], max_dist=32.0)
    assert section["found"] and section["synthetic"] and section["closed"]
    assert section["metrics"]["area_mm2"] == pytest.approx(199.0, rel=0.05)


def test_section_metrics_of_a_square_are_exact():
    metrics = MO.section_metrics(np.array([[0.0, 0.0], [10.0, 0.0], [10.0, 10.0], [0.0, 10.0]]))
    assert metrics["area_mm2"] == pytest.approx(100.0) and metrics["perimeter_mm"] == pytest.approx(40.0)
    assert metrics["max_diameter_mm"] == pytest.approx(math.hypot(10, 10))
    assert metrics["min_diameter_mm"] == pytest.approx(10.0, abs=1e-9)
    assert metrics["equivalent_diameter_mm"] == pytest.approx(2 * math.sqrt(100.0 / math.pi))
    assert metrics["centroid"] == pytest.approx([5.0, 5.0])
    assert MO.section_metrics(np.zeros((2, 2))) is None


def test_lumen_volume_caps_the_open_ends():
    vertices, faces = _tube(1, 40.0, 5.0, 0.0, 0.0, nring=64, step=1.0)
    volume, note = MO.lumen_volume_ml(vertices, faces)
    assert note is None
    assert volume == pytest.approx(math.pi * 25.0 * 40.0 / 1000.0, rel=0.01)
    # A mesh whose boundary cannot be chained (a dangling triangle strip) reports no volume, with a reason.
    broken = np.concatenate([faces, np.array([[0, 1, len(vertices) - 1]])])
    value, message = MO.lumen_volume_ml(vertices, broken)
    assert value is None and "开口" in message


# ------------------------------------------------------------------ §17.1 block
def test_compute_finds_the_sac_the_neck_and_the_reference_diameter(case):
    atlas, vertices, faces = case
    out = MO.compute(vertices, faces, atlas, branch_names=NAMES,
                     cloud={"segment_id": np.zeros(4, int), "wss": np.array([0.1, 1.0, 5.0, 9.0])},
                     thresholds=[0.4, 4.0, 7.0])
    assert out["schema_version"] == "wss-deploy.morphology/v1" and out["station_mm"] == 1.0
    aorta = out["aorta"]
    assert aorta["name"] == "主动脉" and aorta["segment_id"] == 0
    # Reference = 10th percentile of the equivalent diameter: the straight 8 mm tube, not the bulge.
    assert aorta["reference_diameter_mm"] == pytest.approx(15.92, abs=0.1)
    largest = aorta["max"]
    assert largest["max_diameter_mm"] == pytest.approx(2 * (8.0 + SAC_BULGE), abs=0.1)
    assert largest["distance_from_inlet_mm"] == pytest.approx(60.0, abs=1.0)
    assert largest["xyz_mm"][2] == pytest.approx(60.0, abs=1.0) and largest["oblique"] is False
    assert 3 <= len(largest["polygon_world"]) <= MO.MAX_POLYGON_WORLD
    sac = aorta["sac"]
    assert sac["present"] and sac["threshold_mm"] == pytest.approx(1.5 * aorta["reference_diameter_mm"])
    # r >= 1.5 * 7.96 = 11.94 mm happens between s = 45.4 and s = 74.6 for the sine bulge.
    assert sac["s_start_mm"] == pytest.approx(46.0, abs=1.5) and sac["s_end_mm"] == pytest.approx(74.0, abs=1.5)
    assert sac["length_mm"] == pytest.approx(sac["s_end_mm"] - sac["s_start_mm"])
    assert sac["volume_ml"] > 0
    neck = aorta["neck"]
    assert neck["present"] and neck["s_start_mm"] == pytest.approx(2.0, abs=0.01)
    assert neck["diameter_mean_mm"] == pytest.approx(15.92, abs=0.2)
    assert neck["length_mm"] == neck["n_stations"] * out["station_mm"]
    assert out["lumen_volume_ml"] is not None and out["lumen_volume_method"] == "capped_mesh_divergence"


def test_stations_skip_the_openings_and_branches_carry_the_family_columns(case):
    atlas, vertices, faces = case
    segment = np.where(np.arange(600) < 300, 0, 1)
    wss = np.where(segment == 0, 0.2, 6.0)
    out = MO.compute(vertices, faces, atlas, branch_names=NAMES,
                     cloud={"segment_id": segment, "wss": wss}, thresholds=[0.4, 4.0, 7.0],
                     findings={"items": [{"kind": "pressure_drop", "branch": "主动脉", "value": 266.6}]})
    aorta = next(b for b in out["branches"] if b["name"] == "主动脉")
    stations = aorta["stations"]
    # The inlet (s = 0) and the distal 2 mm are skipped; stations sit on whole millimetres.
    assert stations["s_local_mm"][0] == 2.0 and stations["s_local_mm"][-1] == 118.0
    assert aorta["n_stations"] == len(stations["s_local_mm"]) == 117
    assert aorta["tortuosity"] == pytest.approx(1.0, abs=1e-6)
    assert aorta["wss_mean_pa"] == pytest.approx(0.2) and aorta["area_frac_low"] == pytest.approx(1.0)
    assert aorta["area_frac_high"] == pytest.approx(0.0) and aorta["delta_p_pa"] == pytest.approx(266.6)
    left = next(b for b in out["branches"] if b["name"] == "左髂总")
    assert left["wss_mean_pa"] == pytest.approx(6.0) and left["area_frac_high"] == pytest.approx(1.0)
    assert [b["name"] for b in out["branches"]] == ["主动脉", "左髂总", "右髂总"]


def test_no_sac_when_the_tube_is_uniform():
    atlas = _atlas(lengths=(60.0, 30.0, 30.0), radii=(8.0, 5.0, 5.0))
    # Straight 8 mm tube only: no station reaches 1.5 x the reference.
    atlas.table[:, COLUMNS.index("radius_mm")] = 8.0
    vertices, faces = _tube(1, 60.0, 8.0, 0.0, 0.0)
    out = MO.compute(vertices, faces, atlas, branch_names=NAMES)
    assert out["aorta"]["max"]["max_diameter_mm"] == pytest.approx(16.0, rel=1e-6)
    assert out["aorta"]["sac"] == {"present": False} and out["aorta"]["neck"] == {"present": False}
    assert any("无瘤样扩张" in note for note in out["notes"])


def test_oblique_stations_are_flagged_against_the_inscribed_diameter(case):
    atlas, vertices, faces = case
    table = {"min_diameter_mm": [16.0, 60.0, None], "inscribed_diameter_mm": [16.0, 16.0, 16.0]}
    assert MO._oblique_mask(table).tolist() == [False, True, False]
    out = MO.compute(vertices, faces, atlas, branch_names=NAMES)
    # The synthetic tube is a body of revolution around the centreline: nothing is oblique.
    assert all(branch["n_oblique"] == 0 for branch in out["branches"])
    assert out["aorta"]["max"]["obliquity_ratio"] < MO.OBLIQUITY_FACTOR


def test_trim_for_report_drops_the_bulky_station_series(case):
    atlas, vertices, faces = case
    full = MO.compute(vertices, faces, atlas, branch_names=NAMES)
    trimmed = MO.trim_for_report(full)
    assert set(trimmed["aorta"]["stations"]) == set(MO.REPORT_STATION_KEYS)
    assert all(set(branch["stations"]) == set(MO.REPORT_STATION_KEYS) for branch in trimmed["branches"])
    assert trimmed["aorta"]["max"] == full["aorta"]["max"]              # the ring the report draws survives
    assert trimmed["lumen_volume_ml"] == full["lumen_volume_ml"]
    assert len(trimmed["aorta"]["stations"]["s_from_root_mm"]) == len(full["aorta"]["stations"]["s_from_root_mm"])
    # The full block keeps what summary.json must carry.
    assert {"xyz_mm", "closed", "area_mm2"} <= set(full["aorta"]["stations"])
    assert MO.trim_for_report(None) is None


def test_digest_is_station_free_and_exposes_the_headline_diameter(case):
    atlas, vertices, faces = case
    full = MO.compute(vertices, faces, atlas, branch_names=NAMES)
    small = MO.digest(full)
    assert "stations" not in small["aorta"] and all("stations" not in b for b in small["branches"])
    assert "polygon_world" not in small["aorta"]["max"] and "method" not in small
    assert small["aorta"]["sac"] == full["aorta"]["sac"] and small["lumen_volume_ml"] == full["lumen_volume_ml"]
    assert len(json.dumps(small, ensure_ascii=False)) < len(json.dumps(full, ensure_ascii=False)) / 10
    assert MO.max_diameter_mm(small) == MO.max_diameter_mm(full) == full["aorta"]["max"]["max_diameter_mm"]
    assert MO.digest(None) is None and MO.max_diameter_mm(None) is None and MO.max_diameter_mm({"aorta": None}) is None


def test_compute_rejects_a_non_positive_station_spacing(case):
    atlas, vertices, faces = case
    with pytest.raises(ValueError):
        MO.compute(vertices, faces, atlas, branch_names=NAMES, station_mm=0.0)
    with pytest.raises(ValueError):
        MO.MeshSections(np.zeros((0, 3)), np.zeros((0, 3), dtype=np.int64))


# ------------------------------------------------------------------ §17.1 reliability (2026-09-22)
def _swept_prism(levels, centre, profile, nring: int = 72):
    """Closed tube swept along ``centre(z)`` whose in-plane polar radius is ``profile(z, theta)``."""
    angles = np.linspace(0.0, 2.0 * np.pi, nring, endpoint=False)
    vertices, faces = [], []
    for i, z in enumerate(levels):
        origin = centre(float(z))
        radii = profile(float(z), angles)
        for angle, r in zip(angles, radii):
            vertices.append(origin + r * np.array([np.cos(angle), np.sin(angle), 0.0]))
        if i:
            for j in range(nring):
                k = (j + 1) % nring
                low, high = (i - 1) * nring, i * nring
                faces.append([low + j, low + k, high + k])
                faces.append([low + j, high + k, high + j])
    return np.asarray(vertices, dtype=np.float64), np.asarray(faces, dtype=np.int64)


def _single_branch_atlas(points, radii):
    """One-segment atlas following ``points`` with the given (possibly lagging) inscribed radii."""
    points = np.asarray(points, dtype=np.float64)
    n = len(points)
    s = np.concatenate([[0.0], np.cumsum(np.linalg.norm(np.diff(points, axis=0), axis=1))])
    rows = [[0, k, *points[k], 0.0, 0.0, 1.0, radii[k], s[k], s[k], 0.0, 0.0,
             min(s[k], s[-1] - s[k]), min(s[k], s[-1] - s[k]), 0.0, float(k in (0, n - 1)), 0.0, 0.0]
            for k in range(n)]
    table = np.asarray(rows, dtype=np.float64)
    segments = [{"segment_id": 0, "parent_id": -1, "starts_at_root": True, "ends_at_leaf": True,
                 "outlet_name": "out-le", "length_mm": float(s[-1]), "s_offset_mm": 0.0, "descendant_outlets": []}]
    return Atlas(table=table, columns=COLUMNS, segments=segments, semantic_of_segment={0: 0},
                 frame_n=np.tile([1.0, 0.0, 0.0], (n, 1)), frame_b=np.tile([0.0, 1.0, 0.0], (n, 1)),
                 tree_rows=np.arange(n), tree=cKDTree(table[:, 2:5]), provenance={})


TUBE_R, SAC_R, SAC_B, SAC_Z, SHOULDER_Z, SAC_END_Z, SLANT = 8.0, 35.0, 75.0, 110.0, 50.0, 170.0, 45.0


def _sac_radius(z: float) -> float:
    """8 mm neck, then an ellipsoidal sac (radial 35 mm, axial 75 mm) starting abruptly at the shoulder."""
    if z < SHOULDER_Z or z > SAC_END_Z:
        return TUBE_R
    return max(TUBE_R, SAC_R * math.sqrt(max(0.0, 1.0 - ((z - SAC_Z) / SAC_B) ** 2)))


def _sac_centre(z: float) -> np.ndarray:
    """The neck enters the sac at ~42°, so a plane normal to the neck grazes far into the sac."""
    return np.array([SLANT * max(0.0, (SHOULDER_Z - z) / SHOULDER_Z), 0.0, z])


@pytest.fixture(scope="module")
def slanted_sac():
    levels = np.sort(np.concatenate([np.arange(0.0, 220.5, 1.0),
                                     [SHOULDER_Z - 0.01, SHOULDER_Z + 0.01, SAC_END_Z - 0.01, SAC_END_Z + 0.01]]))
    vertices, faces = _swept_prism(levels, _sac_centre, lambda z, ang: np.full(len(ang), _sac_radius(z)))
    z = np.arange(0.0, 220.5, 0.5)
    points = np.stack([_sac_centre(float(v)) for v in z])
    radii = np.array([_sac_radius(float(v)) for v in z])
    return _single_branch_atlas(points, radii), vertices, faces


def test_oblique_shoulder_station_is_reoriented_and_the_sac_diameter_is_the_true_one(slanted_sac):
    """The plane normal to the slanted neck grazes into the sac; re-orientation recovers the neck calibre."""
    atlas, vertices, faces = slanted_sac
    out = MO.compute(vertices, faces, atlas, branch_names={"0": "主动脉"})
    aorta = out["aorta"]
    stations = aorta["stations"]
    # The widest true section is the ellipsoid's equator: 2 × 35 mm.
    assert aorta["max"]["max_diameter_mm"] == pytest.approx(2 * SAC_R, rel=0.05)
    assert aorta["max"]["oblique"] is False and aorta["sac"]["present"]
    assert aorta["sac"]["max_diameter_mm"] == pytest.approx(2 * SAC_R, rel=0.05)
    shoulder = [i for i, flag in enumerate(stations["reoriented"]) if flag]
    assert shoulder, "the shoulder stations must be re-oriented"
    for i in shoulder:
        # Every one of them measured a grazing swath and now reports the local neck calibre.
        assert stations["raw_max_diameter_mm"][i] > 2.5 * TUBE_R
        assert stations["max_diameter_mm"][i] < 0.6 * stations["raw_max_diameter_mm"][i]
        assert stations["max_diameter_mm"][i] == pytest.approx(2 * TUBE_R, rel=0.1)
        assert stations["tilt_deg"][i] in MO.REORIENT_TILTS_DEG
        assert stations["reliable"][i] is True      # re-orientation cleared the flag
    assert aorta["n_reoriented"] == len(shoulder) and aorta["n_reliable"] == aorta["n_reoriented"] + \
        sum(1 for i, flag in enumerate(stations["reliable"]) if flag and not stations["reoriented"][i])
    assert any("重新定向" in note for note in out["notes"])


SPIKE_R, SPIKE_AMP, SPIKE_START, SPIKE_END = 5.0, 26.0, 30.5, 45.5


def _keyhole(z: float, angles: np.ndarray) -> np.ndarray:
    """A 5 mm tube; between two z levels a narrow lobe reaches 31 mm towards +x (the junction keyhole)."""
    amplitude = SPIKE_AMP if SPIKE_START <= z <= SPIKE_END else 0.0
    offset = np.abs((angles + np.pi) % (2.0 * np.pi) - np.pi)
    return SPIKE_R + amplitude * np.exp(-(offset ** 2) / (2 * 0.16 ** 2))


def test_keyhole_stations_do_not_set_the_branch_diameter():
    """A plane that also slices a neighbouring lumen makes a sliver/keyhole; the branch keeps the tube calibre."""
    levels = np.sort(np.concatenate([np.arange(0.0, 75.5, 1.0),
                                     [SPIKE_START - 0.01, SPIKE_START + 0.01, SPIKE_END - 0.01, SPIKE_END + 0.01]]))
    vertices, faces = _swept_prism(levels, lambda z: np.array([0.0, 0.0, z]), _keyhole, nring=96)
    z = np.arange(0.0, 75.5, 0.5)
    atlas = _single_branch_atlas(np.stack([np.zeros_like(z), np.zeros_like(z), z], axis=1),
                                 np.full(len(z), SPIKE_R))
    out = MO.compute(vertices, faces, atlas, branch_names={"0": "主动脉"})
    branch = out["branches"][0]
    stations = branch["stations"]
    keyhole_raw = max(v for v in stations["raw_max_diameter_mm"] if v is not None)
    assert keyhole_raw == pytest.approx(2 * SPIKE_R + SPIKE_AMP, rel=0.05)   # ~36 mm from lobe tip to far wall
    # The reported calibre is the tube, not the keyhole.
    assert branch["diameter_max_mm"] < 0.6 * keyhole_raw
    assert branch["diameter_min_mm"] == pytest.approx(2 * SPIKE_R, rel=0.05)
    flagged = [i for i, flag in enumerate(stations["reoriented"]) if flag]
    assert len(flagged) >= 10 and all(stations["raw_max_diameter_mm"][i] > 3 * SPIKE_R for i in flagged)
    assert branch["n_reliable"] + branch["n_excluded"] == branch["n_closed"]
    # The sliver rule fires on elongation and on solidity, not on obliquity alone.
    spike = MO.section_metrics(np.asarray([[SPIKE_R * math.cos(a), SPIKE_R * math.sin(a)]
                                           for a in np.linspace(0, 2 * np.pi, 64, endpoint=False)]
                                          + [[31.0, 0.3], [31.0, -0.3]]))
    flags = MO.station_flags(spike, 2 * SPIKE_R)
    assert flags["elongated"] and flags["low_solidity"] and flags["suspect"]
    assert MO.station_flags(None, 10.0) == {"oblique": False, "elongated": False,
                                            "low_solidity": False, "suspect": False}

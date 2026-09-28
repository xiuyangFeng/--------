"""cfd_auto invariants on a library case (read-only): the STL wall passes through untouched, caps reproduce the library
cut faces, extension regions are closed and collision-free, an identity UDF render is byte-identical, and the guard
refuses library paths."""
import json
from pathlib import Path

import numpy as np
import pytest

from cfd_auto import guard, surface, udf

ROOT = Path("/public/newhome/cy/Digital_twin/GNN")
CASE = ROOT / "data_new/AG/slow/QIN_SI_FU"
PROFILE = ROOT / "outputs/cfd_auto_trial_20260927/AG/slow/QIN_SI_FU/reference_profile.json"

pytestmark = pytest.mark.skipif(not (CASE / "QIN_SI_FU.stl").exists() or not PROFILE.exists(), reason="library case / trial profile not available")


@pytest.fixture(scope="module")
def built():
    prof = json.loads(PROFILE.read_text())
    pts, wall = surface.read_stl(CASE / "QIN_SI_FU.stl")
    loops = surface.boundary_loops(wall)
    names = surface.name_caps_by_reference(np.array([pts[l].mean(0) for l in loops]), prof["openings"])
    surf, stats = surface.close_surface(CASE / "QIN_SI_FU.stl", names)
    exts = [surface.Extension(e["opening"], e["length_mm"], e["wall_zone"], e["interface_zone"][0]) for e in prof["extensions"]]
    P, zones, _ = surface.extend_surface(surf, exts)
    return prof, pts, wall, surf, stats, exts, P, zones


def test_wall_is_the_stl(built):
    _, pts, wall, surf, *_ = built
    assert np.array_equal(surf.points_mm[: surf.n_wall_nodes], pts)
    assert {tuple(f) for f in np.sort(surf.wall_faces, 1)} == {tuple(f) for f in np.sort(wall, 1)}


def test_caps_match_library_cut_faces(built):
    prof, _, _, surf, stats, *_ = built
    ref = {o["name"]: o for o in prof["openings"]}
    assert sorted(c.name for c in surf.caps) == sorted(ref)
    for c in surf.caps:
        assert c.area_mm2 == pytest.approx(ref[c.name]["area_mm2"], rel=1e-4)
        assert float(np.dot(c.normal, ref[c.name]["outward_normal"])) > 0.9999
    assert min(s["min_angle_deg_min"] for s in stats) > 20


def test_regions_closed_and_collision_free(built):
    _, _, _, _, _, exts, P, zones = built
    chk = surface.check_regions(P, zones, exts)
    assert chk["ok"], chk
    assert chk["anatomy"]["volume_mm3"] == pytest.approx(167032.1, rel=1e-5)   # library anatomy zone volume


def test_udf_identity_render_is_byte_identical(built):
    prof = built[0]
    text = udf.read_udf(CASE / "udf-inlet.c")
    c = prof["udf_constants"]
    assert udf.render(text, c["threads"], c["inlet_area_m2"], c["rcr"]) == text
    moved = udf.render(text, {"outle": 1, "outli": 2, "outri": 3, "outre": 4})
    assert moved.count("Lookup_Thread(d, ") == 4 and "Lookup_Thread(d, 4396)" not in moved


def test_guard_refuses_library(tmp_path):
    with pytest.raises(PermissionError):
        guard.assert_inside(CASE / "x.cas.gz", CASE)
    with pytest.raises(PermissionError):
        guard.check_journal(f"/file/read-case {CASE}/QIN_SI_FU.cas.gz\n", tmp_path)
    guard.check_journal(f"/file/read-case {tmp_path}/a.cas.gz\n", tmp_path)

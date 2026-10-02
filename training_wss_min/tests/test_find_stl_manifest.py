"""find_stl reads the canonical STL manifest (2026-10-02); units outside it keep the folder rule."""
import json

import pytest

from training_wss_min.tools import deployment_stl_simulation as S

MANIFEST = S.ROOT / "data_wss_v5/stl_canonical_v5_2d_20261001/manifest.json"
pytestmark = pytest.mark.skipif(not MANIFEST.is_file(), reason="canonical STL manifest (patient data) not present")


def rows():
    return json.loads(MANIFEST.read_text())["units"]


def test_every_manifest_unit_resolves_to_its_canonical_file():
    for row in rows():
        stl = S.find_stl(row["unit"])
        assert stl == S.ROOT / row["canonical"]["path"] or stl.resolve() == (S.ROOT / row["canonical"]["path"]).resolve()
        assert stl.is_file(), row["unit"]


def test_exact_units_with_a_single_stl_match_the_old_folder_rule():
    checked = 0
    for row in rows():
        if row["category"] == "exact" and len(row["stl_files"]) == 1:
            assert S.find_stl(row["unit"]) == S.find_stl_in_folder(row["unit"])
            checked += 1
    assert checked > 200


def test_non_exact_units_use_the_cfd_wall_export_not_the_folder_file():
    picked = [row for row in rows() if row["category"] != "exact"]
    assert picked
    for row in picked:
        stl = S.find_stl(row["unit"])
        assert row["canonical"]["kind"] != "data_new_stl" and "cfd_wall" in stl.parts
        assert stl != S.find_stl_in_folder(row["unit"])


def test_units_outside_the_manifest_keep_the_folder_rule():
    assert S.find_stl("AG/slow/NOT_A_LIBRARY_UNIT") is None
    assert S.find_stl("AG/slow/SUN_ZONG_GE~m43") == S.find_stl_in_folder("AG/slow/SUN_ZONG_GE~m43")

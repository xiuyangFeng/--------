from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from pipeline_wss_min.new_cohorts.ilo_before import (
    DEFAULT_WHITELIST,
    build_inventory,
    discover_after_derived,
    is_ilo_before_unit,
    normalize_patient_name,
    source_stat_fingerprint,
    validate_active_preprocess_unit,
    wss_hard_failures,
)
from pipeline_wss_min.raw_io import read_inlet_waveform


class ILOBeforeIdentityTests(unittest.TestCase):
    def test_name_normalization_strips_only_ilo_suffix(self):
        self.assertEqual(normalize_patient_name(" zhang-mao jin-0 "), "ZHANG_MAO_JIN")
        self.assertEqual(normalize_patient_name("SHEN_CHUN_WANG-1"), "SHEN_CHUN_WANG")
        self.assertEqual(normalize_patient_name("LIU_JIE"), "LIU_JIE")

    def test_after_is_rejected_from_active_preprocess(self):
        validate_active_preprocess_unit("ILO/WANG_LI_MIN-0/before")
        with self.assertRaisesRegex(ValueError, "只允许 before"):
            validate_active_preprocess_unit("ILO/WANG_LI_MIN-0/after")
        self.assertTrue(is_ilo_before_unit("ILO/WANG_LI_MIN-0/before"))
        self.assertFalse(is_ilo_before_unit("ILO/WANG_LI_MIN-0/after"))

    def test_current_inventory_contract(self):
        with tempfile.TemporaryDirectory() as tmp:
            report_dir = Path(tmp)
            manifest = build_inventory(report_dir)
            overlaps = json.loads(
                (report_dir / "ilo_before_name_overlaps.json").read_text(encoding="utf-8")
            )
        self.assertEqual(manifest["counts"]["before_units"], 61)
        self.assertEqual(manifest["counts"]["included_before"], 41)
        self.assertEqual(manifest["counts"]["excluded_before"], 20)
        self.assertEqual(manifest["counts"]["manual_visual_excluded"], 8)
        self.assertEqual(manifest["counts"]["after_units_in_active_manifest"], 0)
        self.assertEqual(
            {unit.split("/")[1] for unit in manifest["excluded_units"] if "SHEN_CHUN_WANG" in unit or "ZHANG_MAO_JIN" in unit},
            {"SHEN_CHUN_WANG-0", "ZHANG_MAO_JIN-0"},
        )
        self.assertEqual(
            {row["normalized_patient_id"] for row in overlaps["exact_overlaps"]},
            {"GUO_AI_JUN", "SHEN_CHUN_WANG", "ZHANG_MAO_JIN"},
        )
        self.assertTrue(all(row["decision"] == "manual_watch_only" for row in overlaps["near_matches"]))
        self.assertIn("ILO/ZHANG_JIAN_JUN-1/before", manifest["included_units"])
        self.assertIn("ILO/ZHANG_JIN_CHUN-1/before", manifest["included_units"])
        for patient in (
            "DONG_KE_QIN-0", "WEI_QING_FENG-1", "ZHANG_WAN_ZENG-1",
            "LIU_CHUN_YANG-1", "LIU_YUN_ZHANG-0", "LI_YU_GANG-0",
            "WANG_LI_MIN-0", "YANG_QING_REN-1",
        ):
            self.assertNotIn(f"ILO/{patient}/before", manifest["included_units"])
        self.assertTrue(all(is_ilo_before_unit(unit) for unit in manifest["included_units"]))


class ILOBeforeIncrementalTests(unittest.TestCase):
    def test_wss_hard_gates(self):
        steps = np.array([10, 12])
        self.assertEqual(wss_hard_failures(np.ones((2, 100)), steps), [])
        negative = np.ones((2, 100))
        negative[0, 0] = -1
        self.assertIn("negative_wss", wss_hard_failures(negative, steps))
        too_many_zero = np.ones((2, 100))
        too_many_zero[:, :2] = 0
        self.assertIn("nonpositive_wss_over_1pct", wss_hard_failures(too_many_zero, steps))
        zero_step = np.vstack([np.zeros(100), np.ones(100)])
        failures = wss_hard_failures(zero_step, steps)
        self.assertIn("all_zero_timesteps:10", failures)
        nonfinite = np.ones((2, 100)); nonfinite[1, 4] = np.nan
        self.assertIn("wss_nan_or_inf", wss_hard_failures(nonfinite, steps))

    def test_inlet_waveform_prefers_complete_fluent_continuation(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "Global_conditions"
            root.mkdir(parents=True)
            (root / "vf-in-rfile.out").write_text("0 1.0 0\n", encoding="utf-8")
            (root / "vf-in-rfile_1_1.out").write_text(
                "0 1.0 0\n1 3.0 0.1\n2 2.0 0.2\n", encoding="utf-8"
            )
            waveform = read_inlet_waveform(Path(tmp))
        self.assertEqual(waveform, {0: 1.0, 1: 3.0, 2: 2.0})

    def test_stat_fingerprint_changes_with_source_metadata(self):
        with tempfile.TemporaryDirectory() as tmp:
            case = Path(tmp)
            wall = case / "ascii"
            wall.mkdir()
            file_path = wall / "CASE-1120"
            file_path.write_text("a", encoding="utf-8")
            first = source_stat_fingerprint(case)[0]
            file_path.write_text("aa", encoding="utf-8")
            second = source_stat_fingerprint(case)[0]
        self.assertNotEqual(first, second)

    def test_after_discovery_is_narrow(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            after = root / "ILO" / "CASE-0" / "after"
            after.mkdir(parents=True)
            (after / "bundle.npz").write_bytes(b"bundle")
            (after / "report.json").write_text("{}", encoding="utf-8")
            (after / "keep.txt").write_text("keep", encoding="utf-8")
            targets, dirs = discover_after_derived(root)
        self.assertEqual({path.name for path in targets}, {"bundle.npz", "report.json"})
        self.assertEqual(len(dirs), 1)

    def test_default_whitelist_is_before_only_when_present(self):
        if not DEFAULT_WHITELIST.is_file():
            self.skipTest("run ILO inventory first")
        payload = json.loads(DEFAULT_WHITELIST.read_text(encoding="utf-8"))
        self.assertEqual(payload["counts"], {"ILO_before": 41, "ILO_after": 0})
        self.assertTrue(all(is_ilo_before_unit(unit) for unit in payload["included_units"]))


if __name__ == "__main__":
    unittest.main()

from __future__ import annotations

import unittest

import numpy as np

from apply_v4_high_tail_calibrator import validate_package
from calibrate_v4_high_tail_oof import smooth_gate, tail_ratio


class HighTailCalibratorTest(unittest.TestCase):
    def test_smooth_gate_is_zero_below_start_and_one_at_top(self) -> None:
        rank = np.asarray([0.0, 0.79, 0.80, 0.90, 1.0])
        gate = smooth_gate(rank, 0.80)
        np.testing.assert_allclose(gate[:3], 0.0)
        self.assertGreater(gate[3], 0.0)
        self.assertLess(gate[3], 1.0)
        self.assertEqual(gate[4], 1.0)

    def test_tail_ratio_preserves_bulk_and_peak_strength_is_monotone(self) -> None:
        current = np.asarray([1.0, 1.0, 1.0, 1.0])
        learned = np.asarray([1.0, 1.1, 1.2, 1.3])
        base = np.asarray([1.0, 2.0, 3.0, 4.0])
        rank = np.asarray([0.10, 0.70, 0.90, 1.00])
        no_peak = tail_ratio(
            current,
            learned,
            base,
            rank,
            tail_alpha=1.2,
            peak_alpha=1.6,
            gate_start=0.80,
            use_peak_anchor=0.0,
        )
        peak30 = tail_ratio(
            current,
            learned,
            base,
            rank,
            tail_alpha=1.2,
            peak_alpha=1.6,
            gate_start=0.80,
            use_peak_anchor=0.30,
        )
        np.testing.assert_allclose(no_peak[:2], current[:2])
        np.testing.assert_allclose(peak30[:2], current[:2])
        self.assertGreater(peak30[-1], no_peak[-1])

    def test_package_schema_is_strict(self) -> None:
        validate_package(
            {"schema_version": 2, "feature_names": ["a", "b"]},
            ["a", "b"],
        )
        with self.assertRaisesRegex(RuntimeError, "schema mismatch"):
            validate_package(
                {"schema_version": 1, "feature_names": ["a", "b"]},
                ["a", "b"],
            )
        with self.assertRaisesRegex(RuntimeError, "feature schema mismatch"):
            validate_package(
                {"schema_version": 2, "feature_names": ["b", "a"]},
                ["a", "b"],
            )


if __name__ == "__main__":
    unittest.main()

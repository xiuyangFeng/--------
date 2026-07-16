from __future__ import annotations

import unittest
from argparse import Namespace
import json
from pathlib import Path

import numpy as np

from pipeline_wss_min import build_samples, config as C
from pipeline_wss_min.archive.alignment_v3 import legacy_registration_for_case
from pipeline_wss_min.examples.preprocess_pipeline import (
    farthest_point_sample,
    require_included_case,
)
from pipeline_wss_min.new_cohorts.common import split_unit
from pipeline_wss_min.run import _config_from_args


REPO = Path(__file__).resolve().parents[2]
SOURCE = REPO / "pipeline_wss_min/examples/preprocess_pipeline.py"


class TestCompactPreprocessExample(unittest.TestCase):
    def test_example_stays_within_250_lines(self):
        text = SOURCE.read_text()
        self.assertLessEqual(len(text.splitlines()), 250)
        self.assertNotIn("from pipeline_wss_min", text)
        self.assertNotIn("import pipeline_wss_min", text)

    def test_fps_matches_production_implementation(self):
        points = np.random.default_rng(7).normal(size=(80, 3))
        expected = build_samples.farthest_point_sample(points, 20, seed=1234)
        actual = farthest_point_sample(points, 20, seed=1234)
        np.testing.assert_array_equal(actual, expected)

    def test_only_split_included_cases_are_allowed(self):
        split = C.load_split(C.DEFAULT_SPLIT_NAME)
        included = C.split_case_labels(C.DEFAULT_SPLIT_NAME, ("train", "val", "test"))
        subset, case = included[0].split("/", 1)
        self.assertEqual(
            require_included_case(f"AG/{subset}", case, C.split_path(C.DEFAULT_SPLIT_NAME)),
            included[0],
        )
        excluded_subset, excluded_case = split["excluded_cases"][0].split("/", 1)
        with self.assertRaises(ValueError):
            require_included_case(
                f"AG/{excluded_subset}", excluded_case, C.split_path(C.DEFAULT_SPLIT_NAME)
            )

    def test_frozen_stats_are_train_peak_logz(self):
        stats = json.loads((REPO / "data_wss_min/wss_global_stats.json").read_text())
        self.assertEqual(stats["partitions"], ["train"])
        self.assertEqual(stats["timesteps_scope"], "peak")
        self.assertEqual(stats["method"], "log_z")

    def test_cli_overrides_do_not_mutate_default_config(self):
        original = C.DEFAULT.sample.wall_n_points
        args = Namespace(wall_n=1500, timesteps="all", sample_name="demo")
        cfg = _config_from_args(args)
        self.assertEqual(cfg.sample.wall_n_points, 1500)
        self.assertEqual(C.DEFAULT.sample.wall_n_points, original)

    def test_new_cohort_unit_paths_are_centralized(self):
        self.assertEqual(split_unit("AAA/ruputer/CASE"), ("AAA/ruputer", "CASE"))
        self.assertEqual(split_unit("ILO/PATIENT-0/after"), ("ILO", "PATIENT-0/after"))

    def test_archived_alignment_is_frozen_to_legacy_frame(self):
        cfg = legacy_registration_for_case("AG/fast", "CHEN_SHI_MING")
        self.assertEqual(cfg.frame_mode, "legacy_centerline")


if __name__ == "__main__":
    unittest.main()

"""Scientific acceptance and reporting regressions for the single-seed volume matrix."""
from __future__ import annotations

import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock

import numpy as np

from training_wss_min.tools import report_volume_attention as R


def baseline(**changes):
    row = dict(r2_cb=.70, mae=10., vector_rmse=.20, radial_r2=.23, circ_r2=.10,
               lw20=1., lw40=2.)
    row.update(changes)
    return row


def improved(**changes):
    row = baseline(r2_cb=.72, mae=9.5, vector_rmse=.19, lw20=.8, lw40=1.6)
    row.update(changes)
    return row


class ScientificGateTest(unittest.TestCase):
    def test_velocity_rejects_secondary_flow_regression_at_either_checkpoint(self):
        for checkpoint in ("best", "last"):
            for component in ("radial_r2", "circ_r2"):
                with self.subTest(checkpoint=checkpoint, component=component):
                    best, last = improved(), improved()
                    (best if checkpoint == "best" else last)[component] = baseline()[component] - .02
                    gate = R.evaluate_gate(best, last, baseline(), baseline(), "velocity")
                    self.assertFalse(gate["qualified"])
                    self.assertFalse(gate["longwave_improved"])
                    self.assertTrue(any(component in issue for issue in gate["issues"]))

    def test_speed_r2_gain_cannot_replace_vector_error_improvement(self):
        gate = R.evaluate_gate(improved(vector_rmse=.21), improved(), baseline(), baseline(), "velocity")
        self.assertFalse(gate["qualified"])
        self.assertTrue(any("vector_rmse" in issue for issue in gate["issues"]))

    def test_longwave_requires_both_scales_and_precision_gate(self):
        gate = R.evaluate_gate(improved(lw40=1.9), improved(), baseline(), baseline(), "pressure")
        self.assertTrue(gate["qualified"])
        self.assertFalse(gate["longwave_improved"])
        gate = R.evaluate_gate(improved(r2_cb=.705), improved(), baseline(), baseline(), "pressure")
        self.assertFalse(gate["qualified"])
        self.assertFalse(gate["longwave_improved"])
        gate = R.evaluate_gate(improved(), improved(), baseline(), baseline(), "pressure")
        self.assertTrue(gate["qualified"])
        self.assertTrue(gate["longwave_improved"])

    def test_exact_preregistered_thresholds_are_inclusive(self):
        reference = baseline(r2_cb=.40)
        candidate = improved(r2_cb=.41, mae=9.7, vector_rmse=.194, lw20=.9, lw40=1.8)
        gate = R.evaluate_gate(candidate, improved(r2_cb=.42), reference, reference, "pressure")
        self.assertTrue(gate["qualified"], gate["issues"])
        self.assertTrue(gate["longwave_improved"])

    def test_missing_diagnostics_cannot_pass_velocity_guard_or_longwave(self):
        gate = R.evaluate_gate(improved(radial_r2=None), improved(), baseline(), baseline(), "velocity")
        self.assertFalse(gate["qualified"])
        gate = R.evaluate_gate(improved(lw20=None), improved(), baseline(), baseline(), "pressure")
        self.assertTrue(gate["qualified"])
        self.assertFalse(gate["longwave_improved"])


def small_matrix():
    arms = []
    for prefix, task in (("P", "pressure"), ("V", "velocity")):
        for number, parent in (("00", "R5" + prefix), ("01", prefix + "00"), ("03", prefix + "01")):
            arms.append(dict(id=prefix + number, task=task, parent=parent, run_name=prefix + number))
    arms += [dict(id="J00", task="joint", parent="P01+V01", run_name="J00", comparisons=["P01", "V01"]),
             dict(id="J01", task="joint", parent="J00", run_name="J01", comparisons=["J00", "P03", "V03"])]
    return {"matrix_id": "synthetic", "arms": arms, "factorials": []}


def build_synthetic_report(*, joint_best, joint_last, queue_status="complete"):
    matrix = small_matrix()
    state = {"status": queue_status, "source_changed": False,
             "arms": {arm["id"]: {"status": "complete" if queue_status == "complete" else "pending"}
                      for arm in matrix["arms"]}}
    def row(aid, task, checkpoint, run, parent, status, record):
        values = baseline()
        if aid == "J00":
            values = copy.deepcopy(joint_best if checkpoint == "best" else joint_last)
        elif aid == "J01":
            values = improved(r2_cb=.74, mae=9.0, vector_rmse=.18)
        values.update(id=aid, task=task, checkpoint=checkpoint, run_dir=str(run), parent=parent,
                      status=status, delta_r2_parent=None, gate="待完成", metrics_path="unused",
                      rmse=values["mae"] * 1.2, case_median=.7, case_p10=.3, negative_cases=0,
                      axial_r2=.7, parameters=100, eval_seconds=1., train_seconds=1.,
                      groups={}, pressure_query_groups={})
        return values, {}
    with mock.patch.object(R, "manifest", return_value=matrix), \
         mock.patch.object(R, "read_json", return_value=state), \
         mock.patch.object(R, "metrics_row", side_effect=row):
        return R.build_report(validate=False)


class ReportProtocolTest(unittest.TestCase):
    def test_joint_single_primary_comparison_is_last(self):
        report = build_synthetic_report(joint_best=improved(), joint_last=improved())
        comparisons = [c for c in report["comparisons"] if c["id"] == "J00" and c["reference"] == "P01"]
        self.assertEqual({c["checkpoint"]: c["primary_for_joint_vs_single"] for c in comparisons},
                         {"best": False, "last": True})
        joint_pair = [c for c in report["comparisons"] if c["id"] == "J01" and c["reference"] == "J00"]
        self.assertEqual({c["checkpoint"]: c["primary_for_joint_vs_single"] for c in joint_pair},
                         {"best": True, "last": False})

    def test_joint_single_gate_uses_last_threshold_even_when_best_improves_more(self):
        report = build_synthetic_report(joint_best=improved(),
                                        joint_last=baseline(r2_cb=.705, mae=9.9, vector_rmse=.199))
        self.assertFalse(report["gates"]["J00/pressure"]["qualified"])
        self.assertFalse(report["gates"]["J00/velocity"]["qualified"])

    def test_joint_single_last_gain_is_not_vetoed_by_supplementary_best(self):
        report = build_synthetic_report(joint_best=baseline(r2_cb=.69, mae=10.1, vector_rmse=.21),
                                        joint_last=improved())
        self.assertTrue(report["gates"]["J00/pressure"]["qualified"], report["gates"]["J00/pressure"])
        self.assertTrue(report["gates"]["J00/velocity"]["qualified"], report["gates"]["J00/velocity"])

    def test_running_queue_cannot_be_reported_as_complete(self):
        report = build_synthetic_report(joint_best=improved(), joint_last=improved(), queue_status="running")
        self.assertFalse(report["complete"])
        self.assertEqual(report["queue_status"], "running")
        new_rows = [row for row in report["rows"] if not row["id"].startswith("R5")]
        self.assertTrue(all(row["status"] != "complete" for row in new_rows))
        self.assertTrue(all(row["gate"] != "通过开发筛选" for row in new_rows))
        # Exercise Markdown/JSON emission as well as the in-memory completion flag.
        with tempfile.TemporaryDirectory() as temp, mock.patch.object(R, "EXP", Path(temp)):
            R.write_outputs(report)
            saved = json.loads((Path(temp) / "report.json").read_text())
            self.assertFalse(saved["complete"])
            markdown = (Path(temp) / "results.md").read_text()
            self.assertTrue("pending" in markdown or "待" in markdown)
            self.assertNotIn("| J00/pressure | True", markdown)


class IndependentPredictionValidationTest(unittest.TestCase):
    def make_archives(self, run):
        base = run / "eval" / "ckpt_best" / "predictions" / "test"
        volume_path = run / "geometry" / "volume.npz"
        volume_path.parent.mkdir(parents=True)
        np.savez(volume_path, n_cells=np.int64(3))
        entries = []
        for number in range(34):
            unit_id = f"AG/fast/CASE_{number:02d}"
            path = base / unit_id / "predictions.npz"
            path.parent.mkdir(parents=True)
            velocity = np.array([[10000., 10000., 10000.], [1., 0., 0.], [0., 2., 0.], [0., 0., 3.]])
            pressure = np.array([10., 20., 30., 40.]) + number
            true = np.column_stack([velocity, pressure])
            pred = np.column_stack([velocity * .9, pressure + 2.])
            np.savez(path, unit_id=np.asarray(unit_id), target=np.asarray("velocity_pressure"),
                     query_idx=np.arange(4), point_kind=np.array([0, 1, 1, 1], dtype=np.int8),
                     n_wall=np.int64(1), true_raw=true, pred_raw=pred)
            entries.append(dict(unit_id=unit_id, file=str(path.relative_to(base)), n_query=4,
                                bundle_path=str(volume_path.parent / "bundle.npz")))
        (base / "manifest.json").write_text(json.dumps({"target": "velocity_pressure", "cases": entries}))
        # Closed-form expectations: pressure error is exactly +2 Pa everywhere;
        # volume speed is [1,2,3] and predicted speed is 90% of truth.
        pressure_r2 = 1 - 4 / (125 + np.var(np.arange(34.)))
        common = {"per_case": {entry["unit_id"]: {} for entry in entries}}
        pressure_metric = {**common, "field_casebalanced": {"r2": pressure_r2},
                           "field": {"n": 136, "r2": pressure_r2, "mae": 2., "rmse": 2.}}
        velocity_rmse = np.sqrt(.14 / 3)
        velocity_metric = {**common, "field_casebalanced": {"r2": .93},
                           "field": {"n": 102, "r2": .93, "mae": .2, "rmse": velocity_rmse},
                           "vector": {"vector_rmse_m_s": velocity_rmse}}
        return base, pressure_metric, velocity_metric

    def test_joint_pressure_keeps_wall_and_velocity_excludes_wall(self):
        with tempfile.TemporaryDirectory() as temp:
            run = Path(temp)
            _, pressure, velocity = self.make_archives(run)
            p = R.independently_validate_predictions(run, "best", "pressure", pressure)
            v = R.independently_validate_predictions(run, "best", "velocity", velocity)
            self.assertTrue(p["passed"], p)
            self.assertTrue(v["passed"], v)
            self.assertEqual((p["n_points"], v["n_points"]), (136, 102))
            self.assertAlmostEqual(v["checks"]["vector_rmse"]["recomputed"], np.sqrt(.14 / 3))

    def test_duplicate_case_or_wrong_reported_point_count_cannot_pass(self):
        with tempfile.TemporaryDirectory() as temp:
            run = Path(temp)
            base, pressure, _ = self.make_archives(run)
            pressure["field"]["n"] = 135
            check = R.independently_validate_predictions(run, "best", "pressure", pressure)
            self.assertFalse(check["passed"])
            self.assertFalse(check["count_matches_metrics"])
            manifest = json.loads((base / "manifest.json").read_text())
            manifest["cases"][-1] = manifest["cases"][0]
            (base / "manifest.json").write_text(json.dumps(manifest))
            with self.assertRaisesRegex(ValueError, "case set|duplicates"):
                R.independently_validate_predictions(run, "best", "pressure", pressure)


if __name__ == "__main__":
    unittest.main()

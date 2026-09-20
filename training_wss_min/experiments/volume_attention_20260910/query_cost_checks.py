"""Synthetic checks of completion-only resource accounting, with no job mutations."""
import copy
import csv
import json
from pathlib import Path
import tempfile
import unittest

from joint_query_costs import write_joint_query_costs


class QueryCostChecks(unittest.TestCase):
    def fixture(self, folder):
        report = {"complete": True, "rows": []}
        runtime = {"arms": {}}
        for joint, number in (("J00", "01"), ("J01", "03")):
            runtime["arms"][joint] = {"unit_ids": list(range(8)), "output_shape": [79000, 4]}
            for checkpoint in ("best", "last"):
                base = folder / joint / "eval" / f"ckpt_{checkpoint}" / "predictions/test"
                base.mkdir(parents=True)
                (base / "manifest.json").write_text(json.dumps({"cases": [{"n_query": 10} for _ in range(34)]}))
                for aid, task, count, seconds in ((joint, "pressure", 340, 8), (joint, "velocity", 204, 8),
                                                  ("P" + number, "pressure", 340, 6), ("V" + number, "velocity", 204, 5)):
                    metrics_path = folder / f"{aid}_{task}_{checkpoint}.json"
                    metrics_path.write_text(json.dumps({"test": {"field": {"n": count}}}))
                    report["rows"].append(dict(id=aid, task=task, checkpoint=checkpoint,
                                               run_dir=str(folder / aid), metrics_path=str(metrics_path),
                                               train_seconds=seconds * 100, eval_seconds=seconds, parameters=100))
        (folder / "runtime_preflight.json").write_text(json.dumps(runtime))
        return report

    def test_execution_counted_once_and_overlap_saved(self):
        with tempfile.TemporaryDirectory() as temp:
            folder = Path(temp)
            evidence = write_joint_query_costs(self.fixture(folder), folder)
            self.assertEqual(evidence["rows"], 4)
            with Path(evidence["path"]).open() as stream:
                rows = list(csv.DictReader(stream))
            for row in rows:
                self.assertEqual(float(row["joint_eval_seconds"]), 8)
                self.assertEqual(float(row["separate_eval_seconds"]), 11)
                self.assertEqual(int(row["eval_saved_queries_vs_separate"]), 204)
                self.assertEqual(int(row["eval_extra_queries_vs_velocity"]), 136)
                self.assertEqual(int(row["preflight_saved_queries_vs_two_tasks"]), 1000)

    def test_incomplete_or_mismatched_task_cost_refused(self):
        with tempfile.TemporaryDirectory() as temp:
            folder = Path(temp)
            report = self.fixture(folder)
            pending = copy.deepcopy(report)
            pending["complete"] = False
            with self.assertRaisesRegex(ValueError, "completed acceptance"):
                write_joint_query_costs(pending, folder)
            report["rows"][0]["eval_seconds"] = 9
            with self.assertRaisesRegex(ValueError, "inconsistent"):
                write_joint_query_costs(report, folder)

    def test_missing_saved_case_cannot_report_savings(self):
        with tempfile.TemporaryDirectory() as temp:
            folder = Path(temp)
            report = self.fixture(folder)
            manifest = folder / "J00/eval/ckpt_best/predictions/test/manifest.json"
            manifest.write_text(json.dumps({"cases": [{"n_query": 10} for _ in range(33)]}))
            with self.assertRaisesRegex(ValueError, "counts differ"):
                write_joint_query_costs(report, folder)


if __name__ == "__main__":
    unittest.main()

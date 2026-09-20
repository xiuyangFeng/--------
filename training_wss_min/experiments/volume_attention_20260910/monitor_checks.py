"""Experiment-local acceptance checks for incremental publication and monitoring.

All fixtures use temporary directories; these checks never touch live reports,
the workbook, the training queue, or the frozen scientific source set.
"""
from __future__ import annotations

from contextlib import ExitStack
import copy
import fcntl
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import Mock, patch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import monitor as M


class HistoryChecks(unittest.TestCase):
    def test_missing_history_is_empty(self):
        with tempfile.TemporaryDirectory() as folder:
            self.assertEqual(M.history(folder), [])

    def test_complete_last_line_without_newline_is_preserved(self):
        with tempfile.TemporaryDirectory() as folder:
            Path(folder, "history.jsonl").write_text('{"epoch":0}\n{"epoch":1}')
            self.assertEqual(M.history(folder), [{"epoch": 0}, {"epoch": 1}])

    def test_incomplete_tail_is_ignored(self):
        with tempfile.TemporaryDirectory() as folder:
            Path(folder, "history.jsonl").write_text('{"epoch":0}\n{"epoch":1}\n{"epoch":')
            self.assertEqual(M.history(folder), [{"epoch": 0}, {"epoch": 1}])

    def test_corrupt_interior_line_is_not_hidden(self):
        with tempfile.TemporaryDirectory() as folder:
            Path(folder, "history.jsonl").write_text('{"epoch":0}\ncorrupted\n{"epoch":2}')
            with self.assertRaises(json.JSONDecodeError):
                M.history(folder)


def sample_report():
    rows = []
    for aid in ("P00", "P01", "P02", "P03"):
        for ck in ("best", "last"):
            row = {field: 1.25 for field in M.NUMERIC_FIELDS}
            row.update(id=aid, task="pressure", checkpoint=ck, parent="P00",
                       status="eval_last", r2_cb=0.9, delta_r2_parent=0.2,
                       groups={"AG": {"r2": 0.8}}, pressure_query_groups={"wall": {"r2": 0.7}},
                       longwave_summary={"sigma20": 0.1}, gate="未核验来源中的判定")
            rows.append(row)
    return dict(complete=False, rows=rows,
                comparisons=[dict(id="P01", reference="P00", task="pressure", checkpoint="best", qualified=True),
                             dict(id="P03", reference="P01", task="pressure", checkpoint="best", qualified=True)],
                factorials=[dict(task="pressure", arms=["P00", "P01", "P02", "P03"], checkpoint="best", r2_interaction=0.3)],
                gates={"P01/pressure": dict(qualified=True, longwave_improved=True,
                         issues=[], accepted_scientific_result=True, longwave_reductions={"20": 0.5, "40": 0.4})})


class PendingPublicationChecks(unittest.TestCase):
    def setUp(self):
        self.raw = sample_report()
        self.arms = {aid: dict(status="train", epochs=125) for aid in ("P00", "P01", "P02", "P03")}

    def test_unchecked_metrics_are_blank_and_source_is_unchanged(self):
        original = copy.deepcopy(self.raw)
        checked = {"P01/pressure/best", "P03/pressure/best"}
        report = M.pending_report(self.raw, checked, self.arms)
        self.assertFalse(report["complete"])
        for row in report["rows"]:
            key = f"{row['id']}/pressure/{row['checkpoint']}"
            if key not in checked:
                for field in M.NUMERIC_FIELDS:
                    self.assertIsNone(row[field], (key, field))
                self.assertEqual(row["groups"], {})
                self.assertEqual(row["pressure_query_groups"], {})
                self.assertNotIn("longwave_summary", row)
            else:
                self.assertEqual(row["r2_cb"], 0.9)
        self.assertEqual(self.raw, original)
        self.assertEqual([(p["id"], p["reference"]) for p in report["comparisons"]], [("P03", "P01")])

    def test_checked_child_does_not_publish_unchecked_parent_delta(self):
        report = M.pending_report(self.raw, {"P01/pressure/best"}, self.arms)
        child = next(row for row in report["rows"] if row["id"] == "P01" and row["checkpoint"] == "best")
        self.assertIsNone(child["delta_r2_parent"])

    def test_factorial_requires_every_constituent_checked(self):
        report = M.pending_report(self.raw, {"P01/pressure/best", "P03/pressure/best"}, self.arms)
        self.assertEqual(report["factorials"], [])
        all_checked = {f"P{i:02d}/pressure/best" for i in range(4)}
        report = M.pending_report(self.raw, all_checked, self.arms)
        self.assertEqual(report["factorials"], self.raw["factorials"])

    def test_pending_report_cannot_advertise_a_scientific_gate_pass(self):
        report = M.pending_report(self.raw, {"P01/pressure/best"}, self.arms)
        for gate in report["gates"].values():
            self.assertFalse(gate["accepted_scientific_result"])
            self.assertFalse(gate["qualified"])
            self.assertFalse(gate["longwave_improved"])


class FinalAndLockChecks(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.exp = Path(self.temp.name)
        (self.exp / "execution").mkdir()
        self.write("execution/queue_status.json", dict(status="running", source_sha256={"frozen": "same"}, arms={}))
        self.stack = ExitStack()
        self.addCleanup(self.stack.close)
        self.stack.enter_context(patch.object(M, "EXP", self.exp))
        self.stack.enter_context(patch.object(M, "fingerprints", return_value={"frozen": "same"}))
        self.stack.enter_context(patch.object(M, "snapshot", return_value={}))
        self.stack.enter_context(patch.object(M, "verify_data_inputs", return_value={"passed": True}))
        self.build = self.stack.enter_context(patch.object(M.R, "build_report", side_effect=AssertionError("Must not rebuild report")))
        self.write_report = self.stack.enter_context(patch.object(M.R, "write_outputs", side_effect=AssertionError("Must not rewrite report")))
        self.workbook = self.stack.enter_context(patch.object(M, "update_workbook", side_effect=AssertionError("Must not rewrite workbook")))
        self.documents = self.stack.enter_context(patch.object(M, "update_documents", side_effect=AssertionError("Must not rewrite documents")))
        # New scheduler diagnostics remain read-only and are orthogonal to these
        # publication tests. Stub them when available to avoid live Slurm I/O.
        for name in ("scheduler_jobs", "scheduler_status", "slurm_status", "scheduler_snapshot", "gpu_diagnostics"):
            if hasattr(M, name):
                self.stack.enter_context(patch.object(M, name, return_value={}))

    def write(self, relative, value):
        (self.exp / relative).write_text(json.dumps(value))

    def assert_no_publication(self):
        for mock in (self.build, self.write_report, self.workbook, self.documents):
            mock.assert_not_called()

    def test_completed_finalizer_prevents_any_pending_overwrite(self):
        self.write("completion_status.json", dict(status="complete", ended_at="finished"))
        self.write("xlsx_acceptance.json", dict(complete=True, all_historical_values_formulas_merges_preserved=True))
        self.write("report.json", dict(complete=True, final_marker="preserve byte-for-byte"))
        before = (self.exp / "report.json").read_bytes()
        self.assertTrue(M.tick())
        self.assertEqual((self.exp / "report.json").read_bytes(), before)
        self.assert_no_publication()
        state = json.loads((self.exp / "monitor_status.json").read_text())
        self.assertEqual(state["status"], "complete")

    def test_finalizer_claim_without_acceptance_is_rejected(self):
        self.write("completion_status.json", dict(status="complete", ended_at="finished"))
        self.write("xlsx_acceptance.json", dict(complete=False))
        self.write("report.json", dict(complete=True))
        with self.assertRaisesRegex(ValueError, "without matching"):
            M.tick()
        self.assert_no_publication()

    def test_busy_reporting_lock_skips_publication(self):
        with (self.exp / ".reporting.lock").open("a") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            self.assertFalse(M.tick())
        self.assert_no_publication()
        state = json.loads((self.exp / "monitor_status.json").read_text())
        self.assertEqual(state["status"], "finalizer_or_publisher_active")


if __name__ == "__main__":
    unittest.main(verbosity=2)

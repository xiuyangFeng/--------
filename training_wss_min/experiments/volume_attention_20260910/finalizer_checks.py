"""Focused checks for the experiment-local completion formatter.

No real finalizer main, trainer, workbook writer, or document mutation is run.
Historical evaluations are only read; all output checks use temporary folders.
"""
from __future__ import annotations

import copy
import csv
import importlib.util
import json
from pathlib import Path
import re
import tempfile
import unittest
from unittest.mock import patch

HERE=Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location("volume_attention_finalizer",HERE/"finalize.py")
module=importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


class CompletionChecks(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.report=json.loads((HERE/"report.json").read_text())
        cls.history_rows=[r for r in cls.report["rows"] if r["id"] in ("R5P","R5V")]

    def historical_report(self):
        return dict(rows=copy.deepcopy(self.history_rows),comparisons=[
            dict(id=r["id"],task=r["task"],checkpoint=r["checkpoint"],reference=r["id"])
            for r in self.history_rows])

    def test_live_readme_status_and_owned_block(self):
        original=(HERE/"README.md").read_text()
        updated=module.update_readme_status(original)
        self.assertIn("状态：26臂400轮及全部best/last已完成",updated)
        self.assertIn("用户于2026-09-10确认完整执行。",updated)
        self.assertIn("自动汇总 **14041** 已通过",updated)
        self.assertNotIn("14040运行中",updated)
        self.assertEqual(original.split("## 冻结协议",1)[1],updated.split("## 冻结协议",1)[1])
        self.assertEqual(module.update_readme_status(updated),updated)

    def test_legacy_readme_status_adds_completion_block(self):
        original="# Experiment\n\n状态：实现与验收中，尚无本矩阵正式训练成绩。\n\n## 冻结协议\n\nkeep\n"
        updated=module.update_readme_status(original)
        self.assertIn("## 当前执行状态",updated)
        self.assertIn("## 冻结协议\n\nkeep\n",updated)

    def test_missing_or_ambiguous_readme_status_refused(self):
        for document in ("# title\n\n## section\n", "状态：one\n状态：two\n"):
            with self.subTest(document=document),self.assertRaises(ValueError):
                module.update_readme_status(document)

    def test_tracker_links_resolve_to_real_experiment(self):
        text=module.render(self.report)
        links=re.findall(r"\]\((\.\./[^)]+)\)",text)
        self.assertTrue(links)
        for target in links:
            with self.subTest(target=target):
                self.assertEqual((module.TRACKER.parent/target).resolve().parent,HERE)

    def test_complete_and_pending_render_schema(self):
        pending=copy.deepcopy(self.report);pending["complete"]=False
        self.assertIn("不作最终结论",module.render(pending))
        complete=copy.deepcopy(pending);complete["complete"]=True
        for gate in complete["gates"].values():
            gate.update(qualified=False,longwave_improved=False)
        rendered=module.render(complete)
        self.assertIn("26臂及全部best/last已完成",rendered)
        self.assertIn("未同时通过两任务门槛",rendered)

    def test_historical_paired_cases_cover_tasks_and_diagnostics(self):
        with tempfile.TemporaryDirectory() as folder,patch.object(module,"EXP",Path(folder)):
            n=module.write_paired_cases(self.historical_report())
            self.assertEqual(n,4*34)
            with (Path(folder)/"per_case_comparisons.csv").open() as stream:
                rows=list(csv.DictReader(stream))
            self.assertEqual(len(rows),n)
            self.assertEqual({r["task"] for r in rows},{"pressure","velocity"})
            self.assertEqual({r["domain"] for r in rows},{"AG","AAA","ILO"})
            for row in rows:
                self.assertTrue(all(float(v)==0 for k,v in row.items() if k.startswith("delta_") and v))
                self.assertTrue(row["lw40_debiased_residual_mse"])
                if row["task"]=="velocity":
                    self.assertTrue(row["vector_rmse"])
                    self.assertTrue(row["circ_rmse"])
                    self.assertEqual(row["scalar_quantity"],"speed")
                else:
                    self.assertEqual(row["vector_rmse"],"")
                    self.assertEqual(row["scalar_unit"],"Pa")

    def test_empty_pairs_refused_before_output(self):
        with tempfile.TemporaryDirectory() as folder,patch.object(module,"EXP",Path(folder)):
            with self.assertRaisesRegex(ValueError,"No completed paired"):
                module.write_paired_cases(dict(rows=[],comparisons=[]))
            self.assertFalse((Path(folder)/"per_case_comparisons.csv").exists())

    def test_mismatched_case_identities_refused(self):
        report=self.historical_report()
        with tempfile.TemporaryDirectory() as folder,patch.object(module,"EXP",Path(folder)):
            row=report["rows"][0]
            metric=json.loads(Path(row["metrics_path"]).read_text())
            metric["test"]["per_case"].pop(next(iter(metric["test"]["per_case"])))
            path=Path(folder)/"altered_metrics.json";path.write_text(json.dumps(metric))
            row["metrics_path"]=str(path)
            with self.assertRaisesRegex(ValueError,"Longwave case identities"):
                module.write_paired_cases(report)

    def test_guarded_document_write_preserves_concurrent_change(self):
        with tempfile.TemporaryDirectory() as folder:
            path=Path(folder)/"document.md";path.write_text("new content")
            with self.assertRaisesRegex(RuntimeError,"concurrently"):
                module.guarded_write(path,"old content","replacement")
            self.assertEqual(path.read_text(),"new content")
            module.guarded_write(path,"new content","replacement")
            self.assertEqual(path.read_text(),"replacement")


if __name__=="__main__":
    suite=unittest.defaultTestLoader.loadTestsFromTestCase(CompletionChecks)
    result=unittest.TextTestRunner(verbosity=2).run(suite)
    evidence=dict(passed=result.wasSuccessful(),tests=result.testsRun,
                  failures=len(result.failures),errors=len(result.errors),timestamp=module.stamp(),
                  finalizer_sha256=module.sha256(HERE/"finalize.py"),
                  scope="Experiment-local formatter; historical JSON read only; temporary outputs; no real finalizer main or workbook/document mutation")
    module.save_json(HERE/"finalizer_checks.json",evidence)
    raise SystemExit(0 if result.wasSuccessful() else 1)

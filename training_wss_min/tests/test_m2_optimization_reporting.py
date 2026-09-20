"""M2 exploration decisions, independent arithmetic and workbook preservation."""
from __future__ import annotations

import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from openpyxl import Workbook, load_workbook

from training_wss_min.tools import report_m2_optimization as R
from training_wss_min.tools import update_m2_optimization_xlsx as W


def metric_values(score=.6):
    return {**{key: .5 for key in R.PATHS}, "physical_r2_cb": score, "mae": 2.0,
            "case_p10": .4, "top10_iou": .4, "negative_cases": 0}


def arm(aid, family, config, phase="base", members=None):
    return {"id": aid, "family": family, "parent": "MO0", "phase": phase,
            "run_name": "test/" + aid, "hypothesis": aid, "config": aid + ".json",
            "resolved_config": copy.deepcopy(config), "members": members or []}


def fixture(qualified=("L", "P", "S")):
    base = {"name": "test/MO0", "notes": "test", "data": {"support_n_points": 5000},
            "model": {"width": 32}, "train": {"lr": .001, "loss": "mse"}, "eval": {"fixed_support": True}}
    arms = [arm("MO0", "reference", base)]
    for family, change in (("L", ("train", "loss", "weighted")), ("P", ("train", "lr", .0005)), ("S", ("model", "width", 48))):
        cfg = copy.deepcopy(base)
        cfg[change[0]][change[1]] = change[2]
        arms.append(arm("MO-" + family + "1", family, cfg))
    reports = {}
    for ck in ("best", "last"):
        entries = {}
        for a in arms:
            score = .62 if a["family"] in qualified else .6
            entries[a["id"]] = {**a, **metric_values(score), "evidence": {"valid_scientific_result": True, "status": "已评估"}}
        reports[ck] = {"checkpoint": ck, "reference": metric_values(.6), "base_matrix_finalized": True, "arms": entries}
    return arms, reports


class SelectionTests(unittest.TestCase):
    def test_both_checkpoints_and_both_references_are_required(self):
        refs = {"historical": {"best": metric_values(.6), "last": metric_values(.6)},
                "MO0": {"best": metric_values(.61), "last": metric_values(.61)}}
        self.assertTrue(R.candidate_gate(metric_values(.62), metric_values(.615), refs)["qualified"])
        self.assertFalse(R.candidate_gate(metric_values(.619), metric_values(.615), refs)["qualified"])
        self.assertFalse(R.candidate_gate(metric_values(.62), metric_values(.614), refs)["qualified"])

    def test_each_guard_applies_also_to_last(self):
        refs = {"M2": {ck: metric_values() for ck in ("best", "last")}}
        failures = {"mae": 2.041, "case_p10": .379, "top10_iou": .389, "negative_cases": 1}
        for key, value in failures.items():
            with self.subTest(key=key):
                last = metric_values(.62)
                last[key] = value
                self.assertFalse(R.candidate_gate(metric_values(.62), last, refs)["qualified"])

    def test_incomplete_matrix_does_not_select_or_combine(self):
        arms, reports = fixture()
        reports["last"]["base_matrix_finalized"] = False
        self.assertFalse(R.select_family_winners(reports)["ready"])
        self.assertEqual(R.propose_combinations(arms, reports)["arms"], [])

    def test_zero_one_two_three_qualified_families(self):
        for families, expected in (((), 0), (("L",), 0), (("L", "S"), 1), (("L", "P", "S"), 4)):
            with self.subTest(families=families):
                arms, reports = fixture(families)
                plan = R.propose_combinations(arms, reports)
                self.assertEqual(len(plan["arms"]), expected)
                for f in {"L", "P", "S"} - set(families):
                    self.assertEqual(plan["selection"]["families"][f]["winner"], "MO0")

    def test_metadata_does_not_defeat_configuration_deduplication(self):
        arms, reports = fixture(("L", "P"))
        a, b = copy.deepcopy(arms[0]["resolved_config"]), copy.deepcopy(arms[0]["resolved_config"])
        a["name"], b["name"] = "foo", "bar"
        a["notes"], b["notes"] = "a", "b"
        self.assertEqual(R.config_fingerprint(a), R.config_fingerprint(b))
        result = R.propose_combinations(arms, reports)
        self.assertEqual(len(result["arms"]), 1)  # LP and LPS collapse, LS/PS duplicate singles

    def test_conflicting_changes_are_not_silently_overwritten(self):
        arms, reports = fixture()
        arms[1]["resolved_config"]["train"]["lr"] = .002
        result = R.propose_combinations(arms, reports)
        self.assertEqual(len(result["arms"]), 2)
        conflicts = [x for x in result["skipped"] if x["reason"] == "incompatible config changes"]
        self.assertEqual({x["id"] for x in conflicts}, {"MO-CLP", "MO-CLPS"})
        self.assertTrue(all("train.lr" in x["keys"] for x in conflicts))

    def test_combinations_do_not_become_new_family_winners(self):
        arms, reports = fixture()
        for ck in ("best", "last"):
            reports[ck]["arms"]["MO-CLP"] = {**reports[ck]["arms"]["MO-L1"], "id": "MO-CLP", "phase": "combination", "physical_r2_cb": .8}
        selection = R.select_family_winners(reports)
        self.assertEqual(selection["families"]["L"]["winner"], "MO-L1")
        self.assertEqual(selection["highest_qualified_candidate"], "MO-CLP")

    def test_negative_and_below_threshold_results_remain_in_observed_ranking(self):
        _, reports = fixture(())
        reports["best"]["arms"]["MO-L1"]["physical_r2_cb"] = .609
        reports["last"]["arms"]["MO-L1"]["physical_r2_cb"] = .607
        reports["best"]["arms"]["MO-S1"]["physical_r2_cb"] = .55
        selection = R.select_family_winners(reports)
        self.assertEqual(selection["highest_qualified_candidate"], "MO0")
        self.assertEqual(selection["highest_observed_candidate"], "MO-L1")
        self.assertEqual(set(selection["observed_ranking"]), {"MO-L1", "MO-P1", "MO-S1"})

    def add_combination(self, reports, best_score, last_score, mae=2.0):
        for ck, score in (("best", best_score), ("last", last_score)):
            reports[ck]["arms"]["MO-CLP"] = {
                **reports[ck]["arms"]["MO-L1"], "id": "MO-CLP", "phase": "combination",
                "family": "combination", "members": ["MO-L1", "MO-P1"],
                "physical_r2_cb": score, "mae": mae,
            }

    def test_final_combination_priority_rejects_lower_best_despite_original_gate(self):
        _, reports = fixture()
        self.add_combination(reports, .615, .62)
        selection = R.select_family_winners(reports)
        self.assertTrue(selection["candidates"]["MO-CLP"]["qualified"])
        self.assertIn("MO-CLP", selection["ranking"])
        self.assertNotIn("MO-CLP", selection["final_priority_candidates"])
        priority = selection["combination_priority"]["MO-CLP"]
        self.assertFalse(priority["eligible"])
        self.assertTrue(any("strictly exceed" in issue for issue in priority["issues"]))

    def test_final_combination_priority_rejects_best_tie_even_when_last_wins_sort(self):
        _, reports = fixture()
        self.add_combination(reports, .62, .70)
        selection = R.select_family_winners(reports)
        self.assertEqual(selection["ranking"][0], "MO-CLP")
        self.assertEqual(selection["highest_qualified_candidate"], "MO-CLP")
        self.assertFalse(selection["combination_priority"]["MO-CLP"]["eligible"])
        self.assertNotIn("MO-CLP", selection["final_priority_candidates"])

    def test_final_combination_priority_rejects_higher_best_when_guard_fails(self):
        _, reports = fixture()
        self.add_combination(reports, .65, .65, mae=2.05)
        selection = R.select_family_winners(reports)
        priority = selection["combination_priority"]["MO-CLP"]
        self.assertFalse(priority["eligible"])
        self.assertFalse(priority["candidate_gate_qualified"])
        self.assertGreater(priority["delta_vs_highest_valid_single"], 0)
        self.assertTrue(any("candidate_gate failed" in issue for issue in priority["issues"]))

    def test_final_combination_priority_accepts_actual_improvement_with_guards(self):
        _, reports = fixture()
        self.add_combination(reports, .63, .63)
        selection = R.select_family_winners(reports)
        self.assertTrue(selection["combination_priority"]["MO-CLP"]["eligible"])
        self.assertEqual(selection["final_priority_candidates"][0], "MO-CLP")
        self.assertEqual(selection["highest_final_priority_candidate"], "MO-CLP")

    def test_final_combination_priority_compares_even_guard_rejected_valid_singles(self):
        _, reports = fixture()
        reports["best"]["arms"]["MO-S1"].update(physical_r2_cb=.70, negative_cases=1)
        self.add_combination(reports, .65, .65)
        selection = R.select_family_winners(reports)
        self.assertFalse(selection["candidates"]["MO-S1"]["qualified"])
        priority = selection["combination_priority"]["MO-CLP"]
        self.assertEqual(priority["highest_valid_single_arm_ids"], ["MO-S1"])
        self.assertEqual(priority["highest_valid_single_best_physical_r2_cb"], .70)
        self.assertFalse(priority["eligible"])

    def test_pair_and_third_order_interaction_use_contemporaneous_reference(self):
        arms, reports = fixture()
        numbers = {"MO0": .60, "MO-L1": .62, "MO-P1": .63, "MO-S1": .64,
                   "MO-CLP": .67, "MO-CLS": .68, "MO-CPS": .70, "MO-CLPS": .76}
        for families in (("L", "P"), ("L", "S"), ("P", "S"), ("L", "P", "S")):
            aid = "MO-C" + "".join(families)
            members = ["MO-" + f + "1" for f in families]
            a = arm(aid, "combination", arms[0]["resolved_config"], "combination", members)
            arms.append(a)
            reports["best"]["arms"][aid] = {**a, **metric_values(), "evidence": {"valid_scientific_result": True}}
        for aid, number in numbers.items():
            reports["best"]["arms"][aid]["physical_r2_cb"] = number
        selection = {"ready": True, "families": {f: {"winner": "MO-" + f + "1"} for f in ("L", "P", "S")}}
        interactions = R.compute_interactions(reports["best"], selection, arms)
        self.assertAlmostEqual(interactions["pairs"]["LP"]["values"]["physical_r2_cb"], .02)
        self.assertAlmostEqual(interactions["triple"]["values"]["physical_r2_cb"], 0.0)
        reports["best"]["arms"]["MO-CLP"]["evidence"]["valid_scientific_result"] = False
        self.assertIsNone(R.compute_interactions(reports["best"], selection, arms)["triple"])


class WorkbookTests(unittest.TestCase):
    def workbook(self):
        workbook = Workbook()
        workbook.remove(workbook.active)
        for name in W.EXPECTED_SHEETS:
            ws = workbook.create_sheet(name)
            ws["A1"] = "historical"
            ws["G4"] = "=1+2"
            ws["H8"] = 1.234567890123
            ws.merge_cells("A2:D2")
        return workbook

    def rows(self):
        return {sheet: [[f"own row {i}", i] for i in range(W.CAPACITY)] for sheet in W.WIDTHS}

    def test_initial_append_and_repeat_preserve_every_historical_formula(self):
        workbook = self.workbook()
        positions, preserved, merges = W.update_in_memory(workbook, self.rows())
        W.verify_preserved(workbook, preserved, merges)
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "test.xlsx"
            workbook.save(path)
            workbook.close()
            workbook = load_workbook(path, data_only=False)
            W.verify_preserved(workbook, preserved, merges)
            sizes = {ws.title: ws.max_row for ws in workbook.worksheets}
            again, old, old_merges = W.update_in_memory(workbook, self.rows())
            self.assertEqual(positions, again)
            self.assertEqual(sizes, {ws.title: ws.max_row for ws in workbook.worksheets})
            W.verify_preserved(workbook, old, old_merges)

    def test_foreign_rows_after_and_cells_beside_section_survive(self):
        workbook = self.workbook()
        positions, _, _ = W.update_in_memory(workbook, self.rows())
        for sheet, item in positions.items():
            ws = workbook[sheet]
            ws.cell(item["last_row"] + 3, 1, "colleague section")
            ws.cell(item["last_row"] + 4, 7, "=G4*2")
            ws.cell(item["first_row"] + 3, W.WIDTHS[sheet] + 2, "colleague adjacent note")
        starts = {k: v["first_row"] for k, v in positions.items()}
        new, preserved, merges = W.update_in_memory(workbook, self.rows())
        self.assertEqual(starts, {k: v["first_row"] for k, v in new.items()})
        W.verify_preserved(workbook, preserved, merges)
        self.assertGreaterEqual(sum(v == "=G4*2" for v in preserved.values()), 3)

    def test_corrupt_ownership_or_duplicate_marker_rejected(self):
        for corruption in ("comment", "marker"):
            with self.subTest(corruption=corruption):
                workbook = self.workbook()
                positions, _, _ = W.update_in_memory(workbook, self.rows())
                ws = workbook["实验矩阵总览"]
                if corruption == "comment":
                    ws.cell(positions[ws.title]["first_row"] + 2, 1).comment = None
                else:
                    ws.cell(ws.max_row + 1, 1, W.TITLE)
                with self.assertRaises(ValueError):
                    W.update_in_memory(workbook, self.rows())

    def test_local_headers_match_reused_metric_column_conventions(self):
        header = W.make_headers("实验矩阵总览")
        self.assertEqual(header[26], "log_z pooled MAE")
        self.assertEqual(header[27], "log_z pooled RMSE")
        self.assertEqual(header[34], "log_z top10幅值比")
        self.assertEqual(header[38], "Pa fit slope cb")
        self.assertEqual(W.make_headers("教师汇报视图")[14], "log_z p99比")
        self.assertEqual(W.make_headers("汇总对比")[15], "best p99比")

    def test_concurrent_workbook_change_is_not_overwritten(self):
        with tempfile.TemporaryDirectory() as td:
            directory = Path(td)
            book = directory / "workbook.xlsx"
            workbook = self.workbook()
            workbook.save(book)
            workbook.close()
            reports = {ck: {"matrix_finalized": True, "completed_count": 0} for ck in ("best", "last")}
            original_update = W.update_in_memory
            concurrent = book.read_bytes() + b"external modification"
            def modified_elsewhere(wb, rows):
                result = original_update(wb, rows)
                book.write_bytes(concurrent)
                return result
            with patch.object(R, "report_all", return_value=reports), patch.object(R, "matrix_arms", return_value=[]), \
                    patch.object(W, "make_rows", return_value=self.rows()), patch.object(W, "update_in_memory", side_effect=modified_elsewhere):
                with self.assertRaisesRegex(RuntimeError, "changed concurrently"):
                    W.update_workbook(book=book, experiment_dir=directory)
            self.assertEqual(book.read_bytes(), concurrent)


class HistoricalArtifactTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        path = R.RUNS / R.ANCHOR / "eval/ckpt_best/metrics.json"
        if not path.exists():
            raise unittest.SkipTest("Historical M2 evaluation artifacts are not available")
        cls.metric = R.load_metrics(R.ANCHOR, "best")

    def test_all_case_aggregates_and_nochange_contributions(self):
        audit = R.audit_metric(self.metric, self.metric)
        self.assertTrue(audit["passed"])
        self.assertEqual(audit["aggregation_check_count"], 28)
        self.assertGreater(audit["per_case_numeric_fields_checked"], 5000)
        comparison = R.case_comparison(self.metric, self.metric)
        self.assertEqual(len(comparison["per_case"]), 34)
        self.assertEqual(comparison["contribution_sum"], 0)

    def test_corrupted_aggregate_or_case_count_is_detected(self):
        edited = copy.deepcopy(self.metric)
        edited["field"]["mae"] += .01
        self.assertFalse(R.audit_metric(edited, self.metric)["passed"])
        edited = copy.deepcopy(self.metric)
        case = next(iter(edited["per_case"]))
        edited["per_case"][case]["overall"]["n"] -= 1
        self.assertTrue(R.same_evaluation_population(edited, self.metric))

    def test_nonfinite_case_metric_is_detected_even_if_not_a_headline(self):
        edited = copy.deepcopy(self.metric)
        case = next(iter(edited["per_case"]))
        edited["per_case"][case]["distribution"]["pred_p99"] = float("nan")
        result = R.audit_metric(edited, self.metric)
        self.assertFalse(result["passed"])
        self.assertEqual(len(result["nonfinite_fields"]), 1)

    def test_workbook_rows_transcribe_physical_and_normalized_fields(self):
        reference = {"id": "historical_m2", **R.values(self.metric)}
        candidate = arm("MO-L1", "L", {})
        candidate.update(run_name=R.ANCHOR, changes={"train.loss_raw_mse_lambda": .01})
        entry = {**R.values(self.metric), "evidence": {"valid_scientific_result": True, "status": "已评估"}}
        reports = {ck: {"reference": reference, "arms": {"MO-L1": entry}} for ck in ("best", "last")}
        reports["selection"] = {"candidates": {}}
        rows = W.make_rows([candidate], reports)
        overview, teacher, summary = (rows[name][0] for name in ("实验矩阵总览", "教师汇报视图", "汇总对比"))
        self.assertEqual(overview[6], self.metric["field_casebalanced"]["r2"])
        self.assertEqual(overview[26], self.metric["normalized"]["field"]["mae"])
        self.assertEqual(overview[27], self.metric["normalized"]["field"]["rmse"])
        self.assertEqual(teacher[14], self.metric["normalized"]["calibration"]["p99_pred_true_ratio"])
        self.assertEqual(summary[15], self.metric["calibration"]["p99_pred_true_ratio"])
        self.assertEqual(rows["实验矩阵总览"][1][6], self.metric["field_casebalanced"]["r2"])

    def test_checkpoint_selection_is_verified_against_actual_payload(self):
        import torch
        historical_run = R.RUNS / R.ANCHOR
        history = [json.loads(line) for line in (historical_run / "history.jsonl").read_text().splitlines()]
        selected = min(history, key=lambda h: h["train_loss"])
        self.assertEqual(selected["epoch"], 378)
        payload = torch.load(historical_run / "ckpt_best.pt", map_location="cpu", weights_only=False)
        self.assertEqual(payload["epoch"], selected["epoch"])
        self.assertAlmostEqual(payload["metric"], -selected["train_loss"])

    def test_run_evidence_checks_checkpoint_score_and_training_statistics(self):
        run = R.RUNS / R.ANCHOR
        config = json.loads((run / "config.json").read_text())
        a = arm("MO0", "control", config)
        a["run_name"] = R.ANCHOR
        hashes = {str(run / "config.json"): R.sha256(run / "config.json")}
        state = {"status": "complete", "ended_at": "done", "source_sha256": hashes, "source_sha256_end": hashes,
                 "arms": {"MO0": {"run_name": R.ANCHOR, "status": "complete", "stages": {
                     "train": {"returncode": 0}, "eval_best": {"returncode": 0}, "eval_last": {"returncode": 0}}}}}
        _, evidence = R.run_evidence(a, "best", state, self.metric)
        self.assertTrue(evidence["valid_scientific_result"], evidence["issues"])
        with patch("torch.load", return_value={"epoch": 399, "metric": -999}):
            _, evidence = R.run_evidence(a, "best", state, self.metric)
        self.assertFalse(evidence["valid_scientific_result"])
        self.assertTrue(any("checkpoint epoch" in message for message in evidence["issues"]))
        self.assertTrue(any("checkpoint score" in message for message in evidence["issues"]))
        mismatched = copy.deepcopy(a)
        mismatched["resolved_config"]["train"]["lr"] = .05
        _, evidence = R.run_evidence(mismatched, "best", state, self.metric)
        self.assertFalse(evidence["valid_scientific_result"])
        self.assertIn("saved run config differs from manifest: train.lr", evidence["issues"])


if __name__ == "__main__":
    unittest.main()

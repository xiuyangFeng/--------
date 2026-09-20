"""Completion-only resource accounting; no changes to frozen training sources."""
from __future__ import annotations

import csv
import json
import math
from pathlib import Path


def write_joint_query_costs(report, output_dir):
    """Count joint execution once, with comparable separate-task measurements.

    Training query counts come from the real eight-case preflight batch and are
    explicitly not estimates of the full 400-epoch mean. Evaluation counts come
    from independently validated, complete saved-query manifests and metrics.
    """
    if not report.get("complete"):
        raise ValueError("Joint resource accounting requires completed acceptance")
    output_dir = Path(output_dir)
    runtime = json.loads((output_dir / "runtime_preflight.json").read_text())
    indexed = {(r["id"], r["task"], r["checkpoint"]): r for r in report["rows"]}
    rows = []
    for joint, number in (("J00", "01"), ("J01", "03")):
        preflight = runtime["arms"][joint]
        batch_cases = len(preflight["unit_ids"])
        batch_queries = int(preflight["output_shape"][0])
        if not 7500 * batch_cases <= batch_queries <= 10000 * batch_cases:
            raise ValueError("Joint preflight query union outside protocol limits")
        for checkpoint in ("best", "last"):
            p, v = (indexed[(joint, task, checkpoint)] for task in ("pressure", "velocity"))
            sp, sv = (indexed[(prefix + number, task, checkpoint)]
                      for prefix, task in (("P", "pressure"), ("V", "velocity")))
            for field in ("eval_seconds", "train_seconds", "parameters"):
                if not math.isclose(float(p[field]), float(v[field]), rel_tol=1e-10, abs_tol=1e-10):
                    raise ValueError(f"Joint {field} is inconsistent across task reports")
            metrics = [json.loads(Path(r["metrics_path"]).read_text())["test"] for r in (p, v, sp, sv)]
            pn, vn, spn, svn = (int(m["field"]["n"]) for m in metrics)
            base = Path(p["run_dir"]) / "eval" / f"ckpt_{checkpoint}" / "predictions/test"
            saved = json.loads((base / "manifest.json").read_text())
            actual_queries = sum(int(case["n_query"]) for case in saved["cases"])
            if not (len(saved["cases"]) == 34 and actual_queries == pn == spn and vn == svn and vn < pn):
                raise ValueError("Joint/full-field query counts differ from paired single tasks")
            single_eval = float(sp["eval_seconds"]) + float(sv["eval_seconds"])
            single_train = float(sp["train_seconds"]) + float(sv["train_seconds"])
            rows.append(dict(
                id=joint, checkpoint=checkpoint, pressure_reference="P" + number,
                velocity_reference="V" + number, training_queries_per_task_per_case=5000,
                preflight_cases=batch_cases, preflight_union_queries=batch_queries,
                preflight_union_queries_case_mean=batch_queries / batch_cases,
                preflight_extra_queries_vs_one_task_ratio=batch_queries / (5000 * batch_cases) - 1,
                preflight_saved_queries_vs_two_tasks=10000 * batch_cases - batch_queries,
                training_query_scope="real preflight batch only; union varies by case/epoch",
                eval_union_queries=actual_queries, eval_pressure_points=pn, eval_velocity_points=vn,
                eval_extra_queries_vs_pressure=actual_queries - spn,
                eval_extra_queries_vs_velocity=actual_queries - svn,
                eval_separate_queries=spn + svn, eval_saved_queries_vs_separate=spn + svn - actual_queries,
                joint_eval_seconds=float(p["eval_seconds"]), separate_eval_seconds=single_eval,
                joint_over_separate_eval_time=float(p["eval_seconds"]) / single_eval,
                joint_train_seconds=float(p["train_seconds"]), separate_train_seconds=single_train,
                joint_over_separate_train_time=float(p["train_seconds"]) / single_train,
                joint_parameters=int(p["parameters"]), separate_parameters=int(sp["parameters"] + sv["parameters"]),
                timing_scope="observed concurrent queue wall time; joint counted once; best/last share training cost",
            ))
    dest = output_dir / "joint_query_costs.csv"
    with dest.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    return {"path": str(dest), "rows": len(rows), "joint_execution_counted_once": True}

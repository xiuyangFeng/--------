"""Execute the authorized E0--E9 + C1--C4 single-seed recovery matrix.

Uses the repaired queue's GPU memory gates, subprocess cleanup and atomic status
records. Combination choice changes only execution records; frozen configs are
never rewritten. Test34-driven selection is explicitly exploratory.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time

from training_wss_min.tools import run_next_matrix_queue as base

ROOT = base.ROOT
EXPECTED_ARMS = {*(f"E{i}" for i in range(10)), *(f"C{i}" for i in range(1, 5))}
STRUCTURES = ("E2", "E3", "E4", "E5")
METRIC_PATHS = {
    "norm_r2": "test.normalized.field_casebalanced.r2",
    "pa_r2": "test.field_casebalanced.r2",
    "pa_mae": "test.field.mae",
    "iou": "test.hotspot.top10_iou_casemean",
    "pa_p10": "test.aggregate.r2_casep10",
}


def metric_summary(path: Path):
    """Read the current evaluator's actual schema, rejecting partial/NaN data."""
    contents = path.read_bytes()
    document = json.loads(contents)
    values = {}
    for name, dotted in METRIC_PATHS.items():
        value = document
        for key in dotted.split("."):
            value = value[key]
        value = float(value)
        if not math.isfinite(value):
            raise ValueError(f"non-finite selection metric {dotted} in {path}")
        values[name] = value
    test = document["test"]
    cases = test["per_case"]
    normalized_cases = test["normalized"]["per_case"]
    if not cases or set(cases) != set(normalized_cases):
        raise ValueError(f"physical/normalized case sets differ or are empty: {path}")
    negatives = []
    for case, item in cases.items():
        r2 = float(item["overall"]["r2"])
        if not math.isfinite(r2):
            raise ValueError(f"non-finite per-case R2 for {case} in {path}")
        if r2 < 0:
            negatives.append(case)
    if int(test["aggregate"]["n_cases"]) != len(cases):
        raise ValueError(f"aggregate case count does not match per_case: {path}")
    values.update(negative_cases=sorted(negatives), case_ids=sorted(cases),
                  source=str(path), source_sha256=hashlib.sha256(contents).hexdigest())
    return values


def qualification(candidate, reference):
    """Pre-registered E0-relative gates, including negative-case identity."""
    if candidate["case_ids"] != reference["case_ids"]:
        raise ValueError("selection candidate and E0 evaluated different cases")
    extra_negative = sorted(set(candidate["negative_cases"]) - set(reference["negative_cases"]))
    checks = {
        "normalized_cb_r2_gain_at_least_0.005": candidate["norm_r2"] - reference["norm_r2"] >= .005 - 1e-12,
        "physical_cb_r2_improves": candidate["pa_r2"] > reference["pa_r2"],
        "physical_field_mae_increase_at_most_2pct": candidate["pa_mae"] <= reference["pa_mae"] * 1.02 + 1e-12,
        "top10_iou_drop_at_most_0.010": candidate["iou"] >= reference["iou"] - .010 - 1e-12,
        "physical_case_p10_drop_at_most_0.020": candidate["pa_p10"] >= reference["pa_p10"] - .020 - 1e-12,
        "no_additional_negative_cases": not extra_negative,
    }
    return {"qualified": all(checks.values()), "checks": checks,
            "additional_negative_cases": extra_negative,
            "delta_normalized_cb_r2": candidate["norm_r2"] - reference["norm_r2"],
            "delta_physical_cb_r2": candidate["pa_r2"] - reference["pa_r2"]}


def choose_structure(summaries):
    gates = {aid: qualification(summaries[aid], summaries["E0"]) for aid in STRUCTURES}
    eligible = [aid for aid in STRUCTURES if gates[aid]["qualified"]]
    ranked = sorted(eligible or STRUCTURES,
                    key=lambda aid: (-summaries[aid]["norm_r2"], -summaries[aid]["pa_r2"], aid))
    return {"selected_structure": ranked[0], "eligible_structures": eligible,
            "ranking": ranked, "forced_exploratory": not bool(eligible),
            "qualification": gates, "metric_paths": METRIC_PATHS,
            "disclosure": "Adaptive test34 selection; exploratory single-seed evidence."}


class RecoveryQueue(base.MatrixQueue):
    def __init__(self, args):
        super().__init__(args)
        matrix_path = args.matrix or args.config_dir / "matrix.json"
        self.matrix = json.loads(matrix_path.read_text())
        records = self.state["arms"]
        if set(records) != EXPECTED_ARMS:
            raise ValueError("recovery matrix must contain exactly E0-E9 and C1-C4")
        for aid, record in records.items():
            if record["phase"] != (0 if aid.startswith("E") else 1):
                raise ValueError(f"wrong recovery phase for {aid}")
        c4 = next(a for a in self.matrix["arms"] if a["id"] == "C4")
        candidates = c4.get("candidate_configs", {})
        if set(candidates) != set(STRUCTURES):
            raise ValueError("C4 requires frozen E2-E5 candidate configs")
        self.c4_configs = {}
        for aid, filename in candidates.items():
            path = (args.config_dir / filename).resolve()
            if not path.is_relative_to(args.config_dir.resolve()):
                raise ValueError("C4 candidate config escapes frozen config directory")
            config = json.loads(path.read_text())
            if config["name"] != records["C4"]["run_name"]:
                raise ValueError("every C4 candidate must retain the one C4 run name")
            self.c4_configs[aid] = str(path)
        self.state.update(matrix="wss_direct_recovery_20260912", expected_training_runs=14,
                          expected_evaluations=28, phase1_prepared=False)

    def verify_gate(self):
        """Must verify frozen source/config/data against successful GPU preflight."""
        from training_wss_min.tools.wss_recovery_common import verify_preflight
        result = verify_preflight()
        self.preflight_path = Path(result["preflight"])
        self.preflight_sha256 = result["sha256"]
        self.frozen_fingerprints = json.loads(self.preflight_path.read_text())["fingerprints"]
        return result

    def verify_source(self):
        """Check small code/config/protocol hashes before each subprocess stage."""
        from training_wss_min.tools.wss_recovery_common import fingerprints, sha256, execution_fingerprints, EXP
        if (sha256(self.preflight_path) != self.preflight_sha256 or
                fingerprints() != self.frozen_fingerprints):
            raise RuntimeError("source/config/protocol or preflight evidence changed during matrix execution")
        execution = json.loads((EXP / "execution_gate.json").read_text())
        if execution.get("passed") is not True or execution["files"] != execution_fingerprints():
            raise RuntimeError("execution/report/verification tools changed during matrix execution")

    def verify_evaluation(self, aid, checkpoint):
        """Must independently recompute saved full-wall predictions before success."""
        self.verify_source()
        run = ROOT / "training_wss_min/runs" / self.state["arms"][aid]["run_name"]
        command = [base.PY, "-u", "-m", "training_wss_min.tools.verify_wss_recovery",
                   "--run-dir", str(run), "--checkpoint", checkpoint]
        log = self.exp / "logs" / f"{aid}_verify_{checkpoint}.log"
        started = base.stamp()
        with log.open("x") as stream:
            subprocess.run(command, cwd=ROOT, stdout=stream, stderr=subprocess.STDOUT,
                           check=True, timeout=600)
        return {"status": "passed", "command": command, "log": str(log),
                "started_at": started, "ended_at": base.stamp()}

    def refuse_overwrite(self):
        super().refuse_overwrite()
        for aid in self.state["arms"]:
            for checkpoint in ("best", "last"):
                log = self.exp / "logs" / f"{aid}_verify_{checkpoint}.log"
                if log.exists():
                    raise FileExistsError(f"refusing to overwrite historical verification log: {log}")

    def run(self):
        # A missing or failed source/config/data/GPU gate cannot launch training.
        self.state["preflight_verification"] = self.verify_gate()
        return super().run()

    def launch(self, aid, gpu, stage):
        record = self.state["arms"][aid]
        if aid == "C4" and not self.state["phase1_prepared"]:
            raise RuntimeError("C4 cannot use its placeholder config before phase-one selection")
        self.verify_source()
        run = ROOT / "training_wss_min/runs" / record["run_name"]
        if stage == "train" and run.exists():
            raise FileExistsError(f"run appeared while queued: {run}")
        if stage == "train":
            command = [base.PY, "-u", "-m", "training_wss_min.train", "--config", record["config"]]
        else:
            command = [base.PY, "-u", "-m", "training_wss_min.evaluate", "--run-dir", str(run),
                       "--partitions", "test", "--allow-test", "--checkpoint", stage[5:],
                       "--no-plots", "--save-predictions"]
        log = self.exp / "logs" / f"{aid}_{stage}.log"
        record.update(status=stage, gpu=gpu)
        stage_record = {"started_at": base.stamp(), "log": str(log), "cmd": command}
        record["stages"][stage] = stage_record
        env = os.environ.copy()
        env.update(CUDA_VISIBLE_DEVICES=gpu, OMP_NUM_THREADS="2", OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1")
        stream = None
        try:
            stream = log.open("x")
            process = subprocess.Popen(command, cwd=ROOT, env=env, stdout=stream,
                                       stderr=subprocess.STDOUT, start_new_session=True)
        except BaseException as exc:
            if stream is not None:
                stream.close()
            stage_record.update(ended_at=base.stamp(), error=f"{type(exc).__name__}: {exc}")
            record.update(status="failed", failed_stage=stage)
            raise
        self.active[aid] = {"process": process, "stream": stream, "gpu": gpu, "stage": stage}
        stage_record["pid"] = process.pid
        self.next_launch[gpu] = time.monotonic() + self.args.launch_interval_seconds
        self.save()
        print(base.stamp(), aid, stage, gpu, flush=True)

    def poll(self):
        for aid, task in list(self.active.items()):
            process = task["process"]
            code = process.poll()
            if code is None:
                continue
            process.wait()
            task["stream"].close()
            record = self.state["arms"][aid]
            stage = task["stage"]
            step = record["stages"][stage]
            step.update(returncode=code, ended_at=base.stamp())
            del self.active[aid]
            if code:
                record.update(status="failed", failed_stage=stage)
            else:
                try:
                    if stage.startswith("eval_"):
                        step["prediction_verification"] = self.verify_evaluation(aid, stage[5:])
                    record["status"] = {"train": "ready_eval_best", "eval_best": "ready_eval_last",
                                        "eval_last": "complete"}[stage]
                except Exception as exc:
                    step["verification_error"] = f"{type(exc).__name__}: {exc}"
                    record.update(status="failed", failed_stage=f"verify_{stage}")
            self.save()

    def prepare_combinations(self):
        records = self.state["arms"]
        if not all(records[f"E{i}"]["status"] == "complete" for i in range(10)):
            raise RuntimeError("cannot select combinations until every phase-zero arm completes")
        summaries = {
            f"E{i}": metric_summary(ROOT / "training_wss_min/runs" / records[f"E{i}"]["run_name"] /
                                    "eval/ckpt_best/metrics.json")
            for i in range(10)
        }
        selection = choose_structure(summaries)
        selection.update(selected_at=base.stamp(), source_metrics=summaries)
        self.state["combination_selection"] = selection
        constituents = {"C1": ("E2", "E3"), "C2": ("E2", "E6"), "C3": ("E3", "E7"),
                        "C4": (selection["selected_structure"], "E9")}
        for aid, ids in constituents.items():
            gates = {item: qualification(summaries[item], summaries["E0"]) for item in ids}
            records[aid]["constituents"] = list(ids)
            records[aid]["constituent_gates"] = gates
            records[aid]["exploratory"] = True
            records[aid]["forced_exploratory"] = any(not gate["qualified"] for gate in gates.values())
            records[aid]["execution_policy"] = "run requested combination irrespective of efficacy gates"
        records["C4"]["placeholder_config"] = records["C4"]["config"]
        records["C4"]["config"] = self.c4_configs[selection["selected_structure"]]
        records["C4"]["selected_structure"] = selection["selected_structure"]
        self.state["phase1_prepared"] = True
        self.save()

    def schedule(self):
        records = self.state["arms"]
        if (not self.state["phase1_prepared"] and
                all(records[f"E{i}"]["status"] == "complete" for i in range(10))):
            try:
                self.prepare_combinations()
            except Exception as exc:
                self.state["combination_selection_error"] = f"{type(exc).__name__}: {exc}"
                for aid, record in records.items():
                    if aid.startswith("C") and record["status"] not in base.TERMINAL:
                        record.update(status="blocked", reason="combination metrics/config verification failed")
                self.save()
                return
        super().schedule()


def write_report_safely(args):
    """Refresh documents/workbook after every exit without masking queue status."""
    failure = None
    log = args.experiment_dir / "logs" / f"report_{time.time_ns()}.log"
    command = [base.PY, "-u", "-m", "training_wss_min.tools.report_wss_recovery",
               "--experiment-dir", str(args.experiment_dir), "--config-dir", str(args.config_dir)]
    try:
        log.parent.mkdir(parents=True, exist_ok=True)
        with log.open("x") as stream:
            result = subprocess.run(command, cwd=ROOT, stdout=stream, stderr=subprocess.STDOUT, timeout=600)
        if result.returncode:
            failure = {"returncode": result.returncode, "error": "recovery report command failed"}
    except Exception as exc:
        failure = {"error": f"{type(exc).__name__}: {exc}"}
    if failure is not None:
        failure.update(reported_at=base.stamp(), log=str(log), command=command)
        try:
            args.experiment_dir.mkdir(parents=True, exist_ok=True)
            base.atomic_write_status(args.experiment_dir / "report_failure.json", failure)
        except Exception as exc:
            print(f"unable to record recovery report failure: {exc}; original failure: {failure}", file=sys.stderr)


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--config-dir", type=Path, default=ROOT / "training_wss_min/configs/wss_direct_recovery_20260912")
    parser.add_argument("--matrix", type=Path)
    parser.add_argument("--experiment-dir", type=Path, default=ROOT / "training_wss_min/experiments/wss_direct_recovery_20260912")
    parser.add_argument("--slots-per-gpu", type=int, default=2)
    parser.add_argument("--min-free-mib", type=int, default=7000)
    parser.add_argument("--launch-interval-seconds", type=float, default=60)
    parser.add_argument("--poll-seconds", type=float, default=3)
    parser.add_argument("--terminate-timeout-seconds", type=float, default=20)
    args = parser.parse_args(argv)
    if (args.slots_per_gpu < 1 or args.min_free_mib < 0 or args.launch_interval_seconds < 0
            or args.poll_seconds <= 0 or args.terminate_timeout_seconds <= 0):
        parser.error("invalid slot, memory or timing limit")
    try:
        return RecoveryQueue(args).run()
    finally:
        write_report_safely(args)


if __name__ == "__main__":
    raise SystemExit(main())

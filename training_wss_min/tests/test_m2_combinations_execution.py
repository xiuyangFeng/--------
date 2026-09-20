"""Selected combination materialisation and runtime-gate wiring in temporary dirs."""
from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest
import torch

from training_wss_min import config as C
from training_wss_min.tools import m2_optimization_common as common
from training_wss_min.tools import prepare_m2_combinations as prepare
from training_wss_min.tools import preflight_m2_combinations as preflight
from training_wss_min.tools import prepare_m2_optimization as base_prepare
from training_wss_min.tools import report_m2_optimization as reporting
from training_wss_min.tools import run_m2_optimization as queue


@pytest.fixture
def isolated(tmp_path, monkeypatch):
    exp, configs = tmp_path / "experiment", tmp_path / "configs"
    exp.mkdir(); configs.mkdir()
    matrix_path, combination_path = configs / "matrix.json", configs / "combinations.json"
    matrix = base_prepare.payloads()
    common.save_json(matrix_path, matrix)
    common.save_json(exp / "queue_status.json", {"status": "complete"})
    source = tmp_path / "source.py"
    source.write_text("frozen = True\n")
    hashes = {"source.py": common.sha(source)}
    for module in (prepare, preflight):
        monkeypatch.setattr(module, "EXP", exp)
        monkeypatch.setattr(module, "CONFIGS", configs)
        monkeypatch.setattr(module, "MATRIX", matrix_path)
        monkeypatch.setattr(module, "COMBINATIONS", combination_path)
    monkeypatch.setattr(preflight, "fingerprints", lambda phase: copy.deepcopy(hashes))
    common.save_json(exp / "runtime_preflight.json", dict(
        passed=True, source_sha256=hashes, source_sha256_end=hashes,
    ))
    reports = {}
    for checkpoint in ("best", "last"):
        entries = {}
        for arm in matrix["arms"]:
            entries[arm["id"]] = {**copy.deepcopy(arm), "physical_r2_cb": .60,
                "mae": 2., "top10_iou": .4, "case_p10": .4, "negative_cases": 0,
                "evidence": {"valid_scientific_result": True}}
        reports[checkpoint] = dict(base_matrix_finalized=True, arms=entries,
            reference=dict(physical_r2_cb=.60, mae=2., top10_iou=.4, case_p10=.4, negative_cases=0))
    monkeypatch.setattr(prepare, "report_all", lambda **kwargs: copy.deepcopy(reports))
    return dict(exp=exp, configs=configs, matrix_path=matrix_path, combinations=combination_path,
                matrix=matrix, reports=reports, hashes=hashes, root=tmp_path)


def _qualify(fixture, families=("L", "P", "S")):
    for checkpoint in ("best", "last"):
        for family in families:
            fixture["reports"][checkpoint]["arms"][f"MO-{family}1"]["physical_r2_cb"] = .62


def _manifest(fixture):
    return json.loads(fixture["combinations"].read_text())


def test_only_complete_ready_base_results_can_create_combination_files(isolated):
    _qualify(isolated)
    isolated["reports"]["last"]["base_matrix_finalized"] = False
    with pytest.raises(RuntimeError, match="qualified first"):
        prepare.main()
    assert not isolated["combinations"].exists()
    assert not list(isolated["configs"].glob("MO-C*.json"))
    assert not (isolated["exp"] / "combination_decision.json").exists()


@pytest.mark.parametrize("families,ids", [
    ((), []), (("L",), []), (("L", "P"), ["MO-CLP"]),
    (("L", "S"), ["MO-CLS"]), (("P", "S"), ["MO-CPS"]),
    (("L", "P", "S"), ["MO-CLP", "MO-CLS", "MO-CPS", "MO-CLPS"]),
])
def test_materialises_triple_pairs_and_two_family_fallback_without_duplicates(isolated, families, ids):
    _qualify(isolated, families)
    prepare.main()
    manifest = _manifest(isolated)
    assert [arm["id"] for arm in manifest["arms"]] == ids
    assert len(manifest["arms"]) <= 4
    for arm in manifest["arms"]:
        config = json.loads((isolated["configs"] / arm["config"]).read_text())
        assert config == arm["resolved_config"]
        assert reporting.config_fingerprint(config) == arm["selected_configuration_sha256"]
        assert prepare.effective_configuration_sha256(config) == arm["effective_configuration_sha256"]
        assert config["data"] == isolated["matrix"]["arms"][0]["resolved_config"]["data"]
    if not ids:
        with pytest.raises(RuntimeError, match="No additional combinations"):
            preflight.validate_frozen_inputs(manifest)


def test_default_expansion_has_same_effective_identity_without_changing_raw_hash_contract(isolated):
    raw = isolated["matrix"]["arms"][0]["resolved_config"]
    expanded = C.ExpConfig.from_dict(raw).to_dict()
    expanded["name"], expanded["notes"] = "different_label", "metadata"
    assert prepare.effective_configuration_sha256(raw) == prepare.effective_configuration_sha256(expanded)
    assert reporting.config_fingerprint(raw) != reporting.config_fingerprint(expanded)


def test_repeated_prepare_keeps_all_frozen_files_and_ignores_later_combination_scores(isolated, monkeypatch):
    _qualify(isolated)
    monkeypatch.setattr(prepare, "stamp", lambda: "original-time")
    prepare.main()
    paths = list(isolated["configs"].glob("MO-C*.json")) + [isolated["combinations"], isolated["exp"] / "combination_decision.json"]
    before = {path: (path.read_bytes(), path.stat().st_mtime_ns) for path in paths}
    for checkpoint in ("best", "last"):
        extra = copy.deepcopy(isolated["reports"][checkpoint]["arms"]["MO-L1"])
        extra.update(id="MO-CLPS", phase="combination", family="C", physical_r2_cb=.9)
        isolated["reports"][checkpoint]["arms"][extra["id"]] = extra
    monkeypatch.setattr(prepare, "stamp", lambda: "new-time")
    prepare.main()
    assert {path: (path.read_bytes(), path.stat().st_mtime_ns) for path in paths} == before


def test_complete_write_set_is_validated_before_any_new_config_is_written(isolated):
    _qualify(isolated)
    conflict = isolated["configs"] / "MO-CLPS_s1234.json"
    conflict.write_text('{"unrelated":true}\n')
    with pytest.raises(RuntimeError, match="config conflict"):
        prepare.main()
    assert conflict.read_text() == '{"unrelated":true}\n'
    assert not (isolated["configs"] / "MO-CLP_s1234.json").exists()
    assert not isolated["combinations"].exists()


def test_existing_manifest_requires_all_original_config_files(isolated):
    _qualify(isolated, ("L", "P"))
    prepare.main()
    config_path = isolated["configs"] / "MO-CLP_s1234.json"
    config_path.unlink()
    before = isolated["combinations"].read_bytes()
    with pytest.raises(RuntimeError, match="config is missing"):
        prepare.main()
    assert isolated["combinations"].read_bytes() == before
    assert not config_path.exists()


def test_rejects_over_budget_and_forged_raw_identity_before_writes(isolated, monkeypatch):
    _qualify(isolated)
    suggestion = reporting.propose_combinations(isolated["matrix"]["arms"], isolated["reports"])
    oversized = copy.deepcopy(suggestion)
    oversized["arms"].append(copy.deepcopy(oversized["arms"][0]))
    monkeypatch.setattr(prepare, "propose_combinations", lambda *args: oversized)
    with pytest.raises(RuntimeError, match="At most four"):
        prepare.main()
    suggestion["arms"][0]["config_sha256"] = "forged"
    monkeypatch.setattr(prepare, "propose_combinations", lambda *args: suggestion)
    with pytest.raises(RuntimeError, match="raw configuration identity"):
        prepare.main()
    assert not isolated["combinations"].exists()
    assert not list(isolated["configs"].glob("MO-C*.json"))


def test_effective_default_only_duplicate_is_rejected_even_if_raw_hash_differs(isolated, monkeypatch):
    _qualify(isolated, ("L", "P"))
    suggestion = reporting.propose_combinations(isolated["matrix"]["arms"], isolated["reports"])
    base = copy.deepcopy(isolated["matrix"]["arms"][0]["resolved_config"])
    base["train"]["loss_raw_mse_lambda"] = 0.0
    suggestion["arms"][0].update(changes={"train.loss_raw_mse_lambda": 0.0},
                                config_sha256=reporting.config_fingerprint(base))
    monkeypatch.setattr(prepare, "propose_combinations", lambda *args: suggestion)
    with pytest.raises(RuntimeError, match="duplicates an existing effective"):
        prepare.main()
    assert not isolated["combinations"].exists()


def test_preflight_requires_completed_base_and_exact_manifest_provenance(isolated):
    _qualify(isolated, ("L", "P"))
    prepare.main()
    manifest = _manifest(isolated)
    assert len(preflight.validate_frozen_inputs(manifest)) == 1
    common.save_json(isolated["exp"] / "queue_status.json", {"status": "running"})
    with pytest.raises(RuntimeError, match="completed frozen base"):
        preflight.validate_frozen_inputs(manifest)


def _mock_gpu_preflight(fixture, monkeypatch):
    anchor_run = fixture["root"] / "anchor"
    anchor_run.mkdir()
    common.save_json(anchor_run / "feature_stats.json", {})
    common.save_json(anchor_run / "weight_quantiles.json", {"raw_p90": 10.6})
    monkeypatch.setattr(preflight, "ANCHOR_RUN", anchor_run)
    monkeypatch.setenv("SLURM_JOB_ID", "mock-preflight")
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "empty_cache", lambda: None)
    monkeypatch.setattr(preflight, "seed_all", lambda seed: None)
    monkeypatch.setattr(preflight.gc, "collect", lambda: None)
    monkeypatch.setattr(preflight.D, "load_wss_stats", lambda path: {})
    monkeypatch.setattr(preflight.P, "_load_cases", lambda *args: [{"unit_id": f"c{i}"} for i in range(8)])
    monkeypatch.setattr(preflight.P, "_input_fingerprints", lambda *args: {"data": "frozen"})
    monkeypatch.setattr(preflight.D, "query_rows", lambda case: range(17000))

    class DummyDataset:
        def __init__(self, *args, **kwargs):
            pass

        def __getitem__(self, index):
            return {"support_pos": range(5000), "pos": range(5000)}

    class DummyModel:
        def state_dict(self):
            return {"shared": torch.ones(1)}

        def parameters(self):
            return [torch.nn.Parameter(torch.ones(1))]

        def cuda(self):
            return self

    monkeypatch.setattr(preflight.D, "WSSMinDataset", DummyDataset)
    monkeypatch.setattr(preflight.D, "collate", lambda samples: samples)
    monkeypatch.setattr(preflight, "build_model", lambda *args: DummyModel())
    monkeypatch.setattr(preflight, "build_paired_model", lambda cfg: (
        DummyModel(), {"new_keys": [], "changed_shape_keys": []}))
    monkeypatch.setattr(preflight.P, "_initialization_check", lambda *args: {})
    memory = dict(passed=True, max_memory_reserved_mib=3000., max_memory_allocated_mib=2000.)
    monkeypatch.setattr(preflight.P, "_train_check", lambda *args: memory)
    monkeypatch.setattr(preflight.P, "_evaluation_check", lambda *args: memory)
    monkeypatch.setattr(preflight, "execution_tool_hashes", lambda: {
        "prepare_m2_combinations.py": "prepare", "preflight_m2_combinations.py": "preflight"})


def test_runtime_preflight_output_is_accepted_by_real_queue_initializer(isolated, monkeypatch):
    _qualify(isolated, ("L", "P"))
    prepare.main()
    _mock_gpu_preflight(isolated, monkeypatch)
    assert preflight.main() == 0
    gate = json.loads((isolated["exp"] / "combination_runtime_preflight.json").read_text())
    assert gate["passed"] and set(gate["arms"]) == {"MO-CLP"}
    assert gate["source_sha256"] == gate["source_sha256_end"] == isolated["hashes"]
    monkeypatch.setattr(common, "EXP", isolated["exp"])
    for name, value in (("EXP", isolated["exp"]), ("CONFIGS", isolated["configs"]),
                        ("ROOT", isolated["root"]), ("RUNS", isolated["root"] / "runs")):
        monkeypatch.setattr(queue, name, value)
    monkeypatch.setattr(queue, "manifest_path", lambda phase: isolated["combinations"])
    monkeypatch.setattr(queue, "fingerprints", lambda phase: isolated["hashes"])
    queue.initialize("combination")
    state = json.loads(common.state_path("combination").read_text())
    assert state["status"] == "running" and state["arms"]["MO-CLP"]["status"] == "pending"
    assert state["arms"]["MO-CLP"]["min_free_mib"] == 7000.


@pytest.mark.parametrize("failure", ["wrapper_change", "setup_exception", "arm_exception"])
def test_preflight_failure_always_leaves_an_explicit_nonpassing_gate(isolated, monkeypatch, failure):
    _qualify(isolated, ("L", "P"))
    prepare.main()
    _mock_gpu_preflight(isolated, monkeypatch)
    output = isolated["exp"] / "combination_runtime_preflight.json"
    common.save_json(output, {"passed": True, "old_success": True})
    if failure == "wrapper_change":
        calls = [0]

        def tool_hashes():
            calls[0] += 1
            return {"preflight_m2_combinations.py": str(calls[0]), "prepare_m2_combinations.py": "same"}

        monkeypatch.setattr(preflight, "execution_tool_hashes", tool_hashes)
    else:
        def fail(*args, **kwargs):
            raise RuntimeError("simulated failure")

        target = "_load_cases" if failure == "setup_exception" else "_train_check"
        monkeypatch.setattr(preflight.P, target, fail)
    assert preflight.main() == 1
    assert json.loads(output.read_text())["passed"] is False

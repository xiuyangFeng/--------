"""Freeze only combinations selected by the predeclared complete-base gate."""
from __future__ import annotations
import copy
import json
from training_wss_min import config as C
from training_wss_min.tools.m2_optimization_common import (
    COMBINATIONS, CONFIGS, EXP, MATRIX, PREFIX, save_json, sha, stamp,
)
from training_wss_min.tools.prepare_v6_singleframe_matrix import set_path
from training_wss_min.tools.report_m2_optimization import (
    config_fingerprint, propose_combinations, report_all,
)


def effective_configuration_sha256(config):
    """Expand defaults for a second identity check; retain the report's raw hash."""
    expanded = json.loads(json.dumps(C.ExpConfig.from_dict(config).to_dict()))
    return config_fingerprint(expanded)


def validate_manifest(matrix):
    """Validate a frozen combination manifest without GPU work or file mutation."""
    arms = matrix.get("arms", [])
    if matrix.get("phase") != "combination" or not matrix.get("selection", {}).get("ready"):
        raise RuntimeError("Combination manifest must contain a ready base selection")
    if len(arms) > 4 or len({arm["id"] for arm in arms}) != len(arms):
        raise RuntimeError("At most four unique combination arms are authorised")
    if len({arm["run_name"] for arm in arms}) != len(arms):
        raise RuntimeError("Duplicate combination run names")
    identities = set()
    for arm in arms:
        config = arm["resolved_config"]
        if config_fingerprint(config) != arm["selected_configuration_sha256"]:
            raise RuntimeError(f"Selected raw configuration identity mismatch: {arm['id']}")
        effective = effective_configuration_sha256(config)
        if effective != arm["effective_configuration_sha256"] or effective in identities:
            raise RuntimeError(f"Duplicate or mismatched effective configuration: {arm['id']}")
        identities.add(effective)
        families = arm["families"]
        if (tuple(families) not in (("L", "P"), ("L", "S"), ("P", "S"), ("L", "P", "S"))
                or arm["id"] != "MO-C" + "".join(families)):
            raise RuntimeError("Unknown combination family/id")
        expected_members = [matrix["selection"]["families"][family]["winner"] for family in families
                            if matrix["selection"]["families"][family]["winner"] != "MO0"]
        if arm["members"] != expected_members or len(expected_members) < 2:
            raise RuntimeError("Combination members must be the qualified family winners")
        if arm["run_name"] != config["name"] or arm["config"] != f"{arm['id']}_s1234.json":
            raise RuntimeError("Combination configuration/run path mismatch")
    return arms


def main():
    report = report_all(matrix_path=MATRIX, combinations_path=COMBINATIONS,
                        experiment_dir=EXP, write=False)
    matrix = json.loads(MATRIX.read_text())
    # Later combination results must not change a decision based on base arms.
    base_ids = {arm["id"] for arm in matrix["arms"]}
    base_summaries = {checkpoint: {
        **report[checkpoint], "arms": {aid: entry for aid, entry in report[checkpoint]["arms"].items()
                                      if aid in base_ids},
    } for checkpoint in ("best", "last")}
    recommendation = propose_combinations(matrix["arms"], base_summaries)
    if not recommendation["ready"]:
        raise RuntimeError("All base experiments and both checkpoint results must be qualified first")
    if len(recommendation["arms"]) > 4:
        raise RuntimeError("At most four combinations are authorised")
    by_id = {a["id"]: a for a in matrix["arms"]}
    base = by_id["MO0"]["resolved_config"]
    arms = []
    seen = {effective_configuration_sha256(arm["resolved_config"]) for arm in matrix["arms"]}
    for suggested in recommendation["arms"]:
        config = copy.deepcopy(base)
        aid = suggested["id"]
        config["name"] = f"{PREFIX}/{aid}_s1234"
        config["notes"] = "预登记优胜组合：" + "+".join(suggested["members"]) + "；从零训练400轮；原M2数据；单seed探索。"
        for key, value in suggested["changes"].items():
            set_path(config, key, value)
        C.ExpConfig.from_dict(config)
        if config["data"] != base["data"] or config["eval"] != base["eval"]:
            raise RuntimeError("Combination changes frozen M2 data or evaluation")
        if config_fingerprint(config) != suggested["config_sha256"]:
            raise RuntimeError(f"Suggested raw configuration identity mismatch: {aid}")
        effective = effective_configuration_sha256(config)
        if effective in seen:
            raise RuntimeError(f"Combination duplicates an existing effective configuration: {aid}")
        seen.add(effective)
        arm = dict(id=aid, phase="combination", family="C", category="C", parent="MO0", baseline="MO0",
            members=suggested["members"], families=suggested["families"],
            config=f"{aid}_s1234.json", run_name=config["name"], resolved_config=config,
            changes=suggested["changes"], hypothesis=config["notes"],
            selected_configuration_sha256=suggested["config_sha256"],
            effective_configuration_sha256=effective)
        arms.append(arm)
    payload = dict(matrix_id=PREFIX, phase="combination", reference=matrix["reference"], protocol=matrix["protocol"],
        base_matrix_sha256=sha(MATRIX), base_queue_sha256=sha(EXP / "queue_status.json"),
        selection=recommendation["selection"], skipped=recommendation["skipped"], arms=arms)
    validate_manifest(payload)
    if (EXP / "combination_decision.json").exists() and not COMBINATIONS.exists():
        raise RuntimeError("Existing combination decision has no manifest; refusing partial replacement")
    # Validate the whole write set before writing any config. Existing frozen
    # manifests also require their config files to remain present and identical.
    for arm in arms:
        path = CONFIGS / arm["config"]
        if path.exists() and json.loads(path.read_text()) != arm["resolved_config"]:
            raise RuntimeError(f"Combination config conflict: {path}")
        if COMBINATIONS.exists() and not path.exists():
            raise RuntimeError(f"Frozen combination config is missing: {path}")
    if COMBINATIONS.exists():
        if json.loads(COMBINATIONS.read_text()) != payload:
            raise RuntimeError("Combination manifest already exists with different selection/provenance")
    else:
        for arm in arms:
            path = CONFIGS / arm["config"]
            if not path.exists():
                save_json(path, arm["resolved_config"])
        save_json(COMBINATIONS, payload)
    decision_path = EXP / "combination_decision.json"
    decision = dict(additional_training_runs=len(arms), manifest_sha256=sha(COMBINATIONS),
                    selection=recommendation["selection"], skipped=recommendation["skipped"])
    if decision_path.exists():
        previous = json.loads(decision_path.read_text())
        if {key: value for key, value in previous.items() if key != "frozen_at"} != decision:
            raise RuntimeError("Existing combination decision differs; refusing to overwrite")
    else:
        save_json(decision_path, dict(frozen_at=stamp(), **decision))
    print(f"Frozen {len(arms)} qualified combinations; {20 + len(arms)} total runs")


if __name__ == "__main__":
    main()

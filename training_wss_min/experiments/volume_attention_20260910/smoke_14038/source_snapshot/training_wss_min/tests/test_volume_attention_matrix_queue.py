"""Preregistration and execution-gate invariants, independent of GPU training."""
import copy
import json
from pathlib import Path
from unittest import mock

import pytest

from training_wss_min.tools import prepare_volume_attention_matrix as P
from training_wss_min.tools import run_volume_attention_queue as Q


def test_complete_matrix_preserves_protocol_and_two_factorials():
    payloads=P.build_payloads();matrix=payloads[P.MATRIX]
    assert len(matrix["arms"])==26
    assert {a["id"] for a in matrix["arms"]}=={f"{p}{i:02d}" for p in ("P","V") for i in range(12)}|{"J00","J01"}
    for arm in matrix["arms"]:
        c=payloads[P.CONFIGS/arm["config"]]
        assert c["train"]["epochs"]==400 and c["train"]["seed"]==1234
        assert c["train"]["batch_cases"]==8 and c["train"]["loss_pinball_lambda"]==0
        assert c["eval"]["stability_seeds"]==[1234]
        assert len(c["data"]["input_features"])==18
        assert c["data"]["query_n_points"]==5000 and c["data"]["support_n_points"]==5000
        assert c["data"]["feature_stats_path"]
        assert c["model"]["query_decoder"]=="qad_lite"
    for p in ("P","V"):
        cfg=lambda i:payloads[P.CONFIGS/f"{p}{i:02d}_s1234.json"]["model"]
        assert not cfg(0)["local_branch"] and not cfg(0)["bottleneck_transformer"]
        assert cfg(1)["local_branch"] and not cfg(1)["bottleneck_transformer"]
        assert not cfg(2)["local_branch"] and cfg(2)["bottleneck_transformer"]
        assert cfg(3)["local_branch"] and cfg(3)["bottleneck_transformer"]
        assert cfg(4)["bottleneck_layers"]==0 and cfg(5)["bottleneck_mode"]=="token_ffn"
        for i in range(6,12):assert cfg(i)["sa_nsample"][-1]==32
        assert cfg(9)["multiradius_values"]==[.1,.2,.4]
        assert cfg(10)["multiradius_values"]==[.2,.2,.2]
        assert cfg(11)["multiradius_ffn"] and not cfg(11)["multiradius_bottleneck"]


def reference_payload():
    base={"field":{"n":100,"r2":.7,"mae":1.},"field_casebalanced":{"r2":.7},
          "aggregate":{"n_cases":1},"per_case":{"case1":{"overall":{"r2":.7,"n":100}}}}
    return {"test":{**base,"normalized":copy.deepcopy(base),
                     "query_groups":{"wall":copy.deepcopy(base),"interior":copy.deepcopy(base)}}}


def write_reference(tmp_path,changed):
    for ck in ("best","last"):
        for folder,payload in (("original_eval",reference_payload()),("eval",changed)):
            p=tmp_path/folder/f"ckpt_{ck}"/"metrics.json";p.parent.mkdir(parents=True,exist_ok=True)
            p.write_text(json.dumps(payload))


def test_reference_reproduction_rejects_missing_or_wrong_case_group_fields(tmp_path):
    correct=reference_payload();write_reference(tmp_path,correct)
    assert Q.compare_reference(tmp_path)["passed"]
    for change in ("field","per_case","query_groups","normalized"):
        wrong=copy.deepcopy(correct);del wrong["test"][change]
        write_reference(tmp_path,wrong)
        assert not Q.compare_reference(tmp_path)["passed"],change


def test_reference_matching_nan_is_serializable(tmp_path):
    correct=reference_payload();correct["test"]["field"]["r2"]=float("nan")
    write_reference(tmp_path,correct)
    for ck in ("best","last"):
        (tmp_path/"original_eval"/f"ckpt_{ck}"/"metrics.json").write_text(json.dumps(correct))
    result=Q.compare_reference(tmp_path)
    assert result["passed"]
    json.dumps(result,allow_nan=False)


def test_frozen_statistics_change_blocks_gate(tmp_path):
    stats=tmp_path/"stats.json";stats.write_text('{"mean":1}')
    (tmp_path/"runtime_preflight.json").write_text(json.dumps({"data":{str(stats):{"sha256":Q.sha256(stats)}}}))
    with mock.patch.object(Q,"EXP",tmp_path):
        assert Q.verify_data_inputs()["passed"]
        stats.write_text('{"mean":2}')
        with pytest.raises(RuntimeError,match="statistics changed"):
            Q.verify_data_inputs()

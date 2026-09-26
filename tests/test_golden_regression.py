"""Golden regression gate: re-run stage B for reference jobs and require identical numbers.

Opt-in because it needs the frozen release, the conda environment with torch and (normally) a GPU:

    WSS_DEPLOY_GOLDEN_REFERENCE=<dir with <id>/summary.json + field.npz produced by a trusted code version> \
    WSS_DEPLOY_GOLDEN_JOBS_ROOT=outputs/wss_deploy_jobs [WSS_DEPLOY_GOLDEN_DEVICE=cuda] \
    [WSS_DEPLOY_GOLDEN_PRECOMPUTE=1] PYTHONPATH=. python -m pytest -q tests/test_golden_regression.py

``WSS_DEPLOY_GOLDEN_PRECOMPUTE=1`` (v0.14) fills the geometry cache first, so stage B runs on cache hits.
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

REFERENCE = os.environ.get("WSS_DEPLOY_GOLDEN_REFERENCE")
pytestmark = pytest.mark.skipif(not REFERENCE, reason="set WSS_DEPLOY_GOLDEN_REFERENCE to run the golden regression")


def _reference_jobs():
    if not REFERENCE:
        return []
    root = Path(REFERENCE)
    return sorted(p.parent.name for p in root.glob("*/summary.json") if (p.parent / "field.npz").is_file())


@pytest.mark.parametrize("job_id", _reference_jobs())
def test_stage_b_reproduces_reference(job_id, tmp_path):
    from wss_deploy.regress import run_job
    jobs_root = Path(os.environ.get("WSS_DEPLOY_GOLDEN_JOBS_ROOT", "outputs/wss_deploy_jobs"))
    device = os.environ.get("WSS_DEPLOY_GOLDEN_DEVICE", "cuda")
    atol = float(os.environ.get("WSS_DEPLOY_GOLDEN_ATOL", "1e-5"))
    result = run_job(jobs_root / job_id, tmp_path / job_id, device=device, atol=atol, release_root=None,
                     reference=Path(REFERENCE) / job_id,
                     precompute=os.environ.get("WSS_DEPLOY_GOLDEN_PRECOMPUTE") == "1")
    assert result["passed"], json.dumps({k: result[k] for k in ("summary_diffs", "array_failures")}, ensure_ascii=False)[:4000]

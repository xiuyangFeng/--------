"""Evidence gate shared by the GPU preflight and corrected matrix queue."""
from __future__ import annotations
import hashlib
import json
from pathlib import Path
from training_wss_min import config as C

ROOT = C.PROJECT_ROOT
NAME = "wss_direct_recovery_20260912"
CONFIGS = ROOT / "training_wss_min/configs" / NAME
EXP = ROOT / "training_wss_min/experiments" / NAME
RUNS = ROOT / "training_wss_min/runs"


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def save_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")
    tmp.replace(path)


def fingerprints():
    # Freeze every directly importable core module and every candidate config.
    paths = list((ROOT / "training_wss_min").glob("*.py")) + list(CONFIGS.glob("*.json"))
    paths += [EXP / "inlet_audit" / f for f in
              ("case_features.json", "feature_stats_26d.json", "audit.json", "feature_stats_provenance.json")]
    for cfg_path in CONFIGS.glob("*s1234.json"):
        cfg = json.loads(cfg_path.read_text())
        paths += [Path(cfg["data"][k]) for k in ("feature_stats_path", "split_path", "wss_stats_path")]
        paths += [Path(cfg["train"]["init_reference_config"])]
    return {str(p.resolve()): sha256(p) for p in sorted(set(paths))}


def data_fingerprints():
    from training_wss_min import dataset as D
    from training_wss_min.next_geometry import source_h5_for_bundle
    cfg = C.ExpConfig.from_json(CONFIGS / "E0_s1234.json")
    out = {}
    for part in ("train", "test"):
        for cohort, case in D.load_split_cases(cfg.data.split_path, part):
            bundle = Path(cfg.data.data_root) / cohort / case / "bundle.npz"
            sidecar = Path(cfg.data.point_features_root) / cohort / case / "features.npz"
            for path in (bundle, sidecar):
                st = path.stat()
                out[str(path.resolve())] = dict(sha256=sha256(path), bytes=st.st_size, mtime_ns=st.st_mtime_ns)
            source = source_h5_for_bundle(bundle)
            st = source.stat()
            # Relevant inlet/atlas arrays have their own hashes in inlet_audit;
            # avoid hashing multi-GB unrelated CFD solution containers again.
            out[str(source.resolve())] = dict(bytes=st.st_size, mtime_ns=st.st_mtime_ns,
                scope="frozen geometry source; array hashes and identity gates in inlet_audit")
    return out


def verify_preflight():
    path = EXP / "runtime_preflight.json"
    evidence = json.loads(path.read_text())
    if evidence.get("passed") is not True or not evidence.get("slurm_job_id"):
        raise RuntimeError("a successful real Slurm GPU preflight is required")
    if fingerprints() != evidence["fingerprints"]:
        raise RuntimeError("source/config/protocol changed since GPU preflight")
    if data_fingerprints() != evidence["data_fingerprints"]:
        raise RuntimeError("frozen data changed since GPU preflight")
    expected = {p.name for p in CONFIGS.glob("*s1234.json")}
    if set(evidence["arms"]) != expected:
        raise RuntimeError("GPU preflight did not test every arm and C4 candidate")
    if not evidence.get("cli_smoke_passed"):
        raise RuntimeError("train and full-wall evaluation CLI smoke required")
    execution = json.loads((EXP / "execution_gate.json").read_text())
    if execution.get("passed") is not True or execution["files"] != execution_fingerprints():
        raise RuntimeError("execution/report/verification tools changed since final audit")
    real = json.loads((EXP / "real_prediction_verification.json").read_text())
    if real.get("passed") is not True:
        raise RuntimeError("real GPU smoke predictions have not passed independent recomputation")
    return {"preflight": str(path), "sha256": sha256(path), "job_id": evidence["slurm_job_id"]}


def execution_fingerprints():
    names = ("prepare_wss_recovery", "preflight_wss_recovery", "wss_recovery_common",
             "run_next_matrix_queue", "run_wss_recovery_queue", "verify_wss_recovery", "report_wss_recovery")
    paths = [ROOT / "training_wss_min/tools" / f"{name}.py" for name in names]
    paths += [ROOT / "training_wss_min/cluster" / name for name in
              ("preflight_wss_recovery.slurm", "run_wss_recovery.slurm")]
    return {str(p): sha256(p) for p in paths}

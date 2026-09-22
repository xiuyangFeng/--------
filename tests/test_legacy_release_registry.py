"""Old run bindings must come from recorded identity, never the default."""
import hashlib
import json

import pytest

from wss_deploy import registry as registry_module
from wss_deploy.registry import ReleaseError, ReleaseRegistry


def _hash(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _package(root, name):
    path = root / name
    run = path / "models" / "seed1"
    run.mkdir(parents=True)
    (path / "rules").mkdir()
    (path / "release.json").write_text(json.dumps({
        "release": name, "target": "wss",
        "time_axis": [{"index": 0, "step": 1162, "time_s": 0.21}],
        "models": [{"seed": 1, "path": "models/seed1"}],
    }), encoding="utf-8")
    (run / "ckpt_best.pt").write_bytes(b"test checkpoint; must never be loaded")
    (run / "config.json").write_text(json.dumps({"data": {"target": "wss", "timesteps": "peak"}}))
    for name in ("feature_stats.json", "wss_global_stats.json", "target_normalization.json"):
        (run / name).write_text("{}")
    (path / "rules" / "flow_split_rule_train136.json").write_text("{}")
    files = sorted(p for p in path.rglob("*") if p.is_file())
    (path / "MANIFEST.sha256").write_text("".join(
        f"{_hash(p)}  {p.relative_to(path).as_posix()}  {p.stat().st_size}\n" for p in files))
    return path


def test_restores_explicit_package_when_default_differs_without_loading(tmp_path, monkeypatch):
    old = _package(tmp_path, "old-release")
    _package(tmp_path, "new-default")
    loader_calls = []
    registry = ReleaseRegistry(tmp_path, default_id="new-default",
                               loader=lambda *args, **kwargs: loader_calls.append(args))
    audited = []
    original = registry_module._verify_package

    def audit(record):
        audited.append(record.id)
        original(record)

    monkeypatch.setattr(registry_module, "_verify_package", audit)
    before = {p.relative_to(old): p.read_bytes() for p in old.rglob("*") if p.is_file()}
    record = registry.restore_legacy_binding("old-release", manifest_sha256=_hash(old / "MANIFEST.sha256"))
    assert record.id == "old-release"
    assert record.fingerprint == registry.describe("old-release").fingerprint
    assert not record.is_default
    assert audited == ["old-release"] and loader_calls == [] and registry._cache == {}
    assert before == {p.relative_to(old): p.read_bytes() for p in old.rglob("*") if p.is_file()}


@pytest.mark.parametrize("release_id", [None, "", "../old-release", " old-release ", 123])
def test_missing_or_invalid_id_never_uses_default(tmp_path, release_id):
    path = _package(tmp_path, "old-release")
    registry = ReleaseRegistry(tmp_path)
    with pytest.raises(ReleaseError):
        registry.restore_legacy_binding(release_id, manifest_sha256=_hash(path / "MANIFEST.sha256"))


@pytest.mark.parametrize("digest", [None, "", "a" * 63, "g" * 64, "a" * 64 + "\n", 123])
def test_missing_or_malformed_manifest_hash_is_rejected(tmp_path, digest):
    _package(tmp_path, "old-release")
    registry = ReleaseRegistry(tmp_path)
    with pytest.raises(ReleaseError, match="SHA256"):
        registry.restore_legacy_binding("old-release", manifest_sha256=digest)


def test_optional_metadata_hash_is_checked_and_uppercase_hashes_are_valid(tmp_path):
    path = _package(tmp_path, "old-release")
    registry = ReleaseRegistry(tmp_path)
    manifest = _hash(path / "MANIFEST.sha256")
    metadata = _hash(path / "release.json")
    record = registry.restore_legacy_binding("old-release", manifest_sha256=manifest.upper(),
                                             release_json_sha256=metadata.upper())
    assert record.id == "old-release"
    for invalid in ("", "x" * 64, "0" * 64):
        with pytest.raises(ReleaseError):
            registry.restore_legacy_binding("old-release", manifest_sha256=manifest,
                                            release_json_sha256=invalid)


def test_wrong_manifest_and_unknown_release_are_rejected(tmp_path):
    path = _package(tmp_path, "old-release")
    registry = ReleaseRegistry(tmp_path)
    with pytest.raises(ReleaseError, match="不一致"):
        registry.restore_legacy_binding("old-release", manifest_sha256="0" * 64)
    with pytest.raises(ReleaseError, match="不存在"):
        registry.restore_legacy_binding("unknown", manifest_sha256=_hash(path / "MANIFEST.sha256"))


@pytest.mark.parametrize("relative_path", ["models/seed1/ckpt_best.pt", "models/seed1/feature_stats.json",
                                            "rules/flow_split_rule_train136.json"])
def test_matching_manifest_cannot_hide_tampered_package_file(tmp_path, relative_path):
    path = _package(tmp_path, "old-release")
    registry = ReleaseRegistry(tmp_path)
    manifest = _hash(path / "MANIFEST.sha256")
    (path / relative_path).write_bytes(b"tampered")
    with pytest.raises(ReleaseError, match="校验失败"):
        registry.restore_legacy_binding("old-release", manifest_sha256=manifest)


def test_rewritten_manifest_and_missing_manifest_are_rejected(tmp_path):
    path = _package(tmp_path, "old-release")
    manifest = _hash(path / "MANIFEST.sha256")
    (path / "MANIFEST.sha256").write_text((path / "MANIFEST.sha256").read_text() + "\n")
    registry = ReleaseRegistry(tmp_path)
    with pytest.raises(ReleaseError, match="不一致"):
        registry.restore_legacy_binding("old-release", manifest_sha256=manifest)
    (path / "MANIFEST.sha256").unlink()
    registry = ReleaseRegistry(tmp_path)
    with pytest.raises(ReleaseError):
        registry.restore_legacy_binding("old-release", manifest_sha256=manifest)

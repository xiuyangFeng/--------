"""Release discovery and lazy, fail-closed model loading for deployment.

The registry deliberately knows only about release metadata.  Checkpoints are
loaded by :meth:`ReleaseRegistry.load` when a worker is about to run a job;
uploading a file therefore never allocates GPU memory or imports torch models.
"""
from __future__ import annotations

from collections import OrderedDict
import time
from dataclasses import dataclass
import hashlib
import json
import logging
import os
from pathlib import Path
import re
import threading
from typing import Any, Callable

from .families import (FAMILIES, ReleaseError, VOLUME_FEATURES, VOLUME_FIELDS,  # noqa: F401  (re-exported)
                       _validate_volume_model, family_for_info, family_for_protocol)
from .paths import RELEASE_DIR, RETIRED_RELEASE_ROOT

LOG = logging.getLogger("wss_deploy.registry")
RELEASE_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,127}$")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _release_fingerprint(release_file: Path, manifest: Path | None) -> str:
    """Hash metadata and weights so editing a contract changes identity."""
    digest = hashlib.sha256()
    for path in (release_file, manifest):
        if path is None or not path.is_file():
            continue
        digest.update(path.name.encode("utf-8")); digest.update(b"\0")
        with path.open("rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(chunk)
        digest.update(b"\0")
    return digest.hexdigest()


def _safe_id(value: str) -> str:
    value = str(value or "")
    if not RELEASE_ID_RE.fullmatch(value):
        raise ReleaseError("无效的发布包标识。")
    return value


def _is_within(path: Path, root: Path) -> bool:
    """Return whether ``path`` is below ``root`` on supported Python versions."""
    try:
        path.relative_to(root)
        return True
    except ValueError:
        return False


def _verify_package(record) -> None:
    """Verify every manifest entry and every file consumed by inference."""
    root = record.path
    manifest = root / "MANIFEST.sha256"
    if _release_fingerprint(root / "release.json", manifest) != record.fingerprint:
        raise ReleaseError("发布包身份已变化；请使用新的 release 标识。")
    if not manifest.is_file():
        raise ReleaseError("发布包缺少 MANIFEST.sha256，无法验证推理文件。")
    registered = set()
    for line in manifest.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        parts = line.split()
        if len(parts) not in (2, 3) or not re.fullmatch(r"[a-fA-F0-9]{64}", parts[0]):
            raise ReleaseError("MANIFEST.sha256 格式无效。")
        rel = Path(parts[1].lstrip("*"))
        path = (root / rel).resolve()
        if rel.is_absolute() or ".." in rel.parts or not _is_within(path, root):
            raise ReleaseError("发布包文件路径超出发布目录。")
        if rel.as_posix() in registered or not path.is_file() or _sha256(path) != parts[0].lower():
            raise ReleaseError(f"发布包文件校验失败：{rel}")
        if len(parts) == 3 and (not parts[2].isdigit() or int(parts[2]) != path.stat().st_size):
            raise ReleaseError(f"发布包文件大小校验失败：{rel}")
        registered.add(rel.as_posix())
    family = family_for_protocol(record.contract["protocol"])
    try:
        paths = family.model_paths(record.info)
    except ValueError as exc:
        raise ReleaseError(str(exc)) from exc
    declared = record.info.get("models") if isinstance(record.info.get("models"), list) else []
    for model in paths:
        rel = Path(str(model))
        if rel.is_absolute() or ".." in rel.parts or not _is_within((root / rel).resolve(), root):
            raise ReleaseError("模型路径超出发布目录。")
        for name in ("ckpt_best.pt", "config.json", "feature_stats.json", "wss_global_stats.json", "target_normalization.json"):
            if (rel / name).as_posix() not in registered:
                raise ReleaseError(f"推理文件未登记：{rel / name}")
        cfg = json.loads((root / rel / "config.json").read_text(encoding="utf-8"))
        spec = next((m for m in declared if isinstance(m, dict) and m.get("path") == str(model)), None)
        family.verify_model(root / rel, cfg, spec)
    if "rules/flow_split_rule_train136.json" not in registered:
        raise ReleaseError("发布包缺少已验证的分流规则。")
    for directory in (root / "models", root / "rules"):
        for path in directory.rglob("*"):
            if path.is_file() and path.suffix in {".json", ".pt"} and path.relative_to(root).as_posix() not in registered:
                raise ReleaseError(f"推理目录存在未登记文件：{path.relative_to(root)}")


# v0.14: a full package verification hashes every registered file.  It is repeated only when a file's
# stat signature (size, mtime, ctime, inode) changes, a registered/unregistered file appears or
# disappears, or the last full verification is older than WSS_DEPLOY_RELEASE_VERIFY_TTL seconds
# (default 600).  Any failure is never cached, so a changed file still fails closed.
_VERIFIED: dict[str, tuple[Any, float]] = {}
_VERIFIED_LOCK = threading.Lock()


def _verify_ttl_seconds() -> float:
    try:
        return max(0.0, float(os.environ.get("WSS_DEPLOY_RELEASE_VERIFY_TTL", "600")))
    except ValueError:
        return 600.0


def _package_signature(record) -> tuple:
    """Cheap identity of everything :func:`_verify_package` reads: stat of each file, plus the file listing."""
    root = Path(record.path)

    def stat(path: Path) -> tuple:
        info = path.stat()
        return (info.st_size, info.st_mtime_ns, info.st_ctime_ns, info.st_ino)

    entries = [("release.json", stat(root / "release.json"))]
    manifest = root / "MANIFEST.sha256"
    if manifest.is_file():
        entries.append(("MANIFEST.sha256", stat(manifest)))
        for line in manifest.read_text(encoding="utf-8").splitlines():
            parts = line.split()
            if len(parts) >= 2:
                target = root / parts[1].lstrip("*")
                entries.append((parts[1], stat(target) if target.is_file() else None))
    for directory in (root / "models", root / "rules"):
        if directory.is_dir():
            entries.extend((p.relative_to(root).as_posix(), stat(p)) for p in sorted(directory.rglob("*"))
                           if p.is_file() and p.suffix in {".json", ".pt"})
    return (record.fingerprint, tuple(entries))


def _verify_package_cached(record) -> None:
    """:func:`_verify_package` unless the same package state was fully verified within the TTL."""
    key = str(Path(record.path).resolve())
    try:
        signature = _package_signature(record)
    except OSError:
        signature = None
    now = time.monotonic()
    with _VERIFIED_LOCK:
        previous = _VERIFIED.get(key)
    if signature is not None and previous is not None and previous[0] == signature \
            and now - previous[1] < _verify_ttl_seconds():
        return
    try:
        _verify_package(record)
    except Exception:
        with _VERIFIED_LOCK:
            _VERIFIED.pop(key, None)
        raise
    try:
        after = _package_signature(record)
    except OSError:
        after = None
    with _VERIFIED_LOCK:
        if signature is not None and after == signature:
            _VERIFIED[key] = (signature, now)
        else:
            _VERIFIED.pop(key, None)


def _contract(info: dict[str, Any]) -> dict[str, Any]:
    """Return and validate the intentionally narrow deployment contract of the release's model family."""
    return family_for_info(info).contract(info)


def _volume_contract(info: dict[str, Any]) -> dict[str, Any]:
    """Kept for callers that validated volume packages directly; identical to the family contract."""
    from .families import PF6_VF6_VOLUME
    return PF6_VF6_VOLUME.contract(info)


@dataclass(frozen=True)
class ReleaseDescriptor:
    id: str
    path: Path
    fingerprint: str
    info: dict[str, Any]
    contract: dict[str, Any]
    is_default: bool = False

    @property
    def name(self) -> str:
        return self.id

    def public(self) -> dict[str, Any]:
        info = self.info
        models = info.get("models") or info.get("model_runs")
        if models is None:
            models = info.get("source_runs")
        if models is None and (self.path / "models").is_dir():
            models = [p for p in (self.path / "models").iterdir() if p.is_dir()]
        n_models = len(models) if isinstance(models, (list, dict)) else None
        value = {"id": self.id, "release": self.id, "fingerprint": self.fingerprint,
                 "default": self.is_default, "contract": self.contract,
                 "frozen_on": info.get("frozen_on"), "git_commit": info.get("git_commit")}
        if n_models is not None:
            value["models_count"] = (len(self.contract["seeds"])
                                     if self.contract["protocol"] == "single_frame_volume" else n_models)
            value["weights_count"] = n_models
        return {k: v for k, v in value.items() if v is not None}


class ReleaseRegistry:
    """Discover release directories below a fixed root and load them lazily."""

    def __init__(self, root: Path | None = None, *, default_id: str | None = None,
                 loader: Callable[..., Any] | None = None, device: str = "auto"):
        configured = Path(root or os.environ.get("WSS_DEPLOY_RELEASE_ROOT", RELEASE_DIR.parent))
        self.root = configured.resolve()
        self.default_id = default_id
        self.loader = loader
        self.device = device
        # Runtime settings are part of the model identity.  Reusing a full
        # CUDA ensemble for a CPU or reduced-seed job silently violates the
        # manifest, so cache by device and selected ensemble size as well.
        # The cache is a small LRU: every entry is a resident ensemble
        # (GPU memory), so it must not grow with every device/seed variant.
        self._cache: OrderedDict[tuple[str, str, int | None], Any] = OrderedDict()
        self._lock = threading.RLock()
        self._records = self._discover()
        # v0.12.2: room for every discovered release (an ensemble is a few MB of weights), so switching
        # between releases never reloads; WSS_DEPLOY_MODEL_CACHE still overrides.
        default_size = min(6, max(2, len(self._records)))
        self.cache_size = max(1, int(os.environ.get("WSS_DEPLOY_MODEL_CACHE", str(default_size))))

    def _discover(self) -> dict[str, ReleaseDescriptor]:
        root = self.root
        candidates = [root] if (root / "release.json").is_file() else sorted(root.iterdir()) if root.is_dir() else []
        records: dict[str, ReleaseDescriptor] = {}
        for path in candidates:
            try:
                resolved = path.resolve()
                if resolved.parent != root and resolved != root:
                    continue
                release_file = resolved / "release.json"
                if not release_file.is_file():
                    continue
                rid = _safe_id(json.loads(release_file.read_text(encoding="utf-8")).get("release") or resolved.name)
                info = json.loads(release_file.read_text(encoding="utf-8"))
                if rid in records:
                    # An ambiguous id must not select whichever directory was
                    # returned first by the filesystem.
                    records.pop(rid, None)
                    raise ReleaseError(f"重复的发布包标识：{rid}")
                manifest = resolved / "MANIFEST.sha256"
                fingerprint = _release_fingerprint(release_file, manifest if manifest.is_file() else None)
                records[rid] = ReleaseDescriptor(rid, resolved, fingerprint, info, _contract(info))
            except (OSError, ValueError, json.JSONDecodeError, ReleaseError) as exc:
                # A malformed sibling must never become a selectable package;
                # keep it out of the public listing and let explicit selection
                # fail as "not found" rather than executing an unsafe package.
                continue
        if not records:
            raise ReleaseError(f"发布根目录没有可用 release.json：{root}")
        selected = self.default_id
        if selected is None and (RELEASE_DIR / "release.json").is_file():
            selected = RELEASE_DIR.name
        if selected is None or selected not in records:
            selected = sorted(records)[0]
        self.default_id = selected
        return {rid: ReleaseDescriptor(r.id, r.path, r.fingerprint, r.info, r.contract, rid == selected)
                for rid, r in records.items()}

    @property
    def default(self) -> ReleaseDescriptor:
        return self._records[self.default_id]

    def list(self) -> list[dict[str, Any]]:
        return [self._records[rid].public() for rid in sorted(self._records)]

    def describe(self, release_id: str | None = None) -> ReleaseDescriptor:
        rid = self.default_id if release_id in (None, "") else _safe_id(release_id)
        try:
            record = self._records[rid]
        except KeyError:
            if is_retired(rid):
                raise ReleaseError(f"发布包已下线：{rid}。请用现役发布包重跑（复用中心线与已确认出口）。")
            raise ReleaseError("发布包不存在或未被支持。")
        if _release_fingerprint(record.path / "release.json", record.path / "MANIFEST.sha256") != record.fingerprint:
            raise ReleaseError("发布包身份已变化；请使用新的 release 标识。")
        return record

    def restore_legacy_binding(self, release_id: str, *, manifest_sha256: str,
                               release_json_sha256: str | None = None) -> ReleaseDescriptor:
        """Recover an old run's explicit release using its recorded hashes.

        Older summaries stored the SHA256 of the manifest bytes, rather than
        today's combined release fingerprint.  This read-only migration checks
        that historical evidence and the complete package before returning its
        current descriptor.  It never selects a default or loads model weights.
        """
        if not isinstance(release_id, str) or not release_id:
            raise ReleaseError("恢复历史任务必须提供明确的发布包标识。")
        rid = _safe_id(release_id)

        def valid_hash(value, label):
            if not isinstance(value, str) or not re.fullmatch(r"[0-9a-fA-F]{64}", value):
                raise ReleaseError(f"历史任务的 {label} 必须是有效的 SHA256。")
            return value.lower()

        expected_manifest = valid_hash(manifest_sha256, "MANIFEST.sha256 哈希")
        expected_metadata = (valid_hash(release_json_sha256, "release.json 哈希")
                             if release_json_sha256 is not None else None)
        with self._lock:
            try:
                record = self.describe(rid)

                def check_historical_hashes():
                    if _sha256(record.path / "MANIFEST.sha256") != expected_manifest:
                        raise ReleaseError("历史任务的 MANIFEST.sha256 哈希与当前发布包不一致。")
                    if expected_metadata is not None and _sha256(record.path / "release.json") != expected_metadata:
                        raise ReleaseError("历史任务的 release.json 哈希与当前发布包不一致。")

                check_historical_hashes()
                _verify_package(record)
                # Detect metadata changes during the full package audit too.
                check_historical_hashes()
                self.describe(rid)
                return record
            except ReleaseError:
                raise
            except (OSError, ValueError, TypeError) as exc:
                raise ReleaseError("历史任务发布包校验失败，不能恢复绑定。") from exc

    def load(self, release_id: str | None = None, *, device: str | None = None,
             seed_count: int | None = None) -> Any:
        record = self.describe(release_id)
        actual_device = device or self.device
        if actual_device not in {"auto", "cpu", "cuda"}:
            raise ReleaseError("不支持的计算设备。")
        if seed_count is not None and (type(seed_count) is not int or seed_count < 1):
            raise ReleaseError("集成模型数必须是正整数。")
        key = (record.id, actual_device, seed_count)
        with self._lock:
            _verify_package_cached(record)
            if key in self._cache:
                self._cache.move_to_end(key)
                return self._cache[key]
            loader = self.loader
            if loader is None:
                from .infer import Release
                loader = Release
            try:
                obj = loader(record.path, device=actual_device, seed_count=seed_count)
            except TypeError as exc:
                # Keep the small loader injection contract used by tests and
                # downstream tools while production Release accepts the
                # subset explicitly.
                if "seed_count" not in str(exc):
                    raise
                obj = loader(record.path, device=actual_device)
            _verify_package_cached(record)
            # Bind identity to the loaded object so summaries and manifests
            # cannot accidentally report another release.
            if not getattr(obj, "name", None):
                try: obj.name = record.id
                except Exception: pass
            try:
                obj.registry_id, obj.registry_fingerprint, obj.registry_contract = record.id, record.fingerprint, record.contract
            except Exception:
                pass
            self._cache[key] = obj
            while len(self._cache) > self.cache_size:
                evicted_key, evicted = self._cache.popitem(last=False)
                LOG.info("Evicting cached release %s (%s, seeds=%s) to bound resident models",
                         *evicted_key)
                _release_resident_models(evicted)
            return obj


def is_retired(release_id: str) -> bool:
    """Whether ``release_id`` is a retired package (a folder with ``release.json`` below ``RETIRED_RELEASE_ROOT``)."""
    try:
        rid = _safe_id(release_id)
        return (RETIRED_RELEASE_ROOT / rid / "release.json").is_file()
    except (ReleaseError, OSError):
        return False


def retired_release_ids() -> list[str]:
    """Ids of the retired packages (folders with ``release.json`` below ``RETIRED_RELEASE_ROOT``), sorted."""
    try:
        return sorted(p.name for p in RETIRED_RELEASE_ROOT.iterdir()
                      if RELEASE_ID_RE.fullmatch(p.name) and (p / "release.json").is_file())
    except OSError:
        return []


def retired_registry(*, device: str = "auto") -> "ReleaseRegistry | None":
    """Read-only registry of the retired packages (golden regression of results bound to them), or None."""
    root = RETIRED_RELEASE_ROOT
    if not root.is_dir() or not any((p / "release.json").is_file() for p in root.iterdir()):
        return None
    return ReleaseRegistry(root, device=device)


def preload_all(registry: "ReleaseRegistry", *, log=None) -> dict:
    """Load every release into the cache (service start, background thread); returns seconds per release.

    ``WSS_DEPLOY_PRELOAD=0`` disables it.  Failures are logged and skipped: a release that cannot load
    fails again, with its message, when a job asks for it.

    v0.14: each loaded ensemble then runs one throw-away prediction on a small synthetic case
    (``Release.warm_up``) so the first real job does not pay the CUDA / kernel start-up (1.2-2 s).
    ``WSS_DEPLOY_WARMUP=0`` skips it; a failing warm-up is logged and never blocks the service.

    Progress is published as ``registry.preload_state`` (read by ``JobManager.health()``):
    ``{"status": "running" | "done" | "failed" | "disabled", "running": bool, "loaded": [ids],
    "failed": {release_id: message}}``.  A failed load or warm-up is listed in ``failed``; the preload as
    a whole is ``failed`` only when releases were requested and none loaded.
    """
    log = log or LOG

    def publish(status: str, loaded, failed) -> None:
        try:
            registry.preload_state = {"status": status, "running": status == "running",
                                      "loaded": list(loaded), "failed": dict(failed)}
        except Exception:  # noqa: BLE001 — a registry stand-in without attribute support
            pass

    if os.environ.get("WSS_DEPLOY_PRELOAD", "1").strip().lower() in {"0", "false", "off", "no"}:
        publish("disabled", [], {})
        return {}
    warm = os.environ.get("WSS_DEPLOY_WARMUP", "1").strip().lower() not in {"0", "false", "off", "no"}
    timings, loaded, failed = {}, [], {}
    publish("running", loaded, failed)
    ids: list = []
    try:
        ids = sorted((record["id"] for record in registry.list()), key=lambda rid: rid != registry.default_id)[: registry.cache_size]
        for release_id in ids:
            start = time.perf_counter()
            try:
                release = registry.load(release_id)
                timings[release_id] = round(time.perf_counter() - start, 1)
                loaded.append(release_id)
            except Exception as exc:  # the job that needs it reports the real error
                log.warning("预加载发布包 %s 失败：%s", release_id, exc)
                failed[release_id] = str(exc)[:300]
                publish("running", loaded, failed)
                continue
            warm_up = getattr(release, "warm_up", None) if warm else None
            if callable(warm_up):
                try:
                    seconds = warm_up()
                    if seconds is not None:
                        log.info("发布包 %s 预热推理 %.1f s", release_id, seconds)
                except Exception as exc:  # noqa: BLE001 — warm-up is best effort
                    log.warning("发布包 %s 预热推理失败（不影响任务）：%s", release_id, exc)
                    failed[release_id] = f"预热推理失败：{str(exc)[:280]}"
            publish("running", loaded, failed)
    except Exception as exc:  # noqa: BLE001 — listing the releases failed; health reports it
        log.warning("预加载发布包失败：%s", exc)
        failed.setdefault("*", str(exc)[:300])
        publish("failed" if not loaded else "done", loaded, failed)
        return timings
    publish("failed" if ids and not loaded else "done", loaded, failed)
    if timings:
        log.info("已预加载发布包：%s", ", ".join(f"{k} {v}s" for k, v in timings.items()))
    return timings


def _release_resident_models(obj: Any) -> None:
    """Drop model objects of an evicted ensemble and return freed GPU memory to the driver."""
    try:
        models = getattr(obj, "models", None)
        if isinstance(models, list):
            models.clear()
    except Exception:
        pass
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass

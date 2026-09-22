"""Frozen WSS or PF6/VF6 peak models; ensembles average restored physical units."""
from __future__ import annotations
import json, time
from pathlib import Path
import numpy as np
import torch
from training_wss_min import dataset as D, evaluate as E  # noqa: F401  (D/E patched by tests; families use the same modules)
from .families import ModelFamily, family_for_info
from .paths import RELEASE_DIR, SEEDS
from .io_utils import file_sha256
from .registry import _verify_package, _release_fingerprint, ReleaseDescriptor


def load_reference_sidecar(release_dir: Path, release_id: str, info: dict) -> str | None:
    """Merge ``reference.json`` profiles into ``info``; returns the file hash or None when absent.

    Each profile must be bound to this release id; anything else fails closed so a sidecar copied
    from another package can never lend its population to the wrong weights.
    """
    sidecar = Path(release_dir) / "reference.json"
    if not sidecar.is_file():
        return None
    profiles = json.loads(sidecar.read_text(encoding="utf-8"))
    if not isinstance(profiles, dict):
        raise ValueError("reference.json 必须是 JSON 对象。")
    for key in ("geometry_reference", "population_reference"):
        block = profiles.get(key)
        if block is None:
            continue
        if not isinstance(block, dict) or block.get("release") != release_id:
            raise ValueError(f"reference.json 的 {key} 必须绑定当前发布包 {release_id}。")
        info[key] = block
    return file_sha256(sidecar)


class Release:
    def __init__(self, release_dir: Path = RELEASE_DIR, device: str = "auto", seeds=SEEDS,
                 seed_count: int | None = None):
        self.dir = Path(release_dir).resolve()
        self.device = ("cuda" if torch.cuda.is_available() else "cpu") if device == "auto" else device
        self.info = json.loads((self.dir / "release.json").read_text(encoding="utf-8"))
        # The family owns every model-kind specific rule (contract, checkpoint selection, prediction).
        self.family: ModelFamily = family_for_info(self.info)
        self.contract = self.family.contract(self.info)
        self.is_volume = self.contract["protocol"] == "single_frame_volume"
        self.release_id = str(self.info.get("release", self.dir.name))
        if not self.release_id:
            raise ValueError("release.json 必须包含非空 release 标识。")
        # CLI callers can instantiate Release without a registry; they must
        # receive the same complete config/statistics/weights validation.
        _verify_package(ReleaseDescriptor(self.release_id, self.dir,
            _release_fingerprint(self.dir / "release.json", self.dir / "MANIFEST.sha256"),
            self.info, self.contract))
        self.model_specs = self._model_specs(seeds, seed_count=seed_count)
        self.models = []
        self.weight_records = []
        manifest_records = self._manifest_records()
        t = time.perf_counter()
        for spec in self.model_specs:
            s, run = spec["seed"], self.dir / spec["path"]
            if not run.is_dir():
                raise FileNotFoundError(f"发布包模型目录不存在：{run}")
            checkpoint = run / "ckpt_best.pt"
            if not checkpoint.is_file():
                raise FileNotFoundError(f"发布包缺少 ckpt_best.pt：{checkpoint}")
            actual_hash = file_sha256(checkpoint)
            rel_checkpoint = checkpoint.relative_to(self.dir).as_posix()
            expected_hash = manifest_records.get(rel_checkpoint)
            if manifest_records and expected_hash is None:
                raise ValueError(f"权重校验失败：{rel_checkpoint} 未在 MANIFEST.sha256 中登记。")
            if expected_hash and expected_hash != actual_hash:
                raise ValueError(f"权重校验失败：{rel_checkpoint} 的 SHA256 与 MANIFEST.sha256 不一致。")
            self.weight_records.append({"seed": s, "path": rel_checkpoint,
                                        **({"field": spec["field"]} if "field" in spec else {}),
                                        "sha256": actual_hash,
                                        "size_bytes": int(checkpoint.stat().st_size),
                                        "verified": bool(expected_hash)})
            self.family.verify_model(run, json.loads((run / "config.json").read_text(encoding="utf-8")),
                                     spec if "field" in spec else None)
            # Loading reads only packaged config, feature statistics and weights;
            # training-data paths in the config remain provenance, never inputs.
            cfg, feat_stats, model, _ = E.load_model_from_run(run, self.device, "best"); model.eval()
            stats_file = run / "wss_global_stats.json"
            if not stats_file.is_file():
                raise ValueError("发布包必须包含冻结的物理量统计，不能回退到训练数据。")
            self.models.append({"seed": s, "field": spec.get("field", "wss"), "cfg": cfg,
                                "feat_stats": feat_stats, "model": model,
                                "stats": json.loads(stats_file.read_text(encoding="utf-8"))})
        self.load_seconds = time.perf_counter() - t
        if not self.models:
            raise ValueError("发布包至少需要一个模型权重。")
        self.input_features = list(self.models[0]["cfg"].data.input_features)
        if any(list(item["cfg"].data.input_features) != self.input_features for item in self.models[1:]):
            raise ValueError("同一发布包内模型的输入特征合同不一致。")
        # The weights were validated against one specific STL -> feature program.  A release may
        # pin it in release.json ("feature_contract": {"version", "source_hash"}); loading with a
        # different frozen program then fails closed instead of silently drifting.
        from wss_features import contract as feature_contract
        self.feature_contract = feature_contract()
        declared = self.info.get("feature_contract")
        if isinstance(declared, dict) and declared.get("source_hash"):
            if str(declared["source_hash"]).lower() != self.feature_contract["source_hash"]:
                raise ValueError("发布包声明的特征程序哈希与当前 wss_features 不一致，拒绝加载；"
                                 "请使用与该发布包配套的特征程序版本。")
            self.feature_contract["pinned_by_release"] = True
        # Optional reference profiles (geometry ranges, CV3 out-of-fold p99 population) ship as a
        # sidecar so adding them never changes the release fingerprint that finished jobs are bound to.
        # The file's hash is recorded with every run (schema.model_release_metadata).
        self.reference_sha256 = load_reference_sidecar(self.dir, self.release_id, self.info)

    def _model_specs(self, seeds, *, seed_count: int | None = None):
        """Resolve checkpoint paths for this release's family (legacy X5D layout when undeclared).

        Declared ``models`` lists resolve the same way for every family, so this also works on a
        bare release object that has not been bound to a family yet.
        """
        info = getattr(self, "info", None) or {}
        family = getattr(self, "family", None)
        if family is None:
            from .families import WALL_WSS
            family = WALL_WSS
        return family.model_specs(info, getattr(self, "contract", None) or {}, tuple(seeds), seed_count)

    def _manifest_records(self) -> dict[str, str]:
        """Read the release checksum list used to verify loaded checkpoints."""
        path = self.dir / "MANIFEST.sha256"
        if not path.is_file():
            return {}
        records = {}
        for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
            parts = line.strip().split()
            if len(parts) >= 2 and len(parts[0]) == 64 and parts[1].startswith("models/") and "/ckpt_" in parts[1]:
                records[parts[1]] = parts[0].lower()
        return records

    @property
    def name(self) -> str:
        return self.release_id

    def predict(self, case: dict) -> dict:
        """Run the ensemble through the family adapter and return physical-unit predictions."""
        return self.family.predict(self, case)

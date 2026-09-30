"""Locate and validate every signed-off source a case build depends on."""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from wss_pinn.utils import sha256_file

from . import contract as C


def _load_json(path: Path) -> Any:
    return json.loads(Path(path).read_text(encoding="utf-8"))


@dataclass
class Registry:
    """Cohort-level tables loaded once and shared by all case builds."""

    split: dict[str, Any]
    wall_audit: dict[str, Any]
    solver_audit: dict[str, Any]

    @classmethod
    def load(cls) -> "Registry":
        return cls(_load_json(C.SPLIT_PATH), _load_json(C.WALL_AUDIT_SUMMARY), _load_json(C.SOLVER_AUDIT_SUMMARY))

    # ---- split -----------------------------------------------------------
    def role(self, canonical_id: str) -> str:
        if canonical_id in set(self.split["test_cases"]):
            return "test"
        for fold in self.split["cv"]["folds"]:
            if canonical_id in fold["train_cases"] or canonical_id in fold["validation_cases"]:
                return "train"
        raise KeyError(f"{canonical_id}: not in the formal train138/test34 split")

    def all_cases(self) -> list[str]:
        train = set()
        for fold in self.split["cv"]["folds"]:
            train.update(fold["train_cases"])
            train.update(fold["validation_cases"])
        return sorted(train) + sorted(self.split["test_cases"])

    def validation_fold(self, canonical_id: str) -> int | None:
        for fold in self.split["cv"]["folds"]:
            if canonical_id in fold["validation_cases"]:
                return int(fold["fold"])
        return None

    def patient_group(self, canonical_id: str) -> str:
        related = self.split["cv"].get("protected_related_group") or []
        if canonical_id in related:
            return "=".join(sorted(p.split("/")[-1] for p in related))
        parts = canonical_id.split("/")
        if parts[0] == "ILO":  # ILO/<PATIENT>-<k>/<phase>
            return re.sub(r"-\d+$", "", parts[1])
        return parts[-1]

    # ---- audits ----------------------------------------------------------
    def solver_tier(self, canonical_id: str) -> dict[str, Any]:
        for row in self.solver_audit["solver_quality"]["by_case"]:
            if row["canonical_id"] == canonical_id:
                return {
                    "tier": row["tier"],
                    "continuity_last_max": row.get("continuity_last_max"),
                    "convergence_criterion_type": row.get("convergence_criterion_type"),
                }
        return {"tier": "unknown"}

    def wall_audit_status(self, canonical_id: str) -> dict[str, Any]:
        for row in self.wall_audit.get("failures", []):
            if row["canonical_id"] == canonical_id:
                return {"gate_pass": False, "failed_gates": row["failed_gates"]}
        return {"gate_pass": True, "failed_gates": []}


@dataclass
class CaseSources:
    canonical_id: str
    cohort: str
    role: str
    validation_fold: int | None
    patient_group: str
    raw_dir: Path
    cas_path: Path
    cas_sha256_expected: str
    topology_audit: dict[str, Any]
    atlas_npz: Path
    atlas_summary: Path
    udf_path: Path
    udf_alternates: list[dict[str, Any]] = field(default_factory=list)
    solver: dict[str, Any] = field(default_factory=dict)
    wall_audit: dict[str, Any] = field(default_factory=dict)
    monitor_files: list[str] = field(default_factory=list)

    def interface_semantics(self) -> dict[int, str]:
        """distal BC zone id -> semantic label, from the signed-off topology audit."""
        return {int(row["distal_bc_zone_id"]): row["semantic_label"] for row in self.topology_audit["interfaces"]}


def _pick_udf(raw_dir: Path) -> tuple[Path, list[dict[str, Any]]]:
    root = sorted(raw_dir.glob("udf-inlet*.c"), key=lambda p: ("4" not in p.name, p.name))
    compiled = sorted((raw_dir / "libudf" / "src").glob("udf-inlet*.c")) if (raw_dir / "libudf" / "src").is_dir() else []
    candidates = root + compiled
    if not candidates:
        raise FileNotFoundError(f"no udf-inlet*.c under {raw_dir} (root or libudf/src)")
    chosen = candidates[0]
    chosen_sha = sha256_file(chosen)
    alternates = []
    for path in candidates[1:]:
        sha = sha256_file(path)
        alternates.append({"path": str(path), "sha256": sha, "identical_to_chosen": sha == chosen_sha})
    return chosen, alternates


def locate(canonical_id: str, registry: Registry) -> CaseSources:
    audit_path = C.TOPOLOGY_AUDIT_DIR / f"{canonical_id.replace('/', '__')}.json"
    if not audit_path.is_file():
        raise FileNotFoundError(f"{canonical_id}: topology audit missing: {audit_path}")
    audit = _load_json(audit_path)
    if not audit.get("gate_pass", False):
        raise ValueError(f"{canonical_id}: topology audit gate failed; not eligible for V5 build")
    raw_dir = C.RAW_ROOT / canonical_id
    cas_path = Path(audit["fluent_case"]["path"])
    atlas_dir = C.ATLAS_DIR / canonical_id
    atlas_npz = atlas_dir / "atlas.npz"
    atlas_summary = atlas_dir / "summary.json"
    for required in (raw_dir, cas_path, atlas_npz, atlas_summary):
        if not required.exists():
            raise FileNotFoundError(f"{canonical_id}: missing source {required}")
    udf_path, alternates = _pick_udf(raw_dir)
    monitors = raw_dir / "Global_conditions"
    return CaseSources(
        canonical_id=canonical_id,
        cohort=canonical_id.split("/")[0],
        role=registry.role(canonical_id),
        validation_fold=registry.validation_fold(canonical_id),
        patient_group=registry.patient_group(canonical_id),
        raw_dir=raw_dir,
        cas_path=cas_path,
        cas_sha256_expected=audit["fluent_case"]["sha256"],
        topology_audit=audit,
        atlas_npz=atlas_npz,
        atlas_summary=atlas_summary,
        udf_path=udf_path,
        udf_alternates=alternates,
        solver=registry.solver_tier(canonical_id),
        wall_audit=registry.wall_audit_status(canonical_id),
        monitor_files=sorted(p.name for p in monitors.iterdir()) if monitors.is_dir() else [],
    )

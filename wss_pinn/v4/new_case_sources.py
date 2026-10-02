"""Per-case source resolution for cases that were never in the frozen V4 trees (2026-09-22).

The V4 prep tools (topology / wall / solver audits, centerline V2, feature atlas) originally took
three things from ``data_wss_pinn/…`` manifests that were deleted in the 2026-09-03 cleanup:
the case list, the outlet zone-id -> ``out-le|li|ri|re`` map and the ``.cas`` path (+ peak step).
All three are recoverable from the raw case directory itself:

* the outlet map is the ``t1..t4 = Lookup_Thread(d, <zone id>)`` block of the case UDF
  (``t1``=out-le, ``t2``=out-li, ``t3``=out-ri, ``t4``=out-re; this is how the frozen
  qs-smooth-v3 manifests were produced, ``wss_pinn/tools/build_qs_smooth_v3.py``);
* the ``.cas.gz`` is the one the journal reads (by basename), else the unique ``*.cas.gz``;
* the peak step is the frozen contract constant 1162 (all 172 existing wall audits carry 1162).

Every resolver prefers the legacy manifest when it still exists, so existing cases resolve exactly
as before; the fallbacks only engage for cases without a manifest.
"""
from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]  # no wss_pinn.utils import: loadable standalone from the vmtk env

RAW_ROOT = ROOT / "data_new"
LEGACY_BOUNDARY_ROOT = ROOT / "data_wss_pinn/volume_uvwp_peak_qs_smooth_v3_train123_val15_test35/cases"
LEGACY_V4_ROOT = ROOT / "data_wss_pinn/volume_uvwp_bc_rcr_v4_train138_test35/cases"
LEGACY_SOURCE_MANIFEST = ROOT / "data_wss_pinn/volume_uvwp_bc_rcr_v4_train138_test35/manifest.json"
DEFAULT_SPLIT = ROOT / "wss_pinn/configs/splits/split_WSS_PINN_AG_AAA_ILO_bc_rcr_v4_anatomy_only_train136_test34_s1234.json"
PEAK_STEP = 1162
THREAD_TO_LABEL = {"t1": "out-le", "t2": "out-li", "t3": "out-ri", "t4": "out-re"}
READ_CASE_RE = re.compile(r"/file/read-case\s+(\S+)")


def raw_dir(canonical_id: str) -> Path:
    return RAW_ROOT / canonical_id


def udf_path(raw_case: Path) -> Path:
    """Prefer ``udf-inlet4.c`` (AAA/ILO naming) over ``udf-inlet.c`` (AG), then the compiled copy."""
    for name in ("udf-inlet4.c", "udf-inlet.c"):
        if (raw_case / name).is_file():
            return raw_case / name
    hits = sorted(raw_case.glob("libudf/src/udf-inlet*.c"))
    if hits:
        return hits[0]
    raise FileNotFoundError(f"{raw_case}: no udf-inlet*.c")


def outlet_semantics_from_udf(raw_case: Path) -> dict[int, str]:
    """``{mesh zone id: 'out-le'|'out-li'|'out-ri'|'out-re'}`` from the UDF thread ids."""
    text = udf_path(raw_case).read_text(encoding="utf-8", errors="ignore")
    out: dict[int, str] = {}
    for thread, label in THREAD_TO_LABEL.items():
        match = re.search(rf"\b{thread}\s*=\s*Lookup_Thread\s*\(\s*d\s*,\s*([0-9]+)\s*\)", text)
        if not match:
            raise ValueError(f"{raw_case}: cannot find {thread} = Lookup_Thread(d, <id>) in UDF")
        zone = int(match.group(1))
        if zone in out:
            raise ValueError(f"{raw_case}: zone {zone} used for both {out[zone]} and {label} in UDF")
        out[zone] = label
    return out


OUTLET_SEMANTICS_OVERRIDES = ROOT / "wss_pinn/configs/outlet_semantics_overrides_20261001.json"


def outlet_semantics(canonical_id: str) -> dict[int, str]:
    """Explicit override, else the legacy qs-smooth-v3 manifest when present (existing cases), else the UDF.
    2026-10-01 table (library audit, docs/02-…/04-数据处理与CFD/母库全量审计_2026-10-01.md §9): units whose operator
    names do not match the anatomy (internal / external swapped on a side, or left / right against the library convention);
    it replaces the 2026-09-30 table, whose four units now follow their UDF keys."""
    if OUTLET_SEMANTICS_OVERRIDES.is_file():
        row = json.loads(OUTLET_SEMANTICS_OVERRIDES.read_text(encoding="utf-8"))["cases"].get(canonical_id)
        if row:
            return {int(zone): label for zone, label in row.items()}
    manifest = LEGACY_BOUNDARY_ROOT / canonical_id / "manifest.json"
    if manifest.is_file():
        payload = json.loads(manifest.read_text(encoding="utf-8"))
        return {int(row["mesh_zone_id"]): label for label, row in payload["outlets"].items()}
    return outlet_semantics_from_udf(raw_dir(canonical_id))


def udf_to_dataset_outlets(canonical_id: str, raw_case: Path | None = None) -> dict[str, str]:
    """UDF outlet name (t1..t4 -> out-le|li|ri|re) -> dataset outlet name of the zone that thread points at.
    Identity for every unit without an override."""
    udf = outlet_semantics_from_udf(raw_case if raw_case is not None else raw_dir(canonical_id))
    data = outlet_semantics(canonical_id)
    if set(udf) != set(data):
        raise ValueError(f"{canonical_id}: outlet zones of the UDF {sorted(udf)} and of the dataset naming {sorted(data)} differ")
    return {udf[zone]: data[zone] for zone in udf}


def fluent_case_from_raw(raw_case: Path) -> Path:
    """The ``.cas.gz`` the journal reads (matched by basename), else the unique ``*.cas.gz``."""
    candidates = sorted(p for p in raw_case.glob("*.cas.gz") if ".orig_" not in p.name and ".tmp_" not in p.name)
    journal = raw_case / "2.jou"
    if journal.is_file():
        match = READ_CASE_RE.search(journal.read_text(encoding="utf-8", errors="ignore"))
        if match:
            wanted = raw_case / Path(match.group(1)).name
            if wanted.is_file():
                return wanted
    if len(candidates) == 1:
        return candidates[0]
    raise FileNotFoundError(f"{raw_case}: cannot resolve the Fluent case ({len(candidates)} candidates, journal={journal.is_file()})")


def fluent_case(canonical_id: str) -> Path:
    manifest = LEGACY_BOUNDARY_ROOT / canonical_id / "manifest.json"
    if manifest.is_file():
        payload = json.loads(manifest.read_text(encoding="utf-8"))
        return Path(payload["provenance"]["fluent_case"]["path"])
    return fluent_case_from_raw(raw_dir(canonical_id))


def peak_step(canonical_id: str) -> int:
    manifest = LEGACY_V4_ROOT / canonical_id / "manifest.json"
    if manifest.is_file():
        return int(json.loads(manifest.read_text(encoding="utf-8"))["peak_step"])
    return PEAK_STEP


def case_entries(extra: list[str] | None = None, split_path: Path | None = None) -> list[dict[str, Any]]:
    """Case list: legacy source manifest when it exists, else the frozen split (train + test); ``extra`` ids are
    appended with role ``new`` (deduplicated, order preserved)."""
    rows: list[dict[str, Any]] = []
    if LEGACY_SOURCE_MANIFEST.is_file():
        payload = json.loads(LEGACY_SOURCE_MANIFEST.read_text(encoding="utf-8"))
        rows = [{"canonical_id": r["canonical_id"], "role": r["role"]} for r in payload["cases"]]
    else:
        split = json.loads(Path(split_path or DEFAULT_SPLIT).read_text(encoding="utf-8"))
        rows = [{"canonical_id": c, "role": "train"} for c in split["train_cases"]] + [{"canonical_id": c, "role": "test"} for c in split["test_cases"]]
    seen = {r["canonical_id"] for r in rows}
    for cid in extra or []:
        if cid not in seen:
            rows.append({"canonical_id": cid, "role": "new"})
            seen.add(cid)
    return rows


def read_case_list(path: Path | None) -> list[str]:
    if path is None:
        return []
    return [line.strip() for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip() and not line.startswith("#")]

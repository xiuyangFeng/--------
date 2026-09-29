"""Model cards: what each release is, how it was validated and where it does not apply (C3, 2026-09-30).

A card is a small, human-readable JSON document (``wss-deploy.model-card/v1``, contract
``WORKSPACE_V2_CONTRACT.md`` §4) shown next to the numbers of a result: the three evidence tiers
(``geometry`` / ``model`` / ``derived``), the held-out agreement of every predicted field, the flow
protocol the training CFD used and the named colour windows.

Where a card lives:

* ``<release_dir>/model_card.json`` — a future release may carry its own card; it wins.
* ``wss_deploy/model_cards/<release_id>.json`` — the reviewed cards of the current releases (kept in
  the code repository so they are versioned; nothing is ever written below ``outputs/``).

Every number in a card must come from the release's own ``release.json`` (or its ``metrics/`` files);
``tests/test_model_cards.py`` checks the shipped cards against the release packages when they are
present.  A card never changes a prediction, a summary block or the release fingerprint.
"""
from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any, Iterable, Mapping

SCHEMA_VERSION = "wss-deploy.model-card/v1"
CARD_DIR = Path(__file__).resolve().parent / "model_cards"
SIDECAR_NAME = "model_card.json"
TIERS = ("geometry", "model", "derived")
TIER_LABELS = {"geometry": "几何", "model": "模型", "derived": "派生"}
# Sub-keys a card must carry (values may be null / empty where the release declares nothing).
REQUIRED_KEYS = ("schema_version", "release_id", "display_name", "short_name", "version_date", "purpose",
                 "training", "protocol", "validation", "field_tiers", "usage", "weaknesses", "not_applicable",
                 "display_windows", "caveat")


def _read(path: Path) -> dict | None:
    try:
        value = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    return value if isinstance(value, dict) else None


def validate(card: Any, release_id: str | None = None) -> dict | None:
    """The card when it is a well-formed ``wss-deploy.model-card/v1`` document for ``release_id``; else None.

    Fails closed: a card copied from another release (``release_id`` mismatch), an unknown schema or a
    tier outside the three documented ones is ignored rather than shown next to the wrong weights.
    """
    if not isinstance(card, Mapping) or card.get("schema_version") != SCHEMA_VERSION:
        return None
    if any(key not in card for key in REQUIRED_KEYS):
        return None
    if release_id is not None and card.get("release_id") != release_id:
        return None
    tiers = card.get("field_tiers")
    if not isinstance(tiers, Mapping) or any(value not in TIERS for value in tiers.values()):
        return None
    windows = card.get("display_windows")
    if not isinstance(windows, Mapping):
        return None
    for items in windows.values():
        if not isinstance(items, list):
            return None
        for item in items:
            rng = item.get("range") if isinstance(item, Mapping) else None
            if (not isinstance(rng, list) or len(rng) != 2 or not all(isinstance(v, (int, float)) and not isinstance(v, bool) for v in rng)
                    or rng[0] >= rng[1] or not isinstance(item.get("id"), str) or not isinstance(item.get("label"), str)):
                return None
    validation = card.get("validation")
    if not isinstance(validation, Mapping) or not isinstance(validation.get("fields", {}), Mapping):
        return None
    return copy.deepcopy(dict(card))


def load(release_id: str | None, release_dir: str | Path | None = None) -> dict | None:
    """The model card of ``release_id``: ``<release_dir>/model_card.json`` first, then the repository copy.

    Returns a fresh copy (callers may annotate it) or None when no valid card exists.
    """
    if not isinstance(release_id, str) or not release_id.strip() or "/" in release_id or "\\" in release_id \
            or release_id.startswith("."):
        return None
    if release_dir is not None:
        card = validate(_read(Path(release_dir) / SIDECAR_NAME), release_id)
        if card is not None:
            card.setdefault("card_source", "release_dir")
            return card
    card = validate(_read(CARD_DIR / f"{release_id}.json"), release_id)
    if card is not None:
        card.setdefault("card_source", "repository")
    return card


def _registry_entries(registry) -> list[tuple[str, Path | None]]:
    if registry is None:
        return []
    if isinstance(registry, Mapping):          # {release_id: release_dir or None}
        return [(str(key), Path(value) if value else None) for key, value in registry.items()]
    if isinstance(registry, (list, tuple, set)) and all(isinstance(item, str) for item in registry):
        return [(item, None) for item in registry]
    entries = []
    for item in registry.list() if hasattr(registry, "list") else []:
        rid = item.get("id") if isinstance(item, Mapping) else None
        if not isinstance(rid, str):
            continue
        path = None
        try:
            path = registry.describe(rid).path
        except Exception:  # noqa: BLE001 — a release that cannot be described simply has no sidecar card
            path = None
        entries.append((rid, path))
    return entries


def all_cards(registry) -> dict[str, dict | None]:
    """``{release_id: card or None}`` for every release the registry lists (contract §2 ``/api/v2/model-cards``).

    ``registry`` is a :class:`registry.ReleaseRegistry`, a list of release ids or ``{id: release_dir}``.
    """
    return {rid: load(rid, path) for rid, path in _registry_entries(registry)}


def field_tier(card: Mapping[str, Any] | None, field_id: str, default: str | None = None) -> str | None:
    """``geometry`` / ``model`` / ``derived`` for a field of the card, else ``default``."""
    tiers = card.get("field_tiers") if isinstance(card, Mapping) else None
    value = tiers.get(field_id) if isinstance(tiers, Mapping) else None
    return value if value in TIERS else default


def field_validation(card: Mapping[str, Any] | None, field_id: str) -> dict | None:
    """The card's held-out agreement block of one field (plus ``holdout_n``), or None."""
    validation = card.get("validation") if isinstance(card, Mapping) else None
    fields = validation.get("fields") if isinstance(validation, Mapping) else None
    block = fields.get(field_id) if isinstance(fields, Mapping) else None
    if not isinstance(block, Mapping):
        return None
    return {"holdout_n": validation.get("holdout_n"), **copy.deepcopy(dict(block))}


def display_windows(card: Mapping[str, Any] | None, field_id: str) -> list[dict]:
    """Named colour windows of a field (U15; every window carries ``provisional``); [] without a card."""
    windows = card.get("display_windows") if isinstance(card, Mapping) else None
    items = windows.get(field_id) if isinstance(windows, Mapping) else None
    return copy.deepcopy(list(items)) if isinstance(items, list) else []


def validation_rows(card: Mapping[str, Any] | None) -> list[dict]:
    """One row per quantity for the one-page 「模型验证」 table: label, tier, agreement text, usage."""
    if not isinstance(card, Mapping):
        return []
    labels = card.get("field_labels") if isinstance(card.get("field_labels"), Mapping) else {}
    usage = card.get("usage") if isinstance(card.get("usage"), Mapping) else {}
    validation = card.get("validation") if isinstance(card.get("validation"), Mapping) else {}
    fields = validation.get("fields") if isinstance(validation.get("fields"), Mapping) else {}
    rows = []
    tiers = card.get("field_tiers") if isinstance(card.get("field_tiers"), Mapping) else {}
    # ``field_labels`` (when present) chooses and orders the rows a reader sees; else every tiered field.
    order = [key for key in labels if key in tiers] if labels else list(tiers)
    for field_id in order:
        tier = tiers[field_id]
        block = fields.get(field_id) if isinstance(fields.get(field_id), Mapping) else {}
        rows.append({"field": field_id, "label": str(labels.get(field_id) or field_id), "tier": tier,
                     "tier_label": TIER_LABELS.get(tier, tier), "agreement": block.get("summary"),
                     "usage": usage.get(field_id)})
    return rows


def card_ids() -> list[str]:
    """Release ids with a repository card (sorted)."""
    return sorted(path.stem for path in CARD_DIR.glob("*.json"))


def iter_cards() -> Iterable[dict]:
    for rid in card_ids():
        card = load(rid)
        if card is not None:
            yield card


__all__ = ["CARD_DIR", "REQUIRED_KEYS", "SCHEMA_VERSION", "SIDECAR_NAME", "TIERS", "TIER_LABELS", "all_cards", "card_ids",
           "display_windows", "field_tier", "field_validation", "iter_cards", "load", "validate", "validation_rows"]

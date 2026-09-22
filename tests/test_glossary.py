"""C17: static/glossary.json mirrors glossary.py, and every data-gloss key used by the pages exists."""
from __future__ import annotations

import json
import re
from pathlib import Path

from wss_deploy.glossary import GLOSSARY, SCHEMA_VERSION, glossary_document, glossary_json
from wss_deploy.paths import STATIC_DIR

PAGE_SOURCES = ("wss_deploy/report.py", "wss_deploy/volume_report.py", "wss_deploy/static/volume_viewer.js", "wss_deploy/static/report_common.js",
                "wss_deploy/static/index.html", "wss_deploy/static/app.js", "wss_deploy/onepager.py", "wss_deploy/static/compare.js")
ROOT = Path(__file__).resolve().parents[1]


def test_static_json_matches_python_source():
    stored = (STATIC_DIR / "glossary.json").read_text(encoding="utf-8")
    assert stored == glossary_json(), "run: PYTHONPATH=. python -m wss_deploy.glossary"
    document = json.loads(stored)
    assert document["schema_version"] == SCHEMA_VERSION and document["terms"] == GLOSSARY == glossary_document()["terms"]


def test_every_term_has_the_four_fields_and_required_keys_exist():
    for key, term in GLOSSARY.items():
        assert re.fullmatch(r"[a-z0-9_]+", key), key
        assert set(term) == {"zh", "zh_desc", "en", "en_desc"} and all(term.values()), key
    required = {"p99", "max", "area_fraction", "area_weighted_p99", "relative_pressure", "delta_p", "trust_interpolation_uncovered", "trust_rough_surface",
                "trust_geometry_out_of_range", "trust_low_sample_support", "trust_near_opening", "confidence_proxy", "population_percentile", "quality_grade",
                "release", "feature_contract", "run_identity", "standard_views"}
    assert required <= set(GLOSSARY)


def test_page_gloss_keys_are_defined():
    missing = {}
    for relative in PAGE_SOURCES:
        path = ROOT / relative
        if not path.is_file():
            continue
        text = path.read_text(encoding="utf-8")
        keys = set(re.findall(r"""data-gloss=["']([a-z0-9_]+)["']""", text)) | set(re.findall(r"""data-gloss=\$\{[^}]*\}""", text) and [])
        unknown = sorted(k for k in keys if k not in GLOSSARY)
        if unknown:
            missing[relative] = unknown
    assert not missing, f"undefined glossary keys: {missing}"

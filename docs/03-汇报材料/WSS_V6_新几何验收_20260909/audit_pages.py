"""Verify review identity, candidate status and metrics against final source reports.

Run after the complete candidate and page batches have finished. No label reads.
"""
import html
import json
import math
import argparse
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
SOURCE = ROOT / "outputs/wss_v6_geometry_candidate_20260909"
SNAPSHOT = ROOT / "data_wss_v5/anatomy_pointcloud_v5_20260906"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--expected-schema', help='Default: final candidate manifest schema')
    args = parser.parse_args()
    expected_schema = args.expected_schema or json.loads((SOURCE / 'manifest.json').read_text())['schema']
    expected = {p.name for p in (SNAPSHOT / "cases").iterdir() if (p / "case.h5").is_file()}
    pages = {p.name.removesuffix(".summary.json"): p for p in (HERE / "cases").glob("*.summary.json")}
    errors = []
    if set(pages) != expected:
        errors.append({"identity_set_mismatch": {"missing": sorted(expected-set(pages)), "unexpected": sorted(set(pages)-expected)}})
    manifest = json.loads((HERE / "render_manifest.json").read_text())
    if manifest["total_cases"] != len(expected) or manifest["candidate_detail_pages"] != len(expected) or manifest["preview_pages"] != 0:
        errors.append({"manifest_counts": {k: manifest[k] for k in ["total_cases", "candidate_detail_pages", "preview_pages"]}})
    checked = []
    for key in sorted(expected & set(pages)):
        summary = json.loads(pages[key].read_text())
        report = json.loads((SOURCE / "cases" / key / "report.json").read_text())
        issues = []
        if summary["canonical_id"] != report["canonical_id"] or summary["canonical_id"].replace("/", "__") != key:
            issues.append("canonical_id")
        if summary["preview_only"] or summary["training_allowed"] or not summary["awaiting_user_review"]:
            issues.append("candidate_review_status")
        if summary.get("source_schema") != report["schema"] or report["schema"] != expected_schema:
            issues.append("source_schema")
        if summary.get("source_config_sha256") != report["config_sha256"]:
            issues.append("source_config_sha256")
        source_values = {"sections": report["P4_sections"]["count"],
                         "section_valid_fraction": report["P4_sections"]["reference_valid_fraction"],
                         "pc_section_valid_fraction": report["P4_sections"]["pointcloud_valid_fraction"],
                         "wall_map_valid_fraction": report["P5_wall_mapping"]["reference_valid_fraction"]}
        for name, expected_value in source_values.items():
            if not math.isclose(summary[name], expected_value, rel_tol=1e-12, abs_tol=1e-12):
                issues.append(name)
        page = HERE / summary["page"]
        if not page.is_file():
            issues.append("missing_html")
        else:
            if page.stat().st_size != summary["html_bytes"]:
                issues.append("html_file_size")
            with page.open(encoding="utf-8") as stream:
                prefix = stream.read(20000)
            if f'<title>{html.escape(summary["canonical_id"])} ·' not in prefix:
                issues.append("html_title_identity")
        if issues:
            errors.append({"case": key, "issues": issues})
        checked.append(key)
    result = {"pass": not errors, "expected_schema": expected_schema, "expected_cases": len(expected), "checked_cases": len(checked),
              "errors": errors, "checks": ["exact_snapshot_identity_set", "manifest_candidate_counts", "source_schema_and_config_hash", "summary_vs_final_source_metrics", "html_title_existence_size", "unapproved_candidate_status"]}
    (HERE / "page_audit.json").write_text(json.dumps(result, ensure_ascii=False, indent=2))
    print(json.dumps(result, ensure_ascii=False))
    raise SystemExit(0 if result["pass"] else 1)


if __name__ == "__main__":
    main()

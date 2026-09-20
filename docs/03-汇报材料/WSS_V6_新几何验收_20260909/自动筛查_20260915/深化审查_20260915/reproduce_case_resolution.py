"""One-command regeneration from the frozen 29-case review input snapshot."""
from pathlib import Path
import pandas as pd
from resolve_bifurcation_routes import run_case
import build_case_resolution_audit
import make_resolution_review
import validate_resolution_outputs


if __name__ == "__main__":
    base = Path(__file__).resolve().parent
    baseline = pd.read_csv(base / "case_resolution_input_baseline.csv")
    for cid in baseline.loc[baseline.bif_undetermined > 0, "canonical_id"]:
        run_case(cid.replace("/", "__"), base / "bifurcation_route_recovery", step=.25)
    build_case_resolution_audit.main()
    make_resolution_review.main()
    validate_resolution_outputs.main()

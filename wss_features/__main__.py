"""Record or show the frozen feature-program contract.

    python -m wss_features show      # print version, current source hash and whether it matches CONTRACT.json
    python -m wss_features record    # rewrite CONTRACT.json for the current sources (a deliberate bump)
"""
from __future__ import annotations

import datetime as dt
import json
import sys
from pathlib import Path

from . import FEATURE_CONTRACT_VERSION, _ROOT, contract, contract_hash


def main(argv=None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    command = argv[0] if argv else "show"
    if command == "show":
        print(json.dumps(contract(), indent=1))
        return 0
    if command == "record":
        record = {"version": FEATURE_CONTRACT_VERSION, "source_hash": contract_hash(),
                  "recorded_on": dt.date.today().isoformat(),
                  "note": "Bump deliberately: any change to wss_features/*.py must be re-validated against the training implementation."}
        (Path(_ROOT) / "CONTRACT.json").write_text(json.dumps(record, indent=1) + "\n", encoding="utf-8")
        print(json.dumps(record, indent=1))
        return 0
    print(__doc__)
    return 2


if __name__ == "__main__":
    raise SystemExit(main())

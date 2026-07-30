from __future__ import annotations

import argparse

from wss_pinn.config import ExperimentConfig
from wss_pinn.data.sidecar import build_sidecars_from_config


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    config = ExperimentConfig.from_json(args.config)
    result = build_sidecars_from_config(config)
    print(f"P1 completed cases={len(result['cases'])} manifest={result['path']}")
    print(f"sha256={result['sha256']}")


if __name__ == "__main__":
    main()


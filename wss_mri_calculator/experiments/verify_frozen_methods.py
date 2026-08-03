"""Verify that frozen V1/V2 source/config/result artifacts are unchanged."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MANIFESTS = (
    ROOT / "experiments/pointcloud_adaptive_v1/freeze_manifest.json",
    ROOT / "experiments/pointcloud_multiscale_v2/freeze_manifest.json",
    ROOT / "experiments/pointcloud_normal_multiscale_v3/freeze_manifest.json",
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify(manifests: tuple[Path, ...]) -> None:
    failed = False
    for manifest_path in manifests:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        method = manifest["method"]
        if not manifest.get("immutable", False):
            raise ValueError(f"manifest is not immutable: {manifest_path}")
        print(f"[{method}]")
        for artifact in manifest["artifacts"]:
            path = ROOT / artifact["path"]
            if not path.is_file():
                failed = True
                print(f"  MISSING {artifact['path']}")
                continue
            actual = sha256_file(path)
            expected = artifact["sha256"]
            state = "OK" if actual == expected else "CHANGED"
            print(f"  {state:<7} {artifact['path']}")
            failed |= actual != expected
    if failed:
        raise SystemExit("frozen method verification failed")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", nargs="*", type=Path)
    args = parser.parse_args()
    manifests = tuple(path.resolve() for path in args.manifest) or DEFAULT_MANIFESTS
    verify(manifests)


if __name__ == "__main__":
    main()

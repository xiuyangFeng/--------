from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from wss_pinn.utils import ROOT, sha256_file
from wss_pinn.volume_field.data.audit import audit_dataset
from wss_pinn.volume_field.data.builder import (
    GEOMETRY_NAMES,
    SCHEMA_VERSION,
    compute_train_stats,
)


def _write_json(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload), encoding="utf-8")


def _write_array(root: Path, name: str, value: np.ndarray) -> dict:
    path = root / f"{name}.npy"
    np.save(path, value, allow_pickle=False)
    return {
        "path": str(path.resolve()),
        "sha256": sha256_file(path),
        "shape": list(value.shape),
        "dtype": str(value.dtype),
    }


class VolumeDataContractTests(unittest.TestCase):
    def _synthetic_gate(self, root: Path) -> tuple[Path, Path, Path, Path]:
        root.mkdir(parents=True, exist_ok=True)
        case_id = "AG/fast/SYNTHETIC"
        case_root = root / "case"
        case_root.mkdir()
        rng = np.random.default_rng(1234)
        strict_n = 10_000
        duplicate_n = 5
        n = strict_n + duplicate_n
        is_wall = np.zeros(n, dtype=np.bool_)
        is_wall[-duplicate_n:] = True
        velocity = rng.normal(scale=0.1, size=(n, 3)).astype(np.float32)
        velocity[is_wall] = 0.0
        pressure = np.zeros(n, dtype=np.float32)
        pressure[:strict_n] = np.linspace(-1.0, 1.0, strict_n, dtype=np.float32)
        pressure[is_wall] = 5000.0
        arrays = {
            "interior_coords": rng.uniform(-1, 1, size=(n, 3)).astype(np.float32),
            "interior_geometry": np.column_stack(
                [
                    rng.uniform(0, 1, size=n),
                    rng.uniform(1, 5, size=n),
                    rng.normal(size=n),
                ]
            ).astype(np.float32),
            "interior_region": rng.integers(0, 2, size=n, dtype=np.int8),
            "interior_is_wall": is_wall,
            "velocity_m_s": velocity,
            "pressure_relative_pa": pressure,
            "wall_coords": rng.uniform(-1, 1, size=(1024, 3)).astype(np.float32),
            "wall_geometry": np.column_stack(
                [
                    rng.uniform(0, 1, size=1024),
                    rng.uniform(1, 5, size=1024),
                    rng.normal(size=1024),
                ]
            ).astype(np.float32),
        }
        files = {
            name: _write_array(case_root, name, value)
            for name, value in arrays.items()
        }
        parent_bundle = root / "parent_bundle.bin"
        parent_raw = root / "parent_raw.bin"
        parent_bundle.write_bytes(b"bundle")
        parent_raw.write_bytes(b"raw")
        case_manifest = case_root / "manifest.json"
        _write_json(
            case_manifest,
            {
                "schema_version": SCHEMA_VERSION,
                "status": "completed",
                "route": "volume_uvwp_peak_v1",
                "canonical_id": case_id,
                "role": "train",
                "pressure_target": "case_strict_volume_mean_centered_pa",
                "geometry_features": list(GEOMETRY_NAMES),
                "counts": {
                    "interior": n,
                    "strict_interior": strict_n,
                    "volume_wall_duplicates": duplicate_n,
                    "near_wall": int(np.sum(arrays["interior_region"] == 1)),
                    "core": int(np.sum(arrays["interior_region"] == 0)),
                    "wall": 1024,
                },
                "scales": {"length_m": 0.05, "density_kg_m3": 1060.0},
                "vector_frame": {
                    "velocity": (
                        "raw Fluent vector right-multiplied by transform_rotation"
                    )
                },
                "parents": {
                    "bundle": {
                        "path": str(parent_bundle.resolve()),
                        "sha256": sha256_file(parent_bundle),
                    },
                    "raw_peak": {
                        "path": str(parent_raw.resolve()),
                        "sha256": sha256_file(parent_raw),
                    },
                },
                "files": files,
            },
        )
        manifest_sha256 = sha256_file(case_manifest)
        split = root / "split.json"
        _write_json(
            split,
            {
                "cases": [
                    {
                        "canonical_id": case_id,
                        "case_id": "fast/SYNTHETIC",
                        "cohort": "AG",
                        "role": "train",
                        "raw_case_dir": str(root / "unused_raw"),
                        "bundle_path": str(root / "unused_bundle"),
                    }
                ]
            },
        )
        stats = root / "field_stats.json"
        _write_json(
            stats,
            {
                "schema_version": SCHEMA_VERSION,
                "status": "completed",
                "route": "volume_uvwp_peak_v1",
                "scope": "train138 strict-volume only",
                "population": {"interior_is_wall": False},
                "split": {"sha256": sha256_file(split)},
                "case_manifest_sha256": {case_id: manifest_sha256},
                "geometry": {
                    "names": list(GEOMETRY_NAMES),
                    "mean": [0.5, 3.0, 0.0],
                    "std": [0.2, 1.0, 1.0],
                    "curvature_clip_abs": 3.0,
                },
                "velocity_m_s": {
                    "mean": [0.0, 0.0, 0.0],
                    "std": [0.1, 0.1, 0.1],
                },
                "pressure_relative_pa": {"mean": [0.0], "std": [1.0]},
                "physics_nondimensionalization": {
                    "velocity_m_s": 1.0,
                    "density_kg_m3": 1060.0,
                    "pressure_pa": 1060.0,
                },
            },
        )
        aggregate = root / "manifest.json"
        _write_json(
            aggregate,
            {
                "schema_version": SCHEMA_VERSION,
                "status": "completed",
                "route": "volume_uvwp_peak_v1",
                "split": {"sha256": sha256_file(split)},
                "field_stats": {
                    "path": str(stats.resolve()),
                    "sha256": sha256_file(stats),
                },
                "cases": [
                    {
                        "canonical_id": case_id,
                        "role": "train",
                        "manifest": str(case_manifest.resolve()),
                        "manifest_sha256": manifest_sha256,
                    }
                ],
            },
        )
        return aggregate, split, stats, case_manifest

    def test_gate_accepts_embedded_wall_rows_only_when_excluded_and_zero(self):
        with tempfile.TemporaryDirectory() as temporary:
            aggregate, split, stats, _ = self._synthetic_gate(Path(temporary))
            report = audit_dataset(aggregate, split, stats)
            self.assertEqual(report["gate_result"], "pass")
            self.assertEqual(report["rows"][0]["strict_interior_rows"], 10_000)
            self.assertEqual(report["rows"][0]["volume_wall_duplicate_rows"], 5)
            self.assertAlmostEqual(report["rows"][0]["pressure_relative_mean_pa"], 0.0)

    def test_gate_rejects_manifest_hash_and_stats_membership_drift(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            aggregate, split, stats, _ = self._synthetic_gate(root)
            aggregate_payload = json.loads(aggregate.read_text(encoding="utf-8"))
            aggregate_payload["cases"][0]["manifest_sha256"] = "0" * 64
            _write_json(aggregate, aggregate_payload)
            report = audit_dataset(aggregate, split, stats)
            self.assertEqual(report["gate_result"], "fail")
            self.assertIn("manifest hash drift", report["rows"][0]["error"])

            aggregate, split, stats, _ = self._synthetic_gate(root / "second")
            stats_payload = json.loads(stats.read_text(encoding="utf-8"))
            stats_payload["case_manifest_sha256"]["AG/fast/SYNTHETIC"] = "1" * 64
            _write_json(stats, stats_payload)
            aggregate_payload = json.loads(aggregate.read_text(encoding="utf-8"))
            aggregate_payload["field_stats"]["sha256"] = sha256_file(stats)
            _write_json(aggregate, aggregate_payload)
            report = audit_dataset(aggregate, split, stats)
            self.assertEqual(report["gate_result"], "fail")
            self.assertFalse(
                report["field_stats"]["train_manifest_hashes_passed"]
            )

    def test_train_stats_exclude_wall_duplicates_and_test_cases(self):
        data_parent = ROOT / "data_wss_pinn"
        data_parent.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(dir=data_parent) as temporary:
            root = Path(temporary)
            rows = []
            for role, offset in (("train", 0.0), ("test", 1000.0)):
                case_root = root / role
                case_root.mkdir()
                geometry = np.asarray(
                    [
                        [0.0 + offset, 2.0 + offset, -1.0],
                        [1.0 + offset, 4.0 + offset, 1.0],
                        [999.0, 999.0, 999.0],
                    ],
                    dtype=np.float32,
                )
                velocity = np.asarray(
                    [[1.0 + offset, 0.0, 0.0], [3.0 + offset, 0.0, 0.0], [999.0] * 3],
                    dtype=np.float32,
                )
                pressure = np.asarray([-1.0 + offset, 1.0 + offset, 999.0], dtype=np.float32)
                is_wall = np.asarray([False, False, True])
                files = {}
                for name, value in (
                    ("interior_geometry", geometry),
                    ("velocity_m_s", velocity),
                    ("pressure_relative_pa", pressure),
                    ("interior_is_wall", is_wall),
                ):
                    path = case_root / f"{name}.npy"
                    np.save(path, value, allow_pickle=False)
                    files[name] = {"path": str(path.resolve())}
                manifest = case_root / "manifest.json"
                _write_json(manifest, {"files": files})
                rows.append(
                    {
                        "canonical_id": f"AG/{role}/SYNTHETIC",
                        "role": role,
                        "manifest": str(manifest.resolve()),
                        "manifest_sha256": sha256_file(manifest),
                    }
                )
            split = root / "split.json"
            _write_json(split, {"synthetic": True})
            output = root / "field_stats.json"
            compute_train_stats(rows, output, split)
            result = json.loads(output.read_text(encoding="utf-8"))
            np.testing.assert_allclose(result["geometry"]["mean"][:2], [0.5, 3.0])
            np.testing.assert_allclose(result["velocity_m_s"]["mean"], [2.0, 0.0, 0.0])
            np.testing.assert_allclose(result["pressure_relative_pa"]["mean"], [0.0])
            self.assertEqual(
                set(result["case_manifest_sha256"]), {"AG/train/SYNTHETIC"}
            )


if __name__ == "__main__":
    unittest.main()

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from wss_pinn.utils import ROOT, sha256_file
from wss_pinn.volume_field.config import VolumeExperimentConfig
from wss_pinn.volume_field.train import run


class VolumeTrainSmokeTests(unittest.TestCase):
    def test_pointnet_pinn_cpu_single_step_is_finite(self):
        data_parent = ROOT / "data_wss_pinn"
        output_parent = ROOT / "outputs/wss_pinn"
        data_parent.mkdir(parents=True, exist_ok=True)
        output_parent.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(dir=data_parent) as temporary:
            root = Path(temporary)
            case_dir = root / "case"
            case_dir.mkdir()
            rng = np.random.default_rng(1234)
            n = 120
            wall_n = 40
            arrays = {
                "interior_coords": rng.uniform(-1, 1, size=(n, 3)).astype(np.float32),
                "interior_geometry": np.column_stack(
                    [
                        rng.uniform(0, 1, size=n),
                        rng.uniform(1, 10, size=n),
                        rng.normal(size=n),
                    ]
                ).astype(np.float32),
                "interior_region": rng.integers(0, 2, size=n, dtype=np.int8),
                "interior_is_wall": np.zeros(n, dtype=bool),
                "velocity_m_s": rng.normal(scale=0.2, size=(n, 3)).astype(np.float32),
                "pressure_relative_pa": rng.normal(scale=100, size=n).astype(np.float32),
                "wall_coords": rng.uniform(-1, 1, size=(wall_n, 3)).astype(np.float32),
                "wall_geometry": np.column_stack(
                    [
                        rng.uniform(0, 1, size=wall_n),
                        rng.uniform(1, 10, size=wall_n),
                        rng.normal(size=wall_n),
                    ]
                ).astype(np.float32),
            }
            files = {}
            for name, value in arrays.items():
                path = case_dir / f"{name}.npy"
                np.save(path, value, allow_pickle=False)
                files[name] = {"path": str(path)}
            case_manifest = case_dir / "manifest.json"
            case_manifest.write_text(
                json.dumps(
                    {
                        "schema_version": 2,
                        "status": "completed",
                        "route": "volume_uvwp_peak_v1",
                        "canonical_id": "AG/fast/SYNTHETIC",
                        "role": "train",
                        "pressure_target": "case_strict_volume_mean_centered_pa",
                        "scales": {"length_m": 0.05, "density_kg_m3": 1060.0},
                        "files": files,
                    }
                ),
                encoding="utf-8",
            )
            case_manifest_sha256 = sha256_file(case_manifest)
            stats = root / "field_stats.json"
            stats.write_text(
                json.dumps(
                    {
                        "schema_version": 2,
                        "status": "completed",
                        "route": "volume_uvwp_peak_v1",
                        "scope": "train138 strict-volume only",
                        "split": {"sha256": "synthetic-split"},
                        "case_manifest_sha256": {
                            "AG/fast/SYNTHETIC": case_manifest_sha256
                        },
                        "geometry": {
                            "mean": [0.5, 5.0, 0.0],
                            "std": [0.3, 2.0, 1.0],
                            "curvature_clip_abs": 3.0,
                        },
                        "velocity_m_s": {
                            "mean": [0.0, 0.0, 0.0],
                            "std": [0.2, 0.2, 0.2],
                        },
                        "pressure_relative_pa": {"mean": [0.0], "std": [100.0]},
                    }
                ),
                encoding="utf-8",
            )
            aggregate = root / "manifest.json"
            aggregate.write_text(
                json.dumps(
                    {
                        "schema_version": 2,
                        "status": "completed",
                        "route": "volume_uvwp_peak_v1",
                        "split": {"sha256": "synthetic-split"},
                        "field_stats": {
                            "path": str(stats.resolve()),
                            "sha256": sha256_file(stats),
                        },
                        "cases": [
                            {
                                "canonical_id": "AG/fast/SYNTHETIC",
                                "role": "train",
                                "manifest": str(case_manifest),
                                "manifest_sha256": case_manifest_sha256,
                            }
                        ],
                    }
                ),
                encoding="utf-8",
            )
            base = VolumeExperimentConfig.from_json(
                "wss_pinn/configs/volume_uvwp_peak_v1/pointnet_xyz_pinn.json"
            ).as_dict()
            base["paths"]["sidecar_manifest"] = str(aggregate)
            base["paths"]["field_stats"] = str(stats)
            base["paths"]["run_dir"] = str(
                output_parent / "_test_volume_train_smoke"
            )
            base["sampling"].update(
                {
                    "support_points": 32,
                    "query_points": 32,
                    "physics_points": 8,
                    "wall_points": 8,
                }
            )
            base["train"].update(
                {"batch_cases": 1, "epochs": 1, "device": "cpu", "num_workers": 0}
            )
            config = VolumeExperimentConfig(base)
            report = run(config, dry_run=True, device_override="cpu")
            self.assertEqual(report["status"], "dry_run_completed")
            self.assertTrue(report["last_record"]["all_finite"])
            self.assertGreater(report["last_record"]["physics_total"], 0.0)

            with tempfile.TemporaryDirectory(dir=output_parent) as output_temporary:
                formal_payload = config.as_dict()
                formal_payload["paths"]["run_dir"] = str(
                    Path(output_temporary) / "resume_contract"
                )
                formal_payload["train"]["epochs"] = 1
                formal_payload["train"]["log_every_steps"] = 1
                formal = VolumeExperimentConfig(formal_payload)
                first = run(formal, dry_run=False, device_override="cpu")
                self.assertEqual(first["start_epoch"], 0)
                self.assertEqual(first["end_epoch_exclusive"], 1)
                initialization = formal.run_dir / "checkpoints/initialization.pt"
                initialization_sha256 = sha256_file(initialization)
                last = formal.run_dir / "checkpoints/last.pt"

                resume_payload = formal.as_dict()
                resume_payload["train"]["epochs"] = 2
                resume_payload["train"]["resume"] = str(last)
                resumed = run(
                    VolumeExperimentConfig(resume_payload),
                    dry_run=False,
                    device_override="cpu",
                )
                self.assertEqual(resumed["epochs_executed"], 1)
                self.assertEqual(resumed["start_epoch"], 1)
                self.assertEqual(resumed["end_epoch_exclusive"], 2)
                self.assertEqual(sha256_file(initialization), initialization_sha256)
                self.assertTrue((formal.run_dir / "resume_events.jsonl").exists())

                cross_run_payload = formal.as_dict()
                cross_run_payload["paths"]["run_dir"] = str(
                    Path(output_temporary) / "forbidden_cross_run"
                )
                cross_run_payload["train"]["epochs"] = 2
                cross_run_payload["train"]["resume"] = str(last)
                with self.assertRaisesRegex(ValueError, "cross-run"):
                    run(
                        VolumeExperimentConfig(cross_run_payload),
                        dry_run=False,
                        device_override="cpu",
                    )


if __name__ == "__main__":
    unittest.main()

from __future__ import annotations

import unittest

import numpy as np

from wss_pinn.data.sidecar import align_raw_interior_to_bundle
from wss_pinn.tools.audit_full_dataset import audit_split
from wss_pinn.tools.audit_sidecar_dataset import audit
from wss_pinn.utils import ROOT


class DataAuditTests(unittest.TestCase):
    def test_spatial_subset_alignment_recovers_cropped_rows(self) -> None:
        raw = np.asarray(
            [[0.0, 0.0, 0.0], [1.0, 2.0, 3.0], [4.0, 5.0, 6.0]],
            dtype=np.float64,
        )
        bundle = {
            "unit_factor": np.asarray(1.0),
            "transform_centroid": np.zeros(3),
            "transform_rotation": np.eye(3),
            "coord_scale": np.asarray(1.0),
            "int_coords_norm": raw[[2, 0]].astype(np.float32),
        }
        indices, delta, mode = align_raw_interior_to_bundle(
            {"coords": raw}, bundle
        )
        np.testing.assert_array_equal(indices, [2, 0])
        self.assertLessEqual(float(np.max(delta)), 5e-5)
        self.assertEqual(mode, "spatial_subset")

    def test_source_audit_deep_pilot_case_passes(self) -> None:
        report = audit_split(
            ROOT / "wss_pinn/configs/pilot_cases.json",
            deep=True,
            wall_points=5000,
            near_wall_points=8000,
            core_points=8000,
            max_cases=1,
        )
        self.assertEqual(report["gate_result"], "pass")
        self.assertEqual(report["summary"]["passed"], 1)

    def test_existing_pilot_sidecars_pass_integrity_audit(self) -> None:
        report = audit(
            ROOT / "data_wss_pinn/pilot_v1/sampling_manifest.json",
            ROOT / "wss_pinn/configs/pilot_cases.json",
            wall_points=5000,
            near_wall_points=8000,
            core_points=8000,
        )
        self.assertEqual(report["gate_result"], "pass")
        self.assertEqual(report["summary"]["passed"], 3)


if __name__ == "__main__":
    unittest.main()

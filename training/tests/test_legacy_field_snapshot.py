from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import torch
from torch_geometric.data import Data

from training.core.legacy_snapshot import LegacyFieldSnapshotDataset
from training.scripts._figure_utils import _resolve_wall_mask_from_payload
from training.scripts.predict_field import build_prediction_payload


class LegacyFieldSnapshotDatasetTest(unittest.TestCase):
    def test_dataset_reconstructs_static_and_dynamic_tensors(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            cases = root / "cases"
            cases.mkdir()
            x = torch.arange(32, dtype=torch.float32).reshape(2, 16)
            edge_index = torch.tensor([[0, 1], [1, 0]], dtype=torch.long)
            payload = {
                "format": "field_legacy_snapshot_v1",
                "case_name": "slow/CASE_A",
                "x": x,
                "edge_index": edge_index,
                "wall_mask": torch.tensor([True, False]),
                "sample_ids": ["frame-1", "frame-2"],
                "graph_paths": ["old/frame-1.pt", "old/frame-2.pt"],
                "global_cond": torch.tensor([[[0.1] * 6], [[0.2] * 6]]),
                "time_value": torch.tensor([[0.1], [0.2]]),
                "y_true": torch.arange(16, dtype=torch.float32).reshape(2, 2, 4),
                "y_wss_true": torch.arange(16, dtype=torch.float32).reshape(2, 2, 4) + 1,
            }
            torch.save(payload, cases / "slow__CASE_A.pt")
            (root / "manifest.json").write_text(
                json.dumps(
                    {
                        "format": "field_legacy_snapshot_v1",
                        "num_frames": 2,
                        "cases": [
                            {
                                "case_name": "slow/CASE_A",
                                "payload": "cases/slow__CASE_A.pt",
                                "num_frames": 2,
                            }
                        ],
                    }
                ),
                encoding="utf-8",
            )

            dataset = LegacyFieldSnapshotDataset(root)
            self.assertEqual(len(dataset), 2)
            item = dataset[1]
            self.assertTrue(torch.equal(item.x, x))
            self.assertTrue(torch.equal(item.edge_index, edge_index))
            self.assertTrue(torch.equal(item.y, payload["y_true"][1]))
            self.assertTrue(torch.equal(item.y_wss, payload["y_wss_true"][1]))
            self.assertEqual(item.sample_id, "frame-2")
            self.assertEqual(item.case_name, "slow/CASE_A")

    def test_prediction_payload_modes(self) -> None:
        data = Data(
            x=torch.zeros(2, 16),
            edge_index=torch.tensor([[0, 1], [1, 0]], dtype=torch.long),
            y=torch.ones(2, 4),
            y_wss=torch.ones(2, 4) * 2,
            global_cond=torch.zeros(1, 6),
        )
        pred = torch.ones(2, 4) * 3
        wss = torch.ones(2, 4) * 4
        common = dict(
            data=data,
            pred_item=pred,
            wss_pred_item=wss,
            sample_id="frame-1",
            case_name="slow/CASE_A",
            graph_path="old/frame-1.pt",
            wss_target_names=["a", "b", "c", "m"],
            wss_target_frame="global",
        )
        compact = build_prediction_payload(payload_mode="compact", **common)
        full = build_prediction_payload(payload_mode="full", **common)
        self.assertNotIn("x", compact)
        self.assertNotIn("edge_index", compact)
        self.assertNotIn("global_cond", compact)
        self.assertIn("y_true", compact)
        self.assertIn("y_wss_true", compact)
        self.assertIn("x", full)
        self.assertIn("edge_index", full)
        self.assertIn("global_cond", full)
        wall_mask = _resolve_wall_mask_from_payload(compact)
        self.assertEqual(wall_mask.tolist(), [False, False])


if __name__ == "__main__":
    unittest.main()

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from training.scripts.cleanup_field_predictions import cleanup_targets


class CleanupFieldPredictionsTest(unittest.TestCase):
    def test_cleanup_preserves_manifest_and_removes_only_pt(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            target = root / "predictions_test"
            target.mkdir()
            (target / "manifest.json").write_text(
                json.dumps({"num_predictions": 2, "items": []}), encoding="utf-8"
            )
            (target / "a.pt").write_bytes(b"aaa")
            (target / "b.pt").write_bytes(b"bbbb")
            (target / "keep.json").write_text("{}", encoding="utf-8")
            record_path = root / "record.json"

            record = cleanup_targets([target], record_path, apply=True)
            self.assertEqual(record["deleted_files"], 2)
            self.assertEqual(record["deleted_bytes"], 7)
            self.assertTrue((target / "manifest.json").is_file())
            self.assertTrue((target / "keep.json").is_file())
            self.assertTrue((target / "PRUNED.json").is_file())
            self.assertFalse(list(target.glob("*.pt")))
            self.assertTrue(record_path.is_file())


if __name__ == "__main__":
    unittest.main()

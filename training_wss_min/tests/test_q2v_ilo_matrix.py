from __future__ import annotations

import hashlib
import json
import unittest
from pathlib import Path

import numpy as np

from training_wss_min import config as C
from training_wss_min import dataset as D
from training_wss_min.tools import prepare_q2v_ilo_matrix as P
from training_wss_min.tools.export_wss_postview import _resolve_bundle_path


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def require_prepared() -> dict:
    if not P.MANIFEST.is_file():
        raise unittest.SkipTest("Q2V/ILO matrix assets have not been prepared yet")
    return load_json(P.MANIFEST)


class Q2VILOMatrixTests(unittest.TestCase):
    def test_ilo_canonical_before_only_and_postview_path(self) -> None:
        unit = "ILO/CASE_001-0/before"
        self.assertEqual(D.canonical_unit_id(unit), unit)
        canonical, bundle = _resolve_bundle_path(P.DATA_ROOT, unit)
        self.assertEqual(canonical, unit)
        self.assertEqual(bundle, P.DATA_ROOT / unit / "bundle.npz")
        for bad in ("ILO/CASE_001-0/after", "ILO/CASE_001-2/before", "ILO/CASE_001/before"):
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                D.canonical_unit_id(bad)

    def test_stratum_rng_is_deterministic_and_independent(self) -> None:
        a = P.stable_rng(2025, "AG").integers(0, 2**31, size=16)
        b = P.stable_rng(2025, "AG").integers(0, 2**31, size=16)
        c = P.stable_rng(2025, "ILO/0").integers(0, 2**31, size=16)
        d = P.stable_rng(1234, "AG").integers(0, 2**31, size=16)
        self.assertTrue(np.array_equal(a, b))
        self.assertFalse(np.array_equal(a, c))
        self.assertFalse(np.array_equal(a, d))

    def test_prepared_manifest_has_exact_twelve_config_hashes(self) -> None:
        manifest = require_prepared()
        self.assertEqual(manifest["status"], "prepared")
        self.assertEqual(len(manifest["configs"]), 12)
        self.assertEqual({row["family"] for row in manifest["configs"]}, {"data", "architecture"})
        self.assertEqual(sum(row["family"] == "data" for row in manifest["configs"]), 6)
        self.assertEqual(sum(row["family"] == "architecture" for row in manifest["configs"]), 6)
        for row in manifest["configs"]:
            path = Path(row["config"])
            self.assertTrue(path.is_file())
            self.assertEqual(row["sha256"], sha(path))

    def test_promotion_preserves_historical_false_and_accepts_only_before(self) -> None:
        require_prepared()
        promotion = load_json(P.PROMOTION)
        self.assertIs(promotion["training_promotion_authorized"], True)
        self.assertIs(promotion["immutable_audit_training_promotion_authorized"], False)
        self.assertEqual(promotion["counts"], {"ILO_before": 41, "ILO_0": 28, "ILO_1": 13, "ILO_after": 0})
        self.assertEqual(len(promotion["included_units"]), 41)
        self.assertTrue(all(unit.endswith("/before") for unit in promotion["included_units"]))

    def test_split_counts_disjoint_and_before_only(self) -> None:
        require_prepared()
        rows = (
            (P.FIXED_SPLIT, (147, 0, 27, 0)),
            (P.EXTENDED_SPLIT, (138, 0, 36, 0)),
            (P.POOL_SPLIT, (138, 0, 36, 0)),
            (P.POOL_CONTROL_SPLIT, (106, 0, 36, 32)),
            (P.ARCH_SPLIT, (85, 21, 27, 0)),
        )
        for path, counts in rows:
            with self.subTest(path=path.name):
                split = load_json(path)
                train, val, test = (split[f"{part}_cases"] for part in ("train", "val", "test"))
                unused = split.get("unused_cases", [])
                self.assertEqual((len(train), len(val), len(test), len(unused)), counts)
                self.assertEqual(split["expected_counts"], {"train": len(train), "val": len(val), "test": len(test)})
                self.assertFalse(set(train) & set(val) or set(train) & set(test) or set(val) & set(test))
                self.assertFalse(set(unused) & (set(train) | set(val) | set(test)))
                self.assertTrue(all(not unit.startswith("ILO/") or unit.endswith("/before")
                                    for unit in train + val + test + unused))

    def test_five_stratum_quota_and_matched_d3_control(self) -> None:
        require_prepared()
        pool = load_json(P.POOL_SPLIT)
        control = load_json(P.POOL_CONTROL_SPLIT)
        expected_train = {"AG": 61, "AAA/ruputer": 21, "AAA/unruputer": 24, "ILO/0": 22, "ILO/1": 10}
        expected_test = {"AG": 15, "AAA/ruputer": 6, "AAA/unruputer": 6, "ILO/0": 6, "ILO/1": 3}
        self.assertEqual({name: sum(P.stratum(u) == name for u in pool["train_cases"])
                          for name in P.STRATA}, expected_train)
        self.assertEqual({name: sum(P.stratum(u) == name for u in pool["test_cases"])
                          for name in P.STRATA}, expected_test)
        self.assertEqual(control["test_cases"], pool["test_cases"])
        self.assertEqual(control["train_cases"], [u for u in pool["train_cases"] if not u.startswith("ILO/")])
        self.assertEqual(control["unused_cases"], sorted(
            u for u in pool["train_cases"] if u.startswith("ILO/")))

    def test_seed2025_is_not_a_relabeling_of_q2v_test(self) -> None:
        require_prepared()
        old = load_json(P.OLD_SPLIT)
        pool = load_json(P.POOL_SPLIT)
        pool_ag_aaa_test = [u for u in pool["test_cases"] if not u.startswith("ILO/")]
        self.assertNotEqual(set(pool_ag_aaa_test), set(old["test_cases"]))

    def test_stats_are_train_only_or_an_explicit_frozen_train_subset(self) -> None:
        manifest = require_prepared()
        for row in manifest["configs"]:
            with self.subTest(experiment=row["experiment_id"]):
                cfg = C.ExpConfig.from_json(row["config"])
                split = load_json(Path(cfg.data.split_path))
                stats = load_json(Path(cfg.data.wss_stats_path))
                source_split = Path(stats["split_path"])
                self.assertTrue(source_split.is_file())
                self.assertEqual(stats["split_sha256"], sha(source_split))
                self.assertEqual(stats["n_cases"], len(stats["train_units"]))
                self.assertLessEqual(set(stats["train_units"]), set(split["train_cases"]))
                self.assertFalse(set(stats["train_units"]) & set(split["val_cases"] + split["test_cases"]))
                if source_split.resolve() == Path(cfg.data.split_path).resolve():
                    self.assertEqual(stats["train_units"], split["train_cases"])
                if cfg.data.feature_stats_path:
                    feature_path = Path(cfg.data.feature_stats_path)
                    self.assertTrue(feature_path.is_file())
                    feature = load_json(feature_path)
                    self.assertLessEqual(set(cfg.data.input_features) - {"x", "y", "z"}, set(feature))

    def test_data_matrix_reuses_q2v_except_registered_protocol_fields(self) -> None:
        manifest = require_prepared()
        base = load_json(P.Q2V_CONFIG)
        for row in (r for r in manifest["configs"] if r["family"] == "data"):
            cfg = load_json(Path(row["config"]))
            self.assertEqual(cfg["model"], base["model"])
            self.assertEqual(cfg["train"], base["train"])
            self.assertEqual(cfg["eval"], base["eval"])
            for key, value in base["data"].items():
                if key not in {"split_path", "wss_stats_path"}:
                    self.assertEqual(cfg["data"][key], value)

    def test_architecture_factorial_changes_only_width_nsample_and_dev_protocol(self) -> None:
        manifest = require_prepared()
        base = load_json(P.Q2V_CONFIG)
        observed = set()
        for row in (r for r in manifest["configs"] if r["family"] == "architecture"):
            cfg = load_json(Path(row["config"]))
            width = cfg["model"]["width"]
            nsample = tuple(cfg["model"]["sa_nsample"])
            observed.add((nsample[0], width))
            self.assertEqual(nsample, (nsample[0],) * 3)
            for key, value in base["model"].items():
                if key not in {"width", "sa_nsample"}:
                    self.assertEqual(cfg["model"][key], value)
            self.assertEqual(cfg["train"], base["train"])
            self.assertEqual(cfg["eval"], base["eval"])
            self.assertEqual(Path(cfg["data"]["split_path"]), P.ARCH_SPLIT)
            self.assertEqual(Path(cfg["data"]["wss_stats_path"]), P.ARCH_STATS)
            self.assertEqual(Path(cfg["data"]["feature_stats_path"]), P.ARCH_FEATURE_STATS)
        self.assertEqual(observed, {(n, w) for n in (16, 32, 64) for w in (32, 64)})

    def test_arch_dev_keeps_original_test27_untouched(self) -> None:
        require_prepared()
        old = load_json(P.OLD_SPLIT)
        arch = load_json(P.ARCH_SPLIT)
        self.assertEqual(arch["test_cases"], old["test_cases"])
        self.assertEqual(set(arch["train_cases"] + arch["val_cases"]), set(old["train_cases"]))


if __name__ == "__main__":
    unittest.main()

"""AG+AAA v4 分层 split 与 PointNet++ SA3 配置合同。"""

from __future__ import annotations

import hashlib
import json
import unittest
from pathlib import Path

import numpy as np
import torch

from training_wss_min import config as C
from training_wss_min.models import build_model
from training_wss_min.tools.prepare_v4_stratified_protocol import choose_test_groups


REPO = Path(__file__).resolve().parents[2]
SPLIT = REPO / "training/splits/split_AG_AAA_wss_min_v4_stratified_seed1234.json"
STATS = REPO / "data_wss_min/fold_stats/v4/AG_AAA_v4_stratified_train106_global_stats.json"
CONFIGS = (
    REPO / "training_wss_min/configs/pointnetpp_v4/ag_v4_sa3_e2_global_fps2000.json",
    REPO / "training_wss_min/configs/pointnetpp_v4/ag_aaa_v4_locked_sa3_e2_global_fps2000.json",
    REPO / "training_wss_min/configs/pointnetpp_v4/ag_aaa_v4_stratified_sa3_e2_global_fps2000.json",
    REPO / "training_wss_min/configs/pointnet_v4/ag_aaa_v4_stratified_e2_global_fps2000.json",
)
FORBIDDEN = {
    "AAA/ruputer/CHEN_FU", "AAA/ruputer/LIU_YU_MING", "AAA/ruputer/SU_KAI_LI",
    "AAA/ruputer/WANG_SHUN_WEN", "AAA/ruputer/ZHANG_ZAO_SHUAN",
    "AAA/unruputer/CAO_DIAN_HE", "AAA/unruputer/GUO_YU_YING",
    "AAA/unruputer/ZHANG_GUI_HUA", "AG/slow/WANG_DENG_FENG",
}


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


class TestStratifiedSplit(unittest.TestCase):
    def test_counts_exclusions_and_grouping(self):
        split = json.loads(SPLIT.read_text())
        self.assertEqual(split["expected_counts"], {"train": 106, "val": 0, "test": 27})
        self.assertEqual(
            {k: split["counts"][k] for k in (
                "train_AG", "train_AAA_ruputer", "train_AAA_unruputer",
                "test_AG", "test_AAA_ruputer", "test_AAA_unruputer")},
            {"train_AG": 61, "train_AAA_ruputer": 21, "train_AAA_unruputer": 24,
             "test_AG": 15, "test_AAA_ruputer": 6, "test_AAA_unruputer": 6},
        )
        train, test = set(split["train_cases"]), set(split["test_cases"])
        self.assertFalse(train & test)
        self.assertFalse(FORBIDDEN & (train | test))
        pair = {"AG/slow/HOU_SHEN_QIAN", "AG/slow/KANG_XI_MING"}
        self.assertTrue(pair <= train or pair <= test)
        self.assertFalse(split["legacy_AG_test15_locked"])

    def test_stats_are_train_only_and_source_hash_is_frozen(self):
        split, stats = json.loads(SPLIT.read_text()), json.loads(STATS.read_text())
        self.assertEqual(stats["train_units"], split["train_cases"])
        self.assertEqual(stats["n_cases"], 106)
        self.assertEqual(stats["split_sha256"], sha(SPLIT))
        contribution = stats["point_contribution_by_cohort"]
        self.assertAlmostEqual(contribution["AG"]["fraction"] + contribution["AAA"]["fraction"], 1.0)

    def test_exact_quota_selector_is_seed_deterministic(self):
        groups = [[f"x{i}"] for i in range(8)] + [["pair-a", "pair-b"]]
        first = choose_test_groups(groups, 4, np.random.default_rng(1234))
        second = choose_test_groups(groups, 4, np.random.default_rng(1234))
        self.assertEqual(first, second)
        self.assertEqual(sum(map(len, first)), 4)


class TestPointNetPPV4Configs(unittest.TestCase):
    def test_four_config_protocol(self):
        configs = [C.ExpConfig.from_json(path) for path in CONFIGS]
        self.assertEqual([cfg.model.name for cfg in configs].count("pointnetpp"), 3)
        self.assertEqual([cfg.model.name for cfg in configs].count("pointnet"), 1)
        for cfg in configs:
            self.assertEqual(cfg.data.required_frame_version, "stl_landmarks_v4")
            self.assertEqual(cfg.data.wall_n_points, 2000)
            self.assertEqual(cfg.data.sampling, "fps")
            self.assertFalse(cfg.data.resample_each_epoch)
            self.assertFalse(cfg.data.rot_aug)
            self.assertEqual(cfg.train.seed, 1234)
            self.assertEqual(cfg.train.epochs, 400)
            self.assertEqual(cfg.train.selection_rule, "train_loss")
        for cfg in configs[:3]:
            self.assertEqual(cfg.model.sa_ratios, (0.25, 0.25, 0.25))
            self.assertEqual(cfg.model.sa_radius, (0.05, 0.1, 0.2))
            self.assertEqual(cfg.model.sa_nsample, (16, 16, 16))

    def test_sa3_synthetic_two_case_backward(self):
        cfg = C.ExpConfig.from_json(CONFIGS[0])
        model = build_model(cfg.model, C.input_dim(cfg))
        pos = torch.rand(160, 3) * 2 - 1
        x = torch.randn(160, 6)
        batch = torch.tensor([0] * 80 + [1] * 80)
        pred = model(pos, x, batch)
        self.assertEqual(pred.shape, (160,))
        pred.square().mean().backward()


if __name__ == "__main__":
    unittest.main()

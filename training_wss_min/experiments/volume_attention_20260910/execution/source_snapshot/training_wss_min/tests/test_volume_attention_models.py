"""Coarse FFN controls, joint heads and fresh paired initialization for volume tasks."""
from __future__ import annotations

import copy
import json
from pathlib import Path
import tempfile
import unittest

import torch

from training_wss_min import config as C
from training_wss_min.baseline_models import BottleneckTransformer, CapacityMatchedTokenFFN
from training_wss_min.models import build_model
from training_wss_min.paired_initialization import build_paired_model


ROOT = Path(__file__).resolve().parents[2]
VELOCITY = ROOT / "training_wss_min/configs/v5_rerun_20260906/r5v_velocity_qad_s1234.json"


def config(target="velocity", **changes):
    raw = json.loads(VELOCITY.read_text())
    raw["data"]["target"] = target
    raw["data"]["support_n_points"] = 32
    raw["data"]["query_n_points"] = 16
    raw["model"].update(width=8, head_hidden=16, sa_center_counts=[16, 8, 4],
                          sa_nsample=[8, 8, 8], out_dim=3 if target == "velocity" else 1)
    if target == "velocity_pressure":
        raw["model"].update(out_dim=4, output_head="velocity_pressure")
    raw["model"].update(changes)
    return C.ExpConfig.from_dict(raw)


def clouds():
    gen = torch.Generator().manual_seed(37)
    def cloud(counts):
        size = sum(counts)
        return (torch.rand(size, 3, generator=gen) * .2,
                torch.randn(size, 18, generator=gen),
                torch.repeat_interleave(torch.arange(len(counts)), torch.tensor(counts)))
    return cloud((40, 33)), cloud((19, 13))


class VolumeModelTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.old_threads = torch.get_num_threads()
        torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.old_threads)

    def test_old_default_state_rng_and_output_unchanged(self):
        cfg = config()
        raw = cfg.to_dict()
        raw["model"].pop("output_head")
        raw["model"].pop("bottleneck_mode")
        implicit = C.ExpConfig.from_dict(raw)
        torch.manual_seed(1234)
        old = build_model(implicit.model, 18).eval()
        rng = torch.get_rng_state().clone()
        torch.manual_seed(1234)
        explicit = build_model(cfg.model, 18).eval()
        self.assertTrue(torch.equal(rng, torch.get_rng_state()))
        self.assertEqual(old.state_dict().keys(), explicit.state_dict().keys())
        self.assertFalse(any("velocity_head" in key or "bottleneck" in key for key in old.state_dict()))
        explicit.load_state_dict(old.state_dict(), strict=True)
        support, query = clouds()
        with torch.no_grad():
            a = old.forward_support_query(*support, *query, evaluation=True)
            b = explicit.forward_support_query(*support, *query, evaluation=True)
        torch.testing.assert_close(a, b, rtol=0, atol=0)

    def test_pressure_velocity_joint_forward_backward_for_attention_and_ffn(self):
        support, query = clouds()
        for target, dim in (("pressure_mixed", 1), ("velocity", 3), ("velocity_pressure", 4)):
            for mode in ("attention", "token_ffn"):
                with self.subTest(target=target, mode=mode):
                    cfg = config(target, bottleneck_transformer=True, bottleneck_mode=mode,
                                 local_branch=True, local_branch_feature_indices=(14, 15, 16),
                                 local_branch_nsample=(8, 8, 8))
                    model = build_model(cfg.model, 18)
                    pred = model.forward_support_query(*support, *query, evaluation=True)
                    self.assertEqual(pred.shape, (32,) if dim == 1 else (32, dim))
                    pred.square().mean().backward()
                    self.assertTrue(all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None))
                    self.assertEqual(model.support_encode_calls, 1)
                    if dim == 4:
                        self.assertGreater(float(model.velocity_head[-1].weight.grad.abs().sum()), 0)
                        self.assertGreater(float(model.pressure_head[-1].weight.grad.abs().sum()), 0)

    def test_joint_heads_are_independent_and_qad_projections_start_at_zero(self):
        cfg = config("velocity_pressure")
        model = build_model(cfg.model, 18).eval()
        self.assertFalse(hasattr(model, "head"))
        self.assertFalse(hasattr(model, "qad_out"))
        for projection in (model.qad_velocity_out, model.qad_pressure_out):
            self.assertEqual(int(torch.count_nonzero(projection.weight)), 0)
            self.assertEqual(int(torch.count_nonzero(projection.bias)), 0)
        support, query = clouds()
        with torch.no_grad():
            encoded = model.encode_support(*support, evaluation=True)
            before = model.decode_query(encoded, *query)
            model.pressure_head[-1].bias.add_(2)
            after = model.decode_query(encoded, *query)
        torch.testing.assert_close(before[:, :3], after[:, :3], rtol=0, atol=0)
        torch.testing.assert_close(after[:, 3] - before[:, 3], torch.full((32,), 2.0))

    def test_ffn_parameter_match_and_no_token_mixing(self):
        for channels in (64, 256):
            options = dict(channels=channels, in_dim=18, layers=2, heads=8, ffn_ratio=2,
                           dropout=0., pos_enc="input_features", geo_bias=True, residual_scale_init=.001)
            attention = BottleneckTransformer(**options)
            control = BottleneckTransformer(**options, mode="token_ffn")
            self.assertTrue(all(isinstance(layer, CapacityMatchedTokenFFN) for layer in control.layers))
            n_att = sum(p.numel() for p in attention.parameters())
            n_ffn = sum(p.numel() for p in control.parameters())
            self.assertLessEqual(abs(n_ffn / n_att - 1), .01)
            for training in (False, True):
                control.train(training)
                tokens = torch.randn(9, channels, requires_grad=True)
                out = control(torch.randn(9, 3), tokens, torch.tensor([0] * 5 + [1] * 4),
                              feats=torch.randn(9, 18))
                out[0].sum().backward()
                self.assertEqual(int(torch.count_nonzero(tokens.grad[1:])), 0)

    def test_joint_full_query_matches_chunks(self):
        model = build_model(config("velocity_pressure", bottleneck_transformer=True).model, 18).eval()
        support, query = clouds()
        with torch.no_grad():
            model.qad_velocity_out.weight.normal_(std=.1)
            model.qad_pressure_out.weight.normal_(std=.1)
            encoded = model.encode_support(*support, evaluation=True)
            whole = model.decode_query(encoded, *query)
            # Keep all graph IDs represented: chunk each case, preserving its batch ID.
            chunks = torch.cat([model.decode_query(encoded, *(value[start:end] for value in query))
                                for start, end in ((0, 9), (9, 19), (19, 25), (25, 32))])
        torch.testing.assert_close(whole, chunks, rtol=1e-5, atol=1e-6)

    def test_invalid_joint_and_bottleneck_options_rejected(self):
        candidates = [dict(bottleneck_mode="unknown"), dict(bottleneck_mode="token_ffn"),
                      dict(bottleneck_transformer=True, bottleneck_mode="token_ffn", bottleneck_layers=0),
                      dict(output_head="velocity_pressure")]
        for changed in candidates:
            with self.subTest(changed=changed), self.assertRaises(ValueError):
                config(**changed)
        for changed in (dict(output_head="single"), dict(out_dim=3), dict(query_decoder="interpolate"),
                        dict(case_scale_head=True), dict(name="pointnet")):
            with self.subTest(changed=changed), self.assertRaises(ValueError):
                config("velocity_pressure", **changed)
        for group, key, value in (("train", "loss", "huber"), ("train", "loss_pinball_lambda", .2),
                                  ("train", "selection_rule", "field_casebalanced"),
                                  ("data", "timesteps", "random_frame"),
                                  ("eval", "fixed_support", False)):
            raw = config("velocity_pressure").to_dict()
            raw[group][key] = value
            with self.subTest(key=key), self.assertRaises(ValueError):
                C.ExpConfig.from_dict(raw)


class VolumePairedInitializationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.old_threads = torch.get_num_threads()
        torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.old_threads)

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)

    def reference(self, cfg, name="reference.json"):
        path = self.root / name
        cfg.to_json(path)
        return str(path)

    def test_bt_ffn_conditioning_and_wall_branch_preserve_shared_weights_rng(self):
        baseline = config()
        reference_path = self.reference(baseline)
        torch.manual_seed(1234)
        parent = build_model(baseline.model, 18)
        expected_rng = torch.get_rng_state().clone()
        modules = []
        for mode, layers in (("attention", 0), ("attention", 1), ("token_ffn", 1)):
            cfg = config(bottleneck_transformer=True, bottleneck_mode=mode, bottleneck_layers=layers,
                         local_branch=True, local_branch_feature_indices=(14, 15, 16))
            cfg.train.init_reference_config = reference_path
            child, evidence = build_paired_model(cfg)
            self.assertTrue(torch.equal(expected_rng, torch.get_rng_state()))
            self.assertFalse(evidence["trained_checkpoint_loaded"])
            for key, value in parent.state_dict().items():
                torch.testing.assert_close(value, child.state_dict()[key], rtol=0, atol=0)
            modules.append(child)
        for child in modules[1:]:
            for key, value in modules[0].bottleneck.pos.state_dict().items():
                torch.testing.assert_close(value, child.bottleneck.pos.state_dict()[key], rtol=0, atol=0)
            for key, value in modules[0].local_wall_branch.state_dict().items():
                torch.testing.assert_close(value, child.local_wall_branch.state_dict()[key], rtol=0, atol=0)

    def test_multiradius_replaces_only_sa3_and_pairs_middle_scale(self):
        baseline = config()
        reference_path = self.reference(baseline)
        torch.manual_seed(1234)
        parent = build_model(baseline.model, 18)
        modules = []
        for radii, bt, ff in (((.2,), False, False), ((.1, .2, .4), False, False),
                              ((.2,), True, False), ((.1, .2, .4), True, False),
                              ((.2, .2, .2), True, False), ((.1, .2, .4), False, True)):
            cfg = config(multiradius_values=radii, multiradius_bottleneck=bt, multiradius_ffn=ff)
            cfg.train.init_reference_config = reference_path
            child, evidence = build_paired_model(cfg)
            self.assertEqual(evidence["replaced_module_roots"], ["sa.2"])
            for key, value in parent.state_dict().items():
                if not key.startswith("sa.2."):
                    torch.testing.assert_close(value, child.state_dict()[key], rtol=0, atol=0)
            modules.append(child)
        for single, multiple in ((0, 1), (2, 3)):
            for name in ("branches", "contexts"):
                a = getattr(modules[single].sa[-1], name)[0].state_dict()
                b = getattr(modules[multiple].sa[-1], name)[1].state_dict()
                for key in a:
                    torch.testing.assert_close(a[key], b[key], rtol=0, atol=0)
        for key, value in modules[3].sa[-1].state_dict().items():
            torch.testing.assert_close(value, modules[4].sa[-1].state_dict()[key], rtol=0, atol=0)

    def test_joint_chained_reference_reconstructs_actual_fresh_parent(self):
        base = config("velocity_pressure")
        j0 = config("velocity_pressure", local_branch=True, local_branch_feature_indices=(14, 15, 16))
        j0.train.init_reference_config = self.reference(base, "joint_base.json")
        parent, _ = build_paired_model(j0)
        expected_rng = torch.get_rng_state().clone()
        j1 = copy.deepcopy(j0)
        j1.model.bottleneck_transformer = True
        j1.train.init_reference_config = self.reference(j0, "J00.json")
        child, evidence = build_paired_model(j1)
        self.assertIsNotNone(evidence["reference_initialization"])
        self.assertTrue(torch.equal(expected_rng, torch.get_rng_state()))
        for key, value in parent.state_dict().items():
            torch.testing.assert_close(value, child.state_dict()[key], rtol=0, atol=0)

    def test_single_to_joint_head_mapping_preserves_trunk_and_rejects_cycles(self):
        baseline = config()
        joint = config("velocity_pressure")
        joint.train.init_reference_config = self.reference(baseline)
        child, evidence = build_paired_model(joint)
        torch.manual_seed(1234)
        parent = build_model(baseline.model, 18)
        self.assertTrue(all(key.startswith(("head.", "qad_out.")) for key in evidence["discarded_reference_keys"]))
        for key, value in parent.state_dict().items():
            if not key.startswith(("head.", "qad_out.")):
                torch.testing.assert_close(value, child.state_dict()[key], rtol=0, atol=0)
        self.assertEqual(int(torch.count_nonzero(child.qad_velocity_out.weight)), 0)
        cyc = config()
        cyc.train.init_reference_config = str(self.root / "cycle.json")
        self.reference(cyc, "cycle.json")
        with self.assertRaisesRegex(ValueError, "cyclic"):
            build_paired_model(cyc)


if __name__ == "__main__":
    unittest.main()

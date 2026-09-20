"""Exercise every actual matrix config at batch8 x 5000 on a Slurm GPU."""
from __future__ import annotations
import copy
import gc
import json
import os
import subprocess
import sys
import time
from pathlib import Path
import numpy as np
import torch

from training_wss_min import config as C, dataset as D
from training_wss_min.evaluate import predict_case_norm
from training_wss_min.objectives import compute_loss
from training_wss_min.paired_initialization import build_paired_model
from training_wss_min.runtime import seed_all
from training_wss_min.tools.wss_recovery_common import CONFIGS, EXP, ROOT, RUNS, fingerprints, data_fingerprints, save_json


def load_cases(cfg, pairs):
    stats = D.load_wss_stats(cfg.data.wss_stats_path)
    features = D.load_case_feature_table(cfg.data.case_features_path) if cfg.data.case_features_path else {}
    return [D.load_case(cohort, name, stats, target=cfg.data.target,
        target_normalization=cfg.data.target_normalization, data_root=cfg.data.data_root,
        required_frame_version=cfg.data.required_frame_version,
        case_features=features.get(f"{cohort}/{name}"), extra_point_features=C.v6_point_features(cfg),
        point_features_root=cfg.data.point_features_root) for cohort, name in pairs]


def forward(model, batch, training=True):
    return model.forward_support_query(batch["support_pos"].cuda(), batch["support_x"].cuda(),
        batch["support_batch"].cuda(), batch["pos"].cuda(), batch["x"].cuda(), batch["batch"].cuda(),
        unit_ids=batch["unit_ids"], epoch=0, global_seed=1234, evaluation=not training,
        **D.local_model_context(batch, "cuda"))


def main():
    if not os.environ.get("SLURM_JOB_ID") or not torch.cuda.is_available():
        raise RuntimeError("Slurm GPU allocation required")
    torch.set_num_threads(2)
    frozen = fingerprints()
    cfg = C.ExpConfig.from_json(CONFIGS / "E0_s1234.json")
    pairs = D.load_split_cases(cfg.data.split_path, "train")
    chosen = []
    for family in ("AG/", "AAA/", "ILO/"):
        chosen.extend([p for p in pairs if p[0].startswith(family)][:2])
    chosen.extend([p for p in pairs if p not in chosen][:8-len(chosen)])
    evidence = dict(passed=False, slurm_job_id=os.environ["SLURM_JOB_ID"],
                    fingerprints=frozen, arms={}, cli_smoke_passed=False)
    case_cache = {}
    for path in sorted(CONFIGS.glob("*s1234.json")):
        cfg = C.ExpConfig.from_json(path)
        case_key = cfg.data.case_features_path
        if case_key not in case_cache:
            case_cache[case_key] = load_cases(cfg, chosen)
        cases = case_cache[case_key]
        feature_stats = json.loads(Path(cfg.data.feature_stats_path).read_text())
        ds = D.WSSMinDataset(cases, cfg.data, feature_stats, training=True, base_seed=1234)
        batch = D.collate([ds[i] for i in range(8)])
        seed_all(1234)
        model, initial = build_paired_model(cfg)
        model.cuda().train()
        optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.train.lr, weight_decay=cfg.train.weight_decay)
        scaler = torch.amp.GradScaler("cuda")
        torch.cuda.reset_peak_memory_stats()
        started = time.monotonic()
        losses, overflows, components = [], 0, {}
        for attempt in range(12):
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast("cuda", dtype=torch.float16):
                pred = forward(model, batch)
                loss = compute_loss(pred, batch, cfg.train, "cuda",
                                    D.load_wss_stats(cfg.data.wss_stats_path), components=components)
            if not torch.isfinite(loss) or not torch.isfinite(pred).all():
                raise RuntimeError(f"nonfinite forward {path.name}: {components}")
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            finite = all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)
            if finite:
                norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
                if not torch.isfinite(norm):
                    raise RuntimeError(f"nonfinite norm {path.name}")
                losses.append(float(loss.detach()))
            else:
                overflows += 1
            scaler.step(optimizer)
            scaler.update()
            if len(losses) == 3:
                break
        if len(losses) < 3:
            raise RuntimeError(f"AMP never achieved three finite updates: {path.name}")
        if cfg.train.loss_local_diff_lambda and (components["local_difference_pair_count"] <= 0
                or components["local_difference_cases_without_pairs"] != 0):
            raise RuntimeError(f"local loss did not supervise every case: {components}")
        if cfg.train.loss_pairwise_rank_lambda and (components["pairwise_ranking_pair_count"] <= 0
                or components["pairwise_ranking_cases_without_pairs"] != 0):
            raise RuntimeError(f"ranking did not supervise every case: {components}")
        model.eval()
        # Real full-wall inference, then a different query chunk size with the
        # same support and nonzero trained patch projection.
        p1 = predict_case_norm(model, cases[0], cfg.data.input_features, feature_stats, "cuda", cfg)
        cfg2 = copy.deepcopy(cfg)
        cfg2.eval.query_chunk_size = 4096
        p2 = predict_case_norm(model, cases[0], cfg2.data.input_features, feature_stats, "cuda", cfg2)
        np.testing.assert_allclose(p1, p2, rtol=3e-5, atol=3e-5)
        if not np.isfinite(p1).all() or len(p1) != len(cases[0]["pos"]):
            raise RuntimeError("invalid full-wall output")
        torch.cuda.synchronize()
        evidence["arms"][path.name] = dict(parameters=sum(p.numel() for p in model.parameters()),
            peak_memory_mib=torch.cuda.max_memory_allocated()/2**20, seconds=time.monotonic()-started,
            losses=losses, amp_initial_overflows=overflows,
            loss_components={k: float(v) for k,v in components.items()},
            fullwall_points=len(p1), chunk_max_abs_diff=float(np.max(np.abs(p1-p2))),
            initialization=initial, units=batch["unit_ids"])
        save_json(EXP / "runtime_preflight_progress.json", evidence)
        print(path.name, evidence["arms"][path.name]["peak_memory_mib"], losses, flush=True)
        del model, optimizer, scaler, batch, pred, loss, ds
        gc.collect()
        torch.cuda.empty_cache()
    # Exercise the real entrypoints, filesystem serialization, DataLoader,
    # checkpoints and saved predictions using the richest combined config.
    smoke = EXP / "smoke" / os.environ["SLURM_JOB_ID"]
    smoke.mkdir(parents=True, exist_ok=True)
    raw = json.loads((CONFIGS / "C4_E3_s1234.json").read_text())
    raw["name"] = f"{EXP.name}_smoke/job_{os.environ['SLURM_JOB_ID']}"
    split = json.loads(Path(raw["data"]["split_path"]).read_text())
    # load_split_cases accepts entries in cohort/case format.
    split = {"train_cases": [f"{a}/{b}" for a,b in chosen], "val_cases": [],
             "test_cases": [f"{a}/{b}" for a,b in D.load_split_cases(raw['data']['split_path'], 'test')[:2]]}
    save_json(smoke / "split.json", split)
    raw["data"]["split_path"] = str(smoke / "split.json")
    raw["train"].update(epochs=2, min_epoch=2, eval_every=2)
    raw["model"]["local_branch_directional"] = True
    raw["data"]["local_geometry"] = True
    raw["train"].update(loss_local_diff_lambda=.05, local_diff_max_pairs=256,
                        loss_pairwise_rank_lambda=.05, pairwise_rank_max_pairs=256, pairwise_rank_margin=.05)
    save_json(smoke / "config.json", raw)
    for command in ([sys.executable, "-u", "-m", "training_wss_min.train", "--config", str(smoke / "config.json")],
                    [sys.executable, "-u", "-m", "training_wss_min.evaluate", "--run-dir", str(RUNS/raw["name"]),
                     "--checkpoint", "best", "--partitions", "test", "--allow-test", "--no-plots", "--save-predictions"]):
        subprocess.run(command, cwd=ROOT, check=True)
    evidence["cli_smoke_passed"] = True
    evidence["smoke_run"] = str(RUNS/raw["name"])
    evidence["data_fingerprints"] = data_fingerprints()
    if fingerprints() != frozen:
        raise RuntimeError("sources/config changed during GPU preflight")
    evidence["passed"] = True
    save_json(EXP / "runtime_preflight.json", evidence)
    print("PREFLIGHT PASSED", flush=True)


if __name__ == "__main__":
    main()

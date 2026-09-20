"""Real-data batch8/5000-query preflight; run on a Slurm allocated GPU."""
from __future__ import annotations

import argparse
import copy
import gc
import json
import os
import shutil
import time

import torch

from training_wss_min import config as C, dataset as D
from training_wss_min.models import build_model
from training_wss_min.objectives import compute_loss
from training_wss_min.paired_initialization import build_paired_model
from training_wss_min.runtime import seed_all
from wss_v5 import contract as WC
from training_wss_min.tools.volume_attention_common import CONFIGS, EXP, OLD, ROOT, RUNS, fingerprints, manifest, reference_dir, save_json, sha256, stamp


def chosen_pairs(cfg):
    pairs = D.load_split_cases(cfg.data.split_path, "train")
    chosen = []
    for family in ("AG", "AAA", "ILO"):
        chosen.extend([p for p in pairs if p[0].startswith(family + "/")][:2])
    chosen.extend([p for p in pairs if p not in chosen][:8 - len(chosen)])
    return chosen


def load_batch(cfg):
    stats = D.load_wss_stats(cfg.data.wss_stats_path)
    cases = [D.load_case(cohort, name, stats, target=cfg.data.target,
                        target_normalization=cfg.data.target_normalization,
                        data_root=cfg.data.data_root, required_frame_version=cfg.data.required_frame_version)
             for cohort, name in chosen_pairs(cfg)]
    feature_stats = json.loads(open(cfg.data.feature_stats_path).read())
    ds = D.WSSMinDataset(cases, cfg.data, feature_stats, training=True, base_seed=1234)
    batch = D.collate([ds[i] for i in range(8)])
    return {k: v.cuda() if isinstance(v, torch.Tensor) else v for k, v in batch.items()}


def forward(model, batch, training=True):
    return model.forward_support_query(batch["support_pos"], batch["support_x"], batch["support_batch"],
                                       batch["pos"], batch["x"], batch["batch"], unit_ids=batch["unit_ids"],
                                       epoch=0, global_seed=1234, evaluation=not training)


def clone_references():
    records = {}
    for prefix, name in OLD.items():
        src, dest = RUNS / name, reference_dir(prefix)
        dest.mkdir(parents=True, exist_ok=True)
        for f in ("ckpt_best.pt", "ckpt_last.pt", "config.json", "feature_stats.json", "wss_global_stats.json", "target_normalization.json"):
            if (dest / f).exists() and sha256(src / f) != sha256(dest / f):
                raise ValueError(f"Conflicting historical reference copy: {dest/f}")
            if not (dest / f).exists():
                shutil.copy2(src / f, dest / f)
        # Original metrics remain separate from the new evaluation.
        for ck in ("best", "last"):
            p = src / "eval" / f"ckpt_{ck}" / "metrics.json"
            out = dest / "original_eval" / f"ckpt_{ck}" / "metrics.json"
            out.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(p, out)
        records[f"R5{prefix}"] = dict(source=str(src), copy=str(dest),
                                    checkpoint_sha256={ck: sha256(src / f"ckpt_{ck}.pt") for ck in ("best", "last")})
    return records


def data_provenance(cfg):
    records = {}
    for part in ("train", "test"):
        for cohort, name in D.load_split_cases(cfg.data.split_path, part):
            for filename in ("bundle.npz", "volume.npz"):
                path = C.PROJECT_ROOT / cfg.data.data_root / cohort / name / filename
                stat = path.stat()
                records[str(path)] = dict(bytes=stat.st_size, mtime_ns=stat.st_mtime_ns, sha256=sha256(path))
            source=WC.case_dir(f"{cohort}/{name}",WC.SNAPSHOT_ROOT)/"case.h5"
            stat=source.stat()
            records[str(source)]=dict(bytes=stat.st_size,mtime_ns=stat.st_mtime_ns,
                                     scope="existing geometry/volume_static used only in diagnostics; container stat fingerprint")
    for path in (cfg.data.split_path, cfg.data.wss_stats_path, cfg.data.feature_stats_path):
        records[str(path)] = dict(sha256=sha256(path))
    return records


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=3)
    args = ap.parse_args()
    if not os.environ.get("SLURM_JOB_ID"):
        raise RuntimeError("Submit this real-data GPU preflight with sbatch")
    torch.set_num_threads(2)
    hashes = fingerprints()
    matrix = manifest()
    records, batch_cache = {}, {}
    references = clone_references()
    for arm in matrix["arms"]:
        cfg = C.ExpConfig.from_json(CONFIGS / arm["config"])
        if cfg.data.target not in batch_cache:
            batch_cache[cfg.data.target] = load_batch(cfg)
        batch = batch_cache[cfg.data.target]
        seed_all(1234)
        if cfg.train.init_reference_config:
            model, evidence = build_paired_model(cfg)
        else:
            model = build_model(cfg.model, C.input_dim(cfg))
            evidence = {"mode": "fresh_baseline", "trained_checkpoint_loaded": False}
        initial_shared = {k: v.detach().clone() for k,v in model.state_dict().items()}
        model = model.cuda()
        optimizer = torch.optim.AdamW(model.parameters(), lr=.001, weight_decay=.0001)
        scaler = torch.amp.GradScaler("cuda")
        torch.cuda.reset_peak_memory_stats()
        start = time.monotonic()
        losses = []
        for step in range(args.steps):
            model.train(); optimizer.zero_grad(set_to_none=True)
            with torch.autocast("cuda", dtype=torch.float16):
                pred = forward(model, batch)
                loss = compute_loss(pred, batch, cfg.train, "cuda")
            if not torch.isfinite(loss) or not torch.isfinite(pred).all():
                raise ValueError(f"{arm['id']} nonfinite forward")
            scaler.scale(loss).backward(); scaler.unscale_(optimizer)
            if not all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None):
                raise ValueError(f"{arm['id']} nonfinite gradient")
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
            scaler.step(optimizer); scaler.update()
            losses.append(float(loss.detach()))
        model.eval()
        with torch.no_grad():
            result = forward(model, batch, False)
            assert result.shape == batch["y"].shape
        torch.cuda.synchronize()
        records[arm["id"]] = dict(parameters=sum(p.numel() for p in model.parameters()),
                                  peak_memory_mib=torch.cuda.max_memory_allocated()/1024**2,
                                  runtime_seconds=time.monotonic()-start, step_losses=losses,
                                  output_shape=list(result.shape), initialization=evidence,
                                  unit_ids=batch["unit_ids"])
        print(stamp(), arm["id"], records[arm["id"]]["parameters"], records[arm["id"]]["peak_memory_mib"], losses, flush=True)
        del model, optimizer, pred, result, loss, initial_shared
        gc.collect(); torch.cuda.empty_cache()
    cfg = C.ExpConfig.from_json(CONFIGS / "P00_s1234.json")
    provenance = data_provenance(cfg)
    if fingerprints() != hashes:
        raise RuntimeError("Source/config changed during preflight")
    save_json(EXP / "runtime_preflight.json", dict(passed=True, job_id=os.environ["SLURM_JOB_ID"],
              timestamp=stamp(), source_sha256=hashes, references=references, data=provenance, arms=records))


if __name__ == "__main__":
    main()

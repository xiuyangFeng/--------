"""CPU-only replay of historical M2 data preparation and paired initialization."""
from __future__ import annotations
import hashlib
import importlib.util
import json
import os
import pickle
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader
from training_wss_min import config as C, dataset as D
from training_wss_min.models import build_model
from training_wss_min.paired_initialization import build_paired_model, _state_hash
from training_wss_min.runtime import seed_all
from training_wss_min.tools.m2_optimization_common import EXP, ANCHOR_CONFIG, ANCHOR_RUN, save_json, sha


def rng_hashes():
    return dict(cpu=hashlib.sha256(torch.get_rng_state().numpy().tobytes()).hexdigest(),
                numpy=hashlib.sha256(pickle.dumps(np.random.get_state())).hexdigest(),
                initial_seed=torch.initial_seed())


def tensor_digest(batch):
    digest = hashlib.sha256()
    for key in sorted(batch):
        value = batch[key]
        if torch.is_tensor(value):
            digest.update(key.encode()); digest.update(value.numpy().tobytes())
        elif isinstance(value, list) and value and torch.is_tensor(value[0]):
            digest.update(key.encode())
            for tensor in value:
                digest.update(tensor.numpy().tobytes())
        else:
            digest.update(json.dumps([key, value], sort_keys=True).encode())
    return digest.hexdigest()


def loader_for(cases, cfg, stats):
    dataset = D.WSSMinDataset(cases, cfg.data, stats, training=True, base_seed=cfg.train.seed)
    generator = torch.Generator().manual_seed(cfg.train.seed)
    loader = DataLoader(dataset, batch_size=cfg.train.batch_cases, shuffle=True,
        collate_fn=D.collate, num_workers=cfg.data.num_workers, drop_last=False,
        persistent_workers=bool(cfg.data.persistent_workers) and cfg.data.num_workers > 0,
        generator=generator, worker_init_fn=D.worker_init_fn if cfg.data.num_workers > 0 else None)
    return dataset, loader, generator


def main():
    assert os.environ.get("CUDA_VISIBLE_DEVICES") == "-1" and not torch.cuda.is_available()
    torch.set_num_threads(2)
    snapshot = C.PROJECT_ROOT / "training_wss_min/experiments/v6_followup_20260909/source_snapshot_13205/training_wss_min"
    spec = importlib.util.spec_from_file_location("training_wss_min._rng_historical_baseline", snapshot / "baseline_models.py")
    old_models = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(old_models)
    historical = C.ExpConfig.from_json(ANCHOR_CONFIG)
    current = C.ExpConfig.from_json(C.PROJECT_ROOT / "training_wss_min/configs/m2_optimization_20260909/MO0_s1234.json")
    assert historical.data == current.data and historical.eval == current.eval
    result = dict(cpu_only=True, historical_snapshot=str(snapshot), global_rng_stages={}, checks={})
    for name in ("dataset.py", "runtime.py", "surface.py"):
        result["checks"][name + "_byte_identical"] = sha(snapshot / name) == sha(C.PROJECT_ROOT / "training_wss_min" / name)
    seed_all(1234)
    result["global_rng_stages"]["after_seed"] = rng_hashes()
    stats = D.load_wss_stats(historical.data.wss_stats_path)
    cases = D.load_partition(historical.data.split_path, "train", stats, strict=True,
        target=historical.data.target, target_normalization=historical.data.target_normalization,
        data_root=historical.data.data_root, required_frame_version=historical.data.required_frame_version,
        case_features_path=historical.data.case_features_path, timesteps=historical.data.timesteps,
        waveform_path=historical.data.waveform_path, extra_point_features=C.v6_point_features(historical),
        point_features_root=historical.data.point_features_root)
    result["global_rng_stages"]["after_load_train138"] = rng_hashes()
    feature_stats = D.compute_feature_stats(cases, historical.data.input_features, historical.data.curvature_transform)
    quantiles = D.compute_train_weight_quantiles(cases)
    result["global_rng_stages"]["after_training_statistics"] = rng_hashes()
    assert feature_stats == json.loads((ANCHOR_RUN / "feature_stats.json").read_text())
    assert quantiles == json.loads((ANCHOR_RUN / "weight_quantiles.json").read_text())
    old_dataset, old_loader, old_generator = loader_for(cases, historical, feature_stats)
    result["global_rng_stages"]["after_dataset_and_loader_construction"] = rng_hashes()
    old_model = old_models.build_baseline_model(historical.model, C.input_dim(historical))
    old_postbuild = rng_hashes()
    result["global_rng_stages"]["historical_actual_post_model"] = old_postbuild
    old_initial_state = {key: value.clone() for key, value in old_model.state_dict().items()}
    old_state_hash = _state_hash(old_initial_state)
    del old_model
    records = []
    for epoch in (0, 1):
        old_dataset.set_epoch(epoch)
        epoch_records = []
        for batch in old_loader:
            epoch_records.append(dict(unit_ids=batch["unit_ids"], sha256=tensor_digest(batch)))
        records.append(epoch_records)
    old_data_generator = hashlib.sha256(old_generator.get_state().numpy().tobytes()).hexdigest()
    # Reproduce new main's actual order: seed -> deterministic data prep ->
    # construct Dataset/DataLoader -> paired helper. Loading/stats consumed no RNG.
    seed_all(1234)
    new_dataset, new_loader, new_generator = loader_for(cases, current, feature_stats)
    result["global_rng_stages"]["new_pre_paired_model"] = rng_hashes()
    model, evidence = build_paired_model(current)
    result["global_rng_stages"]["paired_post_model"] = rng_hashes()
    differences = [key for key, value in old_initial_state.items() if not torch.equal(value, model.state_dict()[key])]
    result["historical_actual_initial_state_sha256"] = old_state_hash
    result["new_actual_initial_state_sha256"] = _state_hash(model.state_dict())
    result["checks"]["actual_model_parameters_and_buffers_exact"] = not differences
    result["checks"]["actual_post_model_cpu_numpy_rng_exact"] = rng_hashes() == old_postbuild
    result["initial_state_differences"] = differences
    saved_init = json.loads((C.RUNS_ROOT / current.name / "initialization.json").read_text())
    result["checks"]["actual_initial_state_matches_completed_mo0_evidence"] = old_state_hash == saved_init["initial_state_sha256"]
    result["checks"]["actual_post_model_cpu_rng_matches_completed_mo0_evidence"] = old_postbuild["cpu"] == saved_init["post_reference_cpu_rng_sha256"]
    mismatches = []
    for epoch in (0, 1):
        new_dataset.set_epoch(epoch)
        for index, batch in enumerate(new_loader):
            record = dict(unit_ids=batch["unit_ids"], sha256=tensor_digest(batch))
            if record != records[epoch][index]:
                mismatches.append(dict(epoch=epoch, batch=index))
    result["checks"]["first_two_epochs_all_36_batches_identical"] = not mismatches
    result["checks"]["dataloader_generator_state_identical"] = old_data_generator == hashlib.sha256(new_generator.get_state().numpy().tobytes()).hexdigest()
    result["batch_mismatches"] = mismatches
    result["first_two_epochs_batch_hashes"] = records
    before = result["global_rng_stages"]["after_seed"]
    result["checks"]["all_pre_model_data_preparation_preserves_global_rng"] = all(
        result["global_rng_stages"][key] == before for key in
        ("after_load_train138", "after_training_statistics", "after_dataset_and_loader_construction", "new_pre_paired_model"))
    result["cuda_scope"] = "No CUDA execution here. Historical preprocessing/model construction are CPU-only; GPU RNG draws first occur during forward stochastic depth. CPU evidence excludes an erroneous pre-model RNG reset but does not prove CUDA training determinism."
    result["passed"] = all(result["checks"].values())
    save_json(EXP / "initialization_rng_diagnostic.json", result)
    print(json.dumps({"passed":result["passed"],"checks":result["checks"],
                      "global_rng_stages":result["global_rng_stages"],
                      "historical_actual_initial_state_sha256":old_state_hash},ensure_ascii=False,indent=2), flush=True)


if __name__ == "__main__":
    main()

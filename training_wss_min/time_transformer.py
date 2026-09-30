#!/usr/bin/env python3
"""V5.1 frozen-3D-encoder temporal attention experiment, isolated v2 contract.

All arms share 60 input channels, weights at initialization, fixed per-case
context samples and training draws. Raw arms mask query_x after normalization;
the no-cross-time control uses the identical Transformer with a diagonal mask.
The pretrained 3D encoder is represented by fold-matched cached query features.
No old training/evaluation module imports this experimental entry point.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn

from . import time_adapter_probe as TA

SCHEMA = "time_transformer_v51_v2"
N_FRAMES = 81
INPUT_DIMS = {"query_x": 32, "raw": 27, "peak_ln": 1}
DEFAULTS = {
    "schema": SCHEMA, "name": None, "experiment": None, "arm": None, "fold": None,
    "seed": 1234, "title": "", "cache_dir": None, "split_path": None, "data_root": None,
    "frame_stats_path": None, "waveform_path": None, "out_dir": None, "peak_index": 21,
    "inputs": list(INPUT_DIMS),
    "architecture": {"temporal": "transformer", "d_model": 64, "nhead": 4, "num_layers": 2,
                     "dim_feedforward": 128, "dropout": 0.0, "spatial_hidden": 128,
                     "decoder_hidden": [128, 128], "global_pool": "mean"},
    "train": {"epochs": 60, "steps_per_epoch": 30, "batch_cases": 4, "query_points": 1024,
              "train_pool_points": 8192, "stats_points": 4096, "context_points": 1024,
              "lr": 1e-3, "weight_decay": 1e-4, "warmup_steps": 100,
              "min_lr_ratio": 0.01, "grad_clip": 1.0, "selection": "last",
              "allow_tf32": True, "log_every_epochs": 1},
    "eval": {"partition": "test", "chunk_points": 4096}, "notes": "",
}


def load_config(path):
    raw = json.loads(Path(path).read_text())
    if raw.get("schema") != SCHEMA:
        raise ValueError(f"expected {SCHEMA}; old v1 runs are audit-only")
    cfg = TA._merge(DEFAULTS, raw)
    for key in ("name", "experiment", "arm", "fold", "cache_dir", "split_path", "data_root",
                "frame_stats_path", "waveform_path", "out_dir"):
        if cfg[key] is None:
            raise ValueError(f"missing {key}")
    if not cfg['inputs'] or len(set(cfg['inputs'])) != len(cfg['inputs']) or set(cfg['inputs']) - set(INPUT_DIMS):
        raise ValueError("invalid or duplicate inputs")
    a, t = cfg['architecture'], cfg['train']
    if a['temporal'] not in ('transformer', 'frame_mlp') or a['global_pool'] != 'mean':
        raise ValueError("unsupported temporal/pooling mode")
    if a['d_model'] % a['nhead'] or a['dropout'] != 0:
        raise ValueError("d_model must divide by nhead; paired design requires dropout=0")
    if t['selection'] != 'last' or cfg['eval']['partition'] != 'test':
        raise ValueError("only last checkpoint / existing cv3 holdout are supported")
    for key in ('epochs','steps_per_epoch','batch_cases','query_points','train_pool_points','stats_points','context_points'):
        if int(t[key]) < 1:
            raise ValueError(f"train.{key} must be positive")
    return cfg


def _seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def sample_indices(uid, n, count, seed, stream):
    digest = hashlib.sha256(f'{uid}|{seed}|{stream}'.encode()).digest()
    rng = np.random.default_rng(int.from_bytes(digest[:8], 'little'))
    return np.sort(rng.choice(n, min(n, count), replace=False)).astype(np.int64)


def _load_fold(cfg):
    # Existing shared contract checks fold-matched cache and train-only stats.
    f = TA.FoldData({**cfg, 'phase_features': list(TA.PHASE_KEYS)})
    if set(f.train_ids) & set(f.test_ids):
        raise ValueError('train/holdout case overlap')
    if len(f.steps) != N_FRAMES or not np.isfinite(f.mu).all() or not (f.sd > 0).all():
        raise ValueError('invalid frame statistics')
    manifest = f.cache_manifest
    if manifest.get('split_sha256') != TA.sha256_file(cfg['split_path']):
        raise ValueError('cache split fingerprint drift')
    for part, ids in [('train', f.train_ids), ('test', f.test_ids)]:
        for uid in ids:
            if manifest['cases'][uid]['partition'] != part:
                raise ValueError(f'{uid}: cache partition mismatch')
    return dict(train_ids=f.train_ids, test_ids=f.test_ids, steps=f.steps,
                mu=f.mu, sd=f.sd, phase=f.phase.astype(np.float32), q_norm=f.q_norm,
                floor=f.floor, eps=f.eps, peak_index=f.peak_index, manifest=manifest)


def _load_case(cfg, uid, fold, training):
    with np.load(Path(cfg['cache_dir']) / f'{TA.case_key(uid)}.npz', allow_pickle=False) as z:
        if str(z['unit_id']) != uid or int(z['peak_index']) != fold['peak_index']:
            raise ValueError(f'{uid}: cache identity/peak mismatch')
        expected_part = 'train' if training else 'test'
        if str(z['partition']) != expected_part:
            raise ValueError(f'{uid}: cache partition mismatch')
        n = len(z['peak_ln'])
        if n != fold['manifest']['cases'][uid]['n_points']:
            raise ValueError(f'{uid}: cache point count drift')
        indices = sample_indices(uid, n, cfg['train']['train_pool_points'], cfg['seed'], 'query_pool') if training else np.arange(n)
        ctx = sample_indices(uid, n, cfg['train']['context_points'], cfg['seed'], 'context')
        case = {'unit_id': uid, 'idx': indices, 'context_idx': ctx, 'context': {}}
        for key in INPUT_DIMS:
            values = z[key]
            if values.reshape(n, -1).shape[1] != INPUT_DIMS[key]:
                raise ValueError(f'{uid}: {key} width mismatch')
            dtype = np.float64 if key == 'peak_ln' else np.float32
            case[key] = values[indices].astype(dtype)
            case['context'][key] = values[ctx].astype(dtype)
    with np.load(Path(cfg['data_root']) / uid / 'bundle.npz', allow_pickle=False) as z:
        if not np.array_equal(z['steps'], fold['steps']) or int(z['peak_step']) != int(fold['steps'][fold['peak_index']]):
            raise ValueError(f'{uid}: physical frame steps or peak differ')
        wall = z['wall_wss']
        if wall.shape != (N_FRAMES, n) or not np.isfinite(wall).all():
            raise ValueError(f'{uid}: raw label shape/finite mismatch')
        tau = wall[:, indices].astype(np.float64)
    if training:
        case['ln_y'] = TA.ln_floor(tau, fold['floor'], fold['eps']).T.astype(np.float32)
    else:
        # Raw Pa truth is never reconstructed from log/floored labels.
        case['tau_raw'] = tau
    return case


class TemporalFieldModel(nn.Module):
    def __init__(self, input_dim, arch, phase_dim=4):
        super().__init__()
        d, h = int(arch['d_model']), int(arch['spatial_hidden'])
        self.temporal_kind = arch['temporal']
        self.point = nn.Sequential(nn.Linear(input_dim, h), nn.GELU(), nn.Linear(h, d), nn.LayerNorm(d))
        self.phase = nn.Sequential(nn.Linear(phase_dim, d), nn.GELU(), nn.Linear(d, d))
        layer = nn.TransformerEncoderLayer(d_model=d, nhead=int(arch['nhead']),
                    dim_feedforward=int(arch['dim_feedforward']), dropout=float(arch['dropout']),
                    activation='gelu', batch_first=True, norm_first=True)
        self.temporal = nn.TransformerEncoder(layer, num_layers=int(arch['num_layers']), enable_nested_tensor=False)
        layers, last = [], 2*d
        for width in arch['decoder_hidden']:
            layers += [nn.Linear(last, width), nn.GELU()]
            last = width
        layers += [nn.Linear(last, 1)]
        self.decoder = nn.Sequential(*layers)
        nn.init.zeros_(self.decoder[-1].weight)
        nn.init.zeros_(self.decoder[-1].bias)

    def temporal_context(self, point_feat, phase):
        seq = point_feat.mean(1)[:, None, :] + self.phase(phase)[None, :, :]
        mask = None
        if self.temporal_kind == 'frame_mlp':
            # Identical QKV/FFN/normalization/parameter count; only off-diagonal interaction removed.
            mask = ~torch.eye(len(phase), dtype=torch.bool, device=phase.device)
        return self.temporal(seq, mask=mask)

    def decode(self, point_feat, z, peak_index):
        b, p, d = point_feat.shape
        f = z.shape[1]
        x = torch.cat([point_feat[:, :, None, :].expand(-1,-1,f,-1),
                       z[:, None, :, :].expand(-1,p,-1,-1)], -1)
        raw = self.decoder(x).squeeze(-1)
        return raw - raw[:, :, peak_index:peak_index+1]

    def forward(self, x, phase, peak_index, context=None):
        pf = self.point(x)
        z = self.temporal_context(pf, phase) if context is None else context
        return self.decode(pf, z, peak_index)


def _canonical(c, idx):
    return np.concatenate([c[k][idx].reshape(len(idx), -1) for k in INPUT_DIMS], axis=1)


def _stats(pool, max_points, seed):
    # Same 60-D train-only stats for every arm, irrespective of the input mask.
    blocks = []
    for c in pool:
        idx = sample_indices(c['unit_id'], len(c['peak_ln']), max_points, seed, 'input_stats')
        blocks.append(_canonical(c, idx))
    arr = np.concatenate(blocks)
    mean, sd = arr.mean(0), arr.std(0)
    return mean.astype(np.float32), np.where(sd > 1e-6, sd, 1.).astype(np.float32)


def _input(c, idx, inputs, mean, std):
    x = ((_canonical(c, idx) - mean) / std).astype(np.float32)
    offset = 0
    for key, width in INPUT_DIMS.items():
        if key not in inputs:
            x[:, offset:offset+width] = 0
        offset += width
    return x


def _lr(step, total, warm, floor):
    if step < warm:
        return (step+1)/max(warm,1)
    p = min((step-warm)/max(total-warm-1,1),1.)
    return floor + (1-floor)*.5*(1+math.cos(math.pi*p))


def _context_input(c, cfg, mean, std):
    count = len(c['context_idx'])
    return _input(c['context'], np.arange(count), cfg['inputs'], mean, std)


def _batch(pool, ids, cfg, mean, std, rng):
    xs, ys, peaks, contexts = [], [], [], []
    q = cfg['train']['query_points']
    for i in ids:
        c = pool[i]
        idx = rng.choice(len(c['peak_ln']), q, replace=len(c['peak_ln']) < q)
        xs.append(_input(c, idx, cfg['inputs'], mean, std))
        ys.append(c['ln_y'][idx])
        peaks.append(c['peak_ln'][idx].astype(np.float32))
        contexts.append(_context_input(c, cfg, mean, std))
    return tuple(torch.from_numpy(np.stack(a)) for a in (xs, ys, peaks, contexts))


@torch.no_grad()
def _predict_case(model, c, phase, mean, std, fold, cfg, device):
    cx = torch.from_numpy(_context_input(c, cfg, mean, std)[None]).to(device)
    context = model.temporal_context(model.point(cx), phase)
    n, pk = len(c['peak_ln']), fold['peak_index']
    predictions = []
    chunk = cfg['eval']['chunk_points']
    for s in range(0, n, chunk):
        idx = np.arange(s, min(n,s+chunk))
        x = torch.from_numpy(_input(c,idx,cfg['inputs'],mean,std)[None]).to(device)
        delta = model(x, phase, pk, context=context).squeeze(0).double().cpu().numpy()
        # Keep the cached donor anchor / train-fold base in FP64 at evaluation.
        base = TA.tnull_base(c['peak_ln'][idx], fold['mu'], fold['sd'], pk)
        predictions.append((base + delta).T)
    return np.concatenate(predictions, axis=1)


def state_hash(model):
    h = hashlib.sha256()
    for key, value in model.state_dict().items():
        h.update(key.encode()); h.update(value.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


def run(config_path, smoke=False):
    cfg = load_config(config_path)
    out = Path(cfg['out_dir'])
    if out.exists() and any(out.iterdir()):
        raise FileExistsError(f'{out}: refusing to overwrite existing artifacts')
    if not torch.cuda.is_available():
        raise RuntimeError('This node04 training protocol requires CUDA; CPU fallback forbidden')
    out.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    execution = TA.execution_record()
    execution['gpu_uuid'] = os.environ.get('WSS_GPU_UUID', os.environ.get('CUDA_VISIBLE_DEVICES'))
    execution['interpreter'] = __import__('sys').executable
    execution['pid'] = os.getpid()
    cfg_digest = TA.sha256_file(config_path)
    code_hashes = {**TA.code_fingerprint(), 'time_transformer.py': TA.sha256_file(__file__)}
    _seed_all(cfg['seed'])
    device = torch.device('cuda:0')
    tc = cfg['train']
    torch.backends.cuda.matmul.allow_tf32 = tc['allow_tf32']
    torch.backends.cudnn.allow_tf32 = tc['allow_tf32']
    fold = _load_fold(cfg)
    train_ids = fold['train_ids'][:max(tc['batch_cases'], 2)] if smoke else fold['train_ids']
    test_ids = fold['test_ids'][:1] if smoke else fold['test_ids']
    pool = []
    for i, uid in enumerate(train_ids):
        pool.append(_load_case(cfg,uid,fold,True))
        print(f'[load] train {i+1}/{len(train_ids)} {uid}', flush=True)
    mean, std = _stats(pool, tc['stats_points'], cfg['seed'])
    model = TemporalFieldModel(sum(INPUT_DIMS.values()), cfg['architecture']).to(device)
    init_hash = state_hash(model)
    phase = torch.from_numpy(fold['phase']).to(device)
    mu = torch.tensor(fold['mu'],dtype=torch.float32,device=device)
    sd = torch.tensor(fold['sd'],dtype=torch.float32,device=device)
    pk = fold['peak_index']
    opt = torch.optim.AdamW(model.parameters(),lr=tc['lr'],weight_decay=tc['weight_decay'])
    total = tc['epochs']*tc['steps_per_epoch']
    step, history = 0, []
    torch.cuda.synchronize()
    train_started = time.perf_counter()
    with (out/'history.jsonl').open('w') as history_file:
        for epoch in range(tc['epochs']):
            model.train()
            losses = []
            for _ in range(tc['steps_per_epoch']):
                ids = np.random.default_rng(cfg['seed']+step*7919).choice(len(pool),tc['batch_cases'],replace=False)
                batch = _batch(pool,ids,cfg,mean,std,np.random.default_rng(cfg['seed']+step*3571+7))
                xb,yb,peakb,cx = (a.to(device) for a in batch)
                context = model.temporal_context(model.point(cx),phase)
                delta = model(xb,phase,pk,context=context)
                base = mu + sd*((peakb-mu[pk])/sd[pk])[:,:,None]
                loss = (((base+delta-yb)/sd)**2).mean()
                if not torch.isfinite(loss):
                    raise FloatingPointError(f'nonfinite loss at step {step}')
                opt.zero_grad(set_to_none=True); loss.backward()
                gn = float(torch.nn.utils.clip_grad_norm_(model.parameters(),tc['grad_clip'],error_if_nonfinite=True))
                for group in opt.param_groups:
                    group['lr'] = tc['lr']*_lr(step,total,tc['warmup_steps'],tc['min_lr_ratio'])
                opt.step(); step += 1; losses.append(float(loss.detach()))
            torch.cuda.synchronize()
            row = dict(epoch=epoch,loss_z=float(np.mean(losses)),step=step,lr=opt.param_groups[0]['lr'],
                       grad_norm_last=gn,elapsed_s=time.perf_counter()-train_started)
            history.append(row); history_file.write(json.dumps(row)+'\n'); history_file.flush()
            if epoch % tc['log_every_epochs'] == 0 or epoch == tc['epochs']-1:
                print(f"[tt] epoch {epoch+1}/{tc['epochs']} step {step}/{total} loss_z {row['loss_z']:.5f} elapsed_s {row['elapsed_s']:.1f}",flush=True)
    train_seconds = time.perf_counter()-train_started
    torch.save(dict(state_dict=model.state_dict(),input_mean=mean,input_std=std,config=cfg,
                    initialization_sha256=init_hash,config_sha256=cfg_digest,code_sha256=code_hashes),out/'ckpt_last.pt')
    model.eval()
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = True
    truth, predictions, anchor_devs, contexts = [], [], [], {}
    eval_started = time.perf_counter()
    for i,uid in enumerate(test_ids):
        c = _load_case(cfg,uid,fold,False)
        lp = _predict_case(model,c,phase,mean,std,fold,cfg,device)
        dev = float(np.max(np.abs(lp[pk]-c['peak_ln'])))
        if dev > 1e-9 or not np.isfinite(lp).all():
            raise ValueError(f'{uid}: peak anchor/finite gate failed {dev}')
        anchor_devs.append(dev); truth.append(c['tau_raw']); predictions.append(lp)
        contexts[uid] = hashlib.sha256(c['context_idx'].tobytes()).hexdigest()
        print(f'[eval] {i+1}/{len(test_ids)} {uid} points={len(c["peak_ln"])} anchor_max={dev:.2e}',flush=True)
    report = TA.cycle_report(truth,predictions,fold['steps'],pk,fold['q_norm'],fold['floor'],fold['eps'])
    ec = report['eval_compatible']
    required = ['cycle_r2cb_pa','trough_r2cb_pa','tawss_r2cb_pa','cycle_r2cb_ln']
    if not all(np.isfinite(ec[k]) for k in required):
        raise ValueError('nonfinite main metrics')
    if TA.sha256_file(config_path) != cfg_digest:
        raise RuntimeError('config changed during run')
    result = dict(schema=SCHEMA,name=cfg['name'],arm=cfg['arm'],fold=cfg['fold'],seed=cfg['seed'],
                  smoke=smoke,partition='test',n_cases=len(test_ids),config=cfg,config_sha256=cfg_digest,
                  split_sha256=TA.sha256_file(cfg['split_path']),frame_stats_sha256=TA.sha256_file(cfg['frame_stats_path']),
                  cache_manifest_sha256=TA.sha256_file(Path(cfg['cache_dir'])/'manifest.json'),
                  waveform_sha256=TA.sha256_file(cfg['waveform_path']),code_sha256=code_hashes,
                  initialization_sha256=init_hash,n_parameters=sum(p.numel() for p in model.parameters()),
                  input_mean=mean.tolist(),input_std=std.tolist(),steps=step,final_loss_z=history[-1]['loss_z'],
                  context_contract=dict(count=tc['context_points'],seed=cfg['seed'],stream='context',
                    sampling='SHA256(uid|seed|stream) -> PCG64 sorted without replacement; same training/evaluation rule',
                    holdout_index_sha256=contexts),
                  train_seconds=train_seconds,eval_seconds=time.perf_counter()-eval_started,
                  elapsed_seconds=time.perf_counter()-started,execution=execution,
                  evaluation_contract=dict(truth_space='raw_pa_unclipped',frame_steps=fold['steps'].tolist(),
                    peak_index=pk,peak_step=int(fold['steps'][pk]),target_floor=fold['floor'],
                    target_eps=fold['eps'],checkpoint='last'),
                  anchor_check=dict(max_abs_peak_ln_vs_anchor=max(anchor_devs),passed=max(anchor_devs)<=1e-9,
                                    n_cases=len(test_ids),unit='ln Pa'),metrics=report)
    TA.write_json(out/'metrics.json',result)
    print(f"[done] {cfg['name']} steps={step} cycle={ec['cycle_r2cb_pa']:.5f} train_s={train_seconds:.1f}",flush=True)
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config',required=True)
    parser.add_argument('--smoke',action='store_true',help='train on first few training cases and evaluate one holdout only; never a formal matrix run')
    args = parser.parse_args()
    run(args.config,smoke=args.smoke)


if __name__ == '__main__':
    main()

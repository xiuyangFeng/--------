#!/usr/bin/env python3
"""Read-only input/config/compatibility audit before a temporal matrix launch."""
import argparse
import json
import socket
from pathlib import Path

import numpy as np
import torch

from training_wss_min import time_transformer as TT
from training_wss_min import time_adapter_probe as TA


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--config-dir', required=True)
    ap.add_argument('--previous-frozen', required=True)
    args = ap.parse_args()
    cdir = Path(args.config_dir).resolve()
    matrix = json.loads((cdir/'matrix.json').read_text())
    root = Path(__file__).resolve().parents[2]
    previous = Path(args.previous_frozen)
    report = dict(host=socket.gethostname(), experiment=matrix['experiment'], configs={}, folds={}, compatibility={})
    seen = set()
    for job in matrix['arms']:
        path = cdir/job['config']; cfg = TT.load_config(path)
        if cfg['seed'] != 1234 or cfg['train']['epochs']*cfg['train']['steps_per_epoch'] != 1800:
            raise ValueError('seed/budget differs from registered matrix')
        if Path(cfg['out_dir']).exists() and any(Path(cfg['out_dir']).iterdir()):
            raise ValueError('formal output already populated')
        torch.manual_seed(cfg['seed'])
        model = TT.TemporalFieldModel(60,cfg['architecture'])
        report['configs'][job['id']] = dict(sha256=TA.sha256_file(path),initialization=TT.state_hash(model),
                                          parameters=sum(p.numel() for p in model.parameters()))
        if cfg['fold'] in seen:
            continue
        seen.add(cfg['fold']); fold = TT._load_fold(cfg)
        manifest = fold['manifest']; dc = manifest['config']
        from training_wss_min import evaluate as E
        donor = Path(dc['donor_run'])/E.checkpoint_filename(dc['checkpoint'])
        if TA.sha256_file(donor) != manifest['donor_checkpoint_sha256']:
            raise ValueError('donor checkpoint drift')
        files = []
        for uid in fold['train_ids']+fold['test_ids']:
            cache = Path(cfg['cache_dir'])/(TA.case_key(uid)+'.npz')
            bundle = Path(cfg['data_root'])/uid/'bundle.npz'
            if not cache.is_file() or not bundle.is_file():
                raise FileNotFoundError(uid)
            with np.load(bundle,allow_pickle=False) as z:
                if not np.array_equal(z['steps'],fold['steps']) or int(z['peak_step']) != fold['steps'][fold['peak_index']]:
                    raise ValueError('bundle steps mismatch '+uid)
            files.append(dict(uid=uid,cache_bytes=cache.stat().st_size,bundle_bytes=bundle.stat().st_size))
        report['folds'][str(cfg['fold'])] = dict(train=len(fold['train_ids']),holdout=len(fold['test_ids']),files=files,
                split_sha256=TA.sha256_file(cfg['split_path']),cache_manifest_sha256=TA.sha256_file(Path(cfg['cache_dir'])/'manifest.json'),
                stats_sha256=TA.sha256_file(cfg['frame_stats_path']),waveform_sha256=TA.sha256_file(cfg['waveform_path']))
    hashes = {v['initialization'] for v in report['configs'].values()}
    sizes = {v['parameters'] for v in report['configs'].values()}
    if len(hashes)!=1 or len(sizes)!=1:
        raise ValueError('unpaired initialization/parameter count')
    files = ['config.py','dataset.py','evaluate.py','metrics.py','paired_initialization.py','train.py',
             'baseline_models.py','time_adapter_probe.py','time_adapter_finetune.py']
    for name in files:
        a=root/'training_wss_min'/name;b=previous/'training_wss_min'/name
        if TA.sha256_file(a)!=TA.sha256_file(b):
            raise ValueError('old source changed: '+name)
    n=0
    for name in ('wss_time_adapter_v51_20260923','wss_time_adapter_ft_v51_20260923','wss_time_ecc_20260918'):
        for a in (root/'training_wss_min/configs'/name).glob('*.json'):
            b=previous/a.relative_to(root)
            if TA.sha256_file(a)!=TA.sha256_file(b):
                raise ValueError('old config changed: '+str(a))
            n+=1
    report['compatibility']=dict(previous_frozen=str(previous),source_files_unchanged=len(files),configs_unchanged=n)
    report['passed']=True
    dest=Path(matrix['output_experiment_dir'])/'preflight.json'
    TA.write_json(dest,report)
    print(json.dumps(dict(passed=True,configs=len(report['configs']),folds=report['folds'].keys(),parameters=list(sizes),
                         old_configs_unchanged=n,path=str(dest)),default=list),flush=True)


if __name__=='__main__':
    main()

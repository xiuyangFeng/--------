# Joint cycle V5.2 data contract

Implementation: `training_wss_min/joint_cycle_data.py`.

## Split and provenance

- The experimental split is the existing patient-grouped CV5 **fold 0**, with **206 training / 55 held-out cases**, covering all 261 V5.2 units. One seed, 1234.
- Default split: `data_wss_v5/views_v5_2_full_20260923/wss_min_view_v1/cv5_v52/fold0.json`. `GNN_JOINT_SPLIT_PATH` or CLI `--split-path` can select an explicit alternative, and its SHA256 is bound into statistics. `GNN_JOINT_SOURCE_ROOT` defaults to the original absolute GNN repository, so a frozen code snapshot does not silently change data roots.
- Patient identity strips ILO terminal `-0/-1` and `before/after`; other cases use the case name. The audit also unions declared duplicate/related geometry groups. The final fold-0 audit has zero crossing groups.
- The initially proposed IND 170/91 split crosses 44 patient-name groups and was superseded before training. The first failed audit and patient identities are preserved in `data_audit_ind_initial.json`.
- The V5.2 assembly plan authoritatively selects 159 inherited V5.1 H5 files and 102 new/rebuilt V5.2 H5 files. It is not a missing-data fallback. All inputs retain resolved paths, sizes and modification times; prescribed boundary conditions retain their frozen UDF SHA256.
- The 102 new/rebuilt cases initially lacked volume views. Existing `wss_v5.views.volume_view.build_case_volume` regenerates them from their exact H5 and current wall atlas transform under `data_audit_cache/derived_volume`, with builder hash and input signatures. Original datasets are unchanged.
- Cropped wall views are mapped by exact `wall_node_id_cas` identities into the original H5 wall arrays; no row-order assumption is made.

## Targets and time

All targets have point-major shape `[Nquery,80,C]`: velocity `C=3` in atlas-aligned m/s, pressure `C=1` in Pa relative to each frame's stored volume-pressure reference, and scalar WSS magnitude `C=1` in Pa. No nearest-cell wall pressure or placeholder wall velocity is supervised.

Frames 0–79 (steps 1120–1278, 0.01 s spacing) form the primary 0.8 s cycle. Frame 80 (step 1280) is read only for endpoint QA. CFD endpoint values need not be numerically identical to frame 0; discrepancies are recorded, not silently overwritten. Phase inputs are the prescribed waveform's `q_norm,dq_norm,t_sin,t_cos`; H5 nominal flow samples must match this protocol.

Only fold-0 training query candidates fit per-frame linear means/stds for velocity and pressure and log means/stds for WSS. WSS uses `log(max(tau,0.05)+1e-6)`. Inverse normalization returns the floored target for WSS; physical evaluation should still compare to unfloored `y_raw`. These targets do not provide an absolute pressure reference or WSS direction/OSI.

## Geometry and conditions

- Wall support and WSS queries use 27 features: xyz, abscissa, radius, centerline curvature, log radius, rho, theta sin/cos, dr/ds, distances to junction/endpoint, endpoint zone, aligned normal xyz, surface curvature features (`k1,k2,gauss,curvedness,k1_c,k2_c,gauss_c`), `tn_dot`, and geometry-derived Murray log flow/shear priors.
- Volume queries use 20 native features: the first 14 shared geometry fields, aligned radial direction xyz, two Murray priors, and distance to wall. Wall curvature is never fabricated inside the lumen. Murray flow shares map by verified segment ID, while the interior shear prior uses the interior atlas radius.
- Curvature features use signed-log1p and training-only absolute-p99 clipping. Other non-coordinate feature columns are standardized using training candidate points; normalized xyz remain unchanged. Support normalization uses wall statistics.
- Twenty boundary-condition channels come from **the frozen H5 `conditions` attributes**: log(R1,R2,C) in fixed outlet order `out-le,out-li,out-re,out-ri` (12), period, density, five Carreau parameters, and prescribed inlet area. Resistance units are **Pa s/kg**, capacitance **kg/Pa**. No simulated pressure/flux or labels enter these inputs. BC normalization counts each training case once.

## Candidate sampling and API

Preparation streams one complete CFD frame at a time and retains only query candidates; it never reads or stacks an entire 81-frame volume array. Output arrays are uncompressed `.npy`; the Dataset uses read-only memory maps.

Each case has uniform, disjoint training/evaluation candidate pools: requested volume 8192/16384 and wall 8192/8192. If a cloud is smaller, counts shrink proportionally into two nonempty pools, recorded in case metadata. No case is dropped. Support can sample the full cached wall geometry. Training query rows resample deterministically by case/epoch/seed; evaluation rows and support are deterministic and ignore epoch. Velocity and pressure share exactly the same query rows.

`JointCycleDataset(cache_root, partition, stats_path=None, support_n=5000, query_n=512, training=True, seed=1234)` returns:

```text
unit_id
support: pos[Ns,3], x[Ns,27], rows[Ns]
queries[velocity|pressure|wss]: pos[Nq,3], x[Nq,D],
    y[Nq,80,C], y_raw[Nq,80,C], weight[Nq], rows[Nq]
phase[80,4], bc[20]
```

`collate_joint_cycle` concatenates support and per-task queries, adds their point-to-case `batch` indices, and stacks `phase[B,80,4]` and `bc[B,20]`; `unit_ids` is a list. Weights are native cell volumes or wall areas, intentionally unnormalized for the runner's per-case normalization. `query_n<=0` selects the complete cached evaluation candidate pool, which is still a sampled estimate of the complete volume field.

`stats.json` exposes `dimensions`, `phase_dims=4`, `bc_dims=20`, feature/BC normalizers, and `targets[task]={mean[80,C],std[80,C],transform,floor,eps}`. `decode_target(task,pred,stats)` accepts NumPy or Torch. Incomplete training statistics are rejected unless the caller explicitly requests smoke-only `allow_partial_stats=True`.

## Execution and checks

The CPU work runs through Slurm on node03 with 4 allocated CPUs and 32 GB. Submission records are in `data_audit_submission.json` and logs in `data_audit_logs/`. The first one-case smoke passed target inverse transforms, shapes, shared velocity/pressure rows, deterministic training sampling and epoch resampling. The full pipeline repeats smoke checks and additionally asserts disjoint training/evaluation query pools and evaluation epoch invariance.

This work prepares supervision and checks its identity/units. It does not perform model training, held-out model inference or selection.

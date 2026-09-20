# M2 / MO0 GPU training arithmetic diagnostic

Both AMP and FP32 diagnostics completed 12 updates per model on job 13974. The old-code repeat diverges even with identical initialization and observed random choices. The final historical M2 / MO0 quality gap remains unresolved.

| Precision | Comparison to old reference | Step 0 head max Δ | Step 11 head max Δ | Step 11 parameter max Δ |
|---|---|---:|---:|---:|
| amp | old_repeat | 0.0009765625 | 0.0234375 | 0.0056971014 |
| amp | new_log_off | 0.001953125 | 0.022460938 | 0.0057200491 |
| amp | new_log_on | 0.001953125 | 0.026428223 | 0.0061750486 |
| amp | new_paired_log_on | 0.001953125 | 0.029632568 | 0.0046344697 |
| fp32 | old_repeat | 5.6624413e-07 | 0.015524864 | 0.004027456 |
| fp32 | new_log_off | 6.5565109e-07 | 0.016936541 | 0.0034846514 |
| fp32 | new_log_on | 7.4505806e-07 | 0.019197732 | 0.0043753982 |
| fp32 | new_paired_log_on | 6.8545341e-07 | 0.019708112 | 0.0033909082 |

## Verified observations

- Same-code old/old GPU execution diverges with identical initial tensors and identical observed stochastic choices; first observed forward divergence is fp.0 in both precisions.
- Old/new, logging, and paired construction comparisons have the same first observed divergence location and broadly overlapping magnitudes with old/old in this one trace. This is descriptive, not statistical equivalence.
- No CUDA RNG consumption, FPS selection, DropPath, AMP overflow, or skipped optimizer step mismatch was observed in these 12 fixed-batch steps.
- All five initial state hashes and constructor RNG states match in each precision. All source fingerprints stayed unchanged.
- On step 0, stem, all SA outputs, SA attention, and the wall branch match exactly; fp.0 is the first hooked module to differ. Old/old fp.0 max difference is 0.00390625 in AMP and 1.9073486328125e-6 in FP32.
- CPU/CUDA RNG, FPS index hashes and per-call RNG traces, and DropPath masks match at all 12 steps. All gradient norms are finite. AMP scale stays 65536; all five models apply 12/12 optimizer steps. FP32 also applies 12/12 steps.

## First observed divergent module

`fp.0` is `FeaturePropagation`: PyG `knn_interpolate` performs batched kNN lookup, inverse squared-distance weights, weighted-feature scatter sum divided by weight scatter sum; the result is concatenated with skip features and passed through two Linear + BatchNorm1d + ReLU stages. The diagnostic hooks the whole module output. It does not isolate a specific interpolation, reduction, matrix multiplication, or BatchNorm kernel.

## Interpretation limits

- One real batch is reused at epoch 0 for 12 updates; this does not reproduce the original 400-epoch data sequence, evaluation/checkpoint selection, or complete training lifecycle.
- There is only one old/old repeat pair per precision and one trajectory for each new configuration; no distribution or confidence interval establishes implementation equivalence.
- Hooks bracket the whole FeaturePropagation module; they do not isolate kNN ordering, interpolation reduction, or the following MLP as the source of its first numerical difference.
- Unscaled gradients already differ in old/old at step 0, with notable differences in zero-initialized GeoPE output weights in FP32. This does not identify the backward operator responsible.
- Logging and paired construction effects on full tensors are each compared against old_reference, not directly against each other; scalar losses are directly comparable but reflect already-diverged optimizer trajectories.
- Lockstep RNG restoration is deliberate for arithmetic isolation; parity here does not establish historical trainer RNG parity over a complete run.
- GPU same-code divergence is observed evidence, not a justification to dismiss the approximately 0.0358 final best Pa R2 deficit as noise.

Raw machine-readable details and artifact hashes: `analysis_job_13974.json`.

# V4 high-WSS tail calibration study

## Status

This is an experimental add-on to frozen V4 final. It does not replace
`calibrator_frozen_v4.joblib`. Candidate selection used train138 only, followed
by grouped OOF, fixed development73→holdout65 validation, and frozen test35
evaluation.

The selected candidate combines:

- the existing safe V4 final calibration for the bulk of each case;
- a second HistGradientBoosting model trained with extra weight on the
  case-wise upper prediction/truth tail;
- an inference-time gate based on the physics-V4 within-case prediction rank;
- case-level high-tail and peak-scale anchors inferred from diagnostic case
  summaries;
- a conservative peak-anchor strength of `0.30`.

No truth, raw case identity, or absolute wall coordinates are used at inference.

## Metric contract

High WSS follows the existing project convention: each case's CFD-truth top
10%. Two non-equivalent scopes are reported:

1. **pooled high-WSS**: concatenate the case-wise q90 subsets, then calculate
   one R² and mean bias;
2. **case-balanced high-WSS**: calculate high-WSS R²/peak error separately per
   case, then summarize cases equally.

The requested `R² >= 0.90` and underestimation `<= 10%` are achieved only under
the pooled/mean contract. They are not guaranteed for every case or every
extreme peak.

## Train138 grouped OOF

Source: `outputs/wss_mri_calculator/pointcloud_surface_mls_v4/high_tail_oof_train138_v1.json`

- current V4 final pooled high-WSS R²: `0.9062`;
- selected tail model pooled high-WSS R² before partial peak anchoring: `0.9139`;
- selected `0.30` peak strength retains pooled high-WSS R² about `0.9136`;
- case-mean peak underestimation falls to about `9.9%`;
- overall case-balanced raw R² remains about `0.9541` versus current `0.9549`.

Uniform monotone tail stretching, predicted case-specific power stretching, and
raw intrinsic-coordinate ExtraTrees variants were rejected because they reduced
held-out high-WSS R² or overall accuracy.

## Fixed development73 → holdout65

Source: `outputs/wss_mri_calculator/pointcloud_surface_mls_v4/high_tail_dev73_holdout65_v2_peak30.json`

| Metric | Current V4 final | High-tail candidate |
| --- | ---: | ---: |
| pooled high-WSS R² | 0.9112 | **0.9223** |
| pooled high-WSS NRMSE | 0.2606 | **0.2438** |
| pooled high-WSS mean underestimation | 5.14% | **1.55%** |
| case-balanced overall raw R² | **0.95790** | 0.95764 |
| case-balanced high-WSS R² mean | **0.79186** | 0.78899 |
| case-mean peak underestimation | 15.67% | **12.09%** |

## Frozen test35

Source: `outputs/wss_mri_calculator/pointcloud_surface_mls_v4/test35_high_tail_v2_peak30.json`

| Metric | Current V4 final | High-tail candidate |
| --- | ---: | ---: |
| pooled high-WSS R² | 0.9282 | **0.9304** |
| pooled high-WSS NRMSE | 0.2287 | **0.2250** |
| pooled high-WSS mean underestimation | 3.19% | **0.00%** |
| case-balanced overall raw R² | **0.95607** | 0.95514 |
| case-balanced high-WSS R² mean | **0.74027** | 0.73974 |
| case-mean peak underestimation | 11.51% | **9.58%** |

For `ILO/YU_XIANG_SHENG-1/before`, high-WSS R² improves from `0.8317` to
`0.8605`, but its maximum remains underestimated by `32.7%`. This is direct
evidence that pooled success is not a per-case guarantee.

## Reproduction

```bash
export PYTHONPATH=/public/newhome/cy/Digital_twin/GNN/wss_mri_calculator/src:/public/newhome/cy/Digital_twin/GNN
export PYTHON=/public/newhome/cy/.conda/envs/GNN/bin/python

$PYTHON wss_mri_calculator/src/calibrate_v4_high_tail_oof.py \
  --cache-dir outputs/wss_mri_calculator/pointcloud_surface_mls_v4/point_cache_train138_s600 \
  --json-out outputs/wss_mri_calculator/pointcloud_surface_mls_v4/high_tail_oof_train138_v1.json

$PYTHON wss_mri_calculator/src/validate_v4_high_tail_holdout.py \
  --cache-dir outputs/wss_mri_calculator/pointcloud_surface_mls_v4/point_cache_train138_s600 \
  --case-order-result outputs/wss_mri_calculator/pointcloud_surface_mls_v4/train138_s600.json \
  --json-out outputs/wss_mri_calculator/pointcloud_surface_mls_v4/high_tail_dev73_holdout65_v2_peak30.json

$PYTHON wss_mri_calculator/src/apply_v4_high_tail_calibrator.py \
  --cache-dir outputs/wss_mri_calculator/pointcloud_surface_mls_v4/point_cache_test35_s1200 \
  --model wss_mri_calculator/experiments/pointcloud_surface_mls_v4/calibrator_high_tail_v1.joblib \
  --peak-anchor-strength 0.30 \
  --json-out outputs/wss_mri_calculator/pointcloud_surface_mls_v4/test35_high_tail_v2_peak30.json \
  --predictions-out outputs/wss_mri_calculator/pointcloud_surface_mls_v4/test35_high_tail_v2_peak30_predictions.npz
```

## Remaining requirement gap

Achieving high-WSS R² `>= 0.90` and peak underestimation `<= 10%` for every
case is not supported by the current inference features. Grouped OOF screens
show that scalar/rank tail boosts needed by one case often overcorrect another.
The next method should expose raw near-wall profile-shape and mesh-resolution
diagnostics (minimum cell depth, fitted depth curvature, residual structure,
and wall-topology-aware context), or use higher-resolution velocity data.

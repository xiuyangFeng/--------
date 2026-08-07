# Stage 1 confirmation-seed matrix

This independent supplemental matrix was created only after the four-arm
seed-1234 Gate selected `G-PE` and `G-Raw` by the preregistered
`validation_field_score_cb`.  It adds seeds 2345 and 3456 so each selected arm
has the frozen confirmation set `[1234, 2345, 3456]`.

Training randomness follows `train.seed`.  `sampling.seed` remains 1234 so the
post-training fixed-validation support sample is identical across arms and
seeds; the 5000 query indices per case remain bound to the immutable validation
query manifest.  All four jobs are data-only, start from scratch, and preserve
the original optimizer, schedule, early-stop and architecture contracts.

The singleton `groups` are submission/preflight units, not scientific pairs.
Scientific comparisons pair `G-PE` and `G-Raw` within each seed.

# Stage 1 field-v4 matrix

This directory is frozen before formal training.  The four learned arms use
the same train123/val15 cases, fixed 5000-point validation queries, data-only
loss, optimizer, schedule, budget and seed.  `G/L` changes only the case
conditioner; `Raw/PE` changes only the query coordinate encoding.  B0 is the
separate zero-training atlas and is permanently included in the result table.

Primary checkpoint: `best_validation_field_cb.pt`.  Test35, WSS, BC and PDE
are not read or used by this matrix.

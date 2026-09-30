"""Physical velocity diagnostics on unchanged point rows and quadrature."""
from __future__ import annotations

import math
import numpy as np

from .joint_cycle import SEGMENTS, weighted_fields
from .joint_cycle_round2_metrics import case_metrics as original_case_metrics

PHASES = {**SEGMENTS, "early_trough": list(range(5, 10)),
          "late_trough": list(range(43, 58))}


def case_metrics(y, p, weights, frame, valid, angle_floor=0.01, tail_q80=None):
    y, p, weights = (np.asarray(a, np.float64) for a in (y, p, weights))
    out = original_case_metrics("velocity", y, p, weights, angle_floor)
    sy, sp = np.linalg.norm(y, axis=-1), np.linalg.norm(p, axis=-1)
    cosine = np.clip((y * p).sum(-1) / np.maximum(sy * sp, 1e-20), -1, 1)
    wn = weights / weights.sum()
    tangent = np.asarray(frame, np.float64)[:, :, 0]
    valid = np.asarray(valid, bool)
    axial_y, axial_p = np.einsum("nfc,nc->nf", y, tangent), np.einsum("nfc,nc->nf", p, tangent)
    transverse_y, transverse_p = y - axial_y[..., None] * tangent[:, None], p - axial_p[..., None] * tangent[:, None]
    out["error_decomposition"], out["anatomy"], out["train_threshold_tail"] = {}, {}, {}
    for name, ids in PHASES.items():
        yy, pp, ry, rp = y[:, ids], p[:, ids], sy[:, ids], sp[:, ids]
        vector_mse = float(np.einsum("n,nfc->", wn, (pp - yy) ** 2) / len(ids))
        speed_mse = float(np.einsum("n,nf->", wn, (rp - ry) ** 2) / len(ids))
        mask = ry >= angle_floor
        coverage = float(np.einsum("n,nf->", wn, mask) / len(ids))
        norm = float(np.einsum("n,nf->", wn, mask))
        cc = float(np.einsum("n,nf->", wn, cosine[:, ids] * mask) / norm) if norm else None
        if name not in out:
            out[name] = weighted_fields(yy, pp, weights)
            out[name].update(direction_cosine=cc, direction_truth_speed_floor_m_s=angle_floor,
                             direction_weight_coverage=coverage)
            out["speed"][name] = weighted_fields(ry[..., None], rp[..., None], weights)
        out["error_decomposition"][name] = {
            "vector_mse_m2_s2": vector_mse, "speed_mse_m2_s2": speed_mse,
            "direction_coupled_mse_m2_s2": max(0., vector_mse - speed_mse),
            "speed_bias_m_s": float(np.einsum("n,nf->", wn, rp - ry) / len(ids)),
            "direction_truth_threshold_coverage": coverage,
            "prediction_below_0p001_given_truth_above_floor": float(
                np.einsum("n,nf->", wn, (rp < .001) * mask) / norm) if norm else None}
        wv = weights * valid
        frac = float(wv.sum() / weights.sum())
        if wv.sum() > 0:
            wv /= wv.sum()
            ay, ap = axial_y[:, ids], axial_p[:, ids]
            inverse_y, inverse_p = ay < -.01, ap < -.01
            truth_fraction = float(np.einsum("n,nf->", wv, inverse_y) / len(ids))
            pred_fraction = float(np.einsum("n,nf->", wv, inverse_p) / len(ids))
            intersection = float(np.einsum("n,nf->", wv, inverse_y & inverse_p) / len(ids))
            axrmse = math.sqrt(float(np.einsum("n,nf->", wv, (ap - ay) ** 2) / len(ids)))
            trrmse = math.sqrt(float(np.einsum("n,nfc->", wv,
                (transverse_p[:, ids] - transverse_y[:, ids]) ** 2) / len(ids)))
        else:
            axrmse = trrmse = truth_fraction = pred_fraction = intersection = None
        out["anatomy"][name] = {"frame_valid_volume_coverage": frac,
            "axial_rmse_m_s": axrmse, "transverse_vector_rmse_m_s": trrmse,
            "reverse_axial_truth_volume_fraction": truth_fraction,
            "reverse_axial_pred_volume_fraction": pred_fraction,
            "reverse_axial_precision": intersection / pred_fraction if pred_fraction else None,
            "reverse_axial_recall": intersection / truth_fraction if truth_fraction else None}
        if tail_q80 is not None:
            tail = ry >= np.asarray(tail_q80)[None, ids]
            predicted_tail = rp >= np.asarray(tail_q80)[None, ids]
            wt = wn[:, None] * tail
            total = float(wt.sum())
            predicted_mass = float((wn[:, None] * predicted_tail).sum())
            intersection_mass = float((wn[:, None] * (tail & predicted_tail)).sum())
            false_positive_mass = float((wn[:, None] * (~tail & predicted_tail)).sum())
            out["train_threshold_tail"][name] = {
                "volume_phase_coverage": total / len(ids),
                # All coverage denominators are the complete volume x phase
                # domain, not the true-tail subset. Precision/recall instead
                # use their explicit predicted/true positive masses.
                "predicted_volume_phase_coverage": predicted_mass / len(ids),
                "false_positive_volume_phase_coverage": false_positive_mass / len(ids),
                "tail_precision": intersection_mass / predicted_mass if predicted_mass > 0 else None,
                "tail_recall": intersection_mass / total if total > 0 else None,
                "speed_bias_m_s": float((wt * (rp - ry)).sum() / total) if total else None,
                "vector_rmse_m_s": math.sqrt(float((wt[..., None] * (pp - yy) ** 2).sum() / total)) if total else None,
                "pred_true_speed_ratio": float((wt * rp).sum() / (wt * ry).sum()) if (wt * ry).sum() > 0 else None}
    return out

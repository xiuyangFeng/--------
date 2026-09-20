"""Post-training M2 mechanism diagnostics and verified original-wall figures.

Inference uses the existing fixed support, full wall queries and saved statistics.
Nothing is trained, recalibrated, remeshed or selected from the three example cases.
GPU work requires Slurm; --self-test runs small CPU semantic checks only.
"""
from __future__ import annotations

import argparse
from functools import lru_cache
import gc
import json
import math
import os
from pathlib import Path
import tempfile
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import colors
from matplotlib.lines import Line2D
import numpy as np
import torch
from torch_geometric.utils import scatter

from training_wss_min import config as C, dataset as D, evaluate as E, surface as S
from training_wss_min.baseline_models import _knn_group
from training_wss_min.tools import report_m2_optimization as R
from training_wss_min.tools.m2_optimization_common import (
    ANCHOR_CONFIG, ANCHOR_RUN, COMBINATIONS, EXP, ROOT, RUNS,
    fingerprints, save_json, sha, stamp,
)
from training_wss_min.experiments.v6_multiradius_bt_20260909 import plot_paired_wall_fields as V


OUT = EXP / "diagnostics"
REPRESENTATIVES = ("AG/slow/MA_TIAN_YI", "ILO/SUN_XU_XIA-1/before", "ILO/ZHANG_JIN_CHUN-1/before")
VIEWS = ((10., -60.), (10., 120.))
LIMITS = (
    "单seed1234、已暴露test34的训练后机制诊断；激活与门控分布不是因果重要性。"
    "三例在本轮结果出现前固定，不能代表总体收益。图显示原始壁面全部顶点，未重建/平滑几何。"
    "门控作用在固定5000个support点；解码熵和查询残差覆盖完整壁面。"
)
BOUNDARY_CONTEXT = ("MO-P4", "best", "AAA/unruputer/ZHANG_YONG_ZHI", "log_z")
BOUNDARY_METRIC = "distribution.pred_above_one_fraction"
BOUNDARY_PROBE = EXP / "threshold_probe/job_13980/probe.json"
BOUNDARY_PROBE_SHA256 = "9825e4c07b704691dc9d97ebc4eebf6c449e86a5278da5ac800a16d24b6638fe"
BOUNDARY_SOURCE_HASHES = {
    "metrics": "6f4c49970ed2508920c2bac88ae8e5e057f3a41c0e66f483612e1772a9ad197e",
    "checkpoint": "a84e2faad6fc6286d8436b3e52ac3bec42eceaf62ae693c0cf678ee388187e29",
    "config": "71ac891c63d87da0ce87e8bbddc3aeefe67dc50833ec694dd6465099cae7e391",
}


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def record(path):
    path = Path(path)
    return {"path": str(path.resolve()), "sha256": sha(path)}


def slug(value):
    return value.replace("/", "__")


def save_arrays(path, **arrays):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.stem + ".tmp.npz")
    np.savez_compressed(temporary, **arrays)
    os.replace(temporary, path)
    return record(path)


def distribution(value):
    array = np.asarray(value, dtype=np.float64).reshape(-1)
    if not len(array):
        return {"n": 0}
    require(bool(np.isfinite(array).all()), "non-finite diagnostic distribution")
    qs = np.percentile(array, [1, 10, 50, 90, 99])
    return {"n": len(array), "mean": float(array.mean()), "std": float(array.std()),
            "min": float(array.min()), "max": float(array.max()),
            "mean_absolute": float(np.abs(array).mean()), "rms": float(np.sqrt(np.mean(array ** 2))),
            **{f"p{q:02d}": float(v) for q, v in zip((1, 10, 50, 90, 99), qs)}}


def gate_spatial_summary(gate, abscissa, n_scales):
    gate = np.asarray(gate, dtype=np.float64)
    require(gate.ndim == 2 and gate.shape[1] % n_scales == 0, "gate scale/channel shape mismatch")
    by_scale = gate.reshape(len(gate), n_scales, -1).mean(-1)
    abscissa = np.asarray(abscissa, dtype=np.float64)
    require(len(abscissa) == len(gate) and np.isfinite(abscissa).all(), "gate support row mismatch")
    bins = np.clip(np.searchsorted(np.linspace(0., 1., 11), abscissa, side="right") - 1, 0, 9)
    along = []
    for index in range(10):
        selected = bins == index
        along.append({"abscissa_bin": [index / 10, (index + 1) / 10], "support_points": int(selected.sum()),
                      "mean_gate_per_scale": by_scale[selected].mean(0).tolist() if selected.any() else None})
    return {"all_scale_channels": distribution(gate),
            "point_mean_gate_per_scale": [distribution(by_scale[:, i]) for i in range(n_scales)],
            "per_channel_mean": gate.mean(0).tolist(), "per_channel_spatial_std": gate.std(0).tolist(),
            "by_original_abscissa_norm": along,
            "boundary": "existing unstandardized abscissa bins; support-point gate variation, not causal attribution"}


def reconstruction_weights(module, args, kwargs, actual):
    """Reconstruct query weights externally and verify the observed hidden output."""
    support_x, support_pos, query_pos, support_batch, query_batch = args[:5]
    row, col = _knn_group(support_pos, support_batch, query_pos, query_batch, module.k)
    relative = support_pos[col] - query_pos[row]
    squared = (relative * relative).sum(-1).clamp_min(1e-16)
    distance = torch.sqrt(squared)
    scale = scatter(distance, row, dim=0, dim_size=len(query_pos), reduce="mean").clamp_min(1e-9)
    terms = [relative / scale[row, None], (distance / scale[row])[:, None]]
    if module.attr_dim:
        attrs = kwargs["geo_attr"]
        terms.append(attrs[0][col] - attrs[1][row])
    correction = module.correction(torch.cat(terms, -1)).squeeze(-1)
    factor = (-.5 * (module.beta.clamp_min(0.) - 2.) * squared.log() + correction).clamp(-8., 8.)
    unnormalized = squared.reciprocal() * factor.exp()
    denominator = scatter(unnormalized, row, dim=0, dim_size=len(query_pos), reduce="sum").clamp_min(1e-16)
    weights = unnormalized / denominator[row]
    hidden = scatter(support_x[col] * weights[:, None], row, dim=0, dim_size=len(query_pos), reduce="sum")
    torch.testing.assert_close(hidden, actual[0], rtol=2e-5, atol=2e-6)
    entropy = scatter(-(weights.float() * weights.float().clamp_min(1e-12).log()), row,
                      dim=0, dim_size=len(query_pos), reduce="sum")
    return entropy.detach().cpu().numpy(), entropy.exp().detach().cpu().numpy(), float((hidden - actual[0]).abs().max())


class MechanismCollector:
    def __init__(self, model):
        self.model, self.handles = model, []
        self.reset()
        branch = model.local_wall_branch
        if branch is not None and getattr(branch, "modulation", None) is not None:
            self.handles.append(branch.modulation.register_forward_hook(self.modulation_hook))
        if model.local_query is not None:
            self.handles.append(model.local_query.register_forward_hook(self.query_hook, with_kwargs=True))

    def reset(self):
        self.entropy, self.effective, self.residuals = [], [], []
        self.gate, self.modulation_update = None, None
        self.query_reconstruction_max_abs = 0.

    @torch.no_grad()
    def modulation_hook(self, module, args, output):
        require(self.modulation_update is None, "support was encoded more than once for this case")
        self.modulation_update = output.detach().float().cpu().numpy()
        if self.model.local_wall_branch.modulation_mode == "gate":
            self.gate = (2. * torch.sigmoid(output)).detach().float().cpu().numpy()

    @torch.no_grad()
    def query_hook(self, module, args, kwargs, output):
        entropy, effective, error = reconstruction_weights(module, args, kwargs, output)
        self.entropy.append(entropy)
        self.effective.append(effective)
        self.query_reconstruction_max_abs = max(error, self.query_reconstruction_max_abs)
        if output[1] is not None:
            self.residuals.append(output[1].detach().float().cpu().numpy().reshape(-1))

    def close(self):
        for handle in self.handles:
            handle.remove()

    def case_result(self, cfg, case, prediction_norm, prediction_pa, stats):
        arrays, result = {}, {"query_decoder": cfg.model.query_decoder}
        if self.entropy:
            arrays["query_weight_entropy"] = np.concatenate(self.entropy)
            arrays["query_effective_neighbors"] = np.concatenate(self.effective)
            require(len(arrays["query_weight_entropy"]) == len(prediction_norm), "query entropy does not cover full wall")
            result["query_attention"] = {"entropy": distribution(arrays["query_weight_entropy"]),
                "effective_neighbors": distribution(arrays["query_effective_neighbors"]),
                "hidden_reconstruction_max_abs": self.query_reconstruction_max_abs,
                "aggregation": "all full-wall queries; no unweighted mean of chunk averages"}
        if self.residuals:
            residual = np.concatenate(self.residuals)
            require(len(residual) == len(prediction_norm), "query residual does not cover full wall")
            without = np.clip(D.denormalize_wss(np.asarray(prediction_norm) - residual, stats), 0., None)
            arrays.update(query_residual_log_z=residual, query_residual_impact_pa=prediction_pa - without)
            high = case["y_raw"] >= np.percentile(case["y_raw"], 90.)
            result["query_residual"] = {"log_z": distribution(residual),
                "same_weights_output_impact_pa": distribution(arrays["query_residual_impact_pa"]),
                "true_high_wss_output_impact_pa": distribution(arrays["query_residual_impact_pa"][high]),
                "boundary": "same-weight removal of the additive query output; not a retrained ablation"}
        branch = self.model.local_wall_branch
        if self.modulation_update is not None:
            seed = S.stable_seed(cfg.eval.support_seed, case["unit_id"], "eval_support")
            indices = D.sample_support_indices(case, cfg.data, seed, n_points=int(cfg.data.support_n_points or cfg.data.wall_n_points),
                sampling=cfg.data.support_sampling or cfg.data.sampling, stream="eval_support")
            require(len(indices) == len(self.modulation_update), "gate support sampling differs from evaluator")
            arrays.update(support_indices=indices, support_modulation_logits_or_addition=self.modulation_update)
            result["local_modulation"] = {"mode": branch.modulation_mode,
                "support_points": len(indices), "module_output": distribution(self.modulation_update),
                "activation": {key: float(value) for key, value in branch.last_modulation_diagnostics.items()}}
            if self.gate is not None:
                arrays["support_gate"] = self.gate
                result["local_modulation"]["spatial_gate"] = gate_spatial_summary(
                    self.gate, case["abscissa_norm"][indices], len(branch.radii))
        return result, arrays


def parameter_diagnostics(model):
    result = {"parameters": sum(p.numel() for p in model.parameters()),
              "local_wall_parameters": sum(p.numel() for p in model.local_wall_branch.parameters()),
              "learned_residual_scales": {name: float(value.detach()) for name, value in model.named_parameters()
                  if value.numel() == 1 and any(part in name for part in ("gamma_local", "gamma_pointwise", "gamma_attention", "gamma_ffn"))}}
    result["new_sa1_blocks"] = [{"index": i, "gamma_local": float(block.gamma_local.detach()),
                                  "gamma_pointwise": float(block.gamma_pointwise.detach()), "drop_path_rate": block.drop_path_rate}
                                 for i, block in enumerate(model.sa[0].blocks) if i >= 1]
    if model.local_query is not None:
        query = model.local_query
        result["query"] = {"k": query.k, "beta_parameter": float(query.beta.detach()),
                           "beta_effective": float(query.beta.detach().clamp_min(0.)),
                           "correction_last_weight_l2": float(query.correction[-1].weight.detach().norm()),
                           "residual_enabled": query.residual}
        if query.residual:
            result["query"]["residual_output_weight_l2"] = float(query.res_out.weight.detach().norm())
    return result


def select_visual_candidate(reports):
    selection = reports["selection"]
    qualified = selection.get("ranking", [])
    if qualified:
        return {"id": qualified[0], "qualified": True, "label": "最高合格候选", "issues": []}
    best, last = reports["best"]["arms"], reports["last"]["arms"]
    available = [aid for aid, entry in best.items() if aid != "MO0"
                 and entry["evidence"]["valid_scientific_result"]
                 and last.get(aid, {}).get("evidence", {}).get("valid_scientific_result")]
    require(bool(available), "no completed candidate is available for comparison")
    aid = min(available, key=lambda key: (-best[key]["physical_r2_cb"], -last[key]["physical_r2_cb"], best[key]["mae"], key))
    return {"id": aid, "qualified": False, "label": "raw R²最高，未过完整门槛",
            "issues": selection.get("candidates", {}).get(aid, {}).get("issues", []),
            "boundary": "display comparison only; this does not replace historical M2 or change the registered gate"}


@lru_cache(maxsize=1)
def load_boundary_probe():
    """Validate the pinned, case-specific repeated-inference evidence on CPU."""
    require(sha(BOUNDARY_PROBE) == BOUNDARY_PROBE_SHA256, "threshold probe JSON changed")
    probe = json.loads(BOUNDARY_PROBE.read_text())
    require(probe["completed"] and probe["job_id"] == "13980" and len(probe["repeats"]) == 8,
            "incomplete threshold evidence")
    require(sha(ROOT / "training_wss_min/tools/probe_m2_threshold.py") == probe["script_sha256"], "threshold probe source changed")
    require(probe["source_metrics_sha256"] == BOUNDARY_SOURCE_HASHES["metrics"]
            and probe["reference_n"] == 57531 and probe["reference_fraction"] == 13859 / 57531,
            "threshold probe does not match registered source metrics")
    arrays, truth, counts = [], None, []
    for repeat in probe["repeats"]:
        path = Path(repeat["array_path"])
        require(sha(path) == repeat["array_sha256"], "threshold probe array changed")
        with np.load(path) as payload:
            pred, target = payload["prediction_norm"].copy(), payload["truth_norm"].copy()
        require(pred.shape == target.shape == (57531,) and np.isfinite(pred).all() and np.isfinite(target).all(),
                "threshold probe finite population changed")
        require(truth is None or np.array_equal(truth, target), "threshold probe truth ordering changed")
        truth = target
        count = int(np.count_nonzero(pred > 1.))
        require(repeat["n"] == 57531 and count == repeat["above_one_count"], "threshold probe integer count mismatch")
        arrays.append(pred)
        counts.append(count)
    require(set(counts) == {13859, 13860}, "threshold probe counts changed")
    reference_index = counts.index(13859)
    reference = arrays[reference_index]
    upper = float(np.nextafter(np.float32(1.), np.float32(np.inf)))
    require(reference[55137] == 1., "threshold probe reference vertex changed")
    for pred, count in zip(arrays, counts):
        expected = [] if count == 13859 else [55137]
        require(np.flatnonzero((pred > 1.) != (reference > 1.)).tolist() == expected
                and pred[55137] == (1. if count == 13859 else upper), "threshold probe crossing exceeds one known ULP")
    return {"prediction": reference, "truth": truth, "probe": record(BOUNDARY_PROBE),
            "reference_array": record(probe["repeats"][reference_index]["array_path"]),
            "source_script_sha256": probe["script_sha256"], "upper": upper,
            "evidence_inputs": {str(BOUNDARY_PROBE): BOUNDARY_PROBE_SHA256,
                str(ROOT / "training_wss_min/tools/probe_m2_threshold.py"): probe["script_sha256"],
                **{r["array_path"]: r["array_sha256"] for r in probe["repeats"]}}}


def review_boundary_fraction(actual, saved, prediction, truth, context, source_hashes):
    """Review only the one observed discrete boundary; continuous tolerances stay fixed."""
    if context != BOUNDARY_CONTEXT or source_hashes != BOUNDARY_SOURCE_HASHES:
        return None, "case/space/source identity is not the pinned boundary exception"
    n = saved["distribution"]["n"]
    if type(n) is not int or n != 57531 or type(actual["distribution"]["n"]) is not int or actual["distribution"]["n"] != n:
        return None, "finite population does not match the exact probe population"
    reference_fraction = saved["distribution"]["pred_above_one_fraction"]
    fraction = actual["distribution"]["pred_above_one_fraction"]
    if not math.isfinite(reference_fraction) or not math.isfinite(fraction):
        return None, "nonfinite fractions cannot be reviewed"
    reference_count, count = round(reference_fraction * n), round(fraction * n)
    if (abs(reference_fraction * n - reference_count) > 1e-8 or abs(fraction * n - count) > 1e-8
            or reference_count != 13859 or count != 13860):
        return None, "fraction is not the observed one-vertex integer-count difference"
    prediction, truth = np.asarray(prediction), np.asarray(truth)
    if (prediction.shape != (n,) or truth.shape != (n,)
            or not np.isfinite(prediction).all() or not np.isfinite(truth).all()):
        return None, "raw prediction/truth finite population is not identical"
    if int(np.count_nonzero(prediction > 1.)) != count or abs(float(np.mean(prediction > 1.)) - fraction) > 1e-15:
        return None, "raw prediction does not reproduce the reported integer count"
    probe = load_boundary_probe()
    if not np.array_equal(truth, probe["truth"]):
        return None, "raw truth or vertex ordering differs from the probe"
    crossing = np.flatnonzero((prediction > 1.) != (probe["prediction"] > 1.)).tolist()
    if crossing != [55137] or float(prediction[55137]) != probe["upper"]:
        return None, "current threshold sides/value do not match the single probed vertex and ULP"
    return {"review_status": "accepted_discrete_boundary", "metric": BOUNDARY_METRIC,
            "context": dict(zip(("arm", "checkpoint", "case", "space"), context)),
            "n": n, "reference_count": reference_count, "recomputed_count": count, "count_delta": count - reference_count,
            "reference_fraction": reference_fraction, "recomputed_fraction": fraction, "one_vertex_fraction": 1. / n,
            "vertex_index": 55137, "current_vertex_value": float(prediction[55137]), "reference_vertex_value": 1.,
            "float32_ulp": probe["upper"] - 1., "crossing_indices_vs_probe_reference": crossing,
            "max_abs_vs_probe_reference": float(np.max(np.abs(prediction - probe["prediction"]))),
            "finite_mask_matches_probe": True, "truth_matches_probe": True,
            "probe": probe["probe"], "probe_reference_array": probe["reference_array"],
            "evidence_inputs": probe["evidence_inputs"],
            "probe_source_script_sha256": probe["source_script_sha256"],
            "source_metrics_sha256": source_hashes["metrics"], "checkpoint_sha256": source_hashes["checkpoint"],
            "config_sha256": source_hashes["config"],
            "limit": "Post-hoc review of a repeated-inference threshold boundary; original formal/failed per-vertex arrays were not saved. Original metrics and selection are unchanged."}, None


def metric_check(actual, saved, *, prediction=None, truth=None, context=None, source_hashes=None,
                 other_space_strict_passed=False):
    verification = V.verify_metrics(actual, saved)
    failures = [v for v in verification["checks"] if not v["passed"]]
    result = {"passed": verification["passed"], "strict_passed": verification["passed"],
            "checks": len(verification["checks"]), "strict_passed_checks": len(verification["checks"]) - len(failures),
            "strict_failed_checks": len(failures), "boundary_reviews": [],
            "max_absolute_difference": max((v["absolute_difference"] or 0.) for v in verification["checks"]),
            "failed_checks": failures}
    if other_space_strict_passed and len(failures) == 1 and failures[0]["metric"] == BOUNDARY_METRIC:
        review, rejection = review_boundary_fraction(actual, saved, prediction, truth, context, source_hashes)
        if review is not None:
            result.update(passed=True, boundary_reviews=[review])
        else:
            result["boundary_review_rejection"] = rejection
    return result


def metric_summary(provenance):
    verifications = [v for run in provenance["runs"].values() for ck in run.values()
                     for case in ck["cases"].values() for v in case["metric_verification"].values()]
    reviews = [review for v in verifications for review in v["boundary_reviews"]]
    return {"original_per_case_numeric_checks": sum(v["checks"] for v in verifications),
            "strict_passed": bool(verifications) and all(v["strict_passed"] for v in verifications),
            "strict_passed_numeric_checks": sum(v["strict_passed_checks"] for v in verifications),
            "strict_failed_numeric_checks": sum(v["strict_failed_checks"] for v in verifications),
            "boundary_review_count": len(reviews), "boundary_reviews": reviews}


def hotspot_codes(truth, prediction):
    true_hot = truth >= np.percentile(truth, 90.)
    pred_hot = prediction >= np.percentile(prediction, 90.)
    code = np.zeros(len(truth), dtype=np.int8)
    code[true_hot & pred_hot] = 1
    code[true_hot & ~pred_hot] = 2
    code[~true_hot & pred_hot] = 3
    return code, float(np.sum(true_hot & pred_hot) / np.sum(true_hot | pred_hot))


def _save_figure(fig, out, name):
    paths = []
    for extension in ("png", "pdf"):
        path = out / f"{name}.{extension}"
        fig.savefig(path, dpi=170, facecolor="white")
        paths.append(record(path))
    plt.close(fig)
    return paths


def render_figures(output, cases, candidate, result_records):
    V.configure_font()
    figure_dir = output / "wall_figures"
    figure_dir.mkdir(parents=True, exist_ok=True)
    labels = ("M2", "MO0", candidate["id"])
    friendly = {"M2": "历史M2", "MO0": "MO0同期对照", candidate["id"]: candidate["id"]}
    figures, assembled, norms = {}, {}, {}
    for uid in REPRESENTATIVES:
        case = cases[uid]
        xyz, orientation = V.orient(np.asarray(case["pos"], dtype=np.float64) * case["coord_scale_scalar"])
        truth = np.asarray(case["y_raw"], dtype=np.float64)
        fields = {}
        for checkpoint in ("best", "last"):
            for label in labels:
                with np.load(output / "predictions" / label / checkpoint / f"{slug(uid)}.npz") as payload:
                    fields[(checkpoint, label)] = payload["prediction_pa"]
        maximum = max(float(truth.max()), *(float(value.max()) for value in fields.values()))
        error_max = max(float(np.abs(value - truth).max()) for value in fields.values())
        wss_norm = colors.PowerNorm(gamma=.5, vmin=0., vmax=max(maximum, 1e-12))
        error_norm = colors.Normalize(vmin=-max(error_max, 1e-12), vmax=max(error_max, 1e-12))
        norms[uid] = wss_norm
        assembled[uid] = (xyz, truth, fields)
        figures[uid] = {"orientation": orientation, "color_scale": {"wss_pa": [0., maximum],
            "signed_error_pa": [-error_max, error_max], "shared_across": "CFD/M2/MO0/candidate and best/last within this case",
            "wss_color_mapping": "PowerNorm gamma=0.5", "display_clipping": False}, "checkpoints": {}}
        for checkpoint in ("best", "last"):
            per_model_metrics = {label: result_records[label][checkpoint]["cases"][uid]["physical_metrics"] for label in labels}
            files = []
            for kind in ("wss", "error", "hotspot"):
                values = [truth, *(fields[(checkpoint, label)] for label in labels)] if kind == "wss" else [fields[(checkpoint, label)] - truth for label in labels]
                if kind == "hotspot":
                    values = []
                    for label in labels:
                        code, iou = hotspot_codes(truth, fields[(checkpoint, label)])
                        require(abs(iou - per_model_metrics[label]["top10_iou"]) <= 1e-12, "plotted hotspot masks differ from verified evaluator")
                        values.append(code)
                columns = len(values)
                fig = plt.figure(figsize=(4.4 * columns, 9.4))
                cmap = (colors.ListedColormap(["#d5d9df", "#009e73", "#d55e00", "#0072b2"])
                        if kind == "hotspot" else "turbo" if kind == "wss" else "coolwarm")
                norm = (colors.BoundaryNorm([-.5, .5, 1.5, 2.5, 3.5], 4) if kind == "hotspot"
                        else wss_norm if kind == "wss" else error_norm)
                for row, view in enumerate(VIEWS):
                    for column, values_at_points in enumerate(values):
                        label = None if kind == "wss" and column == 0 else labels[column - 1 if kind == "wss" else column]
                        if label is None:
                            title = "CFD真值"
                        else:
                            metrics = per_model_metrics[label]
                            title = (f"{friendly[label]}  R²={metrics['r2']:.3f} · MAE={metrics['mae']:.2f} Pa"
                                     if kind == "wss" else f"{friendly[label]} − CFD" if kind == "error"
                                     else f"{friendly[label]}  IoU={metrics['top10_iou']:.3f}")
                        ax = fig.add_subplot(2, columns, row * columns + column + 1, projection="3d")
                        V.wall(ax, xyz, values_at_points, norm, cmap, view, title if row == 0 else "反面 · " + title)
                fig.subplots_adjust(left=.015, right=.985, bottom=.14, top=.87, wspace=-.10, hspace=.0)
                if kind == "hotspot":
                    handles = [Line2D([], [], marker="o", linestyle="", markersize=7, color=color, label=label)
                               for color, label in zip(cmap.colors, ("其余壁面", "命中真值热点", "漏检真值热点", "额外预测热点"))]
                    fig.legend(handles=handles, loc="lower center", bbox_to_anchor=(.5, .073), ncol=4, frameon=False)
                    color_text = "各场自身q90定义热点；与原legacy_vertex指标一致，不表示幅值准确"
                else:
                    color_axis = fig.add_axes([.22, .088, .56, .018])
                    bar = fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=cmap), cax=color_axis, orientation="horizontal")
                    bar.set_label("WSS (Pa) · √WSS颜色映射" if kind == "wss" else "有符号误差 (Pa) · 蓝色低估 / 红色高估")
                    color_text = "同例所有模型与best/last共用完整色标，无显示截断"
                fig.suptitle(f"{uid}\n{checkpoint} · 峰值帧{case['peak_step']} · 原壁面{len(truth):,}点 · {candidate['label']}", fontsize=14, y=.97)
                fig.text(.5, .025, color_text + "；三例固定示例，单seed，不代表总体收益。", ha="center", fontsize=9)
                files.extend(_save_figure(fig, figure_dir, f"{slug(uid)}__{checkpoint}__{kind}"))
            figures[uid]["checkpoints"][checkpoint] = files
    overviews = {}
    for checkpoint in ("best", "last"):
        fig = plt.figure(figsize=(17., 16.))
        for row, uid in enumerate(REPRESENTATIVES):
            xyz, truth, fields = assembled[uid]
            for column, (label, value) in enumerate([("CFD", truth), *((label, fields[(checkpoint, label)]) for label in labels)]):
                ax = fig.add_subplot(3, 4, row * 4 + column + 1, projection="3d")
                V.wall(ax, xyz, value, norms[uid], "turbo", VIEWS[0], "CFD真值" if label == "CFD" else friendly[label])
            fig.text(.013, .84 - row * .268, uid, fontsize=10, rotation=90, va="center")
            color_axis = fig.add_axes([.945, .68 - row * .274, .012, .15])
            fig.colorbar(plt.cm.ScalarMappable(norm=norms[uid], cmap="turbo"), cax=color_axis).set_label("WSS Pa")
        fig.subplots_adjust(left=.04, right=.94, bottom=.08, top=.90, wspace=-.16, hspace=.005)
        fig.suptitle(f"M2优化：预先固定的三例原壁面WSS对照\n{checkpoint} · {candidate['id']}：{candidate['label']}", fontsize=16, y=.97)
        fig.text(.5, .038, "每行同病例共用色标；不同病例范围可不同。best/last范围一致。单seed例图不替代总体与保护线判断。", ha="center", fontsize=10)
        overviews[checkpoint] = _save_figure(fig, figure_dir, f"wall_overview_{checkpoint}")
    return {"cases": figures, "overviews": overviews}


def _verify_current_training_sources():
    snapshots = {}
    for phase in ("base", "combination"):
        if phase == "combination" and not COMBINATIONS.exists():
            continue
        state_path = EXP / ("queue_status.json" if phase == "base" else "combination_queue_status.json")
        if phase == "combination" and not json.loads(COMBINATIONS.read_text())["arms"]:
            continue
        state = json.loads(state_path.read_text())
        current = fingerprints(phase)
        require(state.get("status") == "complete" and state.get("source_sha256") == state.get("source_sha256_end") == current,
                f"{phase}: completed unchanged training sources required before diagnostic inference")
        snapshots[phase] = current
    return snapshots


def self_test_boundary_review():
    """Exercise the local exception against pinned CPU evidence and rejection cases."""
    from copy import deepcopy
    probe = load_boundary_probe()
    prediction, truth = probe["prediction"].copy(), probe["truth"].copy()
    prediction[55137] = probe["upper"]
    saved = {"distribution": {"n": 57531, "pred_above_one_fraction": 13859 / 57531}, "overall": {"r2": .5}}
    actual = deepcopy(saved)
    actual["distribution"]["pred_above_one_fraction"] = 13860 / 57531
    kwargs = {"prediction": prediction, "truth": truth, "context": BOUNDARY_CONTEXT,
              "source_hashes": BOUNDARY_SOURCE_HASHES, "other_space_strict_passed": True}
    check = metric_check(actual, saved, **kwargs)
    require(check["passed"] and not check["strict_passed"] and check["strict_failed_checks"] == 1
            and len(check["failed_checks"]) == len(check["boundary_reviews"]) == 1, "one-vertex boundary review failed")
    require(metric_check(saved, saved)["strict_passed"], "ordinary strict verification changed")
    rejected = 0
    def reject(candidate=actual, reference=saved, **changes):
        nonlocal rejected
        require(not metric_check(candidate, reference, **{**kwargs, **changes})["passed"], "invalid boundary exception accepted")
        rejected += 1
    second = int(np.flatnonzero(prediction < .9)[0])
    two = prediction.copy(); two[second] = probe["upper"]
    extra = deepcopy(actual); extra["distribution"]["pred_above_one_fraction"] = 13861 / 57531
    reject(extra, prediction=two)
    no_boundary = prediction.copy(); no_boundary[55137] = 1.001
    reject(prediction=no_boundary)
    wrong_vertex = probe["prediction"].copy(); wrong_vertex[second] = probe["upper"]
    reject(prediction=wrong_vertex)
    changed_n = deepcopy(actual); changed_n["distribution"]["n"] = 57530
    reject(changed_n)
    continuous = deepcopy(actual); continuous["overall"]["r2"] = .4
    reject(continuous)
    reject(other_space_strict_passed=False)
    reject(context=("MO-P4", "last", BOUNDARY_CONTEXT[2], "log_z"))
    reject(context=(*BOUNDARY_CONTEXT[:3], "Pa"))
    reject(source_hashes={**BOUNDARY_SOURCE_HASHES, "metrics": "changed"})
    nan_truth = truth.copy(); nan_truth[0] = np.nan
    reject(truth=nan_truth)
    wrong_truth = truth.copy(); wrong_truth[0] += .01
    reject(truth=wrong_truth)
    reject(prediction=probe["prediction"])
    noninteger = deepcopy(actual); noninteger["distribution"]["pred_above_one_fraction"] = 13859.5 / 57531
    reject(noninteger)
    print(f"CPU boundary review: pinned 8-repeat evidence, strict/one-vertex positives, {rejected} rejection cases passed")


def self_test():
    """CPU behavior checks for weight reconstruction, spatial bins and candidate fallback."""
    from training_wss_min.baseline_models import LocalQueryDecoder
    torch.set_num_threads(1)
    torch.manual_seed(12)
    support_pos = torch.rand(30, 3)
    query_pos = torch.rand(21, 3)
    support_x = torch.randn(30, 8)
    support_input, query_input = torch.randn(30, 25), torch.randn(21, 25)
    decoder = LocalQueryDecoder(8, 25, 1, 3, residual=True, attr_dim=4).eval()
    with torch.no_grad():
        decoder.correction[-1].weight.normal_(std=.2)
        decoder.res_out.weight.normal_(std=.1)
        args = (support_x, support_pos, query_pos, torch.zeros(30, dtype=torch.long), torch.zeros(21, dtype=torch.long))
        kwargs = {"support_input_x": support_input, "query_input_x": query_input,
                  "geo_attr": (support_input[:, [4, 14, 15, 16]], query_input[:, [4, 14, 15, 16]])}
        actual = decoder(*args, **kwargs)
        entropy, effective, error = reconstruction_weights(decoder, args, kwargs, actual)
    require(error < 1e-6 and np.all(entropy >= 0.) and np.all(effective <= 3. + 1e-5), "invalid 3NN reconstruction")
    gate = np.ones((12, 96), dtype=np.float32)
    gate[:, 32:64] += np.arange(12, dtype=np.float32)[:, None] / 12
    spatial = gate_spatial_summary(gate, np.linspace(0., 1., 12), 3)
    require(spatial["point_mean_gate_per_scale"][0]["std"] == 0. and spatial["point_mean_gate_per_scale"][1]["std"] > 0., "gate spatial reduction is wrong")
    require(sum(row["support_points"] for row in spatial["by_original_abscissa_norm"]) == 12, "gate bins dropped endpoints")
    entries = {aid: {"physical_r2_cb": score, "mae": 2., "evidence": {"valid_scientific_result": True}}
               for aid, score in (("MO0", .6), ("MO-L1", .64), ("MO-S1", .65))}
    reports = {"best": {"arms": entries}, "last": {"arms": entries}, "selection": {"ranking": [], "candidates": {}}}
    require(select_visual_candidate(reports)["id"] == "MO-S1" and not select_visual_candidate(reports)["qualified"], "raw fallback selection changed")
    reports["selection"]["ranking"] = ["MO-L1"]
    require(select_visual_candidate(reports)["id"] == "MO-L1", "unqualified raw result replaced qualified candidate")
    code, iou = hotspot_codes(np.arange(10.), np.arange(10.))
    require(iou == 1. and np.sum(code == 1) == 1, "hotspot masks changed")
    with tempfile.TemporaryDirectory(prefix="m2_diagnostic_cpu_") as temporary:
        path = Path(temporary) / "values.npz"
        save_arrays(path, gate=gate, entropy=entropy)
        with np.load(path) as payload:
            require(np.array_equal(payload["gate"], gate), "raw NPZ round-trip changed values")
    self_test_boundary_review()
    print("CPU diagnostic self-test passed: learned3NN reconstruction, spatial gates, candidate fallback, hotspot masks, raw NPZ")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--self-test", action="store_true")
    parser.add_argument("--smoke", action="store_true", help="Only historical M2 best on the first fixed example; separate output")
    args = parser.parse_args()
    if args.self_test:
        self_test()
        return
    require(bool(os.environ.get("SLURM_JOB_ID")), "GPU diagnostics require Slurm; this tool does not submit jobs")
    require(torch.cuda.is_available(), "an allocated CUDA GPU is required")
    torch.set_num_threads(2)
    output = EXP / "diagnostics_smoke" if args.smoke else OUT
    output.mkdir(parents=True, exist_ok=True)
    provenance = {"passed": False, "smoke": args.smoke, "started_at": stamp(),
                  "job_id": os.environ["SLURM_JOB_ID"], "script": record(Path(__file__)),
                  "limits": LIMITS, "representative_cases_preselected": list(REPRESENTATIVES), "runs": {}, "inputs": {}}
    provenance.update(metric_summary(provenance))
    save_json(output / "provenance.json", provenance)
    try:
        if args.smoke:
            runs, checkpoints, candidate, before = {"M2": ANCHOR_RUN}, ("best",), None, {"base": fingerprints()}
        else:
            before = _verify_current_training_sources()
            reports = R.report_all(write=False)
            require(all(reports[checkpoint]["matrix_finalized"] for checkpoint in ("best", "last")),
                    "all base and required combination evaluations must be finalized before full diagnostics")
            candidate = select_visual_candidate(reports)
            runs = {"M2": ANCHOR_RUN}
            runs.update({aid: RUNS / entry["run_name"] for aid, entry in reports["best"]["arms"].items()
                         if entry["evidence"]["valid_scientific_result"] and reports["last"]["arms"][aid]["evidence"]["valid_scientific_result"]})
            checkpoints = ("best", "last")
            provenance["visual_candidate"] = candidate
            provenance["report_code"] = record(Path(R.__file__))
        provenance["source_sha256_before"] = before
        anchor = C.ExpConfig.from_json(ANCHOR_CONFIG)
        stats = D.load_wss_stats(anchor.data.wss_stats_path)
        cases = D.load_partition(anchor.data.split_path, "test", stats, strict=True,
            target="wss", target_normalization=anchor.data.target_normalization,
            data_root=anchor.data.data_root, required_frame_version=anchor.data.required_frame_version,
            extra_point_features=C.v6_point_features(anchor), point_features_root=anchor.data.point_features_root)
        if args.smoke:
            cases = [case for case in cases if case["unit_id"] == REPRESENTATIVES[0]]
        else:
            require(len(cases) == 34 and set(REPRESENTATIVES) <= {case["unit_id"] for case in cases}, "test34/example identities differ")
        by_case = {case["unit_id"]: case for case in cases}
        provenance["partition"] = "test34" if not args.smoke else "one preselected test case"
        provenance["case_count"] = len(cases)
        provenance["geometry"] = {}
        for case in cases:
            uid = case["unit_id"]
            bundle = Path(case["bundle_path"])
            sidecar = Path(anchor.data.point_features_root) / case["cohort"] / case["case"] / "features.npz"
            for path in (bundle, sidecar):
                provenance["inputs"][str(path)] = sha(path)
            with np.load(bundle, allow_pickle=True) as payload:
                node_ids = payload["wall_node_id_cas"]
            geometry = save_arrays(output / "geometry" / f"{slug(uid)}.npz",
                wall_node_id_cas=node_ids, pos_norm=case["pos"], wall_coords_raw=case["wall_coords_raw"],
                position_centered_mm=case["pos"] * case["coord_scale_scalar"],
                truth_pa=case["y_raw"], truth_norm=case["y_norm"], abscissa_norm=case["abscissa_norm"])
            provenance["geometry"][uid] = {"arrays": geometry, "wall_points": len(node_ids), "coord_scale_mm": case["coord_scale_scalar"],
                                            "peak_step": int(case["peak_step"])}
        for label, run in runs.items():
            provenance["runs"][label] = {}
            for checkpoint_name in checkpoints:
                started = time.monotonic()
                print(f"{stamp()} diagnose {label}/{checkpoint_name}: {len(cases)} full-wall cases", flush=True)
                cfg, feature_stats, model, checkpoint = E.load_model_from_run(run, "cuda", checkpoint_name)
                collector = MechanismCollector(model)
                try:
                    require((cfg.data.target, cfg.data.timesteps, cfg.model.out_dim, cfg.eval.fixed_support,
                             cfg.eval.full_cloud_query, cfg.eval.support_seed, cfg.eval.query_chunk_size,
                             cfg.eval.surface_metric_mode) == ("wss", "peak", 1, True, True, 1234, 16384, "legacy_vertex"),
                            f"{label}: changed original WSS evaluation protocol")
                    require(tuple(cfg.data.input_features) == tuple(anchor.data.input_features)
                            and cfg.data.data_root == anchor.data.data_root and cfg.data.point_features_root == anchor.data.point_features_root,
                            f"{label}: changed M2 geometry/features")
                    run_stats = E.load_wss_stats_for_run(run)
                    saved_metrics = json.loads((run / "eval" / f"ckpt_{checkpoint_name}" / "metrics.json").read_text())["test"]
                    source_files = [run / f"ckpt_{checkpoint_name}.pt", run / "config.json", run / "feature_stats.json",
                                    run / "wss_global_stats.json", run / "eval" / f"ckpt_{checkpoint_name}" / "metrics.json"]
                    sources = {str(path): sha(path) for path in source_files}
                    item = {"run_name": str(run.relative_to(RUNS)), "checkpoint_epoch": int(checkpoint["epoch"]),
                            "sources": sources, "model": parameter_diagnostics(model), "cases": {}, "passed": False}
                    provenance["runs"][label][checkpoint_name] = item
                    for case in cases:
                        uid = case["unit_id"]
                        collector.reset()
                        result = E._evaluate_partition_frame(model, [case], cfg, feature_stats, run_stats,
                                                             "cuda", return_predictions=True)
                        prediction_norm = np.asarray(result["_pred_norm_by_case"][0], dtype=np.float64)
                        prediction_pa = np.clip(D.denormalize_wss(prediction_norm, run_stats), 0., None)
                        require(np.isfinite(prediction_norm).all() and np.isfinite(prediction_pa).all(), "non-finite prediction arrays")
                        checks = {"Pa": metric_check(result["per_case"][uid], saved_metrics["per_case"][uid])}
                        checks["log_z"] = metric_check(result["normalized"]["per_case"][uid],
                            saved_metrics["normalized"]["per_case"][uid], prediction=prediction_norm, truth=case["y_norm"],
                            context=(label, checkpoint_name, uid, "log_z"), other_space_strict_passed=checks["Pa"]["strict_passed"],
                            source_hashes={"metrics": sources[str(source_files[-1])], "checkpoint": sources[str(source_files[0])],
                                           "config": sources[str(source_files[1])]})
                        for check in checks.values():
                            for review in check["boundary_reviews"]:
                                provenance["inputs"].update(review["evidence_inputs"])
                        if not all(check["passed"] for check in checks.values()):
                            # Preserve the failing raw forward before aborting for independent review.
                            failed_file = save_arrays(output / "predictions" / label / checkpoint_name / f"{slug(uid)}.npz",
                                prediction_norm=prediction_norm, prediction_pa=prediction_pa)
                            item["cases"][uid] = {"wall_points": len(prediction_norm), "arrays": failed_file,
                                                   "metric_verification": checks, "passed": False}
                            raise RuntimeError(f"{label}/{checkpoint_name}/{uid}: saved original metrics not reproduced: {checks}")
                        mechanism, arrays = collector.case_result(cfg, case, prediction_norm, prediction_pa, run_stats)
                        prediction_file = save_arrays(output / "predictions" / label / checkpoint_name / f"{slug(uid)}.npz",
                                                      prediction_norm=prediction_norm, prediction_pa=prediction_pa, **arrays)
                        metrics = result["per_case"][uid]
                        item["cases"][uid] = {"wall_points": len(prediction_norm), "arrays": prediction_file,
                            "metric_verification": checks, "mechanisms": mechanism,
                            "physical_metrics": {"r2": metrics["overall"]["r2"], "mae": metrics["overall"]["mae"],
                                                 "top10_iou": metrics["hotspot"]["top10_iou"]}}
                    require(all(sha(Path(path)) == digest for path, digest in sources.items()), "run artifacts changed during diagnostics")
                    item.update(passed=True, seconds=time.monotonic() - started)
                    save_json(output / "provenance.json", provenance)
                    print(f"verified {label}/{checkpoint_name} in {item['seconds']:.1f}s", flush=True)
                finally:
                    collector.close()
                    del collector, model, checkpoint
                    gc.collect()
                    torch.cuda.empty_cache()
        if not args.smoke:
            provenance["figures"] = render_figures(output, by_case, candidate, provenance["runs"])
        after = {phase: fingerprints(phase) for phase in before}
        require(after == before, "frozen model sources changed during diagnostic inference")
        require(all(sha(Path(path)) == digest for path, digest in provenance["inputs"].items()), "original geometry inputs changed")
        if not args.smoke:
            require(sha(Path(R.__file__)) == provenance["report_code"]["sha256"], "candidate report implementation changed")
        provenance.update(passed=True, completed_at=stamp(), source_sha256_after=after,
            evaluations_verified=sum(len(run) for run in provenance["runs"].values()))
        provenance.update(metric_summary(provenance))
        lines = ["# M2优化训练后诊断", "", LIMITS, "",
                 f"模型/检查点评估：{provenance['evaluations_verified']}；每次{len(cases)}例完整壁面。",
                 f"原逐病例数值核验：{provenance['original_per_case_numeric_checks']}项；原容差直接通过"
                 f"{provenance['strict_passed_numeric_checks']}项；严格失败{provenance['strict_failed_numeric_checks']}项；"
                 f"单顶点阈值边界后验审核{provenance['boundary_review_count']}项。",
                 "原容差核验不表示预测逐位相等。任何边界审核均保留strict_passed=false、原失败数值和probe来源SHA；"
                 "只覆盖已重复验证的特定病例/检查点/log_z阈值，不更改连续指标容差、原metrics、工作簿或筛选值。", "",
                 "参数、残差尺度、解码β、完整query熵/有效邻居、输出残差和support门控分布见provenance.json。",
                 "geometry/*.npz保存原点ID/坐标/真值；predictions/<arm>/<checkpoint>/*.npz保存全部逐点预测及启用机制的原始数组。", ""]
        if candidate is not None:
            lines += [f"图中候选：{candidate['id']}（{candidate['label']}）。", "",
                      "[best总览](wall_figures/wall_overview_best.png) · [last总览](wall_figures/wall_overview_last.png)", ""]
            if not candidate["qualified"]:
                lines += ["该候选仅用于展示raw最高R²结果，不替换M2；未通过项：", "",
                          *[f"- {issue}" for issue in candidate["issues"]], ""]
        (output / "README.md").write_text("\n".join(lines), encoding="utf-8")
        print(json.dumps({"passed": True, "output": str(output), "evaluations": provenance["evaluations_verified"],
                          "case_metric_checks": provenance["original_per_case_numeric_checks"],
                          "strict_passed_numeric_checks": provenance["strict_passed_numeric_checks"],
                          "strict_failed_numeric_checks": provenance["strict_failed_numeric_checks"],
                          "boundary_review_count": provenance["boundary_review_count"]}, ensure_ascii=False), flush=True)
    except Exception as exc:
        provenance.update(error=f"{type(exc).__name__}: {exc}", failed_at=stamp())
        raise
    finally:
        provenance.update(metric_summary(provenance))
        save_json(output / "provenance.json", provenance)


if __name__ == "__main__":
    main()

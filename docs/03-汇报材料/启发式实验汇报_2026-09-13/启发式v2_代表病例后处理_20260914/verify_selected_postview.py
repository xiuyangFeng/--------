#!/usr/bin/env python3
"""Independently verify the nine cached postview cases, without inference.

Run with the GNN environment's Python. Only verification.json is written.
Raw arrays, coordinates, topology, display formulas and CSV samples are checked
against original sources; expected selfmax fields do not use the export helper.
"""
from __future__ import annotations

import argparse
import datetime
import gzip
import hashlib
import json
import sys
from pathlib import Path

import h5py
import numpy as np
import vtk
from scipy.spatial import cKDTree
from vtk.util.numpy_support import vtk_to_numpy

OUT = Path(__file__).resolve().parent
ROOT = next(p for p in OUT.parents if (p / "training_wss_min").is_dir())
sys.path.insert(0, str(ROOT))
from tools.cfdpost_cloud_export.display_fields import (  # noqa: E402
    fit_selfmax,
    normalization_field_data,
    selfmax_fields,
)

DATA = ROOT / "data_wss_v5/views/wss_min_view_v1"
ROLES = ("best", "median", "worst")
KEYS = {"X5X11": "A_X5X11", "PF6": "B_PF6", "VF6": "B_VF6"}
PREFIX = {"X5X11": "wss", "PF6": "pressure", "VF6": "speed"}
SCOPE = {"X5X11": "wall", "PF6": "wall_union_interior", "VF6": "interior"}


def read_vtp(path):
    reader = vtk.vtkXMLPolyDataReader()
    reader.SetFileName(str(path))
    reader.Update()
    poly = reader.GetOutput()
    if poly.GetPoints() is None:
        raise ValueError(f"VTP has no points: {path}")
    pd, fd = poly.GetPointData(), poly.GetFieldData()
    fields = {}
    for i in range(fd.GetNumberOfArrays()):
        a = fd.GetAbstractArray(i)
        fields[a.GetName()] = a.GetValue(0) if a.GetNumberOfValues() == 1 else [
            a.GetValue(j) for j in range(a.GetNumberOfValues())
        ]
    return {
        "xyz": vtk_to_numpy(poly.GetPoints().GetData()).copy(),
        "arrays": {pd.GetArrayName(i): vtk_to_numpy(pd.GetArray(i)).copy()
                   for i in range(pd.GetNumberOfArrays())},
        "triangles": vtk_to_numpy(poly.GetPolys().GetConnectivityArray()).copy().reshape(-1, 3),
        "vertices": poly.GetNumberOfVerts(),
        "field_data": fields,
    }


def load_npz(path):
    with np.load(path, allow_pickle=False) as z:
        return {k: z[k] for k in z.files}


def derived_selfmax(cfd, pred, prefix, cfd_max, pred_max):
    """Independent formulas for this batch's finite, nonzero denominators."""
    a, b = np.asarray(cfd, float) / cfd_max, np.asarray(pred, float) / pred_max
    error = b - a
    return {
        f"{prefix}_cfd_selfmax": a,
        f"{prefix}_pred_selfmax": b,
        f"{prefix}_selfmax_error_pred_minus_cfd": error,
        f"{prefix}_selfmax_abs_error": abs(error),
        f"{prefix}_cfd_selfmax_valid": np.isfinite(a).astype(np.uint8),
        f"{prefix}_pred_selfmax_valid": np.isfinite(b).astype(np.uint8),
        f"{prefix}_selfmax_error_valid": (np.isfinite(a) & np.isfinite(b)).astype(np.uint8),
    }


class Audit:
    def __init__(self):
        self.checks = []
        self.cases = []
        self.hashes = {}
        self.helper_checks = 0

    def check(self, label, passed, detail=None):
        item = {"check": label, "passed": bool(passed)}
        if detail is not None:
            item["detail"] = detail
        self.checks.append(item)
        if not passed:
            print("FAILED", label, detail, flush=True)

    def equal(self, label, a, b):
        a, b = np.asarray(a), np.asarray(b)
        same_shape = a.shape == b.shape
        passed = same_shape and np.array_equal(a, b, equal_nan=True)
        difference = None
        if same_shape and a.size and np.issubdtype(a.dtype, np.number):
            difference = float(np.nanmax(np.abs(a.astype(float) - b.astype(float))))
        self.check(label, passed, {
            "actual_shape": list(a.shape), "expected_shape": list(b.shape),
            "max_absolute_difference": difference,
        })

    def close(self, label, a, b, tolerance):
        self.check(label, abs(float(a) - float(b)) <= tolerance, {
            "actual": float(a), "expected": float(b), "absolute_tolerance": tolerance,
        })

    def sha(self, path):
        path = str(Path(path).resolve())
        if path not in self.hashes:
            h = hashlib.sha256()
            with open(path, "rb") as f:
                for block in iter(lambda: f.read(1 << 20), b""):
                    h.update(block)
            self.hashes[path] = h.hexdigest()
        return self.hashes[path]

    def compare_arrays(self, label, actual, expected):
        self.check(label + "/field_names", set(actual) == set(expected), {
            "actual": sorted(actual), "expected": sorted(expected),
        })
        for name, values in expected.items():
            if name in actual:
                self.equal(label + "/" + name, actual[name], values)

    def csv(self, label, path, xyz, arrays):
        columns = {"x_mm": xyz[:, 0], "y_mm": xyz[:, 1], "z_mm": xyz[:, 2]}
        for name, values in arrays.items():
            if values.ndim == 1:
                columns[name] = values
            else:
                columns.update({f"{name}_{j}": values[:, j] for j in range(values.shape[1])})
        n = len(xyz)
        indices = set(np.linspace(0, n - 1, min(n, 64), dtype=int).tolist())
        samples, rows = [], 0
        opener = gzip.open if path.suffix == ".gz" else open
        with opener(path, "rt", encoding="utf-8") as f:
            header = f.readline().strip().split(",")
            self.check(label + "/header", set(header) == set(columns), {
                "actual": header, "expected": list(columns),
            })
            for index, line in enumerate(f):
                rows += 1
                if index in indices:
                    samples.append((index, np.fromstring(line.strip(), sep=",")))
        self.check(label + "/row_count", rows == n, {"actual": rows, "expected": n})
        valid_header = set(header) == set(columns)
        differences = []
        passed = valid_header and len(samples) == len(indices)
        if valid_header:
            for index, row in samples:
                expected = np.asarray([columns[k][index] for k in header], float)
                same_shape = row.shape == expected.shape
                passed &= same_shape and np.allclose(row, expected, rtol=1e-10, atol=1e-12, equal_nan=True)
                if same_shape:
                    differences.append(float(np.nanmax(abs(row - expected))))
        self.check(label + "/data_sample", passed, {
            "sample_count": len(samples), "rtol": 1e-10, "atol": 1e-12,
            "reason": "CSV uses 12 significant decimal digits",
            "max_absolute_difference": max(differences, default=None),
        })

    def normalization(self, label, vtp, contract, cfd_max, pred_max):
        fd, prefix = vtp["field_data"], contract["prefix"]
        try:
            stored = json.loads(fd["display_normalization"])
            self.check(label + "/contract_json", stored == contract)
        except (KeyError, TypeError, json.JSONDecodeError) as error:
            self.check(label + "/contract_json", False, str(error))
        for name, value in (("cfd", cfd_max), ("pred", pred_max)):
            self.close(label + f"/{name}_max", fd.get(f"{prefix}_{name}_max", np.nan), value, 0)
            self.check(label + f"/{name}_max_valid", fd.get(f"{prefix}_{name}_max_valid") == 1)

    def helper_branches(self):
        start = len(self.checks)
        for title, a, b in [
            ("zero", np.zeros(3), np.zeros(3)),
            ("near_zero", np.full(3, 1e-15), np.full(3, -1e-15)),
        ]:
            contract = fit_selfmax(a, b, "t", "test", "Pa")
            fields = selfmax_fields(a, b, contract)
            self.check("helper/" + title, not contract["cfd_denominator"]["valid"]
                       and not contract["pred_denominator"]["valid"]
                       and all(np.all(fields[k] == 0) for k in fields if k.endswith("valid"))
                       and all(np.isnan(fields[k]).all() for k in fields if not k.endswith("valid")))
        for invalid in (np.nan, np.inf, -np.inf):
            a, b = np.array([1., invalid]), np.array([1., 2.])
            contract = fit_selfmax(a, b, "t", "test", "Pa")
            fields = selfmax_fields(a, b, contract)
            label = "helper/nonfinite_" + str(invalid)
            self.check(label, contract["cfd_max"] is None
                       and contract["cfd_denominator"]["reason"] == "nonfinite_source"
                       and np.isnan(fields["t_cfd_selfmax"]).all()
                       and np.array_equal(fields["t_pred_selfmax"], [.5, 1.])
                       and not fields["t_selfmax_error_valid"].any())
            metadata = normalization_field_data(contract)
            self.check(label + "/metadata", np.isnan(metadata["t_cfd_max"])
                       and json.loads(metadata["display_normalization"])["cfd_max"] is None)
        for title, a, b in [
            ("signed", np.array([-4., 0., 2.]), np.array([-6., 0., 3.])),
            ("all_negative_signed_max", np.array([-4., -2.]), np.array([-6., -3.])),
        ]:
            contract = fit_selfmax(a, b, "t", "test", "Pa")
            expected = derived_selfmax(a, b, "t", float(a.max()), float(b.max()))
            actual = selfmax_fields(a, b, contract)
            self.check("helper/" + title, contract["cfd_max"] == float(a.max())
                       and contract["pred_max"] == float(b.max())
                       and all(np.array_equal(actual[k], v) for k, v in expected.items()))
        contract = fit_selfmax([1., 2.], [2., 4.], "t", "test", "Pa")
        fields = selfmax_fields([np.nan, 1.], [2., np.inf], contract)
        self.check("helper/nonfinite_application", np.array_equal(fields["t_cfd_selfmax_valid"], [0, 1])
                   and np.array_equal(fields["t_pred_selfmax_valid"], [1, 0])
                   and np.isnan(fields["t_selfmax_error_pred_minus_cfd"]).all())
        for label, call in [
            ("vector_fit_rejected", lambda: fit_selfmax(np.ones((2, 3)), np.ones((2, 3)), "t", "test", "m/s")),
            ("empty_fit_rejected", lambda: fit_selfmax([], [], "t", "test", "Pa")),
            ("shape_fit_rejected", lambda: fit_selfmax([1., 2.], [1.], "t", "test", "Pa")),
            ("vector_apply_rejected", lambda: selfmax_fields(np.ones((2, 3)), np.ones((2, 3)), contract)),
            ("shape_apply_rejected", lambda: selfmax_fields([1.], [1., 2.], contract)),
        ]:
            try:
                call()
            except ValueError:
                self.check("helper/" + label, True)
            else:
                self.check("helper/" + label, False)
        self.helper_checks = len(self.checks) - start

    def case(self, model, role, summary):
        d, label = OUT / model / role, f"{model}/{role}"
        m = json.loads((d / "manifest.json").read_text())
        start = len(self.checks)
        self.check(label + "/schema", m.get("schema_version") == 3)
        uid, prefix, run = m["case_id"], PREFIX[model], Path(m["run"])
        p = load_npz(run / "eval/ckpt_best/predictions/test" / uid / "predictions.npz")
        b, v = load_npz(DATA / uid / "bundle.npz"), load_npz(DATA / uid / "volume.npz")
        official = json.loads((run / "eval/ckpt_best/metrics.json").read_text())["test"]["per_case"][uid]["overall"]
        wall, interior = b["wall_coords_aligned_mm"].astype(float), v["vol_coords_aligned_mm"].astype(float)
        nw = len(wall)
        self.check(label + "/frame", m["peak_step"] == int(b["peak_step"]) == int(v["peak_step"]) == 1162
                   and m["peak_index"] == int(np.where(b["steps"] == 1162)[0][0]) == 21
                   and m["frame"] == str(b["transform_frame_version"]) == str(v["transform_frame_version"]) == "v5_atlas_frame_v1"
                   and m["coordinate_unit"] == "mm")
        self.check(label + "/checkpoint", m["seed"] == 1234 and m["checkpoint"] == "ckpt_best.pt" and m["split"] == "test")
        rec = next(a for a in summary[KEYS[model]]["cases"] if a["case_id"] == uid)
        self.equal(label + "/selection_r2", m["selection"]["r2_case_mean"], rec["r2"])
        self.check(label + "/selection_separate", "three-seed" in m["selection"]["basis"]
                   and "not ensemble" in m["selection"]["basis"] and m["visualization_metrics"]["seed"] == 1234)
        for name, path in m["files"].items():
            if path is not None:
                self.check(label + "/relative_pointer/" + name, not Path(path).is_absolute() and (d / path).is_file())
        for link in d.glob("source_*.npz"):
            self.check(label + "/source_link/" + link.name, link.exists())
        point = read_vtp(d / m["files"]["pointcloud"])
        self.check(label + "/pointcloud_vertices", len(point["triangles"]) == 0 and point["vertices"] == len(point["xyz"]))
        if model == "X5X11":
            q = p["row_index"]
            true, pred = p["true_pa"].astype(float), p["pred_pa"].astype(float)
            xyz = wall[q]
            self.equal(label + "/truth_mother", true, b["wall_wss"][21][q])
            expected = {"wss_cfd_pa": true, "wss_pred_pa": pred,
                        "wss_error_pred_minus_cfd_pa": pred - true, "wss_abs_error_pa": abs(pred - true),
                        "wss_cfd_norm": p["true_norm"].astype(float), "wss_pred_norm": p["pred_norm"].astype(float), "row_index": q}
            order = np.argsort(q)
            native_expected = {k: a[order] for k, a in expected.items()}
        else:
            q, kind = p["query_idx"], p["point_kind"]
            raw_true, raw_pred = p["true_raw"].astype(float), p["pred_raw"].astype(float)
            xyz = np.concatenate([wall, interior])[q]
            self.equal(label + "/point_kind", kind, (q >= nw).astype(kind.dtype))
            dist = np.concatenate([np.zeros(nw), v["vol_dist_to_wall_mm"]])[q]
            if model == "VF6":
                self.check(label + "/velocity_interior_only", np.all(kind == 1) and len(q) == len(interior))
                self.equal(label + "/truth_mother_vector", raw_true, v["vol_velocity_aligned_peak"][q - nw])
                true, pred = np.linalg.norm(raw_true, axis=1), np.linalg.norm(raw_pred, axis=1)
                expected = {"speed_cfd": true, "speed_pred": pred, "speed_error_pred_minus_cfd": pred - true,
                            "speed_abs_error": abs(pred - true), "velocity_cfd_vector": raw_true,
                            "velocity_pred_vector": raw_pred, "velocity_error_vector": raw_pred - raw_true,
                            "velocity_vector_error_norm": np.linalg.norm(raw_pred - raw_true, axis=1)}
                for j, a in enumerate("xyz"):
                    expected["velocity_cfd_" + a] = raw_true[:, j]
                    expected["velocity_pred_" + a] = raw_pred[:, j]
                self.check(label + "/no_velocity_surface", all(m["files"][k] is None for k in
                           ["gaussian_surface", "mapping_report", "native_cfd_wall", "aligned_geometry"])
                           and not any(d.rglob("*nearwall*")) and not any(d.rglob("surface*.vtp")))
                native_expected = None
            else:
                true, pred = raw_true, raw_pred
                self.equal(label + "/truth_mother_pressure", true, np.concatenate([v["wall_pressure_rel_peak"], v["vol_pressure_rel_peak"]])[q])
                self.check(label + "/full_pressure_domain", np.array_equal(q, np.arange(nw + len(interior))))
                expected = {"pressure_cfd_pa": true, "pressure_pred_pa": pred,
                            "pressure_error_pred_minus_cfd_pa": pred - true, "pressure_abs_error_pa": abs(pred - true)}
                order = np.argsort(q[kind == 0])
                native_expected = {k: a[kind == 0][order] for k, a in expected.items()}
                self.close(label + "/pressure_reference", m["pressure_reference"]["p_ref_pa"], v["p_ref_pa"], 0)
                self.check(label + "/pressure_zero", m["pressure_reference"]["kind"] == str(v["p_ref_kind"])
                           and m["pressure_reference"]["posthoc_offset_correction"] is False)
            expected.update(query_index=q, point_kind=kind, dist_to_wall_mm=dist)
        self.equal(label + "/coordinates", point["xyz"], xyz)
        cfkey, prkey = ("speed_cfd", "speed_pred") if model == "VF6" else (prefix + "_cfd_pa", prefix + "_pred_pa")
        cmax, pmax = float(true.max()), float(pred.max())
        contract = m["display_normalization"]
        self.check(label + "/positive_finite_denominators", np.isfinite([cmax, pmax]).all() and cmax > 0 and pmax > 0)
        self.close(label + "/cfd_max_domain", contract["cfd_max"], cmax, 0)
        self.close(label + "/pred_max_domain", contract["pred_max"], pmax, 0)
        self.check(label + "/denominator_scope", contract["denominator_scope"] == SCOPE[model]
                   and contract["source_point_count"] == len(true) and contract["prefix"] == prefix
                   and contract["source_unit"] == m["unit"] and contract["unit"] == "dimensionless"
                   and contract["amplitude_difference_removed"] is True)
        expected.update(derived_selfmax(true, pred, prefix, cmax, pmax))
        self.compare_arrays(label + "/point_array", point["arrays"], expected)
        self.normalization(label + "/point_fielddata", point, contract, cmax, pmax)
        self.csv(label + "/point_csv", d / m["files"]["original_points_csv"], xyz, expected)
        if model == "PF6":
            for name, raw in [("cfd", true), ("pred", pred)]:
                normalized = point["arrays"][prefix + "_" + name + "_selfmax"]
                self.check(label + "/pressure_negative_preserved/" + name,
                           np.any(raw < 0) and np.array_equal(normalized < 0, raw < 0),
                           {"negative_count": int((raw < 0).sum()), "selfmax_min": float(normalized.min())})
        r2 = 1 - ((pred - true) ** 2).sum() / ((true - true.mean()) ** 2).sum()
        mae = abs(pred - true).mean()
        self.close(label + "/r2_official", r2, official["r2"], 1e-8)
        self.close(label + "/mae_official", mae, official["mae"], 1e-6)
        self.equal(label + "/metric_metadata_r2", m["visualization_metrics"]["r2"], official["r2"])
        self.equal(label + "/metric_metadata_n", m["visualization_metrics"]["n"], len(true))
        surface_result = None
        if native_expected is not None:
            nt, npred = native_expected[cfkey], native_expected[prkey]
            native_expected.update(derived_selfmax(nt, npred, prefix, cmax, pmax))
            native = read_vtp(d / m["files"]["native_cfd_wall"])
            surface = read_vtp(d / m["files"]["gaussian_surface"])
            aligned = read_vtp(d / m["files"]["aligned_geometry"])
            report = json.loads((d / m["files"]["mapping_report"]).read_text())
            self.equal(label + "/native_coordinates", native["xyz"], wall)
            self.compare_arrays(label + "/native_array", native["arrays"], native_expected)
            self.normalization(label + "/native_fielddata", native, contract, cmax, pmax)
            geometry = m["geometry"]
            with h5py.File(geometry["snapshot_source"]) as h:
                valid = h["wall_static/valid"][:].astype(bool)
                self.equal(label + "/native_node_ids", b["wall_node_id_cas"], h["wall_static/node_id_cas"][:][valid])
                remap = np.full(len(valid), -1, int)
                remap[valid] = np.arange(valid.sum())
                tri = remap[h["topology/wall_triangles"][:]]
                tri = tri[np.all(tri >= 0, axis=1)]
                self.equal(label + "/native_triangles", native["triangles"], tri)
                raw = h["wall_static/xyz_mm"][:][valid]
                reconstructed = (raw - b["transform_centroid"]) @ b["transform_rotation"].T
                self.check(label + "/atlas_registration", np.max(abs(reconstructed - wall)) < 1e-3)
            sx, sa, st = surface["xyz"], surface["arrays"], surface["triangles"]
            self.equal(label + "/surface_coordinates", sx, aligned["xyz"])
            self.equal(label + "/surface_topology", st, aligned["triangles"])
            self.check(label + "/real_surface_triangles", len(st) == len(native["triangles"]) > 0 and st.min() >= 0 and st.max() < len(sx))
            erkey = prefix + "_error_pred_minus_cfd_pa"
            self.equal(label + "/surface_signed_error", sa[erkey], sa[prkey] - sa[cfkey])
            self.equal(label + "/surface_absolute_error", sa[erkey.replace("error", "abs_error")], abs(sa[erkey]))
            self.check(label + "/coverage", np.all(sa["map_valid"] == 1) and report["coverage"]["valid_ratio"] == 1
                       and report["coverage"]["target_points"] == len(sx) and report["source_points"] == nw)
            for field, values in derived_selfmax(sa[cfkey], sa[prkey], prefix, cmax, pmax).items():
                self.equal(label + "/surface_selfmax/" + field, sa[field], values)
            self.normalization(label + "/surface_fielddata", surface, contract, cmax, pmax)
            self.check(label + "/mapping_normalization", report["display_normalization"] == contract)
            tree = cKDTree(wall)
            distances, _ = tree.query(sx, k=1)
            self.check(label + "/mapping_distances", np.max(abs(distances - sa["map_dist_mm"])) < 1e-10)
            self.close(label + "/mapping_max_distance", report["distance_mm"]["max"], distances.max(), 1e-10)
            self.check(label + "/gaussian_parameters", report["params"]["radius_mm"] == 3
                       and report["params"]["sharpness"] == 2 and report["params"]["max_dist_mm"] == 3)
            samples = np.unique(np.linspace(0, len(sx) - 1, 64, dtype=int))
            maxerror = 0.
            for index in samples:
                ids = tree.query_ball_point(sx[index], 3.)
                distance = np.linalg.norm(wall[ids] - sx[index], axis=1)
                weights = np.exp(-2 * (distance / 3) ** 2)
                weights /= weights.sum()
                for field, values in [(cfkey, nt), (prkey, npred)]:
                    maxerror = max(maxerror, abs(float(weights @ values[ids]) - float(sa[field][index])))
            self.check(label + "/independent_gaussian_64", maxerror < 1e-9,
                       {"samples": len(samples), "max_absolute_difference": maxerror})
            qc = report["cross_wall_checks"]
            disconnected, opposite = sa["disconnected_local_patch_weight_fraction"], sa["opposite_normal_weight_fraction"]
            dc, oc = int((disconnected > 1e-12).sum()), int((opposite > 1e-12).sum())
            self.check(label + "/cross_wall_qc", dc == qc["disconnected_stencil_count"] and oc == qc["opposite_normal_stencil_count"]
                       and abs(disconnected.max() - qc["max_disconnected_weight"]) < 1e-12
                       and abs(opposite.max() - qc["max_opposite_normal_weight"]) < 1e-12)
            self.check(label + "/geometry_source_sha", self.sha(geometry["stl_source"]) == geometry["stl_sha256"])
            self.csv(label + "/mapped_csv", d / "surface_gaussian/mapped_vertices.csv", sx, sa)
            source_arrays = {
                cfkey: nt, prkey: npred, erkey: npred - nt,
                erkey.replace("error", "abs_error"): abs(npred - nt),
                **derived_selfmax(nt, npred, prefix, cmax, pmax),
            }
            self.csv(label + "/gaussian_source_csv", d / "_export/gaussian_source.csv.gz", wall, source_arrays)
            surface_result = {"points": len(sx), "triangles": len(st), "coverage": 1.,
                              "nearest_max_mm": float(distances.max()), "independent_kernel_max_error": maxerror,
                              "disconnected_stencils": dc, "max_disconnected_weight": float(disconnected.max()),
                              "opposite_normal_stencils": oc, "max_opposite_normal_weight": float(opposite.max()),
                              "original_cfd_max": cmax, "original_pred_max": pmax,
                              "mapped_cfd_selfmax_max": float(sa[prefix + "_cfd_selfmax"].max()),
                              "mapped_pred_selfmax_max": float(sa[prefix + "_pred_selfmax"].max())}
        for path, sha in m["source_sha256"].items():
            self.check(label + "/source_sha/" + Path(path).name, self.sha(path) == sha)
        for path, sha in m["output_sha256"].items():
            self.check(label + "/output_sha/" + path, self.sha(d / path) == sha)
        self.cases.append({"model": model, "role": role, "case_id": uid, "point_count": len(xyz), "wall_count": nw,
                           "coordinate_max_difference_mm": float(np.max(abs(point["xyz"] - xyz))),
                           "s1234_r2": official["r2"], "three_seed_selection_r2": m["selection"]["r2_case_mean"],
                           "r2_cache_difference": float(r2 - official["r2"]), "mae_cache_difference": float(mae - official["mae"]),
                           "selfmax_domain": SCOPE[model], "cfd_max": cmax, "pred_max": pmax,
                           "surface": surface_result, "checks": len(self.checks) - start,
                           "passed": all(c["passed"] for c in self.checks[start:])})
        print(model, role, "passed", self.cases[-1]["passed"], "checks", self.cases[-1]["checks"], flush=True)

    def write(self):
        errors = [c for c in self.checks if not c["passed"]]
        result = {"status": "passed" if not errors else "failed", "schema_version": 3,
                  "verified_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
                  "scope": "Independent nine-case raw fields and selfmax audit from existing caches; no inference",
                  "verifier": Path(__file__).name, "verifier_sha256": self.sha(__file__),
                  "helper_sha256": self.sha(ROOT / "tools/cfdpost_cloud_export/display_fields.py"),
                  "case_count": len(self.cases), "check_count": len(self.checks), "failed_check_count": len(errors),
                  "helper_branch_checks": self.helper_checks, "unique_sha_files_checked": len(self.hashes),
                  "tolerances": {"r2_absolute": 1e-8, "mae_absolute": 1e-6,
                                 "gaussian_sample_absolute": 1e-9, "csv_rtol": 1e-10, "csv_atol": 1e-12},
                  "cases": self.cases, "errors": errors, "checks": self.checks,
                  "limitations": [
                      "Gaussian uses Euclidean radius 3 mm. Diagnostic weights do not establish absence of cross-wall smoothing; compare native CFD surfaces.",
                      "Coverage is not an area-mapping gate.",
                      "VF6 is an interior point cloud without continuous volume-cell topology.",
                      "Selfmax removes amplitude differences, and signed relative pressure may be negative or below -1. Formal metrics remain physical same-point metrics.",
                  ]}
        (OUT / "verification.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
        print(json.dumps({k: result[k] for k in ["status", "case_count", "check_count", "failed_check_count"]}), flush=True)
        return not errors


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--helper-only", action="store_true", help="Exercise helper branches without writing the package report")
    args = parser.parse_args()
    audit = Audit()
    audit.helper_branches()
    if args.helper_only:
        print(json.dumps(audit.checks, ensure_ascii=False, indent=2))
        return 0 if all(c["passed"] for c in audit.checks) else 1
    summary = json.loads((OUT.parent / "启发式v2_R2分布_20260914/summary.json").read_text())
    summary = {item["key"]: item for item in summary["distributions"]}
    try:
        for model in KEYS:
            for role in ROLES:
                audit.case(model, role, summary)
    except Exception as error:
        audit.check("unhandled_verification_exception", False, repr(error))
        audit.write()
        raise
    return 0 if audit.write() else 1


if __name__ == "__main__":
    raise SystemExit(main())

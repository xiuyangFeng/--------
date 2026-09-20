"""Whole-cohort automatic screen of the V6 geometry candidate (read-only, diagnostic).

Three layers, all independent of the candidate generator code path:
  L1  invariants recomputed from the stored polygons with a separate implementation
      (area / perimeter / centroid / roundness / eccentricity / planarity / containment),
      wall-map interpolation consistency, opening areas vs Fluent interface areas.
  L2  physical oracles that never touch the contour code: cross-section area from the
      Fluent volume mesh (dV/ds with the V5 atlas s-mapping), inscribed radius bound
      A >= pi*Rmis^2, V5-vs-V6 wall projection agreement, adjacent-section jumps.
  L3  cohort robust z-scores of per-case scalars (CIA length / tortuosity / angles /
      valid fractions / opening areas / bifurcation locator s) to rank outliers.

Outputs a per-case CSV, a per-section flag CSV, a ranked markdown list and two contact sheets.
"""
from __future__ import annotations

import json
import sys
from multiprocessing import Pool
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
from matplotlib.path import Path as MplPath
from scipy.spatial import cKDTree

ROOT = Path("/public/newhome/cy/Digital_twin/GNN")
CAND = ROOT / "outputs/wss_v6_geometry_candidate_20260909/cases"
MANIFEST = ROOT / "docs/03-汇报材料/WSS_V6_新几何验收_20260909/render_manifest.json"
BIF_REVIEW = ROOT / "docs/03-汇报材料/WSS_V6_新几何验收_20260909/bifurcation_review.json"
_BIF = {}
for _r in json.load(open(BIF_REVIEW))["rows"]:
    _BIF.setdefault(_r["canonical_id"], {})[_r["parent_segment"]] = _r
OUT = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(__file__).resolve().parent / "screen_out"
OUT.mkdir(parents=True, exist_ok=True)

BIN_HALF_MM = 1.0          # dV/ds bin half width along s (2 mm total = section spacing)
VOL_REL_FLAG = 0.20        # |A_section/A_volume - 1| beyond this -> section flag
JUMP_FLAG = np.log(1.5)    # |log A[i+1]/A[i]| between adjacent valid sections (2 mm apart)
RMIS_LOW = 0.85            # A/(pi Rmis^2) below this is physically impossible if both are right
RMIS_HIGH = 3.0


def frame(normal):
    t = normal / max(np.linalg.norm(normal), 1e-12)
    # deliberately a different construction from section_features.frame
    a = np.array([0.0, 0.0, 1.0]) if abs(t[2]) < 0.9 else np.array([1.0, 0.0, 0.0])
    e1 = np.cross(t, a); e1 /= np.linalg.norm(e1)
    e2 = np.cross(t, e1)
    return e1, e2


def polygon_recompute(poly, center, normal):
    """Independent shoelace on the section plane. Returns dict or None if degenerate."""
    if len(poly) > 1 and np.linalg.norm(poly[0] - poly[-1]) < 1e-7:
        poly = poly[:-1]
    if len(poly) < 3:
        return None
    e1, e2 = frame(normal)
    rel = poly - center
    x, y = rel @ e1, rel @ e2
    z = rel @ (normal / np.linalg.norm(normal))
    x2, y2 = np.roll(x, -1), np.roll(y, -1)
    cr = x * y2 - x2 * y
    sa = 0.5 * cr.sum()
    if abs(sa) < 1e-9:
        return None
    cx = ((x + x2) * cr).sum() / (6 * sa)
    cy = ((y + y2) * cr).sum() / (6 * sa)
    per = np.hypot(x2 - x, y2 - y).sum()
    area = abs(sa)
    req = np.sqrt(area / np.pi)
    # self-intersection proxy: signed area of |cr| vs |sum cr|
    return {
        "area": area, "perimeter": per, "roundness": 4 * np.pi * area / per ** 2,
        "eccentricity": np.hypot(cx, cy) / req,
        "centroid": center + cx * e1 + cy * e2,
        "planarity_mm": float(np.abs(z).max()),
        "contains": bool(MplPath(np.column_stack([x, y])).contains_point((0.0, 0.0))),
        "sign_mix": float(min(np.abs(cr[cr > 0]).sum(), np.abs(cr[cr < 0]).sum()) / np.abs(cr).sum()),
        "self_intersect": self_intersects(x, y),
    }


def self_intersects(x, y):
    """True if any two non-adjacent edges of the closed polygon cross (vectorised)."""
    n = len(x)
    if n < 4:
        return False
    P = np.column_stack([x, y]); Q = np.roll(P, -1, axis=0)
    i, j = np.triu_indices(n, k=2)
    keep = ~((i == 0) & (j == n - 1))
    i, j = i[keep], j[keep]
    p1, p2, p3, p4 = P[i], Q[i], P[j], Q[j]
    d1 = p2 - p1; d2 = p4 - p3
    den = d1[:, 0] * d2[:, 1] - d1[:, 1] * d2[:, 0]
    r = p3 - p1
    t = (r[:, 0] * d2[:, 1] - r[:, 1] * d2[:, 0]) / np.where(np.abs(den) < 1e-14, np.nan, den)
    u = (r[:, 0] * d1[:, 1] - r[:, 1] * d1[:, 0]) / np.where(np.abs(den) < 1e-14, np.nan, den)
    return bool(np.any((t > 1e-9) & (t < 1 - 1e-9) & (u > 1e-9) & (u < 1 - 1e-9)))


def robust_relerr(a, b):
    m = np.isfinite(a) & np.isfinite(b) & (b != 0)
    return np.where(m, a / np.where(m, b, 1) - 1, np.nan)


def screen_case(entry):
    cid, key = entry["canonical_id"], entry["key"]
    cdir = CAND / key
    z = np.load(cdir / "geometry.npz")
    rep = json.load(open(cdir / "report.json"))
    h5p = rep["source"]["case_h5"]
    row = {"canonical_id": cid, "key": key, "cohort": cid.split("/")[0]}
    sec_flags = []
    n = int(z["section_area_mm2"].shape[0])
    row["n_sections"] = n

    # ---------------- L1: recompute polygon metrics (reference + pointcloud) ----------------
    for src, pk, ok in (("ref", "section_polygon", "section"), ("pc", "pc_section_polygon", "pc_section")):
        off = z[f"{pk}_offsets"]; P = z[f"{pk}_xyz_mm"]
        valid = z[f"{ok}_valid"]
        err_a, err_p, err_c, err_e, plan, cont, mix, selfx = [], [], [], [], [], [], [], []
        for i in range(n):
            if not valid[i]:
                continue
            poly = P[off[i]:off[i + 1]]
            r = polygon_recompute(poly, z["section_center_mm"][i].astype(float), z["section_tangent"][i].astype(float))
            if r is None:
                err_a.append(np.inf); continue
            err_a.append(abs(r["area"] / z[f"{ok}_area_mm2"][i] - 1))
            err_p.append(abs(r["perimeter"] / z[f"{ok}_perimeter_mm"][i] - 1))
            err_c.append(np.linalg.norm(r["centroid"] - z[f"{ok}_centroid_mm"][i]))
            err_e.append(abs(r["eccentricity"] - z[f"{ok}_eccentricity"][i]))
            plan.append(r["planarity_mm"]); cont.append(r["contains"]); mix.append(r["sign_mix"]); selfx.append(r["self_intersect"])
            if not r["contains"] or abs(r["area"] / z[f"{ok}_area_mm2"][i] - 1) > 1e-3:
                sec_flags.append({"canonical_id": cid, "section": i, "segment": int(z["section_segment_id"][i]),
                                  "s_mm": float(z["section_s_local_mm"][i]), "layer": "L1", "source": src,
                                  "check": "polygon_recompute", "value": float(abs(r["area"] / z[f"{ok}_area_mm2"][i] - 1)),
                                  "note": f"contains={r['contains']}"})
            if r["self_intersect"]:
                sec_flags.append({"canonical_id": cid, "section": i, "segment": int(z["section_segment_id"][i]),
                                  "s_mm": float(z["section_s_local_mm"][i]), "layer": "L1", "source": src,
                                  "check": "self_intersecting_contour", "value": 1.0, "note": f"sign_mix={r['sign_mix']:.2e}"})
            elif r["sign_mix"] > 1e-3:
                sec_flags.append({"canonical_id": cid, "section": i, "segment": int(z["section_segment_id"][i]),
                                  "s_mm": float(z["section_s_local_mm"][i]), "layer": "L1", "source": src,
                                  "check": "nonstar_from_centerline", "value": float(r["sign_mix"]), "note": "reversed-edge area fraction"})
        row[f"L1_{src}_area_relerr_max"] = float(np.max(err_a)) if err_a else np.nan
        row[f"L1_{src}_perim_relerr_max"] = float(np.max(err_p)) if err_p else np.nan
        row[f"L1_{src}_centroid_mm_max"] = float(np.max(err_c)) if err_c else np.nan
        row[f"L1_{src}_ecc_abs_max"] = float(np.max(err_e)) if err_e else np.nan
        row[f"L1_{src}_planarity_mm_max"] = float(np.max(plan)) if plan else np.nan
        row[f"L1_{src}_not_containing_center"] = int(np.sum(~np.array(cont, bool))) if cont else 0
        row[f"L1_{src}_self_intersecting"] = int(np.sum(selfx)) if selfx else 0
        row[f"L1_{src}_nonstar_from_center"] = int(np.sum(np.array(mix) > 1e-3)) if mix else 0
        # derived-field identities
        v = valid
        row[f"L1_{src}_req_identity_max"] = float(np.nanmax(np.abs(z[f"{ok}_req_mm"][v] - np.sqrt(z[f"{ok}_area_mm2"][v] / np.pi)))) if v.any() else np.nan
        row[f"L1_{src}_roundness_identity_max"] = float(np.nanmax(np.abs(z[f"{ok}_roundness"][v] - 4 * np.pi * z[f"{ok}_area_mm2"][v] / z[f"{ok}_perimeter_mm"][v] ** 2))) if v.any() else np.nan
        row[f"L1_{src}_valid_reason_mismatch"] = int(np.sum(valid != (z[f"{ok}_reason"] == "ok")))

    # wall map vs section interpolation (reference), plus V5-vs-V6 projection agreement
    wseg = z["wall_map_segment_id"]; ws = z["wall_map_s_local_mm"]; wA = z["wall_map_area_mm2"]; wv = z["wall_map_valid"]
    sseg = z["section_segment_id"]; ss = z["section_s_local_mm"]; sA = z["section_area_mm2"]; sv = z["section_valid"]
    interp = np.full(len(wA), np.nan)
    for k in np.unique(sseg):
        m = sseg == k; idx = np.flatnonzero(m); o = np.argsort(ss[idx]); idx = idx[o]
        wm = np.flatnonzero((wseg == k) & wv)
        if len(idx) < 2 or len(wm) == 0:
            continue
        s_k, A_k, v_k = ss[idx], sA[idx], sv[idx]
        hi = np.clip(np.searchsorted(s_k, ws[wm], side="right"), 1, len(s_k) - 1); lo = hi - 1
        good = v_k[lo] & v_k[hi] & (ws[wm] >= s_k[0]) & (ws[wm] <= s_k[-1])
        al = (ws[wm] - s_k[lo]) / np.maximum(s_k[hi] - s_k[lo], 1e-12)
        val = (1 - al) * A_k[lo] + al * A_k[hi]
        interp[wm[good]] = val[good]
    m = wv & np.isfinite(interp) & np.isfinite(wA)
    re = np.abs(wA[m] / interp[m] - 1)
    row["L1_wallmap_area_relerr_p99"] = float(np.quantile(re, 0.99)) if m.any() else np.nan
    row["L1_wallmap_valid_but_uninterpolable"] = int(np.sum(wv & ~np.isfinite(interp)))

    with h5py.File(h5p, "r") as f:
        v5seg = f["wall_static/segment_id"][:]; v5s = f["wall_static/s_local_mm"][:]
        v5junc = f["wall_static/junction_mask"][:]
        vseg = f["volume_static/segment_id"][:]; vs = f["volume_static/s_local_mm"][:]; vV = f["volume_static/volume_m3"][:] * 1e9
        iface = {k: (float(f["interfaces"][k].attrs["area_m2"]) * 1e6,
                     np.array(json.loads(f["interfaces"][k].attrs["center_mm"]))) for k in f["interfaces"]}
        cap_labels = list(json.loads(f["geometry"].attrs["cap_labels"]))
    amb = z["wall_map_ambiguous"]
    cmp = ~amb & ~v5junc & (wseg >= 0)
    row["L2_wall_segment_agree_v5"] = float(np.mean(wseg[cmp] == v5seg[cmp])) if cmp.any() else np.nan
    same = cmp & (wseg == v5seg)
    row["L2_wall_s_absdiff_p95_mm"] = float(np.quantile(np.abs(ws[same] - v5s[same]), 0.95)) if same.any() else np.nan
    row["L2_wall_segment_disagree_count"] = int(np.sum(cmp & (wseg != v5seg)))

    # ---------------- L2: dV/ds oracle from the Fluent volume mesh ----------------
    # Volume cells carry the discrete s of their nearest atlas point (0.5 mm grid), so first
    # aggregate V per atlas s-value, give each value its own slab width (half-gaps to the
    # neighbours), then average dV/ds over the +-BIN_HALF_MM window weighted by slab width.
    A_vol = np.full(n, np.nan)
    for k in np.unique(sseg):
        km = vseg == k
        if not km.any():
            continue
        su, inv = np.unique(vs[km], return_inverse=True)
        Vu = np.bincount(inv, weights=vV[km], minlength=len(su))
        if len(su) < 3:
            continue
        edges = np.concatenate([[su[0] - (su[1] - su[0]) / 2], (su[:-1] + su[1:]) / 2, [su[-1] + (su[-1] - su[-2]) / 2]])
        width = np.diff(edges)
        idx = np.flatnonzero(sseg == k)
        for i in idx:
            w = (su >= ss[i] - BIN_HALF_MM) & (su <= ss[i] + BIN_HALF_MM)
            if w.sum() >= 2:
                A_vol[i] = Vu[w].sum() / width[w].sum()
    for src, ok in (("ref", "section"), ("pc", "pc_section")):
        valid = z[f"{ok}_valid"]; A = z[f"{ok}_area_mm2"]
        r = robust_relerr(A, A_vol); r[~valid] = np.nan
        good = np.isfinite(r)
        row[f"L2_{src}_vs_volume_median_relerr"] = float(np.nanmedian(r)) if good.any() else np.nan
        row[f"L2_{src}_vs_volume_p90_absrelerr"] = float(np.nanquantile(np.abs(r), 0.9)) if good.any() else np.nan
        row[f"L2_{src}_vs_volume_n_over20pct"] = int(np.sum(np.abs(r[good]) > VOL_REL_FLAG))
        row[f"L2_{src}_vs_volume_n_compared"] = int(good.sum())
        for i in np.flatnonzero(good & (np.abs(r) > VOL_REL_FLAG)):
            sec_flags.append({"canonical_id": cid, "section": int(i), "segment": int(sseg[i]), "s_mm": float(ss[i]),
                              "layer": "L2", "source": src, "check": "area_vs_volume_dVds", "value": float(r[i]),
                              "note": f"A_{src}={A[i]:.1f} A_vol={A_vol[i]:.1f} mm2"})
    # inscribed-radius lower bound
    rat = sA / (np.pi * z["section_radius_mis_mm"] ** 2); rat[~sv] = np.nan
    row["L2_ref_A_over_piRmis2_min"] = float(np.nanmin(rat)) if sv.any() else np.nan
    row["L2_ref_A_over_piRmis2_max"] = float(np.nanmax(rat)) if sv.any() else np.nan
    row["L2_ref_A_over_piRmis2_median"] = float(np.nanmedian(rat)) if sv.any() else np.nan
    for i in np.flatnonzero(sv & ((rat < RMIS_LOW) | (rat > RMIS_HIGH))):
        sec_flags.append({"canonical_id": cid, "section": int(i), "segment": int(sseg[i]), "s_mm": float(ss[i]),
                          "layer": "L2", "source": "ref", "check": "A_over_piRmis2", "value": float(rat[i]),
                          "note": f"A={sA[i]:.1f} Rmis={z['section_radius_mis_mm'][i]:.2f}"})
    # adjacent-section jumps within a segment (both valid, 2 mm apart)
    jumps = 0
    for k in np.unique(sseg):
        idx = np.flatnonzero(sseg == k); idx = idx[np.argsort(ss[idx])]
        for a, b in zip(idx[:-1], idx[1:]):
            if sv[a] and sv[b] and abs(ss[b] - ss[a]) < 3.0:
                d = abs(np.log(sA[b] / sA[a]))
                if d > JUMP_FLAG:
                    jumps += 1
                    sec_flags.append({"canonical_id": cid, "section": int(b), "segment": int(k), "s_mm": float(ss[b]),
                                      "layer": "L2", "source": "ref", "check": "adjacent_jump", "value": float(np.exp(d)),
                                      "note": f"A[{a}]={sA[a]:.1f} -> A[{b}]={sA[b]:.1f}"})
    row["L2_ref_adjacent_jump_count"] = jumps
    # ref vs pc disagreement (both valid)
    both = sv & z["pc_section_valid"]
    d = np.abs(z["pc_section_area_mm2"][both] / sA[both] - 1)
    row["L2_pc_vs_ref_p90_absrelerr"] = float(np.quantile(d, 0.9)) if both.any() else np.nan
    row["L2_pc_vs_ref_n_over20pct"] = int(np.sum(d > 0.2))

    # ---------------- L1/L2: openings vs Fluent interface areas ----------------
    ops = rep["openings"]
    row["openings_count"] = len(ops)
    row["openings_valid"] = int(sum(1 for o in ops if o["valid"]))
    max_open_err = 0.0; centre_gap = 0.0
    for o in ops:
        if not o["valid"]:
            continue
        c = np.array(o["centroid_mm"])
        name, (ia, ic) = min(iface.items(), key=lambda kv: np.linalg.norm(kv[1][1] - c))
        max_open_err = max(max_open_err, abs(o["area_mm2"] / ia - 1)); centre_gap = max(centre_gap, np.linalg.norm(ic - c))
    row["L1_opening_vs_fluent_area_relerr_max"] = max_open_err
    row["L1_opening_vs_fluent_centre_gap_mm_max"] = centre_gap
    row["L1_inlet_ref_vs_fluent_relerr"] = abs(rep["inlet_reference_area_mm2"] / iface["inlet"][0] - 1) if "inlet" in iface else np.nan
    row["L1_inlet_pc_vs_fluent_relerr"] = abs(rep["inlet_pointcloud_area_mm2"] / iface["inlet"][0] - 1) if (rep.get("inlet_pointcloud_valid") and "inlet" in iface) else np.nan

    # ---------------- L3: per-case scalars ----------------
    row["ref_valid_frac"] = float(sv.mean()); row["pc_valid_frac"] = float(z["pc_section_valid"].mean())
    row["wall_valid_frac"] = float(wv.mean()); row["wall_ambiguous_frac"] = float(amb.mean())
    row["topology_valid"] = bool(rep["P2_topology"]["valid"]); row["aortoiliac_pattern"] = bool(rep["P2_topology"]["expected_aortoiliac_pattern_valid"])
    row["n_segments"] = len(rep["segments"]); row["n_warnings"] = len(rep["warnings"])
    for seg in rep["segments"]:
        if seg["role"] == "trunk":
            row["trunk_length_mm"] = seg["length_mm"]; row["trunk_tortuosity"] = seg["tortuosity"]
            row["trunk_daughter_angle_deg"] = seg["daughter_angle_deg"]
    for j, c in enumerate(sorted(rep["P3_CIA"], key=lambda c: c["anatomy_label"])):
        tag = c["anatomy_label"].replace("_cia", "")
        row[f"cia_{tag}_length_mm"] = c["length_mm"]; row[f"cia_{tag}_tortuosity"] = c["tortuosity"]
        row[f"cia_{tag}_parent_angle_deg"] = c["parent_branch_angle_deg"]; row[f"cia_{tag}_daughter_angle_deg"] = c["daughter_angle_deg"]
    row["inlet_area_mm2"] = rep["inlet_reference_area_mm2"]
    for name, (ia, _) in iface.items():
        row[f"fluent_{name}_area_mm2"] = ia
    # Murray-type check at the aortic bifurcation: inlet^1.5 vs sum of outlet areas^1.5 is anatomy, keep as descriptor
    outs = [ia for nm, (ia, _) in iface.items() if nm != "inlet"]
    row["outlet_area_sum_over_inlet"] = float(sum(outs) / iface["inlet"][0]) if "inlet" in iface and outs else np.nan
    # bifurcation locator
    for pseg, r in _BIF.get(cid, {}).items():
        tag = f"J_seg{pseg}"
        row[f"{tag}_status"] = r["status"]
        row[f"{tag}_s_mm"] = r["s_mm"] if r["s_mm"] is not None else np.nan
    row["bif_undetermined"] = int(sum(1 for r in _BIF.get(cid, {}).values() if r["status"] != "candidate"))
    # stability (as already summarised in report)
    n_sens = 0; n_cmp = 0
    for st in rep["stability"]:
        for v in st["variants"]:
            if v.get("comparison_valid"):
                n_cmp += 1
                if abs(v.get("relative_to_same_source_baseline") or v.get("relative_area_change") or 0) > 0.2:
                    n_sens += 1
    row["stability_variants_compared"] = n_cmp; row["stability_variants_over20pct"] = n_sens

    # thumbnails data (compact)
    thumb = {
        "wall": z["wall_xyz_mm"][::max(1, len(z["wall_xyz_mm"]) // 3000)].astype(np.float32),
        "cl": z["centerline_xyz_mm"].astype(np.float32), "clseg": z["centerline_segment_id"],
        "sc": z["section_centroid_mm"].astype(np.float32), "sv": sv, "sseg": sseg, "ss": ss.astype(np.float32),
        "sA": sA.astype(np.float32), "pcA": z["pc_section_area_mm2"].astype(np.float32), "pcv": z["pc_section_valid"],
        "Avol": A_vol.astype(np.float32),
        "open": np.array([o["centroid_mm"] for o in ops if o["valid"]], np.float32).reshape(-1, 3),
    }
    np.savez_compressed(OUT / "thumbs" / f"{key}.npz", **thumb)
    return row, sec_flags


def main():
    (OUT / "thumbs").mkdir(exist_ok=True)
    cases = json.load(open(MANIFEST))["cases"]
    with Pool(32) as pool:
        res = pool.map(screen_case, cases, chunksize=1)
    rows = [r for r, _ in res]; flags = [f for _, fl in res for f in fl]
    df = pd.DataFrame(rows).set_index("canonical_id")
    df.to_csv(OUT / "screen_per_case.csv")
    fl = pd.DataFrame(flags)
    fl.to_csv(OUT / "screen_section_flags.csv", index=False)
    print("cases", len(df), "section flags", len(fl))
    print(fl.groupby(["layer", "check"]).size() if len(fl) else "no section flags")


if __name__ == "__main__":
    main()

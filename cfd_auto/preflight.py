"""Set-up-stage checks: is the boundary-condition protocol applicable to this geometry, and how risky is the run?
Written to ``preflight.json`` before any CFD; a ``fail`` blocks the unit until a person decides.

  rcr_R2_positive        protocol R2 = Rt - R1 > 0 for every outlet (fail) — YANG_QING_REN-1/after (7.3 / 5.1 mm² outlets)
  r1_fraction            R1 / Rt per outlet (warn above protocol risk.r1_fraction_warn)
  small_outlets          outlets below risk.small_outlet_mm2 (warn) — WANG_CAI-0/before's 5.9 mm² outlet diverged at the
                         library time step (the run ladder then takes over)
  oblique_openings       cap normal vs vessel axis above extension.oblique_deg (warn; how the extension handled it)
  anatomy_split          steady Poiseuille network on the centreline tree (segment resistance 8 mu ds / (pi r^4), plus
                         each outlet's straight extension) closed by each outlet's RCR total: predicted left share and
                         within-side shares vs the protocol's 50/50 and Murray design (warn when the anatomy takes more
                         than risk.path_resistance_ratio_warn of an outlet's RCR total) — WANG_CAI-0/before: left 0.451
  naming                 automatic opening naming confidence vs openings.naming_min_confidence (fail below, unless a
                         person confirmed the naming)
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

MU_PA_S = 0.0035
RHO = 1060.0


def _item(name: str, level: str, value=None, note: str = "") -> dict:
    return {"check": name, "level": level, "value": value, "note": note}


def tree_from_atlas(atlas_npz: Path) -> dict:
    """Centreline tree of a vessel_geom atlas: per segment parent, outlet name (leaves) and Poiseuille resistance (Pa s / m^3)."""
    d = np.load(atlas_npz, allow_pickle=True)
    cols = [str(c) for c in d["columns"]]
    t = d["table"]
    ci = {c: i for i, c in enumerate(cols)}
    segs = d["segments"]
    segs = segs.item() if segs.ndim == 0 else list(segs)
    segs = json.loads(segs) if isinstance(segs, (str, bytes)) else segs
    out = {}
    for s in segs:
        rows = t[t[:, ci["segment_id"]] == s["segment_id"]]
        rows = rows[np.argsort(rows[:, ci["s_local_mm"]])]
        ds = np.diff(rows[:, ci["s_local_mm"]]) * 1e-3
        r = 0.5 * (rows[1:, ci["radius_mm"]] + rows[:-1, ci["radius_mm"]]) * 1e-3
        out[int(s["segment_id"])] = {"parent": int(s["parent_id"]), "outlet": s.get("outlet_name") or None, "length_mm": float(s["length_mm"]),
                                     "R": float(np.sum(8 * MU_PA_S * ds / (np.pi * r ** 4))), "r_min_mm": float(rows[:, ci["radius_mm"]].min())}
    return out


def network_split(tree: dict, terminal_R: dict[str, float]) -> dict:
    """Steady flow split of a tree whose leaves end in ``terminal_R`` (outlet name -> Pa s / m^3): flows divide at every
    junction in inverse proportion to the children's equivalent resistances."""
    kids: dict[int, list[int]] = {}
    for sid, s in tree.items():
        kids.setdefault(s["parent"], []).append(sid)
    req: dict[int, float] = {}

    def eq(sid: int) -> float:
        s = tree[sid]
        if s["outlet"]:
            req[sid] = s["R"] + terminal_R[s["outlet"]]
        else:
            req[sid] = s["R"] + 1.0 / sum(1.0 / eq(c) for c in kids[sid])
        return req[sid]

    roots = kids[-1]
    if len(roots) != 1:
        raise ValueError(f"tree has {len(roots)} roots")
    eq(roots[0])
    share: dict[str, float] = {}

    def down(sid: int, q: float) -> None:
        if tree[sid]["outlet"]:
            share[tree[sid]["outlet"]] = q; return
        g = {c: 1.0 / req[c] for c in kids[sid]}
        tot = sum(g.values())
        for c, gc in g.items():
            down(c, q * gc / tot)

    down(roots[0], 1.0)
    return share


def anatomy_split(atlas_npz: Path, outlet_key: dict[str, str], rcr: dict[str, dict], cut_area_mm2: dict[str, float], ext_len_mm: dict[str, float],
                  side_pairs: list[list[str]], ratio_warn: float, dev_slope: float = 9.415, dev_warn: float = 0.015) -> dict:
    """``outlet_key``: atlas outlet name -> UDF key. RCR totals are per mass flow (Pa s / kg) -> x rho for volume flow."""
    tree = tree_from_atlas(atlas_npz)
    term, ratio, path = {}, {}, {}
    parent = {sid: s["parent"] for sid, s in tree.items()}
    for sid, s in tree.items():
        if not s["outlet"]:
            continue
        k = outlet_key[s["outlet"]]
        r_ext = np.sqrt(cut_area_mm2[k] / np.pi) * 1e-3
        R_ext = 8 * MU_PA_S * ext_len_mm[k] * 1e-3 / (np.pi * r_ext ** 4)
        R_rcr = (rcr[k]["R1"] + rcr[k]["R2"]) * RHO
        term[s["outlet"]] = R_ext + R_rcr
        R_path, p = 0.0, sid
        while p != -1 and parent[p] != -1:           # the trunk (root segment) is shared by every outlet
            R_path += tree[p]["R"]; p = parent[p]
        path[k] = R_path + R_ext
        ratio[k] = path[k] / R_rcr
    sh = network_split(tree, term)
    by_key = {outlet_key[o]: v for o, v in sh.items()}
    (le, li), (re_, ri) = side_pairs
    left = by_key[le] + by_key[li]
    d3 = {k: cut_area_mm2[k] ** 1.5 for k in by_key}
    within = {k: by_key[k] / (by_key[a] + by_key[b]) - d3[k] / (d3[a] + d3[b]) for a, b in side_pairs for k in (a, b)}
    worst = max(ratio.values())
    dev = dev_slope * (left - 0.5)
    return {"left_share_pred": left, "left_dev_calibrated": dev, "share_pred": by_key, "within_side_dev_vs_murray_pred": within, "path_over_rcr": ratio,
            "path_over_rcr_max": worst, "flag": abs(dev) > dev_warn, "path_ratio_flag": worst > ratio_warn,
            "note": "steady Poiseuille estimate (mu 3.5 mPa s); the library (249 units) shows the CFD left-share deviation is ~9.4x "
                    "the raw estimate (Pearson 0.72), hence left_dev_calibrated = 9.415 (pred - 0.5)"}


def naming_item(proto: dict, naming: dict, confirmed: bool = False) -> dict:
    """Fail when the only naming method is below the protocol confidence (overall or per side) or methods disagree."""
    dep = (naming.get("methods") or {}).get("deploy") or {}
    conf = dep.get("confidence")
    side = dep.get("side_confidence") or {}
    low = [v for v in [conf, *side.values()] if v is not None and v < proto["openings"]["naming_min_confidence"]]
    agree = naming.get("all_agree", True)
    if confirmed:
        return _item("naming", "ok", {"confidence": conf, "confirmed": True}, "confirmed by a person")
    if (low and len(naming.get("methods", {})) == 1) or not agree:
        return _item("naming", "fail", {"confidence": conf, "side_confidence": side, "all_agree": agree},
                     "automatic naming below the confidence threshold or methods disagree: confirm with `orchestrate confirm`")
    return _item("naming", "ok", {"confidence": conf, "all_agree": agree})


def check(proto: dict, protocol_bc: dict, naming: dict | None = None, surface_report: dict | None = None, split: dict | None = None,
          naming_confirmed: bool = False) -> dict:
    risk, keys = proto["risk"], proto["openings"]["outlet_keys"]
    items = []
    rcr = protocol_bc["rcr"]
    bad = {k: rcr[k]["R2"] for k in keys if rcr[k]["R2"] <= 0}
    items.append(_item("rcr_R2_positive", "fail" if bad else "ok", bad or None,
                       "protocol R2 <= 0: the protocol is not applicable to these outlet areas (needs a rule for small outlets)" if bad else ""))
    frac = {k: rcr[k]["R1"] / (rcr[k]["R1"] + rcr[k]["R2"]) for k in keys}
    hi = {k: round(v, 3) for k, v in frac.items() if v > risk["r1_fraction_warn"]}
    items.append(_item("r1_fraction", "warn" if hi else "ok", {k: round(v, 3) for k, v in frac.items()}, f"R1/Rt above {risk['r1_fraction_warn']}: {hi}" if hi else ""))
    small = {k: a for k, a in protocol_bc["cut_area_mm2"].items() if a < risk["small_outlet_mm2"]}
    items.append(_item("small_outlets", "warn" if small else "ok", small or None,
                       f"outlet < {risk['small_outlet_mm2']} mm²: divergence risk at the library time step (run ladder: gentle -> dt2)" if small else ""))
    if surface_report and surface_report.get("extension_directions"):
        ob = {k: v.get("angle_deg") for k, v in surface_report["extension_directions"].items() if v.get("angle_deg") is not None and v["angle_deg"] > proto["extension"]["oblique_deg"]}
        used = {k: v.get("direction") is not None for k, v in surface_report["extension_directions"].items()}
        items.append(_item("oblique_openings", "warn" if ob else "ok", {"angle_deg": ob, "swept_along_axis": used} if ob else None,
                           "cut more than extension.oblique_deg off the vessel axis" if ob else ""))
    else:
        items.append(_item("oblique_openings", "not_evaluated", None, "extension direction = cap normal (library mode)"))
    if split is not None:
        items.append(_item("anatomy_split", "warn" if split["flag"] else "ok",
                           {"left_dev_calibrated": round(split["left_dev_calibrated"], 4), "left_share_pred": round(split["left_share_pred"], 4),
                            "path_over_rcr_max": round(split["path_over_rcr_max"], 4)},
                           "anatomy resistance is a sizeable part of an outlet's RCR total: the split will drift from the 50/50 / Murray design" if split["flag"] else ""))
    else:
        items.append(_item("anatomy_split", "not_evaluated", None, "no centreline atlas"))
    if naming is not None:
        items.append(naming_item(proto, naming, naming_confirmed))
    levels = [i["level"] for i in items]
    return {"items": items, "fail": [i["check"] for i in items if i["level"] == "fail"], "warn": [i["check"] for i in items if i["level"] == "warn"],
            "ok": "fail" not in levels}


def dump(path: Path, rep: dict) -> None:
    path.write_text(json.dumps(rep, indent=1, default=float, ensure_ascii=False))

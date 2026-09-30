"""[2026-09-30 later: v5.2p5 was merged into the single unified root v5.2c (consolidate_v52c.py); NEW265 below is historical.]
Corrected data versions after the 2026-09-30 library label fix (9 units re-simulated with protocol-correct BCs and
swapped into data_new; docs/02-推进与变更/04-数据处理与CFD/库内标签问题核查_RCR挂错与入口除数_2026-09-30.md).

  v5.2c   data_wss_v5/views_v5_2c_20260930           = v5.2 (views_v5_2_full_20260923) with the 9 units re-pointed
  v5.2p5  data_wss_v5/views_v5_2p5_full265_20260930  = v5.2p4 full265 (views_v5_2p4_full265_20260930) with the 9 re-pointed

Both roots mirror their predecessor's layout and FILE NAMES exactly (packs, split files, statistics files), so any
config moves to the corrected data by replacing the root prefix (`training_wss_min/tools/repoint_data_root.py`). Case
directories are symlinks: the 9 units -> data_wss_v5/views_v5_2c_labelfix9_20260930 (rebuilt from the corrected library),
every other case -> the same real directory its predecessor pointed at. Label-derived statistics (WSS log_z, cycle
TAWSS / OSI) are recomputed on the new roots with the same code path that wrote the originals (verified first by
re-deriving the original files bit-exactly); input-feature z-scores are geometry-only and are checked to be unchanged.

Density sidecars (wss_min_density_v1) are geometry-only (subsample rows + surface features, checked against the bundle's
wall_node_id_cas at load) and the 9 units' geometry is unchanged, so every case keeps its predecessor sidecar: a rebuild is
not reproducible (build_density_sidecars seeds with Python's per-process randomized hash(cid)) and would change the
augmentation subsample, i.e. an input, not a label.

Three synthetic units (syn_v52 splits only, never in IND / CV5 / full265) inherited a wrong RCR from their library parent
and are re-simulated separately (outputs/cfd_auto_trial_20260927/_label_fix_syn). Until their rebuilt packs exist in FIX
they stay linked to the old directories and the syn_v52 statistics are not written; `finish-syn` then relinks them and
writes the syn_v52 statistics.

    python prepare_v52c.py assemble | verify-codepath | stats | check-features | finish-syn | all
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import time
from pathlib import Path

G = Path("/public/newhome/cy/Digital_twin/GNN")
HERE = Path(__file__).resolve().parent
FIX = G / "data_wss_v5/views_v5_2c_labelfix9_20260930"
SNAP_FIX = G / "data_wss_v5/anatomy_pointcloud_v5_2c_labelfix_20260930"  # corrected V5 snapshot entries of the relinked units
OLD52, NEW52 = G / "data_wss_v5/views_v5_2_full_20260923", G / "data_wss_v5/views_v5_2c_20260930"
OLD265, NEW265 = G / "data_wss_v5/views_v5_2p4_full265_20260930", G / "data_wss_v5/views_v5_2p5_full265_20260930"
UNITS9 = list(json.loads((G / "outputs/cfd_auto_trial_20260927/_label_fix/batch.json").read_text())["units"])
SYN3 = list(json.loads((G / "outputs/cfd_auto_trial_20260927/_label_fix_syn/batch.json").read_text())["units"])
PACKS52 = ["wss_min_view_v1", "wss_min_geom_v2", "wss_min_flowref_v1", "wss_min_cycle_v1", "wss_min_phys1d_v1",
           "wss_min_density_v1/L70", "wss_min_density_v1/L50", "wss_min_density_v1/L35", "wss_min_density_v1/L25"]
PACKS265 = ["wss_min_view_v1", "wss_min_geom_v2", "wss_min_flowref_v1",
            "wss_min_density_v1/L70", "wss_min_density_v1/L50", "wss_min_density_v1/L35", "wss_min_density_v1/L25"]
MANIFESTS = {"wss_min_view_v1": "view_manifest.json", "wss_min_geom_v2": "geom_manifest.json", "wss_min_flowref_v1": "flowref_manifest.json",
             "wss_min_cycle_v1": "cycle_manifest.json", "wss_min_phys1d_v1": "phys1d_manifest.json", "wss_min_density_v1": "density_manifest.json"}
CASE_FILE = {"wss_min_view_v1": "bundle.npz", "wss_min_cycle_v1": "cycle.npz"}
FOLDS = range(5)


def sha256(p: Path) -> str:
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def case_dirs(pack_root: Path) -> list[str]:
    """Canonical ids of the case directories of a pack (cohort/subset/case, three levels; links or real dirs)."""
    out = []
    for c in sorted(pack_root.iterdir()):
        if not c.is_dir() or c.name not in ("AAA", "AG", "ILO"):
            continue
        for s in sorted(c.iterdir()):
            if s.is_dir():
                out += [f"{c.name}/{s.name}/{k.name}" for k in sorted(s.iterdir()) if k.is_dir()]
    return out


def link(src: Path, dst: Path) -> None:
    real = src.resolve()
    if not real.is_dir():
        raise FileNotFoundError(src)
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.is_symlink():
        if dst.resolve() != real:
            raise FileExistsError(f"{dst} -> {dst.resolve()} (expected {real})")
        return
    if dst.exists():
        raise FileExistsError(dst)
    dst.symlink_to(real)


def repoint_json(src: Path, dst: Path, old_root: Path, new_root: Path) -> None:
    """Copy a split / cohort json, re-rooting any path string under old_root and noting the version."""
    text = src.read_text().replace(str(old_root.relative_to(G)), str(new_root.relative_to(G))).replace(str(old_root), str(new_root))
    d = json.loads(text)
    if isinstance(d, dict):
        d["label_fix_20260930"] = (f"copied from {src} into the corrected root {new_root.name}; case lists unchanged; the 9 label-fixed units "
                                   "carry protocol-correct CFD labels")
    dst.parent.mkdir(parents=True, exist_ok=True)
    dst.write_text(json.dumps(d, indent=2, ensure_ascii=False) + "\n")


def syn_ready(pack: str, cid: str) -> bool:
    base = pack.split("/")[0]
    src = FIX / pack / cid
    return src.is_dir() and (base not in CASE_FILE or (src / CASE_FILE[base]).is_file())


def assemble_root(old: Path, new: Path, packs: list[str], version: str) -> dict:
    counts = {}
    for pack in packs:
        base = pack.split("/")[0]
        ids = case_dirs(old / pack)
        n9, pending = 0, []
        for cid in ids:
            if base == "wss_min_density_v1":
                src = old / pack / cid  # geometry-only; kept (see module docstring)
            elif cid in UNITS9 or (cid in SYN3 and syn_ready(pack, cid)):
                src = FIX / pack / cid
                if base in CASE_FILE and not (src / CASE_FILE[base]).is_file():
                    raise FileNotFoundError(src / CASE_FILE[base])
                n9 += 1
            else:
                src = old / pack / cid
                if cid in SYN3:
                    pending.append(cid)
            link(src, new / pack / cid)
        counts[pack] = {"cases": len(ids), "relinked_label_fix": n9}
        if pending:
            counts[pack]["synthetic_pending_rerun"] = pending
        missing9 = [u for u in UNITS9 if u not in ids]
        if missing9 and base != "wss_min_density_v1":
            counts[pack]["label_fix_units_not_in_pack"] = missing9
    # merged manifests: the predecessor's reports with the 9 units' reports replaced by the rebuilt ones
    for base, mname in MANIFESTS.items():
        om = old / base / mname
        if not om.is_file():
            continue
        m = json.loads(om.read_text())
        fm = FIX / base / mname
        fr = {r.get("canonical_id") or r.get("case"): r for r in json.loads(fm.read_text()).get("reports", json.loads(fm.read_text()).get("cases", []))} if fm.is_file() else {}
        for key in ("reports", "cases"):
            if isinstance(m.get(key), list):
                m[key] = [fr.get(r.get("canonical_id") or r.get("case"), r) if isinstance(r, dict) else r for r in m[key]]
        m["label_fix_20260930"] = {"version": version, "relinked_units": UNITS9 + [u for u in SYN3 if syn_ready(base, u)],
                                   "rebuilt_from": str(FIX), "predecessor": str(old)}
        (new / base).mkdir(parents=True, exist_ok=True)
        (new / base / mname).write_text(json.dumps(m, indent=1, ensure_ascii=False))
    return counts


def cmd_assemble(_a) -> None:
    for p in (FIX, OLD52, OLD265):
        assert p.is_dir(), p
    for new in (NEW52, NEW265):
        if new.exists():
            raise FileExistsError(f"{new} exists")
    c52 = assemble_root(OLD52, NEW52, PACKS52, "v5.2c")
    v = "wss_min_view_v1"
    repoint_json(OLD52 / v / "split_v52_ind_train170_test91.json", NEW52 / v / "split_v52_ind_train170_test91.json", OLD52, NEW52)
    for f in sorted((OLD52 / v / "cv5_v52").glob("*.json")):
        if not f.name.startswith("wss_global_stats"):
            repoint_json(f, NEW52 / v / "cv5_v52" / f.name, OLD52, NEW52)
    for f in sorted((OLD52 / v / "syn_v52").glob("*.json")):
        if not f.name.startswith("wss_global_stats"):
            repoint_json(f, NEW52 / v / "syn_v52" / f.name, OLD52, NEW52)
    c265 = assemble_root(OLD265, NEW265, PACKS265, "v5.2p5")
    repoint_json(OLD265 / v / "split_v52p4_full265_train265_test8.json", NEW265 / v / "split_v52p4_full265_train265_test8.json", OLD265, NEW265)
    fix = {"created": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "label_fix_units": UNITS9, "rebuilt_packs": str(FIX), "snapshot": str(SNAP_FIX),
           "v5.2c": {"root": str(NEW52), "predecessor": str(OLD52), "packs": c52},
           "v5.2p5": {"root": str(NEW265), "predecessor": str(OLD265), "packs": c265},
           "script": str(Path(__file__).resolve())}
    write_plans(fix)
    print(json.dumps(fix, indent=1, ensure_ascii=False)[:3000])


def write_plans(fix: dict) -> None:
    """assembly_plan.json of each new root = the predecessor's plan (same keys; readers such as joint_cycle_data use
    plan[cid] to find the source snapshot) with the relinked units re-tagged "v5.2c-labelfix" (source: SNAP_FIX)."""
    relinked = set(fix["label_fix_units"]) | set(fix.get("synthetic_relinked", {}).get("units", []))
    for old, new, version in ((OLD52, NEW52, "v5.2c"), (OLD265, NEW265, "v5.2p5")):
        d = json.loads((old / "assembly_plan.json").read_text())
        d["plan"] = {cid: (tag if cid not in relinked else "v5.2c-labelfix" if isinstance(tag, str)
                           else {"source": "v5.2c-labelfix", "root": str(FIX), "snapshot": str(SNAP_FIX), "was": tag})
                     for cid, tag in d["plan"].items()}
        d["data_version"] = f"{version} (2026-09-30 label fix of {d.get('data_version', old.name)})"
        d["label_fix_20260930"] = fix
        (new / "assembly_plan.json").write_text(json.dumps(d, indent=1, ensure_ascii=False))


def split_train(path: Path) -> list[str]:
    return json.loads(Path(path).read_text())["train_cases"]


def wss_stats_jobs() -> list[tuple[Path, Path, str, str]]:
    """(view root, split, output name relative to the view root, scope text) for every WSS statistics file."""
    v = "wss_min_view_v1"
    jobs = [(NEW52 / v, NEW52 / v / "split_v52_ind_train170_test91.json", "wss_global_stats_ind_train170.json", "v5.2c IND train170")]
    jobs += [(NEW52 / v, NEW52 / v / "cv5_v52" / f"fold{k}.json", f"cv5_v52/wss_global_stats_fold{k}_train.json", f"v5.2c CV5 fold{k} train") for k in FOLDS]
    jobs += [(NEW265 / v, NEW265 / v / "split_v52p4_full265_train265_test8.json", "wss_global_stats_full265_train265.json", "v5.2p5 full265 train265")]
    return jobs


def syn_stats_jobs() -> list[tuple[Path, Path, str, str]]:
    v = "wss_min_view_v1"
    jobs = [(NEW52 / v, NEW52 / v / "syn_v52" / "ind_syn.json", "syn_v52/wss_global_stats_ind_syn_train.json", "v5.2c IND + synthetic train")]
    jobs += [(NEW52 / v, NEW52 / v / "syn_v52" / f"cv5_fold{k}_syn.json", f"syn_v52/wss_global_stats_cv5_fold{k}_syn_train.json", f"v5.2c CV5 fold{k} + synthetic train") for k in FOLDS]
    return jobs


def cmd_verify_codepath(_a) -> None:
    """Re-derive original statistics from the ORIGINAL roots with this code into a scratch dir: they must match bit-exactly
    (all fields but timestamps / paths), which shows that the new files differ only through the label fix."""
    from wss_v5.views import wall_cycle_v1 as CY
    from wss_v5.views.wss_min_view import write_wss_stats
    scratch = HERE / "verify_codepath"
    if scratch.exists():
        shutil.rmtree(scratch)
    scratch.mkdir(parents=True)
    out = {}
    v = "wss_min_view_v1"
    for name, split, ref in (("ind_train170", OLD52 / v / "split_v52_ind_train170_test91.json", OLD52 / v / "wss_global_stats_ind_train170.json"),
                             ("cv5_fold0", OLD52 / v / "cv5_v52/fold0.json", OLD52 / v / "cv5_v52/wss_global_stats_fold0_train.json")):
        shadow = scratch / f"shadow_{name}"
        for cid in split_train(split):
            link(OLD52 / v / cid, shadow / cid)
        p = write_wss_stats(split_train(split), shadow, split, filename="refit.json", scope="verify")
        out[f"wss_{name}"] = compare_stats(json.loads(ref.read_text()), json.loads(p.read_text()))
    cyc = scratch / "shadow_cycle"
    split = OLD52 / v / "split_v52_ind_train170_test91.json"
    for cid in split_train(split):
        link(OLD52 / "wss_min_cycle_v1" / cid, cyc / cid)
    CY.build_stats(cyc, "ind_train170", split)
    for t in ("tawss", "osi_linear", "osi_logit"):
        out[f"cycle_{t}_ind_train170"] = compare_stats(json.loads((OLD52 / "wss_min_cycle_v1/stats" / f"cycle_stats_{t}_ind_train170.json").read_text()),
                                                       json.loads((cyc / "stats" / f"cycle_stats_{t}_ind_train170.json").read_text()))
    out["passed"] = all(r["max_abs_diff"] == 0.0 for k, r in out.items() if isinstance(r, dict))
    (HERE / "verify_codepath.json").write_text(json.dumps(out, indent=1))
    shutil.rmtree(scratch)
    print(json.dumps(out, indent=1))
    if not out["passed"]:
        raise RuntimeError("code path does not reproduce the original statistics")


SKIP = {"generated_at", "created_at", "split_path", "split_sha256", "statistics_scope", "scope", "_provenance", "label_fix_20260930"}


def compare_stats(a, b, path="") -> dict:
    """max abs difference over numeric leaves (dict / list), ignoring provenance fields."""
    worst = [0.0, ""]

    def walk(x, y, p):
        if isinstance(x, dict):
            for k in x:
                if k in SKIP or k not in y:
                    continue
                walk(x[k], y[k], f"{p}.{k}")
        elif isinstance(x, list) and isinstance(y, list):
            if len(x) != len(y):
                worst[:] = [float("inf"), p]; return
            for i, (xi, yi) in enumerate(zip(x, y)):
                walk(xi, yi, f"{p}[{i}]")
        elif isinstance(x, (int, float)) and not isinstance(x, bool) and isinstance(y, (int, float)):
            d = abs(float(x) - float(y))
            if d > worst[0]:
                worst[:] = [d, p]
        elif x != y and not isinstance(x, (dict, list)):
            if worst[0] != float("inf"):
                worst[:] = [float("inf"), p]
    walk(a, b, path)
    return {"max_abs_diff": worst[0], "field": worst[1]}


def write_stats(jobs, report: dict) -> None:
    from wss_v5.views.wss_min_view import write_wss_stats
    for root, split, name, scope in jobs:
        out = root / name
        if out.exists():
            raise FileExistsError(out)
        p = write_wss_stats(split_train(split), root, split, filename=name, scope=f"{scope} partition only (label fix 2026-09-30)")
        old_root = OLD52 if root.parent == NEW52 else OLD265
        old = json.loads((old_root / "wss_min_view_v1" / name).read_text())
        new = json.loads(p.read_text())
        report[name if root.parent == NEW52 else f"full265:{name}"] = {"n_cases": new["n_cases"], "log_mean": [old["log"]["mean"], new["log"]["mean"]],
                                                                         "log_std": [old["log"]["std"], new["log"]["std"]],
                                                                         "p99": [old["raw_percentiles"].get("99", old["raw_percentiles"].get("p99")) if isinstance(old.get("raw_percentiles"), dict) else None,
                                                                                 new["raw_percentiles"].get("99", new["raw_percentiles"].get("p99")) if isinstance(new.get("raw_percentiles"), dict) else None],
                                                                         "diff_vs_predecessor": compare_stats(old, new)}


def cmd_stats(_a) -> None:
    from wss_v5.views import wall_cycle_v1 as CY
    report = {}
    write_stats(wss_stats_jobs(), report)
    if all(syn_ready("wss_min_view_v1", u) for u in SYN3) and all((NEW52 / "wss_min_view_v1" / u).resolve() == (FIX / "wss_min_view_v1" / u).resolve() for u in SYN3):
        write_stats(syn_stats_jobs(), report)
    else:
        report["syn_v52"] = f"not written: synthetic units {SYN3} wait for their re-simulation (finish-syn)"
    cyc = NEW52 / "wss_min_cycle_v1"
    scopes = {"ind_train170": NEW52 / "wss_min_view_v1/split_v52_ind_train170_test91.json"}
    scopes.update({f"cv5_fold{k}": NEW52 / "wss_min_view_v1/cv5_v52" / f"fold{k}.json" for k in FOLDS})
    for scope, split in scopes.items():
        CY.build_stats(cyc, scope, split)
        for t in ("tawss", "osi_linear", "osi_logit"):
            old = json.loads((OLD52 / "wss_min_cycle_v1/stats" / f"cycle_stats_{t}_{scope}.json").read_text())
            new = json.loads((cyc / "stats" / f"cycle_stats_{t}_{scope}.json").read_text())
            report[f"cycle_{t}_{scope}"] = compare_stats(old, new)
    (HERE / "stats_report.json").write_text(json.dumps(report, indent=1, default=float))
    print(json.dumps(report, indent=1, default=float)[:4000])


FEATURE_FILES = {  # feature z-scores used by the mainline configs (inputs are geometry-only -> must be unchanged)
    "ind": (G / "training_wss_min/experiments/wss_v52_20260923/feature_stats/ind_X5Dcap_train_feature_stats.json", NEW52, "split_v52_ind_train170_test91.json",
            "wss_global_stats_ind_train170.json"),
    "full265": (G / "training_wss_min/experiments/wss_v52p4_full265_20260930/feature_stats/full265_X5Dcap_train_feature_stats.json", NEW265,
                "split_v52p4_full265_train265_test8.json", "wss_global_stats_full265_train265.json"),
}


def cmd_check_features(_a) -> None:
    from training_wss_min import config as C
    from training_wss_min import dataset as D
    cfg = json.loads((G / "training_wss_min/configs/wss_v52_phys_20260926/X5Dcap_asym2_s1234.json").read_text())
    feats = tuple(cfg["data"]["input_features"])
    extra = tuple(f for f in feats if f in C.SIDECAR_FEATURE_KEYS)
    out = {}
    for name, (ref, root, split, stats) in FEATURE_FILES.items():
        v = root / "wss_min_view_v1"
        st = D.load_wss_stats(v / stats)
        cases = D.load_partition(str(v / split), "train", st, strict=True, target="wss", data_root=v, required_frame_version="v5_atlas_frame_v1",
                                 extra_point_features=extra, point_features_root=[str(root / "wss_min_geom_v2"), str(root / "wss_min_flowref_v1")])
        fs = D.compute_feature_stats(cases, feats, cfg["data"]["curvature_transform"])
        out[name] = {"n_cases": len(cases), **compare_stats({k: v for k, v in json.loads(ref.read_text()).items() if not k.startswith("_")}, fs), "reference": str(ref)}
    out["passed"] = all(r["max_abs_diff"] <= 1e-9 for r in out.values() if isinstance(r, dict))
    (HERE / "feature_stats_check.json").write_text(json.dumps(out, indent=1))
    print(json.dumps(out, indent=1))
    if not out["passed"]:
        raise RuntimeError("feature z-scores changed: geometry of the relinked units differs")


def cmd_finish_syn(_a) -> None:
    """After the 3 synthetic re-simulations are rebuilt into FIX: relink them in v5.2c and write the syn_v52 statistics."""
    relink_packs = [p for p in PACKS52 if not p.startswith("wss_min_density_v1")]
    missing = [(p, u) for u in SYN3 for p in relink_packs if (OLD52 / p / u).is_dir() and not syn_ready(p, u)]
    if missing:
        raise FileNotFoundError(f"rebuilt synthetic packs missing in {FIX}: {missing}")
    relinked = []
    for p in relink_packs:
        for u in SYN3:
            dst = NEW52 / p / u
            if not (OLD52 / p / u).is_dir():
                continue
            real_old, real_fix = (OLD52 / p / u).resolve(), (FIX / p / u).resolve()
            if dst.resolve() == real_fix:
                continue
            assert dst.is_symlink() and dst.resolve() == real_old, dst
            dst.unlink(); dst.symlink_to(real_fix); relinked.append(f"{p}/{u}")
    for base, mname in MANIFESTS.items():  # manifests: replace the synthetic units' reports too (density: predecessor kept)
        nm, fm = NEW52 / base / mname, FIX / base / mname
        if not (nm.is_file() and fm.is_file()):
            continue
        m, f = json.loads(nm.read_text()), json.loads(fm.read_text())
        fr = {r.get("canonical_id") or r.get("case"): r for r in f.get("reports", f.get("cases", [])) if isinstance(r, dict)}
        for key in ("reports", "cases"):
            if isinstance(m.get(key), list):
                m[key] = [fr.get(r.get("canonical_id") or r.get("case"), r) if isinstance(r, dict) and (r.get("canonical_id") or r.get("case")) in SYN3 else r for r in m[key]]
        m["label_fix_20260930"]["relinked_units"] = UNITS9 + SYN3
        nm.write_text(json.dumps(m, indent=1, ensure_ascii=False))
    report = {"relinked": relinked}
    write_stats(syn_stats_jobs(), report)
    fix = json.loads((NEW52 / "assembly_plan.json").read_text())["label_fix_20260930"]
    fix["synthetic_relinked"] = {"units": SYN3, "at": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "links": relinked}
    write_plans(fix)
    (HERE / "finish_syn_report.json").write_text(json.dumps(report, indent=1, default=float))
    print(json.dumps(report, indent=1, default=float)[:3000])


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["assemble", "verify-codepath", "stats", "check-features", "finish-syn", "rewrite-plans", "all"])
    a = ap.parse_args()
    steps = ["assemble", "verify-codepath", "stats", "check-features"] if a.cmd == "all" else [a.cmd]
    for s in steps:
        print(f"== {s}", flush=True)
        {"assemble": cmd_assemble, "verify-codepath": cmd_verify_codepath, "stats": cmd_stats, "check-features": cmd_check_features,
         "finish-syn": cmd_finish_syn,
         "rewrite-plans": lambda _a: write_plans(json.loads((NEW52 / "assembly_plan.json").read_text()).get("label_fix_20260930")
                                                 or json.loads((NEW52 / "assembly_plan.json").read_text()))}[s](a)


if __name__ == "__main__":
    main()

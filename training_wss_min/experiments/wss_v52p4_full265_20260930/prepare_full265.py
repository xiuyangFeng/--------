"""v5.2 + 4 = full265 data version and the X5Dcap_asym2 three-seed full-data training (2026-09-30, user decisions).

Data version (train 265, val 0, test = recover8):
  261  the v5.2 cases (IND train170 + test91), views read from data_wss_v5/views_v5_2_full_20260923 (read-only)
    1  AAA/ruputer/YANG_BAO_KUI (CFD rebuilt by cfd_auto, swapped into data_new 09-30; views built in the isolated root E)
    3  ILO/XUE_YOU_TANG-0/after, ILO/LI_FA_XIANG-1/before, ILO/WANG_CAI-0/before (recover11 units whose same-patient partner
       is already in v5.2 -> returned to training; views from E)
  test recover8 = recover11 minus those 3 (views from E; no density sidecar, evaluation only)
The new view root only holds symlinks (plus the split / statistics files and merged manifests); the v5.2 views, the v5.2
anatomy snapshot, data_new and the shared PREP are never written.

Recipe: X5Dcap_asym2 (wss_v52_phys_20260926/X5Dcap_asym2_s{1234,7,2025}.json) copied verbatim except the data paths
(data_root, split_path, wss_stats_path, feature_stats_path, point_features_root, density_aug_root) and the run
identity (name, notes).  WSS log_z statistics and feature z-scores are refitted on the 265 training cases with the same
code path as training_wss_min/tools/prepare_wss_v52.py (write_wss_stats / load_partition + compute_feature_stats);
`verify-codepath` first reproduces the v5.2 IND train170 files with that code.

    PYTHONPATH=<GNN_v52p_frozen_20260926> python prepare_full265.py assemble        # login node: symlinks + manifests
    ... python -m training_wss_min.tools.build_density_sidecars (4 new units, v5.2 parameters) -> DENS_NEW
    ... python prepare_full265.py link-density | verify-codepath | split-stats | check | configs
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path

G = Path("/public/newhome/cy/Digital_twin/GNN")
E = Path("/public/newhome/cy/Digital_twin/GNN_recover_eval_20260929")
NAME = "wss_v52p4_full265_20260930"
HERE = G / "training_wss_min/experiments" / NAME
CONFIGS = G / "training_wss_min/configs" / NAME
SRC_CONFIGS = G / "training_wss_min/configs/wss_v52_phys_20260926"
V52 = G / "data_wss_v5/views_v5_2_full_20260923"
ER = E / "data_wss_v5/views_v5_2_recover_20260929"
EY = E / "data_wss_v5/views_v5_2_yang_20260930"
NEW = G / "data_wss_v5/views_v5_2p4_full265_20260930"
DENS_NEW = G / "data_wss_v5/density_v5_2p4_new4_20260930"
VIEW, GEOM, FLOW, DENS = (NEW / "wss_min_view_v1", NEW / "wss_min_geom_v2", NEW / "wss_min_flowref_v1", NEW / "wss_min_density_v1")
V52_SPLIT = V52 / "wss_min_view_v1/split_v52_ind_train170_test91.json"
V52_STATS = V52 / "wss_min_view_v1/wss_global_stats_ind_train170.json"
V52_FEAT = G / "training_wss_min/experiments/wss_v52_20260923/feature_stats/ind_X5Dcap_train_feature_stats.json"
REC_EVAL = G / "outputs/cfd_auto_trial_20260927/_recover/eval"
R8_SPLIT = REC_EVAL / "split_eval_recover8.json"
YANG_BUILD_SPLIT = REC_EVAL / "split_build_yang_20260930.json"
SPLIT = VIEW / "split_v52p4_full265_train265_test8.json"
STATS = VIEW / "wss_global_stats_full265_train265.json"
FEAT = HERE / "feature_stats/full265_X5Dcap_train_feature_stats.json"
SEEDS = (1234, 7, 2025)
LEVELS = (70, 50, 35, 25)
YANG = "AAA/ruputer/YANG_BAO_KUI"
RETURNED = ["ILO/LI_FA_XIANG-1/before", "ILO/WANG_CAI-0/before", "ILO/XUE_YOU_TANG-0/after"]
NEW4 = sorted([YANG] + RETURNED)
R8 = sorted(["AAA/ruputer/FU_GUO_JUN", "AAA/ruputer/LIU_YU_MING", "AAA/ruputer/SHI_YUN_XI", "AAA/ruputer/WANG_SHUN_WEN",
             "AAA/ruputer/ZHANG_ZAO_SHUAN", "AAA/unruputer/LIU_WEN_QI", "ILO/LI_JIE-1/after", "ILO/LI_JIE-1/before"])
VIEWS = ("wss_min_view_v1", "wss_min_geom_v2", "wss_min_flowref_v1")
MANIFESTS = {"wss_min_view_v1": "view_manifest.json", "wss_min_geom_v2": "geom_manifest.json", "wss_min_flowref_v1": "flowref_manifest.json"}
CHANGED = {"name", "notes", "data.data_root", "data.split_path", "data.wss_stats_path", "data.feature_stats_path",
           "data.point_features_root", "data.density_aug_root"}


def sha256(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_once(path: Path, value) -> None:
    """Preregistered files are immutable: rewriting with different content is refused."""
    path.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(value, ensure_ascii=False, indent=2) + "\n"
    if path.exists():
        old = json.loads(path.read_text()); new = json.loads(text)
        for d in (old, new):
            d.pop("created_at", None) if isinstance(d, dict) else None
        if old != new:
            raise FileExistsError(f"refusing to change preregistered file: {path}")
        return
    path.write_text(text)


def v52_cases() -> list[str]:
    s = json.loads(V52_SPLIT.read_text())
    cases = sorted(s["train_cases"] + s["test_cases"])
    assert len(cases) == 261 == len(set(cases)), len(cases)
    return cases


def plan() -> dict[str, dict]:
    p = {c: {"source": "v5.2-full", "root": str(V52)} for c in v52_cases()}
    for c in NEW4:
        assert c not in p, c
        p[c] = {"source": "E-yang" if c == YANG else "E-recover", "root": str(EY if c == YANG else ER), "role": "train (new)"}
    for c in R8:
        assert c not in p, c
        p[c] = {"source": "E-recover", "root": str(ER), "role": "test (recover8)"}
    return p


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
        raise FileExistsError(f"{dst} exists and is not a link")
    dst.symlink_to(real)


def merged_reports(view: str, roots: list[Path], cases: list[str]) -> tuple[list, list]:
    reps = {}
    for root in roots:
        mp = root / view / MANIFESTS[view]
        if mp.is_file():
            for r in json.loads(mp.read_text()).get("reports", []):
                if r.get("canonical_id") in cases:
                    reps.setdefault(r["canonical_id"], r)
    return [reps[c] for c in sorted(reps)], sorted(set(cases) - set(reps))


def cmd_assemble(_a) -> None:
    for p in (V52, ER, EY):
        if not p.is_dir():
            raise FileNotFoundError(p)
    pl = plan()
    for view in VIEWS:
        for cid, spec in pl.items():
            src = Path(spec["root"]) / view / cid
            if view == "wss_min_view_v1" and not (src / "bundle.npz").is_file():
                raise FileNotFoundError(src / "bundle.npz")
            link(src, NEW / view / cid)
        roots = [V52, EY, ER]
        reps, missing = merged_reports(view, roots, sorted(pl))
        if view == "wss_min_view_v1" and missing:
            raise RuntimeError(f"view reports missing: {missing}")
        write_once(NEW / view / MANIFESTS[view], {"view": view, "assembled": "2026-09-30", "data_version": "v5.2p4 full265",
                   "sources": {"v5.2-full": str(V52), "E-recover": str(ER), "E-yang": str(EY)}, "cases": len(pl),
                   "reports": reps, "missing_reports": missing,
                   "note": "symlinks to the source case directories; reports merged from the source manifests (sidecar manifests of the E runs only keep their last run, hence missing_reports)"})
    counts = {}
    for spec in pl.values():
        counts[spec["source"]] = counts.get(spec["source"], 0) + 1
    write_once(NEW / "assembly_plan.json", {"data_version": "v5.2p4 full265 (2026-09-30)", "counts": counts, "train": 265, "test_recover8": 8,
                                            "new_train_units": NEW4, "recover8": R8, "plan": pl,
                                            "density": {"v5.2 cases": str(V52 / "wss_min_density_v1"), "new4": str(DENS_NEW), "recover8": "not built (evaluation only)"},
                                            "script": str(Path(__file__).resolve())})
    print("assembled", NEW, counts)


def cmd_link_density(_a) -> None:
    train = v52_cases() + NEW4
    for lv in LEVELS:
        for cid in train:
            root = DENS_NEW if cid in NEW4 else V52 / "wss_min_density_v1"
            src = root / f"L{lv}" / cid
            if not (src / "features.npz").is_file():
                raise FileNotFoundError(src / "features.npz")
            link(src, DENS / f"L{lv}" / cid)
    reps = {}
    for root in (V52 / "wss_min_density_v1", DENS_NEW):
        mp = root / "density_manifest.json"
        if mp.is_file():
            for r in json.loads(mp.read_text()).get("reports", []):
                if r["canonical_id"] in train:
                    reps.setdefault(r["canonical_id"], r)
    new_m = json.loads((DENS_NEW / "density_manifest.json").read_text())
    write_once(DENS / "density_manifest.json", {"pack": "wss_min_density_v1", "version": new_m["version"], "levels": list(LEVELS), "seed": new_m["seed"],
               "assembled": "2026-09-30", "sources": {"v5.2 cases": str(V52 / "wss_min_density_v1"), "new4": str(DENS_NEW)},
               "cases": len(train), "reports": [reps[c] for c in sorted(reps)], "missing_reports": sorted(set(train) - set(reps)),
               "note": "training cases only; recover8 has no density sidecar (evaluation does not use it)"})
    print("density links", len(train), "x", len(LEVELS))


def load_train(split: Path, stats_path: Path, view: Path, geom: Path, flow: Path, cfg_src: dict):
    from training_wss_min import config as C
    from training_wss_min import dataset as D
    stats = D.load_wss_stats(stats_path)
    extra = tuple(f for f in cfg_src["data"]["input_features"] if f in C.SIDECAR_FEATURE_KEYS)
    cases = D.load_partition(str(split), "train", stats, strict=True, target="wss", data_root=view,
                             required_frame_version="v5_atlas_frame_v1", extra_point_features=extra, point_features_root=[str(geom), str(flow)])
    expected = json.loads(Path(split).read_text())["counts"]["train"]
    if len(cases) != expected:
        raise RuntimeError(f"expected {expected} train cases, loaded {len(cases)}")
    return cases


def max_abs_diff(a, b, path="") -> tuple[float, str]:
    if isinstance(a, dict):
        out = (0.0, "")
        for k in a:
            if k.startswith("_") or k in ("generated_at", "split_path", "split_sha256", "statistics_scope"):
                continue
            out = max(out, max_abs_diff(a[k], b[k], f"{path}.{k}"))
        return out
    if isinstance(a, list):
        if len(a) != len(b):
            return (float("inf"), path)
        return max([max_abs_diff(x, y, f"{path}[{i}]") for i, (x, y) in enumerate(zip(a, b))], default=(0.0, ""))
    if isinstance(a, (int, float)) and not isinstance(a, bool):
        return (abs(float(a) - float(b)), path)
    return (0.0 if a == b else float("inf"), path)


def cmd_verify_codepath(_a) -> None:
    """Refit the v5.2 IND train170 WSS statistics and X5Dcap feature z-scores with this code into a scratch dir and
    compare with the files the X5Dcap_asym2 IND runs used."""
    from training_wss_min import dataset as D
    from wss_v5.views.wss_min_view import write_wss_stats
    scratch = HERE / "verify_codepath"
    scratch.mkdir(parents=True, exist_ok=True)
    train170 = json.loads(V52_SPLIT.read_text())["train_cases"]
    # write_wss_stats(train, out_root, split, filename) reads out_root/<cid>/bundle.npz and writes out_root/filename:
    # point out_root at a scratch dir of links to the 170 bundles so nothing is written into the v5.2 view root
    shadow = scratch / "shadow_view"
    for cid in train170:
        link(V52 / "wss_min_view_v1" / cid, shadow / cid)
    p = write_wss_stats(train170, shadow, V52_SPLIT, filename="wss_stats_ind_train170_refit.json", scope="verify refit")
    ref = json.loads(V52_STATS.read_text()); new = json.loads(p.read_text())
    ws = max_abs_diff({k: ref[k] for k in ("n_cases", "n_points", "zero_frac", "linear", "log", "raw_min", "raw_max", "raw_percentiles", "train_units")},
                      {k: new[k] for k in ("n_cases", "n_points", "zero_frac", "linear", "log", "raw_min", "raw_max", "raw_percentiles", "train_units")})
    cfg_src = json.loads((SRC_CONFIGS / "X5Dcap_asym2_s1234.json").read_text())
    cases = load_train(V52_SPLIT, V52_STATS, V52 / "wss_min_view_v1", V52 / "wss_min_geom_v2", V52 / "wss_min_flowref_v1", cfg_src)
    fs = D.compute_feature_stats(cases, tuple(cfg_src["data"]["input_features"]), cfg_src["data"]["curvature_transform"])
    fref = json.loads(V52_FEAT.read_text())
    fd = max_abs_diff({k: v for k, v in fref.items() if not k.startswith("_")}, fs)
    out = {"wss_stats_max_abs_diff": ws[0], "wss_stats_field": ws[1], "feature_stats_max_abs_diff": fd[0], "feature_stats_field": fd[1],
           "reference_wss_stats": str(V52_STATS), "reference_feature_stats": str(V52_FEAT), "n_cases": len(cases),
           "passed": bool(ws[0] <= 1e-9 and fd[0] <= 1e-6), "checked_at": time.strftime("%Y-%m-%dT%H:%M:%S%z")}
    (scratch / "verify_codepath.json").write_text(json.dumps(out, indent=2) + "\n")
    for cid in train170:
        (shadow / cid).unlink()
    print("verify-codepath", out)
    if not out["passed"]:
        raise RuntimeError("the v5.2 code path does not reproduce the IND train170 statistics")


def cmd_split_stats(_a) -> None:
    from training_wss_min import dataset as D
    from wss_v5.views.wss_min_view import write_wss_stats
    v52 = json.loads(V52_SPLIT.read_text())
    train = sorted(v52_cases() + NEW4)
    assert len(train) == 265 and not set(train) & set(R8)
    payload = {"schema_version": 1, "pipeline": "wss_v5.views.wss_min_view", "data_root": str(VIEW.relative_to(G)),
               "required_frame_version": "v5_atlas_frame_v1",
               "id_format": "canonical cohort/subset/case; ILO subset is patient-0|1 and case is before|after",
               "created_at": time.strftime("%Y-%m-%dT%H:%M:%S+00:00", time.gmtime()), "split_version": "split_v52p4_full265_train265_test8",
               "source_splits": {str(V52_SPLIT): sha256(V52_SPLIT), str(R8_SPLIT): sha256(R8_SPLIT), str(YANG_BUILD_SPLIT): sha256(YANG_BUILD_SPLIT)},
               "duplicate_geometry_groups": [["AG/slow/HOU_SHEN_QIAN", "AG/slow/KANG_XI_MING"]],
               "explicit_related_groups": v52.get("explicit_related_groups", [["AG/slow/HOU_SHEN_QIAN", "AG/slow/KANG_XI_MING"]]),
               "duplicate_geometry_note": "KANG_XI_MING kept (HOU_SHEN_QIAN excluded since v5.1). The library LIU_WEN_QI (a copy of ZHU_ZI_HAI's CFD) stays excluded; recover8's LIU_WEN_QI is a cfd_auto run on its own STL (another anatomy) and is evaluation-only.",
               "train_cases": train, "val_cases": [], "test_cases": R8, "unused_cases": [],
               "counts": {"train": len(train), "val": 0, "test": len(R8)}, "expected_counts": {"train": len(train), "val": 0, "test": len(R8)},
               "new_train_units": NEW4, "note": "v5.2p4 full265 (2026-09-30 user): train = v5.2 261 (IND train170 + test91) + YANG_BAO_KUI + 3 recover11 units whose partner is in v5.2; no validation (selection = train_loss as in the IND recipe); test = recover8"}
    write_once(SPLIT, payload)
    if not STATS.is_file():
        write_wss_stats(train, VIEW, SPLIT, filename=STATS.name, scope="v5.2p4 full265 train partition only (265 cases, valid anatomy wall nodes)")
    stats = json.loads(STATS.read_text())
    assert stats["n_cases"] == 265 and stats["split_sha256"] == sha256(SPLIT), stats["n_cases"]
    if not FEAT.is_file():
        cfg_src = json.loads((SRC_CONFIGS / "X5Dcap_asym2_s1234.json").read_text())
        cases = load_train(SPLIT, STATS, VIEW, GEOM, FLOW, cfg_src)
        out = D.compute_feature_stats(cases, tuple(cfg_src["data"]["input_features"]), cfg_src["data"]["curvature_transform"])
        out["_provenance"] = {"train_split": str(SPLIT), "n_cases": len(cases), "wss_stats": str(STATS), "sidecar_roots": [str(GEOM), str(FLOW)],
                              "curvature_transform": cfg_src["data"]["curvature_transform"],
                              "note": "v5.2p4 full265: train-partition z-scores for every non-xyz input feature (same code path as prepare_wss_v52.feature_stats)"}
        write_once(FEAT, out)
    print("split", SPLIT, "stats", STATS, "feature stats", FEAT)


def cmd_check(_a) -> None:
    """Patient isolation (Registry.patient_group), density sidecars loadable for every training case at every level,
    new units' WSS inside the training distribution."""
    from training_wss_min import dataset as D
    from wss_v5.sources import Registry
    reg = Registry(split={"cv": {}}, wall_audit={}, solver_audit={})
    sp = json.loads(SPLIT.read_text())
    tr_groups = {reg.patient_group(c) for c in sp["train_cases"]}
    leaks = [c for c in sp["test_cases"] if reg.patient_group(c) in tr_groups]
    out = {"patient_groups_train": len(tr_groups), "recover8_groups": sorted({reg.patient_group(c) for c in sp["test_cases"]}), "leaks": leaks}
    stats = D.load_wss_stats(STATS)
    density = {}
    for cid in sp["train_cases"]:
        cohort, sub, case = cid.split("/", 2)
        c = D.load_case(f"{cohort}/{sub}", case, stats, target="wss", data_root=VIEW, required_frame_version="v5_atlas_frame_v1")
        ns = {}
        for lv in LEVELS:
            d = D.load_density_level(c, DENS, lv)
            ns[lv] = round(len(d["rows"]) / len(c["pos"]), 3)
        if cid in NEW4:
            density[cid] = {"fractions": ns, "n_wall": int(len(c["pos"]))}
        bad = [lv for lv, f in ns.items() if not (0.5 * lv / 100 <= f <= 1.5 * lv / 100)]
        if bad:
            raise RuntimeError(f"{cid}: density fractions off {ns}")
    out["density_new4"] = density
    out["density_checked_cases"] = len(sp["train_cases"])
    out["passed"] = not leaks
    write_once(HERE / "data_checks.json", out)
    print(json.dumps(out, indent=1, ensure_ascii=False))
    if leaks:
        raise RuntimeError(f"patient leakage: {leaks}")


def flat(d: dict, prefix: str = "") -> dict:
    out = {}
    for k, v in d.items():
        key = f"{prefix}.{k}" if prefix else k
        if isinstance(v, dict) and key in ("data", "model", "train", "eval"):
            out.update(flat(v, key))
        else:
            out[key] = v
    return out


def cmd_configs(_a) -> None:
    from training_wss_min import config as C
    for p in (SPLIT, STATS, FEAT, DENS / "density_manifest.json"):
        if not p.is_file():
            raise FileNotFoundError(p)
    diffs, arms = {}, []
    for seed in SEEDS:
        src_path = SRC_CONFIGS / f"X5Dcap_asym2_s{seed}.json"
        src = json.loads(src_path.read_text())
        cfg = json.loads(src_path.read_text())
        arm = f"X5Dcap_asym2_full265_s{seed}"
        cfg["name"] = f"{NAME}/{arm}"
        cfg["notes"] = (f"v5.2p4 full265 X5Dcap_asym2 seed {seed}: recipe = wss_v52_phys_20260926/X5Dcap_asym2_s{seed}.json verbatim; "
                        "only data paths changed (full265 view root: v5.2 261 + YANG_BAO_KUI + 3 returned recover units; test = recover8; "
                        "WSS stats / feature z-scores refitted on the 265 training cases).")
        cfg["data"]["data_root"] = str(VIEW)
        cfg["data"]["split_path"] = str(SPLIT)
        cfg["data"]["wss_stats_path"] = str(STATS)
        cfg["data"]["feature_stats_path"] = str(FEAT)
        cfg["data"]["point_features_root"] = [str(GEOM), str(FLOW)]
        cfg["data"]["density_aug_root"] = str(DENS)
        C.ExpConfig.from_dict(cfg)
        fa, fb = flat(src), flat(cfg)
        diff = {k: {"source": fa.get(k), "full265": fb.get(k)} for k in sorted(set(fa) | set(fb)) if fa.get(k) != fb.get(k)}
        if set(diff) != CHANGED:
            raise RuntimeError(f"{arm}: unexpected changed fields {set(diff) ^ CHANGED}")
        write_once(CONFIGS / f"{arm}.json", cfg)
        diffs[arm] = {"source": str(src_path), "source_sha256": sha256(src_path), "changed_fields": diff}
        arms.append(dict(id=arm, title=f"X5Dcap_asym2 recipe on full265 (v5.2 261 + 4), seed {seed}; test = recover8", config=f"{arm}.json",
                         phase=0, modules=["F6", "density_aug", "cap_prior", "tail_loss"], seed=seed, fold=None, protocol="FULL265",
                         arm="X5Dcap_asym2_full265", parent=f"X5Dcap_asym2_s{seed} (wss_v52_phys_20260926)", depends_on=[],
                         evaluate=["best", "last"], single_change=False, config_diff_vs_parent=diff))
    write_once(HERE / "config_diff_vs_source.json", {"allowed_changes": sorted(CHANGED), "arms": diffs})
    write_once(CONFIGS / "matrix.json", dict(
        schema_version=1, experiment=NAME, anchor=str(G / "training_wss_min/configs/wss_direct_recovery_20260912/C1_s1234.json"),
        anchor_run=str(G / "training_wss_min/runs/wss_v51_wave1_20260916/X5D_v51_s1234"), control_id=None,
        external_reference_runs={f"X5Dcap_asym2_s{s}": str(G / f"training_wss_min/runs/wss_v52_phys_20260926/X5Dcap_asym2_s{s}") for s in SEEDS},
        section="v5.2p4 full265 base training (2026-09-30)",
        data={"views_root": str(NEW), "train": 265, "test": "recover8", "split": str(SPLIT), "wss_stats": str(STATS), "feature_stats": str(FEAT),
              "density": str(DENS), "new_train_units": NEW4},
        expected_training_runs=len(arms), expected_evaluations=2 * len(arms), arms=arms,
        reading_rule=("recover8 physical Pa R2_cb of the three-seed Pa-mean ensemble (evaluate_recover8.sh), compared with the X5Dcap_asym2 IND "
                      "models on recover8: five-seed 0.8393, seeds 1234/7/2025 0.8352 (singles 0.8194/0.8239/0.8188), median unit R2 0.830/0.827. "
                      "test91 is inside the full265 training set and cannot be used. Descriptive (n=8), not a gate.")))
    print("configs", CONFIGS)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["assemble", "link-density", "verify-codepath", "split-stats", "check", "configs"])
    a = ap.parse_args()
    {"assemble": cmd_assemble, "link-density": cmd_link_density, "verify-codepath": cmd_verify_codepath,
     "split-stats": cmd_split_stats, "check": cmd_check, "configs": cmd_configs}[a.cmd](a)


if __name__ == "__main__":
    main()

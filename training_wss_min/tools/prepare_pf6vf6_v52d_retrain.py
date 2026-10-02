"""PF6 (pressure) / VF6 (velocity) on v5.2d: volume data root, statistics, configs (2026-10-02; 01 block tracking §43).

The user confirmed the §43 matrix on 2026-10-02: PF6 and VF6 each CV5 5 folds x seeds 1234 / 7 / 2025 (30 arms) + full265 x 3
seeds evaluated on recover8 (6 arms); recipes verbatim, only the data changes; no prior-ablation arms; no gates.

Volume data root (why a separate root): the v5.2d main view root `wss_min_view_v1` holds `volume.npz` for 159 units only.
`joint_cycle_data.source_paths` prefers the main root's `volume.npz` over its own derived cache, so adding files there would
break the full-cycle cache source signatures. The new root `views_v5_2d_20261001/wss_min_volview_v1/<unit>/` therefore holds,
for the 273 real units (full265 train 265 + recover8): `bundle.npz` -> symlink to the main view root, and `volume.npz` ->
symlink to an existing volume view (main root 159 / full-cycle derived cache 102) *only if* a fresh build from the v5.2d
snapshot is array-for-array identical, otherwise the fresh build itself (the 12 units never built, and any mismatch).

Subcommands (data steps run on node04's CPUs, outside Slurm; nothing here needs a GPU):
  build      fresh `build_case_volume` of all 273 units from the v5.2d snapshot into a staging root (bundle links)
  compare    every existing volume view vs its fresh build, every array (keys, dtype, shape, values); cache provenance check
  assemble   create wss_min_volview_v1 (links / fresh builds) + volview_manifest.json   [refuses unless compare is clean
             or --accept-mismatch-replacement is given after the mismatches were reviewed]
  stats      volume statistics (write_volume_stats, linear z) for the six training partitions, renamed per partition
  featstats  input-feature normalisation of the six partitions (PF6 and VF6 configs; the two are expected identical)
  configs    36 arm configs + matrix (+ host sub-matrices), declared-diff check vs the source configs, anchor and E5 dirs
  cleanup    remove the staging copies that are not referenced by the new root

    python -u -m training_wss_min.tools.prepare_pf6vf6_v52d_retrain <subcommand> [--workers N]
"""
from __future__ import annotations

import argparse
import hashlib
import json
import multiprocessing as mp
import os
import shutil
import socket
import time
from pathlib import Path

import numpy as np

from training_wss_min import config as C

ROOT = C.PROJECT_ROOT
NAME = "pf6vf6_v52d_retrain_20261002"
CONFIGS = ROOT / "training_wss_min/configs" / NAME
EXP = ROOT / "training_wss_min/experiments" / NAME
PREP = EXP / "data_prep"
V = ROOT / "data_wss_v5/views_v5_2d_20261001"
VIEW = V / "wss_min_view_v1"
VOLVIEW = V / "wss_min_volview_v1"
FLOWREF = V / "wss_min_flowref_v1"
STAGE = V / "_volview_staging_20261002"
SNAP = ROOT / "data_wss_v5/anatomy_pointcloud_v5_2d_20261001"
CACHE = ROOT / "training_wss_min/experiments/joint_cycle_v52d_20261001/data_audit_cache/derived_volume"
FULL_SPLIT = VIEW / "split_v52p4_full265_train265_test8.json"
CV5 = VIEW / "cv5_v52"
FOLDS = (0, 1, 2, 3, 4)
SEEDS = (1234, 7, 2025)
FROZEN = "GNN_pf6vf6_v52d_frozen_20261002"
RELEASE = ROOT / "outputs/wss_deploy_release/PF6_VF6_peak_3seed_20260920"
OLD_CODE = ROOT.parent / "GNN_voltime_frozen_20260919"   # code of the v5.1 PF6/VF6 fold bases (cross-version anchor)
TARGETS = {"PF6": "pressure_mixed", "VF6": "velocity"}
# source recipes: seed 1234 from wave 2, seeds 7 / 2025 from wave 3 (identical except seed/name/notes/feature stats/init ref)
SRC = {(p, s): ROOT / "training_wss_min/configs" / ("wss_local_wave2_20260912" if s == 1234 else "wss_local_wave3_20260913")
       / f"{p}_s{s}.json" for p in TARGETS for s in SEEDS}
ALLOWED_DIFF = {"name", "notes", "data.data_root", "data.split_path", "data.wss_stats_path", "data.feature_stats_path",
                "data.point_features_root"}
KNOWN_LIMITS = {"early_exit_iterations": "150 units (AG 105, AAA 25, ILO 20) keep labels from runs whose per-step iterations "
                                         "stopped at 1e-3; one measured unit: outlet pressure off by 1.6-2.1 %, peak WSS rel L2 "
                                         "4.3 % vs the iteration-converged solution; user decision 2026-10-01: leave as is"}


# ---------------------------------------------------------------------------------------------------------------- helpers
def now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%S%z")


def sha256(path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(8 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def signature(path) -> dict:
    p = Path(path).resolve()
    s = p.stat()
    return {"path": str(p), "size": s.st_size, "mtime_ns": s.st_mtime_ns}


def write_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, indent=1, ensure_ascii=False, default=float) + "\n")
    tmp.replace(path)


def save_frozen(path: Path, value) -> None:
    """Pre-registered file: identical rewrite is a no-op, a different one is refused."""
    from training_wss_min.tools import prepare_wss_local_wave1 as W1
    W1.save(path, value)


def run_context() -> dict:
    return {"host": socket.gethostname(), "pid": os.getpid(), "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
            "executed_outside_slurm": not os.environ.get("SLURM_JOB_ID"), "at": now()}


def units() -> list[str]:
    full = json.loads(FULL_SPLIT.read_text())
    out = list(full["train_cases"]) + list(full["test_cases"])
    if len(out) != 273 or len(set(out)) != 273:
        raise RuntimeError(f"expected 273 distinct real units, got {len(out)}")
    return out


def partitions() -> list[tuple[str, Path]]:
    return [(f"cv5_fold{k}", CV5 / f"fold{k}.json") for k in FOLDS] + [("full265", FULL_SPLIT)]


def existing_volume(cid: str) -> tuple[str, Path] | tuple[None, None]:
    main, cache = VIEW / cid / "volume.npz", CACHE / cid / "volume.npz"
    if main.is_file() and cache.is_file():
        raise RuntimeError(f"{cid}: volume view in both the main root and the derived cache")
    if main.is_file():
        return "main_view", main
    if cache.is_file():
        return "joint_cycle_cache", cache
    return None, None


def builder_hashes() -> dict:
    return {str(p.relative_to(ROOT)): sha256(p) for p in (ROOT / "wss_v5/views/volume_view.py", ROOT / "wss_v5/contract.py",
                                                         ROOT / "wss_v5/views/wss_min_view.py")}


# ------------------------------------------------------------------------------------------------------------------ build
def _build_one(cid: str) -> dict:
    from wss_v5.views.volume_view import build_case_volume
    d = STAGE / cid
    try:
        d.mkdir(parents=True, exist_ok=True)
        bundle = (VIEW / cid / "bundle.npz").resolve()
        link = d / "bundle.npz"
        if link.is_symlink() or link.exists():
            if link.resolve() != bundle:
                raise RuntimeError(f"stale staging bundle link -> {link.resolve()}")
        else:
            link.symlink_to(bundle)
        h5 = SNAP / "cases" / cid.replace("/", "__") / "case.h5"
        if not h5.is_file():
            raise FileNotFoundError(f"v5.2d snapshot case.h5 missing: {h5}")
        rep = build_case_volume(cid, SNAP, STAGE)
        rep["sources"] = {"h5": signature(h5), "bundle": signature(bundle)}
        return rep
    except Exception as exc:  # noqa: BLE001 - reported per unit
        return {"canonical_id": cid, "error": f"{type(exc).__name__}: {exc}"}


def cmd_build(args) -> int:
    todo = units()
    report_path = PREP / "build_report.json"
    done = {}
    if report_path.is_file():
        done = {r["canonical_id"]: r for r in json.loads(report_path.read_text())["units"] if "error" not in r
                and (STAGE / r["canonical_id"] / "volume.npz").is_file()}
    jobs = [u for u in todo if u not in done]
    print(f"[build] {len(todo)} units, {len(done)} already built, {len(jobs)} to build, workers={args.workers}", flush=True)
    results = dict(done)
    started = time.time()
    with mp.get_context("fork").Pool(max(1, args.workers)) as pool:
        for i, rep in enumerate(pool.imap_unordered(_build_one, jobs)):
            results[rep["canonical_id"]] = rep
            msg = rep.get("error") or f"n={rep['n_cells']} {rep['seconds']:.1f}s"
            print(f"[build] {i + 1}/{len(jobs)} {rep['canonical_id']} {msg}", flush=True)
            if (i + 1) % 20 == 0 or i + 1 == len(jobs):
                write_json(report_path, {"generated_at": now(), "context": run_context(), "snapshot_root": str(SNAP),
                                         "staging_root": str(STAGE), "builder": builder_hashes(),
                                         "units": [results[u] for u in todo if u in results]})
    errors = [r for r in results.values() if "error" in r]
    write_json(report_path, {"generated_at": now(), "context": run_context(), "snapshot_root": str(SNAP),
                             "staging_root": str(STAGE), "builder": builder_hashes(), "seconds": time.time() - started,
                             "n_units": len(todo), "n_errors": len(errors), "units": [results[u] for u in todo]})
    print(f"[build] done: {len(results)} units, {len(errors)} errors, {time.time() - started:.0f}s", flush=True)
    for r in errors:
        print("[build] ERROR", r["canonical_id"], r["error"], flush=True)
    return 1 if errors else 0


# ---------------------------------------------------------------------------------------------------------------- compare
def compare_npz(existing: Path, fresh: Path) -> dict:
    with np.load(existing, allow_pickle=True) as a, np.load(fresh, allow_pickle=True) as b:
        ka, kb = set(a.files), set(b.files)
        out = {"keys_only_existing": sorted(ka - kb), "keys_only_fresh": sorted(kb - ka), "arrays": len(ka & kb), "diff": {}}
        for k in sorted(ka & kb):
            x, y = a[k], b[k]
            if x.dtype != y.dtype or x.shape != y.shape:
                out["diff"][k] = {"dtype": [str(x.dtype), str(y.dtype)], "shape": [list(x.shape), list(y.shape)]}
                continue
            same = np.array_equal(x, y, equal_nan=True) if x.dtype.kind in "fc" else np.array_equal(x, y)
            if not same:
                d = {"dtype": str(x.dtype), "shape": list(x.shape)}
                if x.dtype.kind in "fiub" and x.size:
                    xd, yd = x.astype(np.float64), y.astype(np.float64)
                    diff = np.abs(xd - yd)
                    d.update(n_diff=int(np.sum(~np.isclose(xd, yd, rtol=0, atol=0, equal_nan=True))),
                             max_abs_diff=float(np.nanmax(diff)),
                             max_rel_diff=float(np.nanmax(diff / np.maximum(np.abs(yd), 1e-12))))
                else:
                    d["values"] = [str(x)[:120], str(y)[:120]]
                out["diff"][k] = d
    out["identical"] = not out["diff"] and not out["keys_only_existing"] and not out["keys_only_fresh"]
    return out


def _compare_one(cid: str) -> dict:
    kind, path = existing_volume(cid)
    fresh = STAGE / cid / "volume.npz"
    rec = {"canonical_id": cid, "existing_kind": kind, "existing_path": str(path) if path else None}
    try:
        if not fresh.is_file():
            raise FileNotFoundError(f"fresh build missing: {fresh}")
        if path is None:
            rec["status"] = "new_unit"
            return rec
        rec.update(compare_npz(path, fresh))
        rec["status"] = "identical" if rec["identical"] else "mismatch"
        if kind == "joint_cycle_cache":   # same provenance rule as joint_cycle_data.source_paths
            prov = json.loads((path.parent / "source.json").read_text())
            h5 = SNAP / "cases" / cid.replace("/", "__") / "case.h5"
            expected = {"h5": signature(h5), "bundle": signature(VIEW / cid / "bundle.npz")}
            rec["cache_provenance_current"] = prov["sources"] == expected
            rec["cache_builder_sha256"] = prov.get("builder_sha256")
    except Exception as exc:  # noqa: BLE001
        rec.update(status="error", error=f"{type(exc).__name__}: {exc}")
    return rec


def cmd_compare(args) -> int:
    todo = units()
    with mp.get_context("fork").Pool(max(1, args.workers)) as pool:
        recs = list(pool.imap(_compare_one, todo))
    by = {}
    for r in recs:
        by.setdefault(r["status"], []).append(r["canonical_id"])
    builder = builder_hashes()["wss_v5/views/volume_view.py"]
    stale_cache = [r["canonical_id"] for r in recs if r.get("cache_provenance_current") is False]
    other_builder = [r["canonical_id"] for r in recs if r.get("cache_builder_sha256") not in (None, builder)]
    summary = {k: len(v) for k, v in by.items()}
    summary.update(existing_main_view=sum(r["existing_kind"] == "main_view" for r in recs),
                   existing_joint_cycle_cache=sum(r["existing_kind"] == "joint_cycle_cache" for r in recs),
                   cache_provenance_stale=len(stale_cache), cache_other_builder=len(other_builder))
    write_json(PREP / "compare_report.json", {"generated_at": now(), "context": run_context(), "summary": summary,
                                              "mismatch_units": by.get("mismatch", []), "error_units": by.get("error", []),
                                              "new_units": by.get("new_unit", []), "cache_provenance_stale": stale_cache,
                                              "cache_other_builder": other_builder, "builder_sha256": builder, "units": recs})
    print("[compare]", json.dumps(summary, ensure_ascii=False), flush=True)
    for r in recs:
        if r["status"] in ("mismatch", "error"):
            print("[compare]", r["status"], r["canonical_id"], r["existing_kind"],
                  r.get("error") or sorted(r["diff"]), flush=True)
    return 0 if not by.get("error") else 1


# --------------------------------------------------------------------------------------------------------------- assemble
def _link(link: Path, target: Path) -> None:
    target = target.resolve()
    if link.is_symlink():
        if link.resolve() != target:
            raise RuntimeError(f"{link} already links to {link.resolve()}, expected {target}")
        return
    if link.exists():
        raise RuntimeError(f"{link} exists and is not a link")
    link.symlink_to(target)


def cmd_assemble(args) -> int:
    comp = json.loads((PREP / "compare_report.json").read_text())
    if comp["error_units"]:
        raise RuntimeError(f"compare has errors: {comp['error_units']}")
    if comp["cache_provenance_stale"]:
        raise RuntimeError(f"derived cache provenance stale for {comp['cache_provenance_stale']}")
    if comp["mismatch_units"] and not args.accept_mismatch_replacement:
        raise RuntimeError(f"{len(comp['mismatch_units'])} existing volume views differ from the v5.2d rebuild; review "
                           f"{PREP / 'compare_report.json'} and rerun with --accept-mismatch-replacement")
    recs = {r["canonical_id"]: r for r in comp["units"]}
    todo = units()
    if sorted(recs) != sorted(todo):
        raise RuntimeError("compare report does not cover the 273 units")
    manifest = []
    for i, cid in enumerate(todo):
        r = recs[cid]
        d = VOLVIEW / cid
        d.mkdir(parents=True, exist_ok=True)
        _link(d / "bundle.npz", VIEW / cid / "bundle.npz")
        vol = d / "volume.npz"
        if r["status"] == "identical":
            _link(vol, Path(r["existing_path"]))
            source = "link_main_view" if r["existing_kind"] == "main_view" else "link_joint_cycle_cache"
        else:
            source = "new_build" if r["status"] == "new_unit" else "rebuilt_replaces_mismatch"
            staged = STAGE / cid / "volume.npz"
            if vol.is_symlink():
                raise RuntimeError(f"{vol} is a link but the unit needs its fresh build")
            if not vol.exists():
                shutil.move(str(staged), str(vol))
            elif staged.exists():
                raise RuntimeError(f"{vol} and the staging copy both exist")
        manifest.append({"unit": cid, "volume_source": source, "volume_target": str(vol.resolve()),
                         "volume_sha256": sha256(vol), "bundle_target": str((d / "bundle.npz").resolve()),
                         "replaced_existing": r["existing_path"] if source == "rebuilt_replaces_mismatch" else None,
                         "mismatched_arrays": sorted(r.get("diff", {})) if source == "rebuilt_replaces_mismatch" else None})
        if (i + 1) % 50 == 0:
            print(f"[assemble] {i + 1}/{len(todo)}", flush=True)
    counts = {}
    for m in manifest:
        counts[m["volume_source"]] = counts.get(m["volume_source"], 0) + 1
    write_json(VOLVIEW / "volview_manifest.json", {
        "schema_version": 1, "generated_at": now(), "context": run_context(), "experiment": NAME,
        "purpose": "PF6/VF6 volume-target data root for v5.2d (273 real units); bundle.npz links the v5.2d main view root, "
                   "volume.npz links an existing volume view only when a fresh build from the v5.2d snapshot is "
                   "array-for-array identical, otherwise holds that fresh build",
        "why_separate_root": "joint_cycle_data.source_paths prefers <main view>/<unit>/volume.npz over its derived cache; "
                             "adding volume.npz to the main view root would break the full-cycle cache source signatures",
        "snapshot_root": str(SNAP), "main_view_root": str(VIEW), "joint_cycle_cache": str(CACHE),
        "builder": builder_hashes(), "compare_report": str(PREP / "compare_report.json"),
        "compare_report_sha256": sha256(PREP / "compare_report.json"), "counts": counts,
        "dependencies": "links resolve into the main view root and the joint_cycle_v52d_20261001 derived cache; do not delete "
                        "those volume.npz files while this root is in use", "units": manifest})
    print("[assemble]", counts, flush=True)
    return 0


# ------------------------------------------------------------------------------------------------------------------ stats
def stats_path(scope: str, n: int) -> Path:
    return VOLVIEW / "stats" / f"volume_stats_{scope}_train{n}.json"


def cmd_stats(args) -> int:
    from wss_v5.views.volume_view import write_volume_stats
    manifest_sha = sha256(VOLVIEW / "volview_manifest.json")
    out = []
    for scope, split_path in partitions():
        split = json.loads(split_path.read_text())
        train = list(split["train_cases"])
        if len(train) != int(split["expected_counts"]["train"]):
            raise RuntimeError(f"{scope}: train count mismatch")
        stale = sorted(VOLVIEW.glob("volume_stats_train*.json"))
        if stale:
            raise RuntimeError(f"stale unrenamed statistics in {VOLVIEW}: {stale}")
        tmp = write_volume_stats(train, VOLVIEW, split_path)   # writes VOLVIEW/volume_stats_train{n}.json
        payload = json.loads(tmp.read_text())
        tmp.unlink()
        payload.update(scope=scope, renamed_from=tmp.name, volview_manifest_sha256=manifest_sha,
                       note="write_volume_stats names the file by train count (cv5 fold2 and fold3 are both 211); "
                            "renamed per partition by prepare_pf6vf6_v52d_retrain")
        dst = stats_path(scope, len(train))
        if dst.is_file():
            old = json.loads(dst.read_text())
            keep = ("linear", "velocity", "speed", "pressure_rel_interior", "pressure_rel_wall", "train_units", "split_sha256")
            if any(old.get(k) != payload.get(k) for k in keep):
                raise FileExistsError(f"refusing to change existing statistics: {dst}")
            print(f"[stats] {scope}: unchanged {dst.name}", flush=True)
        else:
            write_json(dst, payload)
        out.append({"scope": scope, "path": str(dst), "n_cases": payload["n_cases"],
                    "n_interior": payload["n_interior_points"], "n_wall": payload["n_wall_points"],
                    "pressure_linear": payload["linear"], "velocity": payload["velocity"], "speed": payload["speed"]})
        print(f"[stats] {scope}: n={payload['n_cases']} p {payload['linear']['mean']:.2f}±{payload['linear']['std']:.2f} Pa "
              f"speed {payload['speed']['mean']:.4f}±{payload['speed']['std']:.4f}", flush=True)
    ref = json.loads((RELEASE / "models/PF6_s1234/wss_global_stats.json").read_text())
    write_json(PREP / "stats_report.json", {"generated_at": now(), "context": run_context(), "partitions": out,
                                            "v50_train138_reference": {k: ref[k] for k in ("linear", "velocity", "speed")}})
    return 0


# -------------------------------------------------------------------------------------------------------------- featstats
def feature_stats_path(prefix: str, scope: str) -> Path:
    return EXP / "feature_stats" / f"{prefix}_{scope}_train.json"


def _featstats_one(job) -> dict:
    from training_wss_min import dataset as D
    prefix, scope = job
    try:
        raw = derive(prefix, SEEDS[0], scope, with_feature_stats=False)
        cfg = C.ExpConfig.from_dict(raw)
        stats = D.load_wss_stats(cfg.data.wss_stats_path)
        started = time.time()
        cases = D.load_partition(cfg.data.split_path, "train", stats, strict=True, target=cfg.data.target,
                                 target_normalization=cfg.data.target_normalization, data_root=cfg.data.data_root,
                                 required_frame_version=cfg.data.required_frame_version,
                                 case_features_path=cfg.data.case_features_path, timesteps=cfg.data.timesteps,
                                 waveform_path=cfg.data.waveform_path, extra_point_features=C.v6_point_features(cfg),
                                 point_features_root=cfg.data.point_features_root)
        expected = json.loads(Path(cfg.data.split_path).read_text())["counts"]["train"]
        if len(cases) != expected:
            raise RuntimeError(f"{prefix} {scope}: expected {expected} train cases, loaded {len(cases)}")
        out = D.compute_feature_stats(cases, tuple(cfg.data.input_features), cfg.data.curvature_transform)
        n_rows = int(sum(len(c["pos"]) for c in cases))
        out["_provenance"] = {"train_split": cfg.data.split_path, "split_sha256": sha256(cfg.data.split_path),
                              "n_cases": len(cases), "n_rows": n_rows, "data_root": cfg.data.data_root,
                              "volume_stats": cfg.data.wss_stats_path, "target": cfg.data.target,
                              "point_features_root": cfg.data.point_features_root,
                              "scope": "volume point set (wall nodes + interior cells) of the training partition",
                              "builder": "dataset.load_partition + dataset.compute_feature_stats (same call as train.py)",
                              "generated_at": now(), "context": run_context(), "load_seconds": time.time() - started}
        dst = feature_stats_path(prefix, scope)
        if dst.is_file():
            old = json.loads(dst.read_text())
            if {k: v for k, v in old.items() if k != "_provenance"} != {k: v for k, v in out.items() if k != "_provenance"}:
                raise FileExistsError(f"refusing to change existing feature statistics: {dst}")
        else:
            write_json(dst, out)
        return {"prefix": prefix, "scope": scope, "path": str(dst), "n_cases": len(cases), "n_rows": n_rows,
                "seconds": time.time() - started}
    except Exception as exc:  # noqa: BLE001
        import traceback
        return {"prefix": prefix, "scope": scope, "error": f"{type(exc).__name__}: {exc}", "trace": traceback.format_exc()}


def cmd_featstats(args) -> int:
    jobs = [(p, s) for s, _ in partitions() for p in TARGETS]
    with mp.get_context("fork").Pool(max(1, args.workers)) as pool:
        results = list(pool.imap_unordered(_featstats_one, jobs))
    errors = [r for r in results if "error" in r]
    for r in results:
        print("[featstats]", r.get("error") or f"{r['prefix']} {r['scope']} n={r['n_cases']} rows={r['n_rows']} "
              f"{r['seconds']:.0f}s", flush=True)
    same = {}
    if not errors:
        for scope, _ in partitions():
            a, b = (json.loads(feature_stats_path(p, scope).read_text()) for p in TARGETS)
            same[scope] = {k: v for k, v in a.items() if k != "_provenance"} == {k: v for k, v in b.items() if k != "_provenance"}
    write_json(PREP / "featstats_report.json", {"generated_at": now(), "context": run_context(), "results": results,
                                                "pf6_equals_vf6": same})
    print("[featstats] PF6 == VF6 per partition:", same, flush=True)
    return 1 if errors else 0


# ---------------------------------------------------------------------------------------------------------------- configs
def scope_split(scope: str) -> Path:
    return FULL_SPLIT if scope == "full265" else CV5 / f"fold{scope[-1]}.json"


def scope_train_count(scope: str) -> int:
    return int(json.loads(scope_split(scope).read_text())["expected_counts"]["train"])


def arm_id(prefix: str, seed: int, scope: str) -> str:
    return f"{prefix}_full265_s{seed}" if scope == "full265" else f"{prefix}_v52cv_f{scope[-1]}_s{seed}"


def derive(prefix: str, seed: int, scope: str, *, with_feature_stats: bool = True) -> dict:
    src_path = SRC[(prefix, seed)]
    src = json.loads(src_path.read_text())
    if src["data"]["target"] != TARGETS[prefix] or int(src["train"]["seed"]) != seed:
        raise ValueError(f"unexpected source recipe {src_path}")
    cfg = json.loads(json.dumps(src))
    aid = arm_id(prefix, seed, scope)
    n = scope_train_count(scope)
    d = cfg["data"]
    d["data_root"] = str(VOLVIEW)
    d["split_path"] = str(scope_split(scope))
    d["wss_stats_path"] = str(stats_path(scope, n))
    d["point_features_root"] = [str(FLOWREF)]
    d["feature_stats_path"] = str(feature_stats_path(prefix, scope)) if with_feature_stats else None
    cfg["name"] = f"{NAME}/{aid}"
    protocol = ("full265: train 265 units, test = recover8 (8 units)" if scope == "full265" else
                f"CV5 fold {scope[-1]}: patient-grouped 5-fold CV over 261 units, train {n}, test = the held-out fold")
    cfg["notes"] = (f"2026-10-02 PF6/VF6 v5.2d retrain (01 tracking §43) {aid}: source recipe {src_path.parent.name}/{src_path.name} "
                    f"verbatim (paired init reference unchanged); data only: v5.2d volume root wss_min_volview_v1, {protocol}, "
                    f"volume statistics (linear z) and input-feature statistics of this training partition, flowref v5.2d "
                    f"(log_q_branch_murray / log_tau0_murray). Source notes: {src.get('notes', '')}")
    return cfg


def check_config(aid: str, cfg: dict, src: dict) -> list[str]:
    from training_wss_min.tools import prepare_wss_local_wave1 as W1
    classes = {"data": C.DataConfig, "model": C.ModelConfig, "train": C.TrainConfig, "eval": C.EvalConfig}
    for section, cls in classes.items():
        unknown = set(cfg[section]) - set(cls.__dataclass_fields__)
        if unknown:
            raise ValueError(f"{aid}: unknown {section} fields {unknown}")
    parsed = C.ExpConfig.from_dict(cfg)
    C.validate_features(parsed)
    for key in ("wss_stats_path", "feature_stats_path", "split_path"):
        if not Path(cfg["data"][key]).is_file():
            raise FileNotFoundError(f"{aid}: {key} missing: {cfg['data'][key]}")
    if not Path(cfg["train"]["init_reference_config"]).is_file():
        raise FileNotFoundError(f"{aid}: init reference missing")
    diff = sorted(W1.declared_diff(src, cfg)) + [k for k in ("name", "notes") if src.get(k) != cfg.get(k)]
    extra = set(diff) - ALLOWED_DIFF
    if extra:
        raise ValueError(f"{aid}: fields outside the data/name/notes set changed: {sorted(extra)}")
    return sorted(diff)


def anchor_split() -> Path:
    """recover8 without its ILO 'after' unit: the old code (GNN_voltime_frozen_20260919) only accepts ILO 'before' labels."""
    full = json.loads(FULL_SPLIT.read_text())
    test = [u for u in full["test_cases"] if not (u.startswith("ILO/") and u.endswith("/after"))]
    path = EXP / "anchor" / f"split_anchor_recover{len(test)}_old_code_compatible.json"
    save_frozen(path, {"schema_version": 1, "split_version": path.stem, "source_split": str(FULL_SPLIT),
                       "source_split_sha256": sha256(FULL_SPLIT), "train_cases": [], "val_cases": [], "test_cases": test,
                       "counts": {"train": 0, "val": 0, "test": len(test)},
                       "expected_counts": {"train": 0, "val": 0, "test": len(test)},
                       "note": "preflight anchors only: recover8 minus ILO 'after' units, because the old code's split loader "
                               "(GNN_voltime_frozen_20260919, canonical_unit_id) rejects ILO 'after' labels; E5 uses full recover8"})
    return path


def e5_and_anchor_dirs() -> dict:
    """Deployed v5.0 PF6/VF6 (train138) repointed to the v5.2d volume root for recover8 (E5) + the preflight anchors."""
    made = {"e5": {}, "anchor": {}}
    anchor_split_path = anchor_split()
    for prefix in TARGETS:
        for seed in SEEDS:
            src = RELEASE / "models" / f"{prefix}_s{seed}"
            dests = [("e5", EXP / "e5_deployed_v50" / f"{prefix}_s{seed}")]
            if seed == 1234:
                dests.append(("anchor", EXP / "anchor" / f"{prefix}_s1234_v50_on_v52d"))
            for kind, out in dests:
                cfg = json.loads((src / "config.json").read_text())
                new = json.loads(json.dumps(cfg))
                d = new["data"]
                d["data_root"] = str(VOLVIEW)
                d["split_path"] = str(FULL_SPLIT if kind == "e5" else anchor_split_path)
                d["point_features_root"] = [str(FLOWREF)]
                d["wss_stats_path"] = str(out / "wss_global_stats.json")
                d["feature_stats_path"] = str(out / "feature_stats.json")
                split_note = ("split -> full265 (test = recover8)" if kind == "e5" else
                              f"split -> {anchor_split_path.name} (recover8 minus ILO 'after': the old code rejects 'after' labels)")
                new["notes"] = (str(cfg.get("notes", "")) + f" | 2026-10-02 {kind.upper()} read-only copy of the deployed release "
                                f"PF6_VF6_peak_3seed_20260920 (v5.0, train138): data root -> v5.2d wss_min_volview_v1, {split_note}, "
                                "flowref -> v5.2d; weights and statistics unchanged").strip(" |")
                save_frozen(out / "config.json", new)
                for fname in ("ckpt_best.pt", "feature_stats.json", "wss_global_stats.json", "target_normalization.json"):
                    dst = out / fname
                    if dst.is_file():
                        if sha256(dst) != sha256(src / fname):
                            raise FileExistsError(f"refusing to change {dst}")
                    else:
                        shutil.copy2(src / fname, dst)
                save_frozen(out / "SOURCE.json", {"release": str(RELEASE), "model": f"{prefix}_s{seed}",
                                                  "checkpoint_sha256": sha256(src / "ckpt_best.pt"), "kind": kind})
                made[kind][f"{prefix}_s{seed}"] = str(out)
    return made


def cmd_configs(args) -> int:
    from training_wss_min.tools import prepare_wss_local_wave1 as W1
    CONFIGS.mkdir(parents=True, exist_ok=True)
    arms = []
    order = [("full265", p, s) for p in TARGETS for s in SEEDS] + \
            [(f"cv5_fold{k}", p, s) for s in SEEDS for k in FOLDS for p in TARGETS]
    for scope, prefix, seed in order:
        aid = arm_id(prefix, seed, scope)
        cfg = derive(prefix, seed, scope)
        src = json.loads(SRC[(prefix, seed)].read_text())
        diff = check_config(aid, cfg, src)
        save_frozen(CONFIGS / f"{aid}.json", cfg)
        group = ("B-" if scope == "full265" else "A-") + prefix[0]
        arms.append(dict(id=aid, group=group, title=f"{prefix} {'full265' if scope == 'full265' else 'CV5 fold ' + scope[-1]} seed {seed} on v5.2d",
                         config=f"{aid}.json", run_name=cfg["name"], phase=0, seed=seed,
                         fold=None if scope == "full265" else int(scope[-1]), protocol="FULL265" if scope == "full265" else "CV5",
                         target=TARGETS[prefix], arm=prefix, depends_on=[], evaluate=["best", "last"], single_change=False,
                         evaluation="recover8" if scope == "full265" else "held-out CV5 fold",
                         source_config=str(SRC[(prefix, seed)]), source_config_sha256=sha256(SRC[(prefix, seed)]),
                         config_diff_vs_source=diff,
                         volume_stats=cfg["data"]["wss_stats_path"], feature_stats=cfg["data"]["feature_stats_path"]))
    made = e5_and_anchor_dirs()
    matrix = dict(
        schema_version=1, experiment=NAME,
        section="01 block tracking §43: PF6 (pressure) / VF6 (velocity) on v5.2d, recipes verbatim, data only; "
                "CV5 5 folds x 3 seeds per target + full265 x 3 seeds per target (recover8); confirmed by the user 2026-10-02",
        data={"volume_root": str(VOLVIEW), "volview_manifest": str(VOLVIEW / "volview_manifest.json"),
              "volview_manifest_sha256": sha256(VOLVIEW / "volview_manifest.json"), "main_view_root": str(VIEW),
              "flowref_root": str(FLOWREF), "snapshot_root": str(SNAP), "full265_split": str(FULL_SPLIT),
              "full265_split_sha256": sha256(FULL_SPLIT), "cv5_dir": str(CV5),
              "volume_stats": {s: str(stats_path(s, scope_train_count(s))) for s, _ in partitions()},
              "feature_stats": {f"{p}_{s}": str(feature_stats_path(p, s)) for s, _ in partitions() for p in TARGETS}},
        recipe_sources={f"{p}_s{s}": str(SRC[(p, s)]) for p in TARGETS for s in SEEDS},
        allowed_config_diff=sorted(ALLOWED_DIFF),
        execution=f"frozen code copy {FROZEN}; master 4x RTX 4090 (Slurm queue, one arm per GPU) + node04 2x A100 (outside "
                  "Slurm, run_local_train_queue --only); each arm train -> eval best -> eval last with --save-predictions",
        expected_training_runs=len(arms), expected_evaluations=2 * len(arms), arms=arms,
        not_scheduled={"*_murray_cap prior arms": "user 2026-10-02: not in this batch",
                       "IND 170/91": "not requested"},
        known_data_limits=KNOWN_LIMITS,
        preflight_anchors=made["anchor"],
        post_queue={"E3": "the 30 CV5 fold models (best) on recover8 via --split-path full265 -> experiments/<exp>/recover8_cv5/<arm>/",
                    "E5": {"what": "deployed PF6_VF6_peak_3seed_20260920 (v5.0, train138) on recover8 -> "
                                   "experiments/<exp>/e5_deployed_v50/<model>/eval", "run_dirs": made["e5"]}},
        reading_rule=("Baseline on new data, no gates. CV5: per seed the pooled out-of-fold case-balanced R2 over the 261 units, "
                      "three-seed mean +- sd, three-seed out-of-fold ensemble (mean prediction per point; velocity = mean "
                      "vector, speed of the mean); pressure also wall / interior, velocity speed R2_cb plus vector error; per "
                      "cohort. recover8 (n = 8) descriptive only: full265 per seed + ensemble, E3 per-seed five-fold and "
                      "15-model ensembles, E5 deployed v5.0 per seed + ensemble; full265 vs E5 mixes training size (265 vs "
                      "138) and data corrections, not attributed. v5.0 test34 and v5.1 cv3 numbers are other evaluation sets: "
                      "listed side by side, never subtracted."))
    save_frozen(CONFIGS / "matrix.json", matrix)
    if args.node04_arms:
        chosen = [a for a in args.node04_arms.split(",") if a]
        unknown = set(chosen) - {a["id"] for a in arms}
        if unknown:
            raise ValueError(f"unknown node04 arms {unknown}")
        for host, members in (("master", [a for a in arms if a["id"] not in chosen]),
                              ("node04", [a for a in arms if a["id"] in chosen])):
            sub = dict(matrix, host=host, expected_training_runs=len(members), expected_evaluations=2 * len(members),
                       arms=members, parent_matrix="matrix.json")
            save_frozen(CONFIGS / f"matrix_{host}.json", sub)
            print(f"matrix_{host}.json: {len(members)} arms", flush=True)
    print(CONFIGS / "matrix.json", len(arms), "arms", flush=True)
    return 0


# ---------------------------------------------------------------------------------------------------------------- cleanup
def cmd_cleanup(args) -> int:
    manifest = json.loads((VOLVIEW / "volview_manifest.json").read_text())
    removed = 0
    for m in manifest["units"]:
        staged = STAGE / m["unit"] / "volume.npz"
        if staged.is_file():
            if m["volume_source"] not in ("link_main_view", "link_joint_cycle_cache"):
                raise RuntimeError(f"{staged} should have been moved into the new root")
            staged.unlink()
            removed += 1
        link = STAGE / m["unit"] / "bundle.npz"
        if link.is_symlink():
            link.unlink()
    for d in sorted((p for p in STAGE.rglob("*") if p.is_dir()), key=lambda p: -len(p.parts)):
        if not any(d.iterdir()):
            d.rmdir()
    left = [p for p in STAGE.rglob("*") if p.is_file()] if STAGE.exists() else []
    if STAGE.exists() and not left:
        shutil.rmtree(STAGE)
    print(f"[cleanup] removed {removed} staging copies; staging {'removed' if not STAGE.exists() else 'kept: ' + str(left[:5])}")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("command", choices=("build", "compare", "assemble", "stats", "featstats", "configs", "cleanup"))
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--accept-mismatch-replacement", action="store_true")
    ap.add_argument("--node04-arms", default="", help="configs: comma list of arm ids run on node04 (writes matrix_{master,node04}.json)")
    args = ap.parse_args()
    PREP.mkdir(parents=True, exist_ok=True)
    return {"build": cmd_build, "compare": cmd_compare, "assemble": cmd_assemble, "stats": cmd_stats,
            "featstats": cmd_featstats, "configs": cmd_configs, "cleanup": cmd_cleanup}[args.command](args)


if __name__ == "__main__":
    raise SystemExit(main())

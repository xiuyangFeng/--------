"""Make v5.2c the single, self-contained data version and remove the superseded copies (2026-09-30; user: one data
foundation, same 口径 everywhere, old-version data removed; docs/02-推进与变更/04-数据处理与CFD/数据统一与旧版本清理_2026-09-30.md).

    python consolidate_v52c.py plan            # read-only: what moves where, what is left for deletion, sizes, checks
    python consolidate_v52c.py move            # links -> real directories (mv on the same filesystem; nothing is copied)
    python consolidate_v52c.py purge --yes     # delete the superseded roots / caches (only after `move` and the checks)

move
  snapshot  data_wss_v5/anatomy_pointcloud_v5_2c_20260930/cases/<unit>: every link replaced by the real case directory
            it points at (the h5 each current view was built from)
  views     data_wss_v5/views_v5_2c_20260930/<pack>/<unit>: every link replaced by its real directory; view manifest and
            per-case view_report.json `source.case_h5` rewritten to the unified snapshot; other manifests' root strings
            rewritten; split definitions of the old roots kept under wss_min_view_v1/legacy_splits/ (definitions only)
  joint     training_wss_min/experiments/joint_cycle_v52c_20260930/data_audit_cache: linked unit directories made real,
            source paths in metadata / audit / provenance rewritten to the unified roots
purge     the roots and caches listed in PURGE (their current content was moved out by `move`; what is left are the old
          label / old geometry copies and closed-line packs); refuses if any link in the unified roots or the joint cache
          still points into them.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import time
from pathlib import Path

G = Path("/public/newhome/cy/Digital_twin/GNN")
D = G / "data_wss_v5"
EG = G.parent / "GNN_recover_eval_20260929"
C = D / "views_v5_2c_20260930"
S = D / "anatomy_pointcloud_v5_2c_20260930"
JC = G / "training_wss_min/experiments/joint_cycle_v52c_20260930/data_audit_cache"
HERE = Path(__file__).resolve().parent
LOG = HERE / "consolidate_log.json"
SYN3 = list(json.loads((G / "outputs/cfd_auto_trial_20260927/_label_fix_syn/batch.json").read_text())["units"])
SNAP_FIX = D / "anatomy_pointcloud_v5_2c_labelfix_20260930"
OLD_VIEW_ROOTS = [D / "views_v5_1", D / "views_v5_2_20260922", D / "views_v5_2_full_20260923", D / "views_v5_2_pilot_20260922",
                  D / "views_v5_2c_labelfix9_20260930", D / "views_v5_2p4_full265_20260930", D / "views_v5_2p5_full265_20260930",
                  D / "density_v5_2p4_new4_20260930", EG / "data_wss_v5/views_v5_2_recover_20260929", EG / "data_wss_v5/views_v5_2_yang_20260930"]
OLD_SNAP_ROOTS = [D / "anatomy_pointcloud_v5_1_20260916", D / "anatomy_pointcloud_v5_2_20260922", D / "anatomy_pointcloud_v5_2_pilot_20260922",
                  SNAP_FIX, EG / "data_wss_v5/anatomy_pointcloud_v5_2_recover_20260929", EG / "data_wss_v5/anatomy_pointcloud_v5_2_yang_20260930"]
OLD_JOINT = G / "training_wss_min/experiments/joint_cycle_v52_20260929/data_audit_cache"
PURGE = OLD_VIEW_ROOTS + OLD_SNAP_ROOTS + [
    OLD_JOINT,
    G / "training_wss_min/experiments/wss_time_adapter_v51_20260923/cache",            # v5.1 WSS time caches (old labels)
    G / "training_wss_min/experiments/vf6_velocity_to_wss_20260916/chunks",            # v5.1 CFD velocity extractions
    G / "training_wss_min/experiments/vf6_velocity_to_wss_20260916/velocity",
    G / "training_wss_min/experiments/volume_time_20260919/offline",                   # v5.1 volume time caches
    G / "training_wss_min/experiments/v5_velocity_to_wss_20260909/chunks",             # v5.0 CFD velocity extractions
    G / "training_wss_min/experiments/v5_velocity_to_wss_20260909/velocity_best",
    G / "training_wss_min/experiments/velocity_phase_v52_20260930/geometry",           # sidecars + q80 stats of the old joint cache
    G / "outputs/cfd_auto_trial_20260927/_recover/units_variants",                     # recover study variants (WANG_CAI dt2 moved to the library first)
    G / "outputs/cfd_auto_trial_20260927/_recover/units_origbc",
    EG,                                                                                # isolated recover root (data moved out; PREP merged into the main PREP)
    G / "outputs/centerline_v2_full_173_20260828/cases/AAA/unruputer/LIU_WEN_QI",     # centreline of the old ZHU_ZI_HAI-copy mesh
    G / "outputs/centerline_v2_full_173_20260828/cases/AAA/ruputer/YANG_BAO_KUI",      # centreline of the overwritten v4 mesh
]
PREP = G / "outputs/wss_pinn/volume_uvwp_bc_rcr_v4_anatomy_prep_20260903"
RECOVER12 = ["AAA/unruputer/LIU_WEN_QI", "AAA/ruputer/FU_GUO_JUN", "AAA/ruputer/SHI_YUN_XI", "AAA/ruputer/LIU_YU_MING", "AAA/ruputer/WANG_SHUN_WEN",
             "AAA/ruputer/ZHANG_ZAO_SHUAN", "ILO/XUE_YOU_TANG-0/after", "ILO/LI_JIE-1/after", "ILO/LI_JIE-1/before", "ILO/WANG_CAI-0/before",
             "ILO/LI_FA_XIANG-1/before", "AAA/ruputer/YANG_BAO_KUI"]
PACKS = ["wss_min_view_v1", "wss_min_geom_v2", "wss_min_flowref_v1", "wss_min_cycle_v1", "wss_min_phys1d_v1",
         "wss_min_density_v1/L70", "wss_min_density_v1/L50", "wss_min_density_v1/L35", "wss_min_density_v1/L25"]


def du(paths) -> int:
    paths = [str(p) for p in paths if Path(p).exists()]
    tot = 0
    for i in range(0, len(paths), 400):
        out = subprocess.run(["du", "-scb", *paths[i:i + 400]], capture_output=True, text=True).stdout.strip().splitlines()
        tot += int(out[-1].split()[0]) if out else 0
    return tot


def unit_entries(pack_root: Path) -> list[Path]:
    out = []
    for coh in ("AAA", "AG", "ILO"):
        if not (pack_root / coh).is_dir():
            continue
        for sub in sorted((pack_root / coh).iterdir()):
            if sub.is_dir():
                out += [k for k in sorted(sub.iterdir()) if k.is_dir() or k.is_symlink()]
    return out


def links_into(roots: list[Path], old: list[Path]) -> list[str]:
    bad = []
    olds = [str(o) + "/" for o in old] + [str(o.resolve()) + "/" for o in old if o.exists()]
    for r in roots:
        for dirpath, dirnames, filenames in os.walk(r):
            for n in dirnames + filenames:
                p = Path(dirpath) / n
                if p.is_symlink():
                    t = os.readlink(p)
                    t = t if os.path.isabs(t) else str((p.parent / t))
                    if any(t.startswith(o) or t + "/" == o for o in olds):
                        bad.append(f"{p} -> {t}")
    return bad


def checks() -> list[str]:
    problems = []
    for cid in SYN3:
        link = S / "cases" / cid.replace("/", "__")
        if link.is_symlink() and not str(link.resolve()).startswith(str(SNAP_FIX.resolve())):
            problems.append(f"{cid}: synthetic rerun not rebuilt yet (snapshot still -> {link.resolve()})")
        v = C / "wss_min_view_v1" / cid
        if v.is_symlink() and "labelfix9" not in str(v.resolve()):
            problems.append(f"{cid}: view still the old one ({v.resolve()})")
    for pack in PACKS:
        for e in unit_entries(C / pack):
            if e.is_symlink() and not e.exists():
                problems.append(f"dangling {e}")
    return problems


def cmd_plan(_a) -> None:
    moves = {"snapshot": [], "views": [], "joint": []}
    for e in sorted((S / "cases").iterdir()):
        if e.is_symlink():
            moves["snapshot"].append((str(e), str(e.resolve())))
    for pack in PACKS:
        for e in unit_entries(C / pack):
            if e.is_symlink():
                moves["views"].append((str(e), str(e.resolve())))
    if JC.exists():
        for sub in ("cases", "derived_volume"):
            for e in unit_entries(JC / sub):
                if e.is_symlink():
                    moves["joint"].append((str(e), str(e.resolve())))
    moving = {Path(t) for kind in moves.values() for _, t in kind}
    rem = []
    for r in PURGE:
        if not r.exists():
            rem.append({"root": str(r), "exists": False}); continue
        inside = [m for m in moving if str(m).startswith(str(r.resolve()) + "/")]
        total = du([r]); moved = du(inside)
        rem.append({"root": str(r), "total_gb": round(total / 1e9, 2), "moving_out_gb": round(moved / 1e9, 2), "deleted_gb": round((total - moved) / 1e9, 2)})
    out = {"moves": {k: len(v) for k, v in moves.items()}, "purge": rem, "purge_total_gb": round(sum(x.get("deleted_gb", 0) for x in rem), 1),
           "checks": checks()}
    (HERE / "consolidate_plan.json").write_text(json.dumps({"summary": out, "moves": moves}, indent=1))
    print(json.dumps(out, indent=1))


def rewrite_text(p: Path, mapping: dict[str, str]) -> bool:
    t = p.read_text()
    n = t
    for a, b in mapping.items():
        n = n.replace(a, b)
    if n != t:
        p.write_text(n)
        return True
    return False


def cmd_move(_a) -> None:
    prob = checks()
    if prob:
        raise SystemExit("checks failed:\n" + "\n".join(prob))
    log = {"started": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "snapshot": [], "views": [], "joint": [], "rewritten": []}
    # snapshot: real case dirs into S
    snap_map = {}
    for e in sorted((S / "cases").iterdir()):
        if e.is_symlink():
            real = e.resolve(); e.unlink(); shutil.move(str(real), str(e))
            log["snapshot"].append([str(real), str(e)]); snap_map[str(real)] = str(e)
    # views: real dirs into C
    for pack in PACKS:
        for e in unit_entries(C / pack):
            if e.is_symlink():
                real = e.resolve(); e.unlink(); shutil.move(str(real), str(e))
                log["views"].append([str(real), str(e)])
    # view reports: source.case_h5 -> unified snapshot
    for rep in list((C / "wss_min_view_v1").glob("*/*/*/view_report.json")) + list((C / "wss_min_view_v1").glob("ILO/*/*/*/view_report.json")):
        d = json.loads(rep.read_text()); h5 = d["source"]["case_h5"]; parent = str(Path(h5).parent)
        if parent in snap_map:
            d["source"]["case_h5_built_from"] = h5
            d["source"]["case_h5"] = str(Path(snap_map[parent]) / "case.h5")
            rep.write_text(json.dumps(d, indent=1, ensure_ascii=False))
    vm = C / "wss_min_view_v1/view_manifest.json"
    m = json.loads(vm.read_text())
    for r in m["reports"]:
        h5 = r["source"]["case_h5"]; parent = str(Path(h5).parent)
        if parent in snap_map:
            r["source"]["case_h5_built_from"] = h5; r["source"]["case_h5"] = str(Path(snap_map[parent]) / "case.h5")
    m["snapshot_root"] = str(S)
    vm.write_text(json.dumps(m, indent=1, ensure_ascii=False))
    # other manifests: provenance root strings -> unified roots
    roots = {str(r): str(C) for r in OLD_VIEW_ROOTS} | {str(r): str(S) for r in OLD_SNAP_ROOTS}
    for mf in list(C.glob("*/*manifest*.json")) + [C / "assembly_plan.json"]:
        if mf.name != "view_manifest.json" and rewrite_text(mf, roots):
            log["rewritten"].append(str(mf))
    plan = json.loads((C / "assembly_plan.json").read_text())
    plan["unified_snapshot_root"] = str(S)
    plan["unified_note"] = "2026-09-30: every unit's case.h5 is under unified_snapshot_root/cases; plan tags record where each unit was built"
    (C / "assembly_plan.json").write_text(json.dumps(plan, indent=1, ensure_ascii=False))
    # split definitions of the old roots (protocol definitions only; statistics are recomputed on v5.2c when needed)
    legacy = C / "wss_min_view_v1/legacy_splits"; legacy.mkdir(exist_ok=True)
    for r in OLD_VIEW_ROOTS:
        for f in list(r.glob("wss_min_view_v1/split*.json")) + list(r.glob("wss_min_view_v1/cv3*/*.json")) + list(r.glob("wss_min_view_v1/*/fold*.json")):
            if f.is_file() and not f.is_symlink():
                dst = legacy / r.name / f.relative_to(r / "wss_min_view_v1")
                if not dst.exists():
                    dst.parent.mkdir(parents=True, exist_ok=True); shutil.copy2(f, dst)
    # joint cache: linked units made real, source paths rewritten
    if JC.exists():
        for sub in ("cases", "derived_volume"):
            for e in unit_entries(JC / sub):
                if e.is_symlink():
                    real = e.resolve(); e.unlink(); shutil.move(str(real), str(e))
                    log["joint"].append([str(real), str(e)])
        jmap = {**{a: b for a, b in snap_map.items()}, **roots, str(OLD_JOINT): str(JC)}
        for f in list(JC.rglob("metadata.json")) + list(JC.rglob("source.json")) + [JC / "audit.json", JC / "manifest.json"]:
            if f.exists() and rewrite_text(f, jmap):
                log["rewritten"].append(str(f))
    log["finished"] = time.strftime("%Y-%m-%dT%H:%M:%S%z")
    LOG.write_text(json.dumps(log, indent=1))
    print({k: len(v) for k, v in log.items() if isinstance(v, list)})


def cmd_purge(a) -> None:
    if not a.yes:
        raise SystemExit("purge deletes data; pass --yes")
    if not LOG.exists():
        raise SystemExit("run `move` first")
    pre = []
    if not (G / "data_new/ILO/WANG_CAI-0/before/RECOVER_SWAP_20260930.json").exists():
        pre.append("recover units not swapped into the library yet (WANG_CAI dt2 run lives in units_variants)")
    for cid in RECOVER12:
        if not (PREP / "audits/topology/cases" / (cid.replace("/", "__") + ".json")).exists() or not (PREP / "atlas/cases" / cid).exists():
            pre.append(f"{cid}: main PREP entry missing (merge from the isolated root first)")
    if pre:
        raise SystemExit("preconditions:\n" + "\n".join(pre))
    dang = links_into([C, S] + ([JC] if JC.exists() else []), PURGE)
    if dang:
        raise SystemExit("links into purge roots remain:\n" + "\n".join(dang[:20]))
    out = {"date": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "deleted": []}
    for r in PURGE:
        if r.exists():
            n = du([r]); shutil.rmtree(r)
            out["deleted"].append({"path": str(r), "bytes": n}); print(f"deleted {r} ({n / 1e9:.1f} GB)", flush=True)
    out["bytes_deleted"] = sum(x["bytes"] for x in out["deleted"])
    (HERE / "purge_log.json").write_text(json.dumps(out, indent=1))
    print(f"total {out['bytes_deleted'] / 1e9:.1f} GB")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["plan", "move", "purge"])
    ap.add_argument("--yes", action="store_true")
    a = ap.parse_args()
    {"plan": cmd_plan, "move": cmd_move, "purge": cmd_purge}[a.cmd](a)


if __name__ == "__main__":
    main()

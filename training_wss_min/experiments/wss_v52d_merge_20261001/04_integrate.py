"""Step 4 of the 2026-10-01 merge: v5.2c -> v5.2d.

    python 04_integrate.py rename     # the two unified roots get their v5.2d names; root strings rewritten in every json
    python 04_integrate.py units g1 [g2 ...]   # staged units (data_wss_v5/_merge_20261001/{snapshot,views_<group>}) moved in

rename   data_wss_v5/anatomy_pointcloud_v5_2c_20260930 -> anatomy_pointcloud_v5_2d_20261001
         data_wss_v5/views_v5_2c_20260930           -> views_v5_2d_20261001
units    per staged unit and pack (snapshot case, view, geom_v2, cycle, density levels): the current directory goes to
         data_wss_v5/_replaced_v52c_20261001/ (kept until the acceptance checks pass), the staged directory takes its place,
         staging path strings in its json files are rewritten to the final roots; pack manifests get the staged entries.
         flowref / phys1d are rebuilt for every unit afterwards (05), so they are not touched here.
"""
import json, os, shutil, sys, time
from pathlib import Path

G = Path("/public/newhome/cy/Digital_twin/GNN"); D = G / "data_wss_v5"
OLD_S, OLD_C = D / "anatomy_pointcloud_v5_2c_20260930", D / "views_v5_2c_20260930"
NEW_S, NEW_C = D / "anatomy_pointcloud_v5_2d_20261001", D / "views_v5_2d_20261001"
ST = D / "_merge_20261001"; REPL = D / "_replaced_v52c_20261001"
HERE = Path(__file__).resolve().parent; LOG = HERE / "integrate_log.json"
PACK_MANIFEST = {"wss_min_view_v1": ("view_manifest.json", "reports", "canonical_id"), "wss_min_geom_v2": ("geom_manifest.json", "reports", "canonical_id"),
                 "wss_min_cycle_v1": ("cycle_manifest.json", "cases", "case"), "wss_min_density_v1": ("density_manifest.json", "reports", "canonical_id")}


def log(entry):
    L = json.loads(LOG.read_text()) if LOG.exists() else []
    L.append({"t": time.strftime("%Y-%m-%dT%H:%M:%S%z"), **entry}); LOG.write_text(json.dumps(L, indent=1, ensure_ascii=False))


def rewrite_json(root: Path, pairs: list[tuple[str, str]]) -> int:
    n = 0
    for p in root.rglob("*.json"):
        if p.is_symlink(): continue
        t = p.read_text(encoding="utf-8", errors="surrogateescape"); u = t
        for a, b in pairs: u = u.replace(a, b)
        if u != t:
            p.write_text(u, encoding="utf-8", errors="surrogateescape"); n += 1
    return n


def cmd_rename():
    assert OLD_S.is_dir() and OLD_C.is_dir() and not NEW_S.exists() and not NEW_C.exists()
    os.rename(OLD_S, NEW_S); os.rename(OLD_C, NEW_C)
    pairs = [(OLD_S.name, NEW_S.name), (OLD_C.name, NEW_C.name)]
    n = rewrite_json(NEW_S, pairs) + rewrite_json(NEW_C, pairs)
    m = json.loads((NEW_S / "manifest.json").read_text())
    m.update(snapshot=NEW_S.name, data_version="v5.2d (2026-10-01, library audit merge of v5.2c)", predecessor=OLD_S.name)
    (NEW_S / "manifest.json").write_text(json.dumps(m, indent=1, ensure_ascii=False))
    ap = json.loads((NEW_C / "assembly_plan.json").read_text())
    ap["data_version"] = "v5.2d (2026-10-01 library audit merge of v5.2c)"; ap["predecessor_v52c"] = {"views": OLD_C.name, "snapshot": OLD_S.name}
    (NEW_C / "assembly_plan.json").write_text(json.dumps(ap, indent=1, ensure_ascii=False))
    log({"step": "rename", "json_files_rewritten": n}); print("renamed; json files rewritten:", n)


def swap_dir(cur: Path, staged: Path, backup: Path) -> str:
    if not staged.is_dir() and cur.is_dir() and backup.exists():
        return "already"                       # a previous (interrupted) call moved it
    assert staged.is_dir(), staged
    if cur.exists():
        backup.parent.mkdir(parents=True, exist_ok=True); assert not backup.exists(), backup
        shutil.move(str(cur), str(backup)); how = "replaced"
    else:
        cur.parent.mkdir(parents=True, exist_ok=True); how = "added"
    shutil.move(str(staged), str(cur))
    return how


def cmd_units(groups):
    assert NEW_S.is_dir() and NEW_C.is_dir(), "run `rename` first"
    for g in groups:
        VW = ST / f"views_{g}"
        units = [r["canonical_id"] for r in json.loads((VW / "wss_min_view_v1/view_manifest.json").read_text())["reports"]]
        pairs = [(str(VW), str(NEW_C)), (str(ST / "snapshot"), str(NEW_S))]
        rec = {"step": "units", "group": g, "units": {}}
        staged_manifests = {pack: json.loads((VW / pack / mf).read_text()) for pack, (mf, _, _) in PACK_MANIFEST.items() if (VW / pack / mf).exists()}
        for u in units:
            n = u.replace("/", "__"); r = {}
            r["snapshot"] = swap_dir(NEW_S / "cases" / n, ST / "snapshot/cases" / n, REPL / "snapshot" / n)
            rewrite_json(NEW_S / "cases" / n, pairs)
            for pack in ("wss_min_view_v1", "wss_min_geom_v2", "wss_min_cycle_v1"):
                r[pack] = swap_dir(NEW_C / pack / u, VW / pack / u, REPL / "views" / pack / u); rewrite_json(NEW_C / pack / u, pairs)
            for lv in ("L70", "L50", "L35", "L25"):
                if (VW / "wss_min_density_v1" / lv / u).is_dir() or (REPL / "views/wss_min_density_v1" / lv / u).exists():
                    r[f"density/{lv}"] = swap_dir(NEW_C / "wss_min_density_v1" / lv / u, VW / "wss_min_density_v1" / lv / u, REPL / "views/wss_min_density_v1" / lv / u)
            rec["units"][u] = r
        for pack, sm in staged_manifests.items():
            mf, key, idk = PACK_MANIFEST[pack]
            cur = json.loads((NEW_C / pack / mf).read_text())
            new = {e[idk]: e for e in sm[key]}
            cur[key] = [e for e in cur[key] if e[idk] not in new] + list(new.values())
            mg = cur.setdefault("merge_20261001", {"version": "v5.2d", "units": []})
            mg["units"] = sorted(set(mg["units"]) | set(new))
            t = json.dumps(cur, indent=1, ensure_ascii=False)
            for a, b in pairs: t = t.replace(a, b)
            (NEW_C / pack / mf).write_text(t)
        m = json.loads((NEW_S / "manifest.json").read_text())
        for u in units: m["cases"][u] = str(NEW_S / "cases" / u.replace("/", "__"))
        m["n_cases"] = len(m["cases"]); (NEW_S / "manifest.json").write_text(json.dumps(m, indent=1, ensure_ascii=False))
        log(rec); print(g, "integrated", len(units), "units")


if __name__ == "__main__":
    if sys.argv[1] == "rename": cmd_rename()
    elif sys.argv[1] == "units": cmd_units(sys.argv[2:])
    else: raise SystemExit(__doc__)

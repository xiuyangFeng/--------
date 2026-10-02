"""Step 1 of the 2026-10-01 merge: bring the two re-simulated units into the case library.

    python 01_swap_raw.py --dry-run | --apply | --purge-old

HAN_JIAN_FU (same mesh, own-mesh rerun) and WANG_TIAN_QING-1/after (rebuilt from its own STL). Per unit D = data_new/<unit>,
T = the cfd_auto work directory (orchestrator state must be 'checked'):
  archive  every CFD product of the old run -> data_new/_archive/merge_20261001/<unit>/ ; kept in place: the STL(s),
           fluent.slurm and, for the same-mesh unit only, centerline/ (the rebuilt unit's centerline belongs to the wrong mesh)
  bring in ascii/, ascii_in/, Global_conditions/ (moved), the case (T paths rewritten to D), the UDF, 2.jou (paths rewritten);
           small cfd_auto reports -> D/cfd_auto_provenance/ ; D/MERGE_20261001.json
purge    (user 2026-10-01: old data deleted, only the rebuilt data stays) delete the archived exports, cases, libudf;
         old UDFs, journals and gzipped transcripts stay as provenance.
"""
import argparse, gzip, hashlib, json, shutil, subprocess, time
from pathlib import Path

G = Path("/public/newhome/cy/Digital_twin/GNN"); L = G / "data_new"; A = L / "_archive/merge_20261001"
HERE = Path(__file__).resolve().parent
U = json.loads((HERE / "units.json").read_text())
SPECS = {**{u: {**s, "own_mesh": True} for u, s in U["recomputed_same_mesh"].items()}, **{u: {**s, "own_mesh": False} for u, s in U["rebuilt_from_stl"].items()}}
BRING = ("ascii", "ascii_in", "Global_conditions")
PROV = ("own_mesh_report.json", "protocol_bc.json", "sanity.json", "settings_diff.json", "setup.jou", "smoke.jou", "prepare_report.json", "MANIFEST.json", "frames.json",
        "naming.json", "naming.png", "preflight.json", "calibration.json", "mesh_gate.json", "mesh_attempts.json", "surface_report.json", "reference_profile.json", "RELABELLED.json", "logs")


def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 24), b""): h.update(b)
    return h.hexdigest()


def du(p):
    return int(subprocess.run(["du", "-sb", str(p)], capture_output=True, text=True).stdout.split()[0]) if Path(p).exists() else 0


def plan(unit):
    s = SPECS[unit]; D, T = L / unit, G / s["work"]
    st = json.loads((G / s["batch"] / "state" / (unit.replace("/", "__") + ".json")).read_text())
    assert st["state"] == "checked", (unit, st["state"], st.get("message"))
    for need in (*BRING, s["case"]):
        assert (T / need).exists(), f"{unit}: rerun output missing {need}"
    assert len(list((T / "ascii").iterdir())) == 81 and len(list((T / "ascii_in").iterdir())) == 81, unit
    udfs = sorted(T.glob("udf-inlet*.c")); assert len(udfs) == 1, (unit, udfs)
    keep = {"fluent.slurm"} | ({"centerline"} if s["own_mesh"] else set())
    archive = [p.name for p in sorted(D.iterdir()) if p.name not in keep and not p.name.lower().endswith(".stl")]
    return {"D": D, "T": T, "case": s["case"], "udf": udfs[0].name, "archive": archive, "keep": sorted(keep & {p.name for p in D.iterdir()}), "schedule": st.get("last_run", {}).get("schedule")}


def rewrite(data, src, dst):
    n = 0
    for s in {str(src), str(src.resolve())}:
        n += data.count(s.encode()); data = data.replace(s.encode(), str(dst).encode())
    return data, n


def apply(unit, p, log):
    D, T, dst = p["D"], p["T"], A / unit
    dst.mkdir(parents=True, exist_ok=False)
    rec = {"unit": unit, "source_run": str(T), "archived": [], "moved_in": [], "written": [], "kept": p["keep"]}
    for name in p["archive"]:
        shutil.move(str(D / name), str(dst / name)); rec["archived"].append(name)
    for name in BRING:
        shutil.move(str(T / name), str(D / name)); rec["moved_in"].append(name)
    shutil.copy2(T / p["udf"], D / p["udf"]); rec["written"].append(p["udf"])
    run_sha = sha(T / p["case"])
    data, n = rewrite(gzip.open(T / p["case"], "rb").read(), T, D)
    assert b"cfd_auto_trial_20260927" not in data, f"{unit}: run paths remain in the case"
    with gzip.open(D / p["case"], "wb", compresslevel=6) as fh: fh.write(data)
    rec["written"].append(f"{p['case']} ({n} path strings rewritten)")
    jou, nj = rewrite((T / "2.jou").read_bytes(), T, D); (D / "2.jou").write_bytes(jou); rec["written"].append(f"2.jou ({nj} path strings rewritten)")
    prov = D / "cfd_auto_provenance"; prov.mkdir(exist_ok=True)
    for name in PROV:
        if (T / name).is_file(): shutil.copy2(T / name, prov / name)
        elif (T / name).is_dir(): shutil.copytree(T / name, prov / name)
    s = SPECS[unit]
    (D / "MERGE_20261001.json").write_text(json.dumps({
        "what": f"{unit}: CFD replaced by a cfd_auto run ({'same mesh' if s['own_mesh'] else 'rebuilt from its own STL'}), 2026-10-01 library audit",
        "why": s["why"], "source_run": str(T), "schedule": p["schedule"], "case_sha256_run": run_sha, "case_sha256_library": sha(D / p["case"]),
        "note": "the library case differs from the run's case only by the rewritten run-directory path strings",
        "archive": str(dst), "date": log["date"], "docs": "docs/02-推进与变更/04-数据处理与CFD/母库全量审计_2026-10-01.md"}, ensure_ascii=False, indent=1))
    rec["written"].append("MERGE_20261001.json")
    (T / "MOVED_TO_DATA_NEW.txt").write_text(f"ascii/, ascii_in/, Global_conditions/ moved to {D} on {log['date']}; old library data archived in {dst}\n")
    log["units"].append(rec)


def purge():
    out = {"date": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "deleted": [], "gzipped": []}
    for unit in SPECS:
        dst = A / unit
        targets = [dst / "ascii", dst / "ascii_in", dst / "processed", dst / "export", dst / "centerline", *dst.glob("*.cas.gz"), *dst.glob("*.cas"), *dst.glob("libudf*"), *dst.glob("*.dat*")]
        for t in targets:
            if not t.exists(): continue
            n = du(t); files = sum(1 for _ in t.rglob("*")) if t.is_dir() else 1
            shutil.rmtree(t) if t.is_dir() else t.unlink()
            out["deleted"].append({"path": str(t.relative_to(A)), "bytes": n, "files": files})
        for t in list(dst.rglob("*.out")):
            with open(t, "rb") as src, gzip.open(str(t) + ".gz", "wb", compresslevel=6) as g: shutil.copyfileobj(src, g)
            out["gzipped"].append(str(t.relative_to(A))); t.unlink()
    out["bytes_deleted"] = sum(d["bytes"] for d in out["deleted"])
    (A / "PURGE_LOG.json").write_text(json.dumps(out, indent=1, ensure_ascii=False))
    print(f"purged {out['bytes_deleted'] / 1e9:.1f} GB in {len(out['deleted'])} items; gzipped {len(out['gzipped'])}; archive now {du(A) / 1e9:.3f} GB")


def main():
    ap = argparse.ArgumentParser(); g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--dry-run", action="store_true"); g.add_argument("--apply", action="store_true"); g.add_argument("--purge-old", action="store_true")
    ap.add_argument("--units", nargs="*", default=None)
    a = ap.parse_args()
    if a.purge_old: return purge()
    units = a.units or list(SPECS)
    plans = {u: plan(u) for u in units}
    for u, p in plans.items():
        print(f"{u}: archive {p['archive']}\n    keep {p['keep']}; bring in {BRING} {p['case']} {p['udf']} 2.jou (schedule {p['schedule']})")
    if a.dry_run: return
    A.mkdir(parents=True, exist_ok=True)
    logf = A / "SWAP_LOG.json"; log = json.loads(logf.read_text()) if logf.exists() else {"date": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "units": []}
    for u, p in plans.items():
        apply(u, p, log); print("swapped", u, flush=True)
        logf.write_text(json.dumps(log, indent=1, ensure_ascii=False))


if __name__ == "__main__":
    main()

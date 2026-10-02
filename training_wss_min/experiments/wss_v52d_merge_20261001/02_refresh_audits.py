"""Step 2 of the 2026-10-01 merge: refresh the shared-PREP audits of the changed units (same method as
outputs/cfd_auto_trial_20260927/_label_fix/refresh_audits.py: stale per-case reports moved aside, each module's own
audit_case recomputes, summaries merged; summary backups *.bak_merge_20261001).

    python 02_refresh_audits.py --topology <units...> [--wall-solver <units...>] [--centerline-root <root> ...]

topology     units whose case file changed (re-simulated) or whose outlet names changed (the audit records both)
wall-solver  units whose exports / transcript changed (re-simulated)
"""
import argparse, json, shutil, sys, time
from pathlib import Path

G = Path("/public/newhome/cy/Digital_twin/GNN"); sys.path.insert(0, str(G))
from wss_pinn.v4 import audit_anatomy_topology as topo  # noqa: E402
from wss_pinn.v4 import audit_solver_convergence as solv  # noqa: E402
from wss_pinn.v4 import audit_wall_wss as wall  # noqa: E402

TAG = "merge_20261001"
ROOTS = ["outputs/centerline_v2_full_173_20260828", "outputs/centerline_v2_meshwall_20260922_new95", "outputs/centerline_v2_meshwall_20260922_wave5",
         "outputs/centerline_v2_meshwall_20260922_wave3", "outputs/centerline_v2_meshwall_20260922_wave3b", "outputs/centerline_v2_meshwall_20260922_wave4",
         "outputs/centerline_v2_meshwall_20260927_syn60", "outputs/centerline_v2_meshwall_20260929_recover", "outputs/centerline_v2_meshwall_20260904"]


def stash(root: Path, units: list[str]) -> dict[str, str]:
    s, b = root / "summary.json", root / f"summary.json.bak_{TAG}"
    if not b.exists(): shutil.copy2(s, b)
    stale = root / f"_stale_{TAG}"; stale.mkdir(exist_ok=True)
    roles = {}
    for u in units:
        f = root / "cases" / f"{u.replace('/', '__')}.json"; g = stale / f.name
        if f.exists():
            roles[u] = json.loads(f.read_text()).get("role", "train")
            f.unlink() if g.exists() else shutil.move(str(f), str(g))
        elif g.exists():
            roles[u] = json.loads(g.read_text()).get("role", "train")
        else:
            roles[u] = "train"
    return roles


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--topology", nargs="*", default=[]); ap.add_argument("--wall-solver", nargs="*", default=[])
    ap.add_argument("--centerline-root", action="append", default=[]); ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    topo.CENTERLINE_ROOTS[:] = [Path(r) / "cases" if (Path(r) / "cases").is_dir() else Path(r) for r in a.centerline_root] + [G / r / "cases" for r in ROOTS if (G / r / "cases").is_dir()]
    out = {"date": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "topology_units": a.topology, "wall_solver_units": a.wall_solver}
    if a.topology:
        roles = stash(topo.AUDIT_ROOT, a.topology)
        tr = {u: topo.audit_case({"canonical_id": u, "role": roles[u]}) for u in a.topology}
        out["topology"] = {u: {"gate_pass": r.get("gate_pass"), "case_sha256": (r.get("fluent_case") or {}).get("sha256", "")[:12], "error": r.get("error"),
                               "outlets": {row.get("distal_bc_name"): row.get("semantic_label") for row in r.get("interfaces", [])}} for u, r in tr.items()}
        ts = json.loads((topo.AUDIT_ROOT / "summary.json").read_text())
        ts.setdefault("refreshed", []).append({"tag": TAG, "units": a.topology, "all_pass": all(r.get("gate_pass") for r in tr.values())})
        (topo.AUDIT_ROOT / "summary.json").write_text(json.dumps(ts, indent=1, ensure_ascii=False))
    U = a.wall_solver
    if U:
        roles = stash(wall.AUDIT_ROOT, U)
        wr = {u: wall.audit_case({"canonical_id": u, "role": roles[u]}) for u in U}
        ws = json.loads((wall.AUDIT_ROOT / "summary.json").read_text())
        old_fail = {f["canonical_id"] for f in ws.get("failures", []) if f["canonical_id"] in U}
        ws["failures"] = [f for f in ws.get("failures", []) if f["canonical_id"] not in U]
        for u, r in wr.items():
            if "gates" in r and not r["gate_pass"]:
                ws["failures"].append({"canonical_id": u, "role": r["role"], "failed_gates": [k for k, v in r["gates"].items() if not v],
                                       "pressure_delta_median_pa": r.get("provenance", {}).get("pressure_delta_median_pa"),
                                       "wall_lag_days": r.get("files", {}).get("wall_lag_days"), "tangency_p95": r.get("tangency", {}).get("p95"),
                                       "wss_max_pa": max((f["wss_max_pa"] for f in r.get("per_frame", {}).values()), default=None)})
        new_fail = {u for u, r in wr.items() if "gates" in r and not r["gate_pass"]}
        ws["pass"] = ws.get("pass", 0) + len(old_fail) - len(new_fail)
        ws.setdefault("refreshed", []).append({"tag": TAG, "units": U, "failures_before": sorted(old_fail), "failures_after": sorted(new_fail), "errors": [u for u, r in wr.items() if "error" in r]})
        (wall.AUDIT_ROOT / "summary.json").write_text(json.dumps(ws, indent=1, ensure_ascii=False))
        out["wall_wss"] = {u: {"gate_pass": r.get("gate_pass"), "failed": [k for k, v in (r.get("gates") or {}).items() if not v], "error": r.get("error")} for u, r in wr.items()}
        roles = stash(solv.AUDIT_ROOT, U)
        sr = {}
        for u in U:
            r = solv.audit_case({"canonical_id": u, "role": roles[u]})
            if "error" not in r:
                solv.annotate_solver_quality(r)
                (solv.AUDIT_ROOT / "cases" / f"{u.replace('/', '__')}.json").write_text(json.dumps(r, indent=1, ensure_ascii=False))
            sr[u] = r
        ss = json.loads((solv.AUDIT_ROOT / "summary.json").read_text()); q = ss["solver_quality"]
        old_rows = {r["canonical_id"]: r for r in q["by_case"] if r["canonical_id"] in U}
        q["by_case"] = [r for r in q["by_case"] if r["canonical_id"] not in U]
        for u, r in sr.items():
            if "solver_quality" in r:
                q["by_case"].append({"canonical_id": u, "role": r["role"], "tier": r["solver_quality"]["tier"], "continuity_last_max": r["solver_quality"]["continuity_last_max"],
                                     "flag_converged_export_steps": r["solver_quality"]["flag_converged_export_steps"], "convergence_criterion_type": r["solver_quality"]["convergence_criterion_type"]})
        q["by_case"].sort(key=lambda row: (row["tier"], row["canonical_id"]))
        tc = {}
        for row in q["by_case"]:
            b = tc.setdefault(row["tier"], {"all": 0, "train": 0, "test": 0}); b["all"] += 1; b[row["role"]] = b.get(row["role"], 0) + 1
        q["tier_counts"] = tc
        conv = set(ss.get("export_all_81_converged", [])) - set(U)
        conv |= {u for u, r in sr.items() if r.get("status") == "parsed" and r.get("gate_pass")}
        ss["export_all_81_converged"] = sorted(conv)
        ss["not_all_converged"] = [r for r in ss.get("not_all_converged", []) if r["canonical_id"] not in U] + [
            {"canonical_id": u, "role": r["role"], "converged": r["export_steps"]["converged"], "found": r["export_steps"]["found"],
             "iteration_cap_hits": r["export_steps"]["iteration_cap_hits"], "continuity_last_p95": r["export_steps"]["continuity_last_p95"],
             "continuity_last_max": r["export_steps"]["continuity_last_max"]} for u, r in sr.items() if r.get("status") == "parsed" and not r.get("gate_pass")]
        ss.setdefault("refreshed", []).append({"tag": TAG, "units": U, "tiers_before": {u: old_rows.get(u, {}).get("tier") for u in U},
                                              "tiers_after": {u: r.get("solver_quality", {}).get("tier") for u, r in sr.items()}})
        (solv.AUDIT_ROOT / "summary.json").write_text(json.dumps(ss, indent=1, ensure_ascii=False))
        out["solver"] = {u: {"tier_before": old_rows.get(u, {}).get("tier"), "tier_after": r.get("solver_quality", {}).get("tier"), "status": r.get("status"), "error": r.get("error")} for u, r in sr.items()}
    a.out.write_text(json.dumps(out, indent=1, ensure_ascii=False, default=str))
    print(json.dumps(out, indent=1, ensure_ascii=False, default=str))


if __name__ == "__main__":
    main()

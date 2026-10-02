"""Managed-run machinery without Fluent or Slurm: protocol file, schedules (journal export window, kept frames, Slurm
script), relabel, transcript classification, oblique-cut extensions, set-up checks, mesh gate and the orchestrator state
machine on a fake Slurm backend."""
import gzip
import json
import subprocess
from pathlib import Path

import numpy as np
import pytest

from cfd_auto import meshcheck, orchestrate, preflight, protocol, rebuild, relabel, runlog, sanity, schedule, surface, udf

ROOT = Path("/public/newhome/cy/Digital_twin/GNN")
PROTO = protocol.load()


# ------------------------------------------------------------------------------------------------ protocol
def test_protocol_matches_module_constants():
    s, e, bc = PROTO["solver"], PROTO["exports"], PROTO["bc"]
    assert (s["time_step_s"], s["n_steps"], s["max_iter"], s["period_s"]) == (0.005, 1280, 20, 0.8)
    assert protocol.label_steps(PROTO) == sanity.STEPS
    assert bc["mean_pressure_mmHg"] * bc["mmHg_to_Pa"] == pytest.approx(udf.PROTOCOL_P)
    assert {k: v for k, v in bc["A1_kg_s"].items() if k != "AG"} == {k: v for k, v in udf.A1_KG_S.items() if k != "AG"}
    assert bc["A1_kg_s"]["AG"] == pytest.approx(udf.A1_KG_S["AG"], rel=1e-4)    # 0.028421 (2026-09-30 rule) vs the older 0.02842
    assert PROTO["extension"]["outlet_L_over_D"] == rebuild.OUTLET_L_OVER_D
    assert PROTO["openings"]["outlet_keys"] == list(udf.OUTLETS)
    assert {k: tuple(v["fail"]) for k, v in PROTO["sanity_gates"].items()} == sanity.GATES


def test_protocol_overrides_merge():
    p = protocol.load(overrides={"solver": {"max_iter": 30}, "resources_extra": 1})
    assert p["solver"]["max_iter"] == 30 and p["solver"]["n_steps"] == 1280 and PROTO["solver"]["max_iter"] == 20


def test_ag_rule_scales_rcr():
    a = {k: v * 1e-6 for k, v in {"outle": 40.0, "outli": 20.0, "outri": 18.0, "outre": 41.0}.items()}
    r_old, r_new = udf.protocol_rcr(a, "AG"), udf.protocol_rcr(a, "AG", 0.028421)
    assert r_old == udf.protocol_rcr(a, "AG", None)
    assert r_new["outle"]["C"] / r_old["outle"]["C"] == pytest.approx(0.028421 / 0.02842, rel=1e-9)   # C = 1.79 / Rt, Rt = P / flow, flow ∝ A1


# ------------------------------------------------------------------------------------------------ schedules
def _iterates(jou: str) -> list[tuple[str, int]]:
    out, dt = [], None
    for ln in jou.splitlines():
        if ln.startswith("/solve/set/time-step"):
            dt = ln.split()[-1]
        if ln.startswith("/solve/dual-time-iterate"):
            out.append((dt, int(ln.split()[1])))
        if "cfdauto-freq cfdauto-exports" in ln and ln.startswith("(rpsetvar"):
            out.append(("EXPORTS_ON", int(ln.split()[-1].rstrip(")"))))
    return out


def test_library_schedule_exports_only_the_kept_window(tmp_path):
    sch = schedule.make("library", PROTO)
    assert sch.keep == list(range(1120, 1281, 2)) and sch.n_native == 1280 and sch.export_every == 2
    jou = schedule.journal(tmp_path, tmp_path / "c.cas.gz", sch)
    assert "(rpsetvar 'export/automatic '())" in jou.splitlines()[3]
    assert _iterates(jou) == [("0.005", 1119), ("EXPORTS_ON", 2), ("0.005", 161)]


def test_gentle_schedule_keeps_library_phase(tmp_path):
    sch = schedule.make("gentle", PROTO)
    assert sch.stages == [(0.0025, 320), (0.005, 960)] and sch.keep == list(range(1120, 1281, 2))
    # flow time of native step N: 320 * dt/2 + (N - 320) dt = N dt - period -> same phase as library step N
    t = lambda n: 320 * 0.0025 + (n - 320) * 0.005
    assert all(abs((t(n) - n * 0.005) % 0.8) < 1e-9 or abs((t(n) - n * 0.005) % 0.8 - 0.8) < 1e-9 for n in sch.keep)
    assert _iterates(schedule.journal(tmp_path, tmp_path / "c.cas.gz", sch)) == [("0.0025", 320), ("0.005", 799), ("EXPORTS_ON", 2), ("0.005", 161)]


def test_dt2_schedule_frames_and_labels(tmp_path):
    sch = schedule.make("dt2", PROTO)
    assert sch.n_native == 2560 and sch.keep == list(range(2240, 2561, 4)) and sch.export_every == 4
    assert [sch.label(n) for n in sch.keep] == list(range(1120, 1281, 2))
    assert _iterates(schedule.journal(tmp_path, tmp_path / "c.cas.gz", sch)) == [("0.0025", 2239), ("EXPORTS_ON", 4), ("0.0025", 321)]


def test_slurm_script_is_requeue_safe_and_valid_bash(tmp_path):
    sch = schedule.make("library", PROTO)
    rep = schedule.write_run_files(tmp_path, tmp_path / "c.cas.gz", sch, 92, "cfa-test", "fluent/231", make_dirs=("export",))
    text = (tmp_path / "fluent.slurm").read_text()
    assert subprocess.run(["bash", "-n", str(tmp_path / "fluent.slurm")]).returncode == 0
    lines = text.splitlines()
    clean = next(i for i, l in enumerate(lines) if l.startswith("rm -rf ascii ascii_in Global_conditions"))
    assert clean < next(i for i, l in enumerate(lines) if l.startswith("time fluent"))         # cleans BEFORE Fluent starts
    assert schedule.MARKER in text and (tmp_path / schedule.MARKER).exists()
    assert (tmp_path / schedule.KEEP_FILE).read_text().split() == [str(n) for n in sch.keep]
    assert rep["schedule"]["n_native"] == 1280 and "#SBATCH --ntasks-per-node=92" in text and "-t92" in text


def test_slurm_script_cleanup_behaviour(tmp_path):
    """Run the script's own bash (Fluent replaced by a stub that writes a few exports) twice, as a requeue would."""
    sch = schedule.make("library", PROTO)
    schedule.write_run_files(tmp_path, tmp_path / "c.cas.gz", sch, 4, "t", "fluent/231")
    text = (tmp_path / "fluent.slurm").read_text()
    stub = ("for n in 1119 1120 1121 1122; do printf x > ./C-$(printf %04d $n); printf x > ascii/C-$(printf %04d $n); done; "
            "echo 'time step = 1280'")
    text = text.replace("module purge\n", "").replace("module load fluent/231\n", "")
    text = text.replace(f"time fluent 3ddp -g -t4 -mpi=intel -platform=intel -i ./2.jou -ssh", stub)
    (tmp_path / "run.sh").write_text(text)
    for attempt in (1, 2):
        (tmp_path / "ascii" / "C-0999").write_text("stale") if (tmp_path / "ascii").exists() else None
        env = {"SLURM_SUBMIT_DIR": str(tmp_path), "SLURM_JOB_ID": "7", "PATH": "/usr/bin:/bin"}
        r = subprocess.run(["bash", str(tmp_path / "run.sh")], env=env, capture_output=True, text=True)
        assert r.returncode == 0, r.stderr
        assert sorted(p.name for p in (tmp_path / "ascii").iterdir()) == ["C-1120", "C-1122"]
        assert sorted(p.name for p in (tmp_path / "ascii_in").iterdir()) == ["C-1120", "C-1122"]
        assert not list(tmp_path.glob("C-[0-9]*"))


def test_relabel_dt2_layout(tmp_path):
    sch = schedule.make("dt2", PROTO)
    schedule.write_run_files(tmp_path, tmp_path / "c.cas.gz", sch, 4, "t", "fluent/231")
    (tmp_path / "ascii_in").mkdir()
    for n in sch.keep:
        (tmp_path / "ascii" / f"C-{n:04d}").write_text(str(n)); (tmp_path / "ascii_in" / f"C-{n:04d}").write_text(str(n))
    gc = tmp_path / "Global_conditions"; gc.mkdir()
    (gc / "p-outle-rfile.out").write_text('"title"\n"x"\n2239 1.0 5.5975\n2240 2.0 5.6\n')
    (gc / "Fluent_cfdauto_7.out").write_text("header\nUpdating solution at time level N...\n a\ntime step = 2239\n b\ntime step = 2240\n")
    rep = relabel.relabel(tmp_path)
    assert rep["moved"] == {"ascii": 81, "ascii_in": 81}
    assert sorted(int(p.name[-4:]) for p in (tmp_path / "ascii").iterdir()) == list(range(1120, 1281, 2))
    assert (tmp_path / "ascii" / "C-1120").read_text() == "2240"
    assert (gc / "p-outle-rfile.out").read_text().splitlines()[-1] == "1120 2.0 5.6"
    assert "time step = 1120" in (gc / "Fluent_cfdauto_7.out").read_text() and "time step = 2239" not in (gc / "Fluent_cfdauto_7.out").read_text()
    assert relabel.relabel(tmp_path) == rep          # idempotent


# ------------------------------------------------------------------------------------------------ transcript
DIV = PROTO["divergence"]


def _log(tmp_path: Path, body: str, job: str = "5", archived: bool = False) -> None:
    d = tmp_path / "Global_conditions" if archived else tmp_path
    d.mkdir(exist_ok=True)
    (d / f"Fluent_cfdauto_{job}.out").write_text(body)


def _steps(res: list[float], start: int = 1) -> str:
    """Transcript blocks as v231 prints them: residual rows, then the report-file line (step, flow time, report, flow
    time), then the time-step line. The report line must never be read as a residual."""
    return "".join(f"   19  {r * 1.1:.4e}  1.0000e-05  1.0000e-05  1.0000e-05  0:00:01    1\n   20  {r:.4e}  1.0000e-05  1.0000e-05  1.0000e-05  0:00:01    0\n"
                   f"  step  flow-time report-def-0   flow-time\n    {start + i}  {(start + i) * 0.005 + 1.2:.4e}  9.4542e-01  {(start + i) * 0.005 + 1.2:.4e}\n"
                   f"Flow time = {(start + i) * 0.005:.3f}s, time step = {start + i}\n" for i, r in enumerate(res))


def test_runlog_statuses(tmp_path):
    _log(tmp_path, f"{schedule.ATTEMPT_MARK} job=5\n" + _steps([1e-4] * 3))
    assert runlog.classify(tmp_path, 1280, DIV, "5")["status"] == "running"
    _log(tmp_path, f"{schedule.ATTEMPT_MARK}\n" + _steps([1e-4, 0.2, 0.24, 0.3]))
    assert runlog.classify(tmp_path, 1280, DIV, "5")["status"] == "diverging"
    _log(tmp_path, f"{schedule.ATTEMPT_MARK}\n" + _steps([1e-4, 1e-2, 0.24, 130.0]))
    assert runlog.classify(tmp_path, 1280, DIV, "5")["status"] == "diverged"
    _log(tmp_path, f"{schedule.ATTEMPT_MARK}\n" + _steps([1e-4]) + "Error at Node 3: floating point exception\n" + schedule.FINISH_MARK)
    assert runlog.classify(tmp_path, 1280, DIV, "5")["status"] == "diverged"
    _log(tmp_path, f"{schedule.ATTEMPT_MARK}\n" + _steps([1e-4] * 3) + schedule.FINISH_MARK)
    assert runlog.classify(tmp_path, 1280, DIV, "5")["status"] == "ended_early"
    _log(tmp_path, f"{schedule.ATTEMPT_MARK}\n" + _steps([1e-4] * 3, start=1278) + schedule.FINISH_MARK, archived=True)
    (tmp_path / "Fluent_cfdauto_5.out").unlink()
    assert runlog.classify(tmp_path, 1280, DIV, "5")["status"] == "completed"


def test_runlog_ignores_the_unfinished_step(tmp_path):
    _log(tmp_path, f"{schedule.ATTEMPT_MARK}\n    1  1.0000e+00  1.0000e+00  1.0000e+00  1.0000e+00  0:00:05   19\n")
    assert runlog.classify(tmp_path, 1280, DIV, "5")["status"] == "running"


def test_runlog_only_reads_the_last_attempt(tmp_path):
    _log(tmp_path, f"{schedule.ATTEMPT_MARK}\n" + _steps([1e-4]) + "floating point exception\n" + f"{schedule.ATTEMPT_MARK}\n" + _steps([1e-4] * 2))
    assert runlog.classify(tmp_path, 1280, DIV, "5")["status"] == "running"


WANG = ROOT / "outputs/cfd_auto_trial_20260927/_recover/units/ILO/WANG_CAI-0/before/failed_run1_16100"


@pytest.mark.skipif(not WANG.exists(), reason="WANG_CAI failed run not available")
def test_runlog_on_the_real_wang_cai_divergence():
    h = runlog.classify(WANG, 1280, DIV, "16100")
    assert h["status"] == "diverged" and h["last_step"] < 100


# ------------------------------------------------------------------------------------------------ geometry / mesh / set-up
def _oblique_cylinder(path: Path, deg: float = 27.0) -> Path:
    import pyvista as pv
    R, H, K, M = 5.0, 60.0, 60, 48
    th = np.linspace(0, 2 * np.pi, M, endpoint=False)
    P = np.vstack([np.c_[R * np.cos(th), R * np.sin(th), k * H / K + np.tan(np.radians(deg)) * R * np.cos(th) * (1 - k / K)] for k in range(K + 1)])
    F = []
    for k in range(K):
        a = k * M + np.arange(M); a1 = k * M + (np.arange(M) + 1) % M
        F += [np.c_[a, a1, a1 + M], np.c_[a, a1 + M, a + M]]
    F = np.vstack(F)
    pv.PolyData(P, np.hstack([np.full((len(F), 1), 3), F]).ravel()).save(str(path))
    return path


def test_oblique_cut_extension_along_the_axis(tmp_path):
    surf, _ = surface.close_surface(_oblique_cylinder(tmp_path / "c.stl"), {0: ("in", "velocity-inlet"), 1: ("out", "pressure-outlet")})
    d = surface.extension_directions(surf, "auto", PROTO["extension"]["oblique_deg"])
    assert d["in"]["angle_deg"] == pytest.approx(27.0, abs=0.5) and d["in"]["direction"] is not None
    assert d["out"]["direction"] is None                                   # square cut: library behaviour
    assert np.dot(d["in"]["direction"], [0, 0, -1]) > 0.9999
    exts = [surface.Extension(c.name, 30.0, f"wall{i + 1}", c.name + "+", None if d[c.name]["direction"] is None else np.asarray(d[c.name]["direction"]))
            for i, c in enumerate(surf.caps)]
    P, zones, stats = surface.extend_surface(surf, exts)
    assert surface.check_regions(P, zones, exts)["ok"]
    s_in = next(s for s in stats if s["opening"] == "in")
    assert s_in["distal_area_mm2"] == pytest.approx(0.5 * 48 * 25 * np.sin(2 * np.pi / 48), rel=1e-6)   # the true section, not the cut
    assert s_in["distal_planarity_mm"] < 1e-9


def test_oblique_extension_after_other_extensions(tmp_path):
    """The swept tube is not the first extension: its node ids continue after the earlier tubes' nodes (09-30 WANG_CAI)."""
    surf, _ = surface.close_surface(_oblique_cylinder(tmp_path / "c.stl"), {0: ("in", "velocity-inlet"), 1: ("out", "pressure-outlet")})
    d = surface.extension_directions(surf, "auto", PROTO["extension"]["oblique_deg"])
    exts = [surface.Extension(c.name, 30.0, f"wall{i + 1}", c.name + "+", None if d[c.name]["direction"] is None else np.asarray(d[c.name]["direction"]))
            for i, c in enumerate(surf.caps)][::-1]
    P, zones, _ = surface.extend_surface(surf, exts)
    assert surface.check_regions(P, zones, exts)["ok"]


def test_normal_mode_extension_unchanged(tmp_path):
    surf, _ = surface.close_surface(_oblique_cylinder(tmp_path / "c.stl"), {0: ("in", "velocity-inlet"), 1: ("out", "pressure-outlet")})
    e0 = [surface.Extension(c.name, 30.0, f"wall{i + 1}", c.name + "+") for i, c in enumerate(surf.caps)]
    e1 = [surface.Extension(c.name, 30.0, f"wall{i + 1}", c.name + "+", None) for i, c in enumerate(surf.caps)]
    a, b = surface.extend_surface(surf, e0), surface.extend_surface(surf, e1)
    assert np.array_equal(a[0], b[0]) and all(np.array_equal(x[2], y[2]) for x, y in zip(a[1], b[1]))


def test_mesh_quality_gate():
    mp = PROTO["mesh"]
    log = "Mesh Quality:\n\nMinimum Orthogonal Quality =  1.01661e-01 cell 1\n\nMaximum Aspect Ratio =  6.43570e+01 cell 2\n"
    q = meshcheck.fluent_quality(log)
    assert q == {"min_orthogonal_quality": 0.101661, "max_aspect_ratio": 64.357}
    good_bl = {"bl_layers_mode": 10, "bl_layer_hist": {"10": 990, "9": 10}}
    assert meshcheck.quality_gate(q, good_bl, mp)["ok"]
    no_bl = meshcheck.quality_gate({"min_orthogonal_quality": 1.3e-4, "max_aspect_ratio": 1508.0}, {"bl_layers_mode": 0, "bl_layer_hist": {"0": 1000}}, mp)
    assert not no_bl["ok"] and len(no_bl["failures"]) == 3                      # LI_FA_XIANG-1/before's first mesh
    warn = meshcheck.quality_gate({"min_orthogonal_quality": 0.0459, "max_aspect_ratio": 70.0}, good_bl, mp)
    assert warn["ok"] and warn["warnings"]                                       # LIU_YU_MING ran fine
    lib_poly = {"bl_layers_mode": 10, "bl_layer_hist": {"3": 1, "6": 2, "7": 2, "8": 14, "9": 45, "10": 1849, "11": 77, "12": 10}}   # library YU_TIAN_HAI
    assert meshcheck.quality_gate(q, lib_poly, mp)["ok"]
    assert not meshcheck.quality_gate({}, good_bl, mp)["ok"]


def _pbc(areas: dict) -> dict:
    return {"cut_area_mm2": areas, "rcr": udf.protocol_rcr({k: v * 1e-6 for k, v in areas.items()}, "ILO")}


def test_preflight_levels():
    ok = preflight.check(PROTO, _pbc({"outle": 60.0, "outli": 30.0, "outri": 28.0, "outre": 58.0}))
    assert ok["ok"] and not ok["warn"]
    small = preflight.check(PROTO, _pbc({"outle": 60.0, "outli": 5.9, "outri": 28.0, "outre": 58.0}))
    assert small["ok"] and "small_outlets" in small["warn"]
    r2 = preflight.check(PROTO, _pbc({"outle": 60.0, "outli": 30.0, "outri": 5.1, "outre": 7.3}))      # YANG_QING_REN-1/after-like
    assert not r2["ok"] and "rcr_R2_positive" in r2["fail"]


def test_preflight_naming():
    nm = lambda conf, side, methods=("deploy",), agree=True: {"methods": {m: {"confidence": conf, "side_confidence": side} for m in methods}, "all_agree": agree}
    assert preflight.naming_item(PROTO, nm(0.99, {"1": 1.0}))["level"] == "ok"
    assert preflight.naming_item(PROTO, nm(0.99, {"1": 0.943}))["level"] == "fail"                    # LIU_YU_MING's side
    assert preflight.naming_item(PROTO, nm(0.99, {"1": 0.943}), confirmed=True)["level"] == "ok"
    assert preflight.naming_item(PROTO, nm(0.9, {}, ("partner", "deploy")))["level"] == "ok"          # two methods agree
    assert preflight.naming_item(PROTO, nm(0.99, {}, ("partner", "deploy"), agree=False))["level"] == "fail"


def test_network_split_symmetric_and_skewed():
    tree = {0: {"parent": -1, "outlet": None, "R": 1.0}, 1: {"parent": 0, "outlet": None, "R": 0.0}, 2: {"parent": 0, "outlet": None, "R": 0.0},
            3: {"parent": 1, "outlet": "a", "R": 0.0}, 4: {"parent": 1, "outlet": "b", "R": 0.0}, 5: {"parent": 2, "outlet": "c", "R": 0.0}, 6: {"parent": 2, "outlet": "d", "R": 0.0}}
    sh = preflight.network_split(tree, {"a": 1.0, "b": 1.0, "c": 1.0, "d": 1.0})
    assert sh == pytest.approx({"a": 0.25, "b": 0.25, "c": 0.25, "d": 0.25})
    tree[1]["R"] = 1.0
    sh = preflight.network_split(tree, {"a": 2.0, "b": 2.0, "c": 2.0, "d": 2.0})
    assert sh["a"] + sh["b"] == pytest.approx(1 / 2 / (1 / 2 + 1 / 1))


def test_sanity_default_gates_unchanged():
    import inspect
    assert inspect.signature(sanity.check).parameters["gates"].default is None


# ------------------------------------------------------------------------------------------------ orchestrator
class FakeSlurm:
    def __init__(self):
        self.jobs, self.n, self.submitted, self.cancelled = {}, 100, [], []

    def submit(self, script, cwd, name, env=None):
        self.n += 1; j = str(self.n)
        self.jobs[j] = {"name": name, "state": "PENDING", "script": str(script)}
        self.submitted.append((j, name)); return j

    def queued(self, job):
        s = self.jobs.get(job, {}).get("state")
        return s if s in ("PENDING", "RUNNING") else None

    def final_state(self, job):
        return self.jobs.get(job, {}).get("state", "UNKNOWN")

    def find(self, name):
        return next((j for j, v in self.jobs.items() if v["name"] == name), None)

    def cancel(self, job):
        self.cancelled.append(job); self.jobs[job]["state"] = "CANCELLED"

    def finish(self, job, state="COMPLETED"):
        self.jobs[job]["state"] = state


@pytest.fixture
def batch(tmp_path, monkeypatch):
    bf = tmp_path / "batch.json"
    bf.write_text(json.dumps({"batch": "t", "work_root": str(tmp_path / "units"), "library": str(tmp_path / "lib"),
                              "resources": {"cfd_cores": 8, "max_parallel_cfd": 1}, "units": {"AAA/x/U1": {"mode": "own-mesh", "template": "AAA/x/T"},
                                                                                              "AAA/x/U2": {"mode": "own-mesh", "template": "AAA/x/T"}}}))
    fake = FakeSlurm()
    b = orchestrate.Batch(bf, backend=fake)
    b.res["nfs_grace_s"] = 0
    real_tick = b.tick
    def tick(dry_run=False):              # with no grace a finished job is handled on the tick after it left the queue
        real_tick(dry_run)
        return real_tick(dry_run)
    b.tick = tick
    from cfd_auto import prepare
    monkeypatch.setattr(prepare, "rewrite_schedule", lambda work, proto, name, cores, jn, exclude="node05": schedule.write_run_files(work, work / "c.cas.gz", schedule.make(name, proto), cores, jn, "fluent/231",
                                                                                                                  exclude=exclude or None))
    return b, fake


def test_orchestrator_waits_for_nfs_grace(tmp_path):
    bf = tmp_path / "batch.json"
    bf.write_text(json.dumps({"batch": "g", "work_root": str(tmp_path / "units"), "library": str(tmp_path / "lib"), "units": {"AAA/x/U": {"mode": "own-mesh", "template": "AAA/x/T"}}}))
    fake = FakeSlurm(); b = orchestrate.Batch(bf, backend=fake)
    b.tick(); j = _job_of(b, "AAA/x/U"); fake.finish(j, "NODE_FAIL")
    b.tick(); b.tick()
    st = b.load("AAA/x/U")
    assert st["state"] == "preparing" and st["active"]["job"] == j and "left_queue" in st["active"]    # not decided within the grace


def _prepared(b, unit, job):
    w = b.work(unit); w.mkdir(parents=True, exist_ok=True)
    schedule.write_run_files(w, w / "c.cas.gz", schedule.make("library", PROTO), 8, "x", "fluent/231")
    (w / "orchestrator").mkdir(exist_ok=True)
    (w / "orchestrator" / "prepare_result.json").write_text(json.dumps({"status": "ready", "job": job, "message": "ready"}))


def _job_of(b, unit):
    return b.load(unit)["active"]["job"]


def test_orchestrator_happy_path_and_ladder(batch):
    b, fake = batch
    b.tick()
    assert [b.load(u)["state"] for u in b.units] == ["preparing", "preparing"]
    for u in b.units:
        j = _job_of(b, u); _prepared(b, u, j); fake.finish(j)
    b.tick()
    st = {u: b.load(u)["state"] for u in b.units}
    assert sorted(st.values()) == ["cfd", "ready"]                         # max_parallel_cfd = 1
    u1 = next(u for u, s in st.items() if s == "cfd"); w = b.work(u1); j = _job_of(b, u1)
    fake.jobs[j]["state"] = "RUNNING"
    _log(w, f"{schedule.ATTEMPT_MARK}\n" + _steps([1e-4, 0.2, 0.24, 0.5]), job=j)
    b.tick()
    assert j in fake.cancelled                                             # diverging -> cancelled early
    b.tick()
    s1 = b.load(u1)
    assert s1["state"] in ("ready", "cfd") and s1["ladder_index"] == 1
    assert json.loads((w / schedule.FRAMES_FILE).read_text())["schedule"]["name"] == "gentle"
    assert any((w / "attempts").iterdir())
    # gentle run completes
    if s1["state"] == "ready":
        b.tick()
    j2 = _job_of(b, u1)
    _log(w, f"{schedule.ATTEMPT_MARK}\n" + _steps([1e-4] * 3, start=1278) + schedule.FINISH_MARK, job=j2, archived=True)
    fake.finish(j2)
    b.tick()
    assert b.load(u1)["state"] == "post"
    jp = _job_of(b, u1)
    (w / "orchestrator" / "post_result.json").write_text(json.dumps({"status": "checked", "job": jp, "message": "sanity ok"}))
    fake.finish(jp)
    b.tick()
    assert b.load(u1)["state"] == "checked"


def test_orchestrator_recovers_a_lost_submission(batch):
    b, fake = batch
    st = b.load("AAA/x/U1")
    st["active"] = {"kind": "prepare", "name": b.job_name("AAA/x/U1", "prepare", 1), "job": None, "submitted": "t", "n": 1}
    b.save(st)
    fake.jobs["555"] = {"name": st["active"]["name"], "state": "RUNNING"}  # sbatch happened, the tick died before saving the id
    b.tick()
    assert b.load("AAA/x/U1")["active"]["job"] == "555"
    assert sum(1 for _, n in fake.submitted if n.endswith("prepare1") and "U1" not in n) <= 1
    names = [n for _, n in fake.submitted]
    assert b.job_name("AAA/x/U1", "prepare", 1) not in names                # not resubmitted


def test_orchestrator_prepare_outcomes(batch):
    b, fake = batch
    b.tick()
    u1, u2 = list(b.units)
    for u, status in ((u1, "needs_confirmation"), (u2, "blocked")):
        j = _job_of(b, u); w = b.work(u); (w / "orchestrator").mkdir(parents=True, exist_ok=True)
        (w / "orchestrator" / "prepare_result.json").write_text(json.dumps({"status": status, "job": j, "message": status}))
        fake.finish(j)
    b.tick()
    assert b.load(u1)["state"] == "needs_confirmation" and b.load(u2)["state"] == "blocked"
    b.retry(u2, "prepare")
    assert b.load(u2)["state"] == "pending"


def test_orchestrator_infra_retry_then_fail(batch):
    b, fake = batch
    b.tick()
    u1 = list(b.units)[0]
    j = _job_of(b, u1); fake.finish(j, "NODE_FAIL")        # no result file: infrastructure
    b.tick()
    assert b.load(u1)["state"] in ("pending", "preparing") and b.load(u1)["prepare_retries"] == 1


def test_orchestrator_lock(batch):
    b, _ = batch
    fh = orchestrate._locked(b)
    with pytest.raises(SystemExit):
        orchestrate._locked(b)
    fh.close()
    orchestrate._locked(b).close()                       # released -> available again
    run = orchestrate._locked(b, ".orchestrate.run.lock")
    orchestrate._locked(b).close()                       # a running loop does not block manual actions between ticks
    run.close()


def test_polyhexcore_journal_prompt_order(tmp_path):
    from cfd_auto import journals
    j = journals.meshing_polyhexcore(tmp_path, tmp_path / "s.msh.gz", ["wall", "wall1"], ["wall", "wall1", "in", "in+"], tmp_path / "o.msh.gz",
                                     0.0005, 0.002, n_layers=10, first_aspect_ratio=20, growth=1.2, tet_volume_growth=1.2).splitlines()
    i = j.index("/objects/create")
    assert j[i + 1:i + 8] == ["anatomy", "fluid", "3", "(wall wall1 in in+)", "()", "mesh", "yes"]
    k = j.index("/mesh/scoped-prisms/create")
    assert j[k + 1:k + 9] == ["bl", "aspect-ratio", "20", "10", "1.2", "anatomy", "fluid-regions", "only-walls"]
    a = j.index("/mesh/auto-mesh")
    assert j[a + 1:a + 7] == ["anatomy", "no", "scoped", "pyramids", "poly-hexcore", "yes"]
    assert j.index("/size-functions/delete") < i                     # octree follows the surface sizes, not the size field


def test_old_journals_unchanged_by_default(tmp_path):
    """The 09-27/29 CLIs must write the same journals as before the managed additions (committed 3b0874d version)."""
    import importlib.util
    from cfd_auto import journals
    r = subprocess.run(["git", "show", "3b0874d:cfd_auto/journals.py"], cwd=ROOT, capture_output=True, text=True)
    if r.returncode:
        pytest.skip("commit 3b0874d not available")
    src = tmp_path / "journals_head.py"; src.write_text(r.stdout)
    spec = importlib.util.spec_from_file_location("journals_head", src); old = importlib.util.module_from_spec(spec); spec.loader.exec_module(old)
    a = (tmp_path, tmp_path / "s.msh.gz", ["wall", "wall1"], ["wall", "wall1", "in", "in+"], tmp_path / "o.msh.gz")
    assert journals.meshing(*a) == old.meshing(*a)
    assert journals.meshing_poly(*a, 0.0004, 0.002) == old.meshing_poly(*a, 0.0004, 0.002)
    assert journals.smoke(tmp_path, tmp_path / "c.cas.gz") == old.smoke(tmp_path, tmp_path / "c.cas.gz")
    assert journals.run(tmp_path, tmp_path / "c.cas.gz") == old.run(tmp_path, tmp_path / "c.cas.gz")


def test_exclude_resource_reaches_fluent_scripts(tmp_path, monkeypatch):
    """Batch ``resources.exclude`` "" -> no --exclude line in the managed fluent.slurm nor in the small Fluent jobs."""
    from cfd_auto import protocol, schedule, slurm
    proto = protocol.load("aortoiliac_rcr4_v1")
    (tmp_path / "c.cas.gz").write_text("")
    schedule.write_run_files(tmp_path, tmp_path / "c.cas.gz", schedule.make("library", proto), 92, "j", "fluent/231", exclude=None)
    assert "--exclude" not in (tmp_path / "fluent.slurm").read_text()
    schedule.write_run_files(tmp_path, tmp_path / "c.cas.gz", schedule.make("library", proto), 92, "j", "fluent/231")
    assert "#SBATCH --exclude=node05" in (tmp_path / "fluent.slurm").read_text()      # default unchanged
    calls = []
    monkeypatch.setattr(slurm.subprocess, "run", lambda cmd, **kw: calls.append(cmd) or type("R", (), {"stdout": "7\n"})())
    monkeypatch.setattr(slurm, "EXCLUDE", "")
    jou = tmp_path / "x.jou"; jou.write_text("/exit yes\n")
    slurm.submit(tmp_path, jou)
    assert "--exclude" not in (tmp_path / "logs" / "_x.slurm").read_text()

"""Pure-function checks for the 2026-09-29 recovery additions (no library data needed)."""
from pathlib import Path

import pytest

from cfd_auto import compare, journals, recover, sanity, udf

WORK = Path("/public/newhome/cy/Digital_twin/GNN/outputs/cfd_auto_trial_20260927/_recover/units/_test")


def _prof(ext_map, zones, key_zone):
    """Minimal profile: openings (name, type), extensions (opening -> fluid/wall/interface), UDF key -> zone."""
    openings = [{"name": n, "bc_type": t} for n, t in zones]
    extensions = [{"opening": n, "fluid_zone": ext_map[n][0], "wall_zone": ext_map[n][1], "interface_zone": [ext_map[n][2]]} for n, _ in zones]
    return {"anatomy_zone": "blood", "anatomy_wall_zones": ["wall"], "openings": openings, "extensions": extensions, "udf_thread_zones": key_zone}


def test_rename_zones_two_phase_swap_and_types():
    text = journals.rename_zones(WORK, WORK / "a.cas.gz", {"blood2": "blood5", "blood5": "blood2"}, WORK / "b.cas.gz", zone_types={"out-a": "pressure-outlet"})
    lines = text.splitlines()
    tmp = [l for l in lines if "cfdauto-tmp" in l]
    assert len(tmp) == 4                                   # both names leave before either arrives
    assert lines.index("/mesh/modify-zones/zone-name blood2 cfdauto-tmp-0") < lines.index("/mesh/modify-zones/zone-name cfdauto-tmp-1 blood2")
    assert "/define/boundary-conditions/zone-type out-a pressure-outlet" in lines


def test_gentle_start_keeps_library_phase():
    lines = journals.run_gentle_start(WORK, WORK / "x.cas.gz").splitlines()
    it = [l.split() for l in lines if l.startswith("/solve/dual-time-iterate")]
    dts = [float(l.split()[-1]) for l in lines if l.startswith("/solve/set/time-step")]
    assert [int(i[1]) for i in it] == [320, 960] and dts == [0.0025, 0.005]
    assert sum(int(i[1]) for i in it) == 1280
    # step N >= 320: t = 0.8 + (N-320)*0.005 = N*0.005 - 0.8 -> same phase as the library (t = N*0.005)
    for n in (1120, 1162, 1280):
        t = 320 * 0.0025 + (n - 320) * 0.005
        assert abs((t % 0.8) - ((n * 0.005) % 0.8)) < 1e-9 or abs(abs((t % 0.8) - ((n * 0.005) % 0.8)) - 0.8) < 1e-9
    with pytest.raises(ValueError):
        journals.run_gentle_start(WORK, WORK / "x.cas.gz", refine=8)     # the fine cycle would take all 1280 steps


def test_own_mesh_renames_by_opening_key():
    zones = [("inlet", "velocity-inlet"), ("o-lw", "pressure-outlet"), ("o-ln", "pressure-outlet"), ("o-rn", "pressure-outlet"), ("o-rw", "pressure-outlet")]
    own = _prof({"inlet": ("blood1", "wall1", "inlet1"), "o-lw": ("blood2", "wall2", "o-lw1"), "o-ln": ("blood3", "wall3", "o-ln1"),
                  "o-rn": ("blood4", "wall4", "o-rn1"), "o-rw": ("blood5", "wall5", "o-rw1")}, zones,
                 {"outle": "o-lw", "outli": "o-ln", "outri": "o-rn", "outre": "o-rw"})
    tmpl = _prof({"inlet": ("blood1", "wall1", "inlet1"), "o-lw": ("blood5", "wall2", "o-lw1"), "o-ln": ("blood4", "wall3", "o-ln1"),
                  "o-rn": ("blood3", "wall4", "o-rn1"), "o-rw": ("blood2", "wall5", "o-rw1")}, zones,
                 {"outle": "o-lw", "outli": "o-ln", "outri": "o-rn", "outre": "o-rw"})
    assert recover.own_mesh_renames(own, tmpl) == {"blood2": "blood5", "blood3": "blood4", "blood4": "blood3", "blood5": "blood2"}


def test_protocol_rcr_structure():
    rcr = udf.protocol_rcr({"outle": 40e-6, "outli": 20e-6, "outri": 25e-6, "outre": 35e-6}, "ILO")
    for k, v in rcr.items():
        assert v["R2"] > 0 and abs(v["C"] * (v["R1"] + v["R2"]) - 1.79) < 1e-9


def test_verdict_skips_missing_reference():
    res = {k: 0.0 for k in compare.GATES["v2"]}
    res.update(P1_wss_r2=0.99, P4_hotspot_iou=0.9, P5_pressure_centred_r2=0.999, P6_flow_share_abs_max=None, P6_outlet_pressure_rel_max=None)
    out = compare._verdict(res, "v2")
    assert out["all_pass"] and out["gates_skipped"] == ["P6_flow_share_abs_max", "P6_outlet_pressure_rel_max"]
    assert out["gates"]["P6_flow_share_abs_max"]["pass"] is None


def test_step_prints_belong_to_the_step_they_precede(tmp_path):
    log = tmp_path / "t.out"
    log.write_text("P_ave_outle=1 Q_ave_outle=2\nFlow time = 0.005s, time step = 1\nP_ave_outle=3 Q_ave_outle=4\nFlow time = 0.01s, time step = 2\n")
    pr = sanity.step_prints(log)
    assert pr[1]["P_ave_outle"] == 1 and pr[2]["Q_ave_outle"] == 4

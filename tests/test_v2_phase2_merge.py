"""Node tests for the seams between the second-phase lanes after they were merged (PHASE2_LANES.md §7).

Lane E's volume display units (Pa / mmHg, m/s / cm/s) reach lane D's overview numbers, volume findings, evidence
lens and along-vessel curves through ws_display.toDisplay; wall shear stress keeps Pa; without ws_display (or with the
default units) nothing changes.  The references handed to the lens stay in stored units.
"""
from __future__ import annotations

import json

import pytest

from tests.test_v2_display import V2, _node

VOLUME_MANIFEST = {
    "result": {"family": "volume"},
    "fields": [{"id": "pressure", "units": "Pa"}, {"id": "speed", "units": "m/s"}],
    "analysis": {"profiles": {"branches": [{"segment_id": 0, "name": "主动脉", "s_from_root_mm": [0, 2, 4], "radius_mm": [10, 10, 10],
                                            "volume": {"n": [5, 5, 5], "pressure_mean_pa": [1333.22, 666.61, 0],
                                                       "pressure_min_pa": [-133.322, -266.644, -399.966],
                                                       "speed_mean_m_s": [0.5, 0.25, 0.1], "speed_max_m_s": [1, 0.5, 0.2]}}]}},
}


def _scenario(body: str) -> dict:
    files = [V2 / "ws_ui.js", V2 / "ws_overview.js", V2 / "ws_lens.js", V2 / "ws_profile.js"]
    return _node("".join("require(%s);\n" % json.dumps(str(f)) for f in files) + "const M=" + json.dumps(VOLUME_MANIFEST) + ";\n" + body,
                 extra=["ws_display.js"])


def test_volume_numbers_follow_display_units_wall_shear_keeps_pa():
    out = _scenario("""
      const D=ns.display, O=ns.overview, P=ns.profile, L=ns.lens;
      const find=[{kind:'min_pressure',value:-1333.22,units:'Pa'},{kind:'max_speed',value:1.5,units:'m/s'},{kind:'max_wss',value:12,units:'Pa'}];
      const lensRef={kind:'stat',field:'pressure',stat:'range',value:1333.22,units:'Pa',label:'相对压力跨度'};
      const valueRows=()=>L.chain(lensRef,M,{}).filter(s=>s.step==='value')[0].rows.map(r=>r[1]);
      const before={td:D.toDisplay(1333.22,'Pa','volume'),find:find.map(O.findingValue),lens:valueRows(),units:P.spec(M,'pressure').units,
        mean:P.spec(M,'pressure').primary.get(M.analysis.profiles.branches[0])};
      D.setUnit('pressure','mmHg'); D.setUnit('velocity','cm/s');
      out({before,
        td:D.toDisplay(1333.22,'Pa','volume'),wall:D.toDisplay(12,'Pa','wall'),other:D.toDisplay(3,'mm','volume'),missing:D.toDisplay(null,'Pa','volume'),
        find:find.map(O.findingValue),lens:valueRows(),refAfter:lensRef.value,
        p:P.spec(M,'pressure'),pMean:P.spec(M,'pressure').primary.get(M.analysis.profiles.branches[0]),
        s:P.spec(M,'speed').units,sMax:P.spec(M,'speed').secondary.get(M.analysis.profiles.branches[0]),
        model:P.model(M,'speed',0).primary});
    """)
    b = out["before"]
    assert b["td"] == {"value": 1333.22, "units": "Pa"}                        # default units: unchanged
    assert b["units"] == "Pa" and b["mean"] == pytest.approx([1333.22, 666.61, 0])
    assert "Pa" in b["find"][0] and "m/s" in b["find"][1] and "Pa" in b["lens"][0]
    assert out["td"]["units"] == "mmHg" and out["td"]["value"] == pytest.approx(10.0, rel=1e-6)
    assert out["wall"] == {"value": 12, "units": "Pa"}                          # wall shear keeps Pa
    assert out["other"] == {"value": 3, "units": "mm"} and out["missing"]["value"] is None
    f0, f1, f2 = out["find"]
    assert "mmHg" in f0 and "-10" in f0.replace("−", "-")
    assert "cm/s" in f1 and "150" in f1
    assert "Pa" in f2 and "mmHg" not in f2                                     # a wall finding is not converted
    assert "mmHg" in out["lens"][0] and "10" in out["lens"][0]
    assert out["refAfter"] == 1333.22                                          # the lens reference stays raw
    assert out["p"]["units"] == "mmHg" and out["pMean"] == pytest.approx([10, 5, 0], rel=1e-6)
    assert out["s"] == "cm/s" and out["sMax"] == pytest.approx([100, 50, 20])
    assert out["model"] == pytest.approx([50, 25, 10])                         # the drawn curve uses the display unit


def test_numbers_are_raw_without_the_display_module():
    files = [V2 / "ws_ui.js", V2 / "ws_overview.js", V2 / "ws_profile.js"]
    out = _node("".join("require(%s);\n" % json.dumps(str(f)) for f in files) + "const M=" + json.dumps(VOLUME_MANIFEST) + ";\n" + """
      out({has:Boolean(ns.display),find:ns.overview.findingValue({kind:'min_pressure',value:-1333.22,units:'Pa'}),units:ns.profile.spec(M,'pressure').units});
    """)
    assert out["has"] is False and "Pa" in out["find"] and out["units"] == "Pa"

"""Seams between the phase-three lanes after they were merged (PHASE3_LANES.md; scope doc §11.8).

* Lane 4's wall-stress display unit (Pa / dyn/cm²) reaches lane 3's along-vessel curve (values, threshold lines, axis
  title); RRT / ECAP (1/Pa) and OSI keep their own units.
* The in-place re-login dialog (lane 6's ws_main.js) has the same helps as lane 2's login page: show / hide, the Caps
  Lock hint, the wait a 429 names, 5xx said as maintenance.
* The shortcut table lists lane 4's P (hover readout) and the reset key 0.
"""
from __future__ import annotations

import json

from tests.test_v2_display import V2, _node
from tests.test_v2_lane_c import _run

WALL = {
    "result": {"family": "wall"},
    "fields": [{"id": "wss", "units": "Pa"}, {"id": "tawss", "units": "Pa"}, {"id": "rrt", "units": "1/Pa", "display": {"thresholds": [5, 10, 20]}}],
    "analysis": {"profiles": {"branches": [{"segment_id": 0, "name": "主动脉", "s_from_root_mm": [0, 2], "radius_mm": [10, 10],
                                            "wss": {"n": [5, 5], "p99_pa": [2.0, 3.0], "mean_pa": [1.0, 1.5]},
                                            "tawss": {"n": [5, 5], "mean_pa": [0.5, 0.6], "min_pa": [0.1, 0.2]},
                                            "rrt": {"n": [5, 5], "p90": [4.0, 6.0], "mean": [2.0, 3.0]}}],
                              "cycle": None},
                 "cycle": {"stagnation": {"criteria": {"tawss_lt_pa": 0.4, "osi_gt": 0.1}}}},
}


def test_along_vessel_curves_follow_the_wall_stress_unit():
    files = [V2 / "ws_ui.js", V2 / "ws_profile.js"]
    out = _node("".join("require(%s);\n" % json.dumps(str(f)) for f in files) + "const M=" + json.dumps(WALL, ensure_ascii=False) + ";\n" + """
      const P=ns.profile, D=ns.display, b=M.analysis.profiles.branches[0];
      const before={wss:P.spec(M,'wss').units, p99:P.spec(M,'wss').primary.get(b)};
      D.setUnit('wss','dyn/cm²');
      const w=P.spec(M,'wss'), t=P.spec(M,'tawss'), r=P.spec(M,'rrt');
      out({before, wss:w.units, p99:w.primary.get(b), tawss:t.units, tmean:t.primary.get(b), hline:t.hlines[0].v, rrt:r.units, rp90:r.primary.get(b)});
    """, extra=["ws_display.js"])
    assert out["before"] == {"wss": "Pa", "p99": [2.0, 3.0]}
    assert out["wss"] == "dyn/cm²" and out["p99"] == [20.0, 30.0]
    assert out["tawss"] == "dyn/cm²" and out["tmean"] == [5.0, 6.0] and out["hline"] == 4.0   # the 0.4 Pa line moves with it
    assert out["rrt"] == "1/Pa" and out["rp90"] == [4.0, 6.0]                                  # 1/Pa keeps its unit


def test_relogin_dialog_has_the_login_page_helps():
    out = _run(r"""
      await boot();
      canned['/api/session'] = {body: {authenticated: false, login: 'password', csrf_token: null}};
      canned['/api/trash'] = {status: 401, body: {error: {message: '请先登录。'}}};
      await ns.api.request('/api/trash').catch(() => null); await wait(60);
      const dlg = byId('ws-relogin');
      const reveal = walk(dlg, e => e.attrs['aria-label'] === '显示口令')[0];
      const pass = walk(dlg, e => e.attrs['aria-label'] === '口令')[0];
      fire(reveal, 'click'); await wait(5);
      const shown = pass.type;
      const caps = walk(dlg, e => String(e.className || '').includes('login-caps'))[0];
      canned['POST /api/session'] = {status: 429, body: {error: {message: '尝试过于频繁'}, retry_after: 7}};
      pass.value = 'x';
      fire(walk(dlg, e => e.tagName === 'FORM')[0], 'submit'); await wait(60);
      const submit = walk(dlg, e => e.tagName === 'BUTTON' && e.type === 'submit')[0];
      const wait429 = {text: textOf(submit), disabled: submit.disabled, err: textOf(walk(dlg, e => String(e.className || '').includes('login-err'))[0])};
      done({shown, caps: Boolean(caps && caps.hidden), wait429});
    """)
    assert out["errors"] == [], out["errors"]
    assert out["shown"] == "text" and out["caps"] is True
    assert out["wait429"]["disabled"] is True and "秒后可再试" in out["wait429"]["text"] and "7 秒" in out["wait429"]["err"]


def test_shortcut_table_lists_hover_readout_and_reset():
    out = _run(r"""
      canned['/api/jobs'] = {body: {jobs: [jobRecord('A')]}};
      await boot();
      await hashTo('#/job/A', 300);
      await key('?'); await wait(20);
      done({keys: walk(byId('ws-dialog'), e => e.tagName === 'TD').map(textOf)});
    """)
    assert out["errors"] == [], out["errors"]
    keys = out["keys"]
    assert "P" in keys and keys[keys.index("P") + 1] == "悬停读数开关"
    assert "0" in keys and keys[keys.index("0") + 1] == "复位视角"

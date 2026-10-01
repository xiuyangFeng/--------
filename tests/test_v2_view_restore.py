"""Reading positions stay with their own result (user report 10-01: a volume result opened without its interior points).

A view save that came due while the next result was still loading stored the previous result's viewer state (a wall
result's field and layers: no interior points, no outline) under the next result, and every later open restored it.
The save now waits for the result to be shown and checks the viewer shows it; a save due when a result is closed is
made for that result first; a saved viewer state naming a field the result does not have is left out on restore; a
load cut short by the next result is not reported as an error."""
from __future__ import annotations

from tests.test_v2_workspace_js import _run

VOLUME = r"""
      serve('V', {family: 'volume', run_identity: 'run-V'}, {family: 'volume', name: 'CASE_V', resultName: '体内压力与速度',
        fields: [FIELD('pressure', {location: 'interior'}), FIELD('speed', {location: 'interior', units: 'm/s'})]});
      // the viewer shows a result only once its arrays have arrived (the real kernel), and says which one it shows
      const create0 = ns.viewer.create;
      ns.viewer.create = function (c, o) {
        const v = create0(c, o), set0 = v.setResult; let shown = null;
        v.setResult = r => set0(r).then(x => { shown = r; return x; });
        v.result = () => shown;
        return v;
      };
"""


def test_a_save_due_while_the_next_result_loads_never_lands_on_it():
    out = _run(VOLUME + r"""
      canned['/api/v2/jobs/V/manifest'].body.__viewerDelay = 1000;   // V's arrays take a second
      await boot();
      await hashTo('#/job/A', 200);
      const v = shellState().viewerA;
      v.emit('camera', {source: 'user'});            // the user turns A's view: a save is due in 600 ms
      await wait(300);
      await hashTo('#/job/V', 100);                  // … and opens V before it
      const aView = ns.store.readView('run-A');
      v.emit('camera', {source: 'user'});            // the view moves while V loads (still A in the viewer)
      await wait(700);
      const vWhileLoading = ns.store.readView('run-V');
      await wait(600);
      v.emit('camera', {source: 'user'}); await wait(700);
      const vShown = ns.store.readView('run-V');
      done({aField: aView && aView.viewer && aView.viewer.field, aFamily: aView && aView.family, vWhileLoading,
        vShown: vShown && {field: vShown.field, viewer: vShown.viewer && vShown.viewer.field, family: vShown.family}});
    """)
    assert out["errors"] == [], out["errors"]
    assert out["aField"] == "tawss" and out["aFamily"] == "wall"     # A's last turn is kept, for A
    assert out["vWhileLoading"] is None                               # nothing of A under V
    assert out["vShown"] == {"field": "speed", "viewer": "speed", "family": "volume"}


def test_a_saved_view_from_another_result_is_left_out_on_restore():
    out = _run(VOLUME + r"""
      await boot();
      // what the old race left behind: A's viewer state (TAWSS, no interior, no outline) saved under V
      ns.store.writeView('run-V', {field: 'speed', window: 'adaptive', viewer: {field: 'tawss', layers: {interior: false, outline: false},
        camera: {position: [9, 9, 9], target: [0, 0, 0], up: [0, 0, 1], fov: 30}}});
      await hashTo('#/job/V', 200);
      const broken = {restored: vcalls.filter(c => c.endsWith(':applyState')).length, fit: vcalls.filter(c => c.endsWith(':fit')).length, field: shellState().cur.field};
      await hashTo('#/', 100);
      ns.store.writeView('run-V', {field: 'pressure', window: 'adaptive', family: 'wall', viewer: {field: 'pressure'}});
      vcalls.length = 0;
      await hashTo('#/job/V', 200);
      const otherFamily = {restored: vcalls.filter(c => c.endsWith(':applyState')).length, field: shellState().cur.field};
      await hashTo('#/', 100);
      ns.store.writeView('run-V', {field: 'pressure', window: 'adaptive', family: 'volume', viewer: {field: 'pressure', layers: {interior: true}}});
      vcalls.length = 0;
      await hashTo('#/job/V', 200);
      const own = {restored: vcalls.filter(c => c.endsWith(':applyState')).length, field: shellState().cur.field};
      done({broken, otherFamily, own});
    """)
    assert out["errors"] == [], out["errors"]
    assert out["broken"]["restored"] == 0 and out["broken"]["fit"] >= 1 and out["broken"]["field"] == "speed"
    assert out["otherFamily"] == {"restored": 0, "field": "pressure"}
    assert out["own"] == {"restored": 1, "field": "pressure"}


def test_a_load_cut_short_is_not_reported_as_an_error():
    out = _run(r"""
      await boot();
      await hashTo('#/job/A', 200);
      const v = shellState().viewerA, toast = () => { const t = byId('ws-toast'); return t && !t.hidden ? textOf(t) : ''; };
      const ab = new Error('signal is aborted without reason'); ab.name = 'AbortError';
      v.emit('error', ab); await wait(20);
      const afterAbort = toast();
      v.emit('error', new Error('shader failed')); await wait(20);
      done({afterAbort, afterError: toast()});
    """)
    assert out["errors"] == [], out["errors"]
    assert "三维显示出错" not in out["afterAbort"]
    assert "三维显示出错：shader failed" in out["afterError"]

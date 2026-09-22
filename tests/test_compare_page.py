"""The side-by-side comparison page is served with the right headers and relays cameras between frames."""
from __future__ import annotations

import http.client
import json
import shutil
import subprocess
import threading
from pathlib import Path

import pytest

from wss_deploy.jobs import JobManager
from wss_deploy.server import STATIC_FILES, ServiceHTTPServer, SessionStore

STATIC = Path(__file__).resolve().parents[1] / "wss_deploy" / "static"


def test_compare_page_and_script_are_served_with_locked_down_csp(tmp_path):
    manager = JobManager(tmp_path / "jobs", stage_a_fn=lambda *_a, **_k: {}, stage_b_fn=lambda *_a, **_k: {})
    sessions = SessionStore(tmp_path / "sessions.json", shared=False)
    server = ServiceHTTPServer(("127.0.0.1", 0), manager, sessions)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    api = http.client.HTTPConnection("127.0.0.1", server.server_port, timeout=5)
    try:
        api.request("GET", "/compare?left=a&right=b")
        response = api.getresponse(); body = response.read().decode("utf-8")
        assert response.status == 200 and response.getheader("Content-Type").startswith("text/html")
        assert response.getheader("Set-Cookie")  # a local session is created like on the index page
        csp = response.getheader("Content-Security-Policy")
        assert "frame-ancestors 'none'" in csp and "default-src 'self'" in csp
        assert 'id="frame-left"' in body and 'id="frame-right"' in body and 'id="sync-camera"' in body and "/static/compare.js" in body
        # §15.7 controls and the shared helper module the relay needs
        assert 'id="sync-display"' in body and 'id="push-left"' in body and 'id="push-right"' in body
        assert "/static/workbench_core.js" in body
        api.request("GET", "/static/compare.js")
        response = api.getresponse(); script = response.read().decode("utf-8")
        assert response.status == 200 and response.getheader("Content-Type").startswith("application/javascript")
        assert "wss-view:camera" in script and "wss-view:set-camera" in script and "/api/compare" in script
        assert "wss-view:get-state" in script and "wss-view:apply-state" in script
        api.request("GET", "/static/workbench_core.js")
        response = api.getresponse(); response.read()
        assert response.status == 200
        api.request("GET", "/static/compare.html")
        response = api.getresponse(); response.read()
        assert response.status == 200
    finally:
        api.close(); server.shutdown(); server.server_close(); manager.close()


def test_static_whitelist_includes_compare_assets_only():
    assert {"compare.html", "compare.js"} <= STATIC_FILES
    assert "onepage.html" not in STATIC_FILES


def test_compare_script_passes_node_syntax_check():
    if not shutil.which("node"):
        pytest.skip("node not available")
    subprocess.run(["node", "--check", str(STATIC / "compare.js")], check=True)
    subprocess.run(["node", "--check", str(STATIC / "app.js")], check=True)


def test_camera_relay_forwards_between_frames_without_echo():
    """Drive compare.js in Node with a tiny DOM/window stub: a camera event from the left frame is
    forwarded to the right frame as set-camera, the right frame's echo inside the guard window is
    dropped, and nothing is forwarded while sync is off."""
    if not shutil.which("node"):
        pytest.skip("node not available")
    program = r"""
      const script = require('fs').readFileSync(process.argv[1], 'utf8');
      const posted = [];
      const mkWin = name => ({name, postMessage: (msg, origin) => posted.push({to: name, msg, origin})});
      const wins = {left: mkWin('left'), right: mkWin('right')};
      const els = {};
      const el = id => els[id] || (els[id] = {id, textContent: '', className: '', hidden: false, checked: true, src: '',
        contentWindow: id === 'frame-left' ? wins.left : id === 'frame-right' ? wins.right : null,
        listeners: {}, addEventListener(name, fn) { this.listeners[name] = fn; }, replaceChildren() {}, append() {}});
      const listeners = {};
      global.window = {location: {origin: 'http://127.0.0.1:1', search: '?left=A&right=B'}, addEventListener: (n, fn) => { listeners[n] = fn; }};
      global.document = {getElementById: el, createElement: () => ({append() {}, setAttribute() {}, addEventListener() {}, style: {}}), createTextNode: t => t};
      global.Node = class {};
      global.URLSearchParams = URLSearchParams;
      let now = 1000; Date.now = () => now;
      global.fetch = async () => ({ok: true, status: 200, json: async () => ({authenticated: false})});
      eval(script);
      const send = (from, camera) => listeners.message({origin: 'http://127.0.0.1:1', source: wins[from], data: {type: 'wss-view:camera', family: 'wall', camera}});
      send('left', {position: [1, 2, 3], target: [0, 0, 0], up: [0, 0, 1]});
      const first = posted.length;
      send('right', {position: [1, 2, 3], target: [0, 0, 0], up: [0, 0, 1]});   // echo inside the guard window
      const afterEcho = posted.length;
      now += 500;
      send('right', {position: [9, 9, 9], target: [0, 0, 0], up: [0, 0, 1]});   // genuine move later
      const afterMove = posted.length;
      listeners.message({origin: 'http://evil', source: wins.left, data: {type: 'wss-view:camera', camera: {}}});
      el('sync-camera').checked = false; el('sync-camera').listeners.change();
      now += 500; send('left', {position: [5, 5, 5], target: [0, 0, 0], up: [0, 0, 1]});
      console.log(JSON.stringify({first, afterEcho, afterMove, final: posted.length, posted}));
    """
    result = subprocess.run(["node", "-e", program, str(STATIC / "compare.js")], check=True, text=True, capture_output=True)
    out = json.loads(result.stdout.strip().splitlines()[-1])
    assert out["first"] == 1 and out["afterEcho"] == 1 and out["afterMove"] == 2 and out["final"] == 2
    assert out["posted"][0]["to"] == "right" and out["posted"][0]["msg"]["type"] == "wss-view:set-camera"
    assert out["posted"][0]["msg"]["camera"]["position"] == [1, 2, 3] and out["posted"][0]["origin"] == "http://127.0.0.1:1"
    assert out["posted"][1]["to"] == "left" and out["posted"][1]["msg"]["camera"]["position"] == [9, 9, 9]


_DISPLAY_STUB = r"""
  const script = require('fs').readFileSync(process.argv[1], 'utf8');
  global.WssWorkbenchCore = require(process.argv[2]);
  const posted = [];
  const mkWin = name => ({name, postMessage: (msg, origin) => posted.push({to: name, msg, origin})});
  const wins = {left: mkWin('left'), right: mkWin('right')};
  const els = {};
  const el = id => els[id] || (els[id] = {id, textContent: '', className: '', hidden: false, checked: true, src: '',
    contentWindow: id === 'frame-left' ? wins.left : id === 'frame-right' ? wins.right : null,
    listeners: {}, addEventListener(name, fn) { this.listeners[name] = fn; }, replaceChildren() {}, append() {}});
  const listeners = {};
  global.window = {location: {origin: 'http://127.0.0.1:1', search: '?left=A&right=B'}, addEventListener: (n, fn) => { listeners[n] = fn; }};
  global.document = {getElementById: el, createElement: () => ({append() {}, setAttribute() {}, addEventListener() {}, style: {}}), createTextNode: t => t};
  global.Node = class {};
  global.URLSearchParams = URLSearchParams;
  let now = 1000; Date.now = () => now;
  global.fetch = async () => ({ok: true, status: 200, json: async () => ({authenticated: false})});
  eval(script);
  const send = (from, data) => listeners.message({origin: 'http://127.0.0.1:1', source: wins[from], data});
  const typed = type => posted.filter(entry => entry.msg.type === type);
  const tick = () => new Promise(resolve => setTimeout(resolve, 5));
  const viewState = {schema_version: 'wss-deploy.view/v1', family: 'wall', run_identity: 'r1', camera: {position: [1, 2, 3]},
    field: 'wss', mode: 'field', colormap: 'turbo', bands: 6, log: false, range: {mode: 'case', min: 0, max: 10},
    units: 'Pa', thresholds_pa: [0.4, 4, 7], opacity: 0.9, overlay: {trust: true}, slice: {}, highlight: {point: 1}, measurements: []};
  (async () => {
    send('left', {type: 'wss-view:ready', family: 'wall', run_identity: 'r1', case_id: 'A'});
    send('right', {type: 'wss-view:ready', family: 'wall', run_identity: 'r2', case_id: 'B'});
    // two quick camera moves from the left frame: throttled into a single get-state
    send('left', {type: 'wss-view:camera', family: 'wall', camera: {position: [1, 2, 3], target: [0, 0, 0], up: [0, 0, 1]}});
    now += 10;
    send('left', {type: 'wss-view:camera', family: 'wall', camera: {position: [1, 2, 4], target: [0, 0, 0], up: [0, 0, 1]}});
    await tick();
    const asked = typed('wss-view:get-state');
    send('left', {type: 'wss-view:state', request_id: asked[0].msg.request_id, family: 'wall', state: viewState});
    const applied = typed('wss-view:apply-state');
    // the target's applied reply is our own push coming back: it must not start a reverse sync
    send('right', {type: 'wss-view:applied', request_id: applied[0].msg.request_id});
    await tick();
    const afterEcho = typed('wss-view:get-state').length;
    // a state answer whose request id we never sent is ignored
    send('left', {type: 'wss-view:state', request_id: 'stale', family: 'wall', state: viewState});
    const afterStale = typed('wss-view:apply-state').length;
    // later, a genuine move on the right side syncs the other way
    now += 500;
    send('right', {type: 'wss-view:camera', family: 'wall', camera: {position: [9, 9, 9], target: [0, 0, 0], up: [0, 0, 1]}});
    await tick();
    const reverse = typed('wss-view:get-state');
    send('right', {type: 'wss-view:state', request_id: reverse[reverse.length - 1].msg.request_id, family: 'wall', state: viewState});
    const reverseApply = typed('wss-view:apply-state');
    // mixed families keep only the family-independent display keys
    send('right', {type: 'wss-view:ready', family: 'volume', run_identity: 'r2', case_id: 'B'});
    now += 500;
    el('push-left').listeners.click();
    const oneShot = typed('wss-view:get-state');
    send('left', {type: 'wss-view:state', request_id: oneShot[oneShot.length - 1].msg.request_id, family: 'wall', state: viewState});
    const crossApply = typed('wss-view:apply-state');
    // switching the toggle off stops the relay entirely
    el('sync-display').checked = false; el('sync-display').listeners.change();
    now += 500;
    send('left', {type: 'wss-view:camera', family: 'wall', camera: {position: [7, 7, 7], target: [0, 0, 0], up: [0, 0, 1]}});
    await tick();
    console.log(JSON.stringify({asked: asked.length, askedTo: asked[0].to, askedOrigin: asked[0].origin,
      applyTo: applied[0].to, applyState: applied[0].msg.state, applyCount: applied.length,
      afterEcho, afterStale, reverseTo: reverse[reverse.length - 1].to, reverseApplyTo: reverseApply[reverseApply.length - 1].to,
      crossState: crossApply[crossApply.length - 1].msg.state, crossTo: crossApply[crossApply.length - 1].to,
      finalAsked: typed('wss-view:get-state').length, cameras: typed('wss-view:set-camera').length}));
  })();
"""


def test_display_state_round_trip_between_frames():
    """§15.7: an activity signal fetches the active frame's view state and pushes the display subset
    to the other frame; echoes inside the guard window are ignored and families are respected."""
    if not shutil.which("node"):
        pytest.skip("node not available")
    result = subprocess.run(["node", "-e", _DISPLAY_STUB, str(STATIC / "compare.js"), str(STATIC / "workbench_core.js")],
                            check=True, text=True, capture_output=True)
    out = json.loads(result.stdout.strip().splitlines()[-1])
    assert out["asked"] == 1 and out["askedTo"] == "left" and out["askedOrigin"] == "http://127.0.0.1:1"
    assert out["applyTo"] == "right" and out["applyCount"] == 1
    state = out["applyState"]
    assert state["field"] == "wss" and state["colormap"] == "turbo" and state["thresholds_pa"] == [0.4, 4, 7]
    assert "camera" not in state and "highlight" not in state and "run_identity" not in state and "measurements" not in state
    assert out["afterEcho"] == 1                      # the applied echo did not become a new source
    assert out["afterStale"] == 1                     # an unknown request id changes nothing
    assert out["reverseTo"] == "right" and out["reverseApplyTo"] == "left"
    assert set(out["crossState"]) == {"colormap", "bands", "log", "opacity"} and out["crossTo"] == "right"
    assert out["finalAsked"] == 3                     # nothing more once the toggle is off
    assert out["cameras"] >= 1                        # the camera relay keeps working alongside

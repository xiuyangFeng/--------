"""Real-browser acceptance for the operations event feed and management shortcuts.

    PYTHONPATH=. python -m tests.ops_browser_feed --out outputs/wss_ops_feed_check

Uses only a temporary synthetic HTTP service. No compute worker is started.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timedelta, timezone
import json
from pathlib import Path
import socket
import tempfile
import time

from tests._c_helpers import Service, finished
from tests._ops_browser_viewport import set_css_viewport
from wss_deploy.devshot import Browser
from wss_deploy.users import UserStore


def wait(browser, expression, timeout=15):
    end = time.monotonic() + timeout
    while time.monotonic() < end:
        value = browser.js("return " + expression)
        if value:
            return value
        time.sleep(.15)
    raise AssertionError(expression + "\n" + browser.js("return document.body.innerText"))


def view(browser, name):
    browser.js("document.querySelector('#ops-nav [data-view=\"'+arguments[0]+'\"]').click()", [name])
    wait(browser, f"!document.getElementById('view-{name}').hidden")


def poll(browser):
    browser.js("document.activeElement?.blur(); document.dispatchEvent(new Event('visibilitychange'))")


def run(out):
    out.mkdir(parents=True, exist_ok=True)
    result_path = out / "browser-feed.json"
    result_path.unlink(missing_ok=True)
    results = {}
    with tempfile.TemporaryDirectory(prefix="wss-ops-feed-") as temporary:
        root = Path(temporary)
        users = UserStore(root / "users.json")
        users.add("admin", "feed-admin-password", admin=True)
        users.add("alice", "feed-alice-password")
        users.add("bob", "feed-bob-password")
        service = Service(root, shared=True, users=users)
        store = service.server.operations
        try:
            good = finished(service.manager, owner="alice", case_id="BATCH_GOOD", content=b"solid good\nendsolid good")
            bad = finished(service.manager, owner="bob", case_id="BATCH_MISSING", content=b"solid missing\nendsolid missing")
            (service.manager.root / bad["id"] / "input.stl").unlink()
            store.create_ticket(owner="alice", title="待分配工单", description="合成测试", audit_actor="alice")
            now = datetime.now(timezone.utc)
            for index in range(65):
                store.record_event({"at": (now - timedelta(seconds=150-index)).isoformat(), "action": "finished",
                                    "actor": "system", "owner": "alice", "job": good["id"], "source": "job",
                                    "status": "done", "details": {"sequence": index}})
            diagnostic = store.record_event({"action": "failed", "actor": "system", "owner": "alice", "job": good["id"],
                                             "source": "job", "status": "failed",
                                             "details": {"reason": "feed-hidden-diagnostic", "error": {"message": "合成几何检查失败"}}})
            warning = store.record_event({"action": "login_failed", "owner": "alice", "actor": "alice", "source": "http",
                                          "details": {"reason": "feed-export-marker", "ip": "127.0.0.1"}})
            with socket.socket() as sock:
                sock.bind(("127.0.0.1", 0))
                port = sock.getsockname()[1]
            downloads = root / "downloads"
            downloads.mkdir()
            prefs = {"browser.download.folderList": 2, "browser.download.dir": str(downloads),
                     "browser.download.alwaysOpenPanel": False,
                     "browser.helperApps.neverAsk.saveToDisk": "application/json"}
            with Browser(width=1440, height=1050, marionette_port=port, log_path=out / "browser-feed.log", prefs=prefs) as browser:
                browser.go(f"http://127.0.0.1:{service.port}/ops", wait=.4)
                wait(browser, "!document.getElementById('ops-login').hidden")
                browser.js("document.getElementById('login-user').value='admin';document.getElementById('login-password').value='feed-admin-password';document.getElementById('ops-login-form').requestSubmit()")
                wait(browser, "!document.getElementById('ops-content').hidden")
                view(browser, "overview")
                wait(browser, "document.querySelector('[data-shortcut=\"待分派工单\"]')")
                browser.js("document.querySelector('[data-shortcut=\"待分派工单\"]').click()")
                wait(browser, "!document.getElementById('view-tickets').hidden && document.getElementById('tickets-rows').textContent.includes('待分配工单')")
                assert browser.js("return document.getElementById('tickets-unassigned').value==='true' && document.getElementById('tickets-status').value==='unresolved'")
                results["overview_shortcut_applies_matching_filters"] = True
                view(browser, "users")
                browser.js("document.getElementById('users-query').value='bob';document.getElementById('users-disabled').value='false';document.getElementById('users-filter').requestSubmit()")
                wait(browser, "document.querySelectorAll('#users-rows tr').length===1 && document.getElementById('users-rows').textContent.includes('bob')")
                assert not browser.js("return document.getElementById('users-rows').textContent.includes('alice')")
                results["user_search_and_access_filter"] = True
                view(browser, "events")
                entry = f"#events-rows details.event-entry[data-event-id=\"{diagnostic['id']}\"]"
                wait(browser, f"document.querySelector({json.dumps(entry)})")
                assert not browser.js("return document.body.innerText.includes('feed-hidden-diagnostic')")
                assert browser.js("return !document.querySelector('#events-rows table')")
                browser.js("window.reading=document.querySelector(arguments[0]);reading.querySelector('summary').click()", [entry])
                wait(browser, "window.reading.open")
                raw = browser.js("return Boolean(reading.querySelector('details.event-raw'))")
                assert raw
                assert browser.js("return !reading.querySelector('details.event-raw').open")
                browser.js("reading.querySelector('details.event-raw > summary').click()")
                wait(browser, "document.body.innerText.includes('feed-hidden-diagnostic')")
                results["summary_first_expand_details_then_raw"] = True

                live = store.record_event({"action": "created", "actor": "bob", "owner": "bob", "job": "LIVE_FEED_NEW", "source": "job"})
                poll(browser)
                wait(browser, "!document.getElementById('events-new').hidden")
                assert browser.js("return reading.isConnected && reading.open && reading.querySelector('details.event-raw').open")
                assert not browser.js("return Boolean(document.querySelector(arguments[0]))", [f'[data-event-id="{live["id"]}"]'])
                results["new_event_preserves_open_details"] = True
                browser.js("document.getElementById('events-new').click()")
                wait(browser, f"document.querySelector('[data-event-id=\"{live['id']}\"]')")

                browser.js("document.querySelectorAll('details.event-entry[open]').forEach(e=>e.open=false);document.getElementById('events-pause').click()")
                before = browser.js("return document.getElementById('events-rows').textContent")
                paused = store.record_event({"action": "created", "actor": "alice", "owner": "alice", "job": "PAUSED_NEW_EVENT", "source": "job"})
                poll(browser)
                wait(browser, "!document.getElementById('events-new').hidden")
                assert before == browser.js("return document.getElementById('events-rows').textContent")
                browser.js("document.getElementById('events-latest').click()")
                wait(browser, f"document.querySelector('[data-event-id=\"{paused['id']}\"]')")
                results["pause_buffers_and_latest_resumes"] = True

                browser.js("document.getElementById('events-next').click()")
                wait(browser, "document.getElementById('events-page').textContent.startsWith('2 /')")
                history = browser.js("return document.getElementById('events-rows').textContent")
                view(browser, "overview")
                store.record_event({"action": "created", "actor": "alice", "owner": "alice", "job": "HISTORY_NEW_EVENT", "source": "job"})
                poll(browser)
                wait(browser, "!document.getElementById('events-new').hidden")
                view(browser, "events")
                assert browser.js("return document.getElementById('events-page').textContent.startsWith('2 /')")
                assert history == browser.js("return document.getElementById('events-rows').textContent")
                results["history_page_does_not_shift_on_poll"] = True

                browser.js("document.getElementById('events-category').value='auth';document.getElementById('events-severity').value='warning';document.getElementById('events-query').value='feed-export-marker';document.getElementById('events-filter').requestSubmit()")
                wait(browser, "document.querySelectorAll('details.event-entry').length===1")
                assert browser.js("return document.querySelector('details.event-entry').dataset.eventId", []) == str(warning["id"])
                browser.js("document.getElementById('events-query').value='UNAPPLIED-DRAFT';window.exportPayload=null;window.originalCreateURL=URL.createObjectURL;URL.createObjectURL=function(blob){blob.text().then(t=>window.exportPayload=JSON.parse(t));return originalCreateURL.call(this,blob)};document.getElementById('events-export').click()")
                wait(browser, "window.exportPayload !== null")
                exported = browser.js("return window.exportPayload")
                assert [row["id"] for row in exported["events"]] == [warning["id"]]
                assert exported["filters"]["q"] == "feed-export-marker"
                results["filters_and_export_use_applied_snapshot"] = True
                browser.js("URL.createObjectURL=originalCreateURL;document.getElementById('events-reset').click()")
                wait(browser, "document.querySelectorAll('details.event-entry').length>1")

                view(browser, "jobs")
                wait(browser, "document.querySelectorAll('input.job-select').length===2")
                browser.js("document.getElementById('jobs-select-page').click();document.getElementById('jobs-archive-selected').click()")
                wait(browser, "document.querySelectorAll('#jobs-batch-results li').length===2 && !document.getElementById('jobs-archive-selected').disabled")
                assets, total = store.assets()
                assert total == 1 and assets[0]["sources"][0]["job_id"] == good["id"]
                assert browser.js("return document.querySelector(arguments[0]).checked", [f'input.job-select[data-job-id="{bad["id"]}"]'])
                assert not browser.js("return document.querySelector(arguments[0]).checked", [f'input.job-select[data-job-id="{good["id"]}"]'])
                results["batch_archive_reports_partial_success_keeps_failures"] = True
                browser.js("document.getElementById('jobs-user').value='alice';document.getElementById('jobs-filter').requestSubmit()")
                wait(browser, "document.querySelectorAll('input.job-select').length===1")
                assert not browser.js("return document.querySelector('input.job-select').checked")
                results["job_filter_clears_batch_selection"] = True

                view(browser, "assets")
                wait(browser, "document.querySelector('details.ops-asset-provenance')")
                assert not browser.js("return document.querySelector('details.ops-asset-provenance').open")
                browser.js("document.querySelector('details.ops-asset-provenance > summary').click()")
                wait(browser, "document.body.innerText.includes('BATCH_GOOD')")
                results["asset_provenance_expands_with_original_case"] = True

                view(browser, "events")
                browser.js("document.querySelector('#ops-notice .notice-close')?.click();window.scrollTo(0,0)")
                browser.shot(out / "events-desktop.png", full=True)
                results["mobile_viewports"] = []
                for width in (390, 320):
                    for name in ("overview", "jobs", "tickets", "assets", "users", "events"):
                        view(browser, name)
                        metrics = set_css_viewport(browser, width, 844)
                        assert metrics["scrollWidth"] <= metrics["clientWidth"], (name, metrics)
                        results["mobile_viewports"].append({"view": name, **metrics})
                        if name == "events":
                            browser.shot(out / f"events-{width}.png", full=True)
                            browser.js("document.querySelector('details.event-entry > summary').click()")
                            metrics = set_css_viewport(browser, width, 844)
                            assert metrics["scrollWidth"] <= metrics["clientWidth"], ("event-details", metrics)
                            browser.shot(out / f"events-expanded-{width}.png", full=True)
                            browser.js("document.querySelectorAll('details.event-entry[open]').forEach(e=>e.open=false)")
                set_css_viewport(browser, 1440, 1050)
                results["navigation_and_feed_mobile_layout"] = True

                # Hold a completed write response while another tab changes the
                # cookie. Session checks must continue during the mutation, and
                # releasing its old response must not resurrect private content.
                view(browser, "jobs")
                browser.js("window.nativeFetch=window.fetch;window.mutationDelivered=false;window.releaseMutation=null;window.fetch=async(...args)=>{const response=await nativeFetch(...args);if(String(args[0])==='/api/ops/assets'&&args[1]?.method==='POST'){window.mutationDelivered=true;await new Promise(resolve=>window.releaseMutation=resolve);}return response;};document.querySelector('#jobs-rows button').click()")
                wait(browser, "window.mutationDelivered && window.releaseMutation!==null")
                browser.js("window.switched=false;fetch('/api/session',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({username:'bob',password:'feed-bob-password'})}).then(r=>r.json()).then(()=>{window.switched=true;document.dispatchEvent(new Event('visibilitychange'))})")
                wait(browser, "window.switched && document.getElementById('ops-content').hidden")
                assert browser.js("return document.querySelectorAll('details.event-entry').length===0 && document.querySelectorAll('input.job-select:checked').length===0")
                results["identity_switch_clears_feed_and_selections"] = True
                browser.js("window.fetch=nativeFetch;window.releaseMutation()")
                time.sleep(.4)
                assert browser.js("return document.getElementById('ops-content').hidden && document.querySelectorAll('details.event-entry').length===0 && document.getElementById('jobs-rows').childElementCount===0 && document.getElementById('ops-notice').hidden")
                results["late_mutation_response_cannot_restore_previous_account"] = True
                results["javascript_errors"] = browser.errors()
                assert results["javascript_errors"] == []
            result_path.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
            print(json.dumps(results, ensure_ascii=False, indent=2), flush=True)
        finally:
            service.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    run(parser.parse_args().out.resolve())

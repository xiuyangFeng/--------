"""The back button of an open case (user request 10-01): it returns to the workbench page the case was opened from
(首页 / 任务列表 / 队列 / 回收站), the brand returns to the home, and the offline report has neither."""
from __future__ import annotations

from tests.test_v2_lane_c import _run
from tests.test_v2_workspace_js import _run as _run_ws


def test_back_button_returns_to_the_page_the_case_was_opened_from():
    out = _run(r"""
      canned['/api/jobs'] = {body: {jobs: [jobRecord('A')]}};
      await boot();
      const back = () => byClass(app(), 'top-back')[0];
      const home = {hidden: back().hidden};
      await hashTo('#/job/A', 300);
      const fromHome = {hidden: back().hidden, text: textOf(back())};
      fire(back(), 'click'); await wait(10);
      const afterHome = location.hash;
      await hashTo('#/tasks', 150);
      await hashTo('#/job/A', 300);
      const fromTasks = {text: textOf(back())};
      fire(back(), 'click'); await wait(10);
      const afterTasks = location.hash;
      await hashTo('#/job/A', 300);
      fire(byClass(app(), 'brand')[0], 'click'); await wait(10);
      done({home, fromHome, afterHome, fromTasks, afterTasks, brand: location.hash});
    """)
    assert out["errors"] == [], out["errors"]
    assert out["home"]["hidden"] is True
    assert out["fromHome"] == {"hidden": False, "text": "返回工作台"} and out["afterHome"] == "#/"
    assert out["fromTasks"]["text"] == "返回任务列表" and out["afterTasks"] == "#/tasks"
    assert out["brand"] == "#/"


def test_offline_report_has_no_back_button():
    out = _run_ws(r"""
      const m = manifestFor('A');
      const s1 = new Element('script'); s1.id = 'wssv2-manifest'; s1.textContent = JSON.stringify(m); body.appendChild(s1);
      const s2 = new Element('script'); s2.id = 'wssv2-offline'; s2.textContent = JSON.stringify({bookmarks: [], exported_at: '2026-10-01T09:00:00+08:00', view: {}}); body.appendChild(s2);
      const appEl = new Element('div'); appEl.id = 'ws-app'; body.appendChild(appEl);
      await boot();
      await wait(100);
      const b = byClass(app(), 'top-back')[0];
      done({hidden: !b || b.hidden, brandLink: String(byClass(app(), 'brand')[0].className).includes('is-link')});
    """, offline=True)
    assert out["errors"] == [], out["errors"]
    assert out["hidden"] is True and out["brandLink"] is False

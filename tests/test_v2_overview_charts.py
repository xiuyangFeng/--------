"""Home overview band and cohort charts (the 09-30 layout pass after the second-phase merge).

Pure parts: the home statistics (counts, the last 14 local days, running jobs, 待处理 split, latest failure), the
cohort card statistics (results by kind, review, 「关注」 findings, diameter quartiles) and the round axis ticks.  Shell
parts on the lane C harness: four overview cards across the home, the day columns, and the cohort page with its four
cards, the share-scaled distribution (reference on the same axis) and the scatter against the lumen diameter.
"""
from __future__ import annotations

from tests.test_v2_lane_c import _run

ROWS = r"""
canned['/api/jobs/export'] = {body: {columns: [], rows: [
  {job_id: 'A', case_id: 'CASE_A', release_id: 'X5D', family: 'wall', review_status: 'unreviewed', wss_p99_pa: 16.6, max_diameter_mm: 72.2, findings_attention: 1, created_at: '2026-09-20T08:00:00+08:00'},
  {job_id: 'B', case_id: 'CASE_B', release_id: 'X5D', family: 'wall', review_status: 'reviewed', wss_p99_pa: 20.7, max_diameter_mm: 38.6, findings_attention: 0, created_at: '2026-09-18T08:00:00+08:00'},
  {job_id: 'C', case_id: 'CASE_C', release_id: 'X5D', family: 'wall', review_status: 'unreviewed', wss_p99_pa: 25.1, max_diameter_mm: 30.2, findings_attention: 0, created_at: '2026-09-17T08:00:00+08:00'},
  {job_id: 'V', case_id: 'CASE_V', release_id: 'PF6', family: 'volume', review_status: 'unreviewed', speed_p99_m_s: 1.37, max_diameter_mm: 72.2, created_at: '2026-09-21T08:00:00+08:00'}],
  population: {X5D: {values_pa: [10, 12, 14, 16, 18, 20, 22], case_count: 136}}}};
"""


def test_pure_home_stats_cohort_stats_and_ticks():
    out = _run(r"""
      await boot();
      const R = ns.rail, AD = ns.admin, now = Date.parse('2026-09-30T12:00:00+08:00');
      const jobs = [syn(1, {status: 'done', created_at: '2026-09-30T08:00:00+08:00'}), syn(2, {status: 'running', phase: '阶段 B', created_at: '2026-09-29T08:00:00+08:00'}),
        syn(3, {status: 'queued', created_at: '2026-09-29T09:00:00+08:00'}), syn(4, {status: 'awaiting_confirmation', created_at: '2026-09-17T08:00:00+08:00'}),
        syn(5, {status: 'failed', created_at: '2026-09-28T08:00:00+08:00'}), syn(6, {status: 'failed', created_at: '2026-09-30T09:00:00+08:00'}),
        syn(7, {status: 'done', created_at: '2026-09-10T08:00:00+08:00'})];
      const st = R.homeStats(jobs, now);
      const rows = [{release_id: 'X', family: 'wall', review_status: 'reviewed', findings_attention: 2, max_diameter_mm: 30},
        {release_id: 'X', family: 'wall', review_status: 'unreviewed', findings_attention: 0, max_diameter_mm: 40},
        {release_id: 'Y', family: 'volume', review_status: 'unreviewed', max_diameter_mm: 50}, {release_id: 'X', family: 'wall', review_status: 'unreviewed', findings_attention: 1, max_diameter_mm: 60}];
      const cs = AD.cohortStats(rows);
      const t1 = AD.niceTicks(5.21, 57.6, 5), t2 = AD.niceTicks(0, 0.33, 4), t3 = AD.niceTicks(0.21, 0.83, 4);
      done({counts: [st.today, st.active, st.todo, st.failed], days: st.days.length, first: [st.days[0].month, st.days[0].day], last: st.days[13].today,
        perDay: st.days.map(d => d.n), failedToday: st.days[13].failed, running: st.running.map(j => j.id), queued: st.queued, split: [st.confirm, st.review],
        lastFailed: st.lastFailed && st.lastFailed.id, recent: st.recent,
        cs: {n: cs.n, kinds: cs.kinds.map(k => k.n), labels: cs.kinds.map(k => k.label), reviewed: cs.reviewed, attention: cs.attention, known: cs.findingsKnown, med: cs.dMedian, q: [cs.dQ1, cs.dQ3], range: [cs.dMin, cs.dMax]},
        t1: [t1.ticks, t1.fmt(t1.ticks[1])], t2: [t2.ticks, t2.decimals], t3: [t3.step, t3.fmt(0.5)]});
    """)
    assert out["errors"] == [], out["errors"]
    assert out["counts"] == [2, 2, 3, 2]                     # 待处理 = 1 to confirm + 2 finished, not reviewed
    assert out["days"] == 14 and out["first"] == [9, 17] and out["last"] is True
    assert out["perDay"][0] == 1 and out["perDay"][-1] == 2 and out["perDay"][-2] == 2 and sum(out["perDay"]) == 6   # 09-10 is outside
    assert out["failedToday"] == 1 and out["recent"] == 6
    assert out["running"] == ["S002"] and out["queued"] == 1 and out["split"] == [1, 2] and out["lastFailed"] == "S006"
    cs = out["cs"]
    assert cs["n"] == 4 and cs["kinds"] == [3, 1] and cs["reviewed"] == 1 and cs["attention"] == 2 and cs["known"] == 3   # a row without the column is not counted
    assert cs["med"] == 45 and cs["q"] == [37.5, 52.5] and cs["range"] == [30, 60]
    assert out["t1"][0] == [0, 10, 20, 30, 40, 50, 60] and out["t1"][1] == "10"
    assert out["t2"][0] == [0, 0.1, 0.2, 0.3, 0.4] and out["t2"][1] == 1
    assert out["t3"][0] == 0.2 and out["t3"][1] == "0.5"


def test_home_band_and_cohort_page_draw():
    out = _run(ROWS + r"""
      canned['/api/jobs'] = {body: {jobs: [jobRecord('A', {created_at: new Date().toISOString()}), jobRecord('R', {status: 'running', phase: '阶段 A', display_name: 'RUN_ME'}),
        jobRecord('C', {status: 'awaiting_confirmation'}), jobRecord('D', {status: 'failed', display_name: 'BAD_ONE'})]}};
      await boot();
      const band = byClass(app(), 'home-kpis')[0];
      const cardsHome = byClass(band, 'ov-card').map(e => textOf(byClass(e, 'ov-label')[0]));
      const cols = byClass(band, 'ov-col').length, today = byClass(band, 'ov-col').filter(e => e.className.includes('today')).length;
      const runText = textOf(byClass(band, 'ov-card')[1]), failText = textOf(byClass(band, 'ov-card')[3]);
      const segs = byClass(band, 'ov-seg').map(e => e.className);
      const tipBefore = byClass(band, 'chart-tip').map(e => e.hidden);
      fire(byClass(band, 'ov-col')[13], 'pointerenter'); await wait(5);
      const tipAfter = byClass(band, 'chart-tip').map(e => [e.hidden, textOf(e)]);
      await hashTo('#/cohort', 150);
      const kp = byClass(byClass(app(), 'wsc-kpis')[0], 'ov-card').map(e => textOf(byClass(e, 'ov-label')[0]) + '=' + textOf(byClass(e, 'ov-value')[0]));
      const bars = walk(app(), e => String(e.attrs.class || '') === 'wsc-bar').length, ref = walk(app(), e => String(e.attrs.class || '') === 'wsc-ref').length;
      const area = walk(app(), e => String(e.attrs.class || '') === 'wsc-ref-area').length;
      const yTicks = walk(byClass(app(), 'wsc-dist')[0], e => String(e.attrs.class || '') === 'wsc-tick').map(textOf).filter(t => t.endsWith('%'));
      const pts = walk(app(), e => String(e.attrs.class || '') === 'wsc-pt-hit');
      const titles = byClass(app(), 'wsc-card-title').map(textOf), metricSel = walk(byClass(app(), 'wsc-filters')[0], e => e.attrs['aria-label'] === '图上的量').length;
      fire(pts[0], 'click'); await wait(20);
      done({cardsHome, cols, today, runText, failText, segs, tipBefore, tipAfter, kp, bars, ref, area, yTicks, pts: pts.length, titles, metricSel, hash: location.hash});
    """)
    assert out["errors"] == [], out["errors"]
    assert out["cardsHome"] == ["今日上传", "进行中", "待处理", "失败"]
    assert out["cols"] == 14 and out["today"] == 1
    assert "RUN_ME" in out["runText"] and "阶段 A" in out["runText"] and "BAD_ONE" in out["failText"]
    assert out["segs"] == ["ov-seg seg-confirm", "ov-seg seg-review"]
    assert out["tipBefore"] == [True] and out["tipAfter"][0][0] is False and out["tipAfter"][0][1].startswith("1 个今天")
    assert out["kp"][0] == "已完成结果=4" and out["kp"][1] == "已复核=25%" and out["kp"][2] == "有「关注」发现=1" and out["kp"][3].startswith("管腔最大直径 · 中位=55.4")
    assert out["bars"] >= 1 and out["ref"] == 1 and out["area"] == 1
    assert out["yTicks"][0] == "0%" and all(t[:-1].isdigit() for t in out["yTicks"])   # one axis: both groups as shares
    assert out["pts"] == 3 and out["titles"] == ["分布", "与管腔最大直径"] and out["metricSel"] == 1
    assert out["hash"].startswith("#/job/")

"""C5: the owner-level stream carries every transition of the owner's jobs and nothing else."""
from __future__ import annotations

import json

from tests._c_helpers import Service, finished


def _read_event(stream):
    name, data = None, None
    while True:
        line = stream.readline()
        if not line:
            return None, None
        line = line.decode("utf-8").rstrip("\r\n")
        if line.startswith(":"):
            continue
        if line.startswith("event: "):
            name = line[7:]
        elif line.startswith("data: "):
            data = json.loads(line[6:])
        elif line == "" and name is not None:
            return name, data


def test_owner_stream_only_carries_own_jobs(tmp_path):
    service = Service(tmp_path)
    api = service.api()
    stream_conn = service.api(timeout=10)
    try:
        cookie, headers, owner = service.session(api)
        mine = finished(service.manager, owner=owner, case_id="MINE")
        stream_conn.request("GET", "/api/events", headers={"Cookie": cookie})
        stream = stream_conn.getresponse()
        assert stream.status == 200 and stream.getheader("Content-Type").startswith("text/event-stream")
        name, hello = _read_event(stream)
        assert name == "hello" and hello["owner_jobs"] == 1 and hello["active_jobs"] == 0
        finished(service.manager, owner="someone-else", case_id="THEIRS")   # must not appear
        service.manager.review(mine["id"], owner, {"version": mine["version"], "decision": "approve", "reviewer": "R"})
        name, event = _read_event(stream)
        assert name == "job" and event["job_id"] == mine["id"] and event["action"] == "review_approved"
        assert event["case_id"] == "MINE" and event["family"] == "wall" and event["review"] == "reviewed" and event["final"] is True
        record = service.manager.get(mine["id"], owner)
        service.manager.review(mine["id"], owner, {"version": record["version"], "decision": "reopen", "reviewer": "R", "note": "再看"})
        version = service.manager.get(mine["id"], owner)["version"]
        service.manager.delete(mine["id"], owner, {"version": version})
        seen = []
        while len(seen) < 2:
            name, event = _read_event(stream)
            seen.append((name, event["action"], event["status"]))
        assert seen[0] == ("job", "review_reopened", "done") and seen[1] == ("job", "deleted", "deleted")
        assert all(job["case_id"] != "THEIRS" for _, _, job in [(None, None, {"case_id": e[1]}) for e in seen])
    finally:
        stream_conn.close(); api.close(); service.close()

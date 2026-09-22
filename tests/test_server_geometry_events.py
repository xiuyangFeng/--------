"""HTTP round trip for the geometry endpoint (ETag / 304) and the server-sent event stream."""
import http.client
import json
import threading

from wss_deploy.jobs import JobManager
from wss_deploy.server import ServiceHTTPServer, SessionStore


def _stage_a_stub(*_args, **_kwargs):
    return {
        "stage": "A", "created_at": "2026-09-20 10:00:00", "stl": "input.stl", "input_sha256": "a" * 64,
        "input_check": {"ok": True, "status": "pass", "clean_stl": "input_clean_mm.stl", "orientation_source": "unknown_stl"},
        "centerline": {"hard_pass": True, "openings": [{"opening_id": 0, "radius_mm": 9.0, "role": "inlet"}]},
        "preview": {"vertices": [[0, 0, 0], [1, 0, 0], [0, 1, 0]] * 2000, "faces": [[0, 1, 2]] * 2000, "display_only": True},
        "proposal": {"auto_ok": True, "confirmation_required": True, "confidence": 0.5,
                     "mapping": {"3": "out-le", "4": "out-li", "5": "out-re", "6": "out-ri"},
                     "endpoints": [{"segment_id": 3, "kind": "outlet", "auto_name": "out-le"}],
                     "preview_polylines": [{"segment_id": 0, "xyz": [[0, 0, 0], [1, 1, 1]] * 500}], "flags": []},
        "timing_s": {"ingest": 0.1, "centerline": 1.0},
    }


def _part(boundary, name, value, filename=None):
    disposition = f'form-data; name="{name}"'
    if filename:
        disposition += f'; filename="{filename}"'
    header = f"--{boundary}\r\nContent-Disposition: {disposition}\r\n"
    if filename:
        header += "Content-Type: application/octet-stream\r\n"
    return (header + "\r\n").encode() + (value if isinstance(value, bytes) else value.encode()) + b"\r\n"


def _read_event(stream):
    """Read one SSE event (skipping keepalive comments); returns (name, payload)."""
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


def test_geometry_etag_and_event_stream(tmp_path):
    manager = JobManager(tmp_path / "jobs", stage_a_fn=_stage_a_stub, stage_b_fn=lambda *_a, **_k: {},
                         mapping_validator=lambda *_a, **_k: [])
    sessions = SessionStore(tmp_path / "sessions.json", shared=False)
    server = ServiceHTTPServer(("127.0.0.1", 0), manager, sessions)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    api = http.client.HTTPConnection("127.0.0.1", server.server_port, timeout=5)
    stream = None
    try:
        api.request("GET", "/api/session")
        response = api.getresponse(); session = json.loads(response.read())
        cookie = response.getheader("Set-Cookie").split(";", 1)[0]
        boundary = "geom-test"
        body = b"".join([_part(boundary, "stl", b"solid", "case.stl"), f"--{boundary}--\r\n".encode()])
        api.request("POST", "/api/jobs", body=body, headers={"Cookie": cookie, "X-CSRF-Token": session["csrf_token"],
                                                           "Content-Type": f"multipart/form-data; boundary={boundary}"})
        response = api.getresponse(); job = json.loads(response.read())["job"]
        assert response.status == 201 and job["status"] == "queued"

        # Open the live stream before the job runs: first event is the current snapshot.
        stream_conn = http.client.HTTPConnection("127.0.0.1", server.server_port, timeout=10)
        stream_conn.request("GET", f"/api/jobs/{job['id']}/events", headers={"Cookie": cookie})
        stream = stream_conn.getresponse()
        assert stream.status == 200 and stream.getheader("Content-Type").startswith("text/event-stream")
        name, payload = _read_event(stream)
        assert name == "snapshot" and payload["status"] == "queued" and payload["final"] is False

        assert manager.run_next(timeout=0.5)  # executes the stubbed stage A synchronously
        seen = []
        while True:
            name, payload = _read_event(stream)
            assert name == "job"
            seen.append(payload["action"])
            if payload["status"] == "awaiting_confirmation" and payload["action"] == "finished":
                break
        assert seen[0] == "started"

        # Geometry is served separately from the light snapshot, with a weak ETag.
        api.request("GET", f"/api/jobs/{job['id']}", headers={"Cookie": cookie})
        response = api.getresponse(); snapshot = json.loads(response.read())
        assert "preview" not in snapshot["a"]
        api.request("GET", f"/api/jobs/{job['id']}/geometry", headers={"Cookie": cookie})
        response = api.getresponse(); geometry = json.loads(response.read()); etag = response.getheader("ETag")
        assert response.status == 200 and etag and geometry["available"] and len(geometry["preview"]["faces"]) == 2000
        api.request("GET", f"/api/jobs/{job['id']}/geometry", headers={"Cookie": cookie, "If-None-Match": etag})
        response = api.getresponse(); response.read()
        assert response.status == 304

        # Cancelling makes the job final: the stream delivers it and the server closes the connection.
        api.request("POST", f"/api/jobs/{job['id']}/cancel", body=json.dumps({"version": snapshot["version"]}),
                    headers={"Cookie": cookie, "X-CSRF-Token": session["csrf_token"], "Content-Type": "application/json"})
        response = api.getresponse(); assert response.status == 200; response.read()
        name, payload = _read_event(stream)
        assert name == "job" and payload["final"] is True and payload["status"] == "cancelled"
        assert _read_event(stream) == (None, None)
        assert job["id"] not in manager._subscribers
    finally:
        if stream is not None:
            stream.close()
        api.close()
        server.shutdown(); server.server_close(); manager.close()

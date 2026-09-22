import http.client
import json
import threading

from wss_deploy.jobs import JobManager
from wss_deploy.server import ServiceHTTPServer, SessionStore


def _part(boundary, name, value, filename=None):
    disposition = f'form-data; name="{name}"'
    if filename:
        disposition += f'; filename="{filename}"'
    header = f"--{boundary}\r\nContent-Disposition: {disposition}\r\n"
    if filename:
        header += "Content-Type: application/octet-stream\r\n"
    return (header + "\r\n").encode() + (value if isinstance(value, bytes) else value.encode()) + b"\r\n"


def test_batch_upload_and_history_query_round_trip(tmp_path):
    manager = JobManager(tmp_path / "jobs", stage_a_fn=lambda *_a, **_k: {}, stage_b_fn=lambda *_a, **_k: {})
    sessions = SessionStore(tmp_path / "sessions.json", shared=False)
    server = ServiceHTTPServer(("127.0.0.1", 0), manager, sessions)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    connection = http.client.HTTPConnection("127.0.0.1", server.server_port, timeout=5)
    try:
        connection.request("GET", "/api/session")
        response = connection.getresponse()
        session = json.loads(response.read())
        cookie = response.getheader("Set-Cookie").split(";", 1)[0]
        boundary = "p2-test-boundary"
        body = b"".join([
            _part(boundary, "stl", b"one", "one.stl"),
            _part(boundary, "stl", b"two", "two.stl"),
            _part(boundary, "device", "cpu"),
            _part(boundary, "seed_count", "1"),
            _part(boundary, "metadata_json", json.dumps([{"patient_id": "anon-1", "tags": ["AAA"]}, {}])),
            f"--{boundary}--\r\n".encode(),
        ])
        connection.request("POST", "/api/jobs/batch", body=body, headers={
            "Cookie": cookie, "X-CSRF-Token": session["csrf_token"],
            "Content-Type": f"multipart/form-data; boundary={boundary}",
        })
        response = connection.getresponse()
        result = json.loads(response.read())
        assert response.status == 201 and result["created_count"] == 2
        assert result["jobs"][0]["patient_id"] == "anon-1"
        connection.request("GET", "/api/jobs?patient_id=anon-1&page_size=5", headers={"Cookie": cookie})
        response = connection.getresponse()
        history = json.loads(response.read())
        assert response.status == 200 and history["total"] == 1
        assert history["jobs"][0]["compute"]["device"] == "cpu"
    finally:
        connection.close()
        server.shutdown()
        server.server_close()
        manager.close()

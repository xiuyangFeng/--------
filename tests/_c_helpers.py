"""Shared fixtures for the C-class backend tests (cases, duplicates, trash, users, sidecars, exports)."""
from __future__ import annotations

import http.client
import json
import threading

from wss_deploy.jobs import JobManager
from wss_deploy.server import ServiceHTTPServer, SessionStore

META_BLOCK = '<!--WSS_META_START--><script id="wss-report-meta" type="application/json">{}</script><!--WSS_META_END-->'
MAPPING = {"3": "out-le", "4": "out-li", "5": "out-re", "6": "out-ri"}


def stage_a_stub(*_args, **_kwargs):
    return {"stage": "A", "created_at": "2026-09-21 10:00:00", "stl": "input.stl", "input_sha256": "a" * 64,
            "input_check": {"ok": True, "status": "pass", "clean_stl": "input_clean_mm.stl", "orientation_source": "unknown_stl"},
            "centerline": {"hard_pass": True, "openings": [{"opening_id": 0, "radius_mm": 9.0, "role": "inlet"}]},
            "proposal": {"auto_ok": True, "confirmation_required": True, "confidence": 0.5, "mapping": dict(MAPPING), "endpoints": [], "flags": []},
            "timing_s": {"ingest": 0.1, "centerline": 1.0}}


def stage_b_stub(job_dir, mapping, release, **_kwargs):
    return {"peak": {"p99_pa": 1.0}, "timing_s": {"features": 0.1, "inference": 0.2}, "fields": {"wss": {"units": "Pa"}}, "run_identity": "r" * 64}


class FakeRelease:
    registry_id = "REL_A"
    registry_fingerprint = "f" * 64
    registry_contract = {"protocol": "single_frame_wss", "family": "wall_wss_v1"}


def manager(tmp_path, *, release=None, clock=None, stage_a=None, stage_b=None):
    return JobManager(tmp_path, release=release or FakeRelease(), stage_a_fn=stage_a or stage_a_stub, stage_b_fn=stage_b or stage_b_stub,
                      mapping_validator=lambda *_a, **_k: [], clock=clock)


def finished(mgr, owner="owner", case_id="case", *, content=b"solid", with_files=True, summary_extra=None, protocol="single_frame_wss", **create):
    job = mgr.create(owner, content=content, filename=f"{case_id}.stl", case_id=case_id, **create)
    with mgr.lock:
        record = mgr.jobs[job["id"]]
        release = dict(record.get("model_release") or {})
        release["contract"] = {"protocol": protocol}
        summary = {"peak": {"p99_pa": 1.0}, "fields": {"wss": {"units": "Pa"}} if protocol == "single_frame_wss" else {"pressure": {"units": "Pa"}, "velocity": {"units": "m/s"}}}
        record.update(status="done", stage="B", phase="计算完成", mapping=dict(MAPPING), model_release=release, run_identity="r" * 64,
                      a={"stage": "A", "input_sha256": record.get("input_sha256"), "input_check": {"ok": True, "status": "pass", "clean_stl": "input_clean_mm.stl"},
                         "centerline": {"hard_pass": True, "openings": []}, "proposal": {"mapping": dict(MAPPING)}, "timing_s": {"centerline": 1.0}},
                      summary=summary)
        mgr._event(record, "finished")
        job_dir = mgr.root / job["id"]
        if with_files:
            (job_dir / "centerline").mkdir(exist_ok=True)
            (job_dir / "centerline" / "atlas.npz").write_bytes(b"npz")
            full = {"case_id": case_id, "input_sha256": record.get("input_sha256"), "run_identity": "r" * 64, "peak": {"p99_pa": 1.0, "max_pa": 3.0},
                    "wss_field_pa": {"mean": 0.5, "area_frac_low": 0.1, "area_low_mm2": 10.0, "area_frac_high": 0.2, "area_high_mm2": 20.0, "area_frac_very_high": 0.05},
                    "per_branch": {"主动脉": {"wss_p99_pa": 2.5}, "左髂内": {"wss_p99_pa": 4.5}},
                    "reference_assessment": {"population": {"percentile": 32.4}}, "quality": {"level": "good", "label": "模型集成稳定"},
                    "fields": summary["fields"], "time_axis": [{"index": 0, "step": 1162, "time_s": 0.21}],
                    "findings": {"schema_version": "wss-deploy.findings/v1", "items": [
                        {"id": "F1", "kind": "high_wss_cluster", "label": "左髂内高 WSS 区", "branch": "左髂内", "value": 21.9, "units": "Pa", "rank": 1, "severity": "attention", "definition": "≥ p99 的连通簇"},
                        {"id": "F2", "kind": "low_wss_cluster", "label": "主动脉低 WSS 区", "branch": "主动脉", "value": 0.2, "units": "Pa", "rank": 2, "severity": "attention", "definition": "< 0.4 Pa"},
                        {"id": "F3", "kind": "pressure_drop", "label": "主动脉压差", "branch": "主动脉", "value": 266.6, "units": "Pa", "rank": 3, "severity": "note", "definition": "近远端压差"}]},
                    "model_release": release, "timing_s": {"total": 30.0}}
            full.update(summary_extra or {})
            (job_dir / "summary.json").write_text(json.dumps(full, ensure_ascii=False), encoding="utf-8")
            (job_dir / "report.html").write_text("<html><body>" + META_BLOCK + "</body></html>", encoding="utf-8")
            (job_dir / "run_manifest.json").write_text(json.dumps({"outputs": {"summary.json": {"path": "summary.json"}, "report.html": {"path": "report.html"}}}), encoding="utf-8")
            (job_dir / "field.npz").write_bytes(b"PK\x03\x04fake")
    return mgr.get(job["id"], owner)


class Service:
    """Threaded test server; ``api()`` opens a connection, ``session()`` returns (cookie, headers)."""
    def __init__(self, tmp_path, *, shared=False, token=None, users=None, mgr=None, clock=None):
        self.manager = mgr or manager(tmp_path / "jobs", clock=clock)
        self.sessions = SessionStore(tmp_path / "sessions.json", shared=shared, token=token, users=users)
        self.server = ServiceHTTPServer(("127.0.0.1", 0), self.manager, self.sessions)
        threading.Thread(target=self.server.serve_forever, daemon=True).start()
        self.port = self.server.server_port
    def api(self, timeout=5):
        return http.client.HTTPConnection("127.0.0.1", self.port, timeout=timeout)
    def session(self, api):
        api.request("GET", "/api/session")
        response = api.getresponse(); session = json.loads(response.read())
        cookie = response.getheader("Set-Cookie").split(";", 1)[0]
        headers = {"Cookie": cookie, "X-CSRF-Token": session["csrf_token"], "Content-Type": "application/json"}
        return cookie, headers, self.sessions.lookup(cookie)[1]["owner"]
    def login(self, api, body):
        api.request("POST", "/api/session", body=json.dumps(body), headers={"Content-Type": "application/json"})
        response = api.getresponse(); payload = json.loads(response.read())
        cookie = (response.getheader("Set-Cookie") or "").split(";", 1)[0]
        headers = {"Cookie": cookie, "X-CSRF-Token": payload.get("csrf_token") or "", "Content-Type": "application/json"}
        return response.status, payload, cookie, headers
    def close(self):
        self.server.shutdown(); self.server.server_close(); self.manager.close()


def call(api, method, path, headers, body=None):
    api.request(method, path, body=json.dumps(body) if isinstance(body, (dict, list)) else body, headers=headers)
    response = api.getresponse(); raw = response.read()
    try: payload = json.loads(raw)
    except ValueError: payload = raw
    return response.status, payload, response


def multipart(boundary, parts):
    """``parts``: list of (name, value, filename|None)."""
    body = b""
    for name, value, filename in parts:
        disposition = f'form-data; name="{name}"' + (f'; filename="{filename}"' if filename else "")
        header = f"--{boundary}\r\nContent-Disposition: {disposition}\r\n" + ("Content-Type: application/octet-stream\r\n" if filename else "")
        body += (header + "\r\n").encode() + (value if isinstance(value, bytes) else value.encode()) + b"\r\n"
    return body + f"--{boundary}--\r\n".encode(), f"multipart/form-data; boundary={boundary}"

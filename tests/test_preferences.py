"""C6: per-owner preferences are replaced whole, size-capped and isolated per owner."""
from __future__ import annotations

import json

from wss_deploy.server import PreferenceStore, owner_key
from tests._c_helpers import Service, call


def test_owner_key_is_file_safe():
    assert owner_key("alice") == "alice"
    hashed = owner_key("Yx/2+randomsession==")
    assert len(hashed) == 32 and hashed.isalnum()


def test_store_replaces_whole_document_and_caps_size(tmp_path):
    store = PreferenceStore(tmp_path / "preferences")
    assert store.get("alice") == {}
    store.put("alice", {"upload": {"units": "mm"}})
    store.put("alice", {"report_defaults": {"wall": {"units": "dyn"}}})
    assert store.get("alice") == {"report_defaults": {"wall": {"units": "dyn"}}}
    assert store.get("bob") == {}
    try:
        store.put("alice", {"blob": "x" * 17000})
    except Exception as exc:
        assert getattr(exc, "status", None) == 413
    else:
        raise AssertionError("oversized preferences must be refused")


def test_http_preferences_roundtrip_requires_csrf(tmp_path):
    service = Service(tmp_path)
    api = service.api()
    try:
        cookie, headers, owner = service.session(api)
        status, payload, _ = call(api, "GET", "/api/preferences", {"Cookie": cookie})
        assert status == 200 and payload == {"preferences": {}}
        status, payload, _ = call(api, "PUT", "/api/preferences", {"Cookie": cookie, "Content-Type": "application/json"}, {"upload": {"units": "mm"}})
        assert status == 403
        status, payload, _ = call(api, "PUT", "/api/preferences", headers, {"upload": {"units": "mm"}, "presets": {"wall": []}})
        assert status == 200 and payload["preferences"]["upload"]["units"] == "mm"
        status, payload, _ = call(api, "GET", "/api/preferences", {"Cookie": cookie})
        assert payload["preferences"]["presets"] == {"wall": []}
        assert (service.manager.root / "preferences" / f"{owner_key(owner)}.json").is_file()
    finally:
        api.close(); service.close()

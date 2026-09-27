"""v0.15.8: the display preview is a connected vertex-clustered surface, and older jobs are upgraded on read."""
import json
from pathlib import Path

import numpy as np

from wss_deploy.jobs import JobManager
from wss_deploy.preview import MAX_PREVIEW_FACES, display_mesh, needs_upgrade, preview_payload
from wss_features.stl import write_binary_stl


def _tube(n_around=240, n_along=120, radius=10.0, length=200.0):
    """Open cylinder along z: n_around × n_along quads → 2 × n_around × (n_along − 1) triangles."""
    ang = np.arange(n_around) * 2 * np.pi / n_around
    z = np.linspace(0.0, length, n_along)
    verts = np.array([[radius * np.cos(a), radius * np.sin(a), zz] for zz in z for a in ang])
    faces = []
    for j in range(n_along - 1):
        for i in range(n_around):
            a, b = j * n_around + i, j * n_around + (i + 1) % n_around
            c, d = a + n_around, b + n_around
            faces += [[a, b, d], [a, d, c]]
    return verts, np.asarray(faces)


def _manifold_fraction(faces):
    edges = np.sort(np.concatenate([faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]]), axis=1)
    _, counts = np.unique(edges, axis=0, return_counts=True)
    return float((counts == 2).mean())


def test_small_meshes_are_kept_and_large_ones_become_a_connected_surface_within_budget():
    v, f = _tube(n_around=24, n_along=10)
    sv, sf, method = display_mesh(v, f)
    assert method == "full" and len(sf) == len(f) and np.allclose(sv, v)
    v, f = _tube()
    assert len(f) > MAX_PREVIEW_FACES
    dv, df, method = display_mesh(v, f)
    assert method == "vertex_cluster" and 0.5 * MAX_PREVIEW_FACES < len(df) <= MAX_PREVIEW_FACES
    # connected: interior edges are shared by two triangles (the old every-k-th-triangle preview shared none)
    assert _manifold_fraction(df) > 0.95 and len(dv) < 0.7 * len(df)
    assert np.abs(dv.min(0) - v.min(0)).max() < 2.0 and np.abs(dv.max(0) - v.max(0)).max() < 2.0
    payload = preview_payload(v, f)
    assert payload["method"] == "vertex_cluster" and payload["source_faces"] == len(f) and payload["display_only"] is True
    assert needs_upgrade({"vertices": [], "faces": [], "display_faces": MAX_PREVIEW_FACES}) is True
    assert needs_upgrade(payload) is False and needs_upgrade({"display_faces": 500}) is False and needs_upgrade(None) is False


def test_an_older_job_gets_a_connected_preview_on_read_and_keeps_its_records(tmp_path):
    v, f = _tube()

    def stage_a(path, job_dir, **kwargs):
        job_dir = Path(job_dir)
        write_binary_stl(job_dir / "input_clean_mm.stl", v, f)
        sub = f[np.linspace(0, len(f) - 1, MAX_PREVIEW_FACES, dtype=np.int64)]   # the pre-v0.15.8 scatter
        used, inverse = np.unique(sub.reshape(-1), return_inverse=True)
        return {"stage": "A", "created_at": "2026-09-27T17:36:26+08:00",
                "input_check": {"status": "pass", "ok": True, "orientation_source": "unknown_stl", "clean_stl": "input_clean_mm.stl"},
                "centerline": {"hard_pass": True}, "proposal": {"auto_ok": False, "mapping": {}, "preview_polylines": []},
                "preview": {"vertices": v[used].round(3).tolist(), "faces": inverse.reshape(-1, 3).tolist(),
                            "display_faces": MAX_PREVIEW_FACES, "display_only": True},
                "timing_s": {}}

    manager = JobManager(tmp_path / "jobs", stage_a_fn=stage_a, stage_b_fn=lambda *a, **k: {}, mapping_validator=lambda *_: [])
    job = manager.create("owner", content=b"stl", filename="t.stl")
    manager.run_next()
    stage_a_before = (tmp_path / "jobs" / job["id"] / "stage_a.json").read_bytes()
    first = manager.geometry(job["id"], "owner")
    assert first["preview"]["method"] == "vertex_cluster" and first["preview_rev"] == "vertex_cluster"
    assert _manifold_fraction(np.asarray(first["preview"]["faces"])) > 0.95
    cache = list((tmp_path / "jobs" / job["id"]).rglob("preview-*.json"))
    assert len(cache) == 1
    second = manager.geometry(job["id"], "owner")
    assert second["preview"] == json.loads(cache[0].read_text())                       # served from the cache
    assert (tmp_path / "jobs" / job["id"] / "stage_a.json").read_bytes() == stage_a_before   # records untouched

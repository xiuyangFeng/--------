"""Orchestration: stage A (ingest + centreline + naming proposal, CPU) -> [human confirms outlets] -> stage B (geometry + inference + report)."""
from __future__ import annotations
import contextlib, hashlib, json, os, secrets, time, datetime, shutil
from pathlib import Path
import numpy as np
from wss_features import contract as feature_contract
from wss_features.atlas import Atlas
from wss_features.stl import load_stl
from . import centerline as CL, geometry as G, metrics as M, report as R
from .ingest import ingest
from .infer import Release
from .paths import OUTLET_CN
from .io_utils import file_sha256, portable_job_path, resolve_job_path
from .schema import (build_results, field_descriptor, model_release_metadata,
                     single_frame_time_axis, write_run_manifest, wss_compatibility)
from .schema import stable_run_identity, summary_display_name, summary_provenance
from .quality import ensemble_quality, model_count_phrase
from . import analysis as A, morphology as MORPH, narrative as NARR
from .cycle_fields import derived_display_arrays
from . import geometry_cache as GC
from .errors import ResourceError, classify as classify_error

DEFAULT_SMOOTH_MM = 1.0
DEFAULT_SPACING_MM = 0.5
CPU_THREADS_CAP = 32
# Source hash of the deployment-side geometry code, taken at import (the code this process runs).
_GEOMETRY_CODE = GC.source_hash("wss_deploy.geometry", "wss_deploy.geometry_cache")


# ----------------------------------------------------------------------------- v0.14 compute helpers
def default_cpu_threads() -> int:
    """Torch intra-op threads for CPU inference when the job does not set ``compute.threads``.

    torch defaults to one thread per core (192 on the service node), which made one X5D member take
    12.8 s instead of 1.1-1.5 s at 16-32 threads.  ``WSS_DEPLOY_CPU_THREADS`` overrides the cap of 32.
    """
    raw = os.environ.get("WSS_DEPLOY_CPU_THREADS", "").strip()
    if raw:
        try:
            value = int(raw)
            if value >= 1:
                return value
        except ValueError:
            pass
    return max(1, min(CPU_THREADS_CAP, os.cpu_count() or 1))


def inference_threads(device: str | None, threads: int | None) -> int | None:
    """Threads to set around inference: the job's explicit value, the CPU default, or None (GPU: untouched)."""
    if threads is not None:
        return int(threads)
    return default_cpu_threads() if str(device or "").lower() == "cpu" else None


@contextlib.contextmanager
def torch_threads(count: int | None):
    """Set torch's intra-op thread count for the block and restore it afterwards (no-op for None)."""
    old = None
    if count is not None:
        try:
            import torch
            old = torch.get_num_threads()
            torch.set_num_threads(int(count))
        except (ImportError, RuntimeError, ValueError):
            old = None
    try:
        yield
    finally:
        if old is not None:
            try:
                import torch
                torch.set_num_threads(old)
            except (ImportError, RuntimeError, ValueError):
                pass


def run_inference(release, case: dict) -> dict:
    """``release.predict`` with foreign failures typed (errors.classify): CUDA out-of-memory becomes a
    retryable ``ResourceError`` (retry_hint "cpu") that the job manager retries on the CPU."""
    try:
        return release.predict(case)
    except Exception as exc:
        typed = classify_error(exc)
        if isinstance(typed, ResourceError):
            raise typed from exc
        raise


def _tmp_sibling(path: Path) -> Path:
    """Hidden temporary name next to ``path`` keeping its suffix (pyvista picks the writer by suffix)."""
    path = Path(path)
    return path.with_name(f".{path.stem}.{os.getpid()}.{secrets.token_hex(4)}.tmp{path.suffix}")


@contextlib.contextmanager
def atomic_output(path: Path):
    """Yield a temporary path; on success it replaces ``path`` in one rename, on failure it is removed.

    Readers (the service, the report page, a bundle export) therefore never see a half-written output.
    A writer that produced nothing (e.g. VTP without pyvista) leaves ``path`` untouched.
    """
    path = Path(path)
    tmp = _tmp_sibling(path)
    try:
        yield tmp
        if tmp.exists():
            os.replace(tmp, path)
    finally:
        with contextlib.suppress(FileNotFoundError):
            tmp.unlink()


def write_text_atomic(path: Path, text: str) -> None:
    with atomic_output(path) as tmp:
        tmp.write_text(text, encoding="utf-8")


def save_npz_atomic(path: Path, **arrays) -> None:
    """``np.savez_compressed`` (same bytes as before) through a temporary file and ``os.replace``."""
    with atomic_output(path) as tmp:
        with open(tmp, "wb") as stream:
            np.savez_compressed(stream, **arrays)


class VertexInterpolation:
    """``report.interpolate_to_vertices`` / ``report.nearest_label`` for several fields on one (points, vertices)
    pair: the KD-tree is built once and the k=8 neighbours and Gaussian weights are computed once.

    The arithmetic is exactly that of the report helpers (same tree, same query, same weight and sum
    expressions), so every field is bit-identical to a separate call.
    """

    def __init__(self, pts, vertices, sigma_mm: float = 0.5, k: int = 8, max_dist_mm: float = 1.5):
        from scipy.spatial import cKDTree
        pts = np.asarray(pts, dtype=np.float64)
        vertices = np.asarray(vertices, dtype=np.float64)
        if pts.ndim != 2 or pts.shape[1] != 3 or not len(pts) or not np.isfinite(pts).all():
            raise ValueError("Interpolation points must be a non-empty finite (N, 3) array")
        if vertices.ndim != 2 or vertices.shape[1] != 3 or not np.isfinite(vertices).all():
            raise ValueError("Mesh vertices must be finite (N, 3) coordinates")
        if not np.isfinite(sigma_mm) or sigma_mm <= 0 or not np.isfinite(max_dist_mm) or max_dist_mm <= 0:
            raise ValueError("Interpolation sigma and coverage distance must be finite and positive")
        if not isinstance(k, (int, np.integer)) or k < 1:
            raise ValueError("Interpolation neighbour count must be a positive integer")
        self.n_points, self.vertices = len(pts), vertices
        self.tree = cKDTree(pts)
        count = min(k, len(pts))
        d, i = self.tree.query(vertices, k=count)
        d, i = d.reshape(len(vertices), count), i.reshape(len(vertices), count)
        with np.errstate(over="ignore", invalid="ignore"):
            exponent = -0.5 * ((d / sigma_mm) ** 2 - (d[:, :1] / sigma_mm) ** 2)
        weights = np.exp(exponent)
        weights[d > max_dist_mm] = 0.0
        self.index, self.weights, self.den = i, weights, weights.sum(axis=1)
        self._nearest = None

    def values(self, values) -> np.ndarray:
        values = np.asarray(values, dtype=np.float64)
        if values.shape != (self.n_points,) or not np.isfinite(values).all():
            raise ValueError("Interpolation requires one finite value per prediction point")
        out = np.full(len(self.vertices), np.nan, dtype=np.float64)
        np.divide((self.weights * values[self.index]).sum(axis=1), self.den, out=out, where=self.den > 0)
        return out.astype(np.float32)

    def labels(self, labels) -> np.ndarray:
        if self._nearest is None:
            _, self._nearest = self.tree.query(self.vertices, k=1)
        return np.asarray(labels)[self._nearest]


# ----------------------------------------------------------------------------- v0.14 geometry cache
def _clean_mesh(job_dir: Path, clean_stl: Path, cache: "GC.GeometryCache") -> tuple[np.ndarray, np.ndarray]:
    """``load_stl(clean_stl)`` (vertex de-duplication included) cached as arrays keyed by the file's SHA256."""
    from wss_features import stl as stl_module          # looked up at call time (tests substitute the reader)
    if not cache.active:
        return stl_module.load_stl(clean_stl)
    key = GC.key_of("clean-mesh", GC.feature_program_hash(), GC.file_sha256(clean_stl))
    hit = cache.load("mesh", key)
    if hit is not None:
        with contextlib.suppress(Exception):          # an entry of another layout is a miss, never an error
            return hit["vertices"], hit["faces"].astype(np.dtype(str(hit["faces_dtype"])))
    vertices, faces = stl_module.load_stl(clean_stl)
    small = faces.size and int(faces.max()) < 2 ** 31 and int(faces.min()) >= 0
    cache.save("mesh", key, {"vertices": vertices, "faces": faces.astype(np.int32) if small else faces,
                             "faces_dtype": np.asarray(faces.dtype.str)}, compress=True)
    return vertices, faces


def _resampled(vertices, faces, *, smooth_mm: float, spacing_mm: float, seed: int,
               cache: "GC.GeometryCache") -> tuple[np.ndarray, np.ndarray]:
    """``geometry.smooth_and_resample`` through the cache (key: mesh arrays, parameters, seed, code)."""
    if not cache.active:
        return G.smooth_and_resample(vertices, faces, smooth_mm=smooth_mm, spacing_mm=spacing_mm, seed=seed)
    key = GC.key_of("resample", _GEOMETRY_CODE, GC.feature_program_hash(), np.asarray(vertices, np.float64),
                    np.asarray(faces), float(smooth_mm), float(spacing_mm), int(seed))
    hit = cache.load("resample", key)
    if hit is not None and {"smoothed", "points"} <= set(hit):
        return hit["smoothed"], hit["points"]
    smoothed, points = G.smooth_and_resample(vertices, faces, smooth_mm=smooth_mm, spacing_mm=spacing_mm, seed=seed)
    cache.save("resample", key, {"smoothed": smoothed, "points": points})
    return smoothed, points


def _point_geometry_provider(cache: "GC.GeometryCache", lock=None):
    """``build_case`` hook: :func:`geometry.point_geometry` from the cache when the points and the (patched)
    atlas geometry match exactly, else computed and stored."""
    def provider(pts, atlas):
        key = GC.key_of("point-geometry", _GEOMETRY_CODE, GC.feature_program_hash(), np.asarray(pts),
                        np.asarray(atlas.table), np.asarray(atlas.tree_rows), list(atlas.columns))
        with (lock or contextlib.nullcontext()):
            hit = cache.load("pointgeom", key)
            if hit is not None:
                with contextlib.suppress(Exception):   # an entry of another layout is a miss, never an error
                    return _decode_point_geometry(hit)
            value = G.point_geometry(pts, atlas)
            cache.save("pointgeom", key, _encode_point_geometry(value))
            return value
    return provider


def _encode_point_geometry(value) -> dict:
    normals, variation, fine, coarse = value
    arrays = {"normals": normals, "variation": variation,
              "fine_order": GC.json_array(list(fine)), "coarse_order": GC.json_array(list(coarse))}
    for prefix, block in (("fine", fine), ("coarse", coarse)):
        for name, item in block.items():
            if isinstance(item, np.ndarray):
                arrays[f"{prefix}__a__{name}"] = item
            elif isinstance(item, np.generic):   # numpy scalar: 0-d array, restored as the same scalar type
                arrays[f"{prefix}__s__{name}"] = np.asarray(item)
            else:   # Python scalar (e.g. clipped fraction): exact through JSON
                arrays[f"{prefix}__j__{name}"] = GC.json_array(item)
    return arrays


def _decode_point_geometry(arrays) -> tuple:
    blocks = {}
    for prefix in ("fine", "coarse"):
        block = {}
        for key in GC.from_json_array(arrays[f"{prefix}_order"]):      # original key order
            if f"{prefix}__a__{key}" in arrays:
                block[key] = arrays[f"{prefix}__a__{key}"]
            elif f"{prefix}__s__{key}" in arrays:
                block[key] = arrays[f"{prefix}__s__{key}"][()]
            else:
                block[key] = GC.from_json_array(arrays[f"{prefix}__j__{key}"])
        blocks[prefix] = block
    return arrays["normals"], arrays["variation"], blocks["fine"], blocks["coarse"]


def _stage_a_record(job_dir: Path, job: dict | None) -> dict:
    """The stage-A record stage B will read: ``stage_a.json`` on disk (the job record's embedded copy of
    older jobs can carry a stale absolute clean-STL path), falling back to ``job["a"]`` only without it."""
    path = Path(job_dir) / "stage_a.json"
    if path.is_file():
        return json.loads(path.read_text(encoding="utf-8"))
    record = (job or {}).get("a")
    return record if isinstance(record, dict) else {}


def precompute_geometry_cache(job_dir: Path, job: dict, *, release=None, cancel_event=None) -> dict:
    """Compute and store the mesh-only stage-B intermediates while the outlet names are being checked.

    Contract (v0.14): runs in a background thread when a job enters ``awaiting_confirmation``; fills
    ``<job_dir>/geometry_cache/`` with the clean mesh, the smoothed + resampled cloud, (wall families)
    the PCA normals / curvatures of the cloud, and the morphology station tables and lumen volume.
    None of these depend on the outlet mapping (it only relabels branches) or on the release's
    weights, so a later stage B — or a rerun of the same case with another release — reuses them when
    the keys match and recomputes otherwise; results are bit-identical either way.  Never raises:
    returns ``{"ok", "seconds", "cache", ...}`` (``cancelled`` / ``error`` / ``skipped`` on the way out).
    ``cancel_event`` (threading.Event) stops it between steps.
    """
    started = time.perf_counter()
    job_dir = Path(job_dir).resolve()
    cancelled = lambda: bool(cancel_event is not None and cancel_event.is_set())
    out = {"ok": False, "seconds": 0.0}
    if not GC.enabled():
        return {**out, "skipped": "disabled"}
    cache = GC.GeometryCache(job_dir)
    lock = GC.job_lock(job_dir)
    try:
        a = _stage_a_record(job_dir, job)
        if a.get("stage") != "A" or not (a.get("input_check") or {}).get("ok"):
            return {**out, "skipped": "stage A not passed"}
        params = (job or {}).get("params") or {}
        smooth_mm = float(params.get("smooth_mm", DEFAULT_SMOOTH_MM))
        spacing_mm = float(params.get("spacing_mm", DEFAULT_SPACING_MM))
        seed = G.stable_sampling_seed(a["input_sha256"])
        volume = False
        if release is not None:
            from .families import family_for_release
            volume = family_for_release(release).protocol == "single_frame_volume"
        clean_stl = resolve_job_path(job_dir, a["input_check"]["clean_stl"])
        steps, timing = [], {}
        stop = lambda: {**out, "cancelled": True, "steps": steps, "timing_s": timing, "cache": cache.summary()}
        t = time.perf_counter()
        with lock:
            vertices, faces = _clean_mesh(job_dir, clean_stl, cache)
        vertices, faces = np.asarray(vertices, np.float64), np.asarray(faces, np.int64)
        steps.append("mesh")
        if cancelled():
            return stop()
        with lock:
            _, pts = _resampled(vertices, faces, smooth_mm=smooth_mm, spacing_mm=spacing_mm, seed=seed, cache=cache)
        steps.append("resample"); timing["smooth_resample"] = round(time.perf_counter() - t, 2)
        if cancelled():
            return stop()
        atlas = CL.load_vessel_geom_atlas(job_dir / "centerline")
        if not volume:
            # The wall family patches escaped end radii (geometry.build_case) before the normals and before
            # the morphology reads the atlas; the volume family uses the atlas as loaded.
            from wss_features.cloud import patch_atlas_end_radius
            t = time.perf_counter()
            patch_atlas_end_radius(atlas, pts)
            _point_geometry_provider(cache, lock)(pts, atlas)
            steps.append("point_geometry"); timing["features"] = round(time.perf_counter() - t, 2)
            if cancelled():
                return stop()
        t = time.perf_counter()
        with lock:
            MORPH.precompute_sections(vertices, faces, atlas, cache, cancelled=cancelled)
        if cancelled():
            return stop()
        steps.append("morphology"); timing["morphology"] = round(time.perf_counter() - t, 2)
        return {**out, "ok": True, "cancelled": False, "steps": steps, "timing_s": timing,
                "family": "volume" if volume else "wall", "seconds": round(time.perf_counter() - started, 2),
                "cache": cache.summary()}
    except Exception as exc:  # noqa: BLE001 — a precompute failure only means stage B computes everything itself
        return {**out, "error": f"{type(exc).__name__}: {exc}"[:500], "seconds": round(time.perf_counter() - started, 2),
                "cache": cache.summary()}


def cache_record(cache: "GC.GeometryCache") -> dict:
    """Compact ``summary["geometry_cache"]``: which mesh-only steps were reused vs computed in this run
    (lets the ETA / timeline tell a cache-assisted run from a cold one)."""
    stats = cache.summary()
    hits = sorted(set(stats["hits"]))
    return {"enabled": bool(stats["enabled"]), "reused": hits,
            "computed": sorted(set(stats["writes"]) | (set(stats["misses"]) - set(hits))) if stats["enabled"] else []}


def prune_geometry_cache(cache: "GC.GeometryCache") -> None:
    """Drop stage-B cache entries of earlier inputs/code that this run did not use (bounded disk use)."""
    if not cache.active or not cache.dir.is_dir():
        return
    keep = set(cache.used)
    for path in cache.dir.glob("*.npz"):
        kind = path.name.split("-", 1)[0]
        if kind in {"mesh", "resample", "pointgeom", "morph", "morphvol", "volume_case"} and path.name not in keep:
            with contextlib.suppress(OSError):
                path.unlink()


def _preview_surface(path: Path, max_faces: int = 18000, mesh: tuple | None = None) -> dict:
    """Small display mesh; model inference always uses the full cleaned STL.

    ``mesh`` (v0.14) is the already loaded ``load_stl(path)`` result, to avoid reading the file twice.  v0.15.8: a
    connected vertex-clustered surface (:mod:`.preview`) instead of every k-th triangle of the upload.
    """
    from .preview import preview_payload
    vertices, faces = mesh if mesh is not None else load_stl(path)
    return preview_payload(np.asarray(vertices, np.float32), np.asarray(faces, np.int64), max_faces)


def report_meta(meta: dict) -> dict:
    """Summary trimmed for embedding in the report page.

    ``summary.json`` keeps the full morphology stations (positions, closure flags, areas); the page
    only draws the diameter curves and the largest-station ring, so the station tables are reduced
    to the three series contract §17.3 asks for.  Nothing else is changed.
    """
    morphology = meta.get("morphology")
    if not isinstance(morphology, dict):
        return meta
    return {**meta, "morphology": MORPH.trim_for_report(morphology)}


def _now() -> str:
    from .clock import now_iso          # O1: ISO-8601 with offset in the deployment zone (was a naive local string)
    return now_iso()


def _archive_previous_run(job_dir: Path) -> str | None:
    """Preserve a completed result before a retry/manual override rewrites it."""
    summary = Path(job_dir) / "summary.json"
    if not summary.is_file():
        return None
    from .clock import strftime as _strftime
    stamp = _strftime("%Y%m%dT%H%M%S") + f"_{time.time_ns() % 1000000:06d}"
    digest = hashlib.sha256(summary.read_bytes()).hexdigest()[:10]
    archive = Path(job_dir) / "history" / f"{stamp}_{digest}"
    archive.mkdir(parents=True, exist_ok=False)
    for name in ("summary.json", "run_manifest.json", "report.html", "wall_wss.vtp",
                 "points_wss.csv", "field.npz", "quality_audit.json", "volume_fields.vtp",
                 "points_volume.csv", "wall_pressure.vtp", "streamlines.vtp"):
        source = Path(job_dir) / name
        if source.is_file():
            shutil.copy2(source, archive / name)
    return archive.relative_to(job_dir).as_posix()


def stage_a(stl_path: Path, job_dir: Path, *, inlet: int | None = None, smooth_iterations: int = 0,
            units: str = "mm", remove_fragments: bool = False, progress=None, cancelled=None) -> dict:
    job_dir = Path(job_dir).resolve(); job_dir.mkdir(parents=True, exist_ok=True); T = {}
    progress = progress or (lambda phase, detail: None)
    cancelled = cancelled or (lambda: False)
    if cancelled(): raise InterruptedError("任务已取消")
    progress("ingest", "正在读取并检查 STL")
    t = time.perf_counter(); ic = ingest(Path(stl_path), job_dir, units=units, remove_fragments=remove_fragments); T["ingest"] = time.perf_counter() - t
    if ic["status"] != "pass":
        out = {"stage": "input", "created_at": _now(), "stl": portable_job_path(job_dir, stl_path), "input_sha256": file_sha256(Path(stl_path)), "input_check": ic,
               "centerline": None, "proposal": None, "inlet_override": inlet, "timing_s": {k: round(v, 2) for k, v in T.items()}}
        write_text_atomic(job_dir / "stage_a.json", json.dumps(out, ensure_ascii=False, indent=1))
        return out
    if cancelled(): raise InterruptedError("任务已取消")
    progress("centerline", "正在提取中心线和开口")
    cl_dir = job_dir / "centerline"
    clean_stl = resolve_job_path(job_dir, ic["clean_stl"])
    t = time.perf_counter(); cl = CL.run_vessel_geom(clean_stl, cl_dir, inlet=inlet, smooth_iterations=smooth_iterations, cancelled=cancelled); T["centerline"] = time.perf_counter() - t
    if cancelled(): raise InterruptedError("任务已取消")
    atlas = CL.load_vessel_geom_atlas(cl_dir)
    proposal = CL.propose_outlets(atlas, orientation_source=ic.get("orientation_source", "unknown_stl"))
    proposal["direction_note"] = ic.get("direction_note", "")
    # v0.14: the clean mesh is loaded once here (preview) and kept in the geometry cache for stage B.
    with GC.job_lock(job_dir):
        clean_mesh = _clean_mesh(job_dir, clean_stl, GC.GeometryCache(job_dir))
    out = {"stage": "A", "created_at": _now(), "stl": portable_job_path(job_dir, stl_path), "input_sha256": file_sha256(Path(stl_path)), "input_check": ic,
           "centerline": cl, "proposal": proposal, "preview": _preview_surface(clean_stl, mesh=clean_mesh),
           "inlet_override": inlet, "timing_s": {k: round(v, 2) for k, v in T.items()}}
    write_text_atomic(job_dir / "stage_a.json", json.dumps(out, ensure_ascii=False, indent=1))
    return out


def stage_b(job_dir: Path, mapping: dict[str, str], release: Release, *, smooth_mm: float = DEFAULT_SMOOTH_MM, spacing_mm: float = DEFAULT_SPACING_MM,
            confirmed: bool = False, case_id: str | None = None, device: str = "auto",
            seed_count: int | None = None, threads: int | None = None, progress=None, cancelled=None) -> dict:
    """Stage B for any release: the model family decides how inputs, prediction and exports are built."""
    from .families import family_for_release
    family = family_for_release(release)
    return family.stage_b(job_dir, mapping, release, smooth_mm=smooth_mm, spacing_mm=spacing_mm,
                          confirmed=confirmed, case_id=case_id, device=device, seed_count=seed_count,
                          threads=threads, progress=progress, cancelled=cancelled)


def stage_b_wall(job_dir: Path, mapping: dict[str, str], release: Release, *, smooth_mm: float = DEFAULT_SMOOTH_MM, spacing_mm: float = DEFAULT_SPACING_MM,
                 confirmed: bool = False, case_id: str | None = None, device: str = "auto",
                 seed_count: int | None = None, threads: int | None = None, progress=None, cancelled=None) -> dict:
    """Wall WSS family: smooth + resample -> 27-D features -> ensemble -> metrics -> exports."""
    job_dir = Path(job_dir).resolve(); previous_archive = _archive_previous_run(job_dir); a = json.loads((job_dir / "stage_a.json").read_text(encoding="utf-8")); T = dict(a["timing_s"])
    progress = progress or (lambda phase, detail: None)
    cancelled = cancelled or (lambda: False)
    if a.get("stage") != "A" or not a.get("input_check", {}).get("ok"):
        raise ValueError("输入检查尚未通过，不能开始预测。")
    if not confirmed:
        raise ValueError("必须确认出口命名后才能开始预测。")
    if cancelled(): raise InterruptedError("任务已取消")
    progress("geometry", "正在平滑表面并重采样")
    atlas = CL.apply_mapping(CL.load_vessel_geom_atlas(job_dir / "centerline"), mapping)
    # v0.14: mesh-only intermediates come from <job_dir>/geometry_cache when their exact-input keys match
    # (filled by precompute_geometry_cache while the outlets were being checked); otherwise computed here.
    cache, cache_lock = GC.GeometryCache(job_dir), GC.job_lock(job_dir)
    with cache_lock:
        vertices, faces = _clean_mesh(job_dir, resolve_job_path(job_dir, a["input_check"]["clean_stl"]), cache)
    vertices = np.asarray(vertices, np.float64); faces = np.asarray(faces, np.int64)
    t = time.perf_counter()
    with cache_lock:
        v_s, pts = _resampled(vertices, faces, smooth_mm=smooth_mm, spacing_mm=spacing_mm,
                              seed=G.stable_sampling_seed(a["input_sha256"]), cache=cache)
    T["smooth_resample"] = time.perf_counter() - t
    if cancelled(): raise InterruptedError("任务已取消")
    progress("features", "正在构建 27 维几何特征")
    capfit_rule = Path(release.dir) / "rules" / "flow_split_rule_train136.json"
    # Model sampling uses the case unit id.  A task directory is ephemeral and
    # would make identical inputs produce different predictions across reruns.
    stable_case_name = "input-" + str(a["input_sha256"])[:24]
    t = time.perf_counter(); case, aux = G.build_case(pts, atlas, release.input_features,
                                                       case_name=stable_case_name,
                                                       capfit_rule_path=capfit_rule,
                                                       point_geometry_provider=(_point_geometry_provider(cache, cache_lock)
                                                                                if cache.active else None)); T["features"] = time.perf_counter() - t
    if cancelled(): raise InterruptedError("任务已取消")
    models_phrase = model_count_phrase(len(getattr(release, "models", None) or []) or None)
    progress("inference", f"正在运行{'' if models_phrase == '五模型' else ' '}{models_phrase}集成")
    # v0.14: CPU inference without an explicit thread count uses min(32, cores) instead of torch's
    # one-per-core default (WSS_DEPLOY_CPU_THREADS overrides); the GPU path is unchanged.
    cpu_threads = inference_threads(getattr(release, "device", None), threads)
    with torch_threads(cpu_threads):
        t = time.perf_counter(); pred = run_inference(release, case); T["inference_5_models"] = time.perf_counter() - t
    wss = pred["wss_pa"]; geom = aux["geom"]; diag = aux["diag"]
    # Extra wall scalar fields (e.g. TAWSS / OSI from a three-head release) ride along with the prediction;
    # everything below treats them generically: descriptor, statistics, npz/csv/vtp arrays and report colouring.
    extra_fields = {str(k): dict(v) for k, v in (pred.get("extra_fields") or {}).items() if isinstance(v, dict) and "values" in v}
    for key, item in extra_fields.items():
        values = np.asarray(item["values"], dtype=np.float64)
        if values.shape != np.shape(wss) or not np.isfinite(values).all():
            raise ValueError(f"额外壁面字段 {key} 必须与预测点一一对应且全部有限。")
        item["values"] = values
    if extra_fields:
        # RRT / ECAP are computed point by point from the predicted TAWSS / OSI (cycle_fields.with_derived);
        # they travel with the extra fields for statistics, CSV / VTP and the report, but not into field.npz.
        from .cycle_fields import with_derived
        extra_fields = with_derived(extra_fields)
    quality = ensemble_quality(wss, pred["seed_sd_pa"], seed_count=len(pred["seed_pred_pa"]))
    progress("metrics", "正在汇总预测点云统计")
    t = time.perf_counter()
    met = M.compute(pts, wss, geom, atlas, diag, a["input_check"]["area_mm2"])
    # One KD-tree / neighbour set for every interpolated field (bit-identical to report.interpolate_to_vertices).
    interp = VertexInterpolation(pts, vertices)
    vw = interp.values(wss.astype(np.float32)); vseg = interp.labels(geom["segment_id"])
    # Surface statistics are face based; derive a deterministic majority label
    # from the three interpolated vertex labels for the branch breakdown.
    face_segments = np.asarray(vseg, dtype=np.int64)[faces]
    face_labels = np.where(face_segments[:, 0] == face_segments[:, 1], face_segments[:, 0],
                           np.where(face_segments[:, 0] == face_segments[:, 2], face_segments[:, 0], face_segments[:, 1]))
    met["surface_statistics"] = M.surface_metrics(vertices, faces, vw,
                                                   covered=np.isfinite(vw), face_labels=face_labels)
    # centreline arrays for the viewer
    seg = atlas.col("segment_id").astype(int); idx = atlas.col("sample_index").astype(int); edges = []
    for s in atlas.segments:
        rows = np.flatnonzero(seg == int(s["segment_id"])); rows = rows[np.argsort(idx[rows])]; edges += [(rows[i], rows[i + 1]) for i in range(len(rows) - 1)]
    endpoints = []
    shares = {}
    for c in atlas.endpoints():
        lab = c["label"]; endpoints.append({"kind": "inlet" if lab == "inlet" else "outlet", "name": lab, "name_cn": OUTLET_CN.get(lab, lab), "radius_mm": float(diag["caps"].get(lab, {}).get("radius_mm", c["radius_mm"])), "center_mm": np.asarray(c["center_mm"]).round(2).tolist(), "segment_id": int(c["segment_id"])})
    # Murray shares are keyed by segment id in diag; attach to leaf endpoints
    for e in endpoints:
        if e["kind"] == "outlet":
            sh = diag["murray_shares"].get(str(e["segment_id"]));  e["share"] = sh
    per_branch_s = {}
    for sid in np.unique(geom["segment_id"]):
        per_branch_s[str(int(sid))] = float(np.median(geom["s_from_root_mm"][geom["segment_id"] == sid]))
    # Analysis layer (contract §1-§2): along-branch profiles and the findings list.  Derived from
    # the prediction cloud and the atlas only; nothing here changes a prediction or field.npz.
    thresholds = met["wss_field_pa"]["thresholds_pa"]
    # Cycle-integrated fields (three-head release) join the profiles and findings (§19.2); absent → unchanged output.
    cycle_arrays = {key: extra_fields[key]["values"] for key in ("tawss", "osi") if key in extra_fields}
    profile_arrays = {key: extra_fields[key]["values"] for key in ("tawss", "osi", "rrt", "ecap") if key in extra_fields}
    profiles_block = A.profiles(atlas, geom, {"wss": wss, **profile_arrays}, branch_names=met["branch_names"])
    T["metrics_and_interpolation"] = time.perf_counter() - t
    # Lumen morphology (contract §17.1): cross-sections of the same clean mesh the report embeds.
    progress("metrics", "正在测量沿程管腔截面与瘤体形态")
    t = time.perf_counter()
    with cache_lock:
        morphology_block = MORPH.compute(vertices, faces, atlas, branch_names=met["branch_names"],
                                         per_branch=met["per_branch"], thresholds=thresholds,
                                         cloud={"segment_id": geom["segment_id"], "wss": wss},
                                         section_cache=cache)
    T["morphology"] = time.perf_counter() - t
    t = time.perf_counter()
    # U10 (2026-09-30): anatomical zone statistics (neck / sac / iliac pairs) from the same cloud; a new block.
    zones_block = A.zones_wall(geom["segment_id"], geom["s_from_root_mm"], {"wss": wss, **cycle_arrays},
                               morphology=morphology_block, branch_names=met["branch_names"],
                               total_area_mm2=a["input_check"]["area_mm2"],
                               thresholds={"wss": tuple(thresholds[:2])})
    findings_block = A.findings_wall(pts, wss, geom, atlas, met["geometry"], thresholds=thresholds,
                                     total_area_mm2=a["input_check"]["area_mm2"], spacing_mm=diag["spacing_mm"],
                                     branch_names=met["branch_names"], morphology=morphology_block,
                                     cycle=cycle_arrays or None)
    T["metrics_and_interpolation"] += time.perf_counter() - t
    # Mapping provenance lives in the service job record.  Copy only the
    # review fields into the portable run record; never copy owner/session
    # credentials or the rest of job.json.
    mapping_history = []
    job_record_path = job_dir / "job.json"
    if job_record_path.is_file():
        try:
            job_record = json.loads(job_record_path.read_text(encoding="utf-8"))
            for item in job_record.get("mapping_history", []):
                if isinstance(item, dict):
                    mapping_history.append({key: item[key] for key in (
                        "at", "source", "confidence", "inlet", "suggested", "confirmed", "acknowledged",
                        "confidence_gate") if key in item})
        except (OSError, ValueError, TypeError):
            mapping_history = []
    model_frame = {"target": "peak_systole", "step": 1162, "time_s": 0.21, "label": "peak_systole"}
    time_axis = single_frame_time_axis(model_frame)
    fields = {"wss": field_descriptor("wss", label="壁面切应力", units="Pa", location="wall",
                                       kind="scalar", array_key="wss_pa", time_indices=(0,),
                                       axis_order=("point",), statistics_key="wss")}
    if extra_fields:
        from .cycle_fields import descriptor as cycle_descriptor
        for key, item in extra_fields.items():
            fields[key] = cycle_descriptor(key, item["values"])
    case_metadata = {}
    job_record_path = job_dir / "job.json"
    if job_record_path.is_file():
        try:
            jr = json.loads(job_record_path.read_text(encoding="utf-8"))
            case_metadata = {key: jr.get(key, [] if key == "tags" else "")
                             for key in ("patient_id", "scan_label", "scan_date", "tags", "notes")}
        except (OSError, ValueError, TypeError):
            case_metadata = {}
    meta = {"schema_version": "wss-deploy.summary/v1", "case_id": case_id or job_dir.name, "release": release.name, "device": pred["device"], "gpu": pred["gpu"], "created_at": _now(), "input_sha256": a["input_sha256"],
            "case_metadata": case_metadata,
            "input_check": a["input_check"], "centerline": a["centerline"], "outlets_confirmed": confirmed, "mapping": mapping,
            "proposal_confidence": a["proposal"].get("confidence"),
            "proposal_side_confidence": a["proposal"].get("side_confidence", {}),
            "proposal_confirmation_required": a["proposal"].get("confirmation_required"),
            "proposal_confidence_gate": a["proposal"].get("confidence_gate", {}),
            "flags": a["proposal"].get("flags", []),
            "cloud": {"n_points": int(len(pts)), "spacing_mm": diag["spacing_mm"], "smooth_mm": smooth_mm, "surface_variation_median": diag["surface_variation_median"], "junction_ambiguous_fraction": diag["junction_ambiguous_fraction"]},
            "caps": diag["caps"], "murray_shares": diag["murray_shares"], "endpoints": endpoints, "per_branch_s": per_branch_s, "timing_s": {k: round(v, 2) for k, v in T.items()},
            "seconds_per_model": [round(x, 2) for x in pred["seconds_per_model"]],
            "sampling_seed": G.stable_sampling_seed(a["input_sha256"]),
            "model_frame": model_frame,
            # Generic result metadata.  The historical WSS keys below remain
            # in the summary for old reports and scripts.
            "model_release": model_release_metadata(release),
            "time_axis": time_axis,
            "fields": fields,
            "run_parameters": {"smooth_mm": smooth_mm, "spacing_mm": spacing_mm, "device": pred["device"], "seed_count": len(pred["seed_pred_pa"]), "threads": threads},
            # v0.14: torch threads actually used on the CPU (run_parameters keeps the requested value, so the
            # run identity of earlier runs is unchanged); absent for GPU runs.
            **({"inference_threads": cpu_threads} if cpu_threads is not None else {}),
            "frame_transform": A.frame_transform(geom["frame_rotation"], geom["frame_origin_mm"],
                                                 source="vessel_geom atlas + anatomical_frame",
                                                 direction_source=a["input_check"].get("orientation_source")),
            "release_hash": file_sha256(Path(release.dir) / "MANIFEST.sha256") if (Path(release.dir) / "MANIFEST.sha256").is_file() else release.name,
            # Identity of the frozen STL -> feature program; a release may pin it (see infer.Release).
            "feature_contract": feature_contract(),
            "quality": quality["quality"],
            "profiles": profiles_block, "findings": findings_block, "morphology": morphology_block,
            "zones": zones_block,
            "audit": {"mapping_history": mapping_history, "mapping_history_count": len(mapping_history),
                      "previous_run_archive": previous_archive,
                      "quality_audit": quality["audit"], "quality_audit_path": "quality_audit.json"},
            "interpolation": {"method": "Gaussian", "sigma_mm": 0.5, "max_dist_mm": 1.5, "covered_vertices": int(np.isfinite(vw).sum()), "total_vertices": int(len(vw))}, **met}
    if extra_fields:
        from .cycle_fields import cycle_block
        meta["cycle"] = cycle_block(extra_fields, geom["segment_id"], met["branch_names"], a["input_check"]["area_mm2"])
        if isinstance(pred.get("cycle"), dict):
            meta["cycle"]["definition"] = {**meta["cycle"]["definition"], **pred["cycle"]}
    from .reference import evaluate as evaluate_reference, merge_sidecar_v2
    release_info = merge_sidecar_v2(getattr(release, "info", {}), getattr(release, "dir", None),
                                    getattr(release, "release_id", None))
    meta["reference_assessment"] = evaluate_reference(meta, release_info)
    # U11 (2026-09-30): high-WSS findings are graded against the same-protocol cohort p90 (none → 提示).
    A.grade_high_findings(meta["findings"], meta["reference_assessment"])
    meta["display_name"] = summary_display_name(meta)
    # Trust mask (contract §3) needs the reference assessment for the geometry-out-of-range bit.
    trust_vertices, meta["trust"] = A.trust_wall(vertices, vw, pts, geom["surface_variation"], vseg,
                                                 meta["reference_assessment"], met["branch_names"])
    meta["run_identity"] = stable_run_identity(input_sha256=meta["input_sha256"],
                                                release=meta["model_release"],
                                                mapping=meta["mapping"],
                                                parameters=meta["run_parameters"])
    meta["narrative"] = NARR.build_narrative(meta)
    statistics = {"wss": met["wss_field_pa"], "surface": met["surface_statistics"], "quality": quality["quality"]}
    if extra_fields:
        statistics.update({key: meta["cycle"]["fields"][key] for key in extra_fields if key in meta["cycle"]["fields"]})
        if "stagnation" in meta["cycle"]:
            statistics["stagnation"] = meta["cycle"]["stagnation"]
    meta["results"] = build_results(time_axis=time_axis, fields=fields, statistics=statistics,
                                     compatibility=wss_compatibility(met))
    meta["geometry_cache"] = cache_record(cache)
    progress("export", "正在写入 CSV、VTP 和 HTML 报告")
    export_started = time.perf_counter()
    extra_npz, extra_vertex, extra_points = {}, {}, {}
    for key, item in extra_fields.items():
        if item.get("derived"):
            continue
        ak = fields[key]["array_key"]
        extra_vertex[ak] = interp.values(item["values"].astype(np.float32))
        extra_points[ak] = item["values"].astype(np.float32)
        extra_npz[ak] = extra_points[ak]; extra_npz[f"vertex_{ak}"] = extra_vertex[ak]
        if item.get("seed_pred") is not None: extra_npz[f"seed_{ak}"] = np.asarray(item["seed_pred"], np.float32)
        if item.get("seed_sd") is not None: extra_npz[f"seed_sd_{ak}"] = np.asarray(item["seed_sd"], np.float32)
    extra_points_derived, extra_vertex_derived = derived_display_arrays(fields, extra_fields, extra_vertex)
    extra_points.update(extra_points_derived); extra_vertex.update(extra_vertex_derived)
    # v0.14: every output is written to a hidden temporary file and renamed into place (atomic).
    save_npz_atomic(job_dir / "field.npz", pts=pts.astype(np.float32), wss_pa=wss.astype(np.float32), seed_pred_pa=pred["seed_pred_pa"].astype(np.float32), seed_sd_pa=pred["seed_sd_pa"].astype(np.float32), segment_id=geom["segment_id"],
                    s_from_root_mm=geom["s_from_root_mm"], theta_rad=geom["theta_rad"], radius_mm=geom["radius_mm"], vertices=vertices.astype(np.float32), faces=faces.astype(np.int32), vertex_wss_pa=vw, **extra_npz)
    with atomic_output(job_dir / "points_wss.csv") as csv_tmp, open(csv_tmp, "w") as f:
        extra_keys = [fields[key]["array_key"] for key in extra_fields if fields[key]["array_key"] in extra_points]
        f.write("x_mm,y_mm,z_mm,wss_pa,segment_id,branch,s_from_inlet_mm,theta_rad" + "".join("," + k for k in extra_keys) + "\n")
        bn = met["branch_names"]
        # Same row format as before; plain Python floats format ~10x faster than per-element numpy indexing.
        segment_list = geom["segment_id"].astype(int).tolist()
        rows = zip(pts[:, 0].tolist(), pts[:, 1].tolist(), pts[:, 2].tolist(), wss.tolist(), segment_list,
                   [bn.get(str(s), "") for s in segment_list], geom["s_from_root_mm"].tolist(), geom["theta_rad"].tolist(),
                   *[extra_points[k].astype(np.float64).tolist() for k in extra_keys])
        f.write("\n".join(f"{x:.3f},{y:.3f},{z:.3f},{w:.4f},{s},{b},{sr:.2f},{th:.4f}" + "".join(f",{e:.4f}" for e in extras)
                          for x, y, z, w, s, b, sr, th, *extras in rows) + "\n")
    with atomic_output(job_dir / "wall_wss.vtp") as vtp_tmp:
        vtp_ok = R.write_vtp(vtp_tmp, vertices, faces, {"wss_pa": vw, "wss_covered": np.isfinite(vw), "segment_id": vseg.astype(np.int32), **extra_vertex})
    with atomic_output(job_dir / "report.html") as report_tmp:
        R.build_html(report_tmp, report_meta(meta),
                     mesh={"vertices": vertices, "faces": faces, "wss": vw, "segment": vseg, "trust": trust_vertices, "extra": extra_vertex},
                     cloud={"pts": pts, "wss": wss, "segment": geom["segment_id"], "s_from_root_mm": geom["s_from_root_mm"], "theta_rad": geom["theta_rad"], "radius_mm": geom["radius_mm"], "dist_to_junction_mm": geom["dist_to_junction_mm"],
                            "extra": {ak: extra_points[ak] for ak in extra_vertex}},
                     centerline={"xyz": atlas.xyz, "radius_mm": atlas.col("radius_mm"), "edges": np.asarray(edges, dtype=np.uint32), "segment": seg.astype(np.int16)})
        T["export"] = time.perf_counter() - export_started
        T["total"] = float(sum(v for k, v in T.items() if k not in {"total"}))
        meta["timing_s"] = {k: round(v, 2) for k, v in T.items()}
        meta["run_manifest"] = {"path": "run_manifest.json", "schema_version": "wss-deploy.run-manifest/v1"}
        meta["exports"] = {"vtp": bool(vtp_ok), "csv": True, "html": True, "internal_field": True, "run_manifest": True}
        # v0.14 provenance (analysis_version, deploy_version, git_describe, git_dirty, code_source_hash).
        meta.update(summary_provenance())
        R.update_html_meta(report_tmp, report_meta(meta))
    write_text_atomic(job_dir / "summary.json", json.dumps(meta, ensure_ascii=False, indent=1))
    # The manifest is intentionally a separate, portable record.  It is
    # written after summary/report so their hashes are stable.  Changing model
    # weights creates a new, auditable release rather than overwriting a run.
    quality_audit_path = job_dir / "quality_audit.json"
    # Maintenance-only record: keeps numeric ensemble diagnostics and the
    # provenance needed to identify a result, while the public report receives
    # only the compact quality level/reasons.
    quality_audit_record = {
        **quality["audit"], "created_at": meta["created_at"],
        "input_sha256": meta["input_sha256"], "release_hash": meta["release_hash"],
        "model_release": meta["model_release"], "mapping_history": mapping_history,
    }
    write_text_atomic(quality_audit_path, json.dumps(quality_audit_record, ensure_ascii=False, indent=1))
    output_names = ("report.html", "summary.json", "wall_wss.vtp", "points_wss.csv", "field.npz", "quality_audit.json", "stage_a.json")
    write_run_manifest(job_dir, meta, outputs=output_names)
    prune_geometry_cache(cache)
    return meta


def run_all(stl_path: Path, out_dir: Path, release: Release, *, mapping: dict[str, str] | None = None,
            inlet: int | None = None, smooth_mm: float = 1.0, spacing_mm: float = 0.5,
            case_id: str | None = None, units: str = "mm",
            allow_unvalidated_auto: bool = False) -> dict:
    a = stage_a(stl_path, out_dir, inlet=inlet, units=units)
    if a.get("stage") != "A":
        raise RuntimeError("输入检查未通过：" + "; ".join(a["input_check"].get("errors", []) + a["input_check"].get("flags", [])))
    gate = CL.evaluate_confidence_gate(
        a["proposal"],
        orientation_source=a["input_check"].get("orientation_source", "unknown_stl"),
        release=release,
    )
    a["proposal"]["confidence_gate"] = gate
    a["proposal"]["confidence_profile"] = gate.get("profile_id")
    a["proposal"]["confidence_is_calibrated"] = gate.get("calibration_status") == "validated"
    a["proposal"]["confirmation_required"] = not bool(gate.get("passed"))
    write_text_atomic(Path(out_dir) / "stage_a.json", json.dumps(a, ensure_ascii=False, indent=1))
    if mapping is None:
        if not a["proposal"]["auto_ok"]:
            raise RuntimeError("automatic outlet naming failed: " + "; ".join(a["proposal"]["flags"]))
        if not gate["passed"] and not allow_unvalidated_auto:
            reasons = "; ".join(gate.get("reasons") or [])
            raise RuntimeError("automatic outlet naming is not independently validated; "
                               "provide an explicit mapping after review. " + reasons)
        mapping = a["proposal"]["mapping"]
    return stage_b(out_dir, mapping, release, smooth_mm=smooth_mm, spacing_mm=spacing_mm, confirmed=True, case_id=case_id)

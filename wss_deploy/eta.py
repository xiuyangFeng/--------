"""Stage-by-stage progress and remaining-time estimate for active jobs (v0.12.2, contract ``ANALYSIS_CONTRACT.md`` §21.2).

Pure functions: the job manager supplies the finished jobs' timings (history), the input face count and the
stage clock of the running attempt; nothing here touches the file system except :func:`current_feature_hash`.

Expected seconds per stage come from the median of finished jobs of the same family whose feature program is
the current one (``summary.feature_contract.source_hash``), normalised by input size (``seconds / (f/10k)``,
centreline ``(f/10k)^1.4``); with fewer than three such jobs a measured two-point default table is interpolated
linearly in the face count (floored at half the small-case value).
"""
from __future__ import annotations

import functools
import math
import statistics
from typing import Any, Iterable, Mapping

A_STAGES = ("ingest", "centerline")
B_STAGES = {"wall": ("smooth_resample", "features", "inference", "metrics", "morphology", "export"),
            "volume": ("smooth_resample", "volume_features", "inference", "streamlines", "morphology", "export")}
LABELS = {"ingest": "检查输入", "centerline": "提取中心线", "smooth_resample": "平滑与重采样", "features": "几何特征",
          "volume_features": "体内采样与特征", "inference": "模型推理", "metrics": "统计汇总", "streamlines": "流线与插值",
          "morphology": "形态测量", "export": "报告与导出"}
# 2026-09-23 measured after the operator speed-ups (GPU busy): (23 184 faces = LV, 273 227 faces = LIU).
DEFAULT_FACES = (23184.0, 273227.0)
DEFAULTS = {"ingest": (0.2, 6.1), "centerline": (3.4, 65.0), "smooth_resample": (4.2, 12.8), "features": (2.8, 8.0),
            "volume_features": (8.9, 14.1), "inference:wall": (7.0, 9.0), "inference:volume": (3.2, 3.2), "metrics": (0.2, 0.5),
            "streamlines": (3.5, 6.7), "morphology": (1.7, 14.0), "export": (0.8, 1.1)}
# ``summary.timing_s`` / ``stage_a.timing_s`` keys → stage keys (several keys may add into one stage).
TIMING_KEYS = {"ingest": "ingest", "centerline": "centerline", "smooth_resample": "smooth_resample", "features": "features",
               "volume_features": "volume_features", "inference_5_models": "inference", "inference_volume": "inference",
               "inference": "inference", "metrics_and_interpolation": "metrics", "streamlines_and_interpolation": "streamlines",
               "morphology": "morphology", "export": "export", "metrics_and_export": "export"}
MIN_HISTORY = 3
SIZE_EXPONENT = {"centerline": 1.4}
# v0.14 (J2): stage-B work ``pipeline.precompute_geometry_cache`` does while the outlets are being confirmed
# (mesh-only: smoothing + resampling, wall point features, morphology sections).  A cache hit leaves a small
# residual (loading the cached arrays) of PRECOMPUTE_RESIDUAL × the expected stage time.
PRECOMPUTE_STAGES = {"wall": ("smooth_resample", "features", "morphology"), "volume": ("smooth_resample", "morphology")}
PRECOMPUTE_RESIDUAL = 0.1
# Share of a stage's work the precompute does: the point geometry (PCA normals / curvatures) is about two thirds
# of the ``features`` stage; the rest (centreline-relative features) still runs in stage B.
PRECOMPUTE_COVERAGE = {"smooth_resample": 1.0, "features": 2.0 / 3.0, "morphology": 1.0}
# ``summary.geometry_cache.reused`` cache kinds (pipeline / morphology) → the stage whose work they skip.
CACHE_KIND_STAGES = {"mesh": "smooth_resample", "resample": "smooth_resample", "pointgeom": "features",
                     "morph": "morphology", "morphvol": "morphology"}
OVERDUE_TAIL = 0.2          # an overdue stage keeps a shrinking tail of 0.2 × expected, never a negative remainder
ACTIVE = ("queued", "running", "awaiting_input", "awaiting_confirmation")


def _num(value: Any) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    value = float(value)
    return value if math.isfinite(value) else None


def _now_iso(now: float) -> str:
    from .clock import iso
    return iso(now)


@functools.lru_cache(maxsize=1)
def current_feature_hash() -> str | None:
    """Source hash of the feature program this process runs (the service restarts when it changes)."""
    try:
        from wss_features import contract
        return contract().get("source_hash")
    except Exception:  # noqa: BLE001 — an unknown hash simply means "no history"
        return None


def stages_for(family: str | None) -> tuple[str, ...]:
    return A_STAGES + B_STAGES.get(family or "wall", B_STAGES["wall"])


def segment_of(stage: str) -> str:
    return "A" if stage in A_STAGES else "B"


def phase_stage(family: str | None, phase: str | None, detail: str | None = "") -> str | None:
    """Stage key of a pipeline ``progress(phase, detail)`` call (``pipeline`` / ``volume_pipeline`` phase names)."""
    detail = detail or ""
    if phase in {"ingest", "centerline", "export", "inference"}:
        return phase
    if phase == "geometry":
        return "smooth_resample"
    if phase == "features":
        return "volume_features" if family == "volume" else "features"
    if phase == "metrics":
        if "形态" in detail or "截面" in detail:
            return "morphology"
        return "streamlines" if family == "volume" or "流线" in detail else "metrics"
    return None


def stage_seconds(timing: Mapping[str, Any] | None) -> dict[str, float]:
    """``timing_s`` → seconds per stage key (unknown keys and ``total`` are ignored)."""
    out: dict[str, float] = {}
    for key, value in (timing or {}).items():
        stage, seconds = TIMING_KEYS.get(key), _num(value)
        if stage and seconds is not None:
            out[stage] = out.get(stage, 0.0) + seconds
    return out


def _scale(stage: str, faces: float) -> float:
    return (max(faces, 1.0) / 1e4) ** SIZE_EXPONENT.get(stage, 1.0)


DEFAULT_WALL_MODELS = 5     # the wall inference defaults were measured on the five-model X5D ensemble


def default_seconds(stage: str, family: str | None, faces: float | None, *, models_count: int | None = None) -> float:
    """Two-point default, linear in the face count, never below half the small-case value.

    Wall inference scales with the ensemble size (the table is the five-model X5D; M1 has three models).
    """
    small, large = DEFAULTS.get(f"{stage}:{family or 'wall'}") or DEFAULTS.get(stage) or (1.0, 1.0)
    if stage == "inference" and (family or "wall") == "wall" and isinstance(models_count, int) and models_count > 0:
        small, large = small * models_count / DEFAULT_WALL_MODELS, large * models_count / DEFAULT_WALL_MODELS
    if faces is None:
        return small
    f0, f1 = DEFAULT_FACES
    value = small + (large - small) * (float(faces) - f0) / (f1 - f0)
    return max(value, small / 2.0)


def history_sample(*, family: str | None, faces: Any, timing: Mapping[str, Any] | None, source_hash: str | None,
                   release_id: str | None = None, device: str | None = None, cold_start: bool = False) -> dict | None:
    """One finished job as a history record, or None when it cannot be used.

    ``cold_start`` marks the first stage-B run of a release in a service process: its inference includes the
    one-off warm-up (≈ 56 s on CPU for X5D, 2026-09-23), so that stage is left out of the history.
    """
    faces = _num(faces)
    seconds = stage_seconds(timing)
    if cold_start:
        seconds.pop("inference", None)
    if not faces or faces <= 0 or not seconds or family not in B_STAGES:
        return None
    return {"family": family, "faces": faces, "seconds": seconds, "source_hash": source_hash, "release_id": release_id,
            "device": device if device in {"cpu", "cuda"} else None}


def expected_table(family: str | None, faces: float | None, samples: Iterable[Mapping[str, Any]], *,
                   release_id: str | None = None, source_hash: str | None = None, device: str | None = None,
                   models_count: int | None = None) -> tuple[dict[str, float], str, int]:
    """Expected seconds for every stage of ``family``: ``(table, basis, n_history)``.

    History = samples of the same family with the current feature-program hash.  The ``inference`` stage only
    pools runs of the same release on the same device (three vs five models, CPU vs GPU differ by up to 10×);
    with fewer than three of those it uses the default table.
    """
    family = family if family in B_STAGES else "wall"
    usable = [s for s in samples if s.get("family") == family and source_hash and s.get("source_hash") == source_hash]
    basis = "history" if len(usable) >= MIN_HISTORY and faces else "default"
    table = {}
    for stage in stages_for(family):
        pool = usable
        if stage == "inference":
            pool = [s for s in usable if (not release_id or s.get("release_id") == release_id)
                    and (not device or not s.get("device") or s.get("device") == device)]
        values = [s["seconds"][stage] / _scale(stage, s["faces"]) for s in pool if stage in s["seconds"]]
        if basis == "history" and len(values) >= MIN_HISTORY:
            table[stage] = statistics.median(values) * _scale(stage, faces)
        else:
            table[stage] = default_seconds(stage, family, faces, models_count=models_count)
    return table, basis, len(usable)


def stage_remaining(expected: float, elapsed: float) -> float:
    """Remaining seconds of the running stage: ``expected − elapsed`` until 80 %, then a tail that shrinks as
    ``0.2 × expected × (0.8 × expected / elapsed)`` — continuous, decreasing and never negative."""
    expected, elapsed = max(expected, 0.0), max(elapsed, 0.0)
    knee = (1.0 - OVERDUE_TAIL) * expected
    if elapsed <= knee or expected <= 0:
        return max(0.0, expected - elapsed)
    return OVERDUE_TAIL * expected * (knee / elapsed)


def build_eta(*, family: str | None, status: str, segment: str, faces: float | None, table: Mapping[str, float],
              basis: str, n_history: int, a_seconds: Mapping[str, float] | None = None, clock: Mapping[str, Any] | None = None,
              now: float, queue_ahead: int | None = None, queue_ahead_s: float | None = None) -> dict | None:
    """The ``eta`` block of one active job's snapshot (None for final states).

    ``segment`` is the stage the job is queued / running for (``A`` or ``B``).  ``a_seconds`` are the finished
    stage-A timings (``stage_a.timing_s``); ``clock`` is the running attempt's stage clock
    ``{"segment", "current", "current_started_ts", "done": {stage: seconds}}``.
    """
    if status not in ACTIVE:
        return None
    keys = stages_for(family)
    a_seconds = dict(a_seconds or {})
    clock = clock if status == "running" and isinstance(clock, Mapping) and clock.get("segment") == segment else {}
    done_now = dict(clock.get("done") or {})
    current = clock.get("current") if clock else None
    if status == "running" and current not in keys:
        current = next(k for k in keys if segment_of(k) == segment)    # before the first progress call
        started = None
    else:
        started = _num(clock.get("current_started_ts")) if clock else None
    a_finished = status == "awaiting_confirmation" or segment == "B"
    stages = []
    remaining = segment_remaining = 0.0
    for key in keys:
        expected = round(float(table.get(key, 0.0)), 1)
        row = {"key": key, "label": LABELS[key], "expected_s": expected, "state": "pending"}
        if segment_of(key) == "A" and a_finished:
            row["state"] = "done"
            if key in a_seconds:
                row["elapsed_s"] = round(a_seconds[key], 1)
        elif status == "awaiting_input" and key == "ingest":
            row["state"] = "done"
            if key in a_seconds:
                row["elapsed_s"] = round(a_seconds[key], 1)
        elif status == "running" and key in done_now and key != current:
            row["state"] = "done"
            row["elapsed_s"] = round(float(done_now[key]), 1)
        elif status == "running" and key == current:
            elapsed = max(0.0, now - started) if started is not None else 0.0
            elapsed += float(done_now.get(key, 0.0))                     # a stage re-entered after another one
            left = stage_remaining(float(table.get(key, 0.0)), elapsed)
            row.update(state="running", elapsed_s=round(elapsed, 1))
            if elapsed > float(table.get(key, 0.0)):
                row["overdue"] = True
            remaining += left
            if segment_of(key) == segment:
                segment_remaining += left
        else:
            if status == "running" and segment == "B" and segment_of(key) == "B" and keys.index(key) < keys.index(current):
                row["state"] = "done"                                    # skipped without its own progress call
            else:
                remaining += float(table.get(key, 0.0))
                if segment_of(key) == segment:
                    segment_remaining += float(table.get(key, 0.0))
        stages.append(row)
    out = {"basis": basis, "n_history": n_history, "faces": int(faces) if faces else None, "family": family or "wall",
           "status": status, "segment": segment, "current": current if status == "running" else None, "stages": stages,
           "remaining_s": round(remaining, 1), "segment_remaining_s": round(segment_remaining, 1),
           "b_total_s": round(sum(float(table.get(k, 0.0)) for k in keys if segment_of(k) == "B"), 1),
           "updated_at": _now_iso(now)}
    if status == "awaiting_confirmation":
        out["remaining_s"] = out["b_total_s"]
    if queue_ahead is not None:
        out["queue_ahead"] = int(queue_ahead)
        out["queue_ahead_s"] = round(max(0.0, queue_ahead_s or 0.0), 1)
    return out


def precompute_stages(family: str | None) -> tuple[str, ...]:
    """Stage keys a finished background precompute covers for ``family`` (when the result names none)."""
    return PRECOMPUTE_STAGES.get(family or "wall", PRECOMPUTE_STAGES["wall"])


# ``precompute_geometry_cache`` result ``steps`` → the stage keys whose work they did.
PRECOMPUTE_STEP_STAGES = {"mesh": "smooth_resample", "resample": "smooth_resample", "point_geometry": "features",
                          "morphology": "morphology"}


def steps_to_stages(steps: Iterable[str] | None, family: str | None = None) -> set[str]:
    """Stage keys covered by the finished precompute ``steps`` (``features`` only for the wall family)."""
    out = {PRECOMPUTE_STEP_STAGES[step] for step in (steps or ()) if step in PRECOMPUTE_STEP_STAGES}
    if (family or "wall") != "wall":
        out.discard("features")
    return out


def apply_precompute(table: Mapping[str, float], covered: Iterable[str], *, residual: float = PRECOMPUTE_RESIDUAL,
                     coverage: Mapping[str, float] | None = None) -> dict[str, float]:
    """Expected seconds after a background precompute.

    A covered stage keeps its uncovered share plus ``residual`` × the covered share (loading the cached arrays):
    ``expected × (1 − c × (1 − residual))`` with ``c`` from ``coverage`` (default :data:`PRECOMPUTE_COVERAGE`; 1 for
    unknown stages) — smoothing / morphology fall to 10 %, ``features`` (point geometry ≈ ⅔ of it) to 40 %."""
    covered = set(covered)
    coverage = PRECOMPUTE_COVERAGE if coverage is None else coverage
    out = {}
    for key, value in table.items():
        share = min(max(float(coverage.get(key, 1.0)), 0.0), 1.0) if key in covered else 0.0
        out[key] = float(value) * (1.0 - share * (1.0 - residual))
    return out


def cache_reused_stages(geometry_cache: Mapping[str, Any] | None) -> set[str]:
    """Stages whose timings are cache hits in a finished run (``summary.geometry_cache.reused``): such runs report
    ≈ 0 s there, so the per-stage history must leave those stages of the run out."""
    reused = geometry_cache.get("reused") if isinstance(geometry_cache, Mapping) else None
    return {CACHE_KIND_STAGES[kind] for kind in (reused or ()) if kind in CACHE_KIND_STAGES}


def queue_wait(ahead: Iterable[Mapping[str, Any]]) -> float:
    """Seconds until a queued job can start, from the segment work still ahead of it.

    Two workers share the queue; stage B is serialised, stage A is not.  So the wait is at least the stage-B
    work ahead and at least half of all work ahead.
    """
    a_work = b_work = 0.0
    for item in ahead:
        seconds = float(item.get("segment_remaining_s") or 0.0)
        if item.get("segment") == "B":
            b_work += seconds
        else:
            a_work += seconds
    return max(b_work, (a_work + b_work) / 2.0)


def stl_faces(content: bytes) -> int | None:
    """Face count of an STL byte string (binary header count, else ASCII ``endfacet`` lines)."""
    if not isinstance(content, (bytes, bytearray)) or len(content) < 84:
        return None
    count = int.from_bytes(content[80:84], "little")
    if 84 + 50 * count == len(content):
        return count
    ascii_count = content.count(b"endfacet")
    return ascii_count or None


class StlFaceCounter:
    """:func:`stl_faces` for a file read in chunks (v0.14 streamed uploads): feed every chunk to :meth:`update`,
    then :meth:`result` gives the same answer ``stl_faces`` gives for the concatenated bytes."""
    _WORD = b"endfacet"

    def __init__(self):
        self.size = 0
        self.header = b""
        self.ascii = 0
        self._tail = b""

    def update(self, chunk: bytes) -> None:
        self.size += len(chunk)
        if len(self.header) < 84:
            self.header += chunk[:84 - len(self.header)]
        window = self._tail + chunk        # the tail is shorter than the word: no occurrence is counted twice
        self.ascii += window.count(self._WORD)
        self._tail = window[-(len(self._WORD) - 1):]

    def result(self) -> int | None:
        if self.size < 84:
            return None
        count = int.from_bytes(self.header[80:84], "little")
        if 84 + 50 * count == self.size:
            return count
        return self.ascii or None


def faces_from_size(size: int | None) -> int | None:
    """Rough face count from a binary STL file size (used only before stage A has run)."""
    if not size or size < 84:
        return None
    return max(1, int((size - 84) // 50))


__all__ = ["A_STAGES", "B_STAGES", "DEFAULTS", "LABELS", "PRECOMPUTE_STAGES", "apply_precompute", "build_eta",
           "current_feature_hash", "default_seconds", "expected_table", "faces_from_size", "history_sample", "phase_stage",
           "precompute_stages", "queue_wait", "stage_remaining", "stage_seconds", "stages_for", "steps_to_stages", "stl_faces",
           "StlFaceCounter", "cache_reused_stages", "PRECOMPUTE_COVERAGE", "CACHE_KIND_STAGES"]

"""Workspace v2 data layer (``WORKSPACE_V2_CONTRACT.md`` §2–§3): manifest + raw arrays of a finished job.

Phase one reads everything from the job's existing ``report.html`` — the embedded ``wss-report-meta`` JSON and the
``wss-report-arrays`` (wall family) or ``volume-arrays`` (volume family) base64 payload.  That is exactly the data the
classic viewer shows, so the new workspace and the old report read the same numbers: every array served here is the
embedded base64 decoded, byte for byte.  The only arrays computed on the server are the derived ones the contract
names — RRT / ECAP (``cycle_fields.derive_indices`` on the embedded TAWSS / OSI, points and vertices separately) and
the volume speed ``|v|`` (the ``np.linalg.norm`` of the embedded float32 velocity, as ``volume_pipeline`` does) — plus
the streamline polylines concatenated into three flat arrays.  NaN (unsupported vertex) stays NaN.

Review, conclusion edits, findings decisions and case identifiers follow the job record and ``summary.json`` (they
are written after the report and a failed metadata refresh must not show stale text).  Nothing here writes a file.

Parsed reports are cached by ``(path, size, mtime_ns)`` (LRU, ``CACHE_JOBS`` jobs); arrays are decoded lazily, once.
"""
from __future__ import annotations

import base64
import copy
import hashlib
import json
import logging
import os
import threading
from collections import OrderedDict
from pathlib import Path
from typing import Any, Callable

import numpy as np

from .paths import BRANCH_CN

LOG = logging.getLogger("wss_deploy.service")

SCHEMA = "wss-deploy.v2-manifest/v1"
INPUTCHECK_SCHEMA = "wss-deploy.v2-inputcheck/v1"
# Bumped whenever the manifest assembly or a server-side derived array changes: it is part of ``data_version``.
CODE_VERSION = "v2_data/1"
CACHE_JOBS = 6
ARRAY_KEY_PATTERN = r"[A-Za-z0-9_.-]{1,64}"
MAX_INPUT_DISPLAY_FACES = 60000
DTYPES = {"float32": np.dtype("<f4"), "uint32": np.dtype("<u4"), "int32": np.dtype("<i4"), "uint8": np.dtype("u1")}
PLACEHOLDER_NAME = "病例"

# field id -> (label, short label, units); the classic viewer's labels without the trailing code name.
FIELD_TEXT = {
    "wss": ("壁面切应力", "WSS", "Pa"),
    "tawss": ("周期平均壁面切应力", "TAWSS", "Pa"),
    "osi": ("振荡剪切指数", "OSI", "1"),
    "rrt": ("相对滞留时间", "RRT", "1/Pa"),
    "ecap": ("内皮细胞激活势", "ECAP", "1/Pa"),
    "pressure": ("相对压力", "压力", "Pa"),
    "speed": ("速度大小", "速度", "m/s"),
    "velocity": ("速度矢量", "速度矢量", "m/s"),
    "wall_pressure": ("壁面相对压力", "壁面压力", "Pa"),
}
FIELD_DEFINITION = {
    "tawss": "一个心动周期内壁面切应力大小的时间平均（80 帧等权）",
    "osi": "½ ·（1 − |周期平均切应力矢量| / 周期平均 |切应力|）；0 为单向，0.5 为完全往复",
    "rrt": "1 / ((1 − 2·OSI) · TAWSS)，由预测的 TAWSS 与 OSI 逐点计算（TAWSS 取不小于 0.01 Pa，1 − 2·OSI 取不小于 0.01）",
    "ecap": "OSI / TAWSS，由预测的 TAWSS 与 OSI 逐点计算（TAWSS 取不小于 0.01 Pa）",
    "speed": "预测速度矢量的大小 |v|，由服务端逐点计算",
    "velocity": "体内查询点的预测速度矢量，STL 世界坐标",
    "wall_pressure": "壁面查询点的预测相对压力插值到显示网格顶点（仅用于着色，统计仍按查询点）",
}
CYCLE_FIELDS = ("tawss", "osi")
DERIVED_FIELDS = ("rrt", "ecap")
ABOVE_FIELDS = ("osi", "rrt", "ecap")
TRUST_BITS_DEFAULT = {"1": "interpolation_uncovered", "2": "rough_surface", "4": "geometry_out_of_range"}
BRANCH_KEY_OF_NAME = {name: key for key, name in BRANCH_CN.items()}


class DataError(ValueError):
    """The job's report cannot provide the requested data (missing file, missing or malformed embedded payload)."""

    def __init__(self, message: str, status: int = 404):
        super().__init__(message)
        self.status = status


# ---------------------------------------------------------------------------------------------------------- parsing
def _script_payload(text: str, script_id: str) -> str | None:
    """Raw text of ``<script id="<script_id>" …>…</script>`` (first closing tag; ``_script_json`` escapes ``<``)."""
    start = text.find('<script id="' + script_id + '"')
    if start < 0:
        return None
    body = text.find(">", start)
    end = text.find("</script>", body + 1) if body >= 0 else -1
    if body < 0 or end < 0:
        return None
    return text[body + 1:end]


def _b64_size(value: str) -> int | None:
    """Decoded byte count of a base64 string without decoding it (None when it cannot be valid base64)."""
    if not isinstance(value, str) or len(value) % 4:
        return None
    return len(value) // 4 * 3 - (2 if value.endswith("==") else 1 if value.endswith("=") else 0)


class _Spec:
    """One manifest array: an embedded base64 string (served as its decoded bytes) or a computed array."""
    __slots__ = ("key", "dtype", "width", "b64", "compute", "rows")

    def __init__(self, key: str, dtype: str, width: int, *, b64: str | None = None, compute: Callable | None = None,
                 rows: int | None = None):
        self.key, self.dtype, self.width, self.b64, self.compute, self.rows = key, dtype, int(width), b64, compute, rows

    @property
    def shape(self) -> list[int]:
        return [int(self.rows), self.width] if self.width > 1 else [int(self.rows)]

    @property
    def nbytes(self) -> int:
        return int(self.rows) * self.width * DTYPES[self.dtype].itemsize


class _Report:
    """Embedded data of one ``report.html`` (immutable once built; arrays decoded lazily)."""

    def __init__(self, path: Path, signature: tuple[int, int], text: str):
        self.path, self.signature = path, signature
        meta_raw = _script_payload(text, "wss-report-meta")
        if meta_raw is None:
            raise DataError("报告缺少嵌入的结果数据（wss-report-meta）。", 422)
        self.family = "volume" if 'id="volume-arrays"' in text else "wall"
        arrays_id = "volume-arrays" if self.family == "volume" else "wss-report-arrays"
        arrays_raw = _script_payload(text, arrays_id)
        if arrays_raw is None:
            raise DataError(f"报告缺少嵌入的数组（{arrays_id}）。", 422)
        try:
            meta = json.loads(meta_raw)
            arrays = json.loads(arrays_raw)
        except ValueError as exc:
            raise DataError("报告嵌入数据不是有效 JSON。", 422) from exc
        if not isinstance(meta, dict) or not isinstance(arrays, dict):
            raise DataError("报告嵌入数据不是 JSON 对象。", 422)
        if self.family == "wall":
            from .report import _report_schema_metadata
            meta = _report_schema_metadata(meta)     # pre-``fields`` reports: one fixed WSS frame (as the viewer reads them)
        self.meta, self.arrays = meta, arrays
        self.arrays_version = hashlib.sha256((CODE_VERSION + "\0" + arrays_raw).encode("utf-8")).hexdigest()[:16]
        self._bytes: dict[str, bytes] = {}
        self._lock = threading.Lock()
        self.specs: "OrderedDict[str, _Spec]" = OrderedDict()
        self.field_arrays: dict[str, dict[str, str | None]] = {}
        (self._wall_specs if self.family == "wall" else self._volume_specs)()

    # -- specs -----------------------------------------------------------------------------------------------------
    def _embedded(self, key: str, source: dict, name: str, dtype: str, width: int = 1, *, required: bool = False) -> str | None:
        raw = source.get(name) if isinstance(source, dict) else None
        size = _b64_size(raw) if raw is not None else None
        itemsize = DTYPES[dtype].itemsize
        if size is None or size % (itemsize * width):
            if required:
                raise DataError(f"报告缺少数组或数组形状无效：{name}", 422)
            if raw is not None:
                LOG.warning("v2 manifest: skipping malformed embedded array %s in %s", name, self.path)
            return None
        self.specs[key] = _Spec(key, dtype, width, b64=raw, rows=size // (itemsize * width))
        return key

    def _computed(self, key: str, dtype: str, width: int, rows: int, compute: Callable) -> str:
        self.specs[key] = _Spec(key, dtype, width, compute=compute, rows=rows)
        return key

    def _wall_specs(self) -> None:
        A = self.arrays
        dt = A.get("dt") if isinstance(A.get("dt"), dict) else {}
        seg = lambda name: "uint8" if dt.get(name) == "u8" else "int32"   # v0.14 pages record u8 branch ids in ``dt``
        self.mesh = {"vertices": self._embedded("mv", A, "mv", "float32", 3, required=True),
                     "faces": self._embedded("mf", A, "mf", "uint32", 3, required=True),
                     "segment": self._embedded("ms", A, "ms", seg("ms")),
                     "trust": self._embedded("mt", A, "mt", "uint8")}
        self.points = {"xyz": self._embedded("pv", A, "pv", "float32", 3, required=True),
                       "segment": self._embedded("ps", A, "ps", seg("ps")),
                       "s_from_root_mm": self._embedded("p_s", A, "p_s", "float32"),
                       "theta_rad": self._embedded("p_th", A, "p_th", "float32"),
                       "radius_mm": self._embedded("p_r", A, "p_r", "float32"),
                       "dist_to_junction_mm": self._embedded("p_dj", A, "p_dj", "float32")}
        self.centerline = {"xyz": self._embedded("cv", A, "cv", "float32", 3),
                           "radius_mm": self._embedded("cr", A, "cr", "float32"),
                           "edges": self._embedded("ce", A, "ce", "uint32", 2),
                           "segment": self._embedded("cs", A, "cs", seg("cs")),
                           "tangent": None}
        self.streamlines = None
        n_vertices, n_points = self.specs["mv"].rows, self.specs["pv"].rows
        self.field_arrays["wss"] = {"display": self._embedded("f.wss.d", A, "mw", "float32"),
                                    "read": self._embedded("f.wss.r", A, "pw", "float32", required=True)}
        fields = self.meta.get("fields") if isinstance(self.meta.get("fields"), dict) else {}
        xf = A.get("xf") if isinstance(A.get("xf"), dict) else {}
        for field_id, desc in fields.items():
            if field_id == "wss" or not isinstance(desc, dict) or field_id in DERIVED_FIELDS:
                continue
            item = xf.get(str(desc.get("array_key") or ""))
            if not isinstance(item, dict):
                continue
            display = self._embedded(f"f.{field_id}.d", item, "m", "float32")
            read = self._embedded(f"f.{field_id}.r", item, "p", "float32")
            if display and self.specs[display].rows != n_vertices or read and self.specs[read].rows != n_points:
                LOG.warning("v2 manifest: field %s of %s does not match the mesh / cloud sizes; skipped", field_id, self.path)
                for key in (display, read):
                    self.specs.pop(key, None)
                continue
            self.field_arrays[field_id] = {"display": display, "read": read}
        # RRT / ECAP: always recomputed from the embedded TAWSS / OSI (contract §2 派生量), as the classic viewer does.
        if all(self.field_arrays.get(k, {}).get("read") and self.field_arrays.get(k, {}).get("display") for k in CYCLE_FIELDS):
            for name in DERIVED_FIELDS:
                self.field_arrays[name] = {
                    "display": self._computed(f"f.{name}.d", "float32", 1, n_vertices, lambda n=name: self._derived(n, "display")),
                    "read": self._computed(f"f.{name}.r", "float32", 1, n_points, lambda n=name: self._derived(n, "read"))}

    def _volume_specs(self) -> None:
        A = self.arrays
        self.mesh = {"vertices": self._embedded("mv", A, "vertices", "float32", 3, required=True),
                     "faces": self._embedded("mf", A, "faces", "uint32", 3, required=True),
                     "segment": None,
                     "trust": self._embedded("mt", A, "wall_trust", "uint8")}
        self.points = {"xyz": self._embedded("vpts", A, "pts", "float32", 3, required=True),
                       "segment": self._embedded("vseg", A, "segment", "int32"),
                       "is_wall": self._embedded("vis_wall", A, "is_wall", "uint8"),
                       "trust": self._embedded("vtrust", A, "trust", "uint8"),
                       "s_from_root_mm": self._embedded("v_s", A, "s_from_root_mm", "float32"),
                       "radius_mm": self._embedded("v_r", A, "radius_mm", "float32"),
                       "dist_to_wall_mm": self._embedded("v_dw", A, "dist_to_wall_mm", "float32")}
        self.centerline = {"xyz": self._embedded("cv", A, "center", "float32", 3),
                           "radius_mm": self._embedded("cr", A, "center_radius_mm", "float32"),
                           "edges": self._embedded("ce", A, "edges", "uint32", 2),
                           "segment": self._embedded("cs", A, "center_segment", "int32"),
                           "tangent": self._embedded("ct", A, "tangent", "float32", 3)}
        n_points = self.specs["vpts"].rows
        pressure = self._embedded("f.pressure.r", A, "pressure_pa", "float32")
        velocity = self._embedded("f.velocity.r", A, "velocity_m_s", "float32", 3)
        if velocity and self.specs[velocity].rows != n_points:
            self.specs.pop(velocity, None); velocity = None
        if pressure and self.specs[pressure].rows != n_points:
            self.specs.pop(pressure, None); pressure = None
        if pressure:
            self.field_arrays["pressure"] = {"display": None, "read": pressure}
        if velocity:
            self.field_arrays["speed"] = {"display": None,
                                          "read": self._computed("f.speed.r", "float32", 1, n_points, self._speed)}
            self.field_arrays["velocity"] = {"display": None, "read": velocity}
        wall_pressure = self._embedded("f.wall_pressure.d", A, "wall_pressure_pa", "float32")
        if wall_pressure and self.specs[wall_pressure].rows != self.specs["mv"].rows:
            self.specs.pop(wall_pressure, None); wall_pressure = None
        if wall_pressure:
            self.field_arrays["wall_pressure"] = {"display": wall_pressure, "read": None}
        lines = [line for line in (A.get("streamlines") or []) if isinstance(line, dict)]
        sizes = []
        for line in lines:
            n_xyz, n_speed = _b64_size(line.get("points")), _b64_size(line.get("speed_m_s"))
            if n_xyz is None or n_speed is None or n_xyz % 12 or n_speed != n_xyz // 3:
                LOG.warning("v2 manifest: malformed streamline in %s; streamlines skipped", self.path)
                sizes = None
                break
            sizes.append(n_xyz // 12)
        self.streamlines = None
        if lines and sizes:
            total = int(sum(sizes))
            self._line_sizes = sizes
            self.streamlines = {"xyz": self._computed("sl.xyz", "float32", 3, total, lambda: self._streamline("xyz")),
                                "speed": self._computed("sl.speed", "float32", 1, total, lambda: self._streamline("speed")),
                                "offsets": self._computed("sl.offsets", "uint32", 1, len(sizes) + 1, lambda: self._streamline("offsets")),
                                "count": len(sizes), "n_points": total, "speed_units": "m/s"}

    # -- decoding --------------------------------------------------------------------------------------------------
    def raw(self, key: str) -> bytes:
        """Little-endian bytes of one array (decoded / computed once)."""
        spec = self.specs.get(key)
        if spec is None:
            raise DataError("数组不存在。", 404)
        with self._lock:
            hit = self._bytes.get(key)
        if hit is not None:
            return hit
        if spec.b64 is not None:
            try:
                data = base64.b64decode(spec.b64, validate=True)
            except (ValueError, TypeError) as exc:
                raise DataError(f"报告数组无法解码：{key}", 422) from exc
        else:
            data = np.ascontiguousarray(spec.compute(), dtype=DTYPES[spec.dtype]).tobytes()
        if len(data) != spec.nbytes:
            raise DataError(f"报告数组长度不符：{key}", 422)
        with self._lock:
            self._bytes[key] = data
        return data

    def array(self, key: str) -> np.ndarray:
        spec = self.specs[key]
        values = np.frombuffer(self.raw(key), dtype=DTYPES[spec.dtype])
        return values.reshape(spec.shape) if spec.width > 1 else values

    def b64(self, key: str) -> str:
        """Base64 of one array: the embedded string itself when the array is embedded, else of the computed bytes."""
        spec = self.specs[key]
        return spec.b64 if spec.b64 is not None else base64.b64encode(self.raw(key)).decode("ascii")

    def _derived(self, name: str, which: str) -> np.ndarray:
        from .cycle_fields import derive_indices
        tawss = self.array(self.field_arrays["tawss"][which]); osi = self.array(self.field_arrays["osi"][which])
        return derive_indices(tawss, osi)[name].astype(np.float32)

    def _speed(self) -> np.ndarray:
        velocity = np.asarray(self.array(self.field_arrays["velocity"]["read"]), np.float32)
        return np.linalg.norm(velocity, axis=1).astype(np.float32)          # volume_pipeline's definition, float32

    def _streamline(self, part: str) -> np.ndarray:
        lines = self.arrays.get("streamlines") or []
        if part == "offsets":
            return np.concatenate([[0], np.cumsum(self._line_sizes)]).astype(np.uint32)
        name = "points" if part == "xyz" else "speed_m_s"
        return np.frombuffer(b"".join(base64.b64decode(line[name], validate=True) for line in lines), dtype="<f4")


_REPORTS: "OrderedDict[str, _Report]" = OrderedDict()
_SUMMARIES: "OrderedDict[str, tuple[tuple, dict]]" = OrderedDict()
_CACHE_LOCK = threading.Lock()


def _read_report(path: Path) -> _Report:
    key = str(path)
    try:
        with open(path, "rb") as stream:
            stat = os.fstat(stream.fileno())
            signature = (int(stat.st_size), int(stat.st_mtime_ns))
            with _CACHE_LOCK:
                hit = _REPORTS.get(key)
                if hit is not None and hit.signature == signature:
                    _REPORTS.move_to_end(key)
                    return hit
            data = stream.read()
    except FileNotFoundError as exc:
        raise DataError("结果报告不存在。", 404) from exc
    except OSError as exc:
        raise DataError("结果报告无法读取。", 500) from exc
    try:
        text = data.decode("utf-8")
    except UnicodeError as exc:
        raise DataError("结果报告编码无效。", 422) from exc
    report = _Report(path, signature, text)
    with _CACHE_LOCK:
        _REPORTS[key] = report
        _REPORTS.move_to_end(key)
        while len(_REPORTS) > CACHE_JOBS:
            _REPORTS.popitem(last=False)
    return report


def _read_summary(path: Path) -> tuple[dict, int | None]:
    """``(summary, mtime_ns)``; ``({}, None)`` when the file is absent or unreadable."""
    key = str(path)
    try:
        with open(path, "rb") as stream:
            stat = os.fstat(stream.fileno())
            signature = (int(stat.st_size), int(stat.st_mtime_ns))
            with _CACHE_LOCK:
                hit = _SUMMARIES.get(key)
                if hit is not None and hit[0] == signature:
                    _SUMMARIES.move_to_end(key)
                    return hit[1], signature[1]
            data = stream.read()
    except OSError:
        return {}, None
    try:
        value = json.loads(data.decode("utf-8"))
    except (UnicodeError, ValueError):
        LOG.warning("v2 manifest: unreadable %s", path)
        return {}, signature[1]
    value = value if isinstance(value, dict) else {}
    with _CACHE_LOCK:
        _SUMMARIES[key] = (signature, value)
        _SUMMARIES.move_to_end(key)
        while len(_SUMMARIES) > CACHE_JOBS:
            _SUMMARIES.popitem(last=False)
    return value, signature[1]


def clear_cache() -> None:
    with _CACHE_LOCK:
        _REPORTS.clear(); _SUMMARIES.clear()


def cache_info() -> dict:
    with _CACHE_LOCK:
        return {"reports": list(_REPORTS), "summaries": list(_SUMMARIES), "limit": CACHE_JOBS}


class JobData:
    """The report + summary of one job directory, with the contract's ``data_version``."""

    def __init__(self, job_dir: Path):
        self.job_dir = Path(job_dir)
        self.job_id = self.job_dir.name
        self.report = _read_report(self.job_dir / "report.html")
        self.summary, summary_mtime = _read_summary(self.job_dir / "summary.json")
        size, mtime_ns = self.report.signature
        self.data_version = hashlib.sha256(f"{size}:{mtime_ns}:{summary_mtime}:{CODE_VERSION}".encode("ascii")).hexdigest()[:16]

    def array_url(self, key: str) -> str:
        return f"/api/v2/jobs/{self.job_id}/arrays/{key}"

    def array_etag(self, key: str) -> str:
        return f'W/"a-{self.data_version}-{key}"'


def load(job_dir: Path) -> JobData:
    return JobData(job_dir)


# --------------------------------------------------------------------------------------------------------- manifest
def _dict(value) -> dict:
    return value if isinstance(value, dict) else {}


def _num(value):
    """A finite JSON number or None."""
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if np.isfinite(number) else None


def _stats(values: np.ndarray) -> dict:
    finite = np.asarray(values, np.float64)
    finite = finite[np.isfinite(finite)]
    if not finite.size:
        return {"p99": None, "max": None}
    return {"p99": float(np.quantile(finite, 0.99)), "max": float(finite.max())}


def _branch_key(name) -> str | None:
    return BRANCH_KEY_OF_NAME.get(str(name)) if name is not None else None


def _branches(meta: dict) -> list[dict]:
    names = {str(k): str(v) for k, v in _dict(meta.get("branch_names")).items()}
    geometry = _dict(meta.get("geometry")) or _dict(meta.get("branch_geometry"))
    out, seen = [], set()
    for item in _dict(meta.get("profiles")).get("branches") or []:
        if not isinstance(item, dict) or item.get("segment_id") is None:
            continue
        sid = int(item["segment_id"])
        name = item.get("name") or names.get(str(sid))
        parent = item.get("parent_id")
        out.append({"id": sid, "key": _branch_key(name), "name": name, "parent": int(parent) if parent is not None else None,
                    "length_mm": _num(item.get("length_mm") if item.get("length_mm") is not None else _dict(geometry.get(name)).get("length_mm"))})
        seen.add(sid)
    for sid, name in sorted(names.items(), key=lambda kv: int(kv[0]) if kv[0].lstrip("-").isdigit() else 0):
        if not sid.lstrip("-").isdigit() or int(sid) in seen:
            continue
        out.append({"id": int(sid), "key": _branch_key(name), "name": name, "parent": None,
                    "length_mm": _num(_dict(geometry.get(name)).get("length_mm"))})
    return sorted(out, key=lambda b: b["id"])


def _openings(meta: dict) -> list[dict]:
    """Inlet / outlet openings of the confirmed result (the endpoints block, else the volume closure caps)."""
    out = []
    for item in meta.get("endpoints") or []:
        if isinstance(item, dict):
            name = item.get("name")
            out.append({"name": name, "label": item.get("name_cn") or BRANCH_CN.get(str(name)) or name, "kind": item.get("kind"),
                        "segment_id": item.get("segment_id"), "center_mm": item.get("center_mm"),
                        "radius_mm": _num(item.get("radius_mm")), "flow_share": _num(item.get("share"))})
    if out:
        return out
    caps = _dict(_dict(meta.get("volume_geometry")).get("closure")).get("caps") or []
    for item in caps:
        if isinstance(item, dict):
            name = item.get("label")
            out.append({"name": name, "label": "入口（主动脉）" if name == "inlet" else BRANCH_CN.get(str(name)) or name,
                        "kind": "inlet" if name == "inlet" else "outlet", "segment_id": None,
                        "center_mm": item.get("center_mm"), "radius_mm": _num(item.get("radius_mm")), "flow_share": None})
    return out


def _orientation(meta: dict, record: dict) -> tuple[dict, dict]:
    frame = _dict(meta.get("frame_transform"))
    direction = frame.get("direction_source") or _dict(meta.get("input_check")).get("orientation_source")
    history = record.get("mapping_history") if isinstance(record.get("mapping_history"), list) else None
    if history is None:
        history = _dict(meta.get("audit")).get("mapping_history")
    last = history[-1] if isinstance(history, list) and history and isinstance(history[-1], dict) else {}
    # ``outlets_confirmed`` is set on every stage-B run; a name accepted by the automatic high-confidence gate was
    # never looked at by a person, so it stays "inferred".
    automatic = last.get("source") == "automatic_high_confidence" or last.get("acknowledged") is False
    confirmed = bool(meta.get("outlets_confirmed")) and not automatic
    if frame.get("rotation") is None:
        anterior = "unknown"
    else:
        anterior = "confirmed" if direction and direction not in {"unknown_stl", "unknown"} else "inferred"
    block = {"rotation": frame.get("rotation"), "origin_mm": frame.get("origin_mm"), "convention": frame.get("convention"),
             "source": frame.get("source"),
             "orientation": {"left_right": "confirmed" if confirmed else "inferred", "superior_inferior": "derived",
                             "anterior_posterior": anterior},
             "direction_source": direction or "unknown_stl"}
    if frame.get("vector_to_world"):
        block["vector_to_world"] = frame["vector_to_world"]
    return block, last


def _time(meta: dict) -> dict:
    axis = []
    for index, item in enumerate(meta.get("time_axis") or []):
        if isinstance(item, dict):
            axis.append({"index": item.get("index", index), "time_s": _num(item.get("time_s")), "label": item.get("label"),
                         **({"step": item["step"]} if item.get("step") is not None else {})})
    if not axis:
        frame = _dict(meta.get("model_frame"))
        axis = [{"index": 0, "time_s": _num(frame.get("time_s")), "label": frame.get("label") or frame.get("target")}]
    definition = _dict(_dict(meta.get("cycle")).get("definition"))
    cycle = None
    if definition:
        cycle = {"period_s": _num(definition.get("period_s")), "frames": definition.get("frames"), "weight": definition.get("weight"),
                 "peak_frame": definition.get("peak_frame")}
    return {"mode": "single_frame" if len(axis) == 1 else "frames", "axis": axis, "cycle": cycle}


def _card_parts(card: dict | None, field_id: str) -> tuple[list, dict | None]:
    if not isinstance(card, dict):
        return [], None
    windows = _dict(card.get("display_windows")).get(field_id)
    validation = _dict(card.get("validation"))
    per_field = _dict(validation.get("fields")).get(field_id)
    out = None
    if isinstance(per_field, dict):
        out = {"holdout_n": validation.get("holdout_n"), **per_field}
        if validation.get("source") and "source" not in out:
            out["source"] = validation["source"]
    return (list(windows) if isinstance(windows, list) else []), out


def _field(report: _Report, field_id: str, *, card: dict | None) -> dict:
    meta = report.meta
    desc = _dict(_dict(meta.get("fields")).get(field_id))
    label, short, units = FIELD_TEXT.get(field_id, (desc.get("label") or field_id, desc.get("label") or field_id, desc.get("units") or ""))
    arrays = report.field_arrays.get(field_id, {})
    display = _dict(desc.get("display"))
    cycle_stats = _dict(_dict(_dict(meta.get("cycle")).get("fields")).get(field_id))
    statistics: dict | None = None
    tier, source, derived_from, temporal = "model", "prediction", [], "frame"
    definition = FIELD_DEFINITION.get(field_id)
    block = {"thresholds": None, "threshold_direction": None, "log_scale": False, "range": None, "p99": None, "max": None}
    if field_id == "wss":
        field_pa = _dict(meta.get("wss_field_pa")); peak = _dict(meta.get("peak"))
        thresholds = field_pa.get("thresholds_pa") or [0.4, 4.0, 7.0]
        block.update(thresholds=[float(t) for t in thresholds], threshold_direction="below",
                     p99=_num(peak.get("p99_pa", field_pa.get("p99"))), max=_num(peak.get("max_pa", field_pa.get("max"))))
        statistics = {k: field_pa[k] for k in ("n", "mean", "median", "p95", "p99", "max", "min", "area_frac_low", "area_frac_high",
                                                "area_frac_very_high", "area_low_mm2", "area_high_mm2") if k in field_pa} or None
        frame = _dict(meta.get("model_frame"))
        time_s = _num(frame.get("time_s"))
        definition = f"模型在固定收缩期帧（t = {time_s:g} s）预测的壁面切应力大小" if time_s is not None else "模型在固定收缩期帧预测的壁面切应力大小"
    elif field_id in CYCLE_FIELDS or field_id in DERIVED_FIELDS:
        from . import cycle_fields as CF
        spec = CF.FIELD_SPECS[field_id]
        temporal = "cycle_summary"
        if field_id in DERIVED_FIELDS:
            tier, source, derived_from = "derived", "derived", ["tawss", "osi"]
        clip = spec.get("clip") or (0.0, None)
        block.update(thresholds=[float(t) for t in (display.get("thresholds") or spec["thresholds"])],
                     threshold_direction=display.get("threshold_direction") or ("above" if field_id in ABOVE_FIELDS else "below"),
                     log_scale=bool(display.get("log_scale", spec.get("log_scale"))),
                     range=display.get("range") if display.get("range") is not None else ([float(clip[0]), float(clip[1])] if clip[1] is not None else None),
                     p99=_num(display.get("p99", cycle_stats.get("p99"))), max=_num(display.get("max", cycle_stats.get("max"))))
        if block["p99"] is None and arrays.get("read"):
            block.update(_stats(report.array(arrays["read"])))
        statistics = {k: v for k, v in cycle_stats.items() if k not in {"per_branch", "definition"}} or None
    else:
        volume = _dict(meta.get("volume_statistics"))
        stats_key = {"pressure": "pressure_interior_pa", "speed": "speed_m_s", "velocity": "speed_m_s",
                     "wall_pressure": "pressure_wall_pa"}.get(field_id)
        statistics = dict(volume.get(stats_key)) if isinstance(volume.get(stats_key), dict) else None
        if statistics:
            block.update(p99=_num(statistics.get("p99")), max=_num(statistics.get("max")))
        if field_id == "pressure":
            definition = meta.get("pressure_reference") or "相对于该病例当前帧的体积平均压力；不可解释为绝对血压。"
        if field_id == "speed":
            derived_from = ["velocity"]
    kind = "vector" if field_id == "velocity" else "scalar"
    windows, validation = _card_parts(card, field_id)
    location = "wall" if report.family == "wall" or field_id == "wall_pressure" else (desc.get("location") or "interior")
    return {"id": field_id, "label": label, "short_label": short, "units": desc.get("units") or units,
            "location": location, "kind": kind, "components": 3 if kind == "vector" else 1,
            "temporal": temporal, "time_index": None if temporal == "cycle_summary" else 0,
            "source": source, "tier": tier, "derived_from": derived_from, "definition": definition,
            "arrays": {"display": arrays.get("display"), "read": arrays.get("read")},
            "display": block, "windows": windows, "validation": validation, "statistics": statistics}


FIELD_ORDER = {"wall": ("wss", "tawss", "osi", "rrt", "ecap"), "volume": ("pressure", "speed", "velocity", "wall_pressure")}


def _fields(report: _Report, card: dict | None) -> list[dict]:
    order = list(FIELD_ORDER[report.family]) + [k for k in report.field_arrays if k not in FIELD_ORDER[report.family]]
    return [_field(report, field_id, card=card) for field_id in order if field_id in report.field_arrays]


def _review(record: dict, summary: dict, meta: dict) -> dict:
    for source in (record.get("review"), summary.get("review"), meta.get("review")):
        if isinstance(source, dict) and source.get("status"):
            return {"status": source.get("status"), "by": source.get("by") or "", "at": source.get("at") or "",
                    "note": source.get("note") or "", "version": source.get("version")}
    return {"status": "unreviewed", "by": "", "at": "", "note": "", "version": None}


def _narrative(record: dict, summary: dict, meta: dict) -> dict:
    """Latest conclusion: summary.json (rewritten on every edit) over the report copy; the record's edit wins."""
    from .narrative import merge_edit
    block = summary.get("narrative") if isinstance(summary.get("narrative"), dict) else meta.get("narrative")
    block = dict(block) if isinstance(block, dict) else {}
    document = record.get("narrative")
    if isinstance(document, dict) and "edited" in document:
        base = {k: v for k, v in block.items() if k not in {"edited", "edited_by", "edited_at"}} or _dict(document.get("auto"))
        block = merge_edit(base, document.get("edited"), edited_by=document.get("edited_by"), edited_at=document.get("edited_at"))
    block.setdefault("zh", []); block.setdefault("en", []); block.setdefault("edited", None)
    return block


def _findings(summary: dict, meta: dict) -> dict:
    findings = dict(_dict(meta.get("findings"))) if isinstance(meta.get("findings"), dict) else {"items": []}
    review = _dict(summary.get("findings")).get("review")
    if isinstance(review, dict):
        findings["review"] = review
    findings.setdefault("items", [])
    return findings


def _display_name(record: dict) -> str:
    for key in ("display_name", "patient_id", "case_id"):
        value = record.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return ""


def build_manifest(data: JobData, record: dict, *, card: dict | None = None, companions: list | None = None) -> dict:
    """Contract §3 manifest of a finished job.

    ``record`` is the job record (the server passes the owner-checked record; keys used: id, status, case_id,
    patient_id, scan_label, scan_date, created_at, input_sha256, review, narrative, display_name, source_filename,
    model_release, run_identity, mapping_history).  ``companions`` overrides ``record["companions"]``.
    """
    report, summary = data.report, data.summary
    meta = report.meta
    case_metadata = _dict(summary.get("case_metadata")) or _dict(meta.get("case_metadata"))
    release = _dict(record.get("model_release"))
    release_id = release.get("id") or release.get("release") or meta.get("release") or _dict(meta.get("model_release")).get("release")
    narrative = _narrative(record, summary, meta)
    frame, _ = _orientation(meta, record)
    identity = {"case_id": record.get("case_id") or summary.get("case_id") or meta.get("case_id") or "",
                **{key: record.get(key) if record.get(key) is not None else case_metadata.get(key, "")
                   for key in ("patient_id", "scan_label", "scan_date")}}
    job = {"id": data.job_id, "status": record.get("status") or "done",
           # §3: the record's display_name (C7), else a non-empty patient id, else the case id.
           "display_name": _display_name({"display_name": record.get("display_name"), **identity}),
           **identity,
           "created_at": record.get("created_at") or meta.get("created_at"),
           "input_sha256": record.get("input_sha256") or meta.get("input_sha256"),
           "source_filename": record.get("source_filename") or "",
           "review": _review(record, summary, meta),
           "companions": list(companions if companions is not None else (record.get("companions") or [])),
           "narrative_edited": bool(narrative.get("edited"))}
    geometry = {"display_mesh": dict(report.mesh), "points": dict(report.points), "centerline": dict(report.centerline),
                "branches": _branches(meta), "openings": _openings(meta),
                "streamlines": ({k: v for k, v in report.streamlines.items()} if report.streamlines else None)}
    trust = _dict(meta.get("trust"))
    interpolation = _dict(meta.get("interpolation"))
    arrays = {key: {"dtype": spec.dtype, "shape": spec.shape, "bytes": spec.nbytes, "url": data.array_url(key)}
              for key, spec in report.specs.items()}
    model_release = _dict(meta.get("model_release"))
    manifest = {
        "schema": SCHEMA,
        "data_version": data.data_version,
        "arrays_version": report.arrays_version,
        "job": job,
        "result": {"run_identity": record.get("run_identity") or meta.get("run_identity"), "family": report.family,
                   "release_id": release_id, "display_name": card.get("display_name") if isinstance(card, dict) else None,
                   "analysis_version": meta.get("analysis_version"), "deploy_version": meta.get("deploy_version"),
                   "model_frame": {k: _dict(meta.get("model_frame")).get(k) for k in ("target", "time_s")}},
        "units": {"length": "mm"},
        "frame": frame,
        "time": _time(meta),
        "geometry": geometry,
        "fields": _fields(report, card),
        "arrays": arrays,
        "mapping": {"display_interpolation": {k: interpolation.get(k) for k in ("method", "sigma_mm", "max_dist_mm")} if interpolation else None,
                    "trust_bits": trust.get("bits") or (dict(TRUST_BITS_DEFAULT) if report.family == "wall" and report.mesh.get("trust") else {}),
                    "trust_sources": trust.get("sources") or [],
                    "statistics_protocol": _dict(meta.get("statistics_protocol"))},
        "analysis": {"narrative": narrative, "findings": _findings(summary, meta),
                     "zones": meta.get("zones") if isinstance(meta.get("zones"), dict) else (summary.get("zones") if isinstance(summary.get("zones"), dict) else None),
                     "morphology": _dict(meta.get("morphology")), "profiles": _dict(meta.get("profiles")),
                     "per_branch": _dict(meta.get("per_branch")), "cycle": meta.get("cycle") if isinstance(meta.get("cycle"), dict) else None,
                     "peak": _dict(meta.get("peak")), "surface_statistics": _dict(meta.get("surface_statistics")),
                     "reference_assessment": _dict(meta.get("reference_assessment")), "trust": trust, "quality": _dict(meta.get("quality")),
                     "volume_statistics": meta.get("volume_statistics") if isinstance(meta.get("volume_statistics"), dict) else None,
                     "streamlines": meta.get("streamlines") if isinstance(meta.get("streamlines"), dict) else None,
                     "modules": report.arrays.get("modules") if isinstance(report.arrays.get("modules"), list) else None,
                     "pressure_reference": meta.get("pressure_reference"),
                     "branch_geometry": _dict(meta.get("geometry")) or _dict(meta.get("branch_geometry")),
                     "annotations": summary.get("annotations") if isinstance(summary.get("annotations"), dict) else meta.get("annotations"),
                     "flags": list(meta.get("flags") or [])},
        "model_card": card if isinstance(card, dict) else None,
        "provenance": {"release_hash": meta.get("release_hash"), "release_fingerprint": release.get("fingerprint"),
                       "model_release": {k: model_release.get(k) for k in ("name", "release", "frozen_on", "git_commit", "target") if model_release.get(k) is not None},
                       "feature_contract": _dict(meta.get("feature_contract")), "git_describe": meta.get("git_describe"),
                       "code_source_hash": meta.get("code_source_hash"), "timing_s": _dict(meta.get("timing_s")),
                       "interpolation": interpolation, "device": meta.get("device"), "report_schema": meta.get("schema_version")},
        "reserved": {"time_series": None},
    }
    return manifest


def manifest_etag(manifest: dict, body: bytes | None = None) -> str:
    """ETag of a manifest: ``data_version`` plus a digest of the serialised manifest (``body`` when the caller already
    has it), so a change in the job record or the model card revalidates too."""
    if body is None:
        body = json.dumps(manifest, ensure_ascii=False, allow_nan=False).encode("utf-8")
    return f'W/"m-{manifest["data_version"]}-{hashlib.sha256(body).hexdigest()[:12]}"'


def arrays_b64(data: JobData, keys=None) -> dict[str, str]:
    """``{key: base64}`` for the offline package: embedded strings are reused as they are, computed arrays encoded."""
    report = data.report
    return {key: report.b64(key) for key in (keys if keys is not None else report.specs)}


# ------------------------------------------------------------------------------------------------------ model cards
def _model_cards_module():
    try:
        from . import model_cards  # written by the C-line analysis track (contract §4); optional until it lands
    except ImportError as exc:
        if getattr(exc, "name", None) not in {"wss_deploy.model_cards", "model_cards"}:
            LOG.warning("wss_deploy.model_cards failed to import: %s", exc)
        return None
    return model_cards


def load_card(release_id: str | None, release_dir: Path | None = None) -> dict | None:
    """``model_cards.load(release_id, release_dir)``; None when the module or the card is missing or fails."""
    module = _model_cards_module()
    if module is None or not release_id or not hasattr(module, "load"):
        return None
    try:
        card = module.load(release_id, release_dir)
    except Exception:  # noqa: BLE001 — a broken card must not break the result view
        LOG.exception("Model card of %s could not be loaded", release_id)
        return None
    return card if isinstance(card, dict) else None


def all_cards(registry, release_ids) -> dict:
    """``{release_id: card | None}`` for every release the registry lists (contract §2 ``/api/v2/model-cards``)."""
    ids = [str(rid) for rid in release_ids if rid]
    module = _model_cards_module()
    cards: dict = {}
    if module is not None and registry is not None and hasattr(module, "all_cards"):
        try:
            value = module.all_cards(registry)
            if isinstance(value, dict):
                cards = {str(k): (v if isinstance(v, dict) else None) for k, v in value.items()}
        except Exception:  # noqa: BLE001
            LOG.exception("model_cards.all_cards failed; loading the cards one by one")
            cards = {}
    for rid in ids:
        if rid not in cards:
            cards[rid] = load_card(rid, release_dir(registry, rid))
    return cards


def release_dir(registry, release_id: str) -> Path | None:
    root = getattr(registry, "root", None)
    if root is None:
        return None
    path = Path(root) / release_id
    return path if path.is_dir() else None


# ---------------------------------------------------------------------------------------------------- hide the name
def scrub_names(value, names) -> Any:
    """Copy of a JSON value in which every string equal to (or, for names of ≥ 3 characters, containing) one of
    ``names`` is replaced by ``PLACEHOLDER_NAME``.  Keys are left alone; numbers never change."""
    names = sorted({str(n).strip() for n in names if isinstance(n, str) and str(n).strip()}, key=len, reverse=True)
    if not names:
        return copy.deepcopy(value)

    def text(item: str) -> str:
        if item.strip() in names:
            return PLACEHOLDER_NAME
        for name in names:
            if len(name) >= 3 and name in item:
                item = item.replace(name, PLACEHOLDER_NAME)
        return item

    def walk(item):
        if isinstance(item, dict):
            return {key: walk(sub) for key, sub in item.items()}
        if isinstance(item, list):
            return [walk(sub) for sub in item]
        if isinstance(item, str):
            return text(item)
        return item
    return walk(value)


def sensitive_names(manifest: dict, record: dict | None = None) -> list[str]:
    """Strings that identify the case: display name, case id, patient id and the uploaded file name (with and
    without its extension)."""
    job = _dict(manifest.get("job"))
    out = []
    for source in (job, record or {}):
        for key in ("display_name", "case_id", "patient_id", "source_filename", "filename"):
            value = source.get(key)
            if isinstance(value, str) and value.strip() and value.strip() != PLACEHOLDER_NAME:
                out.append(value.strip())
                stem = Path(value.strip()).stem
                if stem and stem != value.strip():
                    out.append(stem)
    return [name for name in dict.fromkeys(out) if name not in {"input", "input.stl", "input_clean_mm", "input_clean_mm.stl"}]


def hide_names(manifest: dict, record: dict | None = None, *, extra=None) -> tuple[dict, list]:
    """``(manifest copy, [extra copies])`` with the case name replaced by 「病例」 everywhere (contract §6.3)."""
    names = sensitive_names(manifest, record)
    out = scrub_names(manifest, names)
    job = out.setdefault("job", {})
    job["display_name"] = PLACEHOLDER_NAME
    job["case_id"] = PLACEHOLDER_NAME
    job["patient_id"] = PLACEHOLDER_NAME if _dict(manifest.get("job")).get("patient_id") else ""
    job["source_filename"] = PLACEHOLDER_NAME if _dict(manifest.get("job")).get("source_filename") else ""
    return out, [scrub_names(item, names) for item in (extra or [])]


# ------------------------------------------------------------------------------------------------------ input check
_INPUT_MESHES: "OrderedDict[tuple, dict]" = OrderedDict()


def _mesh_payload(vertices, faces, *, source: str, scale_factor: float | None, units: str) -> dict:
    from .preview import display_mesh
    v = np.asarray(vertices, np.float64); f = np.asarray(faces, np.int64).reshape(-1, 3)
    if len(f) > MAX_INPUT_DISPLAY_FACES:
        v, f, method = display_mesh(v, f, MAX_INPUT_DISPLAY_FACES)
    else:
        method = "full"
    v32 = np.ascontiguousarray(v, dtype="<f4"); f32 = np.ascontiguousarray(f, dtype="<u4")
    return {"vertices": base64.b64encode(v32.tobytes()).decode("ascii"), "faces": base64.b64encode(f32.tobytes()).decode("ascii"),
            "n_vertices": int(len(v32)), "n_faces": int(len(f32)), "source": source, "method": method,
            "units": units, "scale_factor": scale_factor, "display_only": True}


def _stl_mesh(path: Path, *, source: str, scale_factor: float | None) -> dict | None:
    """Display mesh of an STL inside the job directory (memory-cached by path, size and mtime; nothing written)."""
    try:
        stat = path.stat()
    except OSError:
        return None
    key = (str(path), stat.st_size, stat.st_mtime_ns, scale_factor)
    with _CACHE_LOCK:
        hit = _INPUT_MESHES.get(key)
        if hit is not None:
            _INPUT_MESHES.move_to_end(key)
            return hit
    try:
        from .ingest import _read_checked
        vertices, faces = _read_checked(path)
    except Exception as exc:  # noqa: BLE001 — an unreadable STL is exactly what a failed job may hold
        LOG.info("v2 inputcheck: %s not readable for display: %s", path.name, exc)
        return None
    factor = float(scale_factor) if scale_factor else 1.0
    payload = _mesh_payload(np.asarray(vertices, np.float64) * factor, faces, source=source, scale_factor=scale_factor,
                            units="mm" if scale_factor else "source")
    with _CACHE_LOCK:
        _INPUT_MESHES[key] = payload
        while len(_INPUT_MESHES) > CACHE_JOBS:
            _INPUT_MESHES.popitem(last=False)
    return payload


def _contained(job_dir: Path, name) -> Path | None:
    if not isinstance(name, str) or not name:
        return None
    path = Path(name)
    path = path if path.is_absolute() else job_dir / path
    try:
        resolved = path.resolve()
        if resolved.parent != job_dir.resolve() or not resolved.is_file():
            return None
    except OSError:
        return None
    return resolved


def build_inputcheck(job_dir: Path, record: dict, stage_a: dict | None, *, input_filename: str | None = None) -> dict:
    """Contract §2 ``/inputcheck``: what the input check and stage A left for a job (any status), or null / [].

    Display mesh, in order: the stage-A preview (a connected ≤ 18k-face surface), the cleaned STL, the uploaded STL
    (scaled to mm when the input check recorded the factor), each reduced to ≤ 60k faces in memory."""
    job_dir = Path(job_dir)
    a = stage_a if isinstance(stage_a, dict) else {}
    check = a.get("input_check") if isinstance(a.get("input_check"), dict) else None
    input_check = None
    if check is not None:
        input_check = dict(check)
        quality = check.get("quality") if isinstance(check.get("quality"), dict) else None
        input_check["quality"] = quality
        input_check["opening_geometry"] = (quality or {}).get("opening_geometry") if quality else check.get("opening_geometry")
    centerline = a.get("centerline") if isinstance(a.get("centerline"), dict) else {}
    openings = [dict(item) for item in (centerline.get("openings") or []) if isinstance(item, dict)]
    proposal = a.get("proposal") if isinstance(a.get("proposal"), dict) else None
    polylines = []
    mapping_proposal = None
    if proposal is not None:
        polylines = list(proposal.get("preview_polylines") or [])
        mapping_proposal = {k: v for k, v in proposal.items() if k != "preview_polylines"}
    mesh = None
    preview = a.get("preview") if isinstance(a.get("preview"), dict) else None
    from .preview import needs_upgrade
    if preview and preview.get("vertices") and preview.get("faces") and not needs_upgrade(preview):
        try:
            mesh = _mesh_payload(np.asarray(preview["vertices"], np.float64), np.asarray(preview["faces"], np.int64),
                                 source="stage_a_preview", scale_factor=None, units="mm")
            mesh["method"] = preview.get("method") or mesh["method"]
        except (ValueError, TypeError):
            mesh = None
    if mesh is None:
        clean = _contained(job_dir, (check or {}).get("clean_stl")) or _contained(job_dir, "input_clean_mm.stl")
        if clean is not None:
            mesh = _stl_mesh(clean, source="clean_stl", scale_factor=1.0)
    if mesh is None:
        raw = _contained(job_dir, input_filename or "input.stl")
        if raw is not None:
            factor = (check or {}).get("scale_factor")
            mesh = _stl_mesh(raw, source="input_stl", scale_factor=float(factor) if _num(factor) else None)
    confirmed = record.get("mapping") if isinstance(record.get("mapping"), dict) and record.get("mapping") else None
    return {"schema": INPUTCHECK_SCHEMA, "job_id": job_dir.name, "status": record.get("status"), "stage": record.get("stage"),
            "phase": record.get("phase"), "detail": record.get("detail"), "version": record.get("version"),
            "error": record.get("error") if isinstance(record.get("error"), dict) else None,
            "input_check": input_check, "openings": openings, "mesh": mesh, "mapping_proposal": mapping_proposal,
            "confirmed_mapping": confirmed, "centerline_polylines": polylines,
            "stage_a_created_at": a.get("created_at")}


__all__ = ["CACHE_JOBS", "CODE_VERSION", "DataError", "JobData", "PLACEHOLDER_NAME", "SCHEMA", "all_cards", "arrays_b64",
           "build_inputcheck", "build_manifest", "cache_info", "clear_cache", "hide_names", "load", "load_card", "manifest_etag", "release_dir",
           "scrub_names", "sensitive_names"]

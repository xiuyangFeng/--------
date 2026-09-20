"""HDF5 layout of a V5 case bundle plus digests and manifest helpers."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import h5py
import numpy as np

from . import contract as C


def array_digest(array: np.ndarray) -> str:
    a = np.ascontiguousarray(array)
    h = hashlib.sha256()
    h.update(str(a.dtype).encode())
    h.update(str(a.shape).encode())
    h.update(a.tobytes())
    return h.hexdigest()


class CaseWriter:
    def __init__(self, path: Path):
        self.path = Path(path)
        self.tmp = self.path.with_suffix(".h5.tmp")
        if self.tmp.exists():
            self.tmp.unlink()
        self.h5 = h5py.File(self.tmp, "w")
        self.datasets: dict[str, dict[str, Any]] = {}

    def attrs(self, group: str, payload: dict[str, Any]) -> None:
        g = self.h5.require_group(group)
        for key, value in payload.items():
            if isinstance(value, (dict, list, tuple)):
                g.attrs[key] = json.dumps(value, ensure_ascii=False)
            elif value is None:
                g.attrs[key] = "null"
            else:
                g.attrs[key] = value

    def write(self, name: str, array: np.ndarray, *, units: str, tag: str, description: str = "",
              chunk_points: int | None = None, compress: bool | None = None) -> None:
        array = np.asarray(array)
        kwargs: dict[str, Any] = {}
        if array.ndim >= 2 and array.shape[0] == len(C.EXPECTED_STEPS) and array.shape[1] > 1024:
            n = min(chunk_points or C.HDF5_CHUNK_POINTS, array.shape[1])
            kwargs["chunks"] = (1, n) + tuple(array.shape[2:])
            if compress is None or compress:
                kwargs["compression"] = C.HDF5_COMPRESSION
        elif array.ndim >= 1 and array.size > 4096:
            kwargs["chunks"] = True
            if compress:
                kwargs["compression"] = C.HDF5_COMPRESSION
        ds = self.h5.create_dataset(name, data=array, **kwargs)
        ds.attrs["units"] = units
        ds.attrs["dependency"] = tag
        if description:
            ds.attrs["description"] = description
        self.datasets[name] = {
            "shape": list(array.shape), "dtype": str(array.dtype), "units": units, "dependency": tag,
            "sha256": array_digest(array), "description": description,
        }

    def close(self) -> Path:
        self.h5.flush()
        self.h5.close()
        self.tmp.replace(self.path)
        return self.path

    def abort(self) -> None:
        try:
            self.h5.close()
        finally:
            if self.tmp.exists():
                self.tmp.unlink()


def aggregate_digest(datasets: dict[str, dict[str, Any]]) -> str:
    h = hashlib.sha256()
    for name in sorted(datasets):
        h.update(name.encode())
        h.update(datasets[name]["sha256"].encode())
    return h.hexdigest()

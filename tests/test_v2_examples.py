"""wss_deploy.v2_examples: the example STL keeps every triangle and drops the source header (it carries a case name)."""
from __future__ import annotations

import struct

import pytest

from wss_deploy import v2_examples


def _binary_stl(header: bytes, n: int = 3) -> bytes:
    tri = b"".join(struct.pack("<12fH", *([0.0] * 3), *(float(i) for i in range(9)), 0) for _ in range(n))
    return header.ljust(80, b"\0") + struct.pack("<I", n) + tri


def test_example_stl_replaces_header_and_keeps_triangles(tmp_path):
    src = tmp_path / "in.stl"; src.write_bytes(_binary_stl(b"PATIENT_NAME<stl unit=MM>"))
    out = v2_examples.write_example_stl(src, tmp_path / "out.stl")
    data, raw = out.read_bytes(), src.read_bytes()
    assert b"PATIENT_NAME" not in data and data.startswith(v2_examples.STL_HEADER)
    assert data[80:] == raw[80:] and len(data) == len(raw)


def test_example_stl_refuses_ascii_and_truncated_files(tmp_path):
    ascii_stl = tmp_path / "a.stl"; ascii_stl.write_text("solid NAME\nfacet normal 0 0 1\nendsolid NAME\n")
    with pytest.raises(ValueError):
        v2_examples.write_example_stl(ascii_stl, tmp_path / "x.stl")
    short = tmp_path / "s.stl"; short.write_bytes(_binary_stl(b"x", 3)[:-10])
    with pytest.raises(ValueError):
        v2_examples.write_example_stl(short, tmp_path / "y.stl")

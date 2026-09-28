"""Fluent 2023R1 (v231) TUI journals. Prompt sequences were established by probing v231 in batch mode (2026-09-27):
every prompt needs an explicit answer (blank journal lines are skipped, not taken as defaults) and list prompts take a
parenthesised list."""
from __future__ import annotations

from pathlib import Path

from cfd_auto import guard


def _finish(lines: list[str], workdir: Path) -> str:
    text = "\n".join(lines) + "\n"
    guard.check_journal(text, workdir)
    return text


def meshing(workdir: Path, surface: Path, walls: list[str], all_zones: list[str], out_mesh: Path, n_layers: int = 10,
            first_aspect_ratio: float = 20, growth: float = 1.2, tet_volume_growth: float = 1.2) -> str:
    """Boundary mesh -> prisms (aspect-ratio offsets, geometric growth) on every wall + tet fill; one cell zone per region
    (``merge cell zones? yes`` puts each region's prisms into its tet zone and folds the prism side quads into the caps)."""
    return _finish([
        f"/file/read-boundary-mesh {surface}",
        f"/boundary/check-boundary-mesh ({' '.join(all_zones)})",
        "/boundary/manage/list",
        f"/mesh/prism/controls/zone-specific-growth/apply-growth ({' '.join(walls)}) aspect-ratio geometric {n_layers} {first_aspect_ratio:g} {growth:g} no",
        "/mesh/prism/controls/zone-specific-growth/list-growth",
        f"/mesh/tet/controls/cell-sizing geometric {tet_volume_growth:g}",
        '/mesh/auto-mesh "" zone-specific pyramids tet yes yes',
        "/mesh/manage/list",
        f"/file/write-mesh {out_mesh}",
        "/mesh/check-mesh",
        "/exit yes",
    ], workdir)


def meshing_poly(workdir: Path, surface: Path, walls: list[str], all_zones: list[str], out_mesh: Path, min_size: float, max_size: float,
                 size_growth: float = 1.2, curvature_angle: float = 18, n_layers: int = 10, first_aspect_ratio: float = 20, growth: float = 1.2,
                 tet_volume_growth: float = 1.2) -> str:
    """Library 'P' family (ILO and the larger AAA): the wall is remeshed on a curvature size field (not the STL facets),
    10 aspect-ratio prism layers, tet fill; the solver then converts the whole domain to polyhedra (``finalize(...,
    to_poly=True)``). The library meshes are poly-hexcore (1-2 % octree hexes in the core); v231 auto-mesh has no
    poly-hexcore fill and its octree 'hexcore' fill marks all but one region dead (tried 2026-09-28), so the core is tet->poly.
    Sizes in metres."""
    zones = f"({' '.join(all_zones)})"
    return _finish([
        f"/file/read-boundary-mesh {surface}",
        f"/size-functions/set-global-controls {min_size:g} {max_size:g} {size_growth:g}",
        f"/size-functions/create curvature face {zones} sf-curv {min_size:g} {max_size:g} {size_growth:g} {curvature_angle:g}",
        "/size-functions/list",
        "/size-functions/compute",
        f"/boundary/remesh/remesh-face-zones-conformally {zones} () 40 20 yes",
        "/boundary/manage/delete (*-orig-*) yes",
        "/boundary/manage/list",
        f"/boundary/mark-face-intersection {zones} 56",
        f"/mesh/prism/controls/zone-specific-growth/apply-growth ({' '.join(walls)}) aspect-ratio geometric {n_layers} {first_aspect_ratio:g} {growth:g} no",
        f"/mesh/tet/controls/cell-sizing geometric {tet_volume_growth:g}",
        '/mesh/auto-mesh "" zone-specific pyramids tet yes yes',
        "/mesh/manage/list",
        f"/file/write-mesh {out_mesh}",
        "/exit yes",
    ], workdir)


def finalize(workdir: Path, mesh_in: Path, renames: dict[str, str], case_out: Path, to_poly: bool = False) -> str:
    """Solver: (optionally convert to polyhedra,) rename the region cell zones to the library names, check, write."""
    lines = [f"/file/read-case {mesh_in}"]
    if to_poly:
        lines += ["/mesh/polyhedra/convert-domain"]
    lines += [f"/mesh/modify-zones/zone-name {old} {new}" for old, new in renames.items()]
    lines += ["/mesh/modify-zones/list-zones", "/mesh/check", "/mesh/quality", f"/file/write-case {case_out}", "/exit yes"]
    return _finish(lines, workdir)


def setup(workdir: Path, reference_copy: Path, mesh_case: Path, case_out: Path) -> str:
    """Solver: reference settings (its libudf is auto-compiled from ``workdir/udf-inlet.c``) + replace-mesh."""
    return _finish([
        f"/file/read-case {reference_copy}",
        f"/file/replace-mesh {mesh_case}",
        "/mesh/modify-zones/list-zones",
        "/mesh/check",
        f"/file/write-case {case_out}",
        "/exit yes",
    ], workdir)


def smoke(workdir: Path, case: Path) -> str:
    """Solver: read the final case (auto-compiles libudf), report zones, BCs and mesh check; nothing is written."""
    return _finish([
        f"/file/read-case {case}",
        "/mesh/modify-zones/list-zones",
        "/mesh/check",
        "/define/boundary-conditions/list-zones",
        "/exit yes",
    ], workdir)


def run(workdir: Path, case: Path, time_step: float = 0.005, n_steps: int = 1280, max_iter: int = 20) -> str:
    """Identical to the library 2.jou except for the case path."""
    return _finish([
        f"/file/read-case {case}",
        "/solve/initialize/initialize-flow",
        f"/solve/set/time-step {time_step:g}",
        f"/solve/dual-time-iterate {n_steps} {max_iter}",
        "/exit y",
    ], workdir)

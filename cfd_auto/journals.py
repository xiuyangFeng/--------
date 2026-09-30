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
            first_aspect_ratio: float = 20, growth: float = 1.2, tet_volume_growth: float = 1.2, improve_skew: float | None = None) -> str:
    """Boundary mesh -> prisms (aspect-ratio offsets, geometric growth) on every wall + tet fill; one cell zone per region
    (``merge cell zones? yes`` puts each region's prisms into its tet zone and folds the prism side quads into the caps).
    ``improve_skew``: wall-face skewness improvement before the prisms (managed repair ladder; moves STL nodes)."""
    improve = [f"/boundary/improve/improve ({' '.join(walls)}) skewness {improve_skew:g} 180 5 no"] if improve_skew else []
    return _finish([
        f"/file/read-boundary-mesh {surface}",
        *improve,
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
                 tet_volume_growth: float = 1.2, improve_skew: float | None = None) -> str:
    """Library 'P' family (ILO and the larger AAA): the wall is remeshed on a curvature size field (not the STL facets),
    10 aspect-ratio prism layers, tet fill; the solver then converts the whole domain to polyhedra (``finalize(...,
    to_poly=True)``). The library meshes are poly-hexcore (1-2 % octree hexes in the core); v231 auto-mesh has no
    poly-hexcore fill and its octree 'hexcore' fill marks all but one region dead (tried 2026-09-28), so the core is tet->poly.
    Sizes in metres. ``improve_skew``: optional wall-face skewness improvement after the remesh (a few sliver faces on
    LI_FA_XIANG-1/before stopped the first prism layer, which Fluent then deleted -> a mesh with no boundary layer).
    Call template from Fluent's own scheme: ``/boundary improve improve <zones> skewness <limit> 180 5 no``."""
    zones = f"({' '.join(all_zones)})"
    improve = [f"/boundary/improve/improve ({' '.join(walls)}) skewness {improve_skew:g} 180 5 no"] if improve_skew else []
    return _finish([
        f"/file/read-boundary-mesh {surface}",
        f"/size-functions/set-global-controls {min_size:g} {max_size:g} {size_growth:g}",
        f"/size-functions/create curvature face {zones} sf-curv {min_size:g} {max_size:g} {size_growth:g} {curvature_angle:g}",
        "/size-functions/list",
        "/size-functions/compute",
        f"/boundary/remesh/remesh-face-zones-conformally {zones} () 40 20 yes",
        "/boundary/manage/delete (*-orig-*) yes",
        *improve,
        "/boundary/manage/list",
        f"/boundary/mark-face-intersection {zones} 56",
        f"/mesh/prism/controls/zone-specific-growth/apply-growth ({' '.join(walls)}) aspect-ratio geometric {n_layers} {first_aspect_ratio:g} {growth:g} no",
        f"/mesh/tet/controls/cell-sizing geometric {tet_volume_growth:g}",
        '/mesh/auto-mesh "" zone-specific pyramids tet yes yes',
        "/mesh/manage/list",
        f"/file/write-mesh {out_mesh}",
        "/exit yes",
    ], workdir)


def meshing_polyhexcore(workdir: Path, surface: Path, walls: list[str], all_zones: list[str], out_mesh: Path, min_size: float, max_size: float,
                        size_growth: float = 1.2, curvature_angle: float = 18, n_layers: int = 10, first_aspect_ratio: float = 20, growth: float = 1.2,
                        improve_skew: float | None = None, buffer_layers: int = 2, peel_layers: int = 2, object_name: str = "anatomy", **_) -> str:
    """The library P family as the operator built it: curvature remesh, then a mesh OBJECT over all zones (6 fluid
    volumetric regions), SCOPED aspect-ratio prisms on the walls, poly-hexcore fill (2026-09-30 probe on v231,
    ``_protocol_study/poly_probe``; poly / poly-hexcore fills exist only for object-based meshing with scoped prisms, which
    is why ``auto-mesh ""`` with zone-specific prisms never offered them). The size field is deleted after the remesh so
    the octree refines to the surface sizes (finest hex = ``min_size``; YU_TIAN_HAI library levels 0.5 / 1 / 2 mm ->
    min 0.5 mm). Cell zones come out as ``<object>`` and ``<object>:1..5`` (renamed by the finalize stage). Every
    prompt of /objects/create and /mesh/scoped-prisms/create is answered on its own line (v231 prompt order)."""
    zones = f"({' '.join(all_zones)})"
    improve = [f"/boundary/improve/improve ({' '.join(walls)}) skewness {improve_skew:g} 180 5 no"] if improve_skew else []
    return _finish([
        f"/file/read-boundary-mesh {surface}",
        f"/size-functions/set-global-controls {min_size:g} {max_size:g} {size_growth:g}",
        f"/size-functions/create curvature face {zones} sf-curv {min_size:g} {max_size:g} {size_growth:g} {curvature_angle:g}",
        "/size-functions/compute",
        f"/boundary/remesh/remesh-face-zones-conformally {zones} () 40 20 yes",
        "/boundary/manage/delete (*-orig-*) yes",
        "/size-functions/delete",
        *improve,
        f"/boundary/mark-face-intersection {zones} 56",
        "/boundary/manage/list",
        "/objects/create", object_name, "fluid", "3", zones, "()", "mesh", "yes",
        "/objects/volumetric-regions/compute", object_name, "no",
        f"/objects/volumetric-regions/list {object_name} (*)",
        "/mesh/scoped-prisms/create", "bl", "aspect-ratio", f"{first_aspect_ratio:g}", f"{n_layers:d}", f"{growth:g}", object_name, "fluid-regions", "only-walls",
        "/mesh/scoped-prisms/list",
        f"/mesh/hexcore/controls/buffer-layers {buffer_layers:d}",
        f"/mesh/hexcore/controls/peel-layers {peel_layers:d}",
        "/mesh/poly-hexcore/controls/mark-core-region-cell-type-as-hex? yes",
        "/mesh/auto-mesh", object_name, "no", "scoped", "pyramids", "poly-hexcore", "yes",
        "/mesh/manage/list",
        "/mesh/check-mesh",
        f"/file/write-mesh {out_mesh}",
        "/exit yes",
    ], workdir)


def surface_remesh(workdir: Path, surface: Path, all_zones: list[str], out_mesh: Path, min_size: float, max_size: float, size_growth: float = 1.2,
                   curvature_angle: float = 18) -> str:
    """The surface half of :func:`meshing_poly` only (size-field calibration of new cases; sizes in metres)."""
    zones = f"({' '.join(all_zones)})"
    return _finish([
        f"/file/read-boundary-mesh {surface}",
        f"/size-functions/set-global-controls {min_size:g} {max_size:g} {size_growth:g}",
        f"/size-functions/create curvature face {zones} sf-curv {min_size:g} {max_size:g} {size_growth:g} {curvature_angle:g}",
        "/size-functions/compute",
        f"/boundary/remesh/remesh-face-zones-conformally {zones} () 40 20 yes",
        "/boundary/manage/delete (*-orig-*) yes",
        f"/file/write-mesh {out_mesh}",
        "/exit yes",
    ], workdir)


def rename_zones(workdir: Path, case_in: Path, renames: dict[str, str], case_out: Path, zone_types: dict[str, str] | None = None) -> str:
    """Solver: read a case, rename zones in two phases (via unique temporaries, so swaps like blood2<->blood4 cannot
    collide), optionally change boundary types (keys = final names, e.g. outflow -> pressure-outlet), check, write."""
    tmp = {old: f"cfdauto-tmp-{i}" for i, old in enumerate(renames)}
    lines = [f"/file/read-case {case_in}"]
    lines += [f"/mesh/modify-zones/zone-name {old} {tmp[old]}" for old in renames]
    lines += [f"/mesh/modify-zones/zone-name {tmp[old]} {new}" for old, new in renames.items()]
    lines += [f"/define/boundary-conditions/zone-type {z} {typ}" for z, typ in (zone_types or {}).items()]
    lines += ["/mesh/modify-zones/list-zones", "/mesh/check", f"/file/write-case {case_out}", "/exit yes"]
    return _finish(lines, workdir)


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


def smoke(workdir: Path, case: Path, quality: bool = False) -> str:
    """Solver: read the final case (auto-compiles libudf), report zones, BCs and mesh check (``quality``: also the
    worst-cell report for the managed mesh gate); nothing is written."""
    return _finish([
        f"/file/read-case {case}",
        "/mesh/modify-zones/list-zones",
        "/mesh/check",
        *(["/mesh/quality"] if quality else []),
        "/define/boundary-conditions/list-zones",
        "/exit yes",
    ], workdir)


def run_gentle_start(workdir: Path, case: Path, time_step: float = 0.005, n_steps: int = 1280, max_iter: int = 20, period_s: float = 0.8, refine: int = 2) -> str:
    """The library run with the FIRST cardiac cycle at ``time_step / refine`` (cold start from rest with uncharged
    windkessels diverged for WANG_CAI-0/before, 2026-09-29). The fine cycle covers exactly one period, so from its end
    the step index N has the library phase (t = N*dt - period); the step count stays ``n_steps`` so the exported frames
    1120..1280 and the peak step 1162 keep their library phases. Cost: one cycle fewer of windkessel settling (7 vs 8)."""
    fine = int(round(period_s / (time_step / refine)))
    coarse = n_steps - fine
    if coarse <= 0 or abs(fine * time_step / refine - period_s) > 1e-9:
        raise ValueError("fine stage must be one whole period and shorter than the run")
    return _finish([
        f"/file/read-case {case}",
        "/solve/initialize/initialize-flow",
        f"/solve/set/time-step {time_step / refine:g}",
        f"/solve/dual-time-iterate {fine} {max_iter}",
        f"/solve/set/time-step {time_step:g}",
        f"/solve/dual-time-iterate {coarse} {max_iter}",
        "/exit y",
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

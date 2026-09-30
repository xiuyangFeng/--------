"""cfd_auto — STL -> Fluent Meshing -> Fluent solver -> exports, reproducing the library's CFD protocol (2026-09-27).

The library cases (data_new/<cohort>/...) were built by hand: SpaceClaim flow extensions + Fluent Meshing tet/prism mesh +
a per-case compiled UDF (inlet waveform / area, Carreau viscosity, four RCR outlets). This package rebuilds a case from its
wall STL with the same protocol so that the labels stay comparable:

    surface   STL (mm, open at the inlet/outlets) + planar caps -> Fluent boundary mesh (m)
    mesh      Fluent Meshing: 10 aspect-ratio prism layers on the wall + tet fill (anatomy only)
    extend    Fluent solver: extrude every cap along its normal (straight, opening-shaped extension) and rename the zones
              to the library convention (blood/blood1..5, wall/wall1..5, in/out-*, in+/out-*+)
    setup     copy of a reference case (settings only) + /file/replace-mesh + regenerated UDF (thread ids, constants)
    run       same 2.jou / fluent.slurm as the library (0.005 s x 1280 steps, 20 iterations, ASCII exports)

Managed batches (2026-09-30): ``python -m cfd_auto.orchestrate run <batch.json>`` drives every unit from STL (or a library
mesh) to checked exports with on-disk state, protocol files (``protocols/*.json``), exports limited to the kept frames,
requeue-safe Slurm scripts, a divergence ladder and set-up / mesh / after-run gates (see README).

Hard rule: the source library is read-only. Every Fluent journal and every path string written into a case must point
inside the work directory (``guard.assert_inside``); reference cases are copied before Fluent ever reads them.
"""
__version__ = "0.2.0"

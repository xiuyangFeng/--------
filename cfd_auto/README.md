# cfd_auto — STL → Fluent Meshing → Fluent → exports

Rebuilds a patient CFD case from its wall STL with the library's protocol (Fluent 2023R1, Carreau UDF, four RCR outlets,
0.005 s × 1280 steps, ASCII wall/volume exports), so the labels stay comparable with `data_new/`.
Design notes, the reverse-engineered library protocol, pitfalls and all comparison results:
`docs/02-推进与变更/04-数据处理与CFD/STL全自动CFD工程cfd_auto_试算_2026-09-27.md`.

## Usage

```bash
# library case -> rebuilt case ready to run (optionally submitted); every stage has a gate and writes a JSON report
python -m cfd_auto.pipeline AG/slow/QIN_SI_FU --work-root outputs/cfd_auto_trial_20260927 [--submit]
python -m cfd_auto.pipeline "ILO/GUO_YU_SHU-0/before" --work-root <dir> \
    --mesh-params '{"min_size": 0.00075, "max_size": 0.0016, "curvature_angle": 26}'   # poly family: per-case sizes

# case whose own .cas is lost: settings of a same-template library case + the case's own UDF constants
python -m cfd_auto.rebuild AAA/ruputer/YANG_BAO_KUI --template AAA/ruputer/YU_TIAN_HAI \
    --original-log Global_conditions/Fluent_14165.out --work-root <dir> --mesh-params '{...}'

# library units outside v5.2 recovered with a fresh protocol CFD (plan file: template, mode own-mesh | stl, naming, density ref)
python -m cfd_auto.recover ILO/LI_JIE-1/before --plan <dir>/plan.json --work-root <dir>/units --cores 92 --submit

# numerical checks of a finished run without an original solution (flow ratio, L/R and Murray split, outlet pressure,
# periodicity, solver, exports) -> <work>/sanity.json
python -m cfd_auto.sanity <work dir>

# pre-registered comparison against the library (criteria v2)
python -m cfd_auto.compare data_new/<case> <work-root>/<case> --criteria v2 [--ref-log ...] [--no-ref-case] [--no-volume]

python -m pytest -q cfd_auto/tests
```

## Stages and gates (`pipeline.py`)

| stage | what | gate |
|---|---|---|
| profile | read-only profile of the library case (openings, extension lengths, zone names, UDF constants, mesh family) | — |
| surface | STL self-intersection repair (local, rims fixed, ≤ 0.5 mm) + planar caps + straight extensions → Fluent boundary mesh | every region closed, no collisions |
| mesh | Fluent Meshing: tet-prism family = STL facets as wall; poly family = curvature size-field remesh; 10 prism layers (AR 20, ×1.2) + tet fill | anatomy mesh vs library (dual-equivalent resolution, BL) |
| finalize | solver: region cell zones renamed blood / bloodN | mesh check |
| setup | copy of the library case (paths rewritten into the work dir) + replace-mesh; UDF with final thread ids and **inlet-area divisor = new inlet area** | settings_diff: 0 unexpected differences |
| smoke | read the final case (libudf compiled from the regenerated UDF) | 7 UDF hooks loaded |
| run files | library 2.jou / fluent.slurm (64 cores), empty `ascii/` | — |

## Modules

`surface.py` (STL, caps, extensions, repair, Fluent boundary-mesh writer) · `journals.py` (v231 TUI journals) ·
`refcase.py` (library profile, path-rewritten copy) · `udf.py` (UDF render / byte-preserving IO / RCR protocol) ·
`settings_diff.py` (case-settings comparison) · `meshcheck.py` (mesh statistics, gate, surface-mesh reader) ·
`slurm.py` (Fluent batch jobs) · `guard.py` (library read-only guard) · `compare.py` (label comparison) ·
`rebuild.py` (lost-case rebuild) · `recover.py` (protocol recovery of units: own mesh renamed to a template, or STL
rebuild with calibrated surface density; opening naming by partner centres / rigid registration / deployment proposal) ·
`sanity.py` (after-run numerical checks, gates calibrated on YU/GUO/YANG).

## Requirements and rules

- Runs inside the GNN repository: imports `wss_pinn.v4.fluent_topology` (Fluent case reader); conda env `GNN`;
  Fluent 2023R1 (`module load fluent/231`) through Slurm (`/public/slurm/bin`, CPU partition).
- The library (`data/`, `data_new/`) is read-only: Fluent only ever reads copies in the work directory (reading a library
  `.cas` makes Fluent auto-compile libudf inside the library directory); `guard.check_journal` refuses library paths.
- Known limits: v231 classic TUI has no poly / poly-hexcore fill, so poly-family cases are rebuilt as tet/prism at matched
  dual resolution; poly-family surface sizes must be calibrated per case (`recover.calibrate` does it against a target
  dual wall-face density). `udf.protocol_rcr` on the mesh cut-face areas reproduces AAA/ILO library UDFs to 0.01–0.03 %
  (AG uses a per-case total inflow). Cold start at 0.005 s diverged once on a 5.9 mm² outlet (WANG_CAI-0/before):
  `journals.run_gentle_start` runs the first cycle at half step and keeps the library phase of steps 1120–1280.
- After editing code or a plan, wait ≥ 60 s before submitting: compute nodes can read the previous version (NFS caching,
  node06 clock ~3 min behind).

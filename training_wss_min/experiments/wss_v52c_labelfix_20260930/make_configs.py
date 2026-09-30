"""Mainline X5Dcap_asym2 configs on the corrected data (2026-09-30 label fix) + matrix.json for run_local_train_queue.

    python make_configs.py            # writes training_wss_min/configs/wss_v52c_labelfix_20260930/

Sources (recipe verbatim; only data roots are substituted by tools/repoint_data_root.py, plus name / notes):
  IND  (v5.2c, train170 / test91)    wss_v52_phys_20260926/X5Dcap_asym2_s{1234,7,2025} + wss_v52_phys2e_20260928/X5Dcap_asym2_s{11,2026}
  CV5  (v5.2c, 5 patient folds)      wss_v52_phys2_20260927/X5Dcap_asym2_v52cv_f{0..4}_s{1234,7,2025}
  FULL (v5.2c, train265 / recover8)  wss_v52p4_full265_20260930/X5Dcap_asym2_full265_s{1234,7,2025}
Nothing is trained here.
"""
import json
import shutil
from pathlib import Path

from training_wss_min.tools.repoint_data_root import repoint

G = Path("/public/newhome/cy/Digital_twin/GNN")
EXP = "wss_v52c_labelfix_20260930"
OUT = G / "training_wss_min/configs" / EXP
C = G / "training_wss_min/configs"
SOURCES = ([("IND", None, s, C / "wss_v52_phys_20260926" / f"X5Dcap_asym2_s{s}.json") for s in (1234, 7, 2025)]
           + [("IND", None, s, C / "wss_v52_phys2e_20260928" / f"X5Dcap_asym2_s{s}.json") for s in (11, 2026)]
           + [("CV5", f, s, C / "wss_v52_phys2_20260927" / f"X5Dcap_asym2_v52cv_f{f}_s{s}.json") for f in range(5) for s in (1234, 7, 2025)]
           + [("FULL265", None, s, C / "wss_v52p4_full265_20260930" / f"X5Dcap_asym2_full265_s{s}.json") for s in (1234, 7, 2025)])


def flat(o, p=""):
    if isinstance(o, dict):
        for k, v in o.items():
            yield from flat(v, f"{p}.{k}" if p else k)
    elif isinstance(o, list) and any(isinstance(v, (dict, list)) for v in o):
        for i, v in enumerate(o):
            yield from flat(v, f"{p}[{i}]")
    else:
        yield p, o


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    arms, diffs = [], {}
    for protocol, fold, seed, src in SOURCES:
        cfg = json.loads(src.read_text())
        hits: list = []
        new = repoint(cfg, hits)
        assert hits, src
        fs = Path(new["data"]["feature_stats_path"])        # geometry-only z-scores, verified identical on v5.2c (verify_unified.json)
        dst = G / "data_wss_v5/views_v5_2c_20260930/feature_stats" / fs.name
        if not dst.exists():
            dst.parent.mkdir(parents=True, exist_ok=True); shutil.copy2(fs, dst)
        assert dst.read_bytes() == fs.read_bytes(), dst
        new["data"]["feature_stats_path"] = str(dst)
        new["name"] = f"{EXP}/{src.stem}"
        new["notes"] = (f"2026-09-30 label fix: {src.relative_to(C)} verbatim on the corrected data "
                        f"(unified v5.2c{', full265 split' if protocol == 'FULL265' else ''}; 9 library units re-simulated with protocol-correct BCs); "
                        "only data roots / name / notes changed (training_wss_min/tools/repoint_data_root.py).")
        a, b = dict(flat(cfg)), dict(flat(new))
        d = {k: [a.get(k), b.get(k)] for k in sorted(set(a) | set(b)) if a.get(k) != b.get(k)}
        assert all(k in ("name", "notes") or k.startswith("data.") for k in d), (src, [k for k in d if not k.startswith("data.")])
        diffs[src.stem] = d
        (OUT / src.name).write_text(json.dumps(new, indent=2, ensure_ascii=False) + "\n")
        arms.append({"id": src.stem, "title": f"X5Dcap_asym2 {protocol}{'' if fold is None else f' fold {fold}'} seed {seed} on the corrected data",
                     "config": src.name, "run_name": new["name"], "phase": 0, "seed": seed, "fold": fold, "protocol": protocol,
                     "arm": "X5Dcap_asym2", "parent": str(src.relative_to(C)), "depends_on": [], "evaluate": ["best", "last"],
                     "single_change": False, "config_diff_vs_parent": "data roots only (config_diff_vs_source.json)"})
    matrix = {"schema_version": 1, "experiment": EXP,
              "section": "mainline X5Dcap_asym2 retrained on the unified corrected data v5.2c (2026-09-30)",
              "data": {"v5.2c": str(G / "data_wss_v5/views_v5_2c_20260930"), "snapshot": str(G / "data_wss_v5/anatomy_pointcloud_v5_2c_20260930"),
                       "label_fix_doc": "docs/02-推进与变更/04-数据处理与CFD/库内标签问题核查_RCR挂错与入口除数_2026-09-30.md"},
              "expected_training_runs": len(arms), "expected_evaluations": 2 * len(arms),
              "reading_rule": ("Re-establishes the reference numbers on the corrected labels; compare each protocol with its own "
                               "predecessor arms (IND test91 five-seed ensemble, CV5 pooled out-of-fold, full265 on recover8). "
                               "Differences combine the label change of 9 units (3 of them IND test units) and seed noise."),
              "arms": arms}
    (OUT / "matrix.json").write_text(json.dumps(matrix, indent=2, ensure_ascii=False) + "\n")
    here = Path(__file__).resolve().parent
    (here / "config_diff_vs_source.json").write_text(json.dumps(diffs, indent=1, ensure_ascii=False))
    print(len(arms), "configs ->", OUT)
    print(json.dumps(diffs[SOURCES[0][3].stem], indent=1, ensure_ascii=False)[:2500])


if __name__ == "__main__":
    main()

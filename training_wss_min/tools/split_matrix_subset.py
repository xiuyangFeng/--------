"""Split a prepared matrix (config dir + matrix.json) into host-specific sub-experiments (2026-09-27, node04 + master).

    python -m training_wss_min.tools.split_matrix_subset --source wss_v52_syn_20260926 \
        --subset wss_v52_synm_20260927 "protocol==CV5" "master Slurm 4x4090" \
        --subset wss_v52_synn_20260927 "protocol==IND" "node04 2xA100 manual"

Each arm config is copied verbatim except ``name`` (run directory) and a note; matrix.json keeps every field and lists
only the chosen arms.  A selector is ``field==value`` or ``id in a,b,c``.  Arms must be covered exactly once.
"""
from __future__ import annotations
import argparse, json
from pathlib import Path

ROOT = Path("/public/newhome/cy/Digital_twin/GNN"); CONFIGS = ROOT / "training_wss_min/configs"


def select(arm: dict, rule: str) -> bool:
    if "==" in rule:
        k, v = rule.split("==", 1); return str(arm.get(k.strip())) == v.strip()
    if " in " in rule:
        k, v = rule.split(" in ", 1); return str(arm.get(k.strip())) in {x.strip() for x in v.split(",")}
    raise ValueError(rule)


def main() -> None:
    ap = argparse.ArgumentParser(); ap.add_argument("--source", required=True); ap.add_argument("--subset", nargs=3, action="append", metavar=("NAME", "RULE", "NOTE"), required=True)
    a = ap.parse_args(); src = CONFIGS / a.source; m = json.loads((src / "matrix.json").read_text()); seen: dict = {}
    for name, rule, note in a.subset:
        dst = CONFIGS / name; dst.mkdir(parents=True, exist_ok=True); arms = []
        for arm in m["arms"]:
            if not select(arm, rule):
                continue
            if arm["id"] in seen:
                raise ValueError(f"{arm['id']} selected twice ({seen[arm['id']]} and {name})")
            seen[arm["id"]] = name
            cfg = json.loads((src / arm["config"]).read_text()); cfg["name"] = f"{name}/{arm['id']}"; cfg["notes"] = f"{cfg.get('notes', '')} [{note}]"
            (dst / arm["config"]).write_text(json.dumps(cfg, ensure_ascii=False, indent=2) + "\n"); arms.append(arm)
        mm = dict(m); mm.update(experiment=name, arms=arms, expected_training_runs=len(arms), expected_evaluations=2 * len(arms),
                                split_note=f"subset of {a.source}: rule '{rule}'; {note}", source_matrix=str(src / "matrix.json"))
        (dst / "matrix.json").write_text(json.dumps(mm, ensure_ascii=False, indent=2) + "\n"); print(name, len(arms), "arms:", [x["id"] for x in arms])
    missing = [x["id"] for x in m["arms"] if x["id"] not in seen]
    if missing:
        raise ValueError(f"arms not covered by any subset: {missing}")


if __name__ == "__main__":
    main()

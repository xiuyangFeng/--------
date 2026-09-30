"""Prepare the OSI target-structure matrix on the deployed M1 recipe (v5.1 cv3, seed 1234; "task 2", 2026-09-25).

Origin: tracking §34.12 — loss / hand-crafted-feature changes cannot close the OSI discrimination gap; §34.9 item 4 (mean tangential
vector, OSI derived) and item 2 (rev_frac auxiliary head) were left untested.  Every arm = M1_f{k}_s1234 with one change:

  V1_f{k}  data.multi_aux_channels = (mean_axial, mean_circ), out_dim 5: the frame-0..79 mean wall-shear vector in the local
           (axial, circumferential) tangent frame divided by the mean magnitude; evaluation adds heads.osi_derived =
           0.5(1 - min(1, |(r_a, r_c)|)) next to the direct OSI head
  V2_f{k}  data.multi_aux_channels = (rev_frac,), out_dim 4: fraction of frames reversed against the peak-frame direction

Auxiliary channels are linear z with train-fold statistics, enter only the equal-weight channel MSE, never a deployment output.
Labels come from the view bundle (wall_wss_vec) + cycle.npz; the tangent frame is the deployable direction-head frame (PCA normal,
atlas tangent).  Label cache: experiments/<exp>/aux_labels/<case>.npz (per case, all three aux channels, for statistics only).

    python -u -m training_wss_min.tools.prepare_wss_osi_struct [--workers 16]
"""
from __future__ import annotations

import argparse
import hashlib
import json
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

from training_wss_min import config as C
from training_wss_min import dataset as D
from training_wss_min.tools import prepare_wss_local_wave1 as W1

ROOT = C.PROJECT_ROOT
NAME = "wss_osi_struct_20260925"
M1_NAME = "wss_cycle_m1_20260921"
CONFIGS = ROOT / "training_wss_min/configs" / NAME
EXP = ROOT / "training_wss_min/experiments" / NAME
M1_CONFIGS = ROOT / "training_wss_min/configs" / M1_NAME
VIEW = ROOT / "data_wss_v5/views_v5_1/wss_min_view_v1"
CYC = ROOT / "data_wss_v5/views_v5_1/wss_min_cycle_v1"
SEED = 1234
AUX_ALL = ("rev_frac", "mean_axial", "mean_circ")
ARMS = {"V1": ("mean_axial", "mean_circ"), "V2": ("rev_frac",)}
TITLES = {"V1": "auxiliary mean tangential vector (axial, circumferential) / mean magnitude; OSI also derived from it",
          "V2": "auxiliary rev_frac channel (fraction of frames reversed against the peak direction)"}


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _cache_one(cid: str) -> dict:
    out = EXP / "aux_labels" / f"{cid.replace('/', '__')}.npz"
    if not out.is_file():
        cohort_rel, case_name = cid.rsplit("/", 1)
        cols, _ = D.multi_aux_columns(VIEW / cid / "bundle.npz", AUX_ALL, CYC, cohort_rel, case_name)
        out.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(out, **cols)
    with np.load(out) as z, np.load(CYC / cid / "cycle.npz") as c:
        osi = c["wall_osi"].astype(np.float64)
        der = D.derived_osi(z["mean_axial"], z["mean_circ"])
        return {"case": cid, "osi_consistency_r2": float(1 - np.mean((der - osi) ** 2) / max(np.var(osi), 1e-12)),
                "osi_consistency_mae": float(np.mean(np.abs(der - osi)))}


def aux_stats(fold: int, split_path: str) -> dict:
    cases = [f"{c}/{n}" for c, n in D.load_split_cases(split_path, "train")]
    cols = {a: [] for a in AUX_ALL}
    for cid in cases:
        with np.load(EXP / "aux_labels" / f"{cid.replace('/', '__')}.npz") as z:
            for a in AUX_ALL:
                cols[a].append(z[a].astype(np.float64))
    return {a: {"method": "linear", "linear": {"mean": float(np.concatenate(v).mean()), "std": float(np.concatenate(v).std())},
                "scope": f"fold{fold} train partition ({len(cases)} cases)"} for a, v in cols.items()}


def multi_stats(arm: str, fold: int, base: dict, stats_aux: dict) -> Path:
    path = EXP / "stats" / f"multi_stats_{arm}_fold{fold}.json"
    st = json.loads(Path(base["data"]["wss_stats_path"]).read_text())
    if st.get("method") != "multi" or tuple(st["channels"]) != tuple(C.MULTI_CHANNELS):
        raise ValueError("base M1 statistics must be the three-channel multi file")
    out = dict(st)
    out["channels"] = list(C.MULTI_CHANNELS) + list(ARMS[arm])
    for a in ARMS[arm]:
        out[a] = stats_aux[a]
    out["aux_statistics_scope"] = f"auxiliary channels: fold{fold} train partition only (label cache {EXP / 'aux_labels'})"
    W1.save(path, out)
    return path


def configure(arm: str, fold: int, stats_aux: dict) -> tuple[str, dict, dict]:
    base = json.loads((M1_CONFIGS / f"M1_f{fold}_s{SEED}.json").read_text())
    cfg = json.loads(json.dumps(base))
    aid = f"{arm}_f{fold}_s{SEED}"
    cfg["name"] = f"{NAME}/{aid}"
    cfg["data"]["multi_aux_channels"] = list(ARMS[arm])
    cfg["model"]["out_dim"] = len(C.MULTI_CHANNELS) + len(ARMS[arm])
    cfg["data"]["wss_stats_path"] = str(multi_stats(arm, fold, base, stats_aux))
    cfg["notes"] = (f"OSI target-structure matrix {aid}: M1_f{fold}_s{SEED} (deployed three-head recipe) with one change — {TITLES[arm]}; "
                    f"paired initialisation identical to M1 (reference = fold X5D_v51 config; output layers rebuilt with "
                    f"{cfg['model']['out_dim']} channels); held-out fold read once; test34 unused.")
    classes = {"data": C.DataConfig, "model": C.ModelConfig, "train": C.TrainConfig, "eval": C.EvalConfig}
    for section, cls in classes.items():
        unknown = set(cfg[section]) - set(cls.__dataclass_fields__)
        if unknown:
            raise ValueError(f"unknown {section} fields: {unknown}")
    C.validate_features(C.ExpConfig.from_dict(cfg))
    return aid, cfg, base


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=16)
    args = ap.parse_args()
    split0 = json.loads((M1_CONFIGS / f"M1_f0_s{SEED}.json").read_text())["data"]["split_path"]
    all_cases = sorted({f"{c}/{n}" for k in range(3) for part in ("train", "test")
                        for c, n in D.load_split_cases(split0.replace("fold0", f"fold{k}"), part)})
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        consistency = list(ex.map(_cache_one, all_cases, chunksize=2))
    r2 = np.array([c["osi_consistency_r2"] for c in consistency]); mae = np.array([c["osi_consistency_mae"] for c in consistency])
    report = {"n_cases": len(consistency), "osi_consistency_r2_median": float(np.median(r2)), "osi_consistency_r2_min": float(r2.min()),
              "osi_consistency_mae_median": float(np.median(mae)), "osi_consistency_mae_max": float(mae.max()),
              "worst5": sorted(consistency, key=lambda c: c["osi_consistency_r2"])[:5]}
    W1.save(EXP / "aux_label_consistency.json", report)
    print("label consistency (derived OSI from true mean-vector channels vs true OSI):",
          {k: v for k, v in report.items() if k != "worst5"}, flush=True)
    arms = []
    for arm in ARMS:
        for fold in range(3):
            base_split = json.loads((M1_CONFIGS / f"M1_f{fold}_s{SEED}.json").read_text())["data"]["split_path"]
            aid, cfg, base = configure(arm, fold, aux_stats(fold, base_split))
            W1.save(CONFIGS / f"{aid}.json", cfg)
            arms.append(dict(id=aid, arm=arm, fold=fold, seed=SEED, config=f"{aid}.json", run_name=cfg["name"], title=TITLES[arm],
                             parent=f"M1_f{fold}_s{SEED} ({M1_NAME})", single_change=True, config_diff_vs_parent=W1.declared_diff(base, cfg)))
    W1.save(CONFIGS / "matrix.json", dict(
        schema_version=1, experiment=NAME, parent_experiment=M1_NAME,
        parent_configs_sha256={f"M1_f{k}_s{SEED}": sha256(M1_CONFIGS / f"M1_f{k}_s{SEED}.json") for k in range(3)},
        execution="node04 (2x A100-40GB, outside Slurm) via tools/run_local_train_queue.py; train -> eval best -> eval last (--save-predictions)",
        expected_training_runs=len(arms), expected_evaluations=2 * len(arms), arms=arms,
        reading_rule="fold arms read only on their held-out fold; controls = M1r (node04 A100, wss_osi_tail_20260924) and stored M1; "
                     "V1 primary = derived OSI head, secondary = direct OSI head; V2 = direct OSI head; see PREREG.md"))
    print(CONFIGS / "matrix.json", len(arms), "arms")


if __name__ == "__main__":
    main()

"""Prepare the v5.2d retrain execution matrix (2026-10-01; the matrix the user confirmed on 2026-09-30, resubmitted on v5.2d).

v5.2d = v5.2c after the library audit merge of 2026-10-01 (HAN_JIAN_FU wall labels re-simulated, ILO/WANG_TIAN_QING-1/after
rebuilt from its own STL, 16 units with corrected outlet names; docs/02-推进与变更/04-数据处理与CFD/母库全量审计_2026-10-01.md §11).
The v5.2c retrain (wss_v52c_retrain_20260930) was cancelled before any arm finished; this is the same matrix on v5.2d:

  B  X5Dcap_asym2_full265_s{1234,7,2025}       the v5.2c retrain configs with data roots repointed to v5.2d
  C  M1cap_full265_s{1234,7,2025}              three-head recipe M1cap_v52cv (2026-09-25) unchanged, on v5.2d full265
  A  X5Dcap_asym2_v52cv_f{0..4}_s{1234,7,2025} the v5.2c retrain configs repointed (CV5, 261 units, held-out fold)

Recipes are untouched: A / B differ from their v5.2c sources only in data-root strings, name and notes (checked); C is rebuilt
by the v5.2c builder (prepare_wss_v52c_retrain.m1_config, same declared-difference checks) on the v5.2d paths. All 21 runs go
to runs/wss_v52d_retrain_20261001/ (nothing is written into the cancelled v5.2c run directories).

Also written (idempotent; an existing different file is never changed):
  experiments/<exp>/stats/multi_stats_full265_train265.json   three-channel statistics from the v5.2d statistics files
  experiments/<exp>/anchor/X5Dcap_asym2_full265_s1234_v52d/   GPU-preflight anchor: the pre-label-fix full265 run of job 16192
      repointed to v5.2d. recover8 is not among the 18 units changed by the merge and the model's inputs (geom, murray_cap
      flowref) are bitwise unchanged for unchanged units, so its stored recover8 metrics must reproduce.
  experiments/<exp>/e5_deployed_m1/M1_s{seed}/                 E5 reference: deployed three-head release, config -> v5.2d

    python -u -m training_wss_min.tools.prepare_wss_v52d_retrain
"""
from __future__ import annotations

import json
from pathlib import Path

from training_wss_min import config as C
from training_wss_min.tools import prepare_wss_local_wave1 as W1
from training_wss_min.tools import prepare_wss_v52c_retrain as P
from training_wss_min.tools.repoint_data_root import repoint

ROOT = C.PROJECT_ROOT
NAME = "wss_v52d_retrain_20261001"
SRC_NAME = "wss_v52c_retrain_20260930"
CONFIGS = ROOT / "training_wss_min/configs" / NAME
EXP = ROOT / "training_wss_min/experiments" / NAME
SRC = ROOT / "training_wss_min/configs" / SRC_NAME
V = ROOT / "data_wss_v5/views_v5_2d_20261001"
VIEW = V / "wss_min_view_v1"
CYC = V / "wss_min_cycle_v1"
FULL_SPLIT = VIEW / "split_v52p4_full265_train265_test8.json"
SEEDS, FOLDS = P.SEEDS, P.FOLDS
FROZEN = "GNN_v52d_frozen_20261001"
STALE = ("v5_2c", "v5_2p4", "v5_2p5", "views_v5_2_full", "v52c_retrain", "v52c_labelfix")
# the v5.2c builder's helpers (validate / m1_config) read these module globals at call time
P.NAME, P.CONFIGS, P.EXP, P.V, P.VIEW, P.CYC, P.FULL_SPLIT = NAME, CONFIGS, EXP, V, VIEW, CYC, FULL_SPLIT


def strings(obj):
    if isinstance(obj, dict):
        for v in obj.values():
            yield from strings(v)
    elif isinstance(obj, list):
        for v in obj:
            yield from strings(v)
    elif isinstance(obj, str):
        yield obj


def ab_config(aid: str) -> tuple[dict, dict]:
    src = json.loads((SRC / f"{aid}.json").read_text())
    hits: list = []
    cfg = repoint(src, hits)
    if not hits:
        raise RuntimeError(f"{aid}: source config references no replaced root")
    cfg["name"] = f"{NAME}/{aid}"
    cfg["notes"] = (str(src.get("notes", "")) + f" | 2026-10-01 v5.2d retrain: {SRC_NAME}/{aid}.json with data roots repointed to "
                    "v5.2d (library audit merge); recipe unchanged").strip(" |")
    diff = W1.declared_diff(src, cfg)
    for key, change in diff.items():
        if key in ("name", "notes"):
            continue
        if not key.startswith("data."):
            raise ValueError(f"{aid}: non-data field changed by repointing: {key}")
        old, new = (json.dumps(change[k]) for k in ("anchor", "arm"))
        if old.replace("views_v5_2c_20260930", "views_v5_2d_20261001") != new:
            raise ValueError(f"{aid}: {key} changed by more than the data root")
    data_paths = [s for s in strings(cfg["data"])]
    if any(tag in s for s in data_paths for tag in STALE):
        raise ValueError(f"{aid}: a data field still references an older data version")
    P.validate(cfg, aid)
    return cfg, diff


def multi_stats() -> Path:
    path = EXP / "stats" / "multi_stats_full265_train265.json"
    wss_path = VIEW / "wss_global_stats_full265_train265.json"
    tawss_path = CYC / "stats" / "cycle_stats_tawss_full265_train265.json"
    osi_path = CYC / "stats" / "cycle_stats_osi_logit_full265_train265.json"
    wss, tawss, osi = (json.loads(p.read_text()) for p in (wss_path, tawss_path, osi_path))
    for name, st, expect in (("wss", wss, "log_z"), ("tawss", tawss, "log_z"), ("osi", osi, "logit_z")):
        if st.get("method") != expect:
            raise ValueError(f"{name} statistics must be {expect}")
    split_sha = P.sha256(FULL_SPLIT)
    if tawss["split_sha256"] != split_sha or osi["split_sha256"] != split_sha:
        raise ValueError("cycle statistics were not built on the v5.2d full265 split file")
    if Path(wss["split_path"]).resolve() != FULL_SPLIT.resolve() or int(wss["n_cases"]) != 265 or int(tawss["n_cases"]) != 265:
        raise ValueError("peak-frame statistics are not the v5.2d full265 train265 file")
    out = {"schema_version": 1, "method": "multi", "channels": list(C.MULTI_CHANNELS), "eps": wss["eps"], "floor": tawss["floor"],
           "scope": "full265_train265",
           "statistics_scope": "v5.2d full265 train partition only (265 units; peak-frame log_z = the X5Dcap full265 file; "
                               "cycle = views_v5_2d wss_min_cycle_v1)",
           "sources": {"wss": str(wss_path), "tawss": str(tawss_path), "osi": str(osi_path)},
           "wss": wss, "tawss": tawss, "osi": osi}
    W1.save(path, out)
    return path


def anchor_dir() -> Path:
    out = EXP / "anchor" / "X5Dcap_asym2_full265_s1234_v52d"
    cfg = json.loads((P.OLD_FULL_RUN / "config.json").read_text())
    hits: list = []
    new = repoint(cfg, hits)
    if not hits:
        raise RuntimeError("anchor config references no replaced root")
    new["notes"] = (str(cfg.get("notes", "")) + " | 2026-10-01 preflight anchor copy: data roots repointed to v5.2d; "
                    "checkpoint / statistics / stored metrics copied from job 16192 unchanged").strip(" |")
    W1.save(out / "config.json", new)
    for name in ("ckpt_best.pt", "feature_stats.json", "wss_global_stats.json", "target_normalization.json"):
        P.copy_file(P.OLD_FULL_RUN / name, out / name)
    P.copy_file(P.OLD_FULL_RUN / "eval/ckpt_best/metrics.json", out / "eval/ckpt_best/metrics.json")
    W1.save(out / "ANCHOR_SOURCE.json", {"source_run": str(P.OLD_FULL_RUN), "repointed_fields": len(hits),
                                          "reason": "recover8 labels and this model's inputs are unchanged in v5.2d "
                                                    "(recover8 is not among the 18 merged units; murray_cap / geom arrays "
                                                    "of unchanged units are bitwise identical)"})
    return out


def e5_dirs() -> list[str]:
    old = str(ROOT / "data_wss_v5/views_v5_1")
    made = []
    for seed in SEEDS:
        src = P.RELEASE / "models" / f"M1_s{seed}"
        out = EXP / "e5_deployed_m1" / f"M1_s{seed}"
        cfg = json.loads((src / "config.json").read_text())
        new = json.loads(json.dumps(cfg).replace(old, str(V)))
        new["data"]["split_path"] = str(FULL_SPLIT)
        new["notes"] = (str(cfg.get("notes", "")) + " | 2026-10-01 E5 read-only copy of the deployed release "
                        "M1_3head_3seed_20260922: data roots v5.1 -> v5.2d for the recover8 reference evaluation; "
                        "weights / statistics unchanged").strip(" |")
        for key in ("data_root", "cycle_view_root"):
            if not Path(new["data"][key]).is_dir():
                raise FileNotFoundError(new["data"][key])
        W1.save(out / "config.json", new)
        for name in ("ckpt_best.pt", "feature_stats.json", "wss_global_stats.json", "target_normalization.json"):
            P.copy_file(src / name, out / name)
        made.append(str(out))
    return made


def main() -> None:
    CONFIGS.mkdir(parents=True, exist_ok=True)
    src_arms = {a["id"]: a for a in json.loads((SRC / "matrix.json").read_text())["arms"]}
    b_ids = [f"X5Dcap_asym2_full265_s{s}" for s in SEEDS]
    a_ids = [f"X5Dcap_asym2_v52cv_f{k}_s{s}" for s in SEEDS for k in FOLDS]
    arms = []

    def ab_arm(aid: str, group: str) -> dict:
        cfg, diff = ab_config(aid)
        W1.save(CONFIGS / f"{aid}.json", cfg)
        arm = {k: v for k, v in src_arms[aid].items() if k not in ("config_diff_vs_parent", "source_config_sha256")}
        arm.update(title=arm["title"].replace("on the corrected data", "on v5.2d"), run_name=cfg["name"], group=group,
                   source_config=f"{SRC_NAME}/{aid}.json", source_config_sha256=P.sha256(SRC / f"{aid}.json"),
                   config_diff_vs_source=sorted(diff))
        return arm

    for aid in b_ids:
        arms.append(ab_arm(aid, "B"))
    stats_path = multi_stats()
    for seed in SEEDS:
        aid, cfg, diff_b, diff_parent = P.m1_config(seed, stats_path)
        cfg["notes"] = ("2026-10-01 v5.2d retrain, group C: the latest three-head recipe M1cap_v52cv (2026-09-25; out_dim 3 = "
                        "[peak-frame WSS log_z, TAWSS log_z, OSI logit_z], equal-weight MSE, q90 pinball off, no asym weight) "
                        "unchanged, trained on v5.2d full265 (265 units) and evaluated on recover8; multi statistics on train265; "
                        "paired init = the same-seed X5Dcap_asym2_full265 config (same backbone initialisation, output layers "
                        "rebuilt).")
        if any(tag in s for s in strings({k: v for k, v in cfg.items() if k != "notes"}) for tag in STALE):
            raise ValueError(f"{aid}: still references an older data version")
        P.validate(cfg, aid)
        W1.save(CONFIGS / f"{aid}.json", cfg)
        arms.append(dict(id=aid, group="C", title=f"M1cap three-head FULL265 seed {seed} on v5.2d", config=f"{aid}.json",
                         run_name=cfg["name"], phase=0, seed=seed, fold=None, protocol="FULL265", arm="M1cap",
                         parent=f"wss_cycle_m1_v52_20260925/M1cap_v52cv_f0_s{seed}.json (recipe) + "
                                f"X5Dcap_asym2_full265_s{seed} (data / paired init)",
                         depends_on=[], evaluate=["best", "last"], single_change=False,
                         evaluation="recover8, three channels (peak WSS / TAWSS / OSI)",
                         config_diff_vs_same_seed_B=diff_b, config_diff_vs_recipe_parent=diff_parent))
    for aid in a_ids:
        arms.append(ab_arm(aid, "A"))
    anchor = anchor_dir()
    e5 = e5_dirs()
    W1.save(CONFIGS / "matrix.json", dict(
        schema_version=1, experiment=NAME, parent_experiment=SRC_NAME,
        section="v5.2d retrain after the 2026-10-01 library audit merge: X5Dcap_asym2 CV5 x 3 seeds + full265 x 3 seeds, and the "
                "M1cap three-head on full265 x 3 seeds; recover8 evaluation (matrix confirmed by the user 2026-09-30, "
                "resubmission requested 2026-10-01)",
        data={"view_root": str(V), "cycle_view_root": str(CYC), "full265_split": str(FULL_SPLIT),
              "full265_split_sha256": P.sha256(FULL_SPLIT), "multi_stats": str(stats_path)},
        execution=f"master 4x RTX 4090 (Slurm), one arm per GPU, frozen code copy {FROZEN}; "
                  "train -> eval best -> eval last (--save-predictions); order B, C, A (long arms first)",
        expected_training_runs=len(arms), expected_evaluations=2 * len(arms), arms=arms,
        not_scheduled={"IND X5Dcap_asym2": "not requested"},
        known_data_limits={"early_exit_iterations": "150 units (AG 105, AAA 25, ILO 20) keep labels from runs whose per-step "
                                                    "iterations stopped at 1e-3; one measured unit: peak WSS rel L2 4.3 % vs the "
                                                    "iteration-converged solution; user decision 2026-10-01: leave as is"},
        preflight_anchor=str(anchor),
        post_queue={"E3": "the 15 CV5 fold models (best) on recover8 via --split-path full265 -> experiments/<exp>/recover8_cv5/",
                    "E5": {"what": "deployed M1_3head_3seed_20260922 (v5.1 train136) on recover8 -> experiments/<exp>/e5_deployed_m1/*/eval",
                           "run_dirs": e5}},
        reading_rule=("recover8 has n = 8: descriptive only, no gates. B: per seed + three-seed ensemble physical R2_cb and per-case "
                      "median vs the v5.2p4 full265 runs (job 16192, same seeds: 0.8295 / 0.8171 / 0.8311, ensemble 0.8393; "
                      "recover8 labels unchanged, so the difference = training data changes of v5.2c + v5.2d + run-to-run noise). "
                      "A: CV5 pooled out-of-fold over 261 per seed (mean +- sd) and three-seed OOF ensemble vs the v5.2 runs "
                      "(0.7714). Held-out labels changed for the 9 label-fix units of 09-30 and HAN_JIAN_FU; "
                      "ILO/WANG_TIAN_QING-1/after is a new geometry (different wall points), so 'old model, new labels' is "
                      "reported on the units whose wall rows match. E3: per-seed five-fold ensemble and 15-model ensemble on "
                      "recover8 vs B. C: per channel on recover8 per seed and ensemble; guard = peak channel normalized R2_cb vs "
                      "the same-seed B run (drop <= 0.02); E5 = the deployed three-head on the same 8 units.")))
    print(CONFIGS / "matrix.json", len(arms), "arms;", "anchor", anchor)


if __name__ == "__main__":
    main()

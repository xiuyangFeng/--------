"""Prepare the v5.2c retrain execution matrix (2026-09-30, matrix confirmed by the user).

Training arms (21 runs, master 4x RTX 4090 via Slurm, one arm per GPU; node04 stays free for the full-cycle time line):
  B  X5Dcap_asym2_full265_s{1234,7,2025}       verbatim copies of configs/wss_v52c_labelfix_20260930 (run names unchanged)
  C  M1cap_full265_s{1234,7,2025}              latest three-head recipe (M1cap_v52cv, 2026-09-25) unchanged -- peak-frame WSS log_z,
                                               TAWSS log_z, OSI logit_z; equal-weight MSE; q90 pinball off; no asym (the asym weight
                                               only exists in the scalar loss branch) -- on v5.2c full265 (train265 -> recover8)
  A  X5Dcap_asym2_v52cv_f{0..4}_s{1234,7,2025} verbatim copies (CV5, 261 units, held-out fold)
matrix.json lists B, C, A in that order: the queue launches pending arms in matrix order, so the long arms start first.

Also written (all idempotent; an existing different file is never changed):
  experiments/<exp>/stats/multi_stats_full265_train265.json
      channel 0 = views_v5_2c wss_global_stats_full265_train265 (log_z), 1 = cycle_stats_tawss_full265_train265 (log_z),
      2 = cycle_stats_osi_logit_full265_train265 (logit_z); all three on the same split file (sha256 checked).
  experiments/<exp>/anchor/X5Dcap_asym2_full265_s1234_v52c/
      GPU-preflight anchor: the pre-label-fix full265 run of job 16192 with its config repointed to v5.2c. The old view roots
      are deleted; recover8 labels, the model's inputs (murray_cap flowref, geom) and the run's own statistics are unchanged
      in v5.2c, so its stored recover8 metrics must reproduce with the frozen code.
  experiments/<exp>/e5_deployed_m1/M1_s{seed}/
      E5 reference: the deployed three-head release M1_3head_3seed_20260922 (v5.1 train136) as read-only run dirs whose config
      points at v5.2c (its inputs log_q_branch_murray / log_tau0_murray and geom_v2 exist there for every unit).

    python -u -m training_wss_min.tools.prepare_wss_v52c_retrain
"""
from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path

from training_wss_min import config as C
from training_wss_min.tools import prepare_wss_local_wave1 as W1
from training_wss_min.tools.repoint_data_root import repoint

ROOT = C.PROJECT_ROOT
NAME = "wss_v52c_retrain_20260930"
CONFIGS = ROOT / "training_wss_min/configs" / NAME
EXP = ROOT / "training_wss_min/experiments" / NAME
SRC = ROOT / "training_wss_min/configs/wss_v52c_labelfix_20260930"
M1_SRC = ROOT / "training_wss_min/configs/wss_cycle_m1_v52_20260925"
V = ROOT / "data_wss_v5/views_v5_2c_20260930"
VIEW = V / "wss_min_view_v1"
CYC = V / "wss_min_cycle_v1"
FULL_SPLIT = VIEW / "split_v52p4_full265_train265_test8.json"
OLD_FULL_RUN = ROOT / "training_wss_min/runs/wss_v52p4_full265_20260930/X5Dcap_asym2_full265_s1234"
RELEASE = ROOT / "outputs/wss_deploy_release/M1_3head_3seed_20260922"
SEEDS = (1234, 7, 2025)
FOLDS = range(5)
# the only fields in which the three-head arm may differ from the same-seed X5Dcap_asym2 full265 arm
M1_VS_B = {"data.target", "data.cycle_view_root", "data.wss_stats_path", "model.out_dim", "train.loss_pinball_lambda",
           "train.loss_asym_under_weight", "train.init_reference_config", "eval.threshold_masks_above",
           "eval.threshold_masks_below"}
# ... and from its CV5 parent M1cap_v52cv_f0 (data version / partition only)
M1_VS_PARENT = {"data.data_root", "data.split_path", "data.wss_stats_path", "data.feature_stats_path",
                "data.point_features_root", "data.density_aug_root", "data.cycle_view_root", "train.init_reference_config"}


def sha256(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def copy_file(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists():
        if sha256(dst) != sha256(src):
            raise FileExistsError(f"refusing to change {dst}")
        return
    shutil.copy2(src, dst)


def validate(cfg: dict, aid: str) -> None:
    classes = {"data": C.DataConfig, "model": C.ModelConfig, "train": C.TrainConfig, "eval": C.EvalConfig}
    for section, cls in classes.items():
        unknown = set(cfg[section]) - set(cls.__dataclass_fields__)
        if unknown:
            raise ValueError(f"{aid}: unknown {section} fields {unknown}")
    C.validate_features(C.ExpConfig.from_dict(cfg))
    for key in ("wss_stats_path", "feature_stats_path", "split_path"):
        if not Path(cfg["data"][key]).is_file():
            raise FileNotFoundError(f"{aid}: {key} missing: {cfg['data'][key]}")
    for root in [*cfg["data"]["point_features_root"], cfg["data"]["data_root"], cfg["data"].get("cycle_view_root") or V]:
        if not Path(root).is_dir():
            raise FileNotFoundError(f"{aid}: missing root {root}")


def multi_stats() -> Path:
    path = EXP / "stats" / "multi_stats_full265_train265.json"
    wss_path = VIEW / "wss_global_stats_full265_train265.json"
    tawss_path = CYC / "stats" / "cycle_stats_tawss_full265_train265.json"
    osi_path = CYC / "stats" / "cycle_stats_osi_logit_full265_train265.json"
    wss, tawss, osi = (json.loads(p.read_text()) for p in (wss_path, tawss_path, osi_path))
    for name, st, expect in (("wss", wss, "log_z"), ("tawss", tawss, "log_z"), ("osi", osi, "logit_z")):
        if st.get("method") != expect:
            raise ValueError(f"{name} statistics must be {expect}")
    split_sha = sha256(FULL_SPLIT)
    if tawss["split_sha256"] != split_sha or osi["split_sha256"] != split_sha:
        raise ValueError("cycle statistics were not built on the full265 split")
    if Path(wss["split_path"]).resolve() != FULL_SPLIT.resolve() or int(wss["n_cases"]) != 265 or int(tawss["n_cases"]) != 265:
        raise ValueError("peak-frame statistics are not the full265 train265 file")
    out = {"schema_version": 1, "method": "multi", "channels": list(C.MULTI_CHANNELS), "eps": wss["eps"], "floor": tawss["floor"],
           "scope": "full265_train265",
           "statistics_scope": "v5.2c full265 train partition only (265 units; peak-frame log_z = the X5Dcap full265 file; "
                               "cycle = views_v5_2c wss_min_cycle_v1)",
           "sources": {"wss": str(wss_path), "tawss": str(tawss_path), "osi": str(osi_path)},
           "wss": wss, "tawss": tawss, "osi": osi}
    W1.save(path, out)
    return path


def m1_config(seed: int, stats_path: Path) -> tuple[str, dict, dict, dict]:
    parent_path = M1_SRC / f"M1cap_v52cv_f0_s{seed}.json"
    parent = json.loads(parent_path.read_text())
    b_path = CONFIGS / f"X5Dcap_asym2_full265_s{seed}.json"
    b = json.loads(b_path.read_text())
    cfg = json.loads(json.dumps(parent))
    aid = f"M1cap_full265_s{seed}"
    cfg["name"] = f"{NAME}/{aid}"
    for key in ("data_root", "split_path", "feature_stats_path", "point_features_root", "density_aug_root"):
        cfg["data"][key] = b["data"][key]
    cfg["data"]["cycle_view_root"] = str(CYC)
    cfg["data"]["wss_stats_path"] = str(stats_path)
    cfg["train"]["init_reference_config"] = str(b_path)
    cfg["notes"] = (f"2026-09-30 v5.2c retrain, group C: the latest three-head recipe M1cap_v52cv (2026-09-25; out_dim 3 = "
                    f"[peak-frame WSS log_z, TAWSS log_z, OSI logit_z], equal-weight MSE, q90 pinball off, no asym weight) unchanged, "
                    f"trained on v5.2c full265 (265 units) and evaluated on recover8; multi statistics on train265; paired init = the "
                    f"same-seed X5Dcap_asym2_full265 config (same backbone initialisation, output layers rebuilt).")
    validate(cfg, aid)
    diff_b = W1.declared_diff(b, cfg)
    diff_parent = W1.declared_diff(parent, cfg)
    if set(diff_b) - M1_VS_B:
        raise ValueError(f"{aid}: unexpected differences from X5Dcap_asym2_full265: {sorted(set(diff_b) - M1_VS_B)}")
    if set(diff_parent) - M1_VS_PARENT:
        raise ValueError(f"{aid}: unexpected differences from M1cap_v52cv_f0: {sorted(set(diff_parent) - M1_VS_PARENT)}")
    return aid, cfg, diff_b, diff_parent


def anchor_dir() -> Path:
    out = EXP / "anchor" / "X5Dcap_asym2_full265_s1234_v52c"
    cfg = json.loads((OLD_FULL_RUN / "config.json").read_text())
    hits: list = []
    new = repoint(cfg, hits)
    if not hits:
        raise RuntimeError("anchor config references no replaced root")
    new["notes"] = (str(cfg.get("notes", "")) + " | 2026-09-30 preflight anchor copy: data roots repointed to v5.2c; "
                    "checkpoint / statistics / stored metrics copied from job 16192 unchanged").strip(" |")
    W1.save(out / "config.json", new)
    for name in ("ckpt_best.pt", "feature_stats.json", "wss_global_stats.json", "target_normalization.json"):
        copy_file(OLD_FULL_RUN / name, out / name)
    copy_file(OLD_FULL_RUN / "eval/ckpt_best/metrics.json", out / "eval/ckpt_best/metrics.json")
    W1.save(out / "ANCHOR_SOURCE.json", {"source_run": str(OLD_FULL_RUN), "repointed_fields": len(hits),
                                          "reason": "old view roots deleted 2026-09-30; recover8 labels and inputs unchanged"})
    return out


def e5_dirs() -> list[str]:
    old = str(ROOT / "data_wss_v5/views_v5_1")
    made = []
    for seed in SEEDS:
        src = RELEASE / "models" / f"M1_s{seed}"
        out = EXP / "e5_deployed_m1" / f"M1_s{seed}"
        cfg = json.loads((src / "config.json").read_text())
        new = json.loads(json.dumps(cfg).replace(old, str(V)))
        new["data"]["split_path"] = str(FULL_SPLIT)
        new["notes"] = (str(cfg.get("notes", "")) + " | 2026-09-30 E5 read-only copy of the deployed release "
                        "M1_3head_3seed_20260922: data roots v5.1 -> v5.2c for the recover8 reference evaluation; "
                        "weights / statistics unchanged").strip(" |")
        for key in ("data_root", "cycle_view_root"):
            if not Path(new["data"][key]).is_dir():
                raise FileNotFoundError(new["data"][key])
        W1.save(out / "config.json", new)
        for name in ("ckpt_best.pt", "feature_stats.json", "wss_global_stats.json", "target_normalization.json"):
            copy_file(src / name, out / name)
        made.append(str(out))
    return made


def main() -> None:
    CONFIGS.mkdir(parents=True, exist_ok=True)
    src_matrix = {a["id"]: a for a in json.loads((SRC / "matrix.json").read_text())["arms"]}
    # verbatim copies (A, B)
    b_ids = [f"X5Dcap_asym2_full265_s{s}" for s in SEEDS]
    a_ids = [f"X5Dcap_asym2_v52cv_f{k}_s{s}" for s in SEEDS for k in FOLDS]
    for aid in b_ids + a_ids:
        copy_file(SRC / f"{aid}.json", CONFIGS / f"{aid}.json")
        validate(json.loads((CONFIGS / f"{aid}.json").read_text()), aid)
    stats_path = multi_stats()
    arms = []
    for aid in b_ids:
        arm = dict(src_matrix[aid])
        arms.append({**arm, "group": "B", "evaluation": "recover8 (test partition of the full265 split)",
                     "source_config_sha256": sha256(SRC / f"{aid}.json")})
    for seed in SEEDS:
        aid, cfg, diff_b, diff_parent = m1_config(seed, stats_path)
        W1.save(CONFIGS / f"{aid}.json", cfg)
        arms.append(dict(id=aid, group="C", title=f"M1cap three-head FULL265 seed {seed} on v5.2c", config=f"{aid}.json",
                         run_name=cfg["name"], phase=0, seed=seed, fold=None, protocol="FULL265", arm="M1cap",
                         parent=f"wss_cycle_m1_v52_20260925/M1cap_v52cv_f0_s{seed}.json (recipe) + "
                                f"X5Dcap_asym2_full265_s{seed} (data / paired init)",
                         depends_on=[], evaluate=["best", "last"], single_change=False,
                         evaluation="recover8, three channels (peak WSS / TAWSS / OSI)",
                         config_diff_vs_same_seed_B=diff_b, config_diff_vs_recipe_parent=diff_parent))
    for aid in a_ids:
        arm = dict(src_matrix[aid])
        arms.append({**arm, "group": "A", "evaluation": "held-out CV5 fold (pooled out-of-fold over 261); E3 adds recover8",
                     "source_config_sha256": sha256(SRC / f"{aid}.json")})
    anchor = anchor_dir()
    e5 = e5_dirs()
    W1.save(CONFIGS / "matrix.json", dict(
        schema_version=1, experiment=NAME, parent_experiment="wss_v52c_labelfix_20260930",
        section="v5.2c retrain after the 2026-09-30 label fix: X5Dcap_asym2 CV5 x 3 seeds + full265 x 3 seeds, and the M1cap "
                "three-head on full265 x 3 seeds; recover8 evaluation (user-confirmed matrix 2026-09-30)",
        data={"view_root": str(V), "cycle_view_root": str(CYC), "full265_split": str(FULL_SPLIT),
              "full265_split_sha256": sha256(FULL_SPLIT), "multi_stats": str(stats_path)},
        execution="master 4x RTX 4090 (Slurm), one arm per GPU, frozen code copy GNN_v52c_frozen_20260930; "
                  "train -> eval best -> eval last (--save-predictions); order B, C, A (long arms first)",
        expected_training_runs=len(arms), expected_evaluations=2 * len(arms), arms=arms,
        not_scheduled={"IND X5Dcap_asym2 s1234/s7/s2025/s11/s2026": "prepared in wss_v52c_labelfix_20260930, not requested"},
        preflight_anchor=str(anchor),
        post_queue={"E3": "the 15 CV5 fold models (best) on recover8 via --split-path full265 -> experiments/<exp>/recover8_cv5/",
                    "E5": {"what": "deployed M1_3head_3seed_20260922 (v5.1 train136) on recover8 -> experiments/<exp>/e5_deployed_m1/*/eval",
                           "run_dirs": e5}},
        reading_rule=("recover8 has n = 8: descriptive only, no gates. B: per seed + three-seed ensemble physical R2_cb and per-case "
                      "median vs the pre-label-fix full265 runs (job 16192, same seeds / GPU type: 0.8295 / 0.8171 / 0.8311, "
                      "ensemble 0.8393; recover8 labels unchanged, so the difference = training labels of 9 units + run-to-run "
                      "noise). A: CV5 pooled out-of-fold over 261 per seed (mean +- sd) and three-seed OOF ensemble vs the "
                      "pre-label-fix 0.7714 (the 9 fixed units are both training and held-out labels, so this mixes label change "
                      "and seed noise). E3: per-seed five-fold ensemble and 15-model ensemble on recover8 vs B. C: per channel on "
                      "recover8 (peak Pa / normalized R2_cb, TAWSS Pa R2_cb / CCC / <0.4 Pa IoU, OSI R2 / >0.1 and >0.3 IoU, "
                      "low-TAWSS-high-OSI zone) per seed and ensemble; guard = peak channel normalized R2_cb vs the same-seed B "
                      "run (drop <= 0.02, the historical M1 gate; the Pa tail is expected lower because C has no pinball / asym); "
                      "E5 = the deployed three-head on the same 8 units as the TAWSS / OSI reference.")))
    print(CONFIGS / "matrix.json", len(arms), "arms;", "anchor", anchor)


if __name__ == "__main__":
    main()

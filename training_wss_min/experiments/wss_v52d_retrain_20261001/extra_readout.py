"""Extra numbers for the v5.2d retrain readout (read-only on saved predictions; best checkpoint).
  A  CV5 out-of-fold three-seed ensembles on the 260 pairable units: new models vs v5.2 models on old / new labels;
     per cohort; the units whose labels changed, one by one; ILO/WANG_TIAN_QING-1/after (new geometry) for the new models.
  B  why the stored v5.2p4 full265 ensemble (0.8393, tracker 37.1) differs from the readout's 0.8401.
"""
import json
from pathlib import Path
import numpy as np
from training_wss_min.tools import report_wss_v52d_retrain as D
R = D.R
out = {}
new_s, old_s, oldnew_s, rows_bad = {}, {}, {}, set()
for s in R.SEEDS:
    pn, po, pon = {}, {}, {}
    for k in R.FOLDS:
        rn, ro = {}, {}
        n = R.load_channel(R.pred_dir(R.NEW / f"X5Dcap_asym2_v52cv_f{k}_s{s}", "best"), rows=rn)
        o = R.load_channel(R.pred_dir(R.OLD_CV / f"X5Dcap_asym2_v52cv_f{k}_s{s}", "best"), rows=ro)
        pn.update(n)
        for u, (yn, _) in n.items():
            yo, pp = o[u]
            if len(yo) != len(yn) or not np.array_equal(rn[u], ro[u]):
                rows_bad.add(u); continue
            po[u] = o[u]; pon[u] = (yn, pp)
    new_s[s], old_s[s], oldnew_s[s] = pn, po, pon
en, eo, eon = R.ensemble(list(new_s.values())), R.ensemble(list(old_s.values())), R.ensemble(list(oldnew_s.values()))
pair = sorted(set(en) & set(eon))
sub = lambda d, us: {u: d[u] for u in us}
out["A_ensemble"] = {"n_all": len(en), "new_all": R.r2cb(en), "n_paired": len(pair), "new_paired": R.r2cb(sub(en, pair)),
                     "v52_models_old_labels": R.r2cb(sub(eo, pair)), "v52_models_new_labels": R.r2cb(sub(eon, pair)),
                     "unit_median_new": R.unit_median(sub(en, pair)), "unit_median_v52_new_labels": R.unit_median(sub(eon, pair))}
d = np.array([R.r2(*en[u]) - R.r2(*eon[u]) for u in pair])
out["A_ensemble"]["per_unit_delta"] = {"mean": float(d.mean()), "median": float(np.median(d)), "improved": int((d > 0).sum()), "n": len(d)}
for c in ("AG", "AAA", "ILO"):
    us = [u for u in pair if u.startswith(c + "/")]
    out.setdefault("A_by_cohort", {})[c] = {"n": len(us), "new": R.r2cb(sub(en, us)), "v52_models_new_labels": R.r2cb(sub(eon, us)), "v52_models_old_labels": R.r2cb(sub(eo, us))}
changed = [u for u in pair if not np.array_equal(eo[u][0], eon[u][0])]
out["A_changed_units"] = {u: {"new_model_new_label": R.r2(*en[u]), "v52_model_new_label": R.r2(*eon[u]), "v52_model_old_label": R.r2(*eo[u])} for u in changed}
un = [u for u in pair if u not in changed]
out["A_unchanged_units"] = {"n": len(un), "new": R.r2cb(sub(en, un)), "v52_models": R.r2cb(sub(eon, un))}
out["A_not_pairable"] = {u: {"new_model": R.r2(*en[u])} for u in sorted(rows_bad)}
# B: definition of the stored 0.8393
olds = [R.load_channel(R.pred_dir(R.OLD_FULL / f"X5Dcap_asym2_full265_s{s}", "best")) for s in R.SEEDS]
news = [R.load_channel(R.pred_dir(R.NEW / f"X5Dcap_asym2_full265_s{s}", "best")) for s in R.SEEDS]
def ens(members, mode):
    us = sorted(members[0])
    f = (lambda p: np.mean(p, axis=0)) if mode == "pa" else (lambda p: np.exp(np.mean(np.log(np.maximum(p, 1e-6)), axis=0)))
    return {u: (members[0][u][0], f([m[u][1] for m in members])) for u in us}
out["B_ensemble_definitions"] = {m: {"v52p4": R.r2cb(ens(olds, m)), "v52d": R.r2cb(ens(news, m))} for m in ("pa", "log")}
Path(__file__).with_name("extra_readout.json").write_text(json.dumps(out, indent=1, ensure_ascii=False))
print(json.dumps(out, indent=1, ensure_ascii=False))

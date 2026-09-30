"""A/B: prepared device inputs on/off, alternating, same process; outputs vs run-to-run jitter."""
import json, os, pickle, statistics, sys, time
from pathlib import Path
import numpy as np
ROOT = Path("/public/newhome/cy/Digital_twin/GNN"); sys.path.insert(0, str(ROOT))
import torch
from wss_deploy.infer import Release

RELEASES = {"x5d": "X5D_v51_5seed_20260916", "m1": "M1_3head_3seed_20260922", "vol": "PF6_VF6_peak_3seed_20260920"}

def arrays(result):
    return {k: np.asarray(v) for k, v in result.items() if isinstance(v, np.ndarray) and v.dtype.kind == "f"}

def maxdiff(a, b):
    return {k: float(np.max(np.abs(a[k] - b[k]))) for k in a}

def run(release, case, mode):
    os.environ["WSS_DEPLOY_PREPARED_INPUTS"] = "1" if mode == "prepared" else "0"
    torch.cuda.synchronize(); t = time.perf_counter()
    result = release.predict(case)
    torch.cuda.synchronize()
    return time.perf_counter() - t, result

def main(names, rounds):
    out = {}
    releases = {}
    for name in names:
        family = name.split("_")[1]
        if family not in releases:
            releases[family] = Release(ROOT / "outputs/wss_deploy_release" / RELEASES[family], device="cuda")
        release = releases[family]
        case = pickle.load(open(Path(__file__).parent / f"case_{name}.pkl", "rb"))
        times = {"baseline": [], "prepared": []}; results = {"baseline": [], "prepared": []}; stats = None
        run(release, case, "baseline"); run(release, case, "prepared")          # warm-up both paths
        for r in range(rounds):
            for mode in (("baseline", "prepared") if r % 2 == 0 else ("prepared", "baseline")):
                dt, res = run(release, case, mode)
                times[mode].append(dt); results[mode].append(arrays(res))
                if mode == "prepared": stats = res.get("prepared_inputs")
        base0 = results["baseline"][0]
        jitter = {k: max(maxdiff(base0, b)[k] for b in results["baseline"][1:]) for k in base0}
        prep = {k: max(maxdiff(base0, p)[k] for p in results["prepared"]) for k in base0}
        prep_self = {k: max(maxdiff(results["prepared"][0], p)[k] for p in results["prepared"][1:]) for k in base0}
        bit = all(all(np.array_equal(base0[k], p[k]) for k in base0) for p in results["prepared"])
        out[name] = {"n_pos": int(len(case["pos"])), "models": len(release.models),
                     "baseline_s": [round(x, 3) for x in times["baseline"]], "prepared_s": [round(x, 3) for x in times["prepared"]],
                     "baseline_median_s": round(statistics.median(times["baseline"]), 3),
                     "prepared_median_s": round(statistics.median(times["prepared"]), 3),
                     "baseline_run_to_run_max_abs": jitter, "prepared_vs_baseline_max_abs": prep,
                     "prepared_run_to_run_max_abs": prep_self, "prepared_bit_identical_to_baseline_run0": bit,
                     "prepared_stats": stats}
        print(json.dumps({name: {k: out[name][k] for k in ("baseline_median_s", "prepared_median_s", "prepared_bit_identical_to_baseline_run0", "prepared_stats")}}, ensure_ascii=False), flush=True)
    return out

if __name__ == "__main__":
    rounds = int(sys.argv[1]); names = sys.argv[2:]
    res = main(names, rounds)
    res["_env"] = {"gpu": torch.cuda.get_device_name(0), "torch": torch.__version__, "visible": os.environ.get("CUDA_VISIBLE_DEVICES")}
    Path(__file__).with_name("bench_prepared.json").write_text(json.dumps(res, indent=1, ensure_ascii=False))

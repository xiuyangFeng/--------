import json, os, pickle, sys, time
from pathlib import Path
ROOT = Path("/public/newhome/cy/Digital_twin/GNN"); sys.path.insert(0, str(ROOT))
import torch
from wss_deploy.infer import Release
mode, name = sys.argv[1], sys.argv[2]
os.environ["WSS_DEPLOY_PREPARED_INPUTS"] = "1" if mode == "on" else "0"
rel = Release(ROOT / "outputs/wss_deploy_release/M1_3head_3seed_20260922", device="cuda")
rel.warm_up()
case = pickle.load(open(Path(__file__).parent / f"case_{name}.pkl", "rb"))
out = []
for i in range(3):
    torch.cuda.synchronize(); t = time.perf_counter(); r = rel.predict(case); torch.cuda.synchronize(); out.append(round(time.perf_counter() - t, 3))
print(json.dumps({"mode": mode, "case": name, "calls_s": out, "per_model_first": [round(x, 3) for x in r["seconds_per_model"]]}))

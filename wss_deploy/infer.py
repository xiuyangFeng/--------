"""Stage 3: the frozen release (5 × X5D_v51) -> per-point WSS in Pa; ensemble = arithmetic mean in Pa space."""
from __future__ import annotations
import json, time
from pathlib import Path
import numpy as np
import torch
from training_wss_min import dataset as D, evaluate as E
from .paths import RELEASE_DIR, SEEDS


class Release:
    def __init__(self, release_dir: Path = RELEASE_DIR, device: str = "auto", seeds=SEEDS):
        self.dir = Path(release_dir)
        self.device = ("cuda" if torch.cuda.is_available() else "cpu") if device == "auto" else device
        self.info = json.loads((self.dir / "release.json").read_text(encoding="utf-8"))
        self.models = []
        t = time.perf_counter()
        for s in seeds:
            run = self.dir / "models" / f"X5D_v51_s{s}"
            cfg, feat_stats, model, _ = E.load_model_from_run(run, self.device, "best"); model.eval()
            self.models.append({"seed": s, "cfg": cfg, "feat_stats": feat_stats, "model": model, "stats": E.load_wss_stats_for_run(run)})
        self.load_seconds = time.perf_counter() - t
        self.input_features = list(self.models[0]["cfg"].data.input_features)

    @property
    def name(self) -> str:
        return self.info.get("release", self.dir.name)

    def predict(self, case: dict) -> dict:
        preds, per = [], []
        for m in self.models:
            t = time.perf_counter()
            with torch.no_grad():
                p = E.predict_case_norm(m["model"], case, m["cfg"].data.input_features, m["feat_stats"], self.device, cfg=m["cfg"])
            if self.device == "cuda":
                torch.cuda.synchronize()
            per.append(time.perf_counter() - t)
            preds.append(D.denormalize_wss(np.asarray(p, dtype=np.float64), m["stats"]))
        P = np.stack(preds)
        return {"wss_pa": P.mean(axis=0), "seed_pred_pa": P, "seed_sd_pa": P.std(axis=0), "seconds_per_model": per, "device": self.device,
                "gpu": torch.cuda.get_device_name(0) if self.device == "cuda" else "cpu"}

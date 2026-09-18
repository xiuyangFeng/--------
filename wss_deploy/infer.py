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
        self.release_id = str(self.info.get("release", self.dir.name))
        if not self.release_id:
            raise ValueError("release.json 必须包含非空 release 标识。")
        self.model_specs = self._model_specs(seeds)
        self.models = []
        t = time.perf_counter()
        for spec in self.model_specs:
            s, run = spec["seed"], self.dir / spec["path"]
            if not run.is_dir():
                raise FileNotFoundError(f"发布包模型目录不存在：{run}")
            cfg, feat_stats, model, _ = E.load_model_from_run(run, self.device, "best"); model.eval()
            self.models.append({"seed": s, "cfg": cfg, "feat_stats": feat_stats, "model": model, "stats": E.load_wss_stats_for_run(run)})
        self.load_seconds = time.perf_counter() - t
        if not self.models:
            raise ValueError("发布包至少需要一个模型权重。")
        self.input_features = list(self.models[0]["cfg"].data.input_features)
        if any(list(item["cfg"].data.input_features) != self.input_features for item in self.models[1:]):
            raise ValueError("同一发布包内模型的输入特征合同不一致。")

    def _model_specs(self, seeds):
        """Resolve model paths from release.json, retaining the v5 fallback.

        A new weight family can declare ``models`` as a list of
        ``{"seed": ..., "path": "models/..."}`` records.  Older frozen
        releases omit it and keep the historical X5D_v51_s{seed} layout.
        """
        declared = self.info.get("models") or self.info.get("model_runs")
        if declared:
            if isinstance(declared, dict):
                declared = [{"seed": seed, "path": path} for seed, path in declared.items()]
            if not isinstance(declared, list):
                raise ValueError("release.json 的 models 必须是列表或映射。")
            specs = []
            for index, item in enumerate(declared):
                if isinstance(item, str):
                    item = {"path": item, "seed": index}
                if not isinstance(item, dict) or not item.get("path"):
                    raise ValueError(f"release.json 的 models[{index}] 缺少 path。")
                rel = Path(str(item["path"]))
                if rel.is_absolute() or ".." in rel.parts:
                    raise ValueError(f"release.json 的模型路径必须位于发布包内：{rel}")
                specs.append({"seed": item.get("seed", index), "path": rel.as_posix()})
            return specs
        return [{"seed": seed, "path": f"models/X5D_v51_s{seed}"} for seed in seeds]

    @property
    def name(self) -> str:
        return self.release_id

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

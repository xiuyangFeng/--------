"""Case-scoped device inputs for frozen fixed-support deployment models.

Only inputs are reused: every ensemble member still encodes its own support and
decodes the original query chunks. Keys include array content, feature statistics,
data/evaluation options and frame, never just a mutable object's identity. The
scope owns the tensors and releases them on exit; training/evaluate are unchanged.

``WSS_DEPLOY_PREPARED_INPUTS=0`` restores the evaluator path. The retained input
budget is ``WSS_DEPLOY_PREPARED_INPUTS_MB`` (1024 MiB by default; on a GPU at most a
quarter of the free memory). Chunks beyond it are prepared normally without retention.
CUDA OOM clears this cache and retries the original evaluator. Normal-sized outputs
are copied to CPU once per model; outputs above 64 MiB keep the evaluator's bounded,
per-chunk transfer behavior.

While every block is retained, the builders are called without ``input_memo`` (whose
private copies no member would read) and the first member builds the later query
chunks on a few threads while the GPU runs the earlier ones.  Every chunk is a pure
function of the case, so the arrays are the ones a serial build returns.
"""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
import dataclasses
import hashlib
import inspect
import math
import os
from pathlib import Path
import struct

import numpy as np


def _enabled() -> bool:
    return os.environ.get("WSS_DEPLOY_PREPARED_INPUTS", "1").strip().lower() not in {
        "0", "false", "off", "no",
    }


_PREBUILD_THREADS = 8


def _budget_bytes() -> int:
    try:
        mb = float(os.environ.get("WSS_DEPLOY_PREPARED_INPUTS_MB", "1024"))
        return max(0, int(mb * 1024 * 1024)) if math.isfinite(mb) else 1024 * 1024 * 1024
    except ValueError:
        return 1024 * 1024 * 1024


def _fingerprint(value) -> bytes:
    """Hash exact content, rejecting opaque objects instead of using their id/repr."""
    digest = hashlib.blake2b(digest_size=32)

    def add(item):
        if dataclasses.is_dataclass(item) and not isinstance(item, type):
            item = {field.name: getattr(item, field.name) for field in dataclasses.fields(item)}
        if isinstance(item, np.ndarray):
            if item.dtype.hasobject:
                raise TypeError("object arrays cannot key prepared inputs")
            digest.update(b"array")
            add(item.dtype.str)
            add(item.shape)
            digest.update(np.ascontiguousarray(item).tobytes())
        elif isinstance(item, dict):
            digest.update(b"dict")
            # Deployment input/config dictionaries have string keys.
            if not all(isinstance(key, str) for key in item):
                raise TypeError("non-string input key")
            for key in sorted(item):
                add(key)
                add(item[key])
            digest.update(b"end")
        elif isinstance(item, (tuple, list)):
            digest.update(b"sequence")
            add(len(item))
            for element in item:
                add(element)
        elif isinstance(item, (str, Path)):
            raw = str(item).encode("utf-8")
            digest.update(b"str" + struct.pack("!Q", len(raw)) + raw)
        elif isinstance(item, bytes):
            digest.update(b"bytes" + struct.pack("!Q", len(item)) + item)
        elif item is None:
            digest.update(b"none")
        elif isinstance(item, (bool, np.bool_)):
            digest.update(b"true" if item else b"false")
        elif isinstance(item, (int, np.integer)):
            digest.update(b"int" + str(int(item)).encode("ascii") + b";")
        elif isinstance(item, (float, np.floating)):
            digest.update(b"float" + struct.pack("!d", float(item)))
        else:
            raise TypeError(f"unsupported input key type: {type(item).__name__}")

    add(value)
    return digest.digest()


def _supported_model(model) -> bool:
    from training_wss_min.baseline_models import PointNetPlusPlusRegressor, PointNetRegressor
    # Do not assume unknown subclasses/custom models leave input tensors intact.
    return type(model) in (PointNetPlusPlusRegressor, PointNetRegressor) and not model.training


def _stock_predictor(predictor) -> bool:
    original = inspect.unwrap(predictor)
    return (getattr(original, "__module__", None) == "training_wss_min.evaluate"
            and getattr(original, "__name__", None) == "predict_case_norm")


def _arrays(block):
    for value in block.values():
        if isinstance(value, dict):
            yield from _arrays(value)
        else:
            yield value


def _tensors(block):
    for value in block.values():
        if isinstance(value, dict):
            yield from _tensors(value)
        else:
            yield value


class PreparedInference:
    """One ensemble's read-only inputs; use as a context manager, never globally."""

    def __init__(self):
        self.stats = {"calls": 0, "reused": 0, "built": 0, "fallbacks": 0,
                      "peak_bytes": 0, "output_transfers": 0}
        self._entries = {}
        self._cases = {}
        self._bytes = 0
        self._active = False
        self._overflow = False
        self._budget = _budget_bytes()

    def __enter__(self):
        self._active = True
        return self

    def clear(self):
        self._entries.clear()
        self._cases.clear()
        self._bytes = 0
        self._overflow = False

    def _case_fingerprint(self, case) -> bytes:
        """Content hash of the case's public entries, once per case dict and scope.

        Members receive the same dict (``input_memo`` relies on that identity too); a
        replaced entry is detected and rehashed.  The two private patch caches are
        derived from the public inputs and filled in by the first member.
        """
        public = {k: v for k, v in case.items()
                  if k not in {"_full_wall_tree", "_full_wall_features"}
                  and not k.startswith("_uniform_voxel_groups_")}
        entries = tuple((k, id(v)) for k, v in public.items())
        cached = self._cases.get(id(case))
        if cached is not None and cached[0] is case and cached[1] == entries:
            return cached[2]
        digest = _fingerprint(public)
        self._cases[id(case)] = (case, entries, digest)   # holding the dict keeps its id unique
        return digest

    def __exit__(self, *_):
        self.clear()
        self._active = False

    def _block(self, key, build, device):
        import torch
        entry = self._entries.get(key)
        if entry is not None:
            block, versions, size = entry
            if versions == tuple(t._version for t in _tensors(block)):
                self.stats["reused"] += 1
                return block
            # An unexpected hook must not poison later members' inputs.
            del self._entries[key]
            self._bytes -= size
        block = build()
        self.stats["built"] += 1
        size = sum(t.numel() * t.element_size() for t in _tensors(block))
        budget = self._budget
        if torch.device(device).type == "cuda":
            free, _ = torch.cuda.mem_get_info(device)
            budget = min(budget, self._bytes + free // 4)
        if self._bytes + size <= budget:
            self._entries[key] = (block, tuple(t._version for t in _tensors(block)), size)
            self._bytes += size
            self.stats["peak_bytes"] = max(self.stats["peak_bytes"], self._bytes)
        else:
            self._overflow = True     # later members rebuild this block: let input_memo share it again
        return block

    def predict(self, predictor, model, case, input_features, feat_stats, device, **kwargs):
        """Adapt the stock evaluator; unknown models and patched evaluators pass through."""
        import torch
        cfg = kwargs.get("cfg")
        self.stats["calls"] += 1
        use_prepared = (self._active and _enabled() and self._budget > 0
                        and set(kwargs) <= {"cfg", "return_all_channels", "frame_index"}
                        and cfg is not None and getattr(getattr(cfg, "eval", None), "fixed_support", False)
                        and _stock_predictor(predictor) and _supported_model(model))
        if use_prepared:
            out_of_memory = False
            try:
                with torch.no_grad():
                    return self._predict(model, case, input_features, feat_stats, device, **kwargs)
            except _UnsupportedKey:
                pass
            except torch.cuda.OutOfMemoryError:
                # Retry outside the except block so the failed frame's tensors
                # and traceback are gone before releasing the allocator cache.
                self.clear()
                out_of_memory = True
            if out_of_memory and torch.device(device).type == "cuda":
                torch.cuda.empty_cache()
        self.stats["fallbacks"] += 1
        return predictor(model, case, input_features, feat_stats, device, **kwargs)

    def _predict(self, model, case, input_features, feat_stats, device, *, cfg,
                 return_all_channels=False, frame_index=None):
        import torch
        from training_wss_min import dataset as D, surface as S

        geometry = D.case_geometry(case) if cfg.data.local_geometry else None
        section = D.case_section(case) if getattr(cfg.data, "section_tokens", False) else None
        try:
            # Each member names its own statistics file; the statistics themselves are keyed as feat_stats.
            data_key = {field.name: getattr(cfg.data, field.name) for field in dataclasses.fields(cfg.data)
                        if field.name != "feature_stats_path"}
            key = _fingerprint((self._case_fingerprint(case), data_key, cfg.eval, tuple(input_features), feat_stats,
                                str(torch.device(device)), frame_index))
        except TypeError as exc:
            raise _UnsupportedKey from exc
        rows = D.query_rows(case)
        chunk = int(cfg.eval.query_chunk_size) or len(rows)
        if chunk <= 0 or len(rows) == 0:
            raise _UnsupportedKey

        def builder(name):
            # Unwrapped: input_memo's private copies are never read while the blocks are retained.
            fn = getattr(D, name)
            return fn if self._overflow else inspect.unwrap(fn)

        def host(idx, *, patch=False):
            block = {"pos": np.ascontiguousarray(case["pos"][idx]),
                     "x": builder("build_features")(case, idx, input_features, feat_stats, frame_index=frame_index)}
            if geometry is not None:
                block["geometry"] = np.ascontiguousarray(geometry[idx])
            if section is not None:
                block["section"] = np.ascontiguousarray(section[idx])
            if patch and cfg.data.query_patch_nsample:
                block["patch"] = builder("build_query_patch")(case, idx, input_features, feat_stats,
                                                             nsample=cfg.data.query_patch_nsample, frame_index=frame_index,
                                                             **D._patch_scale_kwargs(cfg.data))
            return block

        def to_device(block):
            out = {"pos": torch.from_numpy(block["pos"]).to(device),
                   "x": torch.from_numpy(block["x"]).to(device),
                   "batch": torch.zeros(len(block["pos"]), dtype=torch.long, device=device)}
            for name in ("geometry", "section"):
                if name in block:
                    out[name] = torch.from_numpy(block[name]).to(device)
            if "patch" in block:
                out["patch"] = {k: torch.as_tensor(v, device=device) for k, v in block["patch"].items()}
            return out

        def support_inputs():
            support_n = int(cfg.data.support_n_points or cfg.data.wall_n_points)
            sampling = cfg.data.support_sampling or cfg.data.sampling
            seed = S.stable_seed(cfg.eval.support_seed, case["unit_id"], "eval_support")
            idx = builder("sample_support_indices")(case, cfg.data, seed, n_points=support_n,
                                                    sampling=sampling, stream="eval_support")
            return to_device(host(idx))

        support = self._block((key, "support"), support_inputs, device)
        encoded = model.encode_support(**support, unit_ids=[case["unit_id"]], epoch=0,
                                       global_seed=cfg.eval.support_seed, evaluation=True)
        starts = list(range(0, len(rows), chunk))
        missing = [start for start in starts if (key, start) not in self._entries]
        futures = {}

        def query_inputs(start):
            future = futures.pop(start, None)
            block = future.result() if future is not None else host(rows[start:start + chunk], patch=True)
            if missing and start == missing[0] and len(missing) > 1 and not self._overflow:
                # The first build fills the case's lazy KD tree and feature table; the other chunks only
                # read them, so they are built concurrently with this chunk's forward pass.
                size = sum(v.nbytes for v in _arrays(block))
                if self._bytes + size * len(missing) <= self._budget:
                    pool = ThreadPoolExecutor(max_workers=min(len(missing) - 1, _PREBUILD_THREADS))
                    try:
                        for later in missing[1:]:
                            futures[later] = pool.submit(host, rows[later:later + chunk], patch=True)
                    finally:
                        pool.shutdown(wait=False)
            return to_device(block)

        outputs = []
        copy_chunks = None
        for start in starts:
            idx = rows[start:start + chunk]
            query = self._block((key, start), lambda: query_inputs(start), device)
            output = model.decode_query(encoded, **query).detach().float()
            if copy_chunks is None:
                row_size = output.numel() // len(idx) * output.element_size()
                copy_chunks = row_size * len(rows) > 64 * 1024 * 1024
            if copy_chunks:
                output = output.cpu()
                self.stats["output_transfers"] += 1
            outputs.append(output)
        output = torch.cat(outputs)
        if output.ndim == 2 and not return_all_channels:
            output = output[:, 0]
        if not copy_chunks:
            output = output.cpu()
            self.stats["output_transfers"] += 1
        return output.numpy()


class _UnsupportedKey(Exception):
    pass


__all__ = ["PreparedInference"]

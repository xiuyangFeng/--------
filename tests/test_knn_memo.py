"""``wss_deploy.knn_memo`` (v0.12.2): exact-input reuse of the ensemble's kNN graph."""
import sys
import threading
import types

import pytest

torch = pytest.importorskip("torch")

from wss_deploy import knn_memo


def _fake_module(monkeypatch):
    calls = []

    def knn(x, y, k, batch_x=None, batch_y=None, cosine=False, num_workers=1, batch_size=None):
        calls.append(k)
        return torch.stack([torch.arange(len(y)).repeat_interleave(k), torch.arange(len(y) * k) % len(x)])

    module = types.ModuleType("training_wss_min.local_refinement")
    module.knn = knn
    monkeypatch.setitem(sys.modules, "training_wss_min.local_refinement", module)
    monkeypatch.setattr(knn_memo, "_TARGET_MODULES", ("training_wss_min.local_refinement",))
    return module, calls


def test_reuses_only_exactly_equal_inputs(monkeypatch):
    module, calls = _fake_module(monkeypatch)
    x = torch.rand(50, 3); batch = torch.zeros(50, dtype=torch.long)
    with knn_memo.shared_knn() as stats:
        first = module.knn(x, x, 8, batch, batch)
        again = module.knn(x.clone(), x.clone(), k=8, batch_x=batch.clone(), batch_y=batch)   # equal content, new tensors
        other_k = module.knn(x, x, 4, batch, batch)
        nudged = x.clone(); nudged[3, 1] += 1e-7
        other_x = module.knn(nudged, nudged, 8, batch, batch)
    assert torch.equal(first, again) and again.data_ptr() != first.data_ptr()          # a copy, not the cached tensor
    assert calls == [8, 4, 8] and stats == {"calls": 4, "reused": 1}
    assert other_k.shape != first.shape and torch.equal(other_x, first)                 # recomputed (fake is position-free)


def test_passes_through_outside_the_scope_and_in_other_threads(monkeypatch):
    module, calls = _fake_module(monkeypatch)
    x = torch.rand(20, 3)
    knn_memo.install()
    module.knn(x, x, 2); module.knn(x, x, 2)
    assert calls == [2, 2]
    seen = []
    with knn_memo.shared_knn():
        module.knn(x, x, 2)
        worker = threading.Thread(target=lambda: seen.append(module.knn(x, x, 2)))
        worker.start(); worker.join()
        module.knn(x, x, 2)
    assert calls == [2, 2, 2, 2]          # scope call + other thread computed; the second scope call reused


def test_switch_off(monkeypatch):
    module, calls = _fake_module(monkeypatch)
    monkeypatch.setenv("WSS_DEPLOY_KNN_MEMO", "0")
    x = torch.rand(10, 3)
    with knn_memo.shared_knn() as stats:
        module.knn(x, x, 2); module.knn(x, x, 2)
    assert calls == [2, 2] and stats == {"calls": 0, "reused": 0}

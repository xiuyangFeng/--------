import copy

import pytest
import torch
from torch import nn

from training_wss_min.joint_cycle_pcgrad import project_conflicting_gradients, pcgrad_backward


def test_conflict_projection_matches_closed_form_and_zero_task_is_safe():
    vectors = torch.tensor([[1., 0.], [-1., 1.]])
    merged, diag = project_conflicting_gradients(vectors, seed=1234)
    torch.testing.assert_close(merged, torch.tensor([.25, .75]))
    assert diag["conflict_fraction"] == 1
    no_conflict, _ = project_conflicting_gradients(torch.tensor([[1., 2.], [0., 0.]]), 1234)
    torch.testing.assert_close(no_conflict, torch.tensor([.5, 1.]))
    with pytest.raises(FloatingPointError):
        project_conflicting_gradients(torch.tensor([[float("nan")], [1.]]), 1234)


class Toy(nn.Module):
    def __init__(self):
        super().__init__()
        self.variant = "J1"
        self.encoders = nn.ModuleDict({"shared": nn.Linear(2, 2, bias=False)})
        self.interaction = None
        self.decoders = nn.ModuleDict({t: nn.Linear(2, 1) for t in ["u", "p", "w"]})

    def losses(self):
        features = self.encoders["shared"](torch.tensor([[1., 2.], [-2., 1.]]))
        return {t: (head(features)-target).square().mean()
                for (t, head), target in zip(self.decoders.items(), [2., -2., 1.])}


@pytest.mark.parametrize("scale", [1., 128.])
def test_private_heads_preserve_mean_loss_gradients_with_scaled_backward(scale):
    torch.manual_seed(7)
    model = Toy()
    control = copy.deepcopy(model)
    sum(control.losses().values()).div(3).backward()
    optimizer = torch.optim.AdamW(model.parameters(), lr=.01)
    scaler = torch.amp.GradScaler("cpu", enabled=True, init_scale=scale)
    diag = pcgrad_backward(model.losses(), model, optimizer, scaler, seed=1234)
    assert diag["tasks"] == ["u", "p", "w"]
    for a, b in zip(model.decoders.parameters(), control.decoders.parameters()):
        torch.testing.assert_close(a.grad, b.grad)
    assert all(torch.isfinite(p.grad).all() for p in model.parameters())
    scaler.step(optimizer)
    scaler.update()
    assert scaler.get_scale() == scale


def test_projector_does_not_modify_input_and_seed_is_reproducible():
    vectors = torch.tensor([[1., -2., 3.], [-3., 2., 1.], [0., -1., -2.]])
    original = vectors.clone()
    a, _ = project_conflicting_gradients(vectors, 1234)
    b, _ = project_conflicting_gradients(vectors, 1234)
    torch.testing.assert_close(a, b, rtol=0, atol=0)
    torch.testing.assert_close(vectors, original, rtol=0, atol=0)

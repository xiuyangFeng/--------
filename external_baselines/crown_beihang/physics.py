from __future__ import annotations

import torch


def fun_u_0(u_x: torch.Tensor, v_y: torch.Tensor, w_z: torch.Tensor) -> torch.Tensor:
    return u_x + v_y + w_z


def fun_r(
    u: torch.Tensor,
    v: torch.Tensor,
    w: torch.Tensor,
    u_t: torch.Tensor,
    u_x: torch.Tensor,
    u_y: torch.Tensor,
    u_z: torch.Tensor,
    f_x: torch.Tensor,
    u_xx: torch.Tensor,
    u_yy: torch.Tensor,
    u_zz: torch.Tensor,
    reynolds: float,
) -> torch.Tensor:
    return u_t + u * u_x + v * u_y + w * u_z + f_x - (u_xx + u_yy + u_zz) / reynolds


def physics_residual_loss(model: torch.nn.Module, pv: torch.Tensor, reynolds: float) -> torch.Tensor:
    """Navier-Stokes + continuity residuals w.r.t. coordinate channels."""
    x = pv[:, 0:1, :]
    y = pv[:, 1:2, :]
    z = pv[:, 2:3, :]
    x = x.detach().clone().requires_grad_(True)
    y = y.detach().clone().requires_grad_(True)
    z = z.detach().clone().requires_grad_(True)

    if model.input_dim > 3:
        extra = pv[:, 3:, :]
        logits = model(torch.cat([x, y, z, extra], dim=1))
    else:
        logits = model(torch.cat([x, y, z], dim=1))

    u = logits[:, 0:1, :]
    v = logits[:, 1:2, :]
    w = logits[:, 2:3, :]
    f = logits[:, 3:4, :]

    u_t = torch.zeros_like(u)
    v_t = torch.zeros_like(v)
    w_t = torch.zeros_like(w)

    u_x = torch.autograd.grad(u, x, grad_outputs=torch.ones_like(x), retain_graph=True, create_graph=True)[0]
    u_y = torch.autograd.grad(u, y, grad_outputs=torch.ones_like(y), retain_graph=True, create_graph=True)[0]
    u_z = torch.autograd.grad(u, z, grad_outputs=torch.ones_like(z), retain_graph=True, create_graph=True)[0]
    v_x = torch.autograd.grad(v, x, grad_outputs=torch.ones_like(x), retain_graph=True, create_graph=True)[0]
    v_y = torch.autograd.grad(v, y, grad_outputs=torch.ones_like(y), retain_graph=True, create_graph=True)[0]
    v_z = torch.autograd.grad(v, z, grad_outputs=torch.ones_like(z), retain_graph=True, create_graph=True)[0]
    w_x = torch.autograd.grad(w, x, grad_outputs=torch.ones_like(x), retain_graph=True, create_graph=True)[0]
    w_y = torch.autograd.grad(w, y, grad_outputs=torch.ones_like(y), retain_graph=True, create_graph=True)[0]
    w_z = torch.autograd.grad(w, z, grad_outputs=torch.ones_like(z), retain_graph=True, create_graph=True)[0]
    f_x = torch.autograd.grad(f, x, grad_outputs=torch.ones_like(x), retain_graph=True, create_graph=True)[0]
    f_y = torch.autograd.grad(f, y, grad_outputs=torch.ones_like(y), retain_graph=True, create_graph=True)[0]
    f_z = torch.autograd.grad(f, z, grad_outputs=torch.ones_like(z), retain_graph=True, create_graph=True)[0]
    u_xx = torch.autograd.grad(u_x, x, grad_outputs=torch.ones_like(x), retain_graph=True, create_graph=True)[0]
    u_yy = torch.autograd.grad(u_y, y, grad_outputs=torch.ones_like(y), retain_graph=True, create_graph=True)[0]
    u_zz = torch.autograd.grad(u_z, z, grad_outputs=torch.ones_like(z), retain_graph=True, create_graph=True)[0]
    v_xx = torch.autograd.grad(v_x, x, grad_outputs=torch.ones_like(x), retain_graph=True, create_graph=True)[0]
    v_yy = torch.autograd.grad(v_y, y, grad_outputs=torch.ones_like(y), retain_graph=True, create_graph=True)[0]
    v_zz = torch.autograd.grad(v_z, z, grad_outputs=torch.ones_like(z), retain_graph=True, create_graph=True)[0]
    w_xx = torch.autograd.grad(w_x, x, grad_outputs=torch.ones_like(x), retain_graph=True, create_graph=True)[0]
    w_yy = torch.autograd.grad(w_y, y, grad_outputs=torch.ones_like(y), retain_graph=True, create_graph=True)[0]
    w_zz = torch.autograd.grad(w_z, z, grad_outputs=torch.ones_like(z), retain_graph=True, create_graph=True)[0]

    r1 = fun_r(u, v, w, u_t, u_x, u_y, u_z, f_x, u_xx, u_yy, u_zz, reynolds)
    r2 = fun_r(u, v, w, v_t, v_x, v_y, v_z, f_y, v_xx, v_yy, v_zz, reynolds)
    r3 = fun_r(u, v, w, w_t, w_x, w_y, w_z, f_z, w_xx, w_yy, w_zz, reynolds)
    r4 = fun_u_0(u_x, v_y, w_z)

    loss_pde = (r1.square() + r2.square() + r3.square()).mean()
    loss_continuity = r4.square().mean()
    return loss_pde + loss_continuity, loss_pde, loss_continuity


def wall_no_slip_loss(model: torch.nn.Module, pv_wall: torch.Tensor) -> torch.Tensor:
    if pv_wall.numel() == 0:
        return torch.tensor(0.0, device=pv_wall.device)
    pred = model(pv_wall)
    u, v, w = pred[:, 0], pred[:, 1], pred[:, 2]
    return (u.square() + v.square() + w.square()).mean()

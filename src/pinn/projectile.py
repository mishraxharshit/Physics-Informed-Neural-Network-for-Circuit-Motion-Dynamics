"""Projectile with linear drag, solved by a PINN with NO training data at all.

The network maps t -> (x, y). Loss = ODE residuals (x'' + k x', y'' + g + k y')
+ initial conditions on position and velocity. Analytical solution is used only for evaluation.
Time is normalised by T_END so that t = T_END * s.
"""
from __future__ import annotations

import numpy as np
import torch

from .models import MLP, grad
from .physics import ProjectileParams, projectile_analytical


def train(p: ProjectileParams = ProjectileParams(), t_end: float = 3.0, epochs: int = 6000,
          lr: float = 2e-3, seed: int = 0, n_colloc: int = 256):
    torch.manual_seed(seed)
    s = torch.linspace(0, 1, n_colloc).reshape(-1, 1).requires_grad_(True)
    s0 = torch.zeros(1, 1, requires_grad=True)
    model = MLP(1, 2, hidden=48, depth=4)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    sched = torch.optim.lr_scheduler.StepLR(opt, step_size=2000, gamma=0.3)
    history = []
    for ep in range(epochs):
        opt.zero_grad()
        out = model(s)
        x, y = out[:, :1], out[:, 1:]
        dx, dy = grad(x, s) / t_end, grad(y, s) / t_end            # d/dt = (1/T_END) d/ds
        ddx, ddy = grad(dx, s) / t_end, grad(dy, s) / t_end
        res_x = ddx + p.k * dx
        res_y = ddy + p.g + p.k * dy
        # Scale residuals so both terms have comparable magnitude
        l_phys = torch.mean(res_x ** 2) / p.vx0 ** 2 + torch.mean(res_y ** 2) / (p.g ** 2)

        o0 = model(s0)
        x0, y0 = o0[:, :1], o0[:, 1:]
        vx0, vy0 = grad(x0, s0) / t_end, grad(y0, s0) / t_end
        l_ic = (x0 ** 2).mean() + (y0 ** 2).mean() \
            + ((vx0 - p.vx0) ** 2).mean() / p.vx0 ** 2 + ((vy0 - p.vy0) ** 2).mean() / p.vy0 ** 2
        loss = l_phys + 10.0 * l_ic
        loss.backward()
        opt.step()
        sched.step()
        if ep % 200 == 0 or ep == epochs - 1:
            history.append((ep, loss.item(), l_phys.item(), l_ic.item()))
    return model, history, t_end


def evaluate(model, p: ProjectileParams, t_end: float, n: int = 400):
    t = np.linspace(0, t_end, n)
    with torch.no_grad():
        out = model(torch.tensor(t / t_end, dtype=torch.float32).reshape(-1, 1)).numpy()
    x_true, y_true = projectile_analytical(t, p)
    x_p, y_p = out[:, 0], out[:, 1]
    rel = lambda a, b: float(np.linalg.norm(a - b) / np.linalg.norm(b))
    return {
        "rel_l2_x": rel(x_p, x_true),
        "rel_l2_y": rel(y_p, y_true),
        "max_abs_error_x_m": float(np.max(np.abs(x_p - x_true))),
        "max_abs_error_y_m": float(np.max(np.abs(y_p - y_true))),
    }, t, (x_p, y_p), (x_true, y_true)

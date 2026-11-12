"""RC charging circuit: PINN vs a purely data-driven network on sparse, noisy data.

Time is normalised: s = t / T_END, V is normalised by Vs, so the ODE becomes
    dv/ds = (T_END / tau) * (1 - v),   v(0) = 0.
Both networks see the SAME 8 noisy measurements taken only in the first 40% of the window.
The question is how well each one extrapolates to the unseen remainder.
"""
from __future__ import annotations

import numpy as np
import torch

from .models import MLP, grad
from .physics import RCParams, rc_analytical

T_END_IN_TAU = 5.0  # simulate five time constants


def make_data(p: RCParams, n_points: int, noise_std: float, seed: int):
    rng = np.random.default_rng(seed)
    t_end = T_END_IN_TAU * p.tau
    t = np.sort(rng.uniform(0, 0.4 * t_end, n_points))  # only early part observed
    v = rc_analytical(t, p) + rng.normal(0, noise_std, n_points)
    return t, v, t_end


def _tensor(a, grad_flag=False):
    return torch.tensor(np.asarray(a), dtype=torch.float32).reshape(-1, 1).requires_grad_(grad_flag)


def train(mode: str, p: RCParams = RCParams(), n_points: int = 8, noise_std: float = 0.05,
          epochs: int = 4000, lr: float = 1e-3, seed: int = 0, n_colloc: int = 200,
          w_phys: float = 1.0, w_ic: float = 1.0):
    """mode: 'pinn' (data + physics) or 'data_only' (data loss only)."""
    assert mode in {"pinn", "data_only"}
    torch.manual_seed(seed)
    t_obs, v_obs, t_end = make_data(p, n_points, noise_std, seed)
    s_obs = _tensor(t_obs / t_end)
    v_obs_n = _tensor(v_obs / p.Vs)
    s_col = _tensor(np.linspace(0, 1, n_colloc), True)  # collocation points over the FULL window
    s0 = _tensor([0.0])
    ratio = t_end / p.tau

    model = MLP(1, 1, hidden=32, depth=3)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    history = []
    for ep in range(epochs):
        opt.zero_grad()
        l_data = torch.mean((model(s_obs) - v_obs_n) ** 2)
        if mode == "pinn":
            v = model(s_col)
            residual = grad(v, s_col) - ratio * (1.0 - v)
            l_phys = torch.mean(residual ** 2)
            l_ic = torch.mean(model(s0) ** 2)
            loss = l_data + w_phys * l_phys + w_ic * l_ic
        else:
            l_phys = l_ic = torch.tensor(0.0)
            loss = l_data
        loss.backward()
        opt.step()
        if ep % 100 == 0 or ep == epochs - 1:
            history.append((ep, loss.item(), l_data.item(), l_phys.item(), l_ic.item()))
    return model, history, (t_obs, v_obs), t_end


def evaluate(model, p: RCParams, t_end: float, n: int = 500):
    t = np.linspace(0, t_end, n)
    with torch.no_grad():
        pred = model(_tensor(t / t_end)).numpy().ravel() * p.Vs
    true = rc_analytical(t, p)
    observed_region = t <= 0.4 * t_end
    rel_l2 = lambda m: float(np.linalg.norm(pred[m] - true[m]) / np.linalg.norm(true[m]))
    return {
        "rel_l2_full": rel_l2(np.ones_like(t, bool)),
        "rel_l2_interpolation": rel_l2(observed_region),
        "rel_l2_extrapolation": rel_l2(~observed_region),
        "max_abs_error_V": float(np.max(np.abs(pred - true))),
    }, t, pred, true

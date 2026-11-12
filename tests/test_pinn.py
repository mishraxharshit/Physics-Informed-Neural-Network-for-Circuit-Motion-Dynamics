import sys
from pathlib import Path

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).parents[1] / "src"))
from pinn import projectile, rc_circuit  # noqa: E402
from pinn.models import MLP, grad  # noqa: E402
from pinn.physics import ProjectileParams, RCParams, projectile_analytical, rc_analytical  # noqa: E402


def test_rc_analytical_boundary_values():
    p = RCParams()
    assert rc_analytical(np.array([0.0]), p)[0] == pytest.approx(0.0)
    assert rc_analytical(np.array([p.tau]), p)[0] == pytest.approx(p.Vs * (1 - np.exp(-1)))
    assert rc_analytical(np.array([50 * p.tau]), p)[0] == pytest.approx(p.Vs)


def test_rc_analytical_satisfies_ode():
    p = RCParams()
    t = np.linspace(0.1, 4, 50)
    h = 1e-6
    dv = (rc_analytical(t + h, p) - rc_analytical(t - h, p)) / (2 * h)
    assert np.allclose(dv, (p.Vs - rc_analytical(t, p)) / p.tau, atol=1e-5)


def test_projectile_analytical_initial_conditions_and_ode():
    p = ProjectileParams()
    x0, y0 = projectile_analytical(np.array([0.0]), p)
    assert x0[0] == pytest.approx(0.0) and y0[0] == pytest.approx(0.0)
    t, h = np.linspace(0.2, 2.5, 40), 1e-4
    x = lambda tt: projectile_analytical(tt, p)[0]
    y = lambda tt: projectile_analytical(tt, p)[1]
    dx, dy = (x(t + h) - x(t - h)) / (2 * h), (y(t + h) - y(t - h)) / (2 * h)
    ddx = (x(t + h) - 2 * x(t) + x(t - h)) / h ** 2
    ddy = (y(t + h) - 2 * y(t) + y(t - h)) / h ** 2
    assert np.allclose(ddx + p.k * dx, 0, atol=1e-3)
    assert np.allclose(ddy + p.g + p.k * dy, 0, atol=1e-3)
    # initial velocities
    assert (x(np.array([h])) - x(np.array([-h])))[0] / (2 * h) == pytest.approx(p.vx0, rel=1e-4)


def test_autograd_derivative_matches_known_function():
    t = torch.linspace(0, 1, 20).reshape(-1, 1).requires_grad_(True)
    y = t ** 3
    assert torch.allclose(grad(y, t), 3 * t ** 2, atol=1e-6)
    assert torch.allclose(grad(grad(y, t), t), 6 * t, atol=1e-5)


def test_mlp_shapes():
    assert MLP(1, 2, hidden=8, depth=2)(torch.zeros(5, 1)).shape == (5, 2)


def test_rc_pinn_beats_data_only_when_extrapolating():
    kw = dict(epochs=1500, seed=0)
    pinn_model, _, _, t_end = rc_circuit.train("pinn", **kw)
    data_model, _, _, _ = rc_circuit.train("data_only", **kw)
    p = RCParams()
    m_pinn = rc_circuit.evaluate(pinn_model, p, t_end)[0]
    m_data = rc_circuit.evaluate(data_model, p, t_end)[0]
    assert m_pinn["rel_l2_extrapolation"] < m_data["rel_l2_extrapolation"]
    assert m_pinn["rel_l2_extrapolation"] < 0.05


def test_projectile_pinn_smoke_trains_and_reduces_loss():
    model, hist, t_end = projectile.train(epochs=600, seed=0)
    assert hist[-1][1] < hist[0][1] * 0.1
    metrics, *_ = projectile.evaluate(model, ProjectileParams(), t_end)
    assert metrics["rel_l2_x"] < 0.5

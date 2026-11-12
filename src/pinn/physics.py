"""Closed-form (analytical) solutions used as ground truth.

RC charging:      dV/dt = (Vs - V) / (R*C),  V(0) = 0
    V(t) = Vs * (1 - exp(-t / (R*C)))

Projectile with linear drag (per unit mass, coefficient k):
    x'' = -k x',        y'' = -g - k y'
    x(t) = (vx0 / k) * (1 - exp(-k t))
    y(t) = (vy0 + g/k) / k * (1 - exp(-k t)) - (g / k) * t
"""
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class RCParams:
    Vs: float = 5.0      # source voltage [V]
    R: float = 1_000.0   # resistance [ohm]
    C: float = 1e-3      # capacitance [F]

    @property
    def tau(self) -> float:
        return self.R * self.C


@dataclass(frozen=True)
class ProjectileParams:
    g: float = 9.81      # gravity [m/s^2]
    k: float = 0.5       # linear drag coefficient / mass [1/s]
    vx0: float = 10.0    # initial horizontal speed [m/s]
    vy0: float = 15.0    # initial vertical speed [m/s]


def rc_analytical(t: np.ndarray, p: RCParams = RCParams()) -> np.ndarray:
    return p.Vs * (1.0 - np.exp(-t / p.tau))


def projectile_analytical(t: np.ndarray, p: ProjectileParams = ProjectileParams()):
    x = (p.vx0 / p.k) * (1.0 - np.exp(-p.k * t))
    y = ((p.vy0 + p.g / p.k) / p.k) * (1.0 - np.exp(-p.k * t)) - (p.g / p.k) * t
    return x, y

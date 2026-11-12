"""Reproduce every number and figure in the README:  python run_experiments.py"""
import json
import sys
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).parent / "src"))
from pinn import projectile, rc_circuit  # noqa: E402
from pinn.physics import ProjectileParams, RCParams  # noqa: E402

OUT = Path(__file__).parent / "results"
OUT.mkdir(exist_ok=True)
SEEDS = [0, 1, 2, 3, 4]


def run_rc():
    p = RCParams()
    per_seed = {"pinn": [], "data_only": []}
    first = {}
    for seed in SEEDS:
        for mode in per_seed:
            model, hist, obs, t_end = rc_circuit.train(mode, p, seed=seed)
            metrics, t, pred, true = rc_circuit.evaluate(model, p, t_end)
            per_seed[mode].append(metrics)
            if seed == SEEDS[0]:
                first[mode] = (t, pred, true, obs, t_end, hist)
        print(f"[RC] seed {seed} done", flush=True)

    summary = {}
    for mode, rows in per_seed.items():
        summary[mode] = {k: {"mean": float(np.mean([r[k] for r in rows])),
                             "std": float(np.std([r[k] for r in rows]))} for k in rows[0]}

    t, _, true, (t_obs, v_obs), t_end, _ = first["pinn"]
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot(t, true, "k-", lw=2, label="Analytical  V = Vs(1 - e^(-t/RC))")
    ax.plot(t, first["data_only"][1], "r--", lw=2, label="Data-only network")
    ax.plot(t, first["pinn"][1], "g-.", lw=2, label="PINN (data + physics)")
    ax.scatter(t_obs, v_obs, c="b", zorder=5, label="8 noisy measurements")
    ax.axvspan(0.4 * t_end, t_end, color="grey", alpha=0.12, label="unobserved region")
    ax.set(xlabel="time [s]", ylabel="capacitor voltage [V]", title="RC charging: extrapolating from sparse noisy data (seed 0)")
    ax.legend(); ax.grid(alpha=0.3); fig.tight_layout()
    fig.savefig(OUT / "rc_pinn_vs_data_only.png", dpi=150); plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 4))
    for mode, c in [("pinn", "g"), ("data_only", "r")]:
        h = np.array(first[mode][5])
        ax.semilogy(h[:, 0], h[:, 1], c, label=f"{mode} total loss")
    ax.set(xlabel="epoch", ylabel="loss", title="RC training loss (seed 0)"); ax.legend(); ax.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(OUT / "rc_loss_curves.png", dpi=150); plt.close(fig)
    return summary, per_seed


def run_projectile():
    p = ProjectileParams()
    model, hist, t_end = projectile.train(p)
    metrics, t, (xp, yp), (xt, yt) = projectile.evaluate(model, p, t_end)

    fig, axs = plt.subplots(1, 2, figsize=(11, 4.2))
    axs[0].plot(xt, yt, "k-", lw=2, label="Analytical")
    axs[0].plot(xp, yp, "g--", lw=2, label="PINN (no training data)")
    axs[0].set(xlabel="x [m]", ylabel="y [m]", title="Trajectory with linear drag"); axs[0].legend(); axs[0].grid(alpha=0.3)
    axs[1].semilogy(t, np.abs(xp - xt) + 1e-12, label="|x error|")
    axs[1].semilogy(t, np.abs(yp - yt) + 1e-12, label="|y error|")
    axs[1].set(xlabel="time [s]", ylabel="absolute error [m]", title="PINN error vs time"); axs[1].legend(); axs[1].grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(OUT / "projectile_pinn.png", dpi=150); plt.close(fig)

    h = np.array(hist)
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.semilogy(h[:, 0], h[:, 1], label="total"); ax.semilogy(h[:, 0], h[:, 2], label="physics residual")
    ax.semilogy(h[:, 0], h[:, 3], label="initial conditions")
    ax.set(xlabel="epoch", ylabel="loss", title="Projectile PINN training loss"); ax.legend(); ax.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(OUT / "projectile_loss_curves.png", dpi=150); plt.close(fig)
    return metrics


if __name__ == "__main__":
    t0 = time.time()
    rc_summary, rc_raw = run_rc()
    proj = run_projectile()
    results = {"rc_circuit": {"summary_mean_std_over_5_seeds": rc_summary, "per_seed": rc_raw},
               "projectile": proj, "runtime_seconds": round(time.time() - t0, 1)}
    (OUT / "results.json").write_text(json.dumps(results, indent=2))
    print(json.dumps({"rc": rc_summary, "projectile": proj, "runtime_s": results["runtime_seconds"]}, indent=2))

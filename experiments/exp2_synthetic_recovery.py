"""Experiment 2 - Parameter recovery on synthetic data: PINN vs classical least squares.

True beta = 0.3, gamma = 0.1. Noise levels 0%, 5%, 10%.
The classical fit is cheap, so it is repeated over several noise draws (mean +/- std);
the PINN is run once per noise level (seed 0) because each run takes minutes.

Outputs: results/tables/exp2_pinn.csv, exp2_classical.csv, exp2_summary.csv
         results/figures/fig2_fits.png, fig3_pinn_training.png
"""
import numpy as np
import pandas as pd

from src import config as C
from src.classical import fit_synthetic
from src.pinn import predict_I, train_pinn
from src.plotting import AQUA, BLUE, INK_2, ORANGE, SERIES, plt, save
from src.sir import rk4, synthetic_data


def rel_err(est, true):
    return 100 * abs(est - true) / true


def trajectory_rmse(beta, gamma, y0, I_true):
    """RMSE between the true I(t) and an RK4 re-simulation using the estimated parameters."""
    t, Y = rk4(beta, gamma, y0, C.T_DAYS, C.H_DATA)
    step = int(round(1 / C.H_DATA))
    return float(np.sqrt(np.mean((Y[::step, 1] - I_true) ** 2)))


def main():
    C.TABLES.mkdir(parents=True, exist_ok=True)
    pinn_rows, cls_rows, pinn_runs = [], [], {}

    for noise in C.NOISE_LEVELS:
        # classical: several noise draws
        for seed in (C.NOISE_SEEDS if noise > 0 else [C.SEED]):
            t, I_obs, I_true, y0 = synthetic_data(noise=noise, seed=seed)
            f = fit_synthetic(t, I_obs, y0)
            cls_rows.append({"noise": noise, "seed": seed, "beta": f["beta"], "gamma": f["gamma"],
                             "beta_err_pct": rel_err(f["beta"], C.TRUE_BETA),
                             "gamma_err_pct": rel_err(f["gamma"], C.TRUE_GAMMA),
                             "traj_rmse": trajectory_rmse(f["beta"], f["gamma"], y0, I_true),
                             "seconds": f["seconds"], "iterations": f["nfev"]})

        # PINN: one run per noise level
        t, I_obs, I_true, y0 = synthetic_data(noise=noise, seed=C.SEED)
        print(f"  PINN, noise {noise:.0%} ...")
        r = train_pinn(t, I_obs, y0, verbose=True)
        pinn_runs[noise] = (t, I_obs, I_true, y0, r)
        pinn_rows.append({"noise": noise, "beta": r["beta"], "gamma": r["gamma"],
                          "beta_err_pct": rel_err(r["beta"], C.TRUE_BETA),
                          "gamma_err_pct": rel_err(r["gamma"], C.TRUE_GAMMA),
                          "traj_rmse": trajectory_rmse(r["beta"], r["gamma"], y0, I_true),
                          "seconds": r["seconds"], "iterations": r["epochs"],
                          "converged_by_tol": r["converged"]})
        np.savetxt(C.TABLES / f"exp2_pinn_history_noise{int(noise*100):02d}.csv", r["history"],
                   delimiter=",", header="epoch,loss,loss_data,loss_ode,loss_ic,beta,gamma",
                   comments="")

    pinn = pd.DataFrame(pinn_rows)
    cls = pd.DataFrame(cls_rows)
    pinn.to_csv(C.TABLES / "exp2_pinn.csv", index=False)
    cls.to_csv(C.TABLES / "exp2_classical.csv", index=False)

    # summary table: classical mean over noise draws vs the single PINN run (seed 0)
    agg = cls.groupby("noise").agg(beta=("beta", "mean"), beta_err_pct=("beta_err_pct", "mean"),
                                   beta_err_sd=("beta_err_pct", "std"), gamma=("gamma", "mean"),
                                   gamma_err_pct=("gamma_err_pct", "mean"),
                                   gamma_err_sd=("gamma_err_pct", "std"),
                                   traj_rmse=("traj_rmse", "mean"), seconds=("seconds", "mean"),
                                   iterations=("iterations", "mean")).reset_index()
    agg.insert(0, "method", "Classical (RK4 + least squares)")
    p = pinn.drop(columns="converged_by_tol").copy()
    p.insert(0, "method", "PINN")
    summary = pd.concat([agg, p], ignore_index=True)
    summary.to_csv(C.TABLES / "exp2_summary.csv", index=False)

    # Figure 2: fitted curves at each noise level
    fig, axes = plt.subplots(1, len(C.NOISE_LEVELS), figsize=(12, 3.6), sharey=True)
    t_fine = np.linspace(0, C.T_DAYS, 400)
    for ax, noise in zip(axes, C.NOISE_LEVELS):
        t, I_obs, I_true, y0, r = pinn_runs[noise]
        c = cls[(cls.noise == noise) & (cls.seed == C.SEED)].iloc[0]
        tc, Yc = rk4(c.beta, c.gamma, y0, C.T_DAYS, C.H_DATA)
        ax.plot(t, I_obs, "o", color=INK_2, ms=3, alpha=0.6, label="observed data")
        ax.plot(tc, Yc[:, 1], color=ORANGE, label="classical fit")
        ax.plot(t_fine, predict_I(r, t_fine), "--", color=BLUE, label="PINN")
        ax.set_title(f"noise {noise:.0%}")
        ax.set_xlabel("day")
    axes[0].set_ylabel("infected fraction I(t)")
    axes[0].legend()
    save(fig, "fig2_fits.png")

    # Figure 3: PINN training (loss components + parameter convergence) at 5% noise
    t, I_obs, I_true, y0, r = pinn_runs[0.05]
    h = r["history"]
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.8))
    ax = axes[0]
    ax.semilogy(h[:, 0], h[:, 2], color=BLUE, label="data loss")
    ax.semilogy(h[:, 0], h[:, 3], color=ORANGE, label="ODE residual loss")
    ax.semilogy(h[:, 0], h[:, 1], color=AQUA, label="total loss")
    ax.set_xlabel("epoch")
    ax.set_ylabel("loss")
    ax.set_title("(a) PINN loss (5% noise)")
    ax.legend()
    ax = axes[1]
    ax.plot(h[:, 0], h[:, 5], color=BLUE, label="beta (true 0.3)")
    ax.plot(h[:, 0], h[:, 6], color=ORANGE, label="gamma (true 0.1)")
    ax.axhline(C.TRUE_BETA, color=BLUE, lw=1, ls=":")
    ax.axhline(C.TRUE_GAMMA, color=ORANGE, lw=1, ls=":")
    ax.set_xlabel("epoch")
    ax.set_ylabel("estimate (1/day)")
    ax.set_title("(b) Parameter estimates during training")
    ax.legend()
    save(fig, "fig3_pinn_training.png")

    print("Experiment 2: synthetic recovery")
    print(summary.to_string(index=False, float_format=lambda x: f"{x:.4g}"))
    return summary


if __name__ == "__main__":
    main()

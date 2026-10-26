"""Experiment 1 - RK4 step size study.

(a) Global error of RK4 vs step size h -> observed order of accuracy (theory: 4).
(b) Effect of the RK4 step size used *inside* the classical fit on the recovered parameters.

Outputs: results/tables/exp1_rk4_convergence.csv, exp1_step_size_in_fit.csv
         results/figures/fig1_rk4_convergence.png
"""
import time

import numpy as np
import pandas as pd

from src import config as C
from src.classical import fit_synthetic
from src.plotting import BLUE, INK_2, ORANGE, plt, save
from src.sir import convergence_study, rk4, initial_state, synthetic_data


def main():
    C.TABLES.mkdir(parents=True, exist_ok=True)

    # (a) convergence order
    errors, orders, drift = convergence_study()
    run_times = []
    for h in C.H_LIST:
        start = time.perf_counter()
        rk4(C.TRUE_BETA, C.TRUE_GAMMA, initial_state(), C.T_DAYS, h)
        run_times.append(time.perf_counter() - start)
    conv = pd.DataFrame({"h": C.H_LIST, "steps": [int(C.T_DAYS / h) for h in C.H_LIST],
                         "global_error": errors, "observed_order": [np.nan] + list(orders),
                         "max_mass_drift": drift, "seconds": run_times})
    conv.to_csv(C.TABLES / "exp1_rk4_convergence.csv", index=False)

    # (b) step size inside the classical fit (clean synthetic data)
    t, I_obs, _, y0 = synthetic_data(noise=0.0)
    rows = []
    for h in C.H_FIT_LIST:
        f = fit_synthetic(t, I_obs, y0, h=h)
        rows.append({"h_fit": h, "beta": f["beta"], "gamma": f["gamma"],
                     "beta_rel_err_pct": 100 * abs(f["beta"] - C.TRUE_BETA) / C.TRUE_BETA,
                     "gamma_rel_err_pct": 100 * abs(f["gamma"] - C.TRUE_GAMMA) / C.TRUE_GAMMA,
                     "seconds": f["seconds"], "nfev": f["nfev"]})
    fitstep = pd.DataFrame(rows)
    fitstep.to_csv(C.TABLES / "exp1_step_size_in_fit.csv", index=False)

    # figure: log-log error vs h with a slope-4 reference line
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.8))
    ax = axes[0]
    h = np.array(C.H_LIST)
    ax.loglog(h, errors, "o-", color=BLUE, label="RK4 global error")
    ax.loglog(h, errors[-1] * (h / h[-1]) ** 4, "--", color=INK_2, lw=1.2, label="slope 4 (theory)")
    ax.set_xlabel("step size h (days)")
    ax.set_ylabel("error at t = 100 days")
    ax.set_title("(a) RK4 converges with order 4")
    ax.legend()
    ax = axes[1]
    ax.loglog(fitstep.h_fit, fitstep.beta_rel_err_pct.clip(lower=1e-8), "o-", color=BLUE, label="beta")
    ax.loglog(fitstep.h_fit, fitstep.gamma_rel_err_pct.clip(lower=1e-8), "s-", color=ORANGE, label="gamma")
    ax.set_xlabel("RK4 step size inside the fit (days)")
    ax.set_ylabel("parameter error (%)")
    ax.set_title("(b) Solver step size limits fit accuracy")
    ax.legend()
    save(fig, "fig1_rk4_convergence.png")

    print("Experiment 1: RK4 convergence")
    print(conv.to_string(index=False, float_format=lambda x: f"{x:.3e}"))
    print(fitstep.to_string(index=False, float_format=lambda x: f"{x:.4g}"))
    return conv, fitstep


if __name__ == "__main__":
    main()

"""Experiment 3 - Stability of the PINN with respect to the physics-loss weight w_ode.

Same 5%-noise data, same network and seed; only w_ode changes. Shows how sensitive the
PINN is to this hyperparameter (classical least squares has no equivalent knob).

Outputs: results/tables/exp3_ode_weight_sweep.csv, results/figures/fig4_ode_weight_sweep.png
"""
import numpy as np
import pandas as pd

from src import config as C
from src.pinn import train_pinn
from src.plotting import BLUE, ORANGE, SERIES, plt, save
from src.sir import synthetic_data


def main():
    C.TABLES.mkdir(parents=True, exist_ok=True)
    t, I_obs, _, y0 = synthetic_data(noise=C.SWEEP_NOISE, seed=C.SEED)
    rows, histories = [], {}
    for w in C.W_ODE_SWEEP:
        print(f"  PINN, w_ode = {w} ...")
        r = train_pinn(t, I_obs, y0, w_ode=w)
        histories[w] = r["history"]
        rows.append({"w_ode": w, "beta": r["beta"], "gamma": r["gamma"],
                     "beta_err_pct": 100 * abs(r["beta"] - C.TRUE_BETA) / C.TRUE_BETA,
                     "gamma_err_pct": 100 * abs(r["gamma"] - C.TRUE_GAMMA) / C.TRUE_GAMMA,
                     "final_data_loss": r["history"][-1, 2], "final_ode_loss": r["history"][-1, 3],
                     "epochs": r["epochs"], "converged_by_tol": r["converged"],
                     "seconds": r["seconds"]})
    df = pd.DataFrame(rows)
    df.to_csv(C.TABLES / "exp3_ode_weight_sweep.csv", index=False)

    fig, axes = plt.subplots(1, 2, figsize=(10, 3.8))
    ax = axes[0]
    x = np.arange(len(df))
    ax.bar(x - 0.2, df.beta_err_pct, 0.38, color=BLUE, label="beta error")
    ax.bar(x + 0.2, df.gamma_err_pct, 0.38, color=ORANGE, label="gamma error")
    ax.set_xticks(x, [f"{w:g}" for w in df.w_ode])
    ax.set_yscale("log")
    ax.set_xlabel("physics-loss weight w_ode")
    ax.set_ylabel("relative error (%)")
    ax.set_title("(a) Final parameter error")
    ax.legend()
    ax = axes[1]
    for color, w in zip(SERIES, C.W_ODE_SWEEP):
        h = histories[w]
        ax.plot(h[:, 0], h[:, 5], color=color, label=f"w_ode = {w:g}")
    ax.axhline(C.TRUE_BETA, color="#52514e", lw=1, ls=":", label="true beta")
    ax.set_xlabel("epoch")
    ax.set_ylabel("beta estimate (1/day)")
    ax.set_title("(b) beta during training")
    ax.legend(fontsize=8)
    save(fig, "fig4_ode_weight_sweep.png")

    print("Experiment 3: w_ode sweep")
    print(df.to_string(index=False, float_format=lambda x: f"{x:.4g}"))
    return df


if __name__ == "__main__":
    main()

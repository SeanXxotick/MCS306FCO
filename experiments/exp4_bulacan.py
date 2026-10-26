"""Experiment 4 - Real data: dengue outbreak seasons in Bulacan (OpenDengue, 1994-2010).

(a) Classical SIR fit to each outbreak season (May-December), giving beta, gamma, R0.
(b) Residual-bootstrap 95% intervals for the 2010 season (error estimation).
(c) PINN on the 2010 season with N_eff and I0 fixed at the classical values, then a
    consistency check: re-simulate the PINN's beta, gamma with RK4 and compare.

Outputs: results/tables/exp4_seasons.csv, exp4_bootstrap_2010.csv, exp4_pinn_2010.csv
         results/figures/fig5_bulacan_overview.png, fig6_bulacan_seasons.png, fig7_bulacan_pinn.png
"""
import numpy as np
import pandas as pd

from src import config as C
from src.classical import bootstrap_real, fit_real, model_incidence, month_edges
from src.data import load_bulacan, load_season
from src.pinn import train_pinn_incidence
from src.plotting import AQUA, BLUE, INK_2, ORANGE, plt, save

MONTHS = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]


def main():
    C.TABLES.mkdir(parents=True, exist_ok=True)
    df = load_bulacan()

    # overview figure of the whole series
    fig, ax = plt.subplots(figsize=(10, 3.2))
    dates = pd.to_datetime(df.calendar_start_date)
    ax.plot(dates, df.dengue_total, color=BLUE, lw=1.5)
    ax.set_ylabel("reported cases / month")
    ax.set_title("Monthly dengue cases, Bulacan, 1994-2010 (OpenDengue)")
    save(fig, "fig5_bulacan_overview.png")

    # (a) per-season classical fits
    rows, fits = [], {}
    for year in C.SEASONS:
        months, cases = load_season(year)
        f = fit_real(cases)
        fits[year] = (months, cases, f)
        rows.append({"season": year, "total_cases": int(cases.sum()), "peak_cases": int(cases.max()),
                     "beta": f["beta"], "gamma": f["gamma"], "R0": f["R0"],
                     "infectious_days": 1 / f["gamma"], "N_eff": f["N_eff"], "I0": f["I0"],
                     "R2": f["r2"], "seconds": f["seconds"], "nfev": f["nfev"]})
        print(f"  season {year}: R0={f['R0']:.2f}  R2={f['r2']:.3f}")
    seasons = pd.DataFrame(rows)
    seasons.to_csv(C.TABLES / "exp4_seasons.csv", index=False)

    fig, axes = plt.subplots(2, 3, figsize=(12, 6), sharex=True)
    for ax, year in zip(axes.ravel(), C.SEASONS):
        months, cases, f = fits[year]
        ax.bar(months, cases, color=BLUE, alpha=0.35, width=0.8, label="reported")
        ax.plot(months, f["fitted"], "o-", color=ORANGE, label="SIR fit")
        ax.set_title(f"{year}  (R0 = {f['R0']:.2f}, R² = {f['r2']:.2f})")
        ax.set_xticks(months, [MONTHS[m - 1] for m in months], fontsize=8)
    axes[0, 0].legend()
    for ax in axes[:, 0]:
        ax.set_ylabel("cases / month")
    save(fig, "fig6_bulacan_seasons.png")

    # (b) bootstrap for 2010
    months, cases, f2010 = fits[2010]
    draws = bootstrap_real(cases, f2010)
    lo, hi = np.percentile(draws, [2.5, 97.5], axis=0)
    boot = pd.DataFrame({"parameter": ["beta", "gamma", "R0", "N_eff"],
                         "estimate": [f2010["beta"], f2010["gamma"], f2010["R0"], f2010["N_eff"]],
                         "ci95_low": lo, "ci95_high": hi, "bootstrap_sd": draws.std(axis=0)})
    boot.to_csv(C.TABLES / "exp4_bootstrap_2010.csv", index=False)

    # (c) PINN on 2010, N_eff and I0 fixed at the classical estimates
    print("  PINN on Bulacan 2010 ...")
    edges = month_edges(len(cases))
    r = train_pinn_incidence(cases, edges, f2010["p"][2], f2010["p"][3], verbose=True)
    resim = model_incidence([r["beta"], r["gamma"], f2010["p"][2], f2010["p"][3]], len(cases))

    def r2(pred):
        return 1 - np.sum((pred - cases) ** 2) / np.sum((cases - cases.mean()) ** 2)

    pinn_tab = pd.DataFrame([
        {"method": "Classical (RK4 + least squares)", "beta": f2010["beta"], "gamma": f2010["gamma"],
         "R0": f2010["R0"], "R2_own_curve": f2010["r2"], "R2_rk4_resimulated": f2010["r2"],
         "seconds": f2010["seconds"]},
        {"method": "PINN (N_eff, I0 fixed)", "beta": r["beta"], "gamma": r["gamma"],
         "R0": r["beta"] / r["gamma"], "R2_own_curve": r2(r["incidence"]),
         "R2_rk4_resimulated": r2(resim), "seconds": r["seconds"]},
    ])
    pinn_tab.to_csv(C.TABLES / "exp4_pinn_2010.csv", index=False)

    fig, ax = plt.subplots(figsize=(7, 3.8))
    ax.bar(months, cases, color=INK_2, alpha=0.25, width=0.8, label="reported cases")
    ax.plot(months, f2010["fitted"], "o-", color=ORANGE, label="classical fit")
    ax.plot(months, r["incidence"], "s--", color=BLUE, label="PINN network output")
    ax.plot(months, resim, "^:", color=AQUA, label="RK4 with PINN's beta, gamma")
    ax.set_xticks(months, [MONTHS[m - 1] for m in months])
    ax.set_ylabel("cases / month")
    ax.set_title("Bulacan 2010: does the PINN's physics agree with its own curve?")
    ax.legend(fontsize=8)
    save(fig, "fig7_bulacan_pinn.png")

    print("Experiment 4: Bulacan")
    print(seasons.to_string(index=False, float_format=lambda x: f"{x:.4g}"))
    print(boot.to_string(index=False, float_format=lambda x: f"{x:.4g}"))
    print(pinn_tab.to_string(index=False, float_format=lambda x: f"{x:.4g}"))
    return seasons, boot, pinn_tab


if __name__ == "__main__":
    main()

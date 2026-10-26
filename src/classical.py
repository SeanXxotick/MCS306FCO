"""Classical baseline: nonlinear least squares on the RK4 solution.

For candidate parameters p, solve the SIR model with RK4, compare with the data, and let
scipy.optimize.least_squares (trust-region reflective) adjust p to minimise the squared
residuals. This is the standard "solve-then-fit" approach the PINN is compared against.
"""
import time

import numpy as np
from scipy.optimize import least_squares

from . import config as C
from .sir import rk4


# ---------------------------------------------------------------- synthetic data: observe I(t)
def fit_synthetic(t_obs, I_obs, y0, h=C.H_FIT, p0=(C.PINN["beta0"], C.PINN["gamma0"])):
    """Estimate (beta, gamma) from daily observations of I(t)."""
    t_end = float(t_obs[-1])
    stride = int(round((t_obs[1] - t_obs[0]) / h))
    if stride < 1 or not np.isclose(stride * h, t_obs[1] - t_obs[0]):
        # step does not divide the observation spacing: interpolate the RK4 grid instead
        stride = None

    def residuals(p):
        t, Y = rk4(p[0], p[1], y0, t_end, h)
        I_model = Y[::stride, 1] if stride else np.interp(t_obs, t, Y[:, 1])
        return (I_model - I_obs) / I_obs.max()

    start = time.perf_counter()
    res = least_squares(residuals, p0, bounds=([1e-4, 1e-4], [5, 5]),
                        ftol=C.LSQ_TOL, xtol=C.LSQ_TOL, gtol=C.LSQ_TOL)
    return dict(beta=res.x[0], gamma=res.x[1], seconds=time.perf_counter() - start,
                nfev=res.nfev, cost=res.cost)


# ---------------------------------------------------------------- real data: monthly new cases
def month_edges(n_months):
    return np.arange(n_months + 1) * C.DAYS_PER_MONTH


def model_incidence(p, n_months, h=C.H_REAL):
    """Monthly new cases predicted by SIR with p = (beta, gamma, log10 N_eff, log10 I0).

    New infections in a month = N_eff * (drop in S over that month).
    """
    beta, gamma, log_N, log_I0 = p
    I0 = 10 ** log_I0
    edges = month_edges(n_months)
    t, Y = rk4(beta, gamma, [1 - I0, I0, 0.0], edges[-1], h)
    S_edges = np.interp(edges, t, Y[:, 0])
    return 10 ** log_N * -np.diff(S_edges)


REAL_BOUNDS = ([0.01, 0.01, 2.0, -8.0], [3.0, 2.0, 7.0, -1.0])
REAL_STARTS = [(b, g) for b in (0.3, 0.5, 0.8) for g in (0.1, 0.2)]


def fit_real(cases, starts=REAL_STARTS):
    """Fit (beta, gamma, N_eff, I0) to monthly case counts; best of several starting points.

    Multiple starts guard against local minima (the problem is non-convex).
    """
    n = len(cases)
    scale = cases.max()
    best, total_nfev = None, 0
    start = time.perf_counter()
    for b0, g0 in starts:
        res = least_squares(lambda p: (model_incidence(p, n) - cases) / scale,
                            [b0, g0, 4.0, -4.0], bounds=REAL_BOUNDS)
        total_nfev += res.nfev
        if best is None or res.cost < best.cost:
            best = res
    fitted = model_incidence(best.x, n)
    r2 = 1 - np.sum((fitted - cases) ** 2) / np.sum((cases - cases.mean()) ** 2)
    return dict(p=best.x, beta=best.x[0], gamma=best.x[1], N_eff=10 ** best.x[2],
                I0=10 ** best.x[3], R0=best.x[0] / best.x[1], r2=r2, fitted=fitted,
                seconds=time.perf_counter() - start, nfev=total_nfev)


def bootstrap_real(cases, fit, B=C.BOOTSTRAP_B, seed=C.SEED):
    """Residual bootstrap: resample the fit's residuals, refit, collect the estimates.

    Gives an error estimate (95% percentile interval) for each parameter.
    """
    rng = np.random.default_rng(seed)
    resid = cases - fit["fitted"]
    n, scale = len(cases), cases.max()
    draws = []
    for _ in range(B):
        fake = np.clip(fit["fitted"] + rng.choice(resid, size=n, replace=True), 0, None)
        res = least_squares(lambda p: (model_incidence(p, n) - fake) / scale, fit["p"],
                            bounds=REAL_BOUNDS)
        draws.append([res.x[0], res.x[1], res.x[0] / res.x[1], 10 ** res.x[2]])
    return np.array(draws)  # columns: beta, gamma, R0, N_eff

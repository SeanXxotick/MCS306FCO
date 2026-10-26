"""SIR epidemic model and the classical 4th-order Runge-Kutta (RK4) solver.

All compartments are *fractions* of the population, so S + I + R = 1 at all times:

    dS/dt = -beta * S * I
    dI/dt =  beta * S * I - gamma * I
    dR/dt =  gamma * I
"""
import numpy as np

from . import config as C


def sir_rhs(y, beta, gamma):
    """Right-hand side f(y) of the SIR system for state y = [S, I, R]."""
    S, I, _ = y
    infection = beta * S * I
    recovery = gamma * I
    return np.array([-infection, infection - recovery, recovery])


def rk4(beta, gamma, y0, t_end, h):
    """Integrate the SIR system from t = 0 to t_end with fixed step size h.

    One RK4 step:  y_{k+1} = y_k + h/6 (k1 + 2 k2 + 2 k3 + k4)
    Local error is O(h^5), so the global error at t_end is O(h^4).

    Returns the time grid t (n+1,) and the solution Y (n+1, 3).
    """
    n = int(round(t_end / h))
    t = np.linspace(0.0, n * h, n + 1)
    Y = np.empty((n + 1, 3))
    y = np.asarray(y0, dtype=float)
    Y[0] = y
    for k in range(n):
        k1 = sir_rhs(y, beta, gamma)
        k2 = sir_rhs(y + 0.5 * h * k1, beta, gamma)
        k3 = sir_rhs(y + 0.5 * h * k2, beta, gamma)
        k4 = sir_rhs(y + h * k3, beta, gamma)
        y = y + h / 6.0 * (k1 + 2 * k2 + 2 * k3 + k4)
        Y[k + 1] = y
    return t, Y


def initial_state(i0=C.I0):
    return [1.0 - i0, i0, 0.0]


def synthetic_data(noise=0.0, seed=C.SEED, beta=C.TRUE_BETA, gamma=C.TRUE_GAMMA,
                   i0=C.I0, t_end=C.T_DAYS):
    """Daily observations of the infected fraction I(t), with optional multiplicative noise.

    The "truth" is an RK4 solution with a small step (H_DATA). Noise model:
        I_obs = I_true * (1 + noise * eps),  eps ~ N(0, 1)
    so 5% noise means each observation is off by about 5% of its value.
    """
    y0 = initial_state(i0)
    t, Y = rk4(beta, gamma, y0, t_end, C.H_DATA)
    step = int(round(1.0 / C.H_DATA))           # keep one point per day
    t_obs, I_true = t[::step], Y[::step, 1].copy()
    I_obs = I_true.copy()
    if noise > 0:
        rng = np.random.default_rng(seed)
        I_obs = np.clip(I_obs * (1 + noise * rng.standard_normal(I_obs.shape)), 0, None)
    return t_obs, I_obs, I_true, y0


def convergence_study(h_list=C.H_LIST, h_ref=C.H_REF):
    """Global error of RK4 at t = T_DAYS for several step sizes, against a tiny-step reference.

    Error = |S_h(T) - S_ref(T)| + |I_h(T) - I_ref(T)|.
    Observed order p between consecutive halvings of h:  p = log2(e_h / e_{h/2}).
    """
    y0 = initial_state()
    _, Y_ref = rk4(C.TRUE_BETA, C.TRUE_GAMMA, y0, C.T_DAYS, h_ref)
    errors, mass_drift = [], []
    for h in h_list:
        _, Y = rk4(C.TRUE_BETA, C.TRUE_GAMMA, y0, C.T_DAYS, h)
        errors.append(abs(Y[-1, 0] - Y_ref[-1, 0]) + abs(Y[-1, 1] - Y_ref[-1, 1]))
        mass_drift.append(np.max(np.abs(Y.sum(axis=1) - 1.0)))   # S+I+R should stay exactly 1
    orders = [np.log(errors[i] / errors[i + 1]) / np.log(h_list[i] / h_list[i + 1])
              for i in range(len(h_list) - 1)]
    return np.array(errors), np.array(orders), np.array(mass_drift)

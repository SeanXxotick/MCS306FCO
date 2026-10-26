"""Physics-Informed Neural Network (PINN) for the SIR inverse problem.

A small neural network N(tau) -> (S, I, R) approximates the solution on rescaled time
tau = t / T in [0, 1]. The unknown rates beta and gamma are trainable parameters learned
together with the network weights by minimising

    L = L_data + w_ode * L_ode + L_ic

    L_data : mismatch between the network's I(t) and the observed data
    L_ode  : mean squared residual of the SIR equations at collocation points
             (computed with automatic differentiation, no time-stepping needed)
    L_ic   : mismatch with the known initial condition

beta and gamma are stored as log-values so they always stay positive.
"""
import time

import numpy as np
import torch
import torch.nn as nn

from . import config as C

torch.set_default_dtype(torch.float64)
torch.set_num_threads(C.TORCH_THREADS)


class SIRNet(nn.Module):
    """Fully connected tanh network: 1 input (time) -> 3 outputs (S, I/s_I, R) in (0, 1)."""

    def __init__(self, width=32, depth=3):
        super().__init__()
        layers, d_in = [], 1
        for _ in range(depth):
            layers += [nn.Linear(d_in, width), nn.Tanh()]
            d_in = width
        layers.append(nn.Linear(d_in, 3))
        self.body = nn.Sequential(*layers)

    def forward(self, tau):
        return torch.sigmoid(self.body(tau))


def _d_dtau(u, tau):
    """Derivative du/dtau by automatic differentiation."""
    return torch.autograd.grad(u, tau, torch.ones_like(u), create_graph=True)[0]


def ode_residual_loss(net, tau_c, beta, gamma, T, s_I):
    """Mean squared SIR residual at the collocation points.

    The network outputs I scaled by s_I (so all outputs are O(1)); the chain rule
    d/dtau = T d/dt converts the time derivatives.
    """
    out = net(tau_c)
    S, I_scaled, R = out[:, :1], out[:, 1:2], out[:, 2:3]
    I = I_scaled * s_I
    r_S = _d_dtau(S, tau_c) - T * (-beta * S * I)
    r_I = _d_dtau(I_scaled, tau_c) - T * (beta * S * I - gamma * I) / s_I
    r_R = _d_dtau(R, tau_c) - T * (gamma * I)
    return (r_S ** 2).mean() + (r_I ** 2).mean() + (r_R ** 2).mean()


def train_pinn(t_obs, I_obs, y0, w_ode=None, seed=C.SEED, settings=None, verbose=False):
    """Learn beta, gamma from observations of the infected fraction I(t).

    Stopping rule: every `check_every` epochs compare beta and gamma with their values at
    the previous check; stop when both relative changes are below `tol`, or at `max_epochs`.

    Returns a dict with the estimates, iteration count, wall time, training history and
    the trained network (for plotting its predicted I(t)).
    """
    p = dict(C.PINN, **(settings or {}))
    w_ode = p["w_ode"] if w_ode is None else w_ode
    torch.manual_seed(seed)

    T = float(t_obs[-1])
    s_I = float(np.max(I_obs))  # scale so the network's I output is O(1)
    net = SIRNet(p["width"], p["depth"])
    log_beta = nn.Parameter(torch.tensor(np.log(p["beta0"])))
    log_gamma = nn.Parameter(torch.tensor(np.log(p["gamma0"])))
    opt = torch.optim.Adam(list(net.parameters()) + [log_beta, log_gamma], lr=p["lr"])

    tau_data = torch.tensor(t_obs / T).reshape(-1, 1)
    I_data = torch.tensor(I_obs / s_I).reshape(-1, 1)
    tau_c = torch.linspace(0, 1, p["n_col"]).reshape(-1, 1).requires_grad_(True)
    tau_0 = torch.zeros(1, 1)
    y0_scaled = torch.tensor([[y0[0], y0[1] / s_I, y0[2]]])

    history = []  # (epoch, total, data, ode, ic, beta, gamma)
    last_check = (p["beta0"], p["gamma0"])
    stop_epoch, converged = p["max_epochs"], False
    start = time.perf_counter()
    for epoch in range(1, p["max_epochs"] + 1):
        opt.zero_grad()
        beta, gamma = torch.exp(log_beta), torch.exp(log_gamma)
        l_ode = ode_residual_loss(net, tau_c, beta, gamma, T, s_I)
        l_data = ((net(tau_data)[:, 1:2] - I_data) ** 2).mean()
        l_ic = ((net(tau_0) - y0_scaled) ** 2).mean()
        loss = l_data + w_ode * l_ode + l_ic
        loss.backward()
        opt.step()

        if epoch == 1 or epoch % 100 == 0:
            history.append((epoch, loss.item(), l_data.item(), l_ode.item(), l_ic.item(),
                            beta.item(), gamma.item()))
            if verbose and epoch % 2000 == 0:
                print(f"    epoch {epoch:6d}  loss={loss.item():.3e}  beta={beta.item():.4f}  "
                      f"gamma={gamma.item():.4f}")
        if epoch % p["check_every"] == 0:
            b, g = beta.item(), gamma.item()
            rel_change = max(abs(b - last_check[0]) / b, abs(g - last_check[1]) / g)
            last_check = (b, g)
            if rel_change < p["tol"]:
                stop_epoch, converged = epoch, True
                break
    elapsed = time.perf_counter() - start

    return dict(beta=torch.exp(log_beta).item(), gamma=torch.exp(log_gamma).item(),
                epochs=stop_epoch, converged=converged, seconds=elapsed,
                history=np.array(history), net=net, T=T, s_I=s_I)


def predict_I(result, t):
    """Infected fraction predicted by a trained PINN at times t (days)."""
    with torch.no_grad():
        tau = torch.tensor(np.asarray(t) / result["T"]).reshape(-1, 1)
        return result["net"](tau)[:, 1].numpy() * result["s_I"]


def train_pinn_incidence(cases, month_edges, log_N, log_I0, w_ode=None, epochs=C.REAL_PINN_EPOCHS,
                         seed=C.SEED, verbose=False):
    """PINN for real monthly case counts (used on Bulacan data).

    Monthly new cases are modelled as N_eff * (S(start of month) - S(end of month)).
    N_eff (effective population) and I0 are held fixed at the classical estimates, so
    the PINN only has to learn beta and gamma -- the easiest possible version of the task.
    """
    p = C.PINN
    w_ode = p["w_ode"] if w_ode is None else w_ode
    torch.manual_seed(seed)
    T = float(month_edges[-1])
    N_eff, I0 = 10 ** log_N, 10 ** log_I0
    s_I = 0.2  # I stays below ~20% of the effective population in the fitted model
    net = SIRNet(p["width"], p["depth"])
    log_beta = nn.Parameter(torch.tensor(np.log(p["beta0"])))
    log_gamma = nn.Parameter(torch.tensor(np.log(p["gamma0"])))
    opt = torch.optim.Adam(list(net.parameters()) + [log_beta, log_gamma], lr=p["lr"])

    tau_c = torch.linspace(0, 1, p["n_col"]).reshape(-1, 1).requires_grad_(True)
    tau_edges = torch.tensor(month_edges / T).reshape(-1, 1)
    scale = float(np.max(cases))
    y_obs = torch.tensor(cases / scale)
    y0_scaled = torch.tensor([[1 - I0, I0 / s_I, 0.0]])

    history = []
    start = time.perf_counter()
    for epoch in range(1, epochs + 1):
        opt.zero_grad()
        beta, gamma = torch.exp(log_beta), torch.exp(log_gamma)
        l_ode = ode_residual_loss(net, tau_c, beta, gamma, T, s_I)
        S_edges = net(tau_edges)[:, 0]
        incidence = N_eff * -(S_edges[1:] - S_edges[:-1])
        l_data = ((incidence / scale - y_obs) ** 2).mean()
        l_ic = ((net(torch.zeros(1, 1)) - y0_scaled) ** 2).mean()
        loss = l_data + w_ode * l_ode + 100 * l_ic  # I0 is tiny, so the IC must be enforced hard
        loss.backward()
        opt.step()
        if epoch == 1 or epoch % 100 == 0:
            history.append((epoch, loss.item(), l_data.item(), l_ode.item(), l_ic.item(),
                            beta.item(), gamma.item()))
            if verbose and epoch % 5000 == 0:
                print(f"    epoch {epoch:6d}  data={l_data.item():.2e}  ode={l_ode.item():.2e}  "
                      f"beta={beta.item():.3f}  gamma={gamma.item():.3f}")
    elapsed = time.perf_counter() - start
    return dict(beta=torch.exp(log_beta).item(), gamma=torch.exp(log_gamma).item(),
                incidence=incidence.detach().numpy(), seconds=elapsed, epochs=epochs,
                history=np.array(history))

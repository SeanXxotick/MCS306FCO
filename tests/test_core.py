"""Fast checks of the numerical core. Run with:  python -m pytest -q"""
import numpy as np
import pytest

from src import config as C
from src.classical import fit_synthetic, model_incidence
from src.sir import convergence_study, initial_state, rk4, synthetic_data


def test_rk4_conserves_population():
    _, Y = rk4(0.3, 0.1, initial_state(), 100, 0.5)
    assert np.allclose(Y.sum(axis=1), 1.0, atol=1e-12)


def test_rk4_order_is_four():
    _, orders, _ = convergence_study(h_list=[2, 1, 0.5, 0.25])
    assert np.all(np.abs(orders - 4) < 0.1)


def test_rk4_no_epidemic_when_r0_below_one():
    _, Y = rk4(0.05, 0.1, initial_state(), 100, 0.5)   # R0 = 0.5
    assert np.all(np.diff(Y[:, 1]) < 0)                  # infections only decline


def test_synthetic_noise_is_reproducible():
    a = synthetic_data(noise=0.05, seed=3)[1]
    b = synthetic_data(noise=0.05, seed=3)[1]
    assert np.array_equal(a, b)


def test_classical_recovers_clean_parameters():
    t, I_obs, _, y0 = synthetic_data(noise=0.0)
    f = fit_synthetic(t, I_obs, y0)
    assert f["beta"] == pytest.approx(C.TRUE_BETA, rel=1e-4)
    assert f["gamma"] == pytest.approx(C.TRUE_GAMMA, rel=1e-4)


def test_incidence_is_nonnegative_and_bounded():
    inc = model_incidence([0.1, 0.03, 3.6, -3.0], 8)
    assert np.all(inc >= 0) and inc.sum() <= 10 ** 3.6


def test_pinn_smoke():
    """A very short PINN run must execute and reduce the loss."""
    torch = pytest.importorskip("torch")
    from src.pinn import train_pinn
    t, I_obs, _, y0 = synthetic_data(noise=0.0)
    r = train_pinn(t, I_obs, y0, settings=dict(max_epochs=300, check_every=10_000))
    h = r["history"]
    assert h[-1, 1] < h[0, 1]
    assert r["beta"] > 0 and r["gamma"] > 0

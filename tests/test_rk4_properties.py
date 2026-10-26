"""Extra RK4 checks: the recovered compartment can only grow."""

import numpy as np

from src.sir import initial_state, rk4


def test_recovered_never_decreases():
    # dR/dt = gamma * I >= 0, so R must never go down
    _, Y = rk4(0.3, 0.1, initial_state(), 100, 0.5)
    assert np.all(np.diff(Y[:, 2]) >= 0)

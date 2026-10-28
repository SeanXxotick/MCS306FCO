"""The classical fit should stay accurate when the data are noisy."""
import pytest

from src import config as C
from src.classical import fit_synthetic
from src.sir import synthetic_data


def test_classical_recovers_noisy_parameters():
    t, I_obs, _, y0 = synthetic_data(noise=0.05, seed=1)
    f = fit_synthetic(t, I_obs, y0)
    assert f["beta"] == pytest.approx(C.TRUE_BETA, rel=0.02)    # within 2%
    assert f["gamma"] == pytest.approx(C.TRUE_GAMMA, rel=0.02)

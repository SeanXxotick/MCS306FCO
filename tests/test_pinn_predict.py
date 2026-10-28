"""The trained PINN returns one non-negative prediction per time point."""
import numpy as np
import pytest

from src.sir import synthetic_data


def test_pinn_prediction_shape_and_range():
    pytest.importorskip("torch")
    from src.pinn import predict_I, train_pinn
    t, I_obs, _, y0 = synthetic_data(noise=0.0)
    r = train_pinn(t, I_obs, y0, settings=dict(max_epochs=50, check_every=10_000))
    pred = predict_I(r, t)
    assert pred.shape == t.shape
    assert np.all(pred >= 0)

import numpy as np
import pytest

from energy_forecast.metrics import mae, mape, mase, rmse, summarize


def test_basic_metrics():
    y = np.array([100.0, 200.0, 300.0])
    p = np.array([110.0, 190.0, 300.0])
    assert mae(y, p) == pytest.approx(20 / 3)
    assert rmse(y, p) == pytest.approx(np.sqrt(200 / 3))
    assert mape(y, p) == pytest.approx((10 + 5 + 0) / 3)


def test_mase_relative_to_naive():
    y = np.array([10.0, 20.0])
    assert mase(y, y + 1, y + 2) == pytest.approx(0.5)


def test_mape_zero_raises():
    with pytest.raises(ValueError):
        mape(np.array([0.0, 1.0]), np.array([1.0, 1.0]))


def test_summarize_keys():
    y = np.array([1.0, 2.0])
    out = summarize(y, y, y + 1)
    assert out["MAE_MW"] == 0 and out["MASE_vs_weekly_naive"] == 0

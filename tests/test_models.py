import numpy as np
import pandas as pd
import pytest

from energy_forecast.backtest import daily_origins, rolling_backtest
from energy_forecast.metrics import mae
from energy_forecast.models import GradientBoostingForecaster, SarimaForecaster, SeasonalNaive


def _split(df, origin):
    hist = df.loc[df.index < origin]
    fut = df.loc[origin : origin + pd.Timedelta(hours=23)]
    return hist, fut


def test_seasonal_naive_repeats_last_week(small_df):
    origin = pd.Timestamp("2023-12-04")
    hist, fut = _split(small_df, origin)
    pred = SeasonalNaive(168).fit(hist).predict(fut)
    np.testing.assert_array_equal(pred, hist["load_mw"].to_numpy()[-168:-144])


def test_gbm_cannot_see_future_target(small_df):
    origin = pd.Timestamp("2023-12-04")
    hist, fut = _split(small_df, origin)
    model = GradientBoostingForecaster(max_iter=50).fit(hist)
    a = model.predict(fut)
    b = model.predict(fut.assign(load_mw=1e9))  # garbage target must be ignored
    np.testing.assert_allclose(a, b)


def test_gbm_rejects_long_horizon(small_df):
    hist = small_df.iloc[:-48]
    model = GradientBoostingForecaster(max_iter=20).fit(hist)
    with pytest.raises(ValueError):
        model.predict(small_df.iloc[-48:])


def test_gbm_beats_daily_naive(small_df):
    origins = daily_origins(small_df.index, 5)
    res = rolling_backtest(
        small_df,
        [lambda: SeasonalNaive(24), lambda: GradientBoostingForecaster(max_iter=150)],
        origins,
        refit_every=5,
    )
    by_model = {m: mae(g.y_true, g.y_pred) for m, g in res.groupby("model")}
    assert by_model["gradient_boosting"] < by_model["seasonal_naive_24h"]
    assert len(res) == 2 * 5 * 24


def test_sarima_produces_finite_forecast(small_df):
    origin = pd.Timestamp("2023-12-04")
    hist, fut = _split(small_df, origin)
    model = SarimaForecaster(order=(1, 0, 0), seasonal_order=(0, 1, 1, 24), window_hours=24 * 14)
    pred = model.fit(hist).predict(fut)
    assert pred.shape == (24,)
    assert np.isfinite(pred).all()
    assert mae(fut["load_mw"], pred) < 0.2 * fut["load_mw"].mean()


def test_daily_origins_requires_enough_days(small_df):
    with pytest.raises(ValueError):
        daily_origins(small_df.index, 10_000)
    origins = daily_origins(small_df.index, 3)
    assert all(o.hour == 0 for o in origins)

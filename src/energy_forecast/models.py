"""Forecasters sharing one interface: ``fit(history)`` then ``predict(origin, horizon)``.

``history`` is the DataFrame of everything known *before* the origin. Models that
need exogenous information for the future (temperature, holidays) receive the
full frame via ``future`` but must only read non-target columns from it.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from typing import Protocol

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from statsmodels.tsa.statespace.sarimax import SARIMAX

from energy_forecast.features import HORIZON, build_features


class Forecaster(Protocol):
    name: str

    def fit(self, history: pd.DataFrame) -> Forecaster: ...

    def predict(self, future: pd.DataFrame) -> np.ndarray: ...


@dataclass
class SeasonalNaive:
    """Repeat the value from ``season`` hours earlier (24 = yesterday, 168 = last week)."""

    season: int = 168
    name: str = ""
    _last: np.ndarray = field(default_factory=lambda: np.empty(0), repr=False)

    def __post_init__(self) -> None:
        self.name = self.name or f"seasonal_naive_{self.season}h"

    def fit(self, history: pd.DataFrame) -> SeasonalNaive:
        self._last = history["load_mw"].to_numpy()[-self.season :]
        return self

    def predict(self, future: pd.DataFrame) -> np.ndarray:
        reps = int(np.ceil(len(future) / self.season))
        return np.tile(self._last, reps)[: len(future)]


@dataclass
class SarimaForecaster:
    """SARIMA(p,d,q)(P,D,Q,24) with temperature/holiday exogenous regressors.

    Weekly seasonality is handled with a weekend/holiday dummy and the daily
    season with s=24; fitting on a sliding window keeps it fast enough to refit
    at every backtest origin.
    """

    order: tuple[int, int, int] = (2, 0, 1)
    seasonal_order: tuple[int, int, int, int] = (1, 1, 1, 24)
    window_hours: int = 24 * 28
    name: str = "sarima"
    _res: object | None = field(default=None, repr=False)

    @staticmethod
    def _exog(df: pd.DataFrame) -> np.ndarray:
        off = ((df.index.dayofweek >= 5) | df["is_holiday"].to_numpy()).astype(float)
        sat = (df.index.dayofweek == 5).astype(float)
        hdd = np.clip(15.0 - df["temp_c"].to_numpy(), 0, None)
        return np.column_stack([off, sat, hdd])

    def fit(self, history: pd.DataFrame) -> SarimaForecaster:
        h = history.iloc[-self.window_hours :]
        model = SARIMAX(
            h["load_mw"].to_numpy(),
            exog=self._exog(h),
            order=self.order,
            seasonal_order=self.seasonal_order,
            enforce_stationarity=False,
            enforce_invertibility=False,
        )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            self._res = model.fit(disp=False, maxiter=50)
        return self

    def update_history(self, history: pd.DataFrame) -> None:
        """Re-run the Kalman filter on the latest window with the fitted parameters."""
        assert self._res is not None, "call fit() first"
        h = history.iloc[-self.window_hours :]
        self._res = self._res.apply(h["load_mw"].to_numpy(), exog=self._exog(h))  # type: ignore[attr-defined]

    def predict(self, future: pd.DataFrame) -> np.ndarray:
        assert self._res is not None, "call fit() first"
        return np.asarray(self._res.forecast(steps=len(future), exog=self._exog(future)))


@dataclass
class GradientBoostingForecaster:
    """Direct day-ahead model: HistGradientBoosting on calendar, weather and lag features."""

    max_iter: int = 400
    learning_rate: float = 0.05
    random_state: int = 0
    name: str = "gradient_boosting"
    _model: HistGradientBoostingRegressor | None = field(default=None, repr=False)
    _full: pd.DataFrame | None = field(default=None, repr=False)

    def fit(self, history: pd.DataFrame) -> GradientBoostingForecaster:
        X = build_features(history).dropna()
        y = history.loc[X.index, "load_mw"]
        self._model = HistGradientBoostingRegressor(
            max_iter=self.max_iter,
            learning_rate=self.learning_rate,
            max_leaf_nodes=31,
            l2_regularization=1.0,
            categorical_features=[X.columns.get_loc(c) for c in ("hour", "dow")],
            random_state=self.random_state,
        )
        self._model.fit(X, y)
        self._full = history
        return self

    def update_history(self, history: pd.DataFrame) -> None:
        """Keep the trained trees but use the latest history for lag features."""
        self._full = history

    def predict(self, future: pd.DataFrame) -> np.ndarray:
        assert self._model is not None and self._full is not None, "call fit() first"
        if len(future) > HORIZON:
            raise ValueError(f"Direct model only supports horizons <= {HORIZON} h")
        # Concatenate history with future rows whose target is masked, so that
        # lag features are built from known values only.
        fut = future.copy()
        fut["load_mw"] = np.nan
        combined = pd.concat([self._full, fut])
        X = build_features(combined).loc[future.index]
        return self._model.predict(X)

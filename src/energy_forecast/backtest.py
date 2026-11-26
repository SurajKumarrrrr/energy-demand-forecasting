"""Rolling-origin (expanding window) backtest for day-ahead forecasts."""

from __future__ import annotations

from collections.abc import Callable, Iterable

import pandas as pd

from energy_forecast.features import HORIZON
from energy_forecast.models import Forecaster


def daily_origins(index: pd.DatetimeIndex, n_days: int) -> list[pd.Timestamp]:
    """The last ``n_days`` midnights for which a full 24 h horizon exists."""
    midnights = index[(index.hour == 0)]
    midnights = [m for m in midnights if m + pd.Timedelta(hours=HORIZON - 1) <= index[-1]]
    if len(midnights) < n_days:
        raise ValueError(f"Only {len(midnights)} complete days available, need {n_days}")
    return list(midnights[-n_days:])


def rolling_backtest(
    df: pd.DataFrame,
    model_factories: Iterable[Callable[[], Forecaster]],
    origins: list[pd.Timestamp],
    refit_every: int = 1,
) -> pd.DataFrame:
    """Forecast 24 h from each origin, using only data strictly before it.

    ``refit_every`` controls how often (in origins) a model is re-trained; between
    refits the model keeps its parameters but still sees the latest history for
    its lag/state inputs (naive and GBM via ``fit`` on cheap paths, SARIMA via refit).

    Returns a long DataFrame: timestamp, origin, model, y_true, y_pred.
    """
    rows: list[pd.DataFrame] = []
    for factory in model_factories:
        model = factory()
        for i, origin in enumerate(origins):
            history = df.loc[df.index < origin]
            future = df.loc[origin : origin + pd.Timedelta(hours=HORIZON - 1)]
            if i % refit_every == 0 or not hasattr(model, "update_history"):
                model.fit(history)
            else:
                model.update_history(history)  # type: ignore[attr-defined]
            pred = model.predict(future.drop(columns=["load_mw"]).assign(load_mw=float("nan")))
            rows.append(
                pd.DataFrame(
                    {
                        "timestamp": future.index,
                        "origin": origin,
                        "model": model.name,
                        "y_true": future["load_mw"].to_numpy(),
                        "y_pred": pred,
                    }
                )
            )
    return pd.concat(rows, ignore_index=True)

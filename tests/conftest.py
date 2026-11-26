import pandas as pd
import pytest

from energy_forecast.data import generate_load


@pytest.fixture(scope="session")
def small_df() -> pd.DataFrame:
    """~10 weeks of synthetic hourly load, enough for 336 h lags plus a short backtest."""
    return generate_load(start="2023-10-01", end="2023-12-10 23:00", seed=7)

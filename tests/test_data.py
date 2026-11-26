import pandas as pd

from energy_forecast.data import generate_load


def test_deterministic_with_seed():
    a = generate_load("2023-01-01", "2023-01-31 23:00", seed=1)
    b = generate_load("2023-01-01", "2023-01-31 23:00", seed=1)
    pd.testing.assert_frame_equal(a, b)


def test_shape_and_ranges(small_df):
    assert small_df.index.freq is None or small_df.index.freqstr == "h"
    assert small_df.index.is_monotonic_increasing
    assert not small_df.isna().any().any()
    assert small_df["load_mw"].between(25_000, 90_000).all()


def test_weekend_and_holiday_lower(small_df):
    daily = small_df["load_mw"].resample("D").mean()
    weekday = daily[daily.index.dayofweek < 5].mean()
    sunday = daily[daily.index.dayofweek == 6].mean()
    assert sunday < 0.9 * weekday
    # 3 Oct 2023 (Tag der Deutschen Einheit) was a Tuesday.
    assert small_df.loc["2023-10-03", "is_holiday"].all()
    assert daily["2023-10-03"] < daily["2023-10-10"]


def test_cold_increases_load():
    df = generate_load("2022-01-01", "2023-12-31 23:00", seed=3)
    daily = df.resample("D").mean()
    winter = daily[daily.index.month.isin([12, 1, 2])]["load_mw"].mean()
    summer = daily[daily.index.month.isin([6, 7, 8])]["load_mw"].mean()
    assert winter > summer

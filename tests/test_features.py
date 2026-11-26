import numpy as np

from energy_forecast.features import LAGS, build_features


def test_lags_are_shifted_target(small_df):
    feats = build_features(small_df)
    for lag in LAGS:
        expected = small_df["load_mw"].shift(lag)
        np.testing.assert_allclose(feats[f"lag_{lag}"].dropna(), expected.dropna())


def test_no_feature_uses_future_or_same_day_target(small_df):
    """Masking the last 24 h of the target must not change any feature for those hours."""
    feats = build_features(small_df)
    masked = small_df.copy()
    masked.iloc[-24:, masked.columns.get_loc("load_mw")] = np.nan
    feats_masked = build_features(masked)
    last = feats.index[-24:]
    np.testing.assert_allclose(
        feats.loc[last].to_numpy(dtype=float), feats_masked.loc[last].to_numpy(dtype=float)
    )


def test_calendar_columns(small_df):
    feats = build_features(small_df)
    assert set(feats["hour"].unique()) == set(range(24))
    assert feats["is_weekend"].sum() > 0

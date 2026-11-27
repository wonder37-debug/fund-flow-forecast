"""Causal feature helpers for chronologically ordered daily rows."""

from dataclasses import dataclass

import numpy as np
import pandas as pd


def causal_target_features(
    values: pd.Series, *, lags: tuple[int, ...], windows: tuple[int, ...]
) -> pd.DataFrame:
    """Calculate each row's target features from earlier rows only."""
    if any(length < 1 for length in (*lags, *windows)):
        raise ValueError("Lag and rolling window lengths must be positive")
    previous = values.shift(1)
    features = {f"lag_{lag}": values.shift(lag) for lag in lags}
    features.update(
        {f"rolling_mean_{window}": previous.rolling(window, min_periods=window).mean() for window in windows}
    )
    return pd.DataFrame(features, index=values.index)


@dataclass(frozen=True)
class Standardizer:
    mean: pd.Series
    scale: pd.Series

    def transform(self, features: pd.DataFrame) -> pd.DataFrame:
        if not features.columns.equals(self.mean.index):
            raise ValueError("Transform columns must match training feature columns")
        return (features - self.mean) / self.scale


def fit_standardizer(training_features: pd.DataFrame) -> Standardizer:
    """Use training feature statistics and unit scale for constant columns."""
    if training_features.empty or np.isinf(training_features.to_numpy(dtype=float)).any():
        raise ValueError("Standardization requires nonempty training features without infinities")
    mean = training_features.mean()
    if mean.isna().any():
        raise ValueError("Every feature must have an observed value in training")
    scale = training_features.std(ddof=0).replace(0, 1)
    return Standardizer(mean, scale)

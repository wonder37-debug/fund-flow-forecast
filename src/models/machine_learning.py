"""Fit one Stage 3 target using samples from a single outer training fold."""

from numbers import Integral

import numpy as np
import pandas as pd
from sklearn.linear_model import HuberRegressor, Ridge
from sklearn.preprocessing import StandardScaler

from src.models.baselines import TARGET_COLUMNS

_RESIDUALS = {"additive", "log_residual"}
_LOG_OBJECTIVES = {"direct_log", "log_residual"}
_OBJECTIVES = _RESIDUALS | {"direct_amount", "direct_log", "direct_relative"}
_DATE_COLUMNS = {"report_date", "origin", "horizon_day", "is_ordinary"}
_FLOAT_LIMIT = np.finfo(float).max


def _numeric(frame: pd.DataFrame, column: str) -> np.ndarray:
    try:
        return pd.to_numeric(frame[column]).to_numpy(dtype=float, na_value=np.nan)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{column} must be numeric") from exc


def _amounts(frame: pd.DataFrame, column: str, *, positive: bool = False) -> np.ndarray:
    if column not in frame:
        raise ValueError(f"Missing amount column: {column}")
    values = _numeric(frame, column)
    invalid = values <= 0 if positive else values < 0
    if not np.isfinite(values).all() or invalid.any():
        bound = "positive" if positive else "nonnegative"
        raise ValueError(f"{column} must contain finite {bound} amounts")
    return values


def _sample_dates(frame: pd.DataFrame, label: str) -> tuple[pd.Series, pd.Series]:
    dates = pd.to_datetime(frame["report_date"], errors="raise")
    origins = pd.to_datetime(frame["origin"], errors="raise")
    if any(
        series.dt.tz is not None or series.isna().any() or not series.eq(series.dt.normalize()).all()
        for series in (dates, origins)
    ):
        raise ValueError(f"{label} report_date and origin must contain calendar dates")
    days = (dates - origins).dt.days.to_numpy()
    if (days < 1).any() or not np.array_equal(days, _numeric(frame, "horizon_day")):
        raise ValueError(f"{label} horizon_day must equal the positive report_date minus origin")
    if not frame["is_ordinary"].isin([True, False]).all():
        raise ValueError(f"{label} is_ordinary must contain boolean flags")
    return dates, origins


def _feature_matrices(
    training: pd.DataFrame,
    future: pd.DataFrame,
    features: tuple[str, ...],
) -> tuple[np.ndarray, np.ndarray]:
    train_columns = []
    future_columns = []
    for feature in features:
        train_values = _numeric(training, feature)
        future_values = _numeric(future, feature)
        observed = np.isfinite(train_values)
        median = np.median(train_values[observed]) if observed.any() else 0.0
        train_columns.append(np.where(observed, train_values, median))
        future_columns.append(np.where(np.isfinite(future_values), future_values, median))
    return np.column_stack(train_columns), np.column_stack(future_columns)


def _fit_predict(
    estimator: str,
    loss: str,
    params: dict,
    features: tuple[str, ...],
    x_train: np.ndarray,
    y_train: np.ndarray,
    weights: np.ndarray,
    x_future: np.ndarray,
) -> tuple[np.ndarray, dict[str, float]]:
    if estimator == "lightgbm":
        from lightgbm import LGBMRegressor

        settings = {
            "n_estimators": 48,
            "learning_rate": 0.05,
            "max_depth": 3,
            "num_leaves": 7,
            "min_child_samples": 32,
            "reg_lambda": 1.0,
            "random_state": 42,
            "n_jobs": 1,
            "deterministic": True,
            "force_col_wise": True,
            "verbosity": -1,
            "importance_type": "gain",
        } | params
        if {"objective", "application", "app", "objective_type", "loss"}.intersection(settings):
            raise ValueError("Select the LightGBM objective through the loss setting")
        model = LGBMRegressor(objective="regression_l2" if loss == "l2" else "regression_l1", **settings)
        model.fit(pd.DataFrame(x_train, columns=list(features)), y_train, sample_weight=weights)
        predictions = model.predict(pd.DataFrame(x_future, columns=list(features)))
        importance = model.feature_importances_
    else:
        scaler = StandardScaler().fit(x_train, sample_weight=weights)
        model = Ridge(**params) if estimator == "ridge" else HuberRegressor(**({"max_iter": 500} | params))
        model.fit(scaler.transform(x_train), y_train, sample_weight=weights)
        predictions = model.predict(scaler.transform(x_future))
        importance = np.abs(model.coef_)
    if not np.isfinite(predictions).all() or not np.isfinite(importance).all():
        raise RuntimeError("The fitted estimator returned nonfinite predictions or feature importance")
    return predictions, {feature: float(value) for feature, value in zip(features, importance, strict=True)}


def forecast_target(
    training: pd.DataFrame,
    future: pd.DataFrame,
    target: str,
    features: tuple[str, ...],
    config: dict,
) -> tuple[np.ndarray, dict[str, float]]:
    """Return nonnegative amounts in future-row order and model feature importance.

    Observation gaps in numeric features use training medians. Each target
    date gets unit total training weight before applying any relative loss.
    Residuals fit only ordinary days by default, leaving special-day amounts
    at the base forecast. Shrink and clip operate on relative corrections.
    """
    if target not in TARGET_COLUMNS:
        raise ValueError(f"Unknown target: {target}")
    options = {"estimator", "objective", "loss", "base", "clip", "shrink", "ordinary_only", "lookback_days", "params"}
    unknown_options = config.keys() - options
    if unknown_options:
        raise ValueError(f"Unknown model config options: {sorted(unknown_options)}")
    estimator = config.get("estimator")
    objective = config.get("objective")
    loss = config.get("loss", "l2")
    if estimator not in {"ridge", "lightgbm", "huber"} or objective not in _OBJECTIVES:
        raise ValueError(f"Unknown estimator or objective: {estimator}, {objective}")
    if loss not in {"l1", "relative_l1", "l2"} or (estimator != "lightgbm" and loss != "l2"):
        raise ValueError(f"Unsupported {estimator} loss: {loss}")
    if loss == "relative_l1" and objective in _LOG_OBJECTIVES:
        raise ValueError("relative_l1 requires an objective with linear conversion to amount")
    params = config.get("params", {})
    if not isinstance(params, dict):
        raise ValueError("params must contain estimator keyword arguments")
    lookback = config.get("lookback_days")
    if lookback is not None and (isinstance(lookback, bool) or not isinstance(lookback, Integral) or lookback < 1):
        raise ValueError("lookback_days must be a positive integer")
    residual = objective in _RESIDUALS
    ordinary_only = config.get("ordinary_only", residual)
    if not isinstance(ordinary_only, bool):
        raise ValueError("ordinary_only must be boolean")
    clip = float(config.get("clip", 0.25))
    shrink = float(config.get("shrink", 1.0))
    if not np.isfinite(clip) or clip < 0 or not 0 <= shrink <= 1:
        raise ValueError("clip must be finite and nonnegative and shrink must be between zero and one")
    if not features or set(features).intersection({"report_date", "origin", *TARGET_COLUMNS}) or len(set(features)) != len(features):
        raise ValueError("Provide unique numeric feature names without raw dates or target labels")

    for label, frame in (("Training", training), ("Future", future)):
        missing = (_DATE_COLUMNS | set(features)) - set(frame.columns)
        if frame.empty or not frame.columns.is_unique or missing:
            raise ValueError(f"{label} samples must be nonempty with unique columns. Missing columns: {sorted(missing)}")
    if set(TARGET_COLUMNS).intersection(future.columns):
        raise ValueError("Future samples cannot contain target label columns")
    train_dates, _ = _sample_dates(training, "Training")
    _, future_origins = _sample_dates(future, "Future")
    if future_origins.nunique() != 1:
        raise ValueError("Future samples must share one outer cutoff (origin)")
    cutoff = future_origins.iloc[0]
    if train_dates.gt(cutoff).any():
        raise ValueError("Training contains target dates after the outer cutoff")

    keep = np.ones(len(training), dtype=bool)
    if lookback is not None:
        keep &= train_dates.gt(cutoff - pd.Timedelta(days=int(lookback))).to_numpy()
    adjust_future = np.ones(len(future), dtype=bool)
    if ordinary_only:
        keep &= training["is_ordinary"].to_numpy(dtype=bool)
        adjust_future = future["is_ordinary"].to_numpy(dtype=bool)
    base_name = f"{config.get('base', 'stat_route_special')}_{target}"
    if residual or ordinary_only:
        future_base = _amounts(future, base_name).copy()
        if not adjust_future.any() or (residual and (shrink == 0 or clip == 0)):
            return future_base, dict.fromkeys(features, 0.0)
    else:
        future_base = np.zeros(len(future))
    if not keep.any():
        raise ValueError("Training has no usable target dates after filtering")
    train_dates = train_dates.iloc[keep]
    training = training.iloc[keep]
    y = _amounts(training, target)
    weights = 1.0 / train_dates.map(train_dates.value_counts()).to_numpy(dtype=float)

    active_future = future.iloc[adjust_future]
    train_scale = future_scale = 1.0
    if objective == "direct_relative":
        train_scale = _amounts(training, f"scale_{target}", positive=True)
        future_scale = _amounts(active_future, f"scale_{target}", positive=True)
    if residual:
        train_base = _amounts(training, base_name)
    if loss == "relative_l1" or objective in _LOG_OBJECTIVES:
        positive = y[(~train_dates.duplicated().to_numpy()) & (y > 0)]
        median_amount = np.median(positive) if len(positive) else 1.0

    if objective in _LOG_OBJECTIVES:
        floor = max(median_amount * 1e-6, np.finfo(float).tiny)
        train_target = np.log(np.maximum(y, floor))
        if residual:
            train_target -= np.log(np.maximum(train_base, floor))
    elif residual:
        train_target = y - train_base
    else:
        train_target = y / train_scale

    if loss == "relative_l1":
        # One unit of error in ratio space changes the amount by the row's scale.
        denominator = np.maximum(y, max(median_amount / 2, np.finfo(float).tiny))
        relative_weights = train_scale / denominator
        weights *= relative_weights / np.average(relative_weights, weights=weights)

    x_train, x_future = _feature_matrices(training, active_future, features)
    estimate, importance = _fit_predict(estimator, loss, params, features, x_train, train_target, weights, x_future)
    base = future_base[adjust_future]
    with np.errstate(over="ignore", divide="ignore"):
        if residual:
            correction = np.expm1(estimate) if objective == "log_residual" else np.divide(
                estimate, base, out=np.zeros_like(estimate), where=base > 0,
            )
            result = base * (1 + np.clip(shrink * correction, -min(clip, 1), clip))
        elif objective == "direct_log":
            result = np.exp(np.minimum(estimate, np.log(_FLOAT_LIMIT)))
        else:
            result = estimate * future_scale
    future_base[adjust_future] = np.clip(result, 0, _FLOAT_LIMIT)
    return future_base, importance

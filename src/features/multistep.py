"""Build direct multi-horizon features from observations available at each origin."""

import json
from collections.abc import Iterable
from numbers import Integral

import numpy as np
import pandas as pd

from experiments.stage2 import DEFAULT_CONFIG, ROOT, build_model_factory
from src.features.calendar import build_calendar
from src.models.baselines import DATE_COLUMN, TARGET_COLUMNS


_MIN_HISTORY = 84
_DAY = pd.Timedelta(days=1)
_SCALE_FLOOR = 1.0
_STAT_MODELS = ("B3", "cal_workday", "blend_equal", "trimmed_weekday", "route_special")
_CALENDAR_FLAGS = ("is_workday", "is_holiday", "is_makeup_workday", "is_pre_holiday", "is_post_holiday")
_WEEKDAY_COLUMNS = tuple(f"weekday_{day}" for day in range(7))
_BUSINESS_COLUMNS = ("purchase_bank_amt", "transfer_amt", "observed_users", "redeem_users")
_EXTERNAL_COLUMNS = ("mfd_7daily_yield", "Interest_O_N")
_RAW_STAT_COLUMNS = tuple(f"stat_{model}_{target}" for model in _STAT_MODELS for target in TARGET_COLUMNS)
_SCALE_COLUMNS = tuple(f"scale_{target}" for target in TARGET_COLUMNS)

FEATURE_GROUPS: dict[str, tuple[str, ...]] = {
    "calendar": ("horizon_day", *_WEEKDAY_COLUMNS, *_CALENDAR_FLAGS, "is_near_month_end"),
    "lags": tuple(
        f"lag_{lag}_{target}"
        for target in TARGET_COLUMNS
        for lag in ("origin_0", "origin_7", "same_weekday_1", "same_weekday_2")
    ),
    "rolling": tuple(
        f"rolling_{window}_{target}"
        for target in TARGET_COLUMNS
        for window in ("mean_7", "median_28", "mean_56")
    ),
    "trend": tuple(
        f"trend_{window}_{target}" for target in TARGET_COLUMNS for window in ("7_to_prior_7", "28_to_prior_28")
    ),
    "statistical": tuple(f"{column}_scaled" for column in _RAW_STAT_COLUMNS),
    "business": (
        "business_bank_purchase_ratio_28", "business_transfer_redeem_ratio_28",
        "business_users_7_to_28", "business_redeem_users_7_to_28",
    ),
    "external": tuple(f"external_{column}_mean_7" for column in _EXTERNAL_COLUMNS),
}
_OUTPUT_COLUMNS = (
    "origin", DATE_COLUMN, "horizon_day", "is_ordinary", *_SCALE_COLUMNS, *_RAW_STAT_COLUMNS,
    *(column for group in FEATURE_GROUPS.values() for column in group if column != "horizon_day"),
)
_DATE_DTYPES = {"origin": "datetime64[ns]", DATE_COLUMN: "datetime64[ns]"}


def _daily_dates(values: Iterable[pd.Timestamp]) -> pd.DatetimeIndex:
    dates = pd.DatetimeIndex(pd.to_datetime(values)).as_unit("ns")
    if dates.hasnans or dates.tz is not None or not dates.equals(dates.normalize()):
        raise ValueError("History and origins must use timezone-naive calendar dates")
    if dates.has_duplicates:
        raise ValueError("History and origin dates must be unique")
    return dates


def _statistical_factories():
    candidates = json.loads((ROOT / DEFAULT_CONFIG).read_text(encoding="utf-8"))["candidates"]
    by_id = {candidate["id"]: candidate for candidate in candidates}
    blend_members = by_id["blend_equal"]["members"]
    # Reuse the already-computed calendar forecast when combining the frozen ensemble members.
    model_ids = dict.fromkeys(("B3", "cal_workday", "trimmed_weekday", *blend_members))
    return {model: build_model_factory(by_id[model], candidates) for model in model_ids}, blend_members


def _observed_features(history: pd.DataFrame, future: pd.DataFrame) -> dict:
    features = {}
    days_before = (history[DATE_COLUMN].iloc[-1].dayofweek - future["weekday"].to_numpy()) % 7
    for target in TARGET_COLUMNS:
        values = history[target].to_numpy(dtype=float)
        scale = max(float(values[-28:].mean()), _SCALE_FLOOR)
        features.update({
            f"scale_{target}": scale,
            f"lag_origin_0_{target}": values[-1] / scale,
            f"lag_origin_7_{target}": values[-8] / scale,
            f"lag_same_weekday_1_{target}": values[-1 - days_before] / scale,
            f"lag_same_weekday_2_{target}": values[-8 - days_before] / scale,
            f"rolling_mean_7_{target}": values[-7:].mean() / scale,
            f"rolling_median_28_{target}": np.median(values[-28:]) / scale,
            f"rolling_mean_56_{target}": values[-56:].mean() / scale,
            f"trend_7_to_prior_7_{target}": values[-7:].mean() / max(values[-14:-7].mean(), _SCALE_FLOOR),
            f"trend_28_to_prior_28_{target}": values[-28:].mean() / max(values[-56:-28].mean(), _SCALE_FLOOR),
        })

    business = history.tail(28).reindex(columns=_BUSINESS_COLUMNS).apply(pd.to_numeric).astype(float)
    features.update({
        "business_bank_purchase_ratio_28": business["purchase_bank_amt"].mean() / features["scale_purchase"],
        "business_transfer_redeem_ratio_28": business["transfer_amt"].mean() / features["scale_redeem"],
    })
    for feature, column in (("business_users_7_to_28", "observed_users"), ("business_redeem_users_7_to_28", "redeem_users")):
        users = business[column]
        features[feature] = users.tail(7).mean() / np.maximum(users.mean(), _SCALE_FLOOR)
    external = history.tail(7).reindex(columns=_EXTERNAL_COLUMNS).apply(pd.to_numeric).astype(float)
    features.update({f"external_{column}_mean_7": external[column].mean() for column in _EXTERNAL_COLUMNS})
    return features


def build_examples(
    daily: pd.DataFrame, origins: Iterable[pd.Timestamp], *, max_horizon: int = 31
) -> pd.DataFrame:
    """Return origin-date rows with 84 days of warmup and no joined labels.

    Pass a horizon of 30 for the September forecast. Every requested target
    date must be inside the public calendar, even when it has no observed flows.
    Raw statistical forecasts are available for reconstruction. Use the scaled
    columns in FEATURE_GROUPS as model inputs.
    """
    if isinstance(max_horizon, bool) or not isinstance(max_horizon, Integral) or not 1 <= max_horizon <= 31:
        raise ValueError("max_horizon must be an integer from 1 to 31")
    origin_dates = _daily_dates(list(origins)).sort_values()
    if not len(origin_dates):
        dtypes = {**_DATE_DTYPES, "is_ordinary": bool, **dict.fromkeys(FEATURE_GROUPS["calendar"], int)}
        return pd.DataFrame({column: pd.Series(dtype=dtypes.get(column, float)) for column in _OUTPUT_COLUMNS})

    dates = _daily_dates(daily[DATE_COLUMN])
    if not origin_dates.isin(dates).all():
        raise ValueError("Every origin must be an observed history date")
    observed = daily.assign(**{DATE_COLUMN: dates}).loc[dates <= origin_dates[-1]].sort_values(DATE_COLUMN)
    dates = pd.DatetimeIndex(observed[DATE_COLUMN])
    if not dates.equals(pd.date_range(dates[0], dates[-1], freq="D", unit="ns")):
        raise ValueError("History through the last origin must contain consecutive daily observations")
    if (origin_dates[0] - dates[0]).days + 1 < _MIN_HISTORY:
        raise ValueError(f"Every origin requires at least {_MIN_HISTORY} observed days")

    forecast_end = origin_dates[-1] + max_horizon * _DAY
    calendar = build_calendar(dates[0], forecast_end).set_index(DATE_COLUMN)
    calendar.index = calendar.index.as_unit("ns")
    factories, blend_members = _statistical_factories()
    rows = []
    for origin in origin_dates:
        history = observed.loc[observed[DATE_COLUMN].le(origin)].reset_index(drop=True)
        future = calendar.loc[origin + _DAY : origin + max_horizon * _DAY].reset_index()
        ordinary = ~future["is_holiday"] & ~future["is_makeup_workday"]
        features = _observed_features(history, future)
        history_calendar = calendar.loc[:origin, ["weekday", *_CALENDAR_FLAGS]].reset_index(drop=True)
        training = history.loc[:, [DATE_COLUMN, *TARGET_COLUMNS]].join(history_calendar)

        for target in TARGET_COLUMNS:
            predictions = {}
            for model, (factory, columns) in factories.items():
                model_at_origin = factory(target)
                model_at_origin.fit(training.loc[:, [DATE_COLUMN, target, *columns]])
                forecast = model_at_origin.predict(future.loc[:, [DATE_COLUMN, *columns]])
                if not isinstance(forecast, pd.Series) or not forecast.index.equals(future.index):
                    raise ValueError(f"Unaligned forecast from {model} for {target} at {origin.date()}")
                values = forecast.to_numpy(dtype=float)
                if not np.isfinite(values).all() or (values < 0).any():
                    raise ValueError(f"Invalid forecast from {model} for {target} at {origin.date()}")
                predictions[model] = values
            predictions["blend_equal"] = sum(predictions[model] for model in blend_members) / len(blend_members)
            predictions["route_special"] = np.where(ordinary, predictions["blend_equal"], predictions["cal_workday"])
            for model in _STAT_MODELS:
                column = f"stat_{model}_{target}"
                features[column] = predictions[model]
                features[f"{column}_scaled"] = predictions[model] / features[f"scale_{target}"]

        calendar_features = future.loc[:, list(_CALENDAR_FLAGS)].astype(int)
        weekday_features = {
            column: future["weekday"].eq(day).astype(int) for day, column in enumerate(_WEEKDAY_COLUMNS)
        }
        target_dates = future[DATE_COLUMN]
        calendar_features["is_near_month_end"] = (target_dates.dt.days_in_month - target_dates.dt.day < 3).astype(int)
        rows.append(calendar_features.assign(
            origin=origin, report_date=target_dates, horizon_day=np.arange(1, len(future) + 1),
            is_ordinary=ordinary, **weekday_features, **features,
        ))
    examples = pd.concat(rows, ignore_index=True).loc[:, list(_OUTPUT_COLUMNS)]
    return examples.astype(_DATE_DTYPES)

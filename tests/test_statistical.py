import numpy as np
import pandas as pd
import pytest

from src.evaluation import predict_targets
from src.models.baselines import BASELINES, SeasonalMeanBaseline
from src.models.statistical import LevelWeekday, RecentWeekday, WeeklyETS


def _history(dates, values):
    return pd.DataFrame({"report_date": dates, "purchase": values})


def _calendar(*dates):
    return pd.DataFrame({"report_date": pd.to_datetime(list(dates))}, index=range(10, 10 + len(dates)))


@pytest.mark.parametrize(("days", "code"), [(28, "B3"), (84, "B5")])
def test_unweighted_recent_weekday_mean_reproduces_stage1(days, code):
    dates = pd.date_range("2024-01-01", periods=100)
    daily = _history(dates, np.random.default_rng(42).uniform(1, 100, len(dates)))
    daily["redeem"] = daily["purchase"] * 0.45
    future_dates = pd.date_range(dates[-1] + pd.Timedelta(days=1), periods=30)
    expected = predict_targets(daily, future_dates, lambda target: SeasonalMeanBaseline(BASELINES[code], target))
    actual = predict_targets(daily, future_dates, lambda target: RecentWeekday(target, days))
    pd.testing.assert_frame_equal(actual, expected, check_exact=True)


def test_weekday_half_life_counts_elapsed_natural_days():
    dates = pd.to_datetime(["2024-01-01", "2024-01-22", "2024-01-29"])
    model = RecentWeekday("purchase", 35, half_life=7).fit(_history(dates, [400, 200, 100]))
    future = _calendar("2024-02-05")
    forecast = model.predict(future)
    assert forecast.iloc[0] == pytest.approx(144)
    assert forecast.index.equals(future.index)


@pytest.mark.parametrize(
    ("estimator", "expected"),
    [("median", 100), ("trimmed", 340 / 3), ("relative_median", 90)],
)
def test_robust_weekday_estimators_limit_the_effect_of_outliers(estimator, expected):
    dates = pd.date_range("2024-01-01", periods=5, freq="W-MON")
    history = _history(dates, [1, 90, 100, 150, 1000])
    forecast = RecentWeekday("purchase", 35, estimator=estimator).fit(history).predict(_calendar("2024-02-05"))
    assert forecast.iloc[0] == pytest.approx(expected)


@pytest.mark.parametrize(
    ("amounts", "expected"),
    [
        pytest.param([150, 1, 100, 1000, 90], 100, id="odd"),
        pytest.param([1000, 90, 1, 150], 120, id="even"),
        pytest.param([123], 123, id="singleton"),
        pytest.param([100, 2, 2, 10, 2], 2, id="duplicate-amounts"),
        pytest.param([0, 20, 0], 0, id="zero-amounts"),
    ],
)
def test_equally_weighted_weekday_median(amounts, expected):
    dates = pd.date_range("2024-01-01", periods=len(amounts), freq="W-MON")
    model = RecentWeekday("purchase", 35, estimator="median").fit(_history(dates, amounts))
    assert model.predict(_calendar(dates[-1] + pd.Timedelta(days=7))).iloc[0] == expected


def test_relative_weekday_median_averages_at_exact_unequal_weight_boundary():
    dates = pd.date_range("2024-01-01", periods=3, freq="W-MON")
    # The amount 2 has the same weight as both observations of 4 combined.
    model = RecentWeekday("purchase", 21, estimator="relative_median").fit(_history(dates, [4, 2, 4]))
    assert model.predict(_calendar("2024-01-22")).iloc[0] == 3


@pytest.mark.parametrize("half_life", [7, 0.001])
def test_weekday_median_applies_recency_to_sorted_amounts(half_life):
    dates = pd.date_range("2024-01-01", periods=3, freq="W-MON")
    model = RecentWeekday("purchase", 21, estimator="median", half_life=half_life)
    model.fit(_history(dates, [40, 20, 10]))
    assert model.predict(_calendar("2024-01-22")).iloc[0] == 10


def test_weekday_median_preserves_weight_difference_lost_by_float_cumulative_sum():
    dates = pd.date_range("2024-01-01", periods=55, freq="W-MON")
    # All older weights add up to 1 - 2**-54, strictly less than the newest weight of 1.
    model = RecentWeekday("purchase", 385, estimator="median", half_life=7)
    model.fit(_history(dates, [0] * 54 + [100]))
    assert model.predict(_calendar(dates[-1] + pd.Timedelta(days=7))).iloc[0] == 100


def test_relative_weekday_median_preserves_small_weight_between_equal_weights():
    dates = pd.to_datetime(["2024-01-08", "2024-01-15", "2024-01-22"])
    middle = 2.0**-499
    # Both outer amounts carry equal weight, so the middle amount decides the median.
    model = RecentWeekday("purchase", 28, estimator="relative_median", half_life=0.007)
    model.fit(_history(dates, [middle, 2.0**-500, 2.0**500]))
    assert model.predict(_calendar("2024-01-29")).iloc[0] == middle


@pytest.mark.parametrize("estimator", ["median", "relative_median"])
def test_weekday_median_preserves_smallest_positive_singleton(estimator):
    amount = np.finfo(float).smallest_subnormal
    dates = pd.to_datetime(["2024-01-01"])
    model = RecentWeekday("purchase", 7, estimator=estimator).fit(_history(dates, [amount]))
    assert model.predict(_calendar("2024-01-08")).iloc[0] == amount


@pytest.mark.parametrize(
    ("lower", "upper", "expected"),
    [
        pytest.param(float.fromhex("0x1p-1074"), float.fromhex("0x2p-1074"), float.fromhex("0x2p-1074"), id="subnormal"),
        pytest.param(1e308, 1.5e308, 1.25e308, id="large"),
    ],
)
def test_weekday_median_averages_extreme_amounts(lower, upper, expected):
    dates = pd.date_range("2024-01-01", periods=2, freq="W-MON")
    model = RecentWeekday("purchase", 14, estimator="median").fit(_history(dates, [lower, upper]))
    assert model.predict(_calendar("2024-01-15")).iloc[0] == expected


def test_relative_weekday_median_handles_exact_zeros_and_missing_weekdays():
    dates = pd.to_datetime(["2024-01-01", "2024-01-02", "2024-01-08", "2024-01-09"])
    history = _history(dates, [0, 0, 50, 0])
    model = RecentWeekday("purchase", 14, estimator="relative_median").fit(history)
    forecast = model.predict(_calendar("2024-01-15", "2024-01-16", "2024-01-17"))
    assert forecast.tolist() == [50, 0, 50]


def test_relative_weekday_median_combines_recency_and_amount_weights_without_underflow():
    dates = pd.date_range("2024-01-01", periods=3, freq="W-MON")
    history = _history(dates, [1e-200, 1e-200, 1e200])
    model = RecentWeekday("purchase", 21, estimator="relative_median", half_life=0.001).fit(history)
    assert model.predict(_calendar("2024-01-22")).iloc[0] == pytest.approx(1e200)


def test_level_weekday_learns_shape_across_weeks_and_level_from_recent_days():
    shape = np.array([1.4, 1.2, 1.1, 1.0, 0.9, 0.7, 0.6])
    levels = np.repeat([20, 50, 100, 200], 7)
    history = _history(pd.date_range("2024-01-01", periods=28), np.tile(shape, 4) * levels)
    model = LevelWeekday("purchase", level_days=7, profile_days=28).fit(history)
    forecast = model.predict(_calendar("2024-01-29", "2024-02-01", "2024-02-04"))
    assert forecast.tolist() == pytest.approx([280, 200, 120])


def test_level_weekday_trend_is_damped_and_leaves_weekday_shape_intact():
    dates = pd.date_range("2024-01-01", periods=56)
    shape = np.array([1.4, 1.2, 1.1, 1.0, 0.9, 0.7, 0.6])
    levels = np.repeat(np.arange(80, 160, 10), 7)
    history = _history(dates, np.tile(shape, 8) * levels)
    flat = LevelWeekday("purchase", level_days=28, profile_days=56).fit(history)
    trending = LevelWeekday("purchase", level_days=28, profile_days=56, trend=True).fit(history)
    future = _calendar("2024-02-26", "2024-03-04", "2024-03-11", "2024-04-22")
    baseline = flat.predict(future).to_numpy()
    forecasts = trending.predict(future).to_numpy()
    assert baseline == pytest.approx([189] * 4)
    assert 0 < forecasts[1] - forecasts[0]
    assert 0 < forecasts[2] - forecasts[1] < forecasts[1] - forecasts[0]
    assert forecasts[-1] < baseline[-1] * 1.25


def test_weekly_ets_preserves_weekly_pattern_and_sparse_forecast_alignment():
    dates = pd.date_range("2024-01-01", periods=56)
    values = np.tile([120, 100, 90, 80, 68, 50, 42], 8) + np.random.default_rng(42).normal(0, 0.5, 56)
    model = WeeklyETS("purchase", days=56).fit(_history(dates, values))
    future = _calendar("2024-02-26", "2024-03-03", "2024-03-11")
    forecast = model.predict(future)
    assert forecast.to_numpy() == pytest.approx([120, 42, 120], rel=0.10)
    assert forecast.index.equals(future.index)


def test_weekly_ets_rejects_incomplete_history():
    dates = pd.date_range("2024-01-01", periods=28).delete(10)
    with pytest.raises(ValueError, match="consecutive daily"):
        WeeklyETS("purchase").fit(_history(dates, np.full(len(dates), 100)))


@pytest.mark.parametrize(
    "model",
    [RecentWeekday("purchase", 28), LevelWeekday("purchase"), WeeklyETS("purchase")],
)
def test_statistical_models_reject_forecasting_on_or_before_cutoff(model):
    dates = pd.date_range("2024-01-01", periods=28)
    model.fit(_history(dates, np.arange(100, 128)))
    with pytest.raises(ValueError, match="after the training cutoff"):
        model.predict(_calendar("2024-01-28"))

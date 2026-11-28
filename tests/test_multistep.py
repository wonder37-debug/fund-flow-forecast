import json

import numpy as np
import pandas as pd
import pytest

from experiments.stage2 import DEFAULT_CONFIG, ROOT, build_model_factory
from src.evaluation import predict_targets
from src.features.multistep import FEATURE_GROUPS, build_examples


def _daily(end="2014-08-31"):
    dates = pd.date_range("2013-07-01", end)
    days = np.arange(len(dates), dtype=float)
    purchase = 100 + days + 15 * dates.dayofweek
    redeem = 600 - days + 30 * dates.dayofweek
    return pd.DataFrame(
        {
            "report_date": dates,
            "purchase": purchase,
            "redeem": redeem,
            "purchase_bank_amt": purchase / 3,
            "transfer_amt": redeem / 2,
            "observed_users": 1000 + days,
            "redeem_users": 10 + days**2,
            "mfd_7daily_yield": days / 100,
            "Interest_O_N": days / 200,
        }
    )


@pytest.mark.parametrize(
    ("origin", "horizon", "last_day"),
    [
        ("2014-01-31", 28, "2014-02-28"),
        ("2014-01-31", 29, "2014-03-01"),
        ("2014-03-31", 30, "2014-04-30"),
        ("2014-04-30", 31, "2014-05-31"),
        ("2014-08-31", 30, "2014-09-30"),
    ],
)
def test_forecast_rows_have_consecutive_calendar_horizons(origin, horizon, last_day):
    origin = pd.Timestamp(origin)
    examples = build_examples(_daily(), iter([origin]), max_horizon=horizon)

    assert examples["horizon_day"].tolist() == list(range(1, horizon + 1))
    assert examples["origin"].dtype == np.dtype("datetime64[ns]")
    assert examples["report_date"].dtype == np.dtype("datetime64[ns]")
    assert examples["origin"].eq(origin).all()
    assert examples["report_date"].tolist() == list(pd.date_range(origin + pd.Timedelta(days=1), last_day))


def test_future_observations_and_forged_calendar_cannot_change_origin_features():
    daily = _daily()
    origin = pd.Timestamp("2014-04-30")
    original = build_examples(daily, [origin])
    changed = daily.copy()
    changed.loc[changed["report_date"] > origin, daily.columns.difference(["report_date"])] = np.nan
    changed["is_holiday"] = False
    changed["weekday"] = -1
    changed["target_label"] = 1e6

    pd.testing.assert_frame_equal(original, build_examples(changed, [origin]))
    assert original.loc[original["report_date"].eq("2014-05-01"), "is_ordinary"].item() is False
    assert original.loc[original["report_date"].eq("2014-05-04"), "is_ordinary"].item() is False
    assert original.loc[original["report_date"].eq("2014-05-05"), "is_ordinary"].item() is True
    assert original["is_ordinary"].dtype == bool
    feature_columns = [column for group in FEATURE_GROUPS.values() for column in group]
    assert len(set(feature_columns)) == len(feature_columns)
    assert set(FEATURE_GROUPS) == {"calendar", "lags", "rolling", "trend", "statistical", "business", "external"}
    assert all(pd.api.types.is_numeric_dtype(original[column]) for column in feature_columns)
    assert {"purchase", "redeem", "target_label", "observed_users", "mfd_7daily_yield"}.isdisjoint(original)


def test_scales_and_same_weekday_lag_use_only_observed_days_through_origin():
    daily = _daily()
    origin = pd.Timestamp("2014-04-30")
    examples = build_examples(daily, [origin], max_horizon=8)
    history = daily.set_index("report_date").loc[:origin]
    scale = history["purchase"].tail(28).mean()

    assert examples["scale_purchase"].eq(scale).all()
    assert examples["lag_same_weekday_1_purchase"].iloc[0] == pytest.approx(history.loc["2014-04-24", "purchase"] / scale)
    assert examples["lag_same_weekday_1_purchase"].iloc[7] == examples["lag_same_weekday_1_purchase"].iloc[0]
    assert examples["lag_same_weekday_1_purchase"].iloc[6] == pytest.approx(history.loc[origin, "purchase"] / scale)


def test_calendar_exposes_one_hot_weekdays_to_learning_models():
    examples = build_examples(_daily(), [pd.Timestamp("2014-03-30")], max_horizon=8)
    weekday_columns = [f"weekday_{day}" for day in range(7)]

    assert "weekday" not in FEATURE_GROUPS["calendar"]
    assert set(weekday_columns).issubset(FEATURE_GROUPS["calendar"])
    np.testing.assert_array_equal(examples[weekday_columns], np.eye(7)[[0, 1, 2, 3, 4, 5, 6, 0]])


def test_statistics_equal_independent_stage_two_forecasts_at_each_origin():
    daily = _daily()
    origins = pd.to_datetime(["2014-04-30", "2014-05-02"])
    examples = build_examples(daily, origins, max_horizon=6)
    candidates = json.loads((ROOT / DEFAULT_CONFIG).read_text(encoding="utf-8"))["candidates"]
    by_id = {candidate["id"]: candidate for candidate in candidates}
    models = ("B3", "cal_workday", "blend_equal", "trimmed_weekday")

    for origin in origins:
        history = daily.loc[daily["report_date"].le(origin)]
        rows = examples.loc[examples["origin"].eq(origin)]
        dates = pd.DatetimeIndex(rows["report_date"])
        for model in models:
            factory, columns = build_model_factory(by_id[model], candidates)
            expected = predict_targets(history, dates, factory, calendar_columns=columns)
            for target in ("purchase", "redeem"):
                raw = f"stat_{model}_{target}"
                scaled = f"{raw}_scaled"
                assert raw not in FEATURE_GROUPS["statistical"]
                assert scaled in FEATURE_GROUPS["statistical"]
                np.testing.assert_allclose(rows[raw], expected[target])
                np.testing.assert_allclose(rows[scaled], expected[target].to_numpy() / rows[f"scale_{target}"])
        for target in ("purchase", "redeem"):
            expected = rows[f"stat_blend_equal_{target}"].where(rows["is_ordinary"], rows[f"stat_cal_workday_{target}"])
            np.testing.assert_allclose(rows[f"stat_route_special_{target}"], expected)


def test_missing_optional_history_and_zero_targets_keep_feature_schema():
    daily = _daily()[["report_date", "purchase", "redeem"]]
    daily = daily.assign(purchase=0, redeem=0)
    examples = build_examples(daily, [pd.Timestamp("2013-09-22")], max_horizon=2)

    assert examples[["scale_purchase", "scale_redeem"]].gt(0).all().all()
    assert examples[[*FEATURE_GROUPS["business"], *FEATURE_GROUPS["external"]]].isna().all().all()
    assert examples[list(FEATURE_GROUPS["statistical"])].eq(0).all().all()


def test_nullable_optional_columns_remain_numeric_when_history_is_missing():
    daily = _daily()
    daily["mfd_7daily_yield"] = pd.Series(pd.NA, index=daily.index, dtype="Float64")
    daily["observed_users"] = pd.Series(pd.NA, index=daily.index, dtype="Int64")
    examples = build_examples(daily, [pd.Timestamp("2014-04-30")], max_horizon=1)

    optional = [*FEATURE_GROUPS["business"], *FEATURE_GROUPS["external"]]
    assert all(pd.api.types.is_numeric_dtype(examples[column]) for column in optional)
    assert examples["external_mfd_7daily_yield_mean_7"].isna().all()
    assert examples["business_users_7_to_28"].isna().all()
    assert examples["business_bank_purchase_ratio_28"].iloc[0] == pytest.approx(1 / 3)
    assert examples["business_transfer_redeem_ratio_28"].iloc[0] == pytest.approx(1 / 2)


def test_business_uses_recent_redeeming_users_relative_to_28_day_history():
    daily = _daily()
    origin = pd.Timestamp("2014-04-30")
    users = daily.loc[daily["report_date"].le(origin), "redeem_users"]
    daily.loc[daily["report_date"].gt(origin), "redeem_users"] = 10**9
    examples = build_examples(daily, [origin], max_horizon=2)

    assert "business_redeem_users_7_to_28" in FEATURE_GROUPS["business"]
    np.testing.assert_allclose(examples["business_redeem_users_7_to_28"], users.tail(7).mean() / users.tail(28).mean())


def test_origins_need_84_contiguous_days_through_an_observed_cutoff():
    daily = _daily()
    with pytest.raises(ValueError, match="84"):
        build_examples(daily, [pd.Timestamp("2013-09-21")])
    with pytest.raises(ValueError, match="observed"):
        build_examples(daily, [pd.Timestamp("2014-09-01")])
    with pytest.raises(ValueError, match="consecutive"):
        build_examples(daily.loc[daily["report_date"].ne("2013-08-31")], [pd.Timestamp("2013-09-24")])

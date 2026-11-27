import numpy as np
import pandas as pd
import pytest

from src import eda


def test_local_relative_flow_removes_constant_level_and_preserves_centered_spike():
    flows = pd.Series([100.0] * 15 + [200.0] + [100.0] * 15)

    relative = eda.local_relative(flows)

    assert relative.iloc[15] == pytest.approx(2.0)
    assert relative.iloc[14] == pytest.approx(1.0)
    assert pd.isna(relative.iloc[0])


def test_event_baselines_exclude_other_events_and_compare_matching_weekday():
    dates = pd.date_range("2014-08-25", "2014-09-14")
    target = pd.Timestamp("2014-09-08")
    frame = pd.DataFrame({"report_date": dates, "weekday": dates.weekday})
    frame["is_holiday"] = frame["report_date"].isin([target, target - pd.Timedelta(days=1)])
    frame["is_makeup_workday"] = frame["report_date"].eq(target - pd.Timedelta(days=2))
    frame["is_workday"] = (frame["weekday"].lt(5) & ~frame["is_holiday"]) | frame["is_makeup_workday"]
    frame["shopping_event"] = np.where(frame["report_date"].eq(target - pd.Timedelta(days=3)), "sale", "")
    frame["purchase"] = np.where(frame["weekday"].eq(0), 50, 100).astype(float)
    frame["redeem"] = frame["purchase"]
    frame.loc[frame["is_holiday"], ["purchase", "redeem"]] = 20
    frame.loc[frame["shopping_event"].ne("") | frame["is_makeup_workday"], ["purchase", "redeem"]] = 1000

    comparison = eda.event_comparisons(frame, pd.Series(frame["report_date"].eq(target), index=frame.index))

    purchase = comparison[comparison["series"].eq("purchase")].set_index("baseline")
    assert purchase.at["nearby_normal", "baseline_median"] == 100
    assert purchase.at["same_weekday", "baseline_median"] == 50
    assert purchase.at["same_weekday", "ratio"] == pytest.approx(0.4)


def test_concentration_uses_full_population_including_zero_traders():
    monthly = pd.DataFrame({"purchase": [8.0, 2.0, 0.0, 0.0], "redeem": [0.0, 0.0, 0.0, 0.0]})

    results = eda.concentration(monthly, "fixture")

    purchase = results[results["series"].eq("purchase")].set_index("top_pct")
    assert purchase.at[20, "users_in_top"] == 1
    assert purchase.at[20, "users_in_population"] == 4
    assert purchase.at[20, "share"] == pytest.approx(0.8)
    assert results.loc[results["series"].eq("redeem"), "share"].isna().all()


def test_segments_suppress_small_groups_without_saving_user_ids():
    users = pd.DataFrame(
        {
            "user_id": [88, 44, 55],
            "sex": ["group", "group", "small"],
            "city": ["group", "group", "small"],
            "constellation": ["group", "group", "small"],
            "purchase": [8, 2, 999],
            "redeem": [1, 9, 999],
            "observed_days": [2, 2, 2],
            "active_days": [1, 2, 1],
            "mean_balance": [1, 5, 999],
        }
    )

    results = eda.segment_summary(users, minimum_users=2)

    assert set(results["segment_value"]) == {"group"}
    assert "user_id" not in results
    assert results["distinct_users"].tolist() == [2, 2, 2]
    assert results["purchase_per_user"].isna().all()


def test_rate_lead_uses_only_past_x_and_respects_missing_previous_day():
    dates = pd.date_range("2014-01-01", periods=12)
    rate = pd.Series([2, 5, np.nan, 8, 3, 9, 4, 10, 2, 6, 1, 5], index=dates)
    flow = rate.shift(1) * 10

    result = eda.rate_associations(
        pd.DataFrame({"Interest_O_N": rate, "purchase": flow, "redeem": flow}),
        periods={"all": (dates.min(), dates.max())},
        lags=(1,),
        minimum_pairs=3,
    )

    purchase = result[result["series"].eq("purchase")].set_index("transformation")
    assert purchase.at["raw", "pearson"] == pytest.approx(1.0)
    assert purchase.at["raw", "pairs"] == 10
    assert purchase.at["difference", "pairs"] == 8


def test_cross_year_compares_only_shared_calendar_month_days():
    dates = pd.date_range("2013-08-29", "2014-08-31")
    frame = pd.DataFrame({"report_date": dates, "purchase": 100.0, "redeem": 100.0})
    frame.loc[frame["report_date"].dt.year.eq(2014), ["purchase", "redeem"]] = 300

    result = eda.year_overlap(frame)

    assert result["month_day"].tolist() == ["08-29", "08-30", "08-31"]
    assert (result["purchase_scale_ratio"] == 3).all()
    assert (result["purchase_shape_2014"] == 1).all()


def test_first_observed_day_includes_zero_flows_and_stays_linked_to_date():
    dates = pd.to_datetime(["2014-01-02", "2014-01-01", "2014-01-02"])
    transactions = pd.DataFrame(
        {
            "report_date": dates,
            "user_id": [2, 2, 8],
            "purchase": [100, 0, 12],
            "redeem": [10, 7, 0],
        }
    )
    cohorts = eda.first_observed_daily(transactions).set_index("report_date")

    assert cohorts.loc["2014-01-01"].to_dict() == {
        "first_observed_users": 1, "first_observed_purchase": 0, "first_observed_redeem": 7,
        "_first_observed_purchase_positive_users": 0, "_first_observed_redeem_positive_users": 1,
    }
    assert cohorts.loc["2014-01-02"].to_dict() == {
        "first_observed_users": 1, "first_observed_purchase": 12, "first_observed_redeem": 0,
        "_first_observed_purchase_positive_users": 1, "_first_observed_redeem_positive_users": 0,
    }


def test_channel_summary_preserves_component_parent_and_raw_volatility():
    history = pd.DataFrame(
        {
            "report_date": pd.date_range("2014-08-01", periods=4),
            "purchase": [10, 10, 10, 10], "direct_purchase_amt": [4, 6, 4, 6],
            "direct_purchase_amt_relative": [1, 1, 1, 1],
            "purchase_bal_amt": [1, 3, 1, 3], "purchase_bal_amt_relative": [1.0, 2.0, 4.0, 3.0],
        }
    )
    channels = eda.channel_summary(history, {"all": ("2014-08-01", "2014-08-04")})
    result = channels.set_index("channel").loc["purchase_bal_amt"]

    assert result["parent"] == "direct_purchase_amt"
    assert result["share_of_parent"] == pytest.approx(0.4)
    assert result["cv"] == pytest.approx(pd.Series([1, 3, 1, 3]).std() / 2)
    assert result["lag7_pairs"] == 0


def test_calendar_profile_separates_ordinary_work_and_weekend_from_makeup():
    frame = pd.DataFrame(
        {
            "report_date": pd.date_range("2014-08-01", periods=3), "weekday": [4, 5, 6],
            "is_workday": [True, False, True], "is_weekend": [False, True, True],
            "is_makeup_workday": [False, False, True], "is_holiday": [False, False, False],
            "shopping_event": [None, None, None], "purchase": [100, 80, 105], "redeem": [100, 95, 105],
            "purchase_relative": [1.00, 0.80, 1.05], "redeem_relative": [1.00, 0.95, 1.05],
        }
    )

    profiles = eda.calendar_profiles(frame, {"all": ("2014-08-01", "2014-08-03")})

    by_status = profiles.loc[
        profiles["dimension"].eq("ordinary_work_status") & profiles["series"].eq("purchase")
    ].set_index("value")
    assert by_status.loc["workday", "days"] == 1
    assert by_status.loc["weekend", "relative_median"] == pytest.approx(0.80)
    assert by_status.loc["makeup", "relative_median"] == pytest.approx(1.05)


def test_first_observed_export_masks_small_cohorts_but_preserves_zero():
    daily = pd.DataFrame(
        {
            "first_observed_users": [0, 20, 21, 20],
            "first_observed_purchase": [0, 901, 902, 903],
            "first_observed_redeem": [0, 801, 802, 803],
            "first_observed_purchase_share": [0, 0.1, 0.2, 0.3],
            "first_observed_redeem_share": [0, 0.1, 0.2, 0.3],
            "_first_observed_purchase_positive_users": [0, 1, 19, 20],
            "_first_observed_redeem_positive_users": [0, 20, 19, 1],
        }
    )

    exported = eda.suppress_small_cohort_amounts(daily)

    assert exported["first_observed_users"].tolist() == [0, 20, 21, 20]
    assert exported.loc[2, exported.columns[1:]].isna().all()
    assert pd.isna(exported.at[1, "first_observed_purchase"])
    assert exported.at[1, "first_observed_redeem"] == 801
    assert exported.at[0, "first_observed_purchase"] == 0
    assert exported.at[3, "first_observed_purchase"] == 903
    assert pd.isna(exported.at[3, "first_observed_redeem"])
    assert len(exported.columns) == 5


def test_monthly_cohort_amount_does_not_reveal_suppressed_daily_amount():
    daily = pd.DataFrame(
        {
            "report_date": pd.to_datetime(["2014-08-01", "2014-08-02"]),
            "purchase": [1000, 1000],
            "redeem": [1000, 1000],
            "first_observed_users": [20, 21],
            "first_observed_purchase": [20, 30],
            "first_observed_redeem": [40, 60],
            "_first_observed_purchase_positive_users": [1, 20],
            "_first_observed_redeem_positive_users": [20, 21],
        }
    )

    exported = eda.monthly_cohorts(daily)

    assert pd.isna(exported.at[0, "first_observed_purchase"])
    assert pd.isna(exported.at[0, "first_observed_purchase_share"])
    assert exported.at[0, "first_observed_redeem"] == 100


def test_anomaly_concentration_is_released_only_for_large_enough_top_group():
    dates = pd.to_datetime(["2014-07-01", "2014-07-02"])
    transactions = pd.DataFrame(
        {
            "report_date": [dates[0]] * 380 + [dates[1]] * 381,
            "purchase": [10] * 761,
            "redeem": [10] * 761,
        }
    )
    anomalies = pd.DataFrame({"series": ["purchase"] * 2, "report_date": dates})

    result = eda.anomaly_concentration(anomalies, transactions)

    assert pd.isna(result.at[0, "active_top_5pct_share"])
    assert result.at[1, "active_top_5pct_share"] == pytest.approx(20 / 381)


def test_behavior_cohort_uses_lookback_only_and_marks_future_first_observations():
    transactions = pd.DataFrame(
        {
            "user_id": [1, 2, 1, 2, 3],
            "report_date": pd.to_datetime(["2014-07-31"] * 2 + ["2014-08-01"] * 3),
            "purchase": [0, 10, 900, 5, 70],
            "redeem": [0, 0, 400, 0, 10],
            "tBalance": [0, 100, 9000, 10, 100],
        }
    )

    groups = eda.behavior_cohort_outcomes(transactions, ["2014-08-01"], minimum_users=1)
    scale = groups.loc[groups["behavior"].eq("flow_scale")].set_index("cohort")

    assert scale.at["zero", "next_purchase"] == 900
    assert scale.at["new_observation", "next_purchase"] == 70
    assert "user_id" not in groups

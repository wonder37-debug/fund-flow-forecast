import numpy as np
import pandas as pd
import pytest

from experiments.stage3 import _fold_samples, _score, build_development_folds, forecast_candidate, read_daily
from src.evaluation import Fold


def test_fold_samples_join_only_labels_observed_at_cutoff():
    days = pd.date_range("2014-04-01", periods=6)
    daily = pd.DataFrame({"report_date": days, "purchase": np.arange(6), "redeem": np.arange(6) + 10})
    examples = pd.DataFrame({
        "origin": days[[0, 1, 1, 2, 2, 3]],
        "report_date": days[[1, 2, 3, 3, 4, 4]],
        "horizon_day": [1, 1, 2, 1, 2, 1],
        "lag": [1., 2., 2., 4., 4., 3.],
    })
    fold = Fold(days[2], days[3:5], "selection")
    training, future = _fold_samples(examples, daily, fold)
    assert training["report_date"].tolist() == list(days[1:3])
    assert training["purchase"].tolist() == [1, 2]
    assert future["report_date"].tolist() == list(days[3:5])
    assert set(future.columns).isdisjoint({"purchase", "redeem"})

    changed = daily.copy()
    changed.loc[changed.report_date.gt(fold.cutoff), ["purchase", "redeem"]] = 9_000_000
    training_changed, future_changed = _fold_samples(examples, changed, fold)
    pd.testing.assert_frame_equal(training, training_changed)
    pd.testing.assert_frame_equal(future, future_changed)


def test_fold_samples_refuse_lookahead_columns_and_missing_horizon():
    days = pd.date_range("2014-04-01", periods=3)
    fold = Fold(days[0], days[1:], "selection")
    daily = pd.DataFrame({"report_date": days, "purchase": 2., "redeem": 3.})
    examples = pd.DataFrame({"report_date": days[1:2], "origin": days[0], "horizon_day": [1]})
    with pytest.raises(ValueError, match="labels"):
        _fold_samples(examples.assign(purchase=2.), daily, fold)
    with pytest.raises(ValueError, match="coverage"):
        _fold_samples(examples, daily, fold)


def test_statistical_control_uses_only_precomputed_prior_forecast():
    future = pd.DataFrame({
        "report_date": pd.date_range("2014-04-01", periods=2),
        "stat_route_special_purchase": [19., 12.], "stat_route_special_redeem": [8., 17.],
    })
    candidate = {"id": "E0", "stat": "route_special"}
    predicted, importance = forecast_candidate(pd.DataFrame(), future, candidate)
    assert importance == []
    assert predicted["purchase"].tolist() == [19., 12.]
    assert predicted["redeem"].tolist() == [8., 17.]


def test_threshold_metrics_keep_target_weights_and_exact_boundary():
    days = pd.date_range("2014-04-01", periods=2)
    actual = pd.DataFrame({"report_date": days, "purchase": [100, 100], "redeem": [100, 100]})
    predicted = actual.assign(purchase=[130, 141], redeem=[120, 70])
    scores, _ = _score(actual, predicted)
    assert scores["weighted_within_30"] == pytest.approx(.775)
    assert scores["weighted_bad_day_rate"] == pytest.approx(.225)
    assert scores["weighted_excess_over_30"] == pytest.approx(.02475)
    assert scores["purchase_severe_overprediction_rate"] == .5
    assert scores["redeem_severe_underprediction_rate"] == 0.


def test_development_folds_include_disjoint_roles_and_no_august():
    folds = build_development_folds()
    selection = [fold for fold in folds if fold.role == "selection"]
    assert len(selection) == 4
    assert [fold.future_dates[0].month for fold in selection] == [4, 5, 6, 7]
    assert all(fold.future_dates[-1] <= pd.Timestamp("2014-07-31") for fold in folds)
    assert all(len(fold.future_dates) == 30 for fold in folds if fold.role == "fixed30")


def test_parquet_reader_does_not_consume_labels_beyond_scope(tmp_path):
    days = pd.date_range("2014-07-29", periods=5, unit="ms")
    daily = pd.DataFrame({"report_date": days, "purchase": [1., 2., 3., np.nan, -1.], "redeem": 4.})
    path = tmp_path / "daily.parquet"
    daily.to_parquet(path)
    ending_july = read_daily(path, pd.Timestamp("2014-07-31"))
    assert ending_july.report_date.dtype == "datetime64[ns]"
    assert ending_july.purchase.tolist() == [1., 2., 3.]
    with pytest.raises(ValueError, match="nonnegative"):
        read_daily(path, pd.Timestamp("2014-08-02"))

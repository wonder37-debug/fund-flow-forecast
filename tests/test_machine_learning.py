import numpy as np
import pandas as pd
import pytest

from src.models.machine_learning import forecast_target


def _rows(dates, *, horizon_days=None, **columns):
    dates = pd.to_datetime(dates)
    horizons = np.ones(len(dates), dtype=int) if horizon_days is None else np.asarray(horizon_days)
    return pd.DataFrame(
        {
            "report_date": dates,
            "origin": dates - pd.to_timedelta(horizons, unit="D"),
            "horizon_day": horizons,
            "is_ordinary": True,
            "x": 0,
            "scale_purchase": 100.0,
            "stat_route_special_purchase": 100.0,
        }
    ).assign(**columns)


def _forecast(training, future, *, estimator="ridge", objective="direct_amount", **config):
    if estimator == "ridge":
        config.setdefault("params", {"alpha": 0})
    return forecast_target(
        training, future, "purchase", ("x",), {"estimator": estimator, "objective": objective, **config}
    )


def test_lookback_uses_outer_origin_and_repeated_label_dates_have_equal_weight():
    training = _rows(
        ["2014-04-04", "2014-04-07", "2014-04-07", "2014-04-08"],
        horizon_days=[1, 1, 2, 1],
        purchase=[10000, 100, 100, 300],
    )
    future = _rows(["2014-04-10", "2014-04-11"], horizon_days=[1, 2])
    predictions, _ = _forecast(training, future, lookback_days=3)
    np.testing.assert_allclose(predictions, [200, 200])


def test_preprocessing_uses_only_training_features_and_preserves_future_order():
    training = _rows(pd.date_range("2014-04-01", periods=3), x=[1, 3, np.inf], purchase=[100, 300, 200])
    future = _rows(["2014-04-05", "2014-04-04", "2014-04-06"], horizon_days=[2, 1, 3], x=[3, 1, np.nan])
    predictions, importance = _forecast(training, future)
    extreme = _rows(["2014-04-07"], horizon_days=[4], x=[10000])
    expanded, _ = _forecast(training, pd.concat([future, extreme]))
    np.testing.assert_allclose(predictions, [300, 100, 200])
    np.testing.assert_array_equal(predictions, expanded[: len(future)])
    assert importance["x"] > 0


def test_training_cannot_include_labels_after_forecast_origin():
    training = _rows(["2014-04-04", "2014-04-06"], purchase=[100, 300])
    future = _rows(["2014-04-06"], horizon_days=[1])
    with pytest.raises(ValueError, match="after.*cutoff"):
        _forecast(training, future)


def test_future_cannot_contain_label_columns():
    training = _rows(["2014-04-04"], purchase=[100])
    with pytest.raises(ValueError, match="label"):
        _forecast(training, _rows(["2014-04-05"], redeem=[np.nan]))


@pytest.mark.parametrize("objective", ["additive", "log_residual"])
def test_residual_model_clips_relative_adjustment_and_preserves_special_days(objective):
    training = _rows(["2014-04-03", "2014-04-04"], purchase=[200, 1], is_ordinary=[True, False])
    future = _rows(["2014-04-05", "2014-04-06"], horizon_days=[1, 2], is_ordinary=[True, False])
    predictions, _ = _forecast(training, future, objective=objective, shrink=0.5, clip=0.1)
    np.testing.assert_allclose(predictions, [110, 100])
    baseline, _ = _forecast(training, future, objective=objective, shrink=0)
    np.testing.assert_array_equal(baseline, [100, 100])


def test_huber_direct_relative_restores_row_scale():
    training = _rows(pd.date_range("2014-04-01", periods=10), purchase=200)
    future = _rows(["2014-04-11", "2014-04-12"], horizon_days=[1, 2], scale_purchase=[50, 300])
    predictions, _ = _forecast(training, future, estimator="huber", objective="direct_relative")
    np.testing.assert_allclose(predictions, [100, 600], rtol=1e-7)


def test_direct_log_inverts_the_mean_log_amount():
    training = _rows(["2014-04-03", "2014-04-04"], purchase=[100, 400])
    predictions, _ = _forecast(training, _rows(["2014-04-05"]), objective="direct_log")
    np.testing.assert_allclose(predictions, [200])


def test_lightgbm_relative_l1_accounts_for_amount_change_per_unit_of_ratio():
    pytest.importorskip("lightgbm")
    training = _rows(
        pd.date_range("2014-04-01", periods=5),
        purchase=[10, 10, 10, 40, 40],
        scale_purchase=[1, 1, 1, 40, 40],
    )
    future = _rows(["2014-04-06"], scale_purchase=10)
    settings = {"estimator": "lightgbm", "objective": "direct_relative", "params": {"n_estimators": 1}}
    plain, _ = _forecast(training, future, **settings, loss="l1")
    relative, importance = _forecast(training, future, **settings, loss="relative_l1")
    np.testing.assert_allclose(plain, [100])
    np.testing.assert_allclose(relative, [10])
    assert importance == {"x": 0.0}


def test_lightgbm_relative_l1_denominator_floor_limits_a_low_amounts_weight():
    pytest.importorskip("lightgbm")
    training = _rows(pd.date_range("2014-04-01", periods=5), purchase=[1, 40, 40, 40, 40])
    future = _rows(["2014-04-06"])
    predictions, _ = _forecast(training, future, estimator="lightgbm", loss="relative_l1")
    np.testing.assert_allclose(predictions, [40])

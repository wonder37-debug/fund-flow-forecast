from pathlib import Path

import numpy as np
import pandas as pd
import pytest


def daily_series(start="2014-01-01", end="2014-08-31"):
    dates = pd.date_range(start, end)
    return pd.DataFrame(
        {
            "report_date": dates,
            "purchase": np.arange(1, len(dates) + 1),
            "redeem": np.arange(1, len(dates) + 1) * 10,
        }
    )


def test_folds_use_month_ends_with_independent_month_and_fixed30_modes():
    from src.evaluation import build_folds, split_fold

    full = build_folds()
    assert [fold.cutoff.strftime("%Y-%m-%d") for fold in full] == [
        "2014-03-31", "2014-04-30", "2014-05-31", "2014-06-30", "2014-07-31"
    ]
    assert [len(fold.future_dates) for fold in full] == [30, 31, 30, 31, 31]
    assert [fold.role for fold in full] == ["selection"] * 4 + ["holdout"]
    fixed = build_folds(horizon="fixed30")
    assert [len(fold.future_dates) for fold in fixed] == [30, 30, 30, 30, 30]
    assert fixed[4].future_dates.equals(full[4].future_dates[:-1])

    frame = daily_series()
    train, actual = split_fold(frame, full[1], rolling_days=28)
    assert len(train) == 28
    assert train["report_date"].iloc[0] == pd.Timestamp("2014-04-03")
    assert train["report_date"].iloc[-1] == full[1].cutoff
    assert actual["report_date"].tolist() == full[1].future_dates.tolist()
    with pytest.raises(ValueError, match="coverage"):
        split_fold(frame[frame["report_date"] != "2014-05-05"], full[1])


def test_causal_features_and_scaling_use_only_available_history():
    from src.features.temporal import causal_target_features, fit_standardizer

    values = pd.Series([1.0, 2.0, 3.0, 900.0, 999.0], name="purchase")
    features = causal_target_features(values, lags=(1, 2), windows=(2,))
    assert pd.isna(features.loc[0, "lag_1"])
    assert features.loc[3, "lag_1"] == 3.0
    assert features.loc[3, "lag_2"] == 2.0
    assert features.loc[3, "rolling_mean_2"] == 2.5
    changed = values.copy()
    changed.iloc[3:] = -1.0
    pd.testing.assert_frame_equal(
        features.iloc[:4], causal_target_features(changed, lags=(1, 2), windows=(2,)).iloc[:4]
    )

    history = features.iloc[2:3]
    standardizer = fit_standardizer(history)
    assert standardizer.mean["lag_1"] == 2.0
    assert standardizer.scale["lag_1"] == 1.0
    assert standardizer.transform(features.loc[[4]]).loc[4, "lag_1"] == 898.0
    nonconstant = fit_standardizer(pd.DataFrame({"feature": [1.0, 3.0]}))
    assert nonconstant.mean["feature"] == 2.0
    assert nonconstant.scale["feature"] == 1.0


def test_baselines_use_expected_window_and_separate_target_values():
    from src.models.baselines import BASELINES, SeasonalMeanBaseline

    history = daily_series(end="2014-04-30")
    future = pd.DataFrame({"report_date": pd.date_range("2014-05-01", periods=8)})
    expected = {
        "B0": (None, False), "B1": (28, False), "B2": (None, True),
        "B3": (28, True), "B4": (56, True), "B5": (84, True),
    }
    for name, (window, use_weekday) in expected.items():
        spec = BASELINES[name]
        training = history if window is None else history.tail(window)
        baseline = SeasonalMeanBaseline(spec, target="purchase").fit(history)
        values = baseline.predict(future)
        for index, date in enumerate(future["report_date"]):
            rows = training[training["report_date"].dt.weekday == date.weekday()] if use_weekday else training
            assert values.iloc[index] == pytest.approx(rows["purchase"].mean())
        other = SeasonalMeanBaseline(spec, target="redeem").fit(history)
        np.testing.assert_allclose(other.predict(future), values * 10)
        assert pd.api.types.is_float_dtype(values.dtype)
        with pytest.raises(ValueError, match="after"):
            baseline.predict(history.tail(1))


def test_predict_fold_ignores_future_targets_and_auto_leakage_columns():
    from src.evaluation import build_folds, predict_fold
    from src.models.baselines import BASELINES

    panel = daily_series()
    fold = build_folds()[1]
    changed = panel.assign(trading_users=123, balance=234, shibor=345, yield_rate=456, weekday=-999)
    changed.loc[changed["report_date"] > fold.cutoff, ["purchase", "redeem"]] = 999_999
    for spec in BASELINES.values():
        forecast = predict_fold(panel, fold, spec)
        changed_forecast = predict_fold(changed, fold, spec)
        pd.testing.assert_frame_equal(forecast, changed_forecast)
        assert forecast.columns.tolist() == ["report_date", "purchase", "redeem"]
        assert forecast["report_date"].tolist() == fold.future_dates.tolist()
        assert forecast[["purchase", "redeem"]].ge(0).all().all()


def test_model_factory_uses_its_own_target_and_declared_known_calendar_values():
    from src.evaluation import predict_targets

    history = daily_series("2014-04-26", "2014-04-30").assign(weekday=900, shibor=999)
    dates = pd.date_range("2014-05-01", periods=4)

    class CalendarModel:
        def __init__(self, target):
            self.target = target

        def fit(self, history):
            assert history.columns.tolist() == ["report_date", self.target, "weekday", "is_holiday"]
            assert history["weekday"].tolist() == [5, 6, 0, 1, 2]
            assert not history["is_holiday"].any()
            self.value = history[self.target].mean()

        def predict(self, calendar):
            assert calendar.columns.tolist() == ["report_date", "weekday", "is_holiday"]
            assert calendar["weekday"].tolist() == [3, 4, 5, 6]
            assert calendar["is_holiday"].tolist() == [True, True, True, False]
            return calendar["weekday"] + self.value

    prediction = predict_targets(history, dates, CalendarModel, calendar_columns=("weekday", "is_holiday"))
    assert prediction["purchase"].tolist() == [6.0, 7.0, 8.0, 9.0]
    assert prediction["redeem"].tolist() == [33.0, 34.0, 35.0, 36.0]
    for unavailable in ("shibor", "balance", "redeem", "purchase", "revenue", "report_date"):
        with pytest.raises(ValueError, match="calendar columns"):
            predict_targets(history, dates, CalendarModel, calendar_columns=(unavailable,))


@pytest.mark.parametrize("prediction", [pd.Series([1.0]), pd.Series([1.0, np.nan]), pd.Series([1.0, -1.0])])
def test_model_factory_rejects_invalid_predictions(prediction):
    from src.evaluation import predict_targets

    class InvalidModel:
        def __init__(self, target):
            self.target = target

        def fit(self, history):
            pass

        def predict(self, calendar):
            return prediction

    history = daily_series("2014-04-26", "2014-04-30")
    dates = pd.date_range("2014-05-01", periods=2)
    with pytest.raises(ValueError, match="predictions"):
        predict_targets(history, dates, InvalidModel)


def test_metrics_document_zero_denominators_and_the_strict_bad_day_threshold():
    from src.evaluation import score_forecast, score_target

    actual = pd.DataFrame({"purchase": [100.0, 100.0, 0.0], "redeem": [100.0, 200.0, 100.0]})
    forecast = pd.DataFrame({"purchase": [70.0, 131.0, 5.0], "redeem": [100.0, 100.0, 100.0]})
    metrics, daily = score_forecast(actual, forecast)
    assert metrics["purchase_mape"] == pytest.approx(0.305)
    assert metrics["purchase_mae"] == pytest.approx(22.0)
    assert metrics["purchase_rmse"] == pytest.approx(np.sqrt((900 + 961 + 25) / 3))
    assert metrics["purchase_wape"] == pytest.approx(66 / 200)
    assert metrics["purchase_bad_day_rate"] == pytest.approx(0.5)
    assert metrics["purchase_relative_error_days"] == 2
    assert metrics["weighted_mape"] == pytest.approx(0.45 * 0.305 + 0.55 / 6)
    assert daily["purchase_relative_error"].iloc[:2].tolist() == [0.30, 0.31]
    assert pd.isna(daily["purchase_relative_error"].iloc[2])
    zero, _ = score_target([0, 0], [0, 8])
    assert np.isnan(zero["mape"]) and np.isnan(zero["wape"])
    assert np.isnan(zero["bad_day_rate"]) and zero["mae"] == 4
    with pytest.raises(ValueError, match="nonnegative"):
        score_target([5], [-1])


def test_model_selection_uses_four_folds_and_ignores_holdout():
    from src.evaluation import summarize_selection

    rows = pd.DataFrame(
        {
            "model": ["B0"] * 5 + ["B1"] * 5,
            "role": (["selection"] * 4 + ["holdout"]) * 2,
            "cutoff": [fold for fold in ("03-31", "04-30", "05-31", "06-30", "07-31")] * 2,
            "weighted_mape": [0.1, 0.2, 0.3, 0.4, 1, 0.3, 0.4, 0.5, 0.6, 0.01],
        }
    )
    summary = summarize_selection(rows).set_index("model")
    assert summary.index[0] == "B0"
    assert summary.loc["B0", "weighted_mape_mean"] == pytest.approx(0.25)
    assert summary.loc["B0", "weighted_mape_std"] == pytest.approx(np.std([0.1, 0.2, 0.3, 0.4], ddof=1))


def test_submission_matches_headerless_sample_and_checks_forecast(tmp_path: Path):
    from src.submission import write_submission

    sample = tmp_path / "sample.csv"
    sample.write_text("20140901,40000000,30000000\n20140902,40000000,30000000\n", encoding="utf-8")
    output = tmp_path / "submission.csv"
    data = daily_series("2014-09-01", "2014-09-30").astype({"purchase": float, "redeem": float})
    data.loc[0, ["purchase", "redeem"]] = [1.5, 3.49]
    write_submission(data, output, sample)
    lines = output.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 30
    assert lines[0] == "20140901,2,3"
    assert lines[-1] == "20140930,30,300"
    assert all(len(line.split(",")) == 3 for line in lines)
    with pytest.raises(ValueError, match="dates"):
        write_submission(data.iloc[::-1], output, sample)
    data.loc[0, "purchase"] = np.nan
    with pytest.raises(ValueError, match="finite"):
        write_submission(data, output, sample)


def test_experiment_writes_metrics_predictions_and_selected_submission(tmp_path: Path):
    from experiments.baselines import run_experiment

    daily = tmp_path / "daily.csv"
    sample = tmp_path / "sample.csv"
    daily_series().to_csv(daily, index=False)
    sample.write_text("20140901,40000000,30000000\n", encoding="utf-8")
    result = run_experiment(daily, sample, tmp_path / "results", seed=7)

    assert len(result.fold_metrics) == 30
    assert len(result.daily_predictions) == 6 * (30 + 31 + 30 + 31 + 31)
    assert len(result.selection_summary) == 6
    assert result.selected_model == result.selection_summary.iloc[0]["model"]
    assert result.submission_path.is_file()
    assert (result.metrics_dir / "manifest.json").is_file()
    assert (result.metrics_dir / "fold_metrics.csv").is_file()
    assert (result.metrics_dir / "daily_predictions.csv").is_file()
    assert (result.metrics_dir / "selection_summary.csv").is_file()


def test_missing_git_still_records_source_digest(monkeypatch):
    import subprocess

    from experiments.baselines import _code_provenance

    def missing_git(*args, **kwargs):
        raise FileNotFoundError("git")

    monkeypatch.setattr(subprocess, "check_output", missing_git)
    evidence = _code_provenance()
    assert evidence["git_commit"] is None
    assert evidence["experiment_sources_dirty"] is None
    assert all(len(digest) == 64 for digest in evidence["experiment_source_sha256"].values())

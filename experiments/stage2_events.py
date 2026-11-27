"""Chronological diagnostics for complete holiday episodes and makeup workdays."""

import argparse
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from experiments.baselines import DEFAULT_DAILY, _read_daily
from src.evaluation import Fold, predict_targets, score_forecast, split_fold
from src.features.calendar import build_calendar
from src.models.baselines import BASELINES, SeasonalMeanBaseline


@dataclass(frozen=True)
class Event:
    name: str
    kind: str
    fold: Fold


def evaluation_events() -> tuple[Event, ...]:
    calendar = build_calendar("2013-07-01", "2014-07-31")
    labels = calendar["holiday"].fillna("")
    episodes = labels.ne(labels.shift(fill_value="")).cumsum()
    events = []
    for _, block in calendar.loc[calendar["is_holiday"]].groupby(episodes):
        dates = pd.DatetimeIndex(block["report_date"])
        if dates[0] >= pd.Timestamp("2014-04-01"):
            fold = Fold(dates[0] - pd.Timedelta(days=1), dates, "development")
            events.append(Event(str(block["holiday"].iloc[0]), "holiday", fold))

    for date in calendar.loc[calendar["is_makeup_workday"], "report_date"]:
        role = "development" if date >= pd.Timestamp("2014-04-01") else "historical_diagnostic"
        fold = Fold(date - pd.Timedelta(days=1), pd.DatetimeIndex([date]), role)
        events.append(Event("makeup_workday", "makeup", fold))
    return tuple(sorted(events, key=lambda event: event.fold.cutoff))


def _hybrid(target: str) -> SeasonalMeanBaseline:
    return SeasonalMeanBaseline(BASELINES["B3" if target == "purchase" else "B5"], target)


def run_diagnostics(daily: pd.DataFrame, output: Path) -> None:
    from src.models.calendar_adjustment import CALENDAR_COLUMNS, CalendarAdjusted

    scores = []
    daily_results = []
    for event in evaluation_events():
        history, actual = split_fold(daily, event.fold)
        adjusted_id = "shared_holiday" if event.kind == "holiday" else "ordinary_workday"

        def adjusted(target: str) -> CalendarAdjusted:
            return CalendarAdjusted(
                target, _hybrid, holidays=event.kind == "holiday",
                makeup="workday" if event.kind == "makeup" else "weekday",
            )

        for code, factory, calendar_columns in (
            ("B3+B5", _hybrid, ()),
            (adjusted_id, adjusted, CALENDAR_COLUMNS),
        ):
            predicted = predict_targets(history, event.fold.future_dates, factory, calendar_columns=calendar_columns)
            result, errors = score_forecast(actual, predicted)
            identity = {
                "event": event.name,
                "kind": event.kind,
                "role": event.fold.role,
                "cutoff": event.fold.cutoff.date().isoformat(),
                "start": event.fold.future_dates[0].date().isoformat(),
                "end": event.fold.future_dates[-1].date().isoformat(),
                "days": len(actual),
                "model": code,
            }
            scores.append({**identity, **result})
            detail = actual.rename(columns={"purchase": "purchase_actual", "redeem": "redeem_actual"})
            detail = detail.join(predicted[["purchase", "redeem"]].add_suffix("_predicted")).join(errors)
            daily_results.append(detail.assign(**identity))

    output.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(scores).to_csv(output / "event_metrics.csv", index=False, lineterminator="\n")
    pd.concat(daily_results, ignore_index=True).to_csv(output / "event_predictions.csv", index=False, lineterminator="\n")
    print(f"{len(scores) // 2} chronological event comparisons written to {output}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--daily", type=Path, default=DEFAULT_DAILY)
    parser.add_argument("--output-dir", type=Path, default=Path("results/metrics/stage2-events"))
    args = parser.parse_args()
    daily = _read_daily(args.daily)
    run_diagnostics(daily.loc[daily["report_date"].le("2014-07-31")].copy(), args.output_dir)


if __name__ == "__main__":
    main()

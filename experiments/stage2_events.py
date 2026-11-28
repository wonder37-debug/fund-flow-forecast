"""Chronological diagnostics for complete holiday episodes and makeup workdays."""

import argparse
import hashlib
import json
import subprocess
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter

import pandas as pd

from experiments.baselines import DEFAULT_DAILY, _sha256
from experiments.stage2 import DEVELOPMENT_END, _read_scoped_daily
from src.evaluation import Fold, predict_targets, score_forecast, split_fold
from src.features.calendar import CONFIG, build_calendar
from src.models.baselines import BASELINES, DATE_COLUMN, SeasonalMeanBaseline


@dataclass(frozen=True)
class Event:
    name: str
    kind: str
    fold: Fold


def evaluation_events() -> tuple[Event, ...]:
    calendar = build_calendar("2013-07-01", DEVELOPMENT_END)
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

    daily = daily.loc[daily[DATE_COLUMN].le(DEVELOPMENT_END)]
    started = perf_counter()
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
    root = Path(__file__).resolve().parents[1]
    try:
        git_head = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True, stderr=subprocess.DEVNULL).strip()
    except (OSError, subprocess.CalledProcessError):
        git_head = None
    scoped = daily[["report_date", "purchase", "redeem"]].sort_values("report_date")
    logical_data = scoped.to_csv(index=False, date_format="%Y-%m-%d", lineterminator="\n").encode("utf-8")
    manifest = {
        "analysis": "complete holiday and makeup events forecast from the prior day",
        "git_head": git_head,
        "development_input_end": scoped["report_date"].max().date().isoformat(),
        "development_input_sha256": hashlib.sha256(logical_data).hexdigest(),
        "calendar_sha256": _sha256(CONFIG),
        "source_sha256": {name: _sha256(root / name) for name in (
            "experiments/stage2_events.py", "src/evaluation.py", "src/models/baselines.py", "src/models/calendar_adjustment.py"
        )},
        "comparison_count": len(scores) // 2,
        "duration_seconds": perf_counter() - started,
    }
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(f"{len(scores) // 2} chronological event comparisons written to {output}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--daily", type=Path, default=DEFAULT_DAILY)
    parser.add_argument("--output-dir", type=Path, default=Path("results/metrics/stage2-events"))
    args = parser.parse_args()
    run_diagnostics(_read_scoped_daily(args.daily, DEVELOPMENT_END), args.output_dir)


if __name__ == "__main__":
    main()

"""Compare historical makeup workday windows with prior ordinary weekend windows."""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from experiments.baselines import DEFAULT_DAILY
from src.features.calendar import CONFIG, build_calendar

HISTORY_START = pd.Timestamp("2013-07-01")
HISTORY_END = pd.Timestamp("2014-08-31")
DATE = "report_date"
METRICS = ("transfer_amt", "tftocard_amt", "redeem")
DESIGNS = {
    "primary": (56, 1),
    "short_lookback": (28, 1),
    "two_followups": (56, 2),
}
MIN_REFERENCE_PAIRS = 2
MATERIAL_CHANGE = 0.10


def read_daily(path: Path) -> pd.DataFrame:
    return pd.read_parquet(
        path,
        columns=[DATE, *METRICS],
        filters=[(DATE, ">=", HISTORY_START), (DATE, "<=", HISTORY_END)],
    )


def classify_pattern(event: float, followup: float, reference_event: float, reference_followup: float) -> str:
    event_delta = event - reference_event
    followup_delta = followup - reference_followup
    expected_total = reference_event + reference_followup
    window_delta = event_delta + followup_delta
    if event_delta < MATERIAL_CHANGE * reference_event:
        return "no_clear_makeup_uplift"
    if window_delta > MATERIAL_CHANGE * expected_total:
        return "partial_offset_with_surplus" if followup_delta < 0 else "surplus_without_offset"
    if followup_delta < 0:
        return "offset_with_shortfall" if window_delta < -MATERIAL_CHANGE * expected_total else "shift_compatible"
    return "inconclusive"


def _reference_anchors(
    calendar: pd.DataFrame,
    event: pd.Timestamp,
    offsets: pd.TimedeltaIndex,
    lookback_days: int,
) -> list[pd.Timestamp]:
    start = event - pd.Timedelta(days=lookback_days)
    history = calendar.loc[start:event]
    candidates = history.loc[history["is_regular_weekend"] & history["weekday"].eq(event.dayofweek)]
    anchors = []
    for anchor in candidates.index:
        followup = anchor + offsets
        if followup[-1] >= event:
            continue
        span = calendar.loc[anchor:followup[-1]]
        blocked = span["is_holiday"] | span["is_makeup_workday"] | span["shopping_event"].notna()
        if not blocked.any() and span.loc[followup, "is_workday"].all():
            anchors.append(anchor)
    return anchors


def _value_comparison(
    amounts: pd.Series,
    event: pd.Timestamp,
    followup: pd.DatetimeIndex,
    anchors: list[pd.Timestamp],
    offsets: pd.TimedeltaIndex,
) -> dict:
    event_cents = int(amounts.at[event])
    followup_cents = int(amounts.loc[followup].sum())
    reference_events = amounts.loc[anchors]
    reference_followups = pd.Series(
        [amounts.loc[anchor + offsets].sum() for anchor in anchors], index=reference_events.index
    )
    reference_event = float(reference_events.median())
    reference_followup = float(reference_followups.median())
    if reference_event <= 0 or reference_followup <= 0:
        return {"status": "nonpositive_reference"}

    reference_totals = reference_events + reference_followups
    event_delta = event_cents - reference_event
    followup_delta = followup_cents - reference_followup
    reference_window = reference_event + reference_followup
    return {
        "status": "estimated",
        "event_cents": event_cents,
        "followup_cents": followup_cents,
        "window_cents": event_cents + followup_cents,
        "reference_event_cents": reference_event,
        "reference_followup_cents": reference_followup,
        "reference_window_cents": reference_window,
        "reference_paired_total_p25_cents": reference_totals.quantile(0.25),
        "reference_paired_total_p75_cents": reference_totals.quantile(0.75),
        "event_lift_pct": 100 * event_delta / reference_event,
        "followup_lift_pct": 100 * followup_delta / reference_followup,
        "window_lift_pct": 100 * (event_delta + followup_delta) / reference_window,
        "offset_share_pct": 100 * -followup_delta / event_delta if event_delta > 0 else np.nan,
        "pattern": classify_pattern(event_cents, followup_cents, reference_event, reference_followup),
    }


def estimate_windows(
    daily: pd.DataFrame,
    calendar: pd.DataFrame,
    *,
    lookback_days: int,
    followup_days: int,
) -> pd.DataFrame:
    if lookback_days < 7 or followup_days < 1:
        raise ValueError("A lookback of at least seven days and at least one followup are required")
    amounts = daily.loc[daily[DATE].between(HISTORY_START, HISTORY_END)].set_index(DATE).sort_index()
    if amounts.empty:
        raise ValueError("No daily amounts within the historical study range")
    schedule = calendar.set_index(DATE).sort_index().loc[amounts.index.min():amounts.index.max()]
    day_range = pd.date_range(amounts.index.min(), amounts.index.max(), name=DATE)
    if not amounts.index.equals(day_range) or not schedule.index.equals(day_range):
        raise ValueError("Daily data and calendar must cover the same consecutive dates")
    for metric in METRICS:
        if (
            not pd.api.types.is_integer_dtype(amounts[metric])
            or amounts[metric].isna().any()
            or amounts[metric].lt(0).any()
        ):
            raise ValueError(f"Daily {metric} must contain nonnegative integer cents")

    regular_workdays = schedule.index[schedule["is_workday"] & ~schedule["is_makeup_workday"]]
    rows = []
    for event, event_day in schedule.loc[schedule["is_makeup_workday"]].iterrows():
        followup = regular_workdays[regular_workdays > event][:followup_days]
        offsets = followup - event
        followup_calendar = schedule.loc[followup]
        row = {
            "event_date": event.date().isoformat(),
            "event_weekday": event.day_name(),
            "event_next_holiday": event_day["next_holiday"],
            "event_days_to_holiday": event_day["days_to_holiday"],
            "event_post_holiday": event_day["is_post_holiday"],
            "followup_dates": "|".join(followup.strftime("%Y-%m-%d")),
            "followup_offsets_days": "|".join(map(str, offsets.days)),
            "followup_pre_holiday": followup_calendar["is_pre_holiday"].any(),
            "followup_post_holiday": followup_calendar["is_post_holiday"].any(),
            "followup_month_end": followup.is_month_end.any(),
            "followup_days": followup_days,
            "lookback_days": lookback_days,
            "reference_count": 0,
            "reference_anchors": "",
            "window_lift_pct": np.nan,
        }
        if len(followup) < followup_days:
            status = "incomplete_followup"
        else:
            span = schedule.loc[event:followup[-1]].iloc[1:]
            if span["is_makeup_workday"].any():
                status = "overlaps_makeup"
            elif span["is_holiday"].any():
                status = "crosses_public_holiday"
            else:
                status = "eligible"
        if status == "eligible":
            anchors = _reference_anchors(schedule, event, offsets, lookback_days)
            row["reference_count"] = len(anchors)
            row["reference_anchors"] = "|".join(day.date().isoformat() for day in anchors)
            status = "eligible" if len(anchors) >= MIN_REFERENCE_PAIRS else "insufficient_reference_pairs"
        for metric in METRICS:
            comparison = (
                _value_comparison(amounts[metric], event, followup, anchors, offsets)
                if status == "eligible" else {"status": status}
            )
            rows.append({**row, "metric": metric, **comparison})
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--daily", type=Path, default=DEFAULT_DAILY)
    parser.add_argument("--output-dir", type=Path, default=Path("results/metrics/stage3-makeup"))
    args = parser.parse_args()

    daily = read_daily(args.daily)
    calendar = build_calendar(HISTORY_START, HISTORY_END)
    results = pd.concat(
        [
            estimate_windows(daily, calendar, lookback_days=lookback, followup_days=followups).assign(design=design)
            for design, (lookback, followups) in DESIGNS.items()
        ],
        ignore_index=True,
    )
    if results.empty:
        raise ValueError("No makeup workdays within the available historical dates")
    manifest = {
        "analysis": "historical makeup workday and subsequent ordinary workdays",
        "history_start": daily[DATE].min().date().isoformat(),
        "history_end": daily[DATE].max().date().isoformat(),
        "history_cutoff": HISTORY_END.date().isoformat(),
        "daily_columns": [DATE, *METRICS],
        "daily_sha256": hashlib.sha256(
            daily.sort_values(DATE).to_csv(index=False, date_format="%Y-%m-%d", lineterminator="\n").encode("utf-8")
        ).hexdigest(),
        "calendar_sha256": hashlib.sha256(CONFIG.read_bytes()).hexdigest(),
        "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "designs": {
            design: {"lookback_days": lookback, "followup_days": followups}
            for design, (lookback, followups) in DESIGNS.items()
        },
        "minimum_reference_pairs": MIN_REFERENCE_PAIRS,
        "material_change_pct": 100 * MATERIAL_CHANGE,
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    results.to_csv(args.output_dir / "event_windows.csv", index=False, float_format="%.6f", lineterminator="\n")
    (args.output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(f"{len(results)} historical event, metric and design combinations written to {args.output_dir}")


if __name__ == "__main__":
    main()

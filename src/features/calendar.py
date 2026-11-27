"""Calendar features transcribed from the historical holiday notices."""

import argparse
import json
from pathlib import Path

import pandas as pd

CONFIG = Path(__file__).resolve().parents[2] / "configs" / "holiday_calendar.json"
DEFAULT_OUTPUT = Path("results/calendar/historical_calendar.csv")


def build_calendar(start: str | pd.Timestamp, end: str | pd.Timestamp) -> pd.DataFrame:
    """Return inclusive dates with public schedule, holiday windows and shopping markers."""
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    coverage = pd.date_range(config["coverage_start"], config["coverage_end"], freq="D")
    start, end = pd.Timestamp(start), pd.Timestamp(end)
    if pd.isna(start) or pd.isna(end) or start not in coverage or end not in coverage or end < start:
        raise ValueError("Requested dates must be ordered calendar dates within the documented coverage")

    calendar = pd.DataFrame(index=coverage.rename("report_date"))
    calendar["weekday"] = coverage.dayofweek
    calendar["is_weekend"] = calendar["weekday"].ge(5)
    calendar["holiday"] = pd.Series(pd.NA, index=coverage, dtype="string")
    calendar["is_statutory_holiday"] = False
    for holiday in config["holidays"]:
        calendar.loc[holiday["start"]:holiday["end"], "holiday"] = holiday["name"]
        calendar.loc[pd.to_datetime(holiday["statutory"]), "is_statutory_holiday"] = True

    calendar["is_holiday"] = calendar["holiday"].notna()
    calendar["is_makeup_workday"] = coverage.isin(pd.to_datetime(config["makeup_workdays"]))
    calendar["is_regular_weekend"] = (
        calendar["is_weekend"] & ~calendar["is_holiday"] & ~calendar["is_makeup_workday"]
    )
    calendar["is_workday"] = calendar["is_makeup_workday"] | (~calendar["is_weekend"] & ~calendar["is_holiday"])

    dates = coverage.to_series()
    holiday_dates = dates.where(calendar["is_holiday"])
    calendar["previous_holiday"] = calendar["holiday"].ffill()
    calendar["next_holiday"] = calendar["holiday"].bfill()
    calendar["days_since_holiday"] = (dates - holiday_dates.ffill()).dt.days.astype("Int16")
    calendar["days_to_holiday"] = (holiday_dates.bfill() - dates).dt.days.astype("Int16")
    calendar["is_pre_holiday"] = calendar["days_to_holiday"].between(1, 3).fillna(False)
    calendar["is_post_holiday"] = calendar["days_since_holiday"].between(1, 3).fillna(False)
    markers = {pd.Timestamp(date): name for date, name in config["shopping_markers"].items()}
    calendar["shopping_event"] = pd.Series(coverage.map(markers), index=coverage, dtype="string")
    return calendar.loc[start:end].reset_index()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--start", default="2013-07-01")
    parser.add_argument("--end", default="2014-09-30")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    calendar = build_calendar(args.start, args.end)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    calendar.to_csv(args.output, index=False, lineterminator="\n")
    print(f"{len(calendar)} calendar dates written to {args.output}")


if __name__ == "__main__":
    main()

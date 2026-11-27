"""Validate and write the competition's three-column submission file."""

import csv
import re
from decimal import Decimal, ROUND_HALF_UP
from pathlib import Path

import pandas as pd

from src.models.baselines import DATE_COLUMN, TARGET_COLUMNS

SUBMISSION_DATES = pd.date_range("2014-09-01", "2014-09-30")
DATE_FORMAT = "%Y%m%d"


def read_sample(sample: Path) -> int:
    """Confirm that the local sample shows ordered dates and integer cents with no header."""
    with sample.open(encoding="utf-8-sig", newline="") as stream:
        rows = list(csv.reader(stream))
    if not 0 < len(rows) <= len(SUBMISSION_DATES):
        raise ValueError("Submission sample must contain a nonempty prefix of the September dates")
    for row, date in zip(rows, SUBMISSION_DATES):
        if len(row) != 3 or row[0] != date.strftime(DATE_FORMAT):
            raise ValueError("Submission sample must have ordered YYYYMMDD dates and three columns without a header")
        if any(re.fullmatch(r"[0-9]+", value) is None for value in row[1:]):
            raise ValueError("Submission sample flows must be nonnegative integer cents")
    return len(rows)


def write_submission(predictions: pd.DataFrame, output: Path, sample: Path) -> Path:
    read_sample(sample)
    dates = pd.DatetimeIndex(predictions[DATE_COLUMN])
    if not dates.equals(SUBMISSION_DATES):
        raise ValueError("Submission dates must be exactly 2014-09-01 through 2014-09-30 in order")

    rows = []
    flows = predictions.loc[:, list(TARGET_COLUMNS)]
    for date, amounts in zip(dates, flows.itertuples(index=False, name=None)):
        rounded = []
        for amount in amounts:
            if pd.isna(amount) or not pd.api.types.is_number(amount) or isinstance(amount, (bool, complex)):
                raise ValueError("Submission amounts must be finite nonnegative numbers")
            cents = Decimal(str(amount))
            if not cents.is_finite() or cents < 0:
                raise ValueError("Submission amounts must be finite nonnegative numbers")
            rounded.append(int(cents.to_integral_value(rounding=ROUND_HALF_UP)))
        rows.append((date.strftime(DATE_FORMAT), *rounded))

    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8", newline="") as stream:
        csv.writer(stream, lineterminator="\n").writerows(rows)
    return output

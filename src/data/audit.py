import pandas as pd

from .inputs import (
    AMOUNT_COLUMNS,
    BALANCE_FILE,
    CATEGORY_COLUMNS,
    DATE,
    FUND_COLUMNS,
    FUND_FILE,
    PROFILE_COLUMNS,
    PROFILE_FILE,
    SHIBOR_COLUMNS,
    SHIBOR_FILE,
    SUBMISSION_FILE,
    USER,
    Inputs,
)


EQUATIONS = {
    "total_purchase_amt": ["direct_purchase_amt", "share_amt"],
    "total_redeem_amt": ["consume_amt", "transfer_amt"],
    "direct_purchase_amt": ["purchase_bal_amt", "purchase_bank_amt"],
    "transfer_amt": ["tftobal_amt", "tftocard_amt"],
}
FORECAST_DATES = pd.date_range("2014-09-01", "2014-09-30", freq="D")
AMOUNT_BOUNDS = [0, 99, 999, 9_999, 99_999, 999_999, 9_999_999, 99_999_999, float("inf")]
AMOUNT_BANDS = [
    "1-99", "100-999", "1000-9999", "10000-99999", "100000-999999",
    "1000000-9999999", "10000000-99999999", "100000000+",
]


def _count_map(values: pd.Series) -> dict[str, int]:
    return {str(key): int(value) for key, value in values.items()}


def _date(value: pd.Timestamp) -> str | None:
    return None if pd.isna(value) else value.date().isoformat()


def _date_range(dates: pd.Series) -> dict:
    return {
        "first_date": _date(dates.min()),
        "last_date": _date(dates.max()),
        "distinct_dates": int(dates.nunique()),
    }


def _table_statistics(inputs: Inputs) -> dict:
    stats = {}
    for filename, table in inputs.tables.items():
        item = {
            "rows": len(table),
            "columns": table.columns.tolist(),
            "column_count": table.shape[1],
            "source_dtypes": inputs.source_dtypes[filename],
            "missing_by_column": _count_map(table.isna().sum()),
            "blank_rows": int(table.isna().all(axis=1).sum()),
            "duplicate_rows_beyond_first": int(table.duplicated().sum()),
        }
        if USER in table:
            item["distinct_users"] = int(table[USER].nunique())
        date_column = DATE if DATE in table else "mfd_date"
        keys = {
            BALANCE_FILE: [USER, DATE],
            PROFILE_FILE: [USER],
        }.get(filename, [date_column])
        item["key_columns"] = keys
        item["duplicate_keys_beyond_first"] = int(table.duplicated(keys).sum())
        if date_column in table:
            item.update(_date_range(table[date_column]))
        if filename == PROFILE_FILE:
            item["distinct_values_by_profile_field"] = _count_map(table[PROFILE_COLUMNS].nunique())
        if filename in (FUND_FILE, SHIBOR_FILE, SUBMISSION_FILE):
            metrics = table.drop(columns=date_column)
            item["negative_by_column"] = _count_map(metrics.lt(0).sum())
            item["zero_by_column"] = _count_map(metrics.eq(0).sum())
        stats[filename] = item
    return stats


def _amount_total(values: pd.Series) -> int | None:
    total = values.sum(min_count=1)
    return None if pd.isna(total) else int(total)


def _amount_bands(values: pd.Series) -> dict[str, int]:
    magnitude = values.astype(float).abs()
    bands = pd.cut(magnitude, bins=AMOUNT_BOUNDS, labels=AMOUNT_BANDS)
    return _count_map(bands.value_counts(sort=False))


def _check_equation(balance: pd.DataFrame, lhs: str, rhs: list[str]) -> dict:
    complete = balance[[lhs, *rhs]].notna().all(axis=1)
    values = balance.loc[complete, [lhs, *rhs]]
    delta = values[lhs] - values[rhs].sum(axis=1)
    mismatch = delta.loc[delta.ne(0)]
    dates = balance.loc[mismatch.index, DATE]
    return {
        "equation": f"{lhs} = {' + '.join(rhs)}",
        "checked_rows": len(values),
        "incomplete_rows": int((~complete).sum()),
        "mismatch_rows": len(mismatch),
        "affected_users": int(balance.loc[mismatch.index, USER].nunique()),
        "mismatch_date_range": _date_range(dates),
        "positive_residual_rows": int(mismatch.gt(0).sum()),
        "negative_residual_rows": int(mismatch.lt(0).sum()),
        "absolute_nonzero_cents_by_band": _amount_bands(mismatch),
        "zero_components_in_mismatch_rows": _count_map(values.loc[mismatch.index].eq(0).sum()),
    }


def _profile_join(balance: pd.DataFrame, profiles: pd.DataFrame) -> dict:
    unmatched = ~balance[USER].isin(profiles[USER])
    return {
        "transaction_rows_with_profile": int((~unmatched).sum()),
        "transaction_rows_without_profile": int(unmatched.sum()),
        "transaction_users_without_profile": int(balance.loc[unmatched, USER].nunique()),
        "profiles_without_transactions": int((~profiles[USER].isin(balance[USER])).sum()),
    }


def _external_coverage(external: pd.DataFrame, dates: pd.Index, metrics: list[str]) -> dict:
    observed = external["mfd_date"].isin(dates)
    available = external.loc[observed, metrics]
    observed_dates = pd.DatetimeIndex(external.loc[observed, "mfd_date"])
    gaps = dates.difference(observed_dates)
    return {
        "calendar_dates": len(dates),
        "first_date": _date(dates.min()),
        "last_date": _date(dates.max()),
        "observed_dates": int(observed.sum()),
        "unobserved_dates": int(len(dates) - observed.sum()),
        "observed_weekend_dates": int((observed_dates.dayofweek >= 5).sum()),
        "unobserved_weekend_dates": int((gaps.dayofweek >= 5).sum()),
        "unobserved_weekday_dates": int((gaps.dayofweek < 5).sum()),
        "complete_observed_dates": int(available.notna().all(axis=1).sum()),
        "available_by_column": _count_map(available.notna().sum()),
    }


def _coverage_by_day(balance: pd.DataFrame, daily: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    user_history = balance.groupby(USER, sort=False)[DATE].agg(first="min", last="max", rows="size")
    columns = [DATE, "observed_users", "first_observed_users", "purchase_users", "redeem_users"]
    coverage = daily[columns].copy()
    coverage["last_observed_users"] = coverage[DATE].map(user_history["last"].value_counts()).fillna(0).astype(int)
    periods = {
        "first_observed_month": _count_map(user_history["first"].dt.to_period("M").value_counts().sort_index()),
        "last_observed_month": _count_map(user_history["last"].dt.to_period("M").value_counts().sort_index()),
        "records_per_user": {str(key): float(value) for key, value in user_history["rows"].describe().items()},
        "record_count_buckets": _count_map(pd.cut(
            user_history["rows"],
            [0, 1, 7, 30, 90, 180, 365, float("inf")],
            labels=["1", "2–7", "8–30", "31–90", "91–180", "181–365", "366+"],
        ).value_counts(sort=False)),
        "first_window_day_users": int(coverage["first_observed_users"].iloc[0]),
        "first_observed_after_first_window_day": int(coverage["first_observed_users"].iloc[1:].sum()),
        "last_window_day_users": int(coverage["last_observed_users"].iloc[-1]),
    }
    return coverage, periods


def _audit_amounts(balance: pd.DataFrame, daily: pd.DataFrame, groups: pd.DataFrame) -> dict:
    actual = daily.set_index(DATE)[AMOUNT_COLUMNS]
    for segment_type, group in groups.groupby("segment_type"):
        segment_sums = group.groupby(DATE)[AMOUNT_COLUMNS].sum(min_count=1)
        if not segment_sums.index.equals(actual.index) or not segment_sums.equals(actual):
            raise ValueError(f"分群 {segment_type} 的金额汇总与全量日级结果不一致")

    columns = {}
    for column in AMOUNT_COLUMNS:
        values = balance[column]
        raw_total = _amount_total(values)
        daily_total = _amount_total(daily[column])
        if raw_total != daily_total:
            raise ValueError(f"日级字段 {column} 的汇总与源数据不一致")
        bands = _amount_bands(values)
        totals_withheld = sum(bands.values()) == 1
        columns[column] = {
            "observed_rows": int(values.notna().sum()),
            "missing_rows": int(values.isna().sum()),
            "zero_rows": int(values.eq(0).sum()),
            "negative_rows": int(values.lt(0).sum()),
            "absolute_nonzero_cents_by_band": bands,
            "totals_withheld": totals_withheld,
            "raw_total_cents": None if totals_withheld else raw_total,
            "daily_total_cents": None if totals_withheld else daily_total,
            "daily_missing_dates": int(daily[column].isna().sum()),
        }
    return columns


def build_audit(inputs: Inputs, daily: pd.DataFrame, groups: pd.DataFrame) -> tuple[dict, pd.DataFrame]:
    balance = inputs.tables[BALANCE_FILE]
    submission = inputs.tables[SUBMISSION_FILE]
    coverage, periods = _coverage_by_day(balance, daily)
    date_index = pd.DatetimeIndex(daily[DATE])
    expected_dates = pd.date_range(date_index.min(), date_index.max(), freq="D")
    category = balance[CATEGORY_COLUMNS]
    targets = balance[["total_purchase_amt", "total_redeem_amt"]]
    all_categories_missing = category.isna().all(axis=1)
    partial_categories_missing = category.isna().any(axis=1) & ~all_categories_missing
    external = {"fund_yield": (FUND_FILE, FUND_COLUMNS), "shibor": (SHIBOR_FILE, SHIBOR_COLUMNS)}
    audit = {
        "tables": _table_statistics(inputs),
        "profile_join": _profile_join(balance, inputs.tables[PROFILE_FILE]),
        "transaction_date_coverage": {
            "observed_dates": len(date_index),
            "calendar_dates": len(expected_dates),
            "missing_dates": [_date(value) for value in expected_dates.difference(date_index)],
        },
        "observation_history": periods,
        "activity": {
            "known_zero_purchase_and_redeem_rows": int(targets.eq(0).fillna(False).all(axis=1).sum()),
            "missing_purchase_or_redeem_rows": int(targets.isna().any(axis=1).sum()),
            "incomplete_purchase_dates": int(daily["purchase_observed_users"].ne(daily["observed_users"]).sum()),
            "incomplete_redeem_dates": int(daily["redeem_observed_users"].ne(daily["observed_users"]).sum()),
        },
        "amount_fields": _audit_amounts(balance, daily, groups),
        "business_checks": {column: _check_equation(balance, column, terms) for column, terms in EQUATIONS.items()},
        "categories": {
            "all_four_missing_rows": int(all_categories_missing.sum()),
            "partially_missing_rows": int(partial_categories_missing.sum()),
            "all_missing_with_positive_consumption": int((all_categories_missing & balance["consume_amt"].gt(0)).sum()),
            "all_observed_with_zero_consumption": int((category.notna().all(axis=1) & balance["consume_amt"].eq(0)).sum()),
            "category_sum_matches": _check_equation(balance, "consume_amt", CATEGORY_COLUMNS),
        },
        "historical_external_availability": {
            label: _external_coverage(inputs.tables[table], date_index, fields)
            for label, (table, fields) in external.items()
        },
        "future_external_availability": {
            label: _external_coverage(inputs.tables[table], FORECAST_DATES, fields)
            for label, (table, fields) in external.items()
        },
        "submission_sample": {
            "header_present": False,
            "column_order": [DATE, "purchase", "redeem"],
            "date_format": "YYYYMMDD",
            "amount_format": "integer cents",
            "rows": len(submission),
            "dated_within_forecast_window": int(submission[DATE].isin(FORECAST_DATES).sum()),
        },
    }
    return audit, coverage

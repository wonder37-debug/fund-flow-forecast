import pandas as pd

from .inputs import (
    AMOUNT_COLUMNS,
    BALANCE_COLUMNS,
    DATE,
    FUND_FILE,
    PROFILE_COLUMNS,
    SHIBOR_FILE,
    USER,
    Inputs,
)


TARGETS = {"purchase": "total_purchase_amt", "redeem": "total_redeem_amt"}


def _transaction_flags(transactions: pd.DataFrame) -> pd.DataFrame:
    flags = pd.DataFrame(index=transactions.index)
    for target, column in TARGETS.items():
        flags[f"{target}_users"] = transactions[column].gt(0).fillna(False)
        flags[f"{target}_observed_users"] = transactions[column].notna()
    first = transactions.groupby(USER, sort=False)[DATE].transform("min")
    flags["first_observed_users"] = transactions[DATE].eq(first)
    return flags


def _aggregate(transactions: pd.DataFrame, flags: pd.DataFrame, keys: list[str]) -> pd.DataFrame:
    groups = transactions.groupby(keys, dropna=False, sort=True)
    amounts = groups[AMOUNT_COLUMNS].sum(min_count=1)
    counts = flags.groupby([transactions[key] for key in keys], dropna=False, sort=True).sum()
    daily = amounts.join(counts)
    daily["observed_users"] = groups.size()
    for target, column in TARGETS.items():
        daily[target] = daily[column]
        active = daily[f"{target}_users"].where(daily[f"{target}_users"].gt(0))
        daily[f"{target}_per_active_user"] = daily[target].div(active)
    return daily.reset_index()


def aggregate_daily(inputs: Inputs, flags: pd.DataFrame) -> pd.DataFrame:
    transactions = inputs.transactions
    daily = _aggregate(transactions, flags, [DATE])
    for target in TARGETS:
        complete = daily[f"{target}_observed_users"].eq(daily["observed_users"])
        daily[target] = daily[target].where(complete)
    balances = transactions.groupby(DATE, sort=True)[BALANCE_COLUMNS]
    means = balances.mean().add_suffix("_mean")
    medians = balances.median().add_suffix("_median")
    percentiles = balances.quantile(0.9).add_suffix("_p90")
    daily = daily.join(means.join(medians).join(percentiles), on=DATE, validate="one_to_one")
    for filename in (FUND_FILE, SHIBOR_FILE):
        external = inputs.tables[filename].rename(columns={"mfd_date": DATE})
        daily = daily.merge(external, on=DATE, how="left", validate="one_to_one")
    return daily


def aggregate_static_groups(transactions: pd.DataFrame, flags: pd.DataFrame) -> pd.DataFrame:
    parts = []
    for field in PROFILE_COLUMNS:
        daily = _aggregate(transactions, flags, [DATE, field])
        daily["segment_type"] = field
        daily["segment_value"] = daily.pop(field).astype("string")
        parts.append(daily)
    groups = pd.concat(parts, ignore_index=True)
    keys = [DATE, "segment_type", "segment_value"]
    return groups[[*keys, *groups.columns.difference(keys, sort=False)]]


def build_aggregates(inputs: Inputs) -> tuple[pd.DataFrame, pd.DataFrame]:
    flags = _transaction_flags(inputs.transactions)
    return aggregate_daily(inputs, flags), aggregate_static_groups(inputs.transactions, flags)


def behavior_features_as_of(transactions: pd.DataFrame, as_of: str | pd.Timestamp) -> pd.DataFrame:
    """为本地分群构造用户快照，严格使用截至指定日期的已观测记录。"""
    cutoff = pd.Timestamp(as_of)
    if pd.isna(cutoff) or cutoff.tzinfo is not None or cutoff != cutoff.normalize():
        raise ValueError("as_of 须为无时区的日级日期")
    target_columns = list(TARGETS.values())
    history = transactions.loc[transactions[DATE].le(cutoff), [USER, DATE, "tBalance", *target_columns]]
    history = history.reset_index(drop=True)
    groups = history.groupby(USER, sort=True)
    features = groups[DATE].agg(
        observed_days="size", first_observed_date="min", last_observed_date="max"
    )
    for target, column in TARGETS.items():
        features[target] = groups[column].sum(min_count=1)
        observed = history[column].notna()
        active = history[column].gt(0).fillna(False)
        features[f"{target}_observed_days"] = observed.groupby(history[USER]).sum()
        features[f"{target}_active_days"] = active.groupby(history[USER]).sum()
    features["tBalance_mean"] = groups["tBalance"].mean()
    last_rows = history.loc[groups[DATE].idxmax()]
    features["tBalance_last"] = last_rows.set_index(USER)["tBalance"]
    return features.reset_index()

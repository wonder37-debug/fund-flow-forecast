"""Exploratory analysis for daily purchase and redemption flows."""

from __future__ import annotations

from argparse import ArgumentParser
from pathlib import Path
import warnings

import numpy as np
import pandas as pd


FLOW_FIELDS = {"total_purchase_amt": "purchase", "total_redeem_amt": "redeem"}
TARGETS = ("purchase", "redeem")
CHANNELS = {
    "direct_purchase_amt": "purchase",
    "share_amt": "purchase",
    "purchase_bal_amt": "direct_purchase_amt",
    "purchase_bank_amt": "direct_purchase_amt",
    "consume_amt": "redeem",
    "transfer_amt": "redeem",
    "tftobal_amt": "transfer_amt",
    "tftocard_amt": "transfer_amt",
    "category1": "consume_amt",
    "category2": "consume_amt",
    "category3": "consume_amt",
    "category4": "consume_amt",
}
RATES = (
    "mfd_daily_yield", "mfd_7daily_yield",
    "Interest_O_N", "Interest_1_W", "Interest_2_W", "Interest_1_M",
    "Interest_3_M", "Interest_6_M", "Interest_9_M", "Interest_1_Y",
)
SEGMENTS = ("sex", "city", "constellation")
LAGS = (1, 7, 14, 28)
MINIMUM_SEGMENT_USERS = 100
MINIMUM_COHORT_USERS = 20
CENT_PER_MILLION_YUAN = 100_000_000
FORECAST_START = "2014-09-01"
FORECAST_END = "2014-09-30"
ANALYSIS_PERIODS = {
    "2013_covered": ("2013-07-01", "2013-12-31"),
    "2014_covered": ("2014-01-01", "2014-08-31"),
    "2014_01_to_05": ("2014-01-01", "2014-05-31"),
    "2014_06_to_08": ("2014-06-01", "2014-08-31"),
    "2014_06": ("2014-06-01", "2014-06-30"),
    "2014_07": ("2014-07-01", "2014-07-31"),
    "2014_08": ("2014-08-01", "2014-08-31"),
}


def ratio(numerator, denominator):
    return numerator / denominator.replace(0, np.nan)


def local_relative(amount: pd.Series, window: int = 29) -> pd.Series:
    """Use a centered median for historical description only."""
    baseline = amount.rolling(window, center=True, min_periods=window).median()
    return ratio(amount, baseline)


def daily_periods(history: pd.DataFrame) -> dict:
    return {"all": (history["report_date"].min(), history["report_date"].max()), **ANALYSIS_PERIODS}


def event_comparisons(history: pd.DataFrame, events: pd.Series, radius: int = 28) -> pd.DataFrame:
    shopping = history["shopping_event"].fillna("").ne("")
    normal = ~history["is_holiday"] & ~history["is_makeup_workday"] & ~shopping
    rows = []
    for event in history.loc[events].itertuples():
        distance = (history["report_date"] - event.report_date).abs().dt.days
        nearby = history.loc[normal & distance.between(1, radius)]
        baselines = (
            ("nearby_normal", nearby),
            ("same_weekday", nearby.loc[nearby["weekday"].eq(event.weekday)]),
            ("same_work_status", nearby.loc[nearby["is_workday"].eq(event.is_workday)]),
        )
        for baseline, peers in baselines:
            if peers.empty:
                continue
            for series in TARGETS:
                median = peers[series].median()
                amount = getattr(event, series)
                rows.append({
                    "report_date": event.report_date, "series": series, "baseline": baseline,
                    "peer_days": len(peers), "baseline_median": median, "amount": amount,
                    "ratio": amount / median if median else np.nan,
                })
    return pd.DataFrame(rows, columns=[
        "report_date", "series", "baseline", "peer_days", "baseline_median", "amount", "ratio",
    ])


def concentration(users: pd.DataFrame, period: str, minimum_users: int = 1) -> pd.DataFrame:
    rows = []
    for series in TARGETS:
        ordered = users[series].sort_values(ascending=False)
        total = ordered.sum()
        for percent in (1, 5, 10, 20):
            count = (len(ordered) * percent + 99) // 100
            rows.append({
                "period": period, "series": series, "top_pct": percent,
                "users_in_top": count, "users_in_population": len(ordered),
                "share": ordered.iloc[:count].sum() / total
                if total and ordered.iloc[:count].gt(0).sum() >= minimum_users else np.nan,
            })
    return pd.DataFrame(rows)


def complementary_mask(counts: pd.Series, minimum_users: int) -> pd.Series:
    hidden = counts.between(1, minimum_users - 1)
    selectable = counts.loc[~hidden & counts.gt(0)]
    if hidden.sum() == 1 and not selectable.empty:
        hidden.loc[selectable.idxmin()] = True
    return hidden


def segment_summary(users: pd.DataFrame, minimum_users: int = MINIMUM_SEGMENT_USERS) -> pd.DataFrame:
    summaries = []
    for segment in SEGMENTS:
        summary = users.groupby(segment, dropna=False, observed=True).agg(
            distinct_users=("user_id", "nunique"),
            purchase=("purchase", "sum"),
            redeem=("redeem", "sum"),
            observed_days_median=("observed_days", "median"),
            active_days_median=("active_days", "median"),
            mean_balance_median=("mean_balance", "median"),
            purchase_contributors=("purchase", lambda amount: amount.gt(0).sum()),
            redeem_contributors=("redeem", lambda amount: amount.gt(0).sum()),
        )
        for series in TARGETS:
            contributors = summary.pop(f"{series}_contributors")
            summary[series] = summary[series].mask(complementary_mask(contributors, minimum_users))
        summary = summary.loc[summary["distinct_users"] >= minimum_users].copy()
        summary["purchase_per_user"] = summary["purchase"] / summary["distinct_users"]
        summary["redeem_per_user"] = summary["redeem"] / summary["distinct_users"]
        summary.insert(0, "segment_type", segment)
        summary.index.name = "segment_value"
        summaries.append(summary.reset_index())
    return pd.concat(summaries, ignore_index=True)


def association(left: pd.Series, right: pd.Series, minimum_pairs: int = 10) -> dict:
    frame = pd.concat([left.rename("left"), right.rename("right")], axis=1).dropna()
    n = len(frame)
    if n < minimum_pairs or frame["left"].nunique() < 2 or frame["right"].nunique() < 2:
        return {"pairs": n, "pearson": np.nan, "spearman": np.nan}
    return {
        "pairs": n,
        "pearson": frame["left"].corr(frame["right"]),
        "spearman": frame["left"].corr(frame["right"], method="spearman"),
    }


def rate_associations(
    indexed: pd.DataFrame,
    periods: dict,
    lags: tuple = (0, 1, 3, 7, 14, 28),
    minimum_pairs: int = 10,
) -> pd.DataFrame:
    """Positive lags mean rate observed before target; gaps stay missing."""
    rows = []
    for rate in (name for name in RATES if name in indexed):
        for series in TARGETS:
            for name, frame in (
                ("raw", indexed),
                ("difference", indexed[[rate, series]].diff()),
            ):
                for lag in lags:
                    shifted = frame[rate].shift(lag)
                    for period, (start, end) in periods.items():
                        rows.append({
                            "period": period, "rate": rate, "series": series,
                            "transformation": name, "lag_days": lag,
                            **association(frame[series].loc[start:end], shifted.loc[start:end], minimum_pairs),
                        })
    return pd.DataFrame(rows)


def year_overlap(daily: pd.DataFrame) -> pd.DataFrame:
    indexed = daily.assign(
        month_day=daily["report_date"].dt.strftime("%m-%d"),
        year=daily["report_date"].dt.year,
    )
    earlier = indexed.loc[indexed["year"].eq(2013), ["month_day", *TARGETS]]
    later = indexed.loc[indexed["year"].eq(2014), ["month_day", *TARGETS]]
    pairs = earlier.merge(later, on="month_day", suffixes=("_2013", "_2014"), validate="one_to_one")
    for series in TARGETS:
        first, second = pairs[f"{series}_2013"], pairs[f"{series}_2014"]
        pairs[f"{series}_scale_ratio"] = ratio(second, first)
        pairs[f"{series}_shape_2013"] = first / first.median()
        pairs[f"{series}_shape_2014"] = second / second.median()
    return pairs


def first_observed_daily(transactions: pd.DataFrame) -> pd.DataFrame:
    first_date = transactions.groupby("user_id", sort=False)["report_date"].transform("min")
    first_rows = transactions.loc[transactions["report_date"].eq(first_date)]
    cohorts = first_rows.groupby("report_date", sort=True)[list(TARGETS)].sum(min_count=1)
    cohorts = cohorts.add_prefix("first_observed_")
    cohorts["first_observed_users"] = first_rows.groupby("report_date", sort=True)["user_id"].nunique()
    for series in TARGETS:
        cohorts[f"_first_observed_{series}_positive_users"] = (
            first_rows[series].gt(0).groupby(first_rows["report_date"], sort=True).sum()
        )
    return cohorts.reset_index()


def calendar_status(frame: pd.DataFrame) -> np.ndarray:
    return np.select(
        [frame["is_holiday"], frame["is_makeup_workday"], frame["is_workday"]],
        ["holiday", "makeup", "workday"],
        default="weekend",
    )


def calendar_profiles(history: pd.DataFrame, periods: dict) -> pd.DataFrame:
    frame = history.copy()
    date = frame["report_date"].dt
    frame["month_day"] = date.day
    frame["month_part"] = pd.cut(date.day, bins=[0, 10, 20, 31], labels=["start", "middle", "end"])
    frame["ordinary_work_status"] = calendar_status(frame)
    frame["work_calendar"] = np.where(frame["is_workday"], "work", "rest")
    frame["calendar_week"] = np.where(frame["is_weekend"], "weekend", "weekday")
    frame["ordinary_weekday"] = frame["weekday"].where(
        ~frame["is_holiday"] & ~frame["is_makeup_workday"] & frame["shopping_event"].fillna("").eq("")
    ).astype("Int8")
    dimensions = (
        "weekday", "ordinary_weekday", "ordinary_work_status", "work_calendar", "calendar_week",
        "month_part", "month_day",
    )
    rows = []
    for period, (start, end) in periods.items():
        subset = frame.loc[frame["report_date"].between(pd.Timestamp(start), pd.Timestamp(end))]
        for series in TARGETS:
            for dimension in dimensions:
                for value, group in subset.groupby(dimension, observed=True):
                    amounts = group[series]
                    relative = group[f"{series}_relative"].dropna()
                    rows.append({
                        "period": period, "series": series, "dimension": dimension, "value": value,
                        "days": len(group), "mean": amounts.mean(), "median": amounts.median(),
                        "std": amounts.std(), "relative_days": len(relative),
                        "relative_mean": relative.mean(), "relative_median": relative.median(),
                        "relative_std": relative.std(),
                    })
    return pd.DataFrame(rows)


def channel_summary(history: pd.DataFrame, periods: dict, monthly_users: pd.DataFrame | None = None) -> pd.DataFrame:
    rows = []
    for period, (start, end) in periods.items():
        subset = history.loc[history["report_date"].between(pd.Timestamp(start), pd.Timestamp(end))]
        for channel, parent in CHANNELS.items():
            if channel not in subset or subset.empty:
                continue
            series = subset[channel]
            relative = subset[f"{channel}_relative"]
            parent_amount = subset[parent].sum(min_count=1)
            lagged = association(relative, relative.shift(7))
            mean = series.mean()
            metrics = {
                "sum": series.sum(min_count=1), "mean": mean, "median": series.median(), "std": series.std(),
                "cv": series.std() / mean if pd.notna(mean) and mean != 0 else np.nan,
                "share_of_parent": series.sum(min_count=1) / parent_amount if parent_amount else np.nan,
                "lag7_pearson_relative": lagged["pearson"], "lag7_pairs": lagged["pairs"],
            }
            hidden = False
            if monthly_users is not None:
                relevant = monthly_users.loc[subset["report_date"].min().to_period("M"):subset["report_date"].max().to_period("M")]
                siblings = [name for name, same_parent in CHANNELS.items() if same_parent == parent]
                hidden = relevant[siblings].lt(MINIMUM_COHORT_USERS).any().any()
            rows.append({
                "period": period, "channel": channel, "parent": parent,
                "days": len(subset), "observed_days": series.count(),
                "metrics_suppressed": hidden, **({key: np.nan for key in metrics} if hidden else metrics),
            })
    return pd.DataFrame(rows)


def channel_contributors(transactions: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    by_day = {}
    by_month = {}
    month = transactions["report_date"].dt.to_period("M")
    for channel in CHANNELS:
        positive = transactions[channel].gt(0).fillna(False)
        by_day[channel] = positive.groupby(transactions["report_date"]).sum()
        by_month[channel] = transactions.loc[positive, "user_id"].groupby(month[positive]).nunique()
    months = pd.period_range(month.min(), month.max(), freq="M")
    return pd.DataFrame(by_day).sort_index(), pd.DataFrame(by_month).reindex(months).fillna(0)


def level_summaries(history: pd.DataFrame, periods: dict) -> tuple[pd.DataFrame, pd.DataFrame]:
    ranges = []
    for period, (start, end) in periods.items():
        subset = history.loc[history["report_date"].between(pd.Timestamp(start), pd.Timestamp(end))]
        if not subset.empty:
            ranges.append((period, subset))
    ranges.extend((str(month), group) for month, group in history.groupby(history["report_date"].dt.to_period("M")))
    rows = []
    associations = []
    transformed = {
        "raw": history,
        "log_difference": np.log1p(history[list(TARGETS)].astype(float)).diff(),
        "trend_relative": history[["purchase_relative", "redeem_relative"]].rename(
            columns={"purchase_relative": "purchase", "redeem_relative": "redeem"}
        ),
        "log_stl_residual": history[["purchase_stl_residual", "redeem_stl_residual"]].rename(
            columns={"purchase_stl_residual": "purchase", "redeem_stl_residual": "redeem"}
        ),
    }
    for period, window in ranges:
        for series in TARGETS:
            amounts = window[series]
            mean, std = amounts.mean(), amounts.std()
            rows.append({
                "period": period, "series": series, "start_date": window["report_date"].min(),
                "end_date": window["report_date"].max(), "days": len(amounts), "sum": amounts.sum(),
                "mean": mean, "median": amounts.median(), "std": std, "cv": std / mean if mean else np.nan,
                "min": amounts.min(), "max": amounts.max(),
            })
        for transformation, data in transformed.items():
            aligned = data.loc[window.index, list(TARGETS)]
            for lag in (0, 1, 7, 14, 28):
                associations.append({
                    "period": period, "transformation": transformation, "purchase_lag_days": lag,
                    **association(aligned["redeem"], data["purchase"].shift(lag).loc[window.index]),
                })
    return pd.DataFrame(rows), pd.DataFrame(associations)


def decompose_history(history: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    from statsmodels.tools.sm_exceptions import InterpolationWarning
    from statsmodels.tsa.seasonal import STL
    from statsmodels.tsa.stattools import acf, adfuller, kpss, pacf

    observed = history.copy()
    correlations = []
    stationarity = []
    decomposition = []
    for series in (*TARGETS, *(channel for channel in CHANNELS if channel in observed)):
        observed[f"{series}_relative"] = local_relative(observed[series])
    for series in TARGETS:
        amount = observed[series].astype(float)
        log_amount = np.log1p(amount)
        result = STL(log_amount, period=7, robust=True).fit()
        observed[f"{series}_stl_residual"] = result.resid
        for part in ("trend", "seasonal", "resid", "weights"):
            decomposition.append(pd.DataFrame({
                "report_date": observed["report_date"], "series": series, "component": part,
                "log_amount": log_amount, "value": getattr(result, part),
            }))
        for name, values in (
            ("raw", amount), ("log", log_amount), ("log_difference", log_amount.diff()),
            ("log_week_difference", log_amount.diff(7)), ("trend_relative", observed[f"{series}_relative"]),
            ("log_stl_residual", result.resid),
        ):
            valid = values.dropna()
            if len(valid) < 20 or np.isclose(valid.var(), 0.0):
                continue
            n_lags = min(42, len(valid) // 2 - 1)
            ac_auto = acf(valid, nlags=n_lags, fft=True)
            partial = pacf(valid, nlags=n_lags, method="ywm")
            correlations.extend(
                {"series": series, "transformation": name, "lag_days": lag, "acf": auto, "pacf": part}
                for lag, (auto, part) in enumerate(zip(ac_auto, partial))
            )
            adf_result = adfuller(valid, regression="c", autolag="AIC")
            with warnings.catch_warnings(record=True) as interpolation:
                warnings.simplefilter("always", InterpolationWarning)
                kpss_result = kpss(valid, regression="c", nlags="auto")
            bound = ""
            if interpolation:
                bound = "upper_estimate" if kpss_result[1] <= 0.01 else "lower_estimate"
            stationarity.append({
                "series": series, "transformation": name, "days": len(valid),
                "adf_statistic": adf_result[0], "adf_p": adf_result[1], "adf_lags": adf_result[2],
                "kpss_statistic": kpss_result[0], "kpss_p": kpss_result[1],
                "kpss_lags": kpss_result[2], "kpss_p_bound": bound,
            })
    return observed, pd.concat(decomposition, ignore_index=True), pd.DataFrame(correlations), pd.DataFrame(stationarity)


def lag_summaries(history: pd.DataFrame, periods: dict) -> pd.DataFrame:
    rows = []
    for series in TARGETS:
        amount = history[series]
        for transformation, values in (
            ("raw", amount), ("log_difference", np.log1p(amount.astype(float)).diff()),
            ("trend_relative", history[f"{series}_relative"]),
        ):
            for lag in LAGS:
                lagged = values.shift(lag)
                for period, (start, end) in periods.items():
                    mask = history["report_date"].between(pd.Timestamp(start), pd.Timestamp(end))
                    rows.append({
                        "period": period, "series": series, "transformation": transformation, "lag_days": lag,
                        **association(values.loc[mask], lagged.loc[mask]),
                    })
    return pd.DataFrame(rows)


def user_period_tables(transactions: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    activity = transactions.assign(active=transactions["purchase"].gt(0) | transactions["redeem"].gt(0))
    periods = {"all": activity}
    periods.update({str(month): group for month, group in activity.groupby(activity["report_date"].dt.to_period("M"))})
    rankings = []
    user_distributions = []
    segments = []
    for period, window in periods.items():
        groups = window.groupby("user_id", sort=False)
        users = groups.agg(
            purchase=("purchase", "sum"), redeem=("redeem", "sum"), observed_days=("report_date", "size"),
            active_days=("active", "sum"), mean_balance=("tBalance", "mean"),
        )
        users["active_day_share"] = users["active_days"] / users["observed_days"]
        rankings.append(concentration(users, period, minimum_users=MINIMUM_COHORT_USERS))
        for field in [*TARGETS, "observed_days", "active_days", "active_day_share", "mean_balance"]:
            values = users[field].dropna()
            quantiles = values.quantile([0.10, 0.25, 0.50, 0.75, 0.90, 0.99]).tolist()
            user_distributions.append({
                "period": period, "measure": field, "users_observed": values.count(),
                "nonzero_users": values.ne(0).sum(), "mean": values.mean(),
                **dict(zip(("p10", "p25", "p50", "p75", "p90", "p99"), quantiles)),
            })
        if period in {"all", "2014-06", "2014-07", "2014-08"}:
            attributes = window.drop_duplicates("user_id").set_index("user_id")[list(SEGMENTS)]
            segments.append(segment_summary(users.join(attributes).reset_index()).assign(period=period))
    return pd.concat(rankings, ignore_index=True), pd.DataFrame(user_distributions), pd.concat(segments, ignore_index=True)


def daily_activity(history: pd.DataFrame, transactions: pd.DataFrame, periods: dict) -> tuple[pd.DataFrame, pd.DataFrame]:
    activity = history[[
        "report_date", *TARGETS, "observed_users", "purchase_users", "redeem_users",
        "purchase_per_active_user", "redeem_per_active_user",
        "tBalance_mean", "tBalance_median", "tBalance_p90", "yBalance_mean", "yBalance_median", "yBalance_p90",
    ]].merge(first_observed_daily(transactions), on="report_date", how="left", validate="one_to_one")
    first_fields = [
        "first_observed_users", "first_observed_purchase", "first_observed_redeem",
        *(f"_first_observed_{series}_positive_users" for series in TARGETS),
    ]
    absent = activity["first_observed_users"].isna()
    activity.loc[absent, first_fields] = 0
    activity["first_observed_share"] = ratio(activity["first_observed_users"], activity["observed_users"])
    association_rows = []
    for series in TARGETS:
        activity[f"first_observed_{series}_share"] = ratio(activity[f"first_observed_{series}"], activity[series])
        features = (f"{series}_users", f"{series}_per_active_user", f"first_observed_{series}")
        for feature in features:
            for transformation, source, target in (
                ("raw", activity[feature], activity[series]),
                ("trend_relative", local_relative(activity[feature]), history[f"{series}_relative"]),
            ):
                for period, (start, end) in periods.items():
                    mask = activity["report_date"].between(pd.Timestamp(start), pd.Timestamp(end))
                    association_rows.append({
                        "period": period, "series": series, "activity_measure": feature,
                        "transformation": transformation, **association(source[mask], target[mask]),
                    })
    return activity, pd.DataFrame(association_rows)


def suppress_small_cohort_amounts(activity: pd.DataFrame) -> pd.DataFrame:
    result = activity.copy()
    for series in TARGETS:
        count = result.pop(f"_first_observed_{series}_positive_users")
        small = count.between(1, MINIMUM_COHORT_USERS - 1)
        for field in (f"first_observed_{series}", f"first_observed_{series}_share"):
            result[field] = result[field].mask(small)
    return result


def monthly_cohorts(activity: pd.DataFrame) -> pd.DataFrame:
    month = activity["report_date"].dt.to_period("M").astype(str)
    totals = activity.groupby(month)[
        ["first_observed_users", "first_observed_purchase", "first_observed_redeem", *TARGETS]
    ].sum(min_count=1)
    for series in TARGETS:
        small_day = activity[f"_first_observed_{series}_positive_users"].between(1, MINIMUM_COHORT_USERS - 1)
        totals[f"first_observed_{series}"] = totals[f"first_observed_{series}"].mask(small_day.groupby(month).any())
        totals[f"first_observed_{series}_share"] = ratio(totals[f"first_observed_{series}"], totals[series])
    return totals.rename_axis("month").reset_index()


def behavior_cohort_outcomes(
    transactions: pd.DataFrame,
    cutoffs: list[str],
    minimum_users: int = MINIMUM_SEGMENT_USERS,
) -> pd.DataFrame:
    rows = []
    for date in cutoffs:
        cutoff = pd.Timestamp(date)
        end = cutoff + pd.offsets.MonthBegin(1)
        dates = transactions["report_date"]
        recent = transactions.loc[dates.ge(cutoff - pd.Timedelta(days=60)) & dates.lt(cutoff)]
        next_month = transactions.loc[dates.ge(cutoff) & dates.lt(end)]
        ever_observed = transactions.loc[dates.lt(cutoff), "user_id"].unique()
        if recent.empty or next_month.empty:
            continue
        recent = recent.assign(active=recent["purchase"].gt(0) | recent["redeem"].gt(0))
        profiles = recent.groupby("user_id", sort=False).agg(
            purchase=("purchase", "sum"), redeem=("redeem", "sum"),
            activity_rate=("active", "mean"), mean_balance=("tBalance", "mean"),
        )
        profiles["flow_scale"] = profiles["purchase"] + profiles["redeem"]
        outcomes = next_month.groupby("user_id", sort=False).agg(
            next_purchase=("purchase", "sum"), next_redeem=("redeem", "sum"),
            observed_next=("report_date", "size"),
        )
        cohort = profiles.join(outcomes, how="outer")
        fields = ["next_purchase", "next_redeem", "observed_next"]
        cohort[fields] = cohort[fields].fillna(0)
        cohort["observed_next"] = cohort["observed_next"].gt(0)
        for measure in ("flow_scale", "activity_rate", "mean_balance"):
            historical = profiles[measure]
            lower, upper = historical[historical.gt(0)].quantile([1 / 3, 2 / 3])
            values = cohort[measure]
            labels = np.select(
                [values.eq(0), values.lt(lower), values.lt(upper), values.notna()],
                ["zero", "low", "middle", "high"],
                default="unobserved_recently",
            )
            new_observation = ~cohort.index.isin(ever_observed)
            labels[new_observation] = "new_observation"
            summary = cohort.groupby(labels)[fields].agg(
                eligible_users=("observed_next", "size"),
                users_observed_next=("observed_next", "sum"),
                next_purchase=("next_purchase", "sum"), next_redeem=("next_redeem", "sum"),
                purchase_contributors=("next_purchase", lambda amount: amount.gt(0).sum()),
                redeem_contributors=("next_redeem", lambda amount: amount.gt(0).sum()),
            )
            for series in TARGETS:
                field = f"next_{series}"
                contributors = summary.pop(f"{series}_contributors")
                hidden = complementary_mask(contributors, minimum_users)
                summary[f"{field}_per_eligible_user"] = summary[field] / summary["eligible_users"]
                total = cohort[field].sum()
                summary[f"{field}_share"] = summary[field] / total if total else np.nan
                for export in (field, f"{field}_per_eligible_user", f"{field}_share"):
                    summary[export] = summary[export].mask(hidden)
            summary = summary.loc[summary["eligible_users"].ge(minimum_users)]
            rows.append(summary.rename_axis("cohort").reset_index().assign(
                as_of=cutoff, behavior=measure, lower_positive_threshold=lower, upper_positive_threshold=upper,
            ))
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


def historical_events(history: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    fields = [
        "report_date", "holiday", "shopping_event", "is_holiday", "is_makeup_workday", "weekday", "is_workday",
        "days_to_holiday", "days_since_holiday", "is_pre_holiday", "is_post_holiday",
        "previous_holiday", "next_holiday",
    ]
    flags = history[["is_holiday", "is_makeup_workday", "is_pre_holiday", "is_post_holiday"]].any(axis=1)
    flags |= history["shopping_event"].notna()
    events = event_comparisons(history, flags).merge(history[fields], on="report_date", validate="many_to_one")
    events["event_type"] = np.select(
        [events["is_holiday"], events["is_makeup_workday"], events["shopping_event"].notna(),
         events["is_pre_holiday"], events["is_post_holiday"]],
        ["holiday", "makeup", "shopping", "pre_holiday", "post_holiday"],
        default="ordinary",
    )
    aggregate = events.groupby(["event_type", "series", "baseline"], as_index=False).agg(
        event_days=("ratio", "count"), median_ratio=("ratio", "median"),
        min_ratio=("ratio", "min"), max_ratio=("ratio", "max"), median_peer_days=("peer_days", "median"),
    )
    return events, aggregate


def rate_coverage(history: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for field in RATES:
        measured = history[field].notna()
        rows.append({
            "rate": field, "history_days": len(history), "available_days": measured.sum(),
            "missing_days": (~measured).sum(), "first_observation": history.loc[measured, "report_date"].min(),
            "last_observation": history.loc[measured, "report_date"].max(),
            "known_forecast_days_at_origin": 0,
        })
    return pd.DataFrame(rows)


def anomaly_summary(history: pd.DataFrame, daily_channel_users: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    records = []
    contributors = []
    for series, components in {
        "purchase": ("direct_purchase_amt", "share_amt", "purchase_bal_amt", "purchase_bank_amt"),
        "redeem": ("consume_amt", "transfer_amt", "tftobal_amt", "tftocard_amt"),
    }.items():
        residual = history[f"{series}_stl_residual"]
        median = residual.median()
        mad = (residual - median).abs().median()
        robust_z = (residual - median) / (1.4826 * mad) if not np.isclose(mad, 0.0) else residual * np.nan
        residual_flags = robust_z.abs().ge(3.5)
        recent = history["report_date"].ge(pd.Timestamp("2014-06-01"))
        highest = history[series].nlargest(3).index
        recent_highest = history.loc[recent, series].nlargest(3).index
        raw_max = history.index.isin(highest.union(recent_highest))
        for idx in history.index[residual_flags | raw_max]:
            row = history.loc[idx]
            distance = (history["report_date"] - row["report_date"]).abs().dt.days
            mask = (
                distance.between(1, 28) & history["weekday"].eq(row["weekday"])
                & ~history["is_holiday"] & ~history["is_makeup_workday"] & history["shopping_event"].isna()
            )
            peers = history.loc[mask]
            baseline = peers[series].median()
            active = f"{series}_users"
            per_user = f"{series}_per_active_user"
            records.append({
                "report_date": row["report_date"], "series": series, "amount": row[series],
                "baseline_same_weekday": baseline, "baseline_days": len(peers),
                "ratio_same_weekday": row[series] / baseline if baseline else np.nan,
                "robust_z": robust_z.loc[idx], "holiday": row["holiday"], "shopping_event": row["shopping_event"],
                "flag_stl_residual": residual_flags.loc[idx], "flag_raw_high": idx in highest or idx in recent_highest,
                "is_makeup_workday": row["is_makeup_workday"],
                "days_to_holiday": row["days_to_holiday"], "days_since_holiday": row["days_since_holiday"],
                "active_users": row[active], "baseline_same_weekday_active_users": peers[active].median(),
                "per_active_user": row[per_user], "baseline_same_weekday_per_active_user": peers[per_user].median(),
            })
            own_counts = daily_channel_users.loc[row["report_date"], list(components)]
            peer_counts = daily_channel_users.loc[peers["report_date"], list(components)]
            own_hidden = own_counts.between(1, MINIMUM_COHORT_USERS - 1).any()
            peer_hidden = ((peer_counts > 0) & (peer_counts < MINIMUM_COHORT_USERS)).any().any()
            for component in components:
                expected = peers[component].median()
                contributors.append({
                    "report_date": row["report_date"], "series": series, "component": component,
                    "amount": row[component] if not own_hidden else np.nan,
                    "baseline_same_weekday": expected if not peer_hidden else np.nan,
                    "difference": row[component] - expected if not own_hidden and not peer_hidden else np.nan,
                })
    days = pd.DataFrame(records)
    if days.empty:
        days = pd.DataFrame(columns=["report_date", "series", "robust_z"])
    return days, pd.DataFrame(contributors)


def anomaly_concentration(anomalies: pd.DataFrame, transactions: pd.DataFrame) -> pd.DataFrame:
    result = anomalies.copy()
    result["active_top_5pct_share"] = np.nan
    for series in TARGETS:
        relevant = result["series"].eq(series)
        dates = result.loc[relevant, "report_date"]
        observed = transactions.loc[transactions["report_date"].isin(dates) & transactions[series].gt(0)]
        shares = {}
        for date, rows in observed.groupby("report_date"):
            count = (5 * len(rows) + 99) // 100
            if count >= MINIMUM_COHORT_USERS:
                shares[date] = rows[series].nlargest(count).sum() / rows[series].sum()
        result.loc[relevant, "active_top_5pct_share"] = dates.map(shares)
    return result


def rolling_flows(history: pd.DataFrame) -> pd.DataFrame:
    rolling = history[["report_date", *TARGETS]].copy()
    for series in TARGETS:
        for window in (7, 14, 28):
            stats = rolling[series].rolling(window)
            mean = stats.mean()
            std = stats.std()
            rolling[f"{series}_mean_{window}d"] = mean
            rolling[f"{series}_std_{window}d"] = std
            rolling[f"{series}_cv_{window}d"] = ratio(std, mean)
        rolling[f"{series}_trend_relative"] = history[f"{series}_relative"]
    return rolling


def reconciliation(history: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for parent, children in {
        "purchase": ("direct_purchase_amt", "share_amt"),
        "direct_purchase_amt": ("purchase_bal_amt", "purchase_bank_amt"),
        "redeem": ("consume_amt", "transfer_amt"),
        "transfer_amt": ("tftobal_amt", "tftocard_amt"),
        "consume_amt": ("category1", "category2", "category3", "category4"),
    }.items():
        comparison = history[parent] - history[list(children)].sum(axis=1, min_count=len(children))
        rows.append({
            "parent": parent, "components": "+".join(children),
            "days_with_all_components": comparison.count(), "exact_days": comparison.eq(0).sum(),
            "net_gap_cent": comparison.sum(min_count=1), "absolute_gap_cent": comparison.abs().sum(min_count=1),
        })
    return pd.DataFrame(rows)


def analyze(daily: pd.DataFrame, transactions: pd.DataFrame) -> dict[str, pd.DataFrame]:
    from src.features.calendar import build_calendar

    daily = daily.sort_values("report_date").reset_index(drop=True)
    dates = daily["report_date"]
    if daily.empty or not pd.DatetimeIndex(dates).equals(pd.date_range(dates.min(), dates.max())):
        raise ValueError("Daily inputs must contain one observation per consecutive date")
    targets = daily[list(TARGETS)]
    if not np.isfinite(targets.to_numpy(dtype=float)).all() or targets.lt(0).any().any():
        raise ValueError("Daily targets must be finite and nonnegative")
    if dates.max() >= pd.Timestamp(FORECAST_START) or transactions["report_date"].max() > dates.max():
        raise ValueError("EDA accepts historical transactions only")

    calendar = build_calendar(dates.min(), dates.max())
    future_calendar = build_calendar(FORECAST_START, FORECAST_END)
    history = daily.merge(calendar, on="report_date", validate="one_to_one")
    history, decomposition, correlations, stationarity = decompose_history(history)
    periods = daily_periods(history)
    levels, flow_associations = level_summaries(history, periods)
    events, event_summary = historical_events(history)
    overlaps = year_overlap(history)
    overlap_stats = pd.DataFrame([
        {
            "series": series, "first_month_day": overlaps["month_day"].min(),
            "last_month_day": overlaps["month_day"].max(),
            "ratio_of_medians": overlaps[f"{series}_2014"].median() / overlaps[f"{series}_2013"].median(),
            "median_day_ratio": overlaps[f"{series}_scale_ratio"].median(),
            **association(overlaps[f"{series}_shape_2014"], overlaps[f"{series}_shape_2013"]),
        }
        for series in TARGETS
    ])
    user_rankings, user_distributions, segments = user_period_tables(transactions)
    activity, activity_associations = daily_activity(history, transactions, periods)
    rates = rate_associations(history.set_index("report_date"), periods)
    daily_channel_users, monthly_channel_users = channel_contributors(transactions)
    anomalies, anomaly_channels = anomaly_summary(history, daily_channel_users)
    channel_periods = periods | {
        str(month): (month.start_time, month.end_time) for month in monthly_channel_users.index
    }

    return {
        "flow_levels": levels,
        "flow_associations": flow_associations,
        "rolling_flows": rolling_flows(history),
        "log_stl_components": decomposition,
        "autocorrelation": correlations,
        "stationarity": stationarity,
        "lag_associations": lag_summaries(history, periods),
        "calendar_profiles": calendar_profiles(history, periods),
        "historical_events": events,
        "event_summary": event_summary,
        "forecast_calendar": future_calendar,
        "cross_year_overlap": overlaps,
        "cross_year_summary": overlap_stats,
        "user_concentration": user_rankings,
        "user_distributions": user_distributions,
        "segment_summary": segments,
        "daily_activity": suppress_small_cohort_amounts(activity),
        "first_observed_monthly": monthly_cohorts(activity),
        "activity_associations": activity_associations,
        "behavior_cohorts": behavior_cohort_outcomes(transactions, ["2014-06-01", "2014-07-01", "2014-08-01"]),
        "channel_summary": channel_summary(history, channel_periods, monthly_channel_users),
        "channel_reconciliation": reconciliation(history),
        "rate_coverage": rate_coverage(history),
        "rate_associations": rates,
        "anomaly_days": anomaly_concentration(anomalies, transactions),
        "anomaly_channels": anomaly_channels,
    }


def write_tables(tables: dict[str, pd.DataFrame], output: Path) -> None:
    output.mkdir(parents=True, exist_ok=True)
    for filename, frame in tables.items():
        if "user_id" in frame.columns:
            raise ValueError(f"User identifiers in aggregate table {filename}")
        frame.to_csv(output / f"{filename}.csv", index=False, lineterminator="\n", float_format="%.9g")


def draw_figures(tables: dict[str, pd.DataFrame], output: Path) -> int:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.dates as mdates
    import matplotlib.pyplot as plt

    output.mkdir(parents=True, exist_ok=True)
    palette = {"purchase": "#176a65", "redeem": "#71378a"}
    ink = "#17243a"
    muted = "#486079"
    style = {
        "axes.spines.top": False, "axes.spines.right": False, "axes.spines.left": False,
        "axes.edgecolor": "#a7b5c5", "axes.labelcolor": ink, "axes.titlecolor": ink,
        "axes.grid": True, "axes.grid.axis": "y", "axes.axisbelow": True,
        "grid.color": "#e6ebf2", "text.color": ink, "xtick.color": muted,
        "ytick.color": muted, "font.size": 10, "legend.frameon": False,
    }
    figure_count = 0

    def save(fig, filename, title):
        nonlocal figure_count
        fig.suptitle(title, x=0.04, ha="left", fontsize=16, fontweight="bold")
        fig.savefig(output / f"{filename}.png", dpi=150, facecolor="white")
        plt.close(fig)
        figure_count += 1

    def date_axis(ax):
        locator = mdates.AutoDateLocator(minticks=4, maxticks=8)
        ax.xaxis.set_major_locator(locator)
        ax.xaxis.set_major_formatter(mdates.ConciseDateFormatter(locator))

    def lines(ax, frame, columns, marker=None):
        for label, values, color in columns:
            ax.plot(frame["report_date"], values, color=color, label=label, linewidth=1.35,
                    marker=marker, markersize=3)
        date_axis(ax)
        ax.legend(loc="upper left")

    with plt.rc_context(style):
        rolling = tables["rolling_flows"]
        fig, axes = plt.subplots(2, 2, figsize=(12, 6.7), sharex=True, layout="constrained")
        for idx, series in enumerate(TARGETS):
            color = palette[series]
            factor = CENT_PER_MILLION_YUAN
            ax = axes[0, idx]
            ax.plot(rolling["report_date"], rolling[series] / factor, color=color, alpha=0.30, label="Daily")
            ax.plot(rolling["report_date"], rolling[f"{series}_mean_28d"] / factor, color=color, label="28-day mean")
            ax.set_title(series.capitalize(), loc="left", fontweight="bold")
            ax.set_ylabel("Million yuan")
            ax.legend(loc="upper left")
            ax = axes[1, idx]
            ax.plot(rolling["report_date"], rolling[f"{series}_std_28d"] / factor, color=color)
            ax.set_ylabel("28-day SD, million yuan")
            date_axis(ax)
        save(fig, "eda_01_flows", "Daily flows and changing volatility")

        decomposition = tables["log_stl_components"]
        fig, axes = plt.subplots(3, 2, figsize=(12, 7.2), sharex=True, layout="constrained")
        for column, series in enumerate(TARGETS):
            data = decomposition.loc[decomposition["series"].eq(series)]
            for row, part in enumerate(("trend", "seasonal", "resid")):
                subset = data.loc[data["component"].eq(part)]
                ax = axes[row, column]
                if part == "trend":
                    ax.plot(subset["report_date"], subset["log_amount"], color=palette[series], alpha=0.25, label="Observed")
                    ax.legend(loc="upper left")
                ax.plot(subset["report_date"], subset["value"], color=palette[series], linewidth=1.25)
                ax.set_ylabel({"trend": "Trend", "seasonal": "7-day seasonal", "resid": "Residual"}[part])
                date_axis(ax)
            axes[0, column].set_title(series.capitalize(), loc="left", fontweight="bold")
        save(fig, "eda_02_log_stl", "Robust weekly STL on log(1 + flow)")

        ac_table = tables["autocorrelation"]
        fig, axes = plt.subplots(2, 2, figsize=(11, 6.3), sharex=True, sharey=True, layout="constrained")
        for column, series in enumerate(TARGETS):
            subset = ac_table.loc[ac_table["series"].eq(series) & ac_table["lag_days"].gt(0)]
            for row, statistic in enumerate(("acf", "pacf")):
                ax = axes[row, column]
                for transformation, color in (("log", palette[series]), ("log_stl_residual", muted)):
                    data = subset.loc[subset["transformation"].eq(transformation)]
                    ax.plot(data["lag_days"], data[statistic], marker=".", markersize=4, linewidth=1.2,
                            color=color, label=transformation.replace("_", " "))
                ax.axhline(0, color="#a7b5c5", linewidth=1)
                ax.set_xticks([1, 7, 14, 21, 28, 35, 42])
                ax.set_ylabel(statistic.upper())
                ax.legend(loc="upper right")
            axes[0, column].set_title(series.capitalize(), loc="left", fontweight="bold")
            axes[1, column].set_xlabel("Lag in calendar days")
        save(fig, "eda_03_acf_pacf", "Autocorrelation before and after weekly decomposition")

        profiles = tables["calendar_profiles"]
        fig, axes = plt.subplots(2, 2, figsize=(12, 6.8), layout="constrained")
        for column, series in enumerate(TARGETS):
            data = profiles.loc[profiles["series"].eq(series)]
            for row, (dimension, title) in enumerate((
                ("ordinary_weekday", "Ordinary weekday"), ("month_day", "Day of month"),
            )):
                ax = axes[row, column]
                for period, alpha, marker in (("all", 1.0, "o"), ("2014_06_to_08", 0.48, "s")):
                    subset = data.loc[data["dimension"].eq(dimension) & data["period"].eq(period)].copy()
                    subset["position"] = subset["value"].astype(int)
                    subset = subset.sort_values("position")
                    ax.plot(subset["position"], subset["relative_median"], marker=marker, color=palette[series],
                            alpha=alpha, label=period.replace("_", " "), linewidth=1.2, markersize=3)
                ax.axhline(1, color="#a7b5c5", linewidth=1)
                ax.set_title(f"{series.capitalize()} · {title}", loc="left", fontweight="bold")
                ax.set_ylabel("Flow / local median")
                if row == 0:
                    ax.set_xticks(range(7), ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"])
                else:
                    ax.set_xticks([1, 5, 10, 15, 20, 25, 31])
                ax.legend(loc="best")
        save(fig, "eda_04_calendar", "Calendar patterns after local detrending")

        overlap = tables["cross_year_overlap"]
        fig, axes = plt.subplots(2, 1, figsize=(11, 6.1), layout="constrained")
        for ax, series in zip(axes, TARGETS):
            for year, color in ((2013, "#71869c"), (2014, palette[series])):
                ax.plot(overlap[f"{series}_shape_{year}"], color=color, linewidth=1.3, label=str(year))
            ax.set_ylabel("Flow / summer median")
            ax.set_title(series.capitalize(), loc="left", fontweight="bold")
            ax.legend(loc="upper right")
            ticks = range(0, len(overlap), 10)
            ax.set_xticks(list(ticks), overlap["month_day"].iloc[list(ticks)])
        save(fig, "eda_05_cross_year", "Same calendar dates, scaled to each year's summer median")

        events = tables["event_summary"]
        fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), sharex=True, layout="constrained")
        marks = ["pre_holiday", "holiday", "post_holiday", "makeup", "shopping"]
        for ax, series in zip(axes, TARGETS):
            subset = events.loc[events["series"].eq(series)]
            for baseline, offset, color in (("nearby_normal", -0.17, "#8ca1b3"), ("same_weekday", 0.17, palette[series])):
                measured = subset.loc[subset["baseline"].eq(baseline)].set_index("event_type").reindex(marks)
                ax.barh(np.arange(len(marks)) + offset, measured["median_ratio"], height=0.29, color=color,
                        label=baseline.replace("_", " "))
            ax.axvline(1, color="#a7b5c5", linewidth=1)
            ax.set_yticks(range(len(marks)), [mark.replace("_", " ") for mark in marks])
            ax.invert_yaxis()
            ax.set_title(series.capitalize(), loc="left", fontweight="bold")
            ax.set_xlabel("Median ratio to reference days")
            ax.legend(loc="lower right")
        save(fig, "eda_06_events", "Historical calendar events and matched reference days")

        activity = tables["daily_activity"]
        top = tables["user_concentration"].loc[lambda frame: frame["period"].eq("all")]
        fig, axes = plt.subplots(2, 2, figsize=(12, 6.8), layout="constrained")
        lines(axes[0, 0], activity, [
            (series.capitalize(), activity[f"{series}_users"], palette[series]) for series in TARGETS
        ])
        axes[0, 0].set_ylabel("Distinct active users")
        lines(axes[0, 1], activity, [
            (series.capitalize(), activity[f"{series}_per_active_user"] / 100, palette[series]) for series in TARGETS
        ])
        axes[0, 1].set_ylabel("Yuan per active user")
        lines(axes[1, 0], activity, [
            ("Users", 100 * activity["first_observed_share"], "#66758d"),
            *((series.capitalize(), 100 * activity[f"first_observed_{series}_share"], palette[series]) for series in TARGETS),
        ])
        axes[1, 0].set_ylabel("First-observed share, %")
        for idx, series in enumerate(TARGETS):
            subset = top.loc[top["series"].eq(series)]
            axes[1, 1].bar(np.arange(len(subset)) + 0.32 * idx, 100 * subset["share"], width=0.28,
                           color=palette[series], label=series.capitalize())
        axes[1, 1].set_xticks(np.arange(4) + 0.16, ["Top 1%", "Top 5%", "Top 10%", "Top 20%"])
        axes[1, 1].set_ylabel("Share of total flow, %")
        axes[1, 1].legend(loc="upper left")
        save(fig, "eda_07_users", "Active users, first observed users and flow concentration")

        channels = tables["channel_summary"].loc[lambda frame: frame["period"].str.fullmatch(r"\d{4}-\d{2}")].copy()
        channels["report_date"] = pd.to_datetime(channels["period"], format="%Y-%m")
        fig, axes = plt.subplots(3, 2, figsize=(12, 8.8), sharex=True, layout="constrained")
        pairs = [
            ("Direct purchase and credited yield", ("direct_purchase_amt", "share_amt")),
            ("Spending and transfers", ("consume_amt", "transfer_amt")),
            ("Purchase via balance and bank", ("purchase_bal_amt", "purchase_bank_amt")),
            ("Transfers to balance and card", ("tftobal_amt", "tftocard_amt")),
            ("Spending categories 1 and 2", ("category1", "category2")),
            ("Spending categories 3 and 4", ("category3", "category4")),
        ]
        for ax, (title, columns) in zip(axes.flat, pairs):
            panel = channels.pivot(index="report_date", columns="channel", values="mean").reset_index()
            lines(ax, panel, [
                (field, panel[field] / CENT_PER_MILLION_YUAN, color)
                for field, color in zip(columns, palette.values())
            ], marker="o")
            ax.set_title(title, loc="left", fontweight="bold")
            ax.set_ylabel("Monthly mean, million yuan")
        save(fig, "eda_08_channels", "Channel trends by transaction component")

        rates = tables["rate_associations"]
        fig, axes = plt.subplots(2, 2, figsize=(12, 7.9), sharex=True, sharey=True, layout="constrained")
        for row, transformation in enumerate(("raw", "difference")):
            for col, series in enumerate(TARGETS):
                subset = rates.loc[
                    rates["period"].eq("all") & rates["series"].eq(series) & rates["transformation"].eq(transformation)
                ]
                matrix = subset.pivot(index="rate", columns="lag_days", values="pearson").reindex(RATES)
                ax = axes[row, col]
                ax.grid(False)
                image = ax.imshow(matrix.to_numpy(), cmap="RdBu_r", vmin=-1, vmax=1, aspect="auto")
                ax.set_title(f"{series.capitalize()} · {transformation}", loc="left", fontweight="bold")
                ax.set_yticks(np.arange(len(RATES)), RATES)
                ax.set_xticks(np.arange(len(matrix.columns)), matrix.columns)
                ax.set_xlabel("Rate lag, calendar days")
        fig.colorbar(image, ax=axes, label="Pearson correlation", shrink=0.8)
        save(fig, "eda_09_rates", "Sensitivity of rate associations to differencing and lag")

        anomalies = tables["anomaly_days"]
        fig, axes = plt.subplots(2, 1, figsize=(12, 6.0), sharex=True, layout="constrained")
        for ax, series in zip(axes, TARGETS):
            residual = decomposition.loc[decomposition["series"].eq(series) & decomposition["component"].eq("resid")]
            points = anomalies.loc[anomalies["series"].eq(series)].merge(residual, on=["report_date", "series"])
            ax.plot(residual["report_date"], residual["value"], color="#98a8b8", linewidth=0.85)
            for flagged, subset in points.groupby("flag_stl_residual"):
                ax.scatter(subset["report_date"], subset["value"], color=palette[series], s=16 if flagged else 26,
                           marker="o" if flagged else "x", label="STL residual flag" if flagged else "Raw high only",
                           zorder=4)
            labels = points.loc[points["robust_z"].abs().nlargest(3).index]
            for point in labels.itertuples():
                ax.annotate(point.report_date.strftime("%m-%d"), (point.report_date, point.value),
                            xytext=(4, 5 if point.value < 0 else -12), textcoords="offset points", fontsize=8)
            ax.set_ylabel("Log STL residual")
            ax.set_title(series.capitalize(), loc="left", fontweight="bold")
            date_axis(ax)
            ax.legend(loc="upper right")
        save(fig, "eda_10_anomalies", "Weekly residual anomalies and raw flow peaks")
    return figure_count


def main() -> None:
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=Path("data/processed"))
    parser.add_argument("--tables", type=Path, default=Path("results/eda"))
    parser.add_argument("--figures", type=Path, default=Path("results/figures"))
    args = parser.parse_args()
    daily = pd.read_parquet(args.input / "daily.parquet")
    transactions = pd.read_parquet(
        args.input / "transactions.parquet",
        columns=["report_date", "user_id", *FLOW_FIELDS, "tBalance", *SEGMENTS, *CHANNELS],
    ).rename(columns=FLOW_FIELDS)
    tables = analyze(daily, transactions)
    write_tables(tables, args.tables)
    figures = draw_figures(tables, args.figures)
    print(f"{len(daily)} dates, {len(tables)} aggregate tables written to {args.tables}")
    print(f"{figures} figures written to {args.figures}")


if __name__ == "__main__":
    main()

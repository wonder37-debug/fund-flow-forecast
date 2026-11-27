from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd


BALANCE_FILE = "user_balance_table.csv"
PROFILE_FILE = "user_profile_table.csv"
FUND_FILE = "mfd_day_share_interest.csv"
SHIBOR_FILE = "mfd_bank_shibor.csv"
SUBMISSION_FILE = "comp_predict_table.csv"

DATE = "report_date"
USER = "user_id"
PROFILE_COLUMNS = ["sex", "city", "constellation"]
BALANCE_COLUMNS = ["tBalance", "yBalance"]
FLOW_COLUMNS = [
    "total_purchase_amt",
    "direct_purchase_amt",
    "purchase_bal_amt",
    "purchase_bank_amt",
    "total_redeem_amt",
    "consume_amt",
    "transfer_amt",
    "tftobal_amt",
    "tftocard_amt",
    "share_amt",
]
CATEGORY_COLUMNS = ["category1", "category2", "category3", "category4"]
AMOUNT_COLUMNS = [*BALANCE_COLUMNS, *FLOW_COLUMNS, *CATEGORY_COLUMNS]
FUND_COLUMNS = ["mfd_daily_yield", "mfd_7daily_yield"]
SHIBOR_COLUMNS = [
    "Interest_O_N", "Interest_1_W", "Interest_2_W", "Interest_1_M",
    "Interest_3_M", "Interest_6_M", "Interest_9_M", "Interest_1_Y",
]
SCHEMAS = {
    BALANCE_FILE: [USER, DATE, *AMOUNT_COLUMNS],
    PROFILE_FILE: [USER, *PROFILE_COLUMNS],
    FUND_FILE: ["mfd_date", *FUND_COLUMNS],
    SHIBOR_FILE: ["mfd_date", *SHIBOR_COLUMNS],
    SUBMISSION_FILE: [DATE, "purchase", "redeem"],
}


@dataclass
class Inputs:
    tables: dict[str, pd.DataFrame]
    source_dtypes: dict[str, dict[str, str]]
    transactions: pd.DataFrame


def _parse_date(values: pd.Series, filename: str) -> pd.Series:
    distinct = values.drop_duplicates()
    text = distinct.astype("string")
    if text.isna().any() or not text.str.fullmatch(r"[0-9]{8}").all():
        raise ValueError(f"{filename} 的日期列 {values.name} 须采用无空值的 YYYYMMDD")
    try:
        dates = pd.to_datetime(text, format="%Y%m%d", errors="raise")
    except ValueError as error:
        raise ValueError(f"{filename} 的日期列 {values.name} 含无效日期") from error
    return values.map(dict(zip(distinct, dates)))


def _checked_numeric(values: pd.Series, filename: str, *, integer: bool) -> pd.Series:
    try:
        numbers = pd.to_numeric(values, errors="raise")
        if not np.isfinite(numbers.dropna()).all():
            raise ValueError("含非有限数值")
        if integer:
            integers = numbers.astype("Int64" if numbers.isna().any() else "int64")
            if not integers.eq(numbers).fillna(True).all():
                raise ValueError("含非整数")
            return integers
    except (TypeError, ValueError, OverflowError) as error:
        raise ValueError(f"{filename} 的 {values.name} 含无效数值") from error
    return numbers


def _read_tables(raw_dir: Path) -> tuple[dict[str, pd.DataFrame], dict[str, dict[str, str]]]:
    tables = {}
    dtypes = {}
    for name, columns in SCHEMAS.items():
        table = pd.read_csv(
            raw_dir / name,
            header=None if name == SUBMISSION_FILE else 0,
            skip_blank_lines=False,
        )
        if name == SUBMISSION_FILE and table.shape[1] == len(columns):
            table.columns = columns
        if table.empty or table.columns.tolist() != columns:
            raise ValueError(f"{name} 为空或列名、列序不符合数据字典")
        dtypes[name] = {column: str(dtype) for column, dtype in table.dtypes.items()}
        tables[name] = table
    return tables, dtypes


def _validate_keys(table: pd.DataFrame, keys: list[str], filename: str) -> None:
    if table[keys].isna().any().any() or table.duplicated(keys).any():
        raise ValueError(f"{filename} 的键 {', '.join(keys)} 含空值或重复")


def load_inputs(raw_dir: Path) -> Inputs:
    tables, source_dtypes = _read_tables(Path(raw_dir))
    date_columns = {
        BALANCE_FILE: DATE, FUND_FILE: "mfd_date", SHIBOR_FILE: "mfd_date", SUBMISSION_FILE: DATE,
    }
    for filename, column in date_columns.items():
        tables[filename][column] = _parse_date(tables[filename][column], filename)

    for filename, columns in {
        BALANCE_FILE: [USER, *AMOUNT_COLUMNS],
        PROFILE_FILE: [USER, "sex", "city"],
        FUND_FILE: FUND_COLUMNS,
        SHIBOR_FILE: SHIBOR_COLUMNS,
        SUBMISSION_FILE: ["purchase", "redeem"],
    }.items():
        for column in columns:
            tables[filename][column] = _checked_numeric(
                tables[filename][column], filename, integer=filename not in (FUND_FILE, SHIBOR_FILE)
            )

    _validate_keys(tables[BALANCE_FILE], [USER, DATE], BALANCE_FILE)
    _validate_keys(tables[PROFILE_FILE], [USER], PROFILE_FILE)
    for filename in (FUND_FILE, SHIBOR_FILE, SUBMISSION_FILE):
        _validate_keys(tables[filename], [date_columns[filename]], filename)

    profiles = tables[PROFILE_FILE].astype({"sex": "Int64", "city": "Int64"})
    transactions = tables[BALANCE_FILE].merge(profiles, on=USER, how="left", validate="many_to_one")
    return Inputs(tables, source_dtypes, transactions)

import importlib
import json

import pandas as pd
import pytest


@pytest.fixture
def pipeline():
    return importlib.import_module("src.data.pipeline")


@pytest.fixture
def source_tables(tmp_path):
    raw = tmp_path / "raw"
    raw.mkdir()
    files = {
        "user_balance_table.csv": (
            "user_id,report_date,tBalance,yBalance,total_purchase_amt,direct_purchase_amt,"
            "purchase_bal_amt,purchase_bank_amt,total_redeem_amt,consume_amt,transfer_amt,"
            "tftobal_amt,tftocard_amt,share_amt,category1,category2,category3,category4\n"
            "101,20140830,100,80,35,30,20,10,0,0,0,0,0,5,,,,\n"
            "101,20140831,90,100,0,0,0,0,10,1,9,4,5,0,0,0,0,0\n"
            "102,20140831,150,100,50,50,20,29,2,0,2,0,2,0,,,,\n"
        ),
        "user_profile_table.csv": (
            "user_id,sex,city,constellation\n101,1,31,A\n102,0,31,B\n103,1,32,C\n"
        ),
        "mfd_day_share_interest.csv": (
            "mfd_date,mfd_daily_yield,mfd_7daily_yield\n20140830,1.2,4.1\n20140901,1.1,4.0\n"
        ),
        "mfd_bank_shibor.csv": (
            "mfd_date,Interest_O_N,Interest_1_W,Interest_2_W,Interest_1_M,Interest_3_M,"
            "Interest_6_M,Interest_9_M,Interest_1_Y\n20140829,1,2,3,4,5,6,7,8\n"
        ),
        "comp_predict_table.csv": "20140901,100,200\n20140902,110,210\n",
    }
    for name, content in files.items():
        (raw / name).write_text(content, encoding="utf-8")
    return raw


def test_pipeline_preserves_source_totals_and_missingness(pipeline, source_tables, tmp_path):
    processed, output = tmp_path / "processed", tmp_path / "audit"
    pipeline.run_pipeline(source_tables, processed, output)

    daily = pd.read_parquet(processed / "daily.parquet").set_index("report_date")
    assert daily.index.tolist() == [pd.Timestamp("2014-08-30"), pd.Timestamp("2014-08-31")]
    assert daily["purchase"].tolist() == [35, 50]
    assert daily["redeem"].tolist() == [0, 12]
    assert daily["purchase_bank_amt"].tolist() == [10, 29]
    assert daily["observed_users"].tolist() == [1, 2]
    assert daily["purchase_users"].tolist() == [1, 1]
    assert daily["redeem_users"].tolist() == [0, 2]
    assert daily["first_observed_users"].tolist() == [1, 1]
    assert daily["purchase_per_active_user"].tolist() == [35.0, 50.0]
    assert pd.isna(daily["redeem_per_active_user"].iloc[0])
    assert pd.isna(daily["category1"].iloc[0])
    assert daily["category1"].iloc[1] == 0
    assert daily["tBalance_median"].iloc[1] == 120
    assert pd.isna(daily["mfd_daily_yield"].iloc[1])
    assert daily["Interest_O_N"].isna().all()

    transactions = pd.read_parquet(processed / "transactions.parquet")
    assert transactions["purchase_bank_amt"].tolist() == [10, 0, 29]
    assert transactions["constellation"].tolist() == ["A", "A", "B"]
    groups = pd.read_parquet(processed / "group_daily.parquet")
    city = groups.loc[groups["segment_type"].eq("city")].set_index("report_date")
    assert city["purchase"].tolist() == [35, 50]
    assert city["segment_value"].tolist() == ["31", "31"]

    audit = json.loads((output / "data_audit.json").read_text(encoding="utf-8"))
    assert audit["tables"]["comp_predict_table.csv"]["rows"] == 2
    assert audit["profile_join"]["profiles_without_transactions"] == 1
    purchase_channels = audit["business_checks"]["direct_purchase_amt"]
    assert purchase_channels["mismatch_rows"] == 1
    assert purchase_channels["absolute_nonzero_cents_by_band"]["1-99"] == 1
    assert "net_residual_cents" not in purchase_channels
    assert "absolute_residual_cents" not in purchase_channels
    assert "min_nonzero_residual_cents" not in purchase_channels
    assert "max_nonzero_residual_cents" not in purchase_channels
    amount = audit["amount_fields"]["total_purchase_amt"]
    assert amount["absolute_nonzero_cents_by_band"]["1-99"] == 2
    assert amount["raw_total_cents"] == amount["daily_total_cents"] == 85
    assert "min_observed_cents" not in amount
    assert "max_observed_cents" not in amount
    share = audit["amount_fields"]["share_amt"]
    assert share["totals_withheld"] is True
    assert share["raw_total_cents"] is share["daily_total_cents"] is None
    assert audit["activity"]["known_zero_purchase_and_redeem_rows"] == 0
    assert audit["activity"]["missing_purchase_or_redeem_rows"] == 0
    assert audit["future_external_availability"]["fund_yield"]["observed_dates"] == 1
    assert audit["future_external_availability"]["shibor"]["observed_dates"] == 0
    assert audit["historical_external_availability"]["shibor"]["observed_weekend_dates"] == 0
    coverage_path = output / "coverage_by_day.csv"
    assert b"\r" not in coverage_path.read_bytes()
    coverage = pd.read_csv(coverage_path)
    assert coverage["first_observed_users"].tolist() == [1, 1]
    assert coverage["last_observed_users"].tolist() == [0, 2]
    assert "user_id" not in coverage.columns


def test_incomplete_daily_target_is_missing_and_retains_observed_subtotal(pipeline, source_tables, tmp_path):
    path = source_tables / "user_balance_table.csv"
    content = path.read_text(encoding="utf-8")
    content = content.replace("101,20140831,90,100,0,", "101,20140831,90,100,5,")
    content = content.replace("102,20140831,150,100,50,", "102,20140831,150,100,,")
    path.write_text(content, encoding="utf-8")
    processed, output = tmp_path / "processed", tmp_path / "audit"
    pipeline.run_pipeline(source_tables, processed, output)

    daily = pd.read_parquet(processed / "daily.parquet").set_index("report_date")
    assert daily.loc[pd.Timestamp("2014-08-30"), "purchase"] == 35
    partial = daily.loc[pd.Timestamp("2014-08-31")]
    assert partial["purchase_observed_users"] == 1
    assert partial["observed_users"] == 2
    assert pd.isna(partial["purchase"])
    assert partial["total_purchase_amt"] == 5
    assert partial["purchase_per_active_user"] == 5
    assert partial["redeem"] == 12


def test_behavior_input_includes_cutoff_day_and_is_invariant_to_future_rows(pipeline, source_tables):
    transactions = pipeline.load_inputs(source_tables).transactions
    as_of = pipeline.behavior_features_as_of(transactions, "2014-08-31")
    assert as_of["user_id"].tolist() == [101, 102]
    assert as_of["observed_days"].tolist() == [2, 1]
    assert as_of["purchase"].tolist() == [35, 50]
    assert as_of["redeem"].tolist() == [10, 2]
    assert as_of["tBalance_mean"].tolist() == [95.0, 150.0]
    assert as_of["tBalance_last"].tolist() == [90, 150]
    assert as_of["last_observed_date"].tolist() == [pd.Timestamp("2014-08-31")] * 2
    assert pipeline.behavior_features_as_of(transactions, "2014-08-29").empty

    future = transactions.iloc[:2].copy()
    future["report_date"] = pd.Timestamp("2014-09-01")
    future.loc[future.index[1], "user_id"] = 103
    future["tBalance"] = 900_000
    future["total_purchase_amt"] = 900_000
    future["total_redeem_amt"] = 900_000
    extended = pd.concat([transactions, future], ignore_index=True)
    pd.testing.assert_frame_equal(pipeline.behavior_features_as_of(extended, "2014-08-31"), as_of)


def test_absent_calendar_day_is_reported_without_zero_fill(pipeline, source_tables, tmp_path):
    path = source_tables / "user_balance_table.csv"
    path.write_text(path.read_text(encoding="utf-8").replace("20140830", "20140828"), encoding="utf-8")
    processed, output = tmp_path / "processed", tmp_path / "audit"
    pipeline.run_pipeline(source_tables, processed, output)
    daily = pd.read_parquet(processed / "daily.parquet")
    assert daily["report_date"].tolist() == [pd.Timestamp("2014-08-28"), pd.Timestamp("2014-08-31")]
    audit = json.loads((output / "data_audit.json").read_text(encoding="utf-8"))
    assert audit["transaction_date_coverage"]["missing_dates"] == ["2014-08-29", "2014-08-30"]


def test_duplicate_user_date_key_is_rejected(pipeline, source_tables, tmp_path):
    path = source_tables / "user_balance_table.csv"
    content = path.read_text(encoding="utf-8")
    path.write_text(content + content.splitlines()[1] + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="user_id.*report_date"):
        pipeline.run_pipeline(source_tables, tmp_path / "processed", tmp_path / "audit")
    assert not (tmp_path / "processed").exists()

import csv
import json

import pandas as pd
import pytest

from experiments.stage2 import run_development, run_holdout
from experiments.stage2_forecast import run_forecast


@pytest.fixture
def frozen_run(tmp_path):
    daily = tmp_path / "daily.csv"
    dates = pd.date_range("2013-07-01", "2014-09-01")
    pd.DataFrame({"report_date": dates, "purchase": 100 + dates.dayofweek, "redeem": 200 + dates.dayofweek}).to_csv(
        daily, index=False,
    )
    config = tmp_path / "config.json"
    config.write_text(json.dumps({"candidates": [
        {"id": "blend_equal", "kind": "baseline", "specs": {"purchase": "B3", "redeem": "B3"}},
        {"id": "cal_workday", "kind": "baseline", "specs": {"purchase": "B5", "redeem": "B5"}},
    ]}), encoding="utf-8")
    freeze = run_development(daily, config, tmp_path / "development")
    holdout = run_holdout(daily, freeze, tmp_path / "holdout")
    sample = tmp_path / "sample.csv"
    sample.write_text("20140901,40000000,30000000\n20140902,40000000,30000000\n", encoding="utf-8")
    return daily, freeze, holdout, sample


def test_forecast_writes_confirmed_roles_and_ignores_september_actuals(tmp_path, frozen_run):
    daily, freeze, holdout, sample = frozen_run
    output = tmp_path / "submission"
    replay = tmp_path / "replay"
    manifest = json.loads(run_forecast(daily, freeze, holdout, sample, output).read_text(encoding="utf-8"))
    assert manifest["training_input"]["scope_end"] == "2014-08-31"
    assert manifest["models"] == {"primary": "blend_equal", "backup": "cal_workday"}
    daily.write_text(daily.read_text(encoding="utf-8").replace("2014-09-01,100,200", "2014-09-01,unavailable,unavailable"), encoding="utf-8")
    run_forecast(daily, freeze, holdout, sample, replay)

    for model in ("primary", "backup"):
        name = f"stage2_{model}.csv"
        assert (output / name).read_bytes() == (replay / name).read_bytes()
        with (output / name).open(encoding="utf-8", newline="") as stream:
            rows = list(csv.reader(stream))
        assert len(rows) == 30
        assert rows[0] == ["20140901", "100", "200"]
        assert rows[-1] == ["20140930", "101", "201"]
    compare = pd.read_csv(output / "calendar_comparison.csv").set_index("report_date")
    assert compare.loc["2014-09-08", "is_holiday"]
    assert compare.loc["2014-09-28", "is_makeup_workday"]
    assert compare.loc["2014-09-28", "is_pre_holiday"]
    assert "2014-09-02" not in compare.index


@pytest.mark.parametrize("tamper", ["august_input", "holdout_artifact"])
def test_forecast_rejects_changed_validated_august(tmp_path, frozen_run, tamper):
    daily, freeze, holdout, sample = frozen_run
    if tamper == "august_input":
        daily.write_text(daily.read_text(encoding="utf-8").replace("2014-08-31,106,206", "2014-08-31,107,206"), encoding="utf-8")
    else:
        (holdout.parent / "fold_metrics.csv").write_text("changed\n", encoding="utf-8")
    output = tmp_path / "submission"
    with pytest.raises(ValueError, match="August"):
        run_forecast(daily, freeze, holdout, sample, output)
    assert not output.exists()

import hashlib
import json

import pandas as pd
import pytest

from experiments.stage3_release import _approved_candidates, _read_pinned, _validate_august_records


def test_approved_mix_copies_purchase_config_and_routes_redeem():
    candidates = [
        {"id": "route_special", "stat": "route_special"},
        {"id": "lightgbm_ratio", "groups": ["calendar"], "model": {"estimator": "lightgbm", "shrink": .35}},
    ]
    frozen, roles = _approved_candidates(candidates)
    assert roles == {"primary": "route_special", "backup": "lightgbm_purchase_route_redeem"}
    assert frozen[:-1] == candidates
    assert frozen[-1]["model"] == candidates[1]["model"]
    assert frozen[-1]["groups"] == ["calendar"]
    assert frozen[-1]["targets"] == {"redeem": {"stat": "route_special"}}


def test_digest_is_independent_and_accepts_windows_line_endings(tmp_path):
    path = tmp_path / "manifest.json"
    original = b'{\n  "role": "holdout"\n}\n'
    digest = hashlib.sha256(original).hexdigest()
    path.write_bytes(original.replace(b"\n", b"\r\n"))
    assert _read_pinned(path, digest) == {"role": "holdout"}
    path.write_bytes(json.dumps({"role": "holdout", "artifacts_sha256": {"file": "a" * 64}}).encode())
    with pytest.raises(ValueError, match="digest"):
        _read_pinned(path, digest)


def test_august_requires_one_prediction_per_date_per_successful_model():
    meta = {"model": "route_special", "role": "holdout", "cutoff": "2014-07-31"}
    folds = pd.DataFrame([{**meta, "forecast_days": 31, "forecast_start": "2014-08-01", "forecast_end": "2014-08-31"}])
    daily = pd.DataFrame({**meta, "report_date": pd.date_range("2014-08-01", "2014-08-31"),
                          "purchase_predicted": 3., "redeem_predicted": 4.})
    failures = pd.DataFrame(columns=[*meta, "error_type", "message"])
    _validate_august_records(folds, daily, failures, {"route_special"}, {"route_special"})
    with pytest.raises(ValueError, match="August dates"):
        _validate_august_records(folds, daily.iloc[1:], failures, {"route_special"}, {"route_special"})
    with pytest.raises(ValueError, match="August dates"):
        _validate_august_records(folds, pd.concat([daily.iloc[1:], daily.iloc[[-1]]]), failures,
                                 {"route_special"}, {"route_special"})
    with pytest.raises(ValueError, match="candidate"):
        _validate_august_records(folds, daily.assign(model="other"), failures, {"route_special"}, {"route_special"})

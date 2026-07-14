import pandas as pd
import pytest
from types import SimpleNamespace

from dataloader import SWEDataLoader
from train import add_conformal_intervals, conformal_quantiles_by_group


def test_group_quantiles_use_global_fallback_for_sparse_snow_days():
    calibration_df = pd.DataFrame(
        [
            {"snow_day": 1, "actual_swe": 3.0, "predicted_swe": 2.0},
            {"snow_day": 1, "actual_swe": 4.0, "predicted_swe": 2.0},
            {"snow_day": 2, "actual_swe": 5.0, "predicted_swe": 2.0},
        ]
    )

    q_hats = conformal_quantiles_by_group(
        calibration_df,
        group_col="snow_day",
        alpha=0.1,
        min_calibration_points=2,
        fallback_q_hat=4.0,
    )

    assert q_hats[1] > 0
    assert q_hats[2] == 4.0


def test_conformal_intervals_are_raw_symmetric_intervals():
    predictions_df = pd.DataFrame(
        [
            {
                "station": "A",
                "snow_day": 1,
                "actual_swe": 0.0,
                "predicted_swe": 2.0,
            }
        ]
    )

    intervals = add_conformal_intervals(
        predictions_df,
        q_hat_by_group={1: 5.0},
        group_col="snow_day",
    )

    assert intervals.loc[0, "conformal_lower_swe"] == -3.0
    assert intervals.loc[0, "conformal_upper_swe"] == 7.0


def test_disabled_conformal_does_not_require_calibration_years():
    cfg = SimpleNamespace(
        train_start_year=2001,
        train_end_year=2002,
        val_start_year=2003,
        val_end_year=2003,
        test_start_year=2004,
        test_end_year=2004,
        conformal=SimpleNamespace(enabled=False),
    )

    dataloader = SWEDataLoader(cfg)

    assert dataloader.calibration_years is None


def test_enabled_conformal_requires_calibration_years():
    cfg = SimpleNamespace(
        train_start_year=2001,
        train_end_year=2002,
        val_start_year=2003,
        val_end_year=2003,
        test_start_year=2004,
        test_end_year=2004,
        conformal=SimpleNamespace(enabled=True),
    )

    with pytest.raises(ValueError, match="calibration_start_year"):
        SWEDataLoader(cfg)

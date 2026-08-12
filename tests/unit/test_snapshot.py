from __future__ import annotations

import pytest
from demand_forecast.data.snapshot import (
    SNAPSHOT_COLUMNS,
    build_reference_snapshot,
    load_reference_artifacts,
    save_reference_artifacts,
)


def test_build_reference_snapshot_has_one_row_per_store(features_df):
    snapshot = build_reference_snapshot(features_df)
    n_stores = features_df["store_nbr"].nunique()
    assert len(snapshot) == n_stores
    assert list(snapshot.columns) == SNAPSHOT_COLUMNS


def test_snapshot_holds_the_most_recent_week_per_store(features_df):
    snapshot = build_reference_snapshot(features_df).set_index("store_nbr")
    for store_nbr, group in features_df.groupby("store_nbr", observed=True):
        latest_row = group.sort_values("date").iloc[-1]
        assert snapshot.loc[store_nbr, "sales_lag_1"] == pytest.approx(
            latest_row["sales_lag_1"], nan_ok=True
        )


def test_save_and_load_reference_artifacts_roundtrip(features_df, tmp_path):
    save_reference_artifacts(features_df, tmp_path)
    snapshot = load_reference_artifacts(tmp_path)
    assert len(snapshot) > 0
    assert "store_nbr" in snapshot.columns


def test_load_reference_artifacts_raises_when_missing(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_reference_artifacts(tmp_path / "does_not_exist")

from __future__ import annotations

import pytest
from demand_forecast.data.snapshot import (
    SNAPSHOT_COLUMNS,
    build_reference_snapshot,
    load_reference_artifacts,
    save_reference_artifacts,
)


def test_build_reference_snapshot_has_one_row_per_store_family(features_df):
    snapshot = build_reference_snapshot(features_df)
    n_groups = features_df.groupby(["store_nbr", "family"], observed=True).ngroups
    assert len(snapshot) == n_groups
    assert list(snapshot.columns) == SNAPSHOT_COLUMNS


def test_save_and_load_reference_artifacts_roundtrip(features_df, raw_tables, tmp_path):
    save_reference_artifacts(features_df, raw_tables["stores"], raw_tables["holidays"], tmp_path)
    snapshot, holidays = load_reference_artifacts(tmp_path)
    assert len(snapshot) > 0
    assert "date" in holidays.columns


def test_load_reference_artifacts_raises_when_missing(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_reference_artifacts(tmp_path / "does_not_exist")

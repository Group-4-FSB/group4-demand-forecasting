from __future__ import annotations

from demand_forecast.models.train import sample_param_grid, time_series_cv_splits


def test_time_series_cv_splits_are_chronological_and_disjoint(features_df):
    splits = time_series_cv_splits(features_df["date"], n_splits=2)
    assert len(splits) == 2
    for train_idx, valid_idx in splits:
        assert len(set(train_idx) & set(valid_idx)) == 0
        train_dates = features_df["date"].iloc[train_idx]
        valid_dates = features_df["date"].iloc[valid_idx]
        # every validation date must come after every training date (rolling-origin)
        assert train_dates.max() < valid_dates.min()


def test_sample_param_grid_respects_n_samples():
    grid = {"a": [1, 2, 3], "b": [10, 20]}
    combos = sample_param_grid(grid, n_samples=4, seed=0)
    assert len(combos) == 4
    for combo in combos:
        assert set(combo.keys()) == {"a", "b"}


def test_sample_param_grid_is_deterministic_given_seed():
    grid = {"a": [1, 2, 3], "b": [10, 20]}
    combos_1 = sample_param_grid(grid, n_samples=3, seed=42)
    combos_2 = sample_param_grid(grid, n_samples=3, seed=42)
    assert combos_1 == combos_2


def test_sample_param_grid_caps_at_full_grid_size():
    grid = {"a": [1, 2]}
    combos = sample_param_grid(grid, n_samples=100, seed=0)
    assert len(combos) == 2  # only 2 possible combos exist

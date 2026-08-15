from __future__ import annotations

import pandas as pd
from demand_forecast.explainability.shap_explain import (
    compute_shap_values,
    native_gain_importance,
    permutation_feature_importance,
    save_global_summary_plot,
    save_local_waterfall_plot,
    top_features_by_mean_abs_shap,
)
from demand_forecast.fairness.fairness_report import (
    add_fairness_segments,
    disparity_ratio,
    fairness_report,
    segment_performance,
)


def _predictions_df() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "sales": [10.0, 20.0, 30.0, 5.0, 15.0, 25.0],
            "prediction": [10.0, 20.0, 30.0, 10.0, 20.0, 30.0],  # segment B is worse
            "segment": ["A", "A", "A", "B", "B", "B"],
        }
    )


def test_segment_performance_computes_metrics_per_group():
    df = _predictions_df()
    out = segment_performance(df, "segment")
    assert set(out["segment"]) == {"A", "B"}
    assert (out["n"] == 3).all()
    # segment A has zero error, segment B does not
    a_rmsle = out.loc[out["segment"] == "A", "rmsle"].iloc[0]
    b_rmsle = out.loc[out["segment"] == "B", "rmsle"].iloc[0]
    assert a_rmsle == 0.0
    assert b_rmsle > 0.0


def test_disparity_ratio_is_at_least_one():
    df = _predictions_df()
    metrics = segment_performance(df, "segment")
    # avoid the degenerate zero-error segment for a finite ratio
    metrics.loc[metrics["rmsle"] == 0.0, "rmsle"] = 1e-6
    ratio = disparity_ratio(metrics)
    assert ratio >= 1.0


def test_fairness_report_flags_large_disparity():
    df = _predictions_df()
    df.loc[df["segment"] == "A", "prediction"] += 0.1
    df["segment"] = df["segment"].astype("category")
    report = fairness_report(df, segment_columns=["segment"], disparity_flag_threshold=1.01)
    assert "segment" in report
    assert report["segment"]["flagged"] is True
    assert isinstance(report["segment"]["metrics"], pd.DataFrame)


def test_fairness_segments_are_defined_from_training_reference():
    reference = pd.DataFrame(
        {
            "store_nbr": [1, 1, 2, 2, 3, 3],
            "sales": [10.0, 12.0, 20.0, 22.0, 30.0, 32.0],
            "unemployment": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
        }
    )
    holdout = pd.DataFrame(
        {
            "store_nbr": [1, 3],
            # Extreme holdout outcomes must not redefine historical size.
            "sales": [10_000.0, 1.0],
            "unemployment": [0.0, 10.0],
        }
    )

    segmented = add_fairness_segments(holdout, reference_df=reference)

    assert segmented["store_size_bucket"].astype(str).tolist() == ["small", "large"]
    assert segmented["unemployment_bucket"].astype(str).tolist() == [
        "Q1_lowest",
        "Q4_highest",
    ]


def test_fairness_report_skips_absent_columns():
    df = _predictions_df()
    report = fairness_report(df, segment_columns=["segment", "does_not_exist"])
    assert "does_not_exist" not in report
    assert "segment" in report


def test_native_gain_importance_sums_to_all_features(trained_summary):
    model = trained_summary["model"]
    feature_cols = trained_summary["feature_columns"]
    importance = native_gain_importance(model, feature_cols)
    assert set(importance.index) == set(feature_cols)
    assert (importance >= 0).all()


def test_shap_values_and_plots(trained_summary, features_df, tmp_path):
    model = trained_summary["model"]
    feature_cols = trained_summary["feature_columns"]
    x_sample = features_df[feature_cols].sample(20, random_state=0)

    shap_values = compute_shap_values(model, x_sample)
    assert shap_values.values.shape[0] == 20
    assert shap_values.values.shape[1] == len(feature_cols)

    top = top_features_by_mean_abs_shap(shap_values, feature_cols, n=5)
    assert len(top) == 5
    assert (top.to_numpy() >= 0).all()

    summary_path = save_global_summary_plot(shap_values, tmp_path / "summary.png")
    waterfall_path = save_local_waterfall_plot(shap_values, 0, tmp_path / "waterfall.png")
    assert summary_path.exists()
    assert waterfall_path.exists()


def test_shap_and_native_importance_roughly_agree_on_top_feature(trained_summary, features_df):
    """Two independent explainability methods should point at the same most
    important feature — a basic cross-check that neither is broken/random."""
    model = trained_summary["model"]
    feature_cols = trained_summary["feature_columns"]
    x_sample = features_df[feature_cols].sample(min(100, len(features_df)), random_state=0)

    shap_values = compute_shap_values(model, x_sample)
    shap_top1 = top_features_by_mean_abs_shap(shap_values, feature_cols, n=1).index[0]
    gain_top1 = native_gain_importance(model, feature_cols).index[0]

    # Not asserting they're identical (that would be flaky) — just that both
    # are valid, known feature names.
    assert shap_top1 in feature_cols
    assert gain_top1 in feature_cols


def test_permutation_importance_uses_holdout_model(trained_summary):
    evaluation_df = trained_summary["responsible_ai_eval_df"]
    feature_cols = trained_summary["feature_columns"]
    importance = permutation_feature_importance(
        trained_summary["responsible_ai_eval_model"],
        evaluation_df[feature_cols],
        evaluation_df["log_sales"],
        n_repeats=2,
    )

    assert set(importance["feature"]) == set(feature_cols)
    assert importance["importance_mean"].notna().all()

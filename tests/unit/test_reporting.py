from __future__ import annotations

from demand_forecast.reporting import generate_responsible_ai_report


def test_generate_responsible_ai_report_writes_expected_artifacts(
    trained_summary, features_df, tmp_path
):
    result = generate_responsible_ai_report(
        model=trained_summary["model"],
        df=features_df,
        feature_cols=trained_summary["feature_columns"],
        output_dir=tmp_path,
        mlflow_run_id=trained_summary["run_id"],
        shap_sample_size=30,
    )

    assert result["fairness_report_path"].exists()
    assert result["shap_summary_path"].exists()
    assert result["shap_waterfall_path"].exists()
    assert (tmp_path / "native_gain_importance.csv").exists()
    assert (tmp_path / "shap_top_features.csv").exists()

    markdown = result["fairness_report_path"].read_text(encoding="utf-8")
    assert "Fairness Report" in markdown
    assert len(result["fairness_report"]) > 0


def test_generate_responsible_ai_report_without_mlflow_logging(
    trained_summary, features_df, tmp_path
):
    # mlflow_run_id=None should skip artifact logging without raising
    result = generate_responsible_ai_report(
        model=trained_summary["model"],
        df=features_df,
        feature_cols=trained_summary["feature_columns"],
        output_dir=tmp_path,
        mlflow_run_id=None,
        shap_sample_size=10,
    )
    assert result["fairness_report_path"].exists()

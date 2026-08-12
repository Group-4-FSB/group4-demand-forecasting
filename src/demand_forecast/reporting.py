"""Generate the Responsible AI report artifacts (fairness + explainability)
after training, and optionally attach them to the MLflow run. Kept separate
from `models/train.py` so training stays focused on model fitting."""

from __future__ import annotations

import logging
from pathlib import Path

import mlflow
import numpy as np
import pandas as pd

from demand_forecast.explainability.shap_explain import (
    compute_shap_values,
    native_gain_importance,
    save_global_summary_plot,
    save_local_waterfall_plot,
    top_features_by_mean_abs_shap,
)
from demand_forecast.fairness.fairness_report import DEFAULT_SEGMENT_COLUMNS, fairness_report

logger = logging.getLogger(__name__)


def _render_fairness_markdown(report: dict[str, dict]) -> str:
    lines = [
        "# Fairness Report\n",
        "Disparity ratio = worst-segment RMSLE / best-segment RMSLE " "(flagged if > 1.5x).\n",
    ]
    for segment, info in report.items():
        flag = "⚠️ FLAGGED" if info["flagged"] else "OK"
        lines.append(
            f"\n## Segment: `{segment}` — disparity ratio "
            f"{info['disparity_ratio']:.3f} ({flag})\n"
        )
        lines.append(info["metrics"].to_markdown(index=False))
        lines.append("")
    return "\n".join(lines)


def generate_responsible_ai_report(
    model,
    df: pd.DataFrame,
    feature_cols: list[str],
    output_dir: Path,
    mlflow_run_id: str | None = None,
    shap_sample_size: int = 500,
    random_seed: int = 42,
) -> dict:
    """Compute fairness + explainability artifacts and write them to output_dir.
    If mlflow_run_id is given, also log everything as artifacts on that run."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    df = df.copy()
    df["prediction"] = np.expm1(model.predict(df[feature_cols]))

    fair_report = fairness_report(df, segment_columns=DEFAULT_SEGMENT_COLUMNS)
    markdown = _render_fairness_markdown(fair_report)
    fairness_path = output_dir / "fairness_report.md"
    fairness_path.write_text(markdown, encoding="utf-8")
    logger.info("Wrote fairness report to %s", fairness_path)

    sample_n = min(shap_sample_size, len(df))
    x_sample = df[feature_cols].sample(sample_n, random_state=random_seed)
    shap_values = compute_shap_values(model, x_sample)
    shap_summary_path = save_global_summary_plot(shap_values, output_dir / "shap_summary.png")
    shap_waterfall_path = save_local_waterfall_plot(
        shap_values, 0, output_dir / "shap_waterfall_example.png"
    )
    gain_importance = native_gain_importance(model, feature_cols)
    gain_importance.to_csv(output_dir / "native_gain_importance.csv", header=["gain"])
    shap_top = top_features_by_mean_abs_shap(shap_values, feature_cols)
    shap_top.to_csv(output_dir / "shap_top_features.csv", header=["mean_abs_shap"])

    if mlflow_run_id:
        mlflow.set_tracking_uri(mlflow.get_tracking_uri())
        with mlflow.start_run(run_id=mlflow_run_id):
            mlflow.log_artifact(str(fairness_path), artifact_path="responsible_ai")
            mlflow.log_artifact(str(shap_summary_path), artifact_path="responsible_ai")
            mlflow.log_artifact(str(shap_waterfall_path), artifact_path="responsible_ai")
            for segment, info in fair_report.items():
                ratio = info["disparity_ratio"]
                if ratio == ratio:  # not NaN
                    mlflow.log_metric(f"fairness_disparity_ratio_{segment}", ratio)

    return {
        "fairness_report": fair_report,
        "fairness_report_path": fairness_path,
        "shap_summary_path": shap_summary_path,
        "shap_waterfall_path": shap_waterfall_path,
        "native_gain_importance": gain_importance,
        "shap_top_features": shap_top,
    }

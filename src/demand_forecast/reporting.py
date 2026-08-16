"""Generate the Responsible AI report artifacts (fairness + explainability)
after training, and optionally attach them to the MLflow run. Kept separate
from `models/train.py` so training stays focused on model fitting."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import mlflow
import numpy as np
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
    DEFAULT_SEGMENT_COLUMNS,
    add_fairness_segments,
    fairness_report,
)

logger = logging.getLogger(__name__)


def _render_fairness_markdown(
    report: dict[str, dict], evaluation_df: pd.DataFrame, threshold: float
) -> str:
    start_date = pd.to_datetime(evaluation_df["date"]).min().date()
    end_date = pd.to_datetime(evaluation_df["date"]).max().date()
    lines = [
        "# Fairness Report\n",
        "This report uses predictions from the chronological holdout model: "
        "the model was fitted only on earlier weeks and never trained on the "
        "rows evaluated below. Proxy-segment boundaries are also derived from "
        "the earlier training pool, not from holdout outcomes.\n",
        f"Evaluation window: **{start_date} to {end_date}** "
        f"({evaluation_df['date'].nunique()} weeks, {len(evaluation_df)} rows).\n",
        "Disparity ratio = worst-segment RMSLE / best-segment RMSLE "
        f"(operational review flag if > {threshold:.1f}x). This is subgroup "
        "performance monitoring, not demographic-parity evidence: the source "
        "dataset contains no protected attributes.\n",
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


def _flatten_fairness_metrics(report: dict[str, dict]) -> pd.DataFrame:
    """Return one machine-readable table across all segment dimensions."""
    tables = []
    for segment, info in report.items():
        table = info["metrics"].copy().rename(columns={segment: "segment_value"})
        table.insert(0, "segment_dimension", segment)
        table["disparity_ratio"] = info["disparity_ratio"]
        table["flagged"] = info["flagged"]
        tables.append(table)
    return pd.concat(tables, ignore_index=True) if tables else pd.DataFrame()


def _json_safe(value: Any) -> Any:
    if isinstance(value, pd.Timestamp | np.datetime64):
        return pd.Timestamp(value).isoformat()
    if value is None or pd.isna(value):
        return None
    if isinstance(value, np.generic):
        return value.item()
    return value


def generate_responsible_ai_report(
    model,
    df: pd.DataFrame,
    feature_cols: list[str],
    output_dir: Path,
    fairness_evaluation_df: pd.DataFrame,
    fairness_reference_df: pd.DataFrame,
    fairness_evaluation_model,
    mlflow_run_id: str | None = None,
    shap_sample_size: int = 500,
    random_seed: int = 42,
    disparity_flag_threshold: float = 1.5,
) -> dict:
    """Create auditable fairness and explainability artifacts.

    ``model`` is the final production model and is used for SHAP/native gain.
    Fairness and permutation importance use ``fairness_evaluation_model`` and
    its chronological holdout predictions so neither result is in-sample.
    If ``mlflow_run_id`` is given, every generated artifact and disparity
    metric is attached to that production run.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    evaluation_df = add_fairness_segments(
        fairness_evaluation_df, reference_df=fairness_reference_df
    )
    fair_report = fairness_report(
        evaluation_df,
        segment_columns=DEFAULT_SEGMENT_COLUMNS,
        disparity_flag_threshold=disparity_flag_threshold,
    )

    markdown = _render_fairness_markdown(
        fair_report, evaluation_df, threshold=disparity_flag_threshold
    )
    fairness_path = output_dir / "fairness_report.md"
    fairness_path.write_text(markdown, encoding="utf-8")
    logger.info("Wrote fairness report to %s", fairness_path)

    fairness_metrics_path = output_dir / "fairness_metrics.csv"
    _flatten_fairness_metrics(fair_report).to_csv(fairness_metrics_path, index=False)

    fairness_summary = {
        "evaluation_protocol": "chronological_holdout",
        "evaluation_start_date": pd.to_datetime(evaluation_df["date"]).min().date().isoformat(),
        "evaluation_end_date": pd.to_datetime(evaluation_df["date"]).max().date().isoformat(),
        "evaluation_weeks": int(evaluation_df["date"].nunique()),
        "evaluation_rows": len(evaluation_df),
        "disparity_flag_threshold": disparity_flag_threshold,
        "segments": {
            segment: {
                "disparity_ratio": info["disparity_ratio"],
                "flagged": info["flagged"],
            }
            for segment, info in fair_report.items()
        },
    }
    fairness_summary_path = output_dir / "fairness_summary.json"
    fairness_summary_path.write_text(
        json.dumps(fairness_summary, indent=2, allow_nan=False), encoding="utf-8"
    )

    sample_n = min(shap_sample_size, len(df))
    sampled_df = df.sample(sample_n, random_state=random_seed)
    x_sample = sampled_df[feature_cols]
    shap_values = compute_shap_values(model, x_sample)
    shap_summary_path = save_global_summary_plot(shap_values, output_dir / "shap_summary.png")

    # Make the local explanation actionable: explain the most recent holdout
    # prediction for the store with the worst holdout RMSLE, rather than an
    # arbitrary training row.
    store_metrics = fair_report["store_nbr"]["metrics"]
    worst_store = store_metrics.iloc[0]["store_nbr"]
    local_sample = (
        evaluation_df.loc[evaluation_df["store_nbr"] == worst_store].sort_values("date").tail(1)
    )
    local_shap_values = compute_shap_values(model, local_sample[feature_cols])
    shap_waterfall_path = save_local_waterfall_plot(
        local_shap_values, 0, output_dir / "shap_waterfall_example.png"
    )
    gain_importance = native_gain_importance(model, feature_cols)
    gain_path = output_dir / "native_gain_importance.csv"
    gain_importance.to_csv(gain_path, header=["gain"])
    shap_top = top_features_by_mean_abs_shap(shap_values, feature_cols)
    shap_top_path = output_dir / "shap_top_features.csv"
    shap_top.to_csv(shap_top_path, header=["mean_abs_shap"])

    permutation = permutation_feature_importance(
        fairness_evaluation_model,
        evaluation_df[feature_cols],
        evaluation_df["log_sales"],
        random_seed=random_seed,
    )
    permutation_path = output_dir / "permutation_importance_holdout.csv"
    permutation.to_csv(permutation_path, index=False)

    local_row = local_sample.iloc[0]
    predicted_log_sales = float(model.predict(local_sample[feature_cols])[0])
    local_example = {
        "selection_reason": "latest_holdout_prediction_for_worst_rmsle_store",
        "explanation_space": "log1p_weekly_sales",
        "store_nbr": _json_safe(local_row.get("store_nbr")),
        "date": _json_safe(local_row.get("date")),
        "actual_sales": _json_safe(local_row.get("sales")),
        "predicted_log1p_sales": predicted_log_sales,
        "predicted_sales": float(np.expm1(predicted_log_sales)),
        "features": {name: _json_safe(local_row[name]) for name in feature_cols},
    }
    local_example_path = output_dir / "shap_local_example.json"
    local_example_path.write_text(
        json.dumps(local_example, indent=2, allow_nan=False), encoding="utf-8"
    )

    artifact_paths = [
        fairness_path,
        fairness_metrics_path,
        fairness_summary_path,
        shap_summary_path,
        shap_waterfall_path,
        gain_path,
        shap_top_path,
        permutation_path,
        local_example_path,
    ]

    if mlflow_run_id:
        mlflow.set_tracking_uri(mlflow.get_tracking_uri())
        with mlflow.start_run(run_id=mlflow_run_id):
            for artifact_path in artifact_paths:
                mlflow.log_artifact(str(artifact_path), artifact_path="responsible_ai")
            for segment, info in fair_report.items():
                ratio = info["disparity_ratio"]
                if ratio == ratio:  # not NaN
                    mlflow.log_metric(f"fairness_disparity_ratio_{segment}", ratio)

    return {
        "fairness_report": fair_report,
        "fairness_report_path": fairness_path,
        "fairness_metrics_path": fairness_metrics_path,
        "fairness_summary_path": fairness_summary_path,
        "shap_summary_path": shap_summary_path,
        "shap_waterfall_path": shap_waterfall_path,
        "shap_local_example_path": local_example_path,
        "native_gain_importance": gain_importance,
        "shap_top_features": shap_top,
        "permutation_importance": permutation,
        "permutation_importance_path": permutation_path,
    }

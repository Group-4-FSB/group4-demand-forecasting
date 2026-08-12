"""Fairness / subgroup-performance analysis (Responsible AI §E).

This dataset has no demographic attributes (no customer-level data at all —
only store and product data), so classical protected-attribute fairness does
not apply. Instead we use the accepted proxy for retail forecasting systems:
does the model serve every *store segment* (type, cluster, city/state) with
comparable accuracy? A model that is much worse for small/rural stores would
systematically cause more stock-outs or overstock there — a real fairness-of-
outcome concern for those store operators and their customers.
"""

from __future__ import annotations

import pandas as pd

from demand_forecast.models.evaluate import evaluate_all

DEFAULT_SEGMENT_COLUMNS = ["store_type", "cluster", "city", "state"]


def segment_performance(
    df: pd.DataFrame, segment_col: str, y_true_col: str = "sales", y_pred_col: str = "prediction"
) -> pd.DataFrame:
    rows = []
    for segment_value, group in df.groupby(segment_col, observed=True):
        metrics = evaluate_all(group[y_true_col], group[y_pred_col])
        rows.append({segment_col: segment_value, "n": len(group), **metrics})
    return pd.DataFrame(rows).sort_values("rmsle", ascending=False).reset_index(drop=True)


def disparity_ratio(segment_metrics: pd.DataFrame, metric: str = "rmsle") -> float:
    """Worst-segment / best-segment error ratio — the core fairness signal.
    A ratio near 1.0 means uniform performance; the further above 1.0, the
    more disparate the model's quality is across that segment."""
    if segment_metrics.empty or segment_metrics[metric].min() <= 0:
        return float("nan")
    return float(segment_metrics[metric].max() / segment_metrics[metric].min())


def fairness_report(
    df: pd.DataFrame,
    segment_columns: list[str] = DEFAULT_SEGMENT_COLUMNS,
    y_true_col: str = "sales",
    y_pred_col: str = "prediction",
    disparity_flag_threshold: float = 1.5,
) -> dict[str, dict]:
    """Run subgroup performance + disparity analysis across every segment column
    present in `df`. Returns one entry per segment with its metrics table,
    disparity ratio, and whether it exceeds the flag threshold."""
    report: dict[str, dict] = {}
    for col in segment_columns:
        if col not in df.columns:
            continue
        metrics_df = segment_performance(df, col, y_true_col, y_pred_col)
        ratio = disparity_ratio(metrics_df)
        is_nan = ratio != ratio  # noqa: PLR0124 (NaN-safe check without importing math/numpy)
        report[col] = {
            "metrics": metrics_df,
            "disparity_ratio": ratio,
            "flagged": False if is_nan else bool(ratio > disparity_flag_threshold),
        }
    return report

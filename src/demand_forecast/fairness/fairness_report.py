"""Fairness / subgroup-performance analysis (Responsible AI §E).

This dataset has no demographic attributes (no customer-level data at all —
only store-week aggregates), so classical protected-attribute fairness does
not apply. Instead we use the accepted proxy for retail forecasting systems:
does the model serve every *store segment* with comparable accuracy? A model
that is much worse for small stores, or for weeks/regions under high
unemployment, would systematically cause more stock-outs or overstock for
those stores and their local shoppers — a real fairness-of-outcome concern.

Segments used (derived — this dataset ships no store-metadata file):
- `store_size_bucket`: tercile of each store's average weekly sales — a
  proxy for "flagship vs small" store, since the source data has no
  store-type/format field.
- `unemployment_bucket`: quartile of the regional unemployment rate at
  prediction time — a proxy for local economic conditions, since the data
  has no city/state/region field but CPI/unemployment vary meaningfully by
  store (see docs/RESPONSIBLE_AI.md).
- `store_nbr`: per-store breakdown, for spotting any individual outlier.
"""

from __future__ import annotations

import pandas as pd

from demand_forecast.models.evaluate import evaluate_all

DEFAULT_SEGMENT_COLUMNS = ["store_size_bucket", "unemployment_bucket", "store_nbr"]


def add_fairness_segments(df: pd.DataFrame) -> pd.DataFrame:
    """Derive the segment columns used by `fairness_report` from raw
    store_nbr/sales/unemployment — call this once before scoring."""
    df = df.copy()
    store_avg_sales = df.groupby("store_nbr", observed=True)["sales"].transform("mean")
    df["store_size_bucket"] = pd.qcut(store_avg_sales, q=3, labels=["small", "medium", "large"])
    df["unemployment_bucket"] = pd.qcut(
        df["unemployment"], q=4, labels=["Q1_lowest", "Q2", "Q3", "Q4_highest"]
    )
    return df


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

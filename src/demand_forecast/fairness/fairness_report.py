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

import numpy as np
import pandas as pd

from demand_forecast.models.evaluate import evaluate_all

DEFAULT_SEGMENT_COLUMNS = ["store_size_bucket", "unemployment_bucket", "store_nbr"]


def add_fairness_segments(
    df: pd.DataFrame, reference_df: pd.DataFrame | None = None
) -> pd.DataFrame:
    """Derive stable proxy segments for a fairness evaluation table.

    ``reference_df`` should contain only the historical training pool. Its
    distributions define the bucket boundaries, which are then applied to
    ``df`` (normally the chronological holdout). This prevents the held-out
    outcomes from influencing their own group definitions. Passing no
    reference preserves the convenient exploratory behaviour of deriving
    segments from ``df`` itself.
    """
    df = df.copy()
    reference = df if reference_df is None else reference_df

    # Rank before qcut so equal averages cannot collapse quantile boundaries.
    # Each store gets equal weight, irrespective of how many weekly rows it has.
    store_avg_sales = reference.groupby("store_nbr", observed=True)["sales"].mean()
    if len(store_avg_sales) >= 3:
        store_buckets = pd.qcut(
            store_avg_sales.rank(method="first"),
            q=3,
            labels=["small", "medium", "large"],
        ).astype(str)
    elif len(store_avg_sales) == 2:
        ordered = store_avg_sales.sort_values().index
        store_buckets = pd.Series({ordered[0]: "small", ordered[1]: "large"})
    else:
        store_buckets = pd.Series("medium", index=store_avg_sales.index)
    df["store_size_bucket"] = pd.Categorical(
        df["store_nbr"].map(store_buckets), categories=["small", "medium", "large"]
    )

    # Define socioeconomic buckets from training-only quartiles. Infinite
    # outer bounds keep later values outside the historical range classifiable.
    unemployment_edges = (
        reference["unemployment"].dropna().quantile([0.25, 0.5, 0.75]).drop_duplicates().to_numpy()
    )
    unemployment_values = df["unemployment"].to_numpy(dtype=float)
    bucket_ids = np.searchsorted(unemployment_edges, unemployment_values, side="right")
    bucket_labels = np.array(["Q1_lowest", "Q2", "Q3", "Q4_highest"], dtype=object)
    assigned = bucket_labels[bucket_ids]
    assigned[pd.isna(unemployment_values)] = None
    df["unemployment_bucket"] = pd.Categorical(
        assigned, categories=bucket_labels.tolist(), ordered=True
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

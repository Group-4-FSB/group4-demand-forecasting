"""Model explainability — two complementary methods (Responsible AI §E):

1. SHAP (TreeExplainer) — global summary (which features matter overall) and
   local waterfall (why THIS particular prediction was made).
2. LightGBM native gain importance — a fast, model-native sanity check that
   should broadly agree with SHAP's global ranking.

Two independent methods, one shared feature-engineered input, no extra
dependency beyond `shap` (already required for method 1).
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless-safe for containers / CI
import matplotlib.pyplot as plt
import pandas as pd
import shap


def compute_shap_values(model, x_sample: pd.DataFrame) -> shap.Explanation:
    """TreeExplainer works directly on LightGBM sklearn estimators / boosters."""
    explainer = shap.TreeExplainer(model)
    return explainer(x_sample)


def save_global_summary_plot(shap_values: shap.Explanation, output_path: Path) -> Path:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.figure()
    shap.summary_plot(shap_values, show=False, plot_size=(9, 6))
    plt.tight_layout()
    plt.savefig(output_path, dpi=120)
    plt.close()
    return output_path


def save_local_waterfall_plot(shap_values: shap.Explanation, index: int, output_path: Path) -> Path:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.figure()
    shap.plots.waterfall(shap_values[index], show=False)
    plt.tight_layout()
    plt.savefig(output_path, dpi=120)
    plt.close()
    return output_path


def native_gain_importance(model, feature_names: list[str]) -> pd.Series:
    """LightGBM's built-in split-gain importance — the second explainability method."""
    gains = model.booster_.feature_importance(importance_type="gain")
    return pd.Series(gains, index=feature_names).sort_values(ascending=False)


def top_features_by_mean_abs_shap(
    shap_values: shap.Explanation, feature_names: list[str], n: int = 10
) -> pd.Series:
    import numpy as np

    mean_abs = pd.Series(np.abs(shap_values.values).mean(axis=0), index=feature_names).sort_values(
        ascending=False
    )
    return mean_abs.head(n)

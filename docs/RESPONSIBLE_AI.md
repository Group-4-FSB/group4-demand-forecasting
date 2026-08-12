# Responsible AI

This document covers the four areas required by the DDM501 rubric: fairness,
explainability, data privacy, and ethics. Supporting code lives in
[`src/demand_forecast/fairness/`](../src/demand_forecast/fairness/) and
[`src/demand_forecast/explainability/`](../src/demand_forecast/explainability/).

## 1. Fairness analysis & bias detection

**Why the usual definition doesn't apply.** The dataset (Kaggle "Walmart
Sales") has no customer-level or demographic data at all — it is a single
table of store-week aggregates (`Store`, `Date`, `Weekly_Sales`,
`Holiday_Flag`, `Temperature`, `Fuel_Price`, `CPI`, `Unemployment`). There is
no protected attribute (gender, age, ethnicity, ...) to check disparate
impact against, and no store-metadata file (no city/state/format field).

**Our proxy definition.** We treat *store segments*, derived from the data
that is available, as the relevant subgroup (`fairness/fairness_report.py`,
`add_fairness_segments`):
- **`store_size_bucket`** — tercile of each store's average weekly sales
  ("small" / "medium" / "large"), a proxy for flagship-vs-small store since
  the data has no store-format field.
- **`unemployment_bucket`** — quartile of the regional unemployment rate at
  prediction time, a proxy for local economic conditions.
- **`store_nbr`** — per-store breakdown, to catch any individual outlier
  segment-level buckets would average away.

A model that is much worse for small stores, or for a specific store
regardless of its size bucket, causes real harm: more stock-outs (lost
sales, understaffing) or more overstock (wasted cost) concentrated on that
store's staff and local shoppers.

**Method** (`fairness/fairness_report.py`):
1. Score the full dataset, derive the three segment columns above.
2. Group by each segment column and compute RMSLE / MAE / RMSE per group
   (`segment_performance`).
3. Compute a **disparity ratio** = worst-segment RMSLE / best-segment RMSLE
   (`disparity_ratio`). A ratio close to 1.0 means uniform quality; we flag
   any segment column whose ratio exceeds **1.5x** (`fairness_report`,
   `disparity_flag_threshold`) for follow-up.

**Actual finding on the full dataset** (see `reports/fairness_report.md`
after running `python scripts/run_pipeline.py`):

| Segment | Disparity ratio | Flagged? |
|---|---|---|
| `store_size_bucket` | 1.06 | No |
| `unemployment_bucket` | 1.24 | No |
| `store_nbr` (per-store) | **2.22** | **⚠️ Yes** |

The coarse buckets look fine, but the per-store breakdown is not: individual
stores (e.g. store 18) have roughly **2.2x** the RMSLE of the best-served
store, even though they fall in the same size/unemployment bucket as
better-served stores. This means the coarse segments hide real, store-level
disparity — exactly the kind of finding this analysis exists to surface, and
a genuine limitation of a single global model at this dataset's scale (~143
weeks/store).

**Mitigation strategies:**
- Add store-specific features (e.g. a store-level historical volatility
  feature) so the shared model can express store-specific patterns instead
  of relying only on `store_nbr` as a bare categorical.
- Sample weighting during training to upweight consistently under-served
  stores instead of letting high-volume stores dominate the loss.
- As a last resort, a small per-store bias correction rather than a fully
  separate model, to keep the system simple.
- Re-run `fairness_report()` after each mitigation and track the disparity
  ratio in MLflow as a regression-tested metric.

## 2. Model explainability

Two independent, complementary methods are implemented
(`explainability/shap_explain.py`):

1. **SHAP (TreeExplainer)** — game-theoretic feature attribution.
   - *Global*: `save_global_summary_plot()` ranks features by mean absolute
     SHAP value across a sample of predictions — "what does the model rely on
     overall?"
   - *Local*: `save_local_waterfall_plot()` explains one specific prediction —
     "why did the model predict $X for store 20 this week?" — useful for a
     planner questioning a specific forecast.
2. **LightGBM native gain importance** (`native_gain_importance()`) — a fast,
   model-native cross-check. On the full dataset both methods agree on the
   same top-3 ranking: `store_nbr` dominates (each store has a very
   different baseline sales level — $260K to $2.1M/week), followed by
   `sales_lag_1` and `sales_roll_mean_4` (recent momentum). Two independent
   methods landing on the same ranking increases confidence the explanation
   isn't an artifact of one method.

Both are cheap enough to run as part of the standard training pipeline and
are logged as MLflow artifacts on the final run.

## 3. Data privacy considerations

- The dataset is **public, store-level, and anonymized** — it contains no
  personally identifiable information (no customer names, IDs, addresses, or
  transaction-level receipts). The finest grain is (date, store) weekly
  aggregate sales plus regional economic indicators.
- The system never collects or stores any additional personal data; the API
  only accepts store/date/economic-indicator fields.
- **If this system were extended** with customer-level or loyalty-card data
  (e.g. for personalized demand signals), it would need: explicit data
  minimization (only fields required for the model), anonymization/hashing of
  customer identifiers, access controls on raw data, and a documented
  retention policy — general applicable principles (e.g. GDPR-style purpose
  limitation), not addressed here because no such data is in scope.
- Model artifacts and logs (MLflow, Prometheus) contain only store
  identifiers and aggregate numbers — no PII risk from artifact leakage.

## 4. Ethical implications

- **Over/under-forecast harms are asymmetric.** Under-forecasting causes
  understaffing and stock-outs (lost revenue, worse customer experience,
  disproportionately affecting smaller stores with thinner staffing
  buffers). Over-forecasting causes overstaffing/overstock cost. RMSLE
  already penalizes relative error symmetrically in log-space, a reasonable
  default, but a real deployment should let planners tune the cost
  asymmetry per store.
- **Automation bias / human-in-the-loop.** The model's forecasts should
  support, not replace, human planners — especially for holiday weeks or
  economic shocks the model has not seen the like of before. The API exposes
  raw predictions with no artificial confidence theater; planners should
  treat large predicted swings as a prompt to double-check, not blind ground
  truth.
- **Fairness-related labor impact.** The model systematically under-serves
  certain individual stores (disparity ratio 2.22, §1) — that store's staff
  bear more manual correction workload than staff at better-served stores,
  an equity concern addressed by the fairness monitoring above, not just an
  accuracy one.
- **Environmental footprint.** Training uses a single lightweight gradient
  boosting model (seconds on the full ~6.4K-row dataset) rather than deep
  learning, keeping the compute/carbon footprint of routine retraining low —
  itself a deliberate, documented design trade-off (see
  [ARCHITECTURE.md](../ARCHITECTURE.md)).

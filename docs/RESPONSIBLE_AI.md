# Responsible AI

This document covers the four areas required by the DDM501 rubric: fairness,
explainability, data privacy, and ethics. Supporting code lives in
[`src/demand_forecast/fairness/`](../src/demand_forecast/fairness/) and
[`src/demand_forecast/explainability/`](../src/demand_forecast/explainability/).

## 1. Fairness analysis & bias detection

**Why the usual definition doesn't apply.** The dataset (Corporación Favorita
store sales) has no customer-level or demographic data at all — only
store attributes (city, state, type, cluster) and product `family`. There is
no protected attribute (gender, age, ethnicity, ...) to check disparate
impact against.

**Our proxy definition.** We treat *store segments* as the relevant subgroup:
does the model forecast demand equally well for every store type, cluster,
city/state, and product family? A model that is systematically worse for a
segment (e.g. small rural stores) causes real harm to that segment's
operators and shoppers — more stock-outs (lost sales, unhappy customers) or
more overstock (wasted inventory, spoilage for perishables).

**Method** (`fairness/fairness_report.py`):
1. Score the holdout set, join predictions back onto store/product metadata.
2. Group by each segment column and compute RMSLE / MAE / RMSE per group
   (`segment_performance`).
3. Compute a **disparity ratio** = worst-segment RMSLE / best-segment RMSLE
   (`disparity_ratio`). A ratio close to 1.0 means uniform quality; we flag
   any segment column whose ratio exceeds **1.5x** (`fairness_report`,
   `disparity_flag_threshold`) for follow-up.

**Mitigation strategies if a segment is flagged:**
- Add segment-specific features (e.g. store cluster interacts with holiday
  effects) so the shared model can express segment-specific patterns.
- Stratified sampling / sample weighting during training to upweight
  under-served segments instead of letting high-volume segments dominate the
  loss.
- As a last resort, train a small per-segment correction (bias term) rather
  than a fully separate model, to keep the system simple.
- Re-run `fairness_report()` after each mitigation and track the disparity
  ratio in MLflow as a regression-tested metric.

Run `python scripts/run_pipeline.py` then see the generated
`reports/fairness_report.md`/console output (produced by
`scripts/run_pipeline.py` at the end of training) for current numbers on the
full dataset.

## 2. Model explainability

Two independent, complementary methods are implemented
(`explainability/shap_explain.py`):

1. **SHAP (TreeExplainer)** — game-theoretic feature attribution.
   - *Global*: `save_global_summary_plot()` ranks features by mean absolute
     SHAP value across a sample of predictions — "what does the model rely on
     overall?"
   - *Local*: `save_local_waterfall_plot()` explains one specific prediction —
     "why did the model predict X units for store 44, family DAIRY, on this
     date?" — useful for an inventory planner questioning a specific forecast.
2. **LightGBM native gain importance** (`native_gain_importance()`) — a fast,
   model-native cross-check. In our runs it broadly agrees with SHAP's
   ranking (recent lag sales and calendar/promotion features dominate), which
   increases confidence that the explanations are not an artifact of one
   method.

Both are cheap enough to run as part of the standard training pipeline and
are logged as MLflow artifacts on the final run.

## 3. Data privacy considerations

- The dataset is **public, aggregated, and anonymized** — it contains no
  personally identifiable information (no customer names, IDs, addresses, or
  transaction-level receipts). The finest grain is (date, store, product
  family) daily aggregate sales.
- The system never collects or stores any additional personal data; the API
  only accepts store/product/date/promotion fields.
- **If this system were extended** with loyalty-card or customer-level data
  (e.g. for personalized demand signals), it would need: explicit data
  minimization (only fields required for the model), anonymization/hashing of
  customer identifiers, access controls on raw data, and a documented
  retention policy — general applicable principles (e.g. GDPR-style purpose
  limitation), not addressed here because no such data is in scope.
- Model artifacts and logs (MLflow, Prometheus) contain only store/product
  identifiers and aggregate numbers — no PII risk from artifact leakage.

## 4. Ethical implications

- **Over/under-forecast harms are asymmetric and store-dependent.**
  Under-forecasting causes stock-outs (lost revenue, customer dissatisfaction,
  disproportionately affecting smaller stores with thinner buffer stock).
  Over-forecasting causes overstock — for perishable families (`DAIRY`,
  `PRODUCE`) this means food waste, an environmental and cost concern. The
  RMSLE loss already penalizes relative error symmetrically in log-space,
  which is a reasonable default, but a real deployment should let inventory
  planners tune the cost asymmetry per product family.
- **Automation bias / human-in-the-loop.** The model's forecasts should
  support, not replace, human inventory planners — especially near holidays,
  promotions, or supply disruptions the model has not seen before (e.g. the
  2016 Ecuador earthquake present in this dataset). The API exposes raw
  predictions with no artificial confidence theater; planners should treat
  large predicted swings as a prompt to double-check, not blind ground truth.
- **Fairness-related labor impact.** If the model systematically under-serves
  smaller/rural stores (see §1), those stores' staff bear more manual
  correction workload than staff at flagship stores — an equity concern
  addressed by the fairness monitoring above, not just an accuracy one.
- **Environmental footprint.** Training uses a single lightweight gradient
  boosting model (seconds to ~1 minute on the full dataset) rather than deep
  learning, keeping the compute/carbon footprint of routine retraining low —
  itself a deliberate, documented design trade-off (see
  [ARCHITECTURE.md](../ARCHITECTURE.md)).

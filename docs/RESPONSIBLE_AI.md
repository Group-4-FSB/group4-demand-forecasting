# Responsible AI

This document covers four Responsible AI areas:
fairness, explainability, data privacy, and ethics. Supporting code lives in
[`src/demand_forecast/fairness/`](../src/demand_forecast/fairness/) and
[`src/demand_forecast/explainability/`](../src/demand_forecast/explainability/).

## Summary

This section is a quick summary. Detailed evidence and
implementation notes follow in Sections 1-6.

### Fairness

- Approach: fairness-of-outcome across store-level proxy segments, because the
  dataset has no demographic/protected-attribute fields.
- Segments: `store_size_bucket`, `unemployment_bucket`, `store_nbr`.
- Signal: disparity ratio = worst RMSLE / best RMSLE, flag when > 1.5.
- Evaluation protocol: predictions from the final 8-week chronological
  holdout, produced by a model trained only on earlier weeks. Segment
  boundaries are derived from that earlier training pool as well.
- Latest result: `store_size_bucket` 1.109 (OK), `unemployment_bucket` 1.119
  (OK), `store_nbr` 6.440 (FLAGGED; only 8 observations per store, so treat
  the magnitude as a review signal rather than a stable population estimate).
- Mitigation: store-specific features, weighting for under-served stores,
  then retrain and compare disparity before/after in MLflow.

### Explainability

- Methods: SHAP TreeExplainer (global + local), LightGBM native gain, and
  model-agnostic permutation importance on the chronological holdout.
- Latest evidence: all three methods agree that `store_nbr` and
  `sales_lag_1` are the two most important features. SHAP/native gain rank
  `sales_roll_mean_4` third; permutation importance ranks `sales_lag_52`
  third and `sales_roll_mean_4` fourth.
- Interpretation: agreement on the dominant signals improves confidence;
  disagreement lower in the ranking is retained rather than hidden.

### Privacy

- Data scope: public, store-level, anonymized; no customer PII.
- Operational controls: localhost developer scope, aggregate-only artifacts,
  no personal identifiers in MLflow/monitoring outputs.
- If scope expands to customer data: apply minimization, access control,
  identifier hashing, and retention policy.

### Ethics

- Key risks: stock-out/overstock harms, automation bias, uneven correction load
  across stores, and compute footprint.
- Position: human-in-the-loop for planning decisions, especially under shocks;
  treat large prediction swings as review triggers.

### Governance and Response

- Fairness flagged (>1.5): mitigation ticket + retrain comparison.
- Staleness alerts: verify scheduler/data freshness, trigger retrain.
- Model unavailable: restore serving path and validate production alias.
- Cadence: review fairness/explainability every production promotion and run a
  monthly retrospective on recurring under-served stores.

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
1. Hold out the most recent 8 weeks. Fit the evaluation model only on all
   earlier weeks, then predict the untouched holdout.
2. Define the store-size and unemployment bucket boundaries using only the
   earlier training pool, then apply those fixed boundaries to the holdout.
3. Group by each segment column and compute RMSLE / MAE / RMSE per group
   (`segment_performance`).
4. Compute a **disparity ratio** = worst-segment RMSLE / best-segment RMSLE
   (`disparity_ratio`). A ratio close to 1.0 means uniform quality; we flag
   any segment column whose ratio exceeds **1.5x** (`fairness_report`,
   `disparity_flag_threshold`) for follow-up.

**Actual finding on the chronological holdout** (2012-09-07 to 2012-10-26,
360 rows; see `reports/fairness_report.md` after running
`python scripts/run_pipeline.py`):

| Segment | Disparity ratio | Flagged? |
|---|---|---|
| `store_size_bucket` | 1.109 | No |
| `unemployment_bucket` | 1.119 | No |
| `store_nbr` (per-store) | **6.440** | **⚠️ Yes** |

The coarse buckets look similar, but the per-store breakdown is not: store 17
has holdout RMSLE 0.0927 versus 0.0144 for store 32. This means aggregation
can hide an individual-store failure and is a genuine limitation of one
global model. The magnitude must be interpreted cautiously: each store has
only 8 observations in this holdout, so a holiday or isolated shock can move
the ratio substantially. The correct response is investigation and repeated
measurement across later windows, not a claim that 6.440x is a permanent
property of either store.

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

Three complementary methods are implemented
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
3. **Permutation importance** (`permutation_feature_importance()`) — a
   model-agnostic cross-check that shuffles one feature at a time and measures
   degradation in log-target RMSE on the untouched chronological holdout.

All three are cheap enough to run as part of the standard training pipeline and
are logged as MLflow artifacts on the final run.

**Evidence from the latest run artifacts** (`reports/shap_top_features.csv`,
`reports/native_gain_importance.csv`, and
`reports/permutation_importance_holdout.csv`):

| Rank | SHAP (mean absolute value) | Native gain importance |
|---|---|---|
| 1 | `store_nbr` (0.2895) | `store_nbr` (7680.69) |
| 2 | `sales_lag_1` (0.1318) | `sales_lag_1` (2315.48) |
| 3 | `sales_roll_mean_4` (0.0498) | `sales_roll_mean_4` (1195.06) |

Permutation importance independently ranks `store_nbr` (0.3025) and
`sales_lag_1` (0.1541) first and second, followed by `sales_lag_52` (0.0788)
and `sales_roll_mean_4` (0.0635). Agreement on the top two is consistent with
store baselines and recent momentum; the third/fourth-place disagreement
shows why multiple explanation methods are reported.

The local waterfall intentionally explains the latest holdout prediction for
the worst-RMSLE store, rather than a convenient random row. Because the model
is trained on `log1p(sales)`, SHAP values in the waterfall are additive in
log-sales space; `reports/shap_local_example.json` records both the log-space
output and its dollar-scale inverse transform to prevent misinterpretation.

**Why not LIME.** The rubric allows SHAP, LIME, *or equivalent*. For this
tree model, TreeExplainer is efficient and stable, while held-out permutation
importance supplies the model-agnostic check LIME would otherwise provide.
Native gain is retained as a third, model-specific diagnostic.

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

**Operational controls currently applied in this project setup:**
- Access to local observability tools is developer-scoped (Docker Desktop /
  localhost environment), not internet-exposed by default.
- MLflow/Grafana are used for course project operations only; no customer or
  account identifiers are ingested into artifacts.
- Responsible-AI artifacts (`fairness_report.md`, SHAP plots, feature
  importance CSVs) are aggregate-level outputs and can be shared for review
  without personal-data redaction.

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
- **Fairness-related labor impact.** The current holdout flags substantial
  individual-store disparity (ratio 6.440x, §1) — the affected store's staff
  bear more manual correction workload than staff at better-served stores,
  an equity concern addressed by the fairness monitoring above, not just an
  accuracy one.
- **Environmental footprint.** Training uses a single lightweight gradient
  boosting model (seconds on the full ~6.4K-row dataset) rather than deep
  learning, keeping the compute/carbon footprint of routine retraining low —
  itself a deliberate, documented design trade-off (see
  [ARCHITECTURE.md](../ARCHITECTURE.md)).

## 5. Monitoring, governance, and response plan

To convert analysis into operational accountability, we use the following
minimum response playbook:

| Trigger | Threshold | Owner | Action | SLA |
|---|---|---|---|---|
| Fairness disparity flagged (`store_nbr`) | disparity ratio > 1.5 | ML engineer | Open mitigation ticket, test weighting/feature fix, retrain, compare before/after ratios | 7 days |
| Model staleness warning | `ProductionModelApproachingStaleness` alert | MLOps owner | Verify scheduler health and fresh raw data availability | 1 day |
| Model stale critical | `ProductionModelStale` alert | MLOps owner + model owner | Force retrain attempt, document gate outcome, escalate if repeatedly rejected | 1 day |
| Model unavailable | `ModelNotLoaded` alert or `/health` degraded | API owner | Restore serving path (reload model / restart API / verify registry alias) | 4 hours |

Review cadence:
- Fairness and explainability artifacts are reviewed on every successful
  production promotion. Rejected candidates do not replace the production
  report.
- A monthly retrospective checks whether recurring under-served stores remain
  the same and whether mitigation reduced disparity.

## 6. Reproducing the evidence

```bash
python scripts/setup_data.py
python scripts/run_pipeline.py
```

A promoted run creates and logs the following MLflow artifacts under
`responsible_ai/`:

- `fairness_report.md`, `fairness_metrics.csv`, `fairness_summary.json`
- `shap_summary.png`, `shap_waterfall_example.png`
- `shap_top_features.csv`, `shap_local_example.json`
- `native_gain_importance.csv`, `permutation_importance_holdout.csv`

The local `reports/` directory is gitignored because these outputs belong to
a particular model/data run. The corresponding MLflow run is the auditable
source that binds them to model parameters, metrics, code, and data
fingerprint.

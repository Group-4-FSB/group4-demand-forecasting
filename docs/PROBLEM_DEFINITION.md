# Problem Definition & Requirements

## 1. Problem statement

Walmart operates 45 stores whose weekly sales are influenced by seasonality,
major US holidays, and regional economic conditions (fuel price, consumer
price index, unemployment, weather). Store and regional planners need to
know **how much a given store will sell in an upcoming week** to plan
staffing, inventory, and local budgets — a decision that today leans on
manual, experience-based estimates that don't systematically account for
holiday-week spikes or shifting local economic conditions.

- **Under-forecasting** → understaffing / stock-outs → lost sales.
- **Over-forecasting** → overstaffing / overstock → wasted cost.

**This project builds an end-to-end ML system** that ingests historical
weekly store sales and economic indicators; trains and tracks a demand
forecasting model; and serves weekly forecasts through a REST API — with
monitoring, testing, and responsible-AI practices appropriate for a
production system.

**Dataset**: [Kaggle "Walmart
Sales"](https://www.kaggle.com/datasets/mikhail1681/walmart-sales) — 45
stores, weekly sales from 2010-02-05 to 2012-10-26 (143 weeks/store, 6,435
rows, no gaps). One row per (store, week): `Store`, `Date`, `Weekly_Sales`,
`Holiday_Flag`, `Temperature`, `Fuel_Price`, `CPI`, `Unemployment`. There is
no separate store-metadata, promotions, or product-category breakdown — this
is a single-file, store-level weekly panel, simpler than a multi-table daily
retail dataset, which keeps the pipeline lean while still exercising the
full ML-system lifecycle end to end.

## 2. User requirements & use cases

### Primary users
- **Store/regional planner** — wants a next-week (or short-horizon) sales
  forecast per store to plan staffing and inventory.
- **ML/platform engineer** (this course's grading perspective) — wants a
  reproducible, observable, tested system they can retrain and redeploy.

### Use cases
1. *As a planner*, I request a forecast for a specific store and week and
   get back a predicted sales figure I can act on.
2. *As a planner*, I optionally supply a forecast/estimate for that week's
   fuel price, CPI, unemployment, or whether it's a major-holiday week, if I
   have better information than the model's default (last-known-value)
   assumption.
3. *As a planner*, I submit a batch of (store, week) combinations (e.g. next
   month's full planning horizon) and get all forecasts back in one call.
4. *As an ML engineer*, I retrain the model on a schedule (or on-demand),
   compare it against previous experiments in MLflow, and promote a new
   model version to serving without changing any client code.
5. *As an on-call engineer*, I get alerted if the API goes down, prediction
   latency spikes, or the error rate climbs — via Grafana/Prometheus.

### Functional requirements
| # | Requirement | Priority |
|---|---|---|
| F1 | Predict sales for a single (store, week) request | Must |
| F2 | Predict sales for a batch of up to 500 requests in one call | Must |
| F3 | Reject requests for a store never seen in training, with a clear error | Must |
| F4 | Allow the caller to override holiday-week flag / economic indicators when a better estimate is available | Should |
| F5 | Track every training run (params, metrics, artifacts) in MLflow | Must |
| F6 | Expose explainability (why did the model predict this?) for at least one prediction | Must |
| F7 | Report subgroup (store segment) performance disparity | Must |
| F8 | Auto-generated interactive API docs (Swagger) | Should |
| F9 | One-command full-stack local deployment (`docker compose up`) | Should |

### Non-functional requirements
| # | Requirement | Priority |
|---|---|---|
| N1 | API must respond to a single prediction well under 1s (p95 target: 500ms) | Must |
| N2 | System must run on a single developer machine (no cloud dependency required) for grading | Must |
| N3 | All components containerized; reproducible via `docker-compose.yml` | Must |
| N4 | Test coverage ≥ 80% with automated CI on every push | Must |
| N5 | No PII is ever collected or stored (see [RESPONSIBLE_AI.md](RESPONSIBLE_AI.md)) | Must |
| N6 | Model retraining must not require API code changes (registry-based versioning) | Should |

## 3. Success metrics

| Level | Metric | Target |
|---|---|---|
| **Business** | Forecast usefulness for planning | Qualitative — a well-engineered, well-monitored **system** is the goal, not a leaderboard score (this dataset has no public competition/benchmark) |
| **Business** | No store segment systematically under-served | Disparity ratio (fairness report) < 1.5x across store-size and regional-economic segments |
| **System** | API availability | ≥ 99% during grading/demo window (tracked via Prometheus `up`) |
| **System** | API latency | p95 < 500ms per prediction (Grafana panel + alert rule) |
| **System** | Test coverage | ≥ 80% (enforced in CI via `--cov-fail-under=80`) |
| **Model** | RMSLE on a chronological held-out test set (primary) | Beat the naive previous-week-persistence baseline **on the same held-out weeks**, tracked in MLflow — achieved: **baseline 0.072 → held-out test RMSLE 0.043** (~40% improvement), on 8 weeks the model never trained on (see [README.md § Train / CV / Test split](../README.md#train--cv--test-split) for the full methodology) |
| **Model** | CV RMSLE (secondary, used only to pick hyperparameters) | 0.090, measured via 3-fold rolling-origin CV on the training pool |
| **Model** | MAE / RMSE | Tracked per training run for interpretability alongside RMSLE |

## 4. Scope & constraints

**In scope:**
- Weekly-granularity forecasts for the 45 stores in the dataset.
- A single global gradient-boosting model (LightGBM) — see
  [ARCHITECTURE.md](../ARCHITECTURE.md) for why this was chosen over
  per-store models, LSTM, or Prophet.
- Local/single-node deployment via Docker Compose (no Kubernetes, no cloud
  infra) — appropriate for a 4-week course project graded on a reviewer's
  machine.

**Out of scope (documented, not silently dropped):**
- Product-category / SKU-level forecasting — the source dataset is
  store-level only, with no product breakdown.
- Real-time/streaming feature updates (lag features and economic indicators
  use a last-known-value snapshot — see the serving trade-off in
  ARCHITECTURE.md).
- Multi-region cloud deployment, autoscaling, and A/B testing infrastructure.
- Forecasting for stores not present in the training data (the API
  explicitly rejects these rather than guessing).
- Alertmanager-based notification routing (alert *rules* are implemented and
  visible in Prometheus; routing to Slack/email/PagerDuty is a documented
  future extension, not required for this course's grading scope).

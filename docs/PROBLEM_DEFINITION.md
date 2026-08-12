# Problem Definition & Requirements

## 1. Problem statement

Corporación Favorita, a large Ecuadorian grocery retailer operating dozens of
stores across multiple regions, needs to know **how many units of each
product family will sell at each store, each day**, so that store and
regional planners can order the right amount of stock. Today this decision
leans heavily on manual, experience-based estimates that don't systematically
account for promotions, holidays, regional economic signals (oil price, a
major driver of Ecuador's economy), or store-specific seasonality.

- **Under-forecasting** → stock-outs → lost sales and dissatisfied customers.
- **Over-forecasting** → overstock → wasted inventory (especially costly for
  perishable families like `DAIRY` or `PRODUCE`) and tied-up working capital.

**This project builds an end-to-end ML system** that ingests historical
sales, store, promotion, holiday, and oil-price data; trains and tracks a
demand forecasting model; and serves next-day-style forecasts through a REST
API that an inventory planning tool (or a planner directly) can call — with
monitoring, testing, and responsible-AI practices appropriate for a
production system.

**Dataset**: [Kaggle "Store Sales - Time Series
Forecasting"](https://www.kaggle.com/competitions/store-sales-time-series-forecasting)
(Corporación Favorita) — 54 stores × 33 product families, daily sales from
2013-01-01 to 2017-08-15 (~3M rows), plus store metadata, national/regional/
local holidays, oil price, and promotion counts.

## 2. User requirements & use cases

### Primary users
- **Inventory planner** — wants a next-day (or short-horizon) sales forecast
  per store/product family to decide replenishment quantities.
- **ML/platform engineer** (this course's grading perspective) — wants a
  reproducible, observable, tested system they can retrain and redeploy.

### Use cases
1. *As a planner*, I request a forecast for a specific store, product family,
   and date (optionally with a planned promotion count) and get back a
   predicted unit-sales number I can act on.
2. *As a planner*, I submit a batch of (store, family, date) combinations
   (e.g. tomorrow's full replenishment list) and get all forecasts back in
   one call.
3. *As an ML engineer*, I retrain the model on a schedule (or on-demand),
   compare it against previous experiments in MLflow, and promote a new
   model version to serving without changing any client code.
4. *As an on-call engineer*, I get alerted if the API goes down, prediction
   latency spikes, or the error rate climbs — via Grafana/Prometheus.

### Functional requirements
| # | Requirement | Priority |
|---|---|---|
| F1 | Predict sales for a single (store, family, date, promotion) request | Must |
| F2 | Predict sales for a batch of up to 500 requests in one call | Must |
| F3 | Reject requests for store/family combinations never seen in training, with a clear error | Must |
| F4 | Track every training run (params, metrics, artifacts) in MLflow | Must |
| F5 | Expose explainability (why did the model predict this?) for at least one prediction | Must |
| F6 | Report subgroup (store segment) performance disparity | Must |
| F7 | Auto-generated interactive API docs (Swagger) | Should |
| F8 | One-command full-stack local deployment (`docker compose up`) | Should |

### Non-functional requirements
| # | Requirement | Priority |
|---|---|---|
| N1 | API must respond to a single prediction in well under 1s (p95 target: 500ms) | Must |
| N2 | System must run on a single developer machine (no cloud dependency required) for grading | Must |
| N3 | All components containerized; reproducible via `docker-compose.yml` | Must |
| N4 | Test coverage ≥ 80% with automated CI on every push | Must |
| N5 | No PII is ever collected or stored (see [RESPONSIBLE_AI.md](RESPONSIBLE_AI.md)) | Must |
| N6 | Model retraining must not require API code changes (registry-based versioning) | Should |

## 3. Success metrics

| Level | Metric | Target |
|---|---|---|
| **Business** | Forecast accuracy good enough to reduce manual override rate by planners | Qualitative — RMSLE competitive with the original Kaggle competition's top-quartile solutions (~0.4–0.5) *is not the goal*; our target is a well-engineered, well-monitored **system**, not a leaderboard score |
| **Business** | Stock-out / overstock signal | Disparity ratio (see fairness report) < 1.5x across store segments, so no segment is systematically under-served |
| **System** | API availability | ≥ 99% during grading/demo window (tracked via Prometheus `up`) |
| **System** | API latency | p95 < 500ms per prediction (Grafana panel + alert rule) |
| **System** | Test coverage | ≥ 80% (enforced in CI via `--cov-fail-under=80`) |
| **Model** | RMSLE (primary, matches Kaggle's own metric) | Beat the seasonal-naive baseline (last-week-same-weekday) tracked in MLflow |
| **Model** | MAE / RMSE | Tracked per training run for interpretability alongside RMSLE |

## 4. Scope & constraints

**In scope:**
- Daily granularity forecasts for the 54 stores × 33 families in the dataset.
- A single global gradient-boosting model (LightGBM) — see
  [ARCHITECTURE.md](../ARCHITECTURE.md) for why this was chosen over
  per-family models, LSTM, or Prophet.
- Local/single-node deployment via Docker Compose (no Kubernetes, no cloud
  infra) — appropriate for a 4-week course project graded on a reviewer's
  machine.

**Out of scope (documented, not silently dropped):**
- Real-time/streaming feature updates (lag features use a daily snapshot —
  see the serving trade-off in ARCHITECTURE.md).
- Multi-region cloud deployment, autoscaling, and A/B testing infrastructure.
- Forecasting for stores/products not present in the training data (the API
  explicitly rejects these rather than guessing).
- Alertmanager-based notification routing (alert *rules* are implemented and
  visible in Prometheus; routing to Slack/email/PagerDuty is a documented
  future extension, not required for this course's grading scope).

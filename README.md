# Demand Forecast — DDM501 Final Project

> End-to-end ML system that forecasts weekly sales per Walmart store, from
> raw data to a monitored production API. Built for DDM501 — *AI in
> Production: From Models to Systems*.

<!-- Replace <YOUR_GH_ORG>/<YOUR_GH_REPO> once pushed to GitHub -->
![CI](https://github.com/khanhtq2994/group4-demand-forecasting/actions/workflows/ci.yml/badge.svg)
![Python](https://img.shields.io/badge/python-%E2%89%A53.10-blue)
![License](https://img.shields.io/badge/license-MIT-green)
![Coverage](https://img.shields.io/badge/coverage-%E2%89%A596%25-brightgreen)

## Overview

### What this is

Predicts next-week sales for each of 45 Walmart stores using the Kaggle
["Walmart Sales"](https://www.kaggle.com/datasets/mikhail1681/walmart-sales)
dataset (6,435 rows, 45 stores × 143 weeks, 2010–2012 — weekly sales plus
holiday-week flag, temperature, fuel price, CPI, and unemployment). See
[docs/PROBLEM_DEFINITION.md](docs/PROBLEM_DEFINITION.md) for the full
business problem, requirements, and success metrics, and
[ARCHITECTURE.md](ARCHITECTURE.md) for the system design and trade-offs.

## Quick navigation

- [Overview](#overview)
- [Quick Start](#quick-start)
- [Architecture](#architecture)
- [Operations](#operations)
- [CI -> CD Local](docs/CI_CD_LOCAL.md)
- [Appendix](#appendix)

## Quick Start

### Quick start checklists

Use this section if you just want to run the system quickly.

### Local (fast iteration)

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements-dev.txt
pip install -e .
python scripts/setup_data.py
python scripts/run_pipeline.py
uvicorn demand_forecast.api.main:app --reload
```

API docs: http://localhost:8000/docs

### Docker (full stack)

```bash
docker compose up -d mlflow
docker compose run --rm trainer python scripts/run_pipeline.py
docker compose up --build
```

MLflow UI: http://localhost:5001

Notes:
- Host uses port `5001` for MLflow to avoid macOS AirPlay conflicts on `5000`.
- Containers still use `http://mlflow:5000` internally.

## Architecture

### Train / CV / Test split

Each run uses a chronological split (never random):

1. Rolling-origin CV on train+CV pool to select hyperparameters.
2. One-time evaluation on locked holdout test weeks for honest performance.
3. Refit on full data only after evaluation, then register/serve.

Details: `chronological_holdout_split()` and `train_and_log()` in
[`src/demand_forecast/models/train.py`](src/demand_forecast/models/train.py).

Latest full-dataset results (`python scripts/run_pipeline.py`):

| Metric | Value | Meaning |
|---|---|---|
| Baseline RMSLE (all data) | 0.1218 | naive previous-week persistence, whole dataset |
| CV RMSLE (train+CV pool) | 0.0903 | used only to pick hyperparameters |
| Baseline RMSLE (test weeks) | 0.0717 | naive baseline on the *same* 8 held-out weeks |
| **Held-out TEST RMSLE** | **0.0428** | model that never saw those 8 weeks — beats the same-weeks baseline by ~40% |

### Quality gate & scheduled retraining

Two safeguards keep production reliable:

1. Promotion gate: candidate model is promoted only when holdout TEST RMSLE is
   better than or equal to current production; otherwise it is rejected and
   `production` alias stays unchanged.
2. Staleness retrain: scheduler checks model age daily and triggers retrain
   attempts at `RETRAIN_MAX_AGE_DAYS` (default 7), with warning lead time via
   `RETRAIN_WARNING_LEAD_DAYS` (default 2).

Monitor and alerting:

- Metric: `demand_forecast_production_model_age_days`
- Alerts: `ProductionModelApproachingStaleness`, `ProductionModelStale`, `ModelNotLoaded`

Implementation references:

- [`src/demand_forecast/models/train.py`](src/demand_forecast/models/train.py)
- [`scripts/retrain_if_stale.py`](scripts/retrain_if_stale.py)
- [`monitoring/prometheus/alert_rules.yml`](monitoring/prometheus/alert_rules.yml)
- [`tests/model/test_quality_gate.py`](tests/model/test_quality_gate.py)

### Data lineage

`data/raw/Walmart_Sales.csv` has no version control of its own (gitignored —
too large/licensed to commit, and would only grow). To still be able to trace
a served model back to exactly which data snapshot produced it, every
training run computes a fingerprint of the raw file — SHA256 of its exact
bytes, row count, and `(min_date, max_date)` coverage — and logs it as tags
(`raw_data_sha256`, `raw_data_rows`, `raw_data_min_date`,
`raw_data_max_date`, `raw_data_n_stores`) on the two runs that matter most:
`holdout_test_evaluation` (the quality-gate decision) and
`best_lightgbm_final` (the model actually registered). This is distinct from
a run's own `start_time` — that answers *when training happened*; the
fingerprint answers *which data, current as of which week, it happened on*
— together with MLflow's own auto-logged `mlflow.source.git.commit` tag,
this pins the full (code, data, params) triple behind any served model.

Implementation reference:
[`compute_raw_data_fingerprint()`](src/demand_forecast/data/ingest.py) in
`data/ingest.py`, called from `run_training_pipeline()` in
[`scripts/run_pipeline.py`](scripts/run_pipeline.py) and threaded through to
`train_and_log()`.

## Operations

### Local Development (no Docker)

Fastest loop for iterating on the ML pipeline — everything runs directly on
your machine against a local `sqlite:///mlflow.db`.

If you only need the minimal happy-path commands, use [Quick start checklists](#quick-start-checklists) above.

```bash
# 1. Python venv + deps (Python >=3.10)
python3 -m venv .venv && source .venv/bin/activate
# Windows (PowerShell): py -m venv .venv ; .venv\Scripts\Activate.ps1
# Windows (CMD): py -m venv .venv && .venv\Scripts\activate.bat
pip install -r requirements-dev.txt
pip install -e .   # installs demand_forecast in editable mode (needed to import it)

# 2. Extract the dataset (zip provided with the assignment, project root)
python scripts/setup_data.py

# 3. Train (ingest -> validate -> features -> train/CV/tune -> held-out
#    test eval -> MLflow -> reference snapshot -> Responsible AI report)
python scripts/run_pipeline.py
# Trains in well under a minute on this dataset (~6.4K rows); prints
# baseline/CV/held-out-test RMSLE side by side (see "Train / CV / Test split").

# 4. Serve
uvicorn demand_forecast.api.main:app --reload
# -> http://localhost:8000/docs

# Optional: inspect experiments
mlflow ui --port 5001
# -> http://localhost:5001
```

### Docker (full stack — production-like)

Runs the API behind the same containers used for grading/demo: FastAPI +
MLflow (its own tracking server, not the local `mlflow.db` above) + a
`scheduler` container running the 7-day auto-retrain check (see
[Quality gate & scheduled retraining](#quality-gate--scheduled-retraining))
+ Prometheus + Grafana. Requires **Docker Desktop running**.

```bash
# 1. Start MLflow first and wait for it to report healthy before continuing
docker compose up -d mlflow
docker compose ps   # wait for STATUS = healthy

# 2. Register a model into Dockerized MLflow
docker compose run --rm trainer python scripts/run_pipeline.py

# 3. Bring up the rest of the stack
docker compose up --build
```

If you want to run training from your host Python instead of the Docker `trainer` service,
set `MLFLOW_TRACKING_URI` to the Dockerized MLflow host port (`5001`):

```bash
# bash
export MLFLOW_TRACKING_URI=http://localhost:5001
python scripts/run_pipeline.py
```

```powershell
# PowerShell
$env:MLFLOW_TRACKING_URI = "http://localhost:5001"
python scripts/run_pipeline.py
```

```bat
:: Windows CMD
set MLFLOW_TRACKING_URI=http://localhost:5001
python scripts/run_pipeline.py
```

# NOTE: 5001 is the host port (helps avoid macOS AirPlay conflicts on 5000).
# Containers still talk to MLflow internally via http://mlflow:5000.

> If port 8000 is already bound by a local `uvicorn` from the section above,
> stop it first (`Ctrl+C`) — otherwise the `api` container fails to start
> with "port is already allocated".

Test it (new terminal, keep step 3 running): `curl http://localhost:8000/health`,
then see [Example request](#example-request) below — same API either way.

| Service | URL |
|---|---|
| API (Swagger) | http://localhost:8000/docs |
| MLflow | http://localhost:5001 |
| `scheduler` | *(no UI)* — `docker logs demand-forecast-scheduler` |
| Prometheus | http://localhost:9090 (see `/alerts`) |
| Grafana | http://localhost:3000 (`admin` / `admin`) |

```bash
docker compose down       # stop, keep volumes
docker compose down -v    # stop and wipe volumes too
```

Full instructions, all service URLs, and troubleshooting:
[docs/USER_GUIDE.md](docs/USER_GUIDE.md).

### Example request

```bash
curl -X POST http://localhost:8000/api/v1/predict \
  -H "Content-Type: application/json" \
  -d '{"store_nbr": 1, "date": "2012-12-28"}'
# Windows (PowerShell, curl is aliased to Invoke-WebRequest there):
# Invoke-RestMethod -Method Post -Uri http://localhost:8000/api/v1/predict -ContentType "application/json" -Body '{"store_nbr": 1, "date": "2012-12-28"}'
```

```json
{
  "store_nbr": 1,
  "date": "2012-12-28",
  "predicted_sales": 1535863.08,
  "model_name": "demand-forecast-lgbm",
  "model_alias": "production"
}
```

`store_nbr` and `date` are the only required fields; `holiday_flag`,
`temperature`, `fuel_price`, `cpi`, and `unemployment` are optional
overrides — omitted ones default to the store's last known value (or a
rule-based holiday-week detector for `holiday_flag`). See `/docs` for the
full schema.

**Batch** (up to 500 items per call):

```bash
curl -X POST http://localhost:8000/api/v1/predict/batch \
  -H "Content-Type: application/json" \
  -d '{"items": [
        {"store_nbr": 1, "date": "2012-11-02"},
        {"store_nbr": 2, "date": "2012-11-09", "temperature": 55.2}
      ]}'
# Windows (PowerShell):
# Invoke-RestMethod -Method Post -Uri http://localhost:8000/api/v1/predict/batch -ContentType "application/json" -Body '{"items": [{"store_nbr": 1, "date": "2012-11-02"}, {"store_nbr": 2, "date": "2012-11-09", "temperature": 55.2}]}'
```

### Try the quality gate & staleness alert yourself

Both are timing-based by design (7-day retrain, 8-week test holdout), so
here's how to see them fire in seconds instead of days:

- **Force a gate rejection**: after a model is registered (§ above),
  artificially make it "unbeatable" so the *next* training run is guaranteed
  worse, then retrain:
  ```python
  import mlflow
  client = mlflow.MlflowClient()  # same MLFLOW_TRACKING_URI as your training run
  mv = client.get_model_version_by_alias("demand-forecast-lgbm", "production")
  client.log_metric(mv.run_id, "test_rmsle", 1e-9)
  ```
  Then `python scripts/run_pipeline.py` again — it prints
  `REJECTED by quality gate — production unchanged`, and the MLflow UI shows
  the new run tagged `quality_gate=fail`.
- **Trigger the staleness warning without waiting 5 days**: lower the
  threshold instead of the clock —
  `python scripts/retrain_if_stale.py --max-age-days 1 --warning-lead-days 1`
  logs the WARNING immediately against a freshly-registered model (age ~0
  already satisfies `age >= max-age-days - warning-lead-days` = 0). Drop
  `--max-age-days` to `0` instead to see it actually retrain.
- **See it in Grafana**: http://localhost:3000 → "Demand Forecast API"
  dashboard → "Production model age (days)" panel goes green→yellow→red at
  the 5/7-day thresholds; http://localhost:9090/alerts shows
  `ProductionModelApproachingStaleness` / `ProductionModelStale` /
  `ModelNotLoaded` alongside the original 5 rules. These reflect the real
  registration age of whatever's currently `production`, so they only move
  once that model is genuinely old (or after a real retrain).

> **Note:** the WARNING log line above is instant because it only depends on
> `--max-age-days`/`--warning-lead-days`, not on the model's actual age. The
> Prometheus/Grafana side reads `demand_forecast_production_model_age_days`,
> which is computed from the `production` model version's real MLflow
> registration timestamp — so to make *that* move without waiting days,
> you'd need to backdate it: connect to MLflow's SQLite backend
> (`mlflow.db` locally, or inside the `mlflow` container at
> `/mlflow/mlflow.db`) and `UPDATE model_versions SET creation_time = ?
> WHERE name = 'demand-forecast-lgbm' AND version = (SELECT version FROM
> registered_model_aliases WHERE name = 'demand-forecast-lgbm' AND alias =
> 'production')`, with `creation_time` in epoch-milliseconds N days in the
> past, then `docker compose restart api` so it re-reads the timestamp. Not
> included as a script here since it pokes MLflow's internal schema
> directly rather than any public API — treat it as a one-off dev/demo tool
> if you build it, not something to wire into the app.

### Comprehensive system and ML metrics (including custom metrics)

The project tracks metrics across four layers so operational health and model
quality are both observable.

```mermaid
flowchart LR
  subgraph Serving[Online Serving Path]
    Client[Client]
    API[FastAPI]
    Model[PredictionService + Model]
    Client --> API --> Model
  end

  subgraph Obs[Observability]
    Metrics[/metrics endpoint]
    Prom[Prometheus]
    Alerts[Alert Rules]
    Graf[Grafana]
  end

  subgraph Train[Offline Training and Governance]
    Pipeline[Training Pipeline]
    Eval[Baseline + CV + Held-out Test]
    Gate[Quality Gate]
    Fair[Fairness Report]
    MLflow[MLflow Registry]
  end

  API --> Metrics --> Prom
  Prom --> Alerts
  Prom --> Graf

  Pipeline --> Eval --> Gate --> MLflow
  Pipeline --> Fair
  MLflow -.production alias.-> Model

  Prom -.monitoring signal.-> Gate
  Fair -.responsible AI signal.-> Gate
```

| Layer | Metric | Type | Formula / definition | Threshold / alert | Action when breached |
|---|---|---|---|---|---|
| **Service availability** | `up{job="demand-forecast-api"}` | System | API scrape health from Prometheus | `APIDown`: `up == 0` for 1m | Check container/process health, logs, and network |
| **HTTP reliability** | HTTP 5xx error rate | System | $\frac{\sum rate(http\_requests\_total\{status=~"5.."\}[5m])}{\sum rate(http\_requests\_total[5m])}$ | `HighHTTPErrorRate` > 5% for 5m | Investigate stack traces and dependency failures |
| **Inference latency (p95)** | `demand_forecast_prediction_latency_seconds` | Custom ML-serving | Model-only latency histogram (excludes HTTP overhead) | `HighPredictionLatencyP95` > 0.5s for 5m | Profile predict path, model load state, and host resources |
| **Prediction failures** | `demand_forecast_prediction_errors_total{reason=*}` | Custom ML-serving | Failed predictions by reason (e.g., `unknown_store`) | `HighPredictionErrorRate` > 0.5 errors/sec for 5m | Validate request payloads and reference snapshot coverage |
| **Traffic health** | `demand_forecast_predictions_total{store_nbr=*}` | Custom ML-serving | Successful predictions served | `NoPredictionTraffic` = 0 rate for 15m | Verify upstream integration / client traffic |
| **Model readiness** | `demand_forecast_model_loaded` | Custom ML-serving | Gauge: 1 if serving model is loaded, else 0 | `ModelNotLoaded` = 0 for 2m | Train/register model and restart API |
| **Model freshness** | `demand_forecast_production_model_age_days` | Custom governance | Age of current `production` alias model in days | `ProductionModelApproachingStaleness`: 5-7 days; `ProductionModelStale`: >=7 days for 24h | Check scheduler health, retrain outcome, and quality gate rejects |
| **Primary model quality** | RMSLE | Offline ML | $RMSLE = \sqrt{\frac{1}{n}\sum_i (\log(1+\hat{y}_i)-\log(1+y_i))^2}$ | Compared against baseline and production RMSLE | Promote only if candidate <= current production |
| **Secondary model quality** | MAE, RMSE | Offline ML | $MAE = \frac{1}{n}\sum_i |\hat{y}_i-y_i|$, $RMSE = \sqrt{\frac{1}{n}\sum_i (\hat{y}_i-y_i)^2}$ | Tracked per run and segment | Diagnose magnitude vs. relative error behavior |
| **Fairness parity** | Disparity ratio by segment | Custom Responsible AI | $\frac{\max(RMSLE_{segment})}{\min(RMSLE_{segment})}$ across `store_size_bucket`, `unemployment_bucket`, `store_nbr` | Flag when > 1.5 | Open mitigation task (features/weighting), retrain, compare before/after |

Where these are implemented:

- Custom online metrics: `src/demand_forecast/api/metrics.py`
- Alert logic: `monitoring/prometheus/alert_rules.yml`
- Offline evaluation metrics: `src/demand_forecast/models/evaluate.py`
- Quality gate + promotion logic: `src/demand_forecast/models/train.py`
- Fairness disparity metrics: `src/demand_forecast/fairness/fairness_report.py`

### What's implemented

| Area | Highlights |
|---|---|
| **ML pipeline** | Validated ingestion (`data/ingest.py`, `data/validate.py`), causal feature engineering incl. rule-based holiday-week detection (`data/features.py`), naive baseline + LightGBM with time-series CV + hyperparameter search + a **true chronological held-out test set** (never trained on — see [Train / CV / Test split](#train--cv--test-split)), a **promotion quality gate** and a **7-day auto-retrain scheduler** (see [Quality gate & scheduled retraining](#quality-gate--scheduled-retraining)), full MLflow tracking & model registry (`models/train.py`) |
| **Deployment** | FastAPI (`/api/v1/predict`, `/api/v1/predict/batch`, `/health`, auto Swagger at `/docs`), multi-stage non-root Dockerfile, 5-service `docker-compose.yml` (API, MLflow, retrain `scheduler`, Prometheus, Grafana) |
| **Monitoring** | Auto HTTP metrics + custom ML metrics (prediction count/latency/value distribution, model loaded/version, **production model age**) via `prometheus-fastapi-instrumentator`; provisioned Grafana dashboard; 8 Prometheus alert rules incl. model-not-loaded and a day-5 "approaching staleness" warning ahead of the 7-day auto-retrain |
| **Testing & CI/CD** | 117 tests across unit / integration / data-quality / model-validation (`tests/`, incl. the quality gate and retrain scheduler), 96% coverage, GitHub Actions pipeline (lint → test → Docker build) |
| **Responsible AI** | Store-segment fairness/disparity analysis (flagged a real ~1.9x per-store disparity — see docs), SHAP + native gain-importance explainability, privacy & ethics discussion — [docs/RESPONSIBLE_AI.md](docs/RESPONSIBLE_AI.md) |
| **Docs** | This README, [ARCHITECTURE.md](ARCHITECTURE.md), [CONTRIBUTING.md](CONTRIBUTING.md), [docs/PROBLEM_DEFINITION.md](docs/PROBLEM_DEFINITION.md), [docs/USER_GUIDE.md](docs/USER_GUIDE.md), OpenAPI/Swagger at `/docs` |

## Appendix

### Project structure

```
demand-forecast/
├── src/demand_forecast/     # installable package: data, models, api, explainability, fairness
├── scripts/                 # setup_data.py, run_pipeline.py (CLI entrypoints)
├── tests/                   # unit / integration / data / model + fixtures
├── monitoring/               # Prometheus config + alert rules, Grafana provisioning + dashboard
├── docs/                    # PROBLEM_DEFINITION, RESPONSIBLE_AI, USER_GUIDE
├── notebooks/                # exploratory analysis
├── Dockerfile, Dockerfile.mlflow, docker-compose.yml
└── .github/workflows/ci.yml
```

### Testing

```bash
pytest --cov=src --cov-report=term-missing   # unit + integration + data + model tests
ruff check src tests scripts                  # lint
black --check src tests scripts               # format check
```

### Team

See [CONTRIBUTING.md](CONTRIBUTING.md) for roles, git workflow, and the
individual-contribution tracking approach.

### License

[MIT](LICENSE)

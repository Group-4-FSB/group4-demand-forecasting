# Demand Forecast — DDM501 Final Project

> End-to-end ML system that forecasts weekly sales per Walmart store, from
> raw data to a monitored production API. Built for DDM501 — *AI in
> Production: From Models to Systems*.

<!-- Replace <YOUR_GH_ORG>/<YOUR_GH_REPO> once pushed to GitHub -->
![CI](https://github.com/<YOUR_GH_ORG>/<YOUR_GH_REPO>/actions/workflows/ci.yml/badge.svg)
![Python](https://img.shields.io/badge/python-3.10-blue)
![License](https://img.shields.io/badge/license-MIT-green)
![Coverage](https://img.shields.io/badge/coverage-%E2%89%A596%25-brightgreen)

## What this is

Predicts next-week sales for each of 45 Walmart stores using the Kaggle
["Walmart Sales"](https://www.kaggle.com/datasets/mikhail1681/walmart-sales)
dataset (6,435 rows, 45 stores × 143 weeks, 2010–2012 — weekly sales plus
holiday-week flag, temperature, fuel price, CPI, and unemployment). See
[docs/PROBLEM_DEFINITION.md](docs/PROBLEM_DEFINITION.md) for the full
business problem, requirements, and success metrics, and
[ARCHITECTURE.md](ARCHITECTURE.md) for the system design and trade-offs.

## Train / CV / Test split

Every training run does a **chronological, three-stage split** (never
random — see `chronological_holdout_split()` and `train_and_log()` in
[`models/train.py`](src/demand_forecast/models/train.py)) so the reported
accuracy is honest, not just a number from data the model already saw:

```
Week 1 (2010-02-05) ───────────────────────► Week 135 ─► Week 143 (2012-10-26)
│◄──────────── TRAIN + CV POOL (135 weeks) ─────────────►│◄─ TEST (8 weeks) ─►│
│   rolling-origin CV picks the best hyperparameters      │  locked away until │
│   (3 folds, same mechanism as before)                   │  the very last step│
```

1. **Pool** (all but the most recent `TEST_HOLDOUT_WEEKS`, default 8) feeds a
   3-fold rolling-origin `TimeSeriesSplit` to pick the best LightGBM
   hyperparameters — same CV mechanism the pipeline always had.
2. Those hyperparameters are fit on the **pool only** and scored **once** on
   the **test** weeks — data the model has never seen in any form. This is
   the one honest, apples-to-apples number to trust.
3. The same hyperparameters are then refit on **100% of the data** (pool +
   test) for the model that actually gets registered/served — recent weeks
   matter for a time series, so the deployed model shouldn't discard them
   once the honest test score above has been recorded.

**Real numbers from the full dataset** (`python scripts/run_pipeline.py`):

| Metric | Value | Meaning |
|---|---|---|
| Baseline RMSLE (all data) | 0.1218 | naive previous-week persistence, whole dataset |
| CV RMSLE (train+CV pool) | 0.0903 | used only to pick hyperparameters |
| Baseline RMSLE (test weeks) | 0.0717 | naive baseline on the *same* 8 held-out weeks |
| **Held-out TEST RMSLE** | **0.0428** | model that never saw those 8 weeks — beats the same-weeks baseline by ~40% |

## Quality gate & scheduled retraining

A held-out test score is only useful if it actually gates what gets served.
Two mechanisms sit on top of the split above (see `get_current_production_version()`,
the gate block in `train_and_log()` in
[`models/train.py`](src/demand_forecast/models/train.py), and
[`scripts/retrain_if_stale.py`](scripts/retrain_if_stale.py)):

**1. Promotion gate — a worse candidate never reaches `production`.** Every
run compares its own held-out TEST RMSLE against the TEST RMSLE the
*current* `production`-aliased model was registered with:

- **Better or equal → promoted.** Refit on 100% of data, registered, alias
  moved. (The very first run for a model name has nothing to compare
  against, so it always promotes.)
- **Worse → rejected.** Nothing is registered, the `production` alias is
  left untouched, and the attempt is tagged `quality_gate=fail` (vs. `pass`)
  on its `holdout_test_evaluation` MLflow run — searchable/filterable
  straight from the MLflow UI — plus a bordered warning in the pipeline log
  so a worse model can't quietly slip by unnoticed:
  ```
  ======================================================================
  QUALITY GATE FAILED — candidate NOT promoted to production
    candidate test_rmsle=0.0512  vs  current production test_rmsle=0.0428
    production alias left pointing at the existing model.
  ======================================================================
  ```
  `run_pipeline.py` reflects this in its exit summary too:
  `REJECTED by quality gate — production unchanged`.

Verified in [`tests/model/test_quality_gate.py`](tests/model/test_quality_gate.py)
(first-registration-always-promotes, worse-candidate-rejected-alias-unchanged,
tying-candidate-still-promotes, gate-disabled-when-`register_model=False`).

**2. A production model can't silently go stale.** `scripts/retrain_if_stale.py`
checks the age of the current `production` model (from its MLflow registration
timestamp) and forces a retrain attempt once it's at least `RETRAIN_MAX_AGE_DAYS`
(default **7**) days old, against whatever is currently in `data/raw/` — keeping
that folder fresh is an external data pipeline's job (out of scope here). Being
"due" only guarantees an *attempt*; the gate above still decides whether it's
actually promoted. Runs as its own `scheduler` container in
`docker-compose.yml` (reuses the `api` image, checks once a day):

```bash
# standalone / cron use
python scripts/retrain_if_stale.py --max-age-days 7
# or run continuously (what the Docker service does):
python scripts/retrain_if_stale.py --loop --check-interval-hours 24
```

Real log line from a fresh registration:
`Production model is 0.0 day(s) old (limit 7) — next check-in due in ~7.0 day(s). Nothing to do.`

**Heads-up before it happens, not just at the moment it does.** Starting
`RETRAIN_WARNING_LEAD_DAYS` (default **2**) days before the limit — i.e. from
day 5 for the 7-day default — every check logs a **WARNING** instead of the
usual INFO line:

```
Production model is 6.0 day(s) old (limit 7) — approaching staleness: only
~1.0 day(s) left before an automatic retrain is triggered.
```

The same signal doubles as a Prometheus gauge,
`demand_forecast_production_model_age_days` (scraped from the API, updated
live at scrape time — see `api/metrics.py`), with two matching alert rules in
[`monitoring/prometheus/alert_rules.yml`](monitoring/prometheus/alert_rules.yml):
`ProductionModelApproachingStaleness` (warning, day 5–7) and
`ProductionModelStale` (critical, still stale a full day past the limit —
the scheduler container is down, or every retrain keeps failing the gate). A
third new rule, `ModelNotLoaded`, pages if `/api/v1/predict` is serving 503s
because nothing is registered yet.

## Local Development (no Docker)

Fastest loop for iterating on the ML pipeline — everything runs directly on
your machine against a local `sqlite:///mlflow.db`.

```bash
# 1. Python 3.10 venv + deps
python3.10 -m venv .venv && source .venv/Scripts/activate
# Windows (PowerShell): py -3.10 -m venv .venv ; .venv\Scripts\Activate.ps1
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
mlflow ui
# -> http://localhost:5000
```

## Docker (full stack — production-like)

Runs the API behind the same containers used for grading/demo: FastAPI +
MLflow (its own tracking server, not the local `mlflow.db` above) + a
`scheduler` container running the 7-day auto-retrain check (see
[Quality gate & scheduled retraining](#quality-gate--scheduled-retraining))
+ Prometheus + Grafana. Requires **Docker Desktop running**.

```powershell
# 1. Start MLflow first and wait for it to report healthy
docker compose up -d mlflow
docker compose ps   # wait for STATUS = healthy before continuing

# 2. Point training at the Dockerized MLflow and register a model into IT
#    (it has its own tracking store, separate from the local one above —
#    the API container can only load a model that's registered here)
$env:MLFLOW_TRACKING_URI = "http://localhost:5000"        # PowerShell
# bash: export MLFLOW_TRACKING_URI=http://localhost:5000
python scripts/run_pipeline.py

# 3. Bring up the rest of the stack
docker compose up --build
```

> If port 8000 is already bound by a local `uvicorn` from the section above,
> stop it first (`Ctrl+C`) — otherwise the `api` container fails to start
> with "port is already allocated".

Test it (new terminal, keep step 3 running): `curl http://localhost:8000/health`,
then see [Example request](#example-request) below — same API either way.

| Service | URL |
|---|---|
| API (Swagger) | http://localhost:8000/docs |
| MLflow | http://localhost:5000 |
| `scheduler` | *(no UI)* — `docker logs demand-forecast-scheduler` |
| Prometheus | http://localhost:9090 (see `/alerts`) |
| Grafana | http://localhost:3000 (`admin` / `admin`) |

```bash
docker compose down       # stop, keep volumes
docker compose down -v    # stop and wipe volumes too
```

Full instructions, all service URLs, and troubleshooting:
[docs/USER_GUIDE.md](docs/USER_GUIDE.md).

## Example request

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

## Try the quality gate & staleness alert yourself

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
- **See the log-level WARNING immediately** (no infra needed): lower the
  threshold instead of the clock —
  `python scripts/retrain_if_stale.py --max-age-days 1 --warning-lead-days 1`
  logs the WARNING right away against a freshly-registered model (age ~0
  already satisfies `age >= max-age-days - warning-lead-days` = 0). This only
  affects that one process's log output, not what Prometheus/Grafana see.
- **See the actual Prometheus alert and Grafana panel move** — this needs
  the *model* to look old, not just the script's threshold, so there's a
  small dev-only helper for it,
  [`scripts/dev_simulate_stale_model.py`](scripts/dev_simulate_stale_model.py)
  (pokes MLflow's own SQLite store directly; not part of the production
  pipeline):
  ```bash
  docker cp scripts/dev_simulate_stale_model.py demand-forecast-mlflow:/tmp/sim.py
  docker exec demand-forecast-mlflow python /tmp/sim.py --age-days 6
  docker compose restart api   # API only reads the timestamp at startup
  ```
  Within one scrape (~15s) `demand_forecast_production_model_age_days` reads
  `6.0`, the "Production model age (days)" panel in Grafana
  (http://localhost:3000 → "Demand Forecast API") turns yellow, and
  `ProductionModelApproachingStaleness` shows up **pending** at
  http://localhost:9090/alerts — **firing** (red) once its `for: 10m` window
  elapses (verified end-to-end while building this, including watching it
  reach `firing`). `--age-days 8` crosses into `ProductionModelStale`
  instead. Reset with `--age-days 0`, or just retrain for a real timestamp.

## What's implemented

| Area | Highlights |
|---|---|
| **ML pipeline** | Validated ingestion (`data/ingest.py`, `data/validate.py`), causal feature engineering incl. rule-based holiday-week detection (`data/features.py`), naive baseline + LightGBM with time-series CV + hyperparameter search + a **true chronological held-out test set** (never trained on — see [Train / CV / Test split](#train--cv--test-split)), a **promotion quality gate** and a **7-day auto-retrain scheduler** (see [Quality gate & scheduled retraining](#quality-gate--scheduled-retraining)), full MLflow tracking & model registry (`models/train.py`) |
| **Deployment** | FastAPI (`/api/v1/predict`, `/api/v1/predict/batch`, `/health`, auto Swagger at `/docs`), multi-stage non-root Dockerfile, 5-service `docker-compose.yml` (API, MLflow, retrain `scheduler`, Prometheus, Grafana) |
| **Monitoring** | Auto HTTP metrics + custom ML metrics (prediction count/latency/value distribution, model loaded/version, **production model age**) via `prometheus-fastapi-instrumentator`; provisioned Grafana dashboard; 8 Prometheus alert rules incl. model-not-loaded and a day-5 "approaching staleness" warning ahead of the 7-day auto-retrain |
| **Testing & CI/CD** | 117 tests across unit / integration / data-quality / model-validation (`tests/`, incl. the quality gate and retrain scheduler), 96% coverage, GitHub Actions pipeline (lint → test → Docker build) |
| **Responsible AI** | Store-segment fairness/disparity analysis (flagged a real ~1.9x per-store disparity — see docs), SHAP + native gain-importance explainability, privacy & ethics discussion — [docs/RESPONSIBLE_AI.md](docs/RESPONSIBLE_AI.md) |
| **Docs** | This README, [ARCHITECTURE.md](ARCHITECTURE.md), [CONTRIBUTING.md](CONTRIBUTING.md), [docs/PROBLEM_DEFINITION.md](docs/PROBLEM_DEFINITION.md), [docs/USER_GUIDE.md](docs/USER_GUIDE.md), OpenAPI/Swagger at `/docs` |

## Project structure

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

## Testing

```bash
pytest --cov=src --cov-report=term-missing   # unit + integration + data + model tests
ruff check src tests scripts                  # lint
black --check src tests scripts               # format check
```

## Team

See [CONTRIBUTING.md](CONTRIBUTING.md) for roles, git workflow, and the
individual-contribution tracking approach.

## License

[MIT](LICENSE)

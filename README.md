# Demand Forecast — DDM501 Final Project

> End-to-end ML system that forecasts daily unit sales per (store, product
> family) for Corporación Favorita (Ecuador), from raw data to a monitored
> production API. Built for DDM501 — *AI in Production: From Models to
> Systems*.

<!-- Replace <YOUR_GH_ORG>/<YOUR_GH_REPO> once pushed to GitHub -->
![CI](https://github.com/<YOUR_GH_ORG>/<YOUR_GH_REPO>/actions/workflows/ci.yml/badge.svg)
![Python](https://img.shields.io/badge/python-3.10-blue)
![License](https://img.shields.io/badge/license-MIT-green)
![Coverage](https://img.shields.io/badge/coverage-%E2%89%A596%25-brightgreen)

## What this is

Predicts next-period unit sales for each (store, product family) combination
using the Kaggle ["Store Sales - Time Series
Forecasting"](https://www.kaggle.com/competitions/store-sales-time-series-forecasting)
dataset (~3M rows, 54 stores × 33 families, 2013–2017). See
[docs/PROBLEM_DEFINITION.md](docs/PROBLEM_DEFINITION.md) for the full
business problem, requirements, and success metrics, and
[ARCHITECTURE.md](ARCHITECTURE.md) for the system design and trade-offs.

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

# 3. Train (ingest -> validate -> features -> train/CV/tune -> MLflow ->
#    reference snapshot -> Responsible AI report)
python scripts/run_pipeline.py
# ~10-15 min on the full ~3M-row dataset; watch for "[i/6] cv_mean_rmsle=..."
# progress lines — a long silent gap right after it is expected, not a hang.

# 4. Serve
uvicorn demand_forecast.api.main:app --reload
# -> http://localhost:8000/docs

# Optional: inspect experiments
mlflow ui
# -> http://localhost:5000
```

## Docker (full stack — production-like)

Runs the API behind the same containers used for grading/demo: FastAPI +
MLflow (its own tracking server, not the local `mlflow.db` above) +
Prometheus + Grafana. Requires **Docker Desktop running**.

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
  -d '{"store_nbr": 1, "family": "GROCERY I", "date": "2017-08-20", "onpromotion": 5}'
# Windows (PowerShell, curl is aliased to Invoke-WebRequest there):
# Invoke-RestMethod -Method Post -Uri http://localhost:8000/api/v1/predict -ContentType "application/json" -Body '{"store_nbr": 1, "family": "GROCERY I", "date": "2017-08-20", "onpromotion": 5}'
```

```json
{
  "store_nbr": 1,
  "family": "GROCERY I",
  "date": "2017-08-20",
  "predicted_sales": 1042.37,
  "model_name": "demand-forecast-lgbm",
  "model_alias": "production"
}
```

## What's implemented

| Area | Highlights |
|---|---|
| **ML pipeline** | Validated ingestion (`data/ingest.py`, `data/validate.py`), causal feature engineering (`data/features.py`), seasonal-naive baseline + LightGBM with time-series cross-validation + hyperparameter search, full MLflow tracking & model registry (`models/train.py`) |
| **Deployment** | FastAPI (`/api/v1/predict`, `/api/v1/predict/batch`, `/health`, auto Swagger at `/docs`), multi-stage non-root Dockerfile, 4-service `docker-compose.yml` (API, MLflow, Prometheus, Grafana) |
| **Monitoring** | Auto HTTP metrics + custom ML metrics (prediction count/latency/value distribution, model version) via `prometheus-fastapi-instrumentator`; provisioned Grafana dashboard; 5 Prometheus alert rules |
| **Testing & CI/CD** | 65+ tests across unit / integration / data-quality / model-validation (`tests/`), 96% coverage, GitHub Actions pipeline (lint → test → Docker build) |
| **Responsible AI** | Store-segment fairness/disparity analysis, SHAP + native gain-importance explainability, privacy & ethics discussion — [docs/RESPONSIBLE_AI.md](docs/RESPONSIBLE_AI.md) |
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

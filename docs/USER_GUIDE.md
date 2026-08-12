# User Guide — Deployment & Operation

## Prerequisites

- Python **3.10** (project pins to this version — see `pyproject.toml`)
- Docker Desktop (for the full stack: API + MLflow + Prometheus + Grafana)
- The Kaggle dataset zip `walmart_sales.zip` in the project root (already
  provided with this assignment)

## 1. Local development (no Docker)

```bash
# 1. Create and activate a Python 3.10 virtual environment
python3.10 -m venv .venv
# Windows (PowerShell):
.venv\Scripts\Activate.ps1
# Windows (Git Bash) / macOS / Linux:
source .venv/Scripts/activate   # or .venv/bin/activate on macOS/Linux

# 2. Install dependencies (+ demand_forecast itself, in editable mode —
#    without this, `import demand_forecast` / `uvicorn demand_forecast...`
#    fails with ModuleNotFoundError since it lives under src/)
pip install -r requirements-dev.txt
pip install -e .

# 3. Extract the dataset
python scripts/setup_data.py

# 4. Run the full training pipeline (ingest -> validate -> features ->
#    train + tune + MLflow log -> reference snapshot -> Responsible AI report)
python scripts/run_pipeline.py

# 5. Inspect experiments
mlflow ui   # http://localhost:5000 (uses the local sqlite:///mlflow.db by default)

# 6. Run the API locally
uvicorn demand_forecast.api.main:app --reload --host 0.0.0.0 --port 8000
# Docs at http://localhost:8000/docs
```

`Makefile` wraps the common commands (`make setup`, `make data`, `make
train`, `make api`, `make test`) if you prefer.

## 2. Full stack via Docker Compose

Requires **Docker Desktop running** (`docker info` should succeed — if it
errors with `failed to connect to the docker API`, start Docker Desktop and
wait until it reports the engine is running before continuing).

This runs the API against its own Dockerized MLflow — a separate tracking
store from the local `sqlite:///mlflow.db` used in §1, so a model trained
locally is **not** automatically visible to the container; you register one
into the Dockerized MLflow explicitly in step 2 below.

```bash
# 1. Start MLflow first and wait for it to report healthy before continuing
docker compose up -d mlflow
docker compose ps   # wait for STATUS = healthy

# 2. Point training at the Dockerized MLflow and register a model into it
export MLFLOW_TRACKING_URI=http://localhost:5000
# Windows (PowerShell): $env:MLFLOW_TRACKING_URI = "http://localhost:5000"
python scripts/run_pipeline.py
# Trains in well under a minute on this dataset (~6.4K rows).

# 3. Bring up the rest of the stack (api, prometheus, grafana)
docker compose up --build
```

If port `8000` is already bound by a local `uvicorn` from §1, stop it first
(`Ctrl+C`) — otherwise the `api` container fails to start with "port is
already allocated".

| Service | URL | Notes |
|---|---|---|
| API (Swagger UI) | http://localhost:8000/docs | Interactive OpenAPI docs |
| API (health) | http://localhost:8000/health | Liveness/readiness |
| API (metrics) | http://localhost:8000/metrics | Raw Prometheus exposition format |
| MLflow | http://localhost:5000 | Experiments, runs, model registry |
| Prometheus | http://localhost:9090 | Query metrics, see `/alerts` for firing rules |
| Grafana | http://localhost:3000 | Login `admin` / `admin` (change for anything beyond local grading demo); dashboard "Demand Forecast API" is pre-provisioned |

### Example requests

```bash
curl -X POST http://localhost:8000/api/v1/predict \
  -H "Content-Type: application/json" \
  -d '{"store_nbr": 1, "date": "2012-12-28"}'

curl -X POST http://localhost:8000/api/v1/predict/batch \
  -H "Content-Type: application/json" \
  -d '{"items": [
        {"store_nbr": 1, "date": "2012-11-02"},
        {"store_nbr": 2, "date": "2012-11-09", "temperature": 55.2}
      ]}'
```

`store_nbr` and `date` are required; `holiday_flag`, `temperature`,
`fuel_price`, `cpi`, and `unemployment` are all optional — omitted fields
default to the store's last known value (economic indicators) or a
rule-based holiday-week detector (`holiday_flag`). See `/docs` for the full
schema.

### Shutting down

```bash
docker compose down       # stop containers, keep volumes (data persists)
docker compose down -v    # also remove volumes (MLflow/Prometheus/Grafana data)
```

## 3. Retraining / promoting a new model

1. Run `python scripts/run_pipeline.py` again (locally, or with
   `MLFLOW_TRACKING_URI` pointed at the running `mlflow` container as above).
   Each run logs a baseline + several tuned LightGBM candidates, and the best
   one is registered and aliased `production` in the model registry —
   automatically replacing the previous alias target.
2. **Restart the API container** so it picks up the newly-aliased model:
   `docker compose restart api` (the service loads the model once at
   startup — see the trade-offs note in ARCHITECTURE.md; a hot-reload
   endpoint is a natural next enhancement but out of scope here).
3. Compare runs in the MLflow UI before trusting a promotion — the pipeline
   always registers the best candidate from that run, but nothing stops you
   from manually re-aliasing an older, better-understood version via the
   MLflow UI or `MlflowClient.set_registered_model_alias(...)`.

## 4. Troubleshooting

| Symptom | Likely cause | Fix |
|---|---|---|
| `ModuleNotFoundError: No module named 'demand_forecast'` | The package under `src/` was never installed into the venv (`pip install -r requirements-dev.txt` alone does **not** do this) | Run `pip install -e .` once per venv (already included in `make setup` and §1 above) |
| `FileNotFoundError: Missing raw file ...` | Dataset not extracted yet | `python scripts/setup_data.py` |
| API `/health` returns `"status": "degraded"` | No model has been trained/registered yet, or `data/processed/` is empty | Run `python scripts/run_pipeline.py`, then restart the API |
| `libgomp.so.1: cannot open shared object file` (only if you modify the Dockerfile and drop the apt step) | LightGBM's OpenMP runtime dependency missing from the image | Keep the `libgomp1` install step in `Dockerfile` |
| MLflow artifact download fails from the API container | MLflow server started without `--serve-artifacts` | Use the provided `Dockerfile.mlflow` command as-is, or ensure `--serve-artifacts --artifacts-destination ...` are both set |
| `404 Not Found` on `/api/v1/predict` for a store number that looks valid | That `store_nbr` wasn't in the training data used to build the reference snapshot (valid range is 1-45, but only stores actually present in `data/raw/` get a snapshot entry) | Confirm the store exists in `Walmart_Sales.csv`; this is intentional — the API refuses to guess for unseen stores |
| Port already in use (`8000`/`5000`/`9090`/`3000`) | Another process/compose stack is already bound | Stop the conflicting process, or change the left-hand port in `docker-compose.yml`'s `ports:` mapping |
| Tests are slow / hang | Full dataset accidentally used instead of `tests/fixtures/` | Tests should only ever read from `tests/fixtures/` (see `tests/conftest.py`) — check you haven't changed `raw_dir` in a fixture |
| `run_pipeline.py` seems to pause briefly after `"Will assume non-transactional DDL"` | Normal — that's the last log line before hyperparameter search/CV; on this dataset (~6.4K rows) the whole run finishes in well under a minute, so any pause here is brief | If it's genuinely frozen for minutes (0% CPU, no progress, and `mlflow.db` stays exclusively locked with no growth), kill the process and re-run — a real hang here usually means another process already had `mlflow.db` open (e.g. two training runs launched at once, or `mlflow ui` pointed at the same file) |

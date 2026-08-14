# User Guide — Deployment & Operation

## Prerequisites

- Python **>= 3.10** (project supports Python 3.10 and 3.11 — see `pyproject.toml`)
- Docker Desktop (for the full stack: API + MLflow + Prometheus + Grafana)
- The Kaggle dataset zip `walmart_sales.zip` in the project root (already
  provided with this assignment)

## 1. Local development (no Docker)

```bash
# 1. Create and activate a Python virtual environment
python3 -m venv .venv
# macOS / Linux:
source .venv/bin/activate
# Windows (PowerShell):
.venv\Scripts\Activate.ps1
# Windows (CMD):
.venv\Scripts\activate.bat
```
> **Note for Windows (PowerShell):** If script execution is blocked, run `Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope Process` first.

```bash
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
mlflow ui --port 5001   # http://localhost:5001 (uses local sqlite:///mlflow.db by default)

# 6. Run the API locally
uvicorn demand_forecast.api.main:app --reload --host 0.0.0.0 --port 8000
# Docs at http://localhost:8000/docs
```

`Makefile` wraps the common commands (`make setup`, `make data`, `make
train`, `make api`, `make test`) if you prefer (on Windows without native `make`,
run the `python`/`pip` commands directly or use Git Bash / WSL).

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
export MLFLOW_TRACKING_URI=http://localhost:5001
# Windows (PowerShell): $env:MLFLOW_TRACKING_URI = "http://localhost:5001"
# Windows (CMD): set MLFLOW_TRACKING_URI=http://localhost:5001
docker compose run --rm trainer python scripts/run_pipeline.py
# Trains in well under a minute on this dataset (~6.4K rows).

# NOTE: 5001 is host-to-container mapping (avoids macOS AirPlay conflicts on 5000).
# In the Docker network, services still use http://mlflow:5000 internally.

# 3. Bring up the rest of the stack (api, scheduler, prometheus, grafana)
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
| MLflow | http://localhost:5001 | Experiments, runs, model registry |
| `scheduler` | *(no exposed port)* | Background container; checks daily whether `production` is ≥ `RETRAIN_MAX_AGE_DAYS` old and, if so, triggers a retrain attempt — `docker logs demand-forecast-scheduler` to watch it. Logs a WARNING from `RETRAIN_WARNING_LEAD_DAYS` (default 2) days before that; same signal is scraped as `demand_forecast_production_model_age_days` and alertable in Prometheus/Grafana |
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
   Each run: holds out the most recent `TEST_HOLDOUT_WEEKS` (default 8) weeks,
   picks the best LightGBM hyperparameters via CV on everything before that,
   scores them once on the held-out weeks for an honest `test_rmsle`, then —
   **only if that `test_rmsle` is at least as good as the current
   `production` model's** — refits on 100% of the data and registers/aliases
   it `production`. A worse candidate is silently *not* registered; the
   pipeline exits with `REJECTED by quality gate — production unchanged` and
   the run is tagged `quality_gate=fail` in MLflow. See
   [README.md § Quality gate & scheduled retraining](../README.md#quality-gate--scheduled-retraining)
   for the full mechanism, and pass `--test-weeks N` to change the holdout size.
2. **Restart the API container** so it picks up the newly-aliased model:
   `docker compose restart api` (the service loads the model once at
   startup — see the trade-offs note in ARCHITECTURE.md; a hot-reload
   endpoint is a natural next enhancement but out of scope here). Not needed
   if the gate rejected the candidate — there's nothing new to load.
3. Compare runs in the MLflow UI before trusting a promotion — the pipeline
   only *auto*-promotes a strictly-better-or-tying candidate, but nothing
   stops you from manually re-aliasing an older, better-understood version
   via the MLflow UI or `MlflowClient.set_registered_model_alias(...)`.
4. **This also happens automatically.** The `scheduler` container (started
   as part of `docker compose up` in §2) checks once a day whether
   `production` is ≥ `RETRAIN_MAX_AGE_DAYS` (default 7) days old and, if so,
   runs step 1 for you against whatever is currently in `data/raw/` —
   keeping that folder fresh is assumed to be handled by an external data
   pipeline. A scheduler-triggered retrain still goes through the same gate
   above, so staleness alone can never push a worse model into production.
   One-off / cron use outside Docker: `python scripts/retrain_if_stale.py`.

## 4. Troubleshooting

| Symptom | Likely cause | Fix |
|---|---|---|
| `cannot be loaded because running scripts is disabled on this system` (Windows PowerShell) | PowerShell Script Execution Policy restricts running `.ps1` scripts | Run `Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope Process` in PowerShell |
| `make: command not found` (Windows) | Windows doesn't include `make` by default | Run the `python`/`pip` commands directly as shown in §1, or run `make` inside Git Bash / WSL / Chocolatey |
| `ModuleNotFoundError: No module named 'demand_forecast'` | The package under `src/` was never installed into the venv (`pip install -r requirements-dev.txt` alone does **not** do this) | Run `pip install -e .` once per venv (already included in `make setup` and §1 above) |
| `FileNotFoundError: Missing raw file ...` | Dataset not extracted yet | `python scripts/setup_data.py` |
| API `/health` returns `"status": "degraded"` | No model has been trained/registered yet, or `data/processed/` is empty | Run `python scripts/run_pipeline.py`, then restart the API |
| `libgomp.so.1: cannot open shared object file` (only if you modify the Dockerfile and drop the apt step) | LightGBM's OpenMP runtime dependency missing from the image | Keep the `libgomp1` install step in `Dockerfile` |
| MLflow artifact download fails from the API container | MLflow server started without `--serve-artifacts` | Use the provided `Dockerfile.mlflow` command as-is, or ensure `--serve-artifacts --artifacts-destination ...` are both set |
| `503 Model is not loaded` from `/api/v1/predict`, and the API container logs show `OSError: No such file or directory: '/E:/...'` (or any local host path) | The experiment was first created while `MLFLOW_TRACKING_URI` pointed at a **local** backend (e.g. `sqlite:///mlflow.db`) but training since then has been pointed at the **Dockerized** MLflow — the experiment's `artifact_location` was set once, at creation, to a path only the host machine can read; the API container can't reach it | Wipe and recreate: `docker compose down -v` (removes the `mlflow_data` volume), `docker compose up -d mlflow`, wait for healthy, then retrain with `MLFLOW_TRACKING_URI=http://localhost:5001` before `docker compose up --build`. Fixed going forward as of `_ensure_experiment()` in `train.py`, which now leaves `artifact_location` unset for remote (`http(s)://`) tracking URIs so the server manages its own `mlflow-artifacts:/` storage |
| `404 Not Found` on `/api/v1/predict` for a store number that looks valid | That `store_nbr` wasn't in the training data used to build the reference snapshot (valid range is 1-45, but only stores actually present in `data/raw/` get a snapshot entry) | Confirm the store exists in `Walmart_Sales.csv`; this is intentional — the API refuses to guess for unseen stores |
| Port already in use (`8000`/`5001`/`9090`/`3000`) | Another process/compose stack is already bound | Stop the conflicting process, or change the left-hand port in `docker-compose.yml`'s `ports:` mapping |
| Tests are slow / hang | Full dataset accidentally used instead of `tests/fixtures/` | Tests should only ever read from `tests/fixtures/` (see `tests/conftest.py`) — check you haven't changed `raw_dir` in a fixture |
| `run_pipeline.py` seems to pause briefly after `"Will assume non-transactional DDL"` | Normal — that's the last log line before hyperparameter search/CV; on this dataset (~6.4K rows) the whole run finishes in well under a minute, so any pause here is brief | If it's genuinely frozen for minutes (0% CPU, no progress, and `mlflow.db` stays exclusively locked with no growth), kill the process and re-run — a real hang here usually means another process already had `mlflow.db` open (e.g. two training runs launched at once, or `mlflow ui` pointed at the same file) |
| A retrain ran (manually or via `scheduler`) but the API still serves the old predictions | The API only loads the model once, at container startup — see step 2 in §3 | `docker compose restart api` |
| `scheduler` container logs show nothing happening for days | Expected if `production` is still fresh — it logs `Production model is N day(s) old ... Nothing to do.` on every check, switches to a WARNING once within `RETRAIN_WARNING_LEAD_DAYS` of the limit, and only retrains once the age crosses `RETRAIN_MAX_AGE_DAYS` | `docker logs demand-forecast-scheduler`; lower `RETRAIN_MAX_AGE_DAYS` (env var in `docker-compose.yml`) if you want to demo it sooner |
| `ProductionModelStale` firing in Prometheus even though `scheduler` looks fine | A retrain has been *attempted* (per the scheduler logs) but every attempt is being rejected by the quality gate — the age clock only resets on an actual promotion | Check MLflow runs tagged `quality_gate=fail`; either the data genuinely hasn't improved the model, or something about the raw data changed for the worse — this is the alert working as intended, not a bug |
| `docker compose up` hangs indefinitely at `Container ... Creating` (any service, not just `scheduler`) | A previous `docker`/`docker compose` CLI invocation was killed (e.g. Ctrl+C, a tool timeout) without the *process itself* exiting, and it's still holding a client connection the daemon is serialized behind | Find and kill the stale client processes (Windows: `Get-Process docker,docker-compose \| Stop-Process -Force`), then retry — Docker Desktop's engine itself is usually fine, it's the leaked CLI client that's stuck |

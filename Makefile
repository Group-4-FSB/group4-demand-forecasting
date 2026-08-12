.PHONY: setup data train test lint fmt api up down clean

PY ?= python3.10

setup:
	$(PY) -m venv .venv
	.venv/Scripts/pip install -r requirements-dev.txt
	.venv/Scripts/pip install -e .

data:
	$(PY) scripts/setup_data.py

train:
	$(PY) scripts/run_pipeline.py

test:
	pytest --cov=src --cov-report=term-missing

lint:
	ruff check src tests scripts
	black --check src tests scripts

fmt:
	ruff check --fix src tests scripts
	black src tests scripts

api:
	uvicorn demand_forecast.api.main:app --reload --host 0.0.0.0 --port 8000

up:
	docker compose up --build

down:
	docker compose down -v

clean:
	find . -type d -name "__pycache__" -prune -exec rm -rf {} \;
	rm -rf .pytest_cache htmlcov .coverage

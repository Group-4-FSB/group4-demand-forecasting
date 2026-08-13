.PHONY: setup data train test lint fmt api up down clean

PY ?= python3
VENV ?= .venv

ifeq ($(OS),Windows_NT)
    BIN = $(VENV)/Scripts
else
    BIN = $(VENV)/bin
endif

setup:
	$(PY) -m venv $(VENV)
	$(BIN)/pip install -r requirements-dev.txt
	$(BIN)/pip install -e .

data:
	$(BIN)/python scripts/setup_data.py

train:
	$(BIN)/python scripts/run_pipeline.py

test:
	$(BIN)/pytest --cov=src --cov-report=term-missing

lint:
	$(BIN)/ruff check src tests scripts
	$(BIN)/black --check src tests scripts

fmt:
	$(BIN)/ruff check --fix src tests scripts
	$(BIN)/black src tests scripts

api:
	$(BIN)/uvicorn demand_forecast.api.main:app --reload --host 0.0.0.0 --port 8000

up:
	docker compose up --build

down:
	docker compose down -v

clean:
	find . -type d -name "__pycache__" -prune -exec rm -rf {} \;
	rm -rf .pytest_cache htmlcov .coverage


# Contributing & Team Roles

## Team roles & responsibilities

> Fill in with your team's actual names/GitHub handles before submission —
> this table is what the instructor uses (alongside git history) for the
> ±20% individual contribution adjustment described in the assignment
> (§3.3 Individual contribution).

| Member | GitHub handle | Primary role | Key responsibilities |
|---|---|---|---|
| _Name 1_ | `@handle1` | ML Pipeline | Data ingestion/validation, feature engineering, training + MLflow, Responsible AI (fairness/SHAP) |
| _Name 2_ | `@handle2` | Serving & Deployment | FastAPI service, Dockerfile/docker-compose, API docs |
| _Name 3_ | `@handle3` | Monitoring & CI/CD | Prometheus/Grafana, alert rules, GitHub Actions pipeline |
| _Name 4_ | `@handle4` | Testing & Documentation | Test suite (unit/integration/data/model), README/ARCHITECTURE/USER_GUIDE, presentation deck |

Each member should also be prepared to answer questions specifically about
their area in the individual Q&A portion of grading.

## Git workflow

- `main` is the protected, always-deployable branch.
- Work on feature branches named `feature/<short-description>` or
  `fix/<short-description>` (e.g. `feature/fairness-report`).
- Open a pull request into `main`; at least one other team member reviews
  before merging. CI (lint + tests + Docker build) must pass.
- Prefer small, frequent, meaningful commits over one large commit at the
  end — commit frequency/quality/distribution is part of the individual
  contribution grading criteria.

### Commit message convention

Use a short imperative summary, optionally prefixed with a type:

```
feat: add SHAP waterfall explanation to prediction report
fix: correct RMSLE clipping for negative predictions
docs: add troubleshooting section to USER_GUIDE
test: add data-quality tests for referential integrity
chore: pin lightgbm version in requirements.txt
```

## Local development setup

See [docs/USER_GUIDE.md](docs/USER_GUIDE.md) for full setup instructions.
Quick version:

```bash
python3 -m venv .venv
# macOS / Linux: source .venv/bin/activate
# Windows (PowerShell): .venv\Scripts\Activate.ps1
# Windows (CMD): .venv\Scripts\activate.bat
pip install -r requirements-dev.txt
pip install -e .              # installs demand_forecast in editable mode (needed to import it)
python scripts/setup_data.py  # extracts the Kaggle dataset zip into data/raw/
pytest --cov=src              # run the full test suite before opening a PR
ruff check src tests scripts && black --check src tests scripts
```

## Before opening a pull request

- [ ] `pytest --cov=src --cov-report=term-missing` passes with ≥80% coverage
- [ ] `ruff check` and `black --check` pass
- [ ] New/changed behavior has a corresponding test (unit, integration,
      data-quality, or model-validation as appropriate)
- [ ] Docs updated if you changed the API, architecture, or setup steps

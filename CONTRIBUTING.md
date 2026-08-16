# Contributing & Team Roles

## Team roles & responsibilities

| Member | Roles | Key responsibilities |
|---|---|---|
| Lê Thị Kim Chi | Team lead | Project coordination, high-level architecture, deployment flow, presentation organization |
| Trương Quốc Khánh | Member| Initial codebase,  FastAPI service endpoints, model serving flow, Dockerfile/docker-compose integration |
| Trương Sỹ Quảng | Member | Data ingestion/validation, feature engineering, fairness/SHAP analysis|
| Nguyễn Viết Anh Minh | Monitoring, Testing & Documentation | training + MLflow,Prometheus/Grafana setup, alert rules, CI workflow support, test coverage |

### Shared team contributions

- Build and maintain the end-to-end MLOps workflow (data -> training -> registry -> serving -> monitoring).
- Review pull requests, improve code quality, and keep commit history meaningful.
- Coordinate documentation updates, testing activities, and regular group review sessions.
- Prepare demo materials, validate results, and align deliverables with the assignment rubric.

## Git workflow

- `main` is the protected, always-deployable branch.
- Work on feature branches named `feature/<short-description>` or
  `fix/<short-description>` (e.g. `feature/fairness-report`).
- Open a pull request into `main`; at least one other team member reviews
  before merging. CI (lint + tests + Docker build) must pass.
- After merge to `main`, CD runs on the project's local self-hosted GitHub
  runner (see `docs/CI_CD_LOCAL.md`) and deploys with Docker Compose on the
  local machine.
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

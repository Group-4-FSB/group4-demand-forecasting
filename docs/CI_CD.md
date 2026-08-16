# Full CI/CD Guide

This repository now has a full CI/CD path:

- CI: lint, format check, tests, coverage, Docker build validation.
- CD Staging: build/push images to GHCR, then deploy over SSH to staging.
- CD Production: build/push release-tagged images, then deploy over SSH to production.
- Rollback: manual rollback to any previously pushed image tags.

## 1. Workflow files

- CI: `.github/workflows/ci.yml`
- Staging deploy: `.github/workflows/cd-staging.yml`
- Production deploy: `.github/workflows/cd-production.yml`
- Rollback: `.github/workflows/rollback.yml`

## 2. What changed for deployability

`docker-compose.yml` now supports image overrides through env vars:

- `API_IMAGE` (default: `demand-forecast-api:latest`)
- `MLFLOW_IMAGE` (default: `demand-forecast-mlflow:latest`)

This allows CI/CD to deploy exact image tags without editing compose files on every release.

## 3. One-time GitHub setup (required)

### 3.1 Create environments

Create two GitHub Environments in the repo settings:

- `staging`
- `production`

Recommended:

- Add required reviewer approvals on `production`.
- Keep `staging` auto-approved.

### 3.2 Add repository/organization secrets

Set these secrets (Settings -> Secrets and variables -> Actions):

Staging:

- `STAGING_SSH_HOST`
- `STAGING_SSH_USER`
- `STAGING_SSH_KEY`
- `STAGING_APP_DIR`

Production:

- `PROD_SSH_HOST`
- `PROD_SSH_USER`
- `PROD_SSH_KEY`
- `PROD_APP_DIR`

Notes:

- `*_APP_DIR` is the absolute path on remote host where this repo is checked out and where `docker compose` will run.
- SSH key should be private key content for the deploy user.

### 3.3 Remote host prerequisites

Both staging and production servers must have:

- Docker + Docker Compose plugin
- Git
- Repo cloned at `*_APP_DIR`
- Ability to pull GHCR images for this repository

If repo/images are private, authenticate GHCR on remote host once:

```bash
echo "$GHCR_PAT" | docker login ghcr.io -u <github-user> --password-stdin
```

## 4. End-to-end flow

### 4.1 CI flow

Trigger:

- Push/PR to `main`, `master`, `develop`

Pipeline:

1. Lint (`ruff`) + format check (`black --check`)
2. Tests + coverage threshold
3. Build Docker images (no push)

### 4.2 CD Staging flow

Trigger:

- `cd-staging.yml` runs automatically when `CI` completes successfully on branch `main`.

Pipeline:

1. Checkout tested commit (`workflow_run.head_sha`)
2. Build + push two images to GHCR:
   - `ghcr.io/<owner>/demand-forecast-api:staging-<short_sha>`
   - `ghcr.io/<owner>/demand-forecast-mlflow:staging-<short_sha>`
3. SSH to staging host
4. Export `API_IMAGE` and `MLFLOW_IMAGE`
5. `docker compose pull` + `docker compose up -d --remove-orphans`
6. Run `/health` smoke check

### 4.3 CD Production flow

Trigger:

- Release published, or manual `workflow_dispatch` with `release_tag`

Pipeline:

1. Checkout release tag (example: `v1.2.3`)
2. Build + push two images:
   - `ghcr.io/<owner>/demand-forecast-api:v1.2.3`
   - `ghcr.io/<owner>/demand-forecast-mlflow:v1.2.3`
3. Deploy to `production` environment (approval gate if configured)
4. Pull + up with exact tags
5. `/health` smoke check

## 5. Rollback flow

Use `rollback.yml` manually from Actions tab.

Inputs:

- `target_environment`: `staging` or `production`
- `api_image_tag`: target API tag
- `mlflow_image_tag`: target MLflow tag

Pipeline:

1. Resolve full image references in GHCR
2. SSH to target host
3. Set `API_IMAGE` + `MLFLOW_IMAGE`
4. `docker compose pull` + `docker compose up -d`
5. Health check

Example rollback tags:

- `staging-abc123def456`
- `v1.2.3`

## 6. Recommended branch and release policy

1. Feature branches -> PR into `main`
2. CI must pass before merge
3. Merge into `main` auto deploys to staging
4. Validate staging
5. Create GitHub Release tag (`vX.Y.Z`) to deploy production
6. If incident occurs, run `rollback.yml`

## 7. Operational checks after deploy

1. API health endpoint: `/health`
2. API docs endpoint: `/docs`
3. Prometheus targets: `/targets`
4. Grafana dashboard loads
5. Optional: run one prediction request to verify inference path

## 8. Security and hardening next steps

Optional improvements for stricter enterprise-grade CI/CD:

- Add Trivy image scanning gate before deploy
- Sign images (cosign)
- Add SBOM generation and upload
- Add post-deploy integration test job against staging endpoint
- Add auto-rollback if latency/error SLO is breached

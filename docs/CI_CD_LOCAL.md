# CI -> CD Local (Self-hosted Runner)

This project supports the following flow:

1. Push source code to GitHub.
2. Run CI on GitHub-hosted runner.
3. If CI passes on `main`, run CD on your local machine via a self-hosted runner.

## 1. Workflow files

- CI: `.github/workflows/ci.yml`
- CD local: `.github/workflows/cd-local.yml`

## 2. How this CD works

`cd-local.yml` listens for the completion of the `CI` workflow.

- If CI is successful and the branch is `main`, job `deploy-local` runs.
- The job runs on `runs-on: self-hosted`, meaning the deployment commands are executed on your local machine (where the runner is installed).
- Deployment command:

```bash
docker compose up -d --build mlflow api scheduler prometheus grafana
```

- Then it checks `/health`. If `model_loaded=false`, it auto-runs bootstrap
   training and restarts the API:

```bash
docker compose run --rm trainer python scripts/run_pipeline.py
docker compose restart api
```

- Finally it verifies model-ready health and runs a prediction smoke test.

```bash
curl -X POST http://localhost:8000/api/v1/predict \
   -H "Content-Type: application/json" \
   -d '{"store_nbr": 1, "date": "2012-12-28"}'
```

If bootstrap training is rejected by the quality gate and no production model
exists yet, the CD job fails by design because the service is not ready to
serve predictions.

- Initial health endpoint check command:

```bash
curl http://localhost:8000/health
```

## 3. One-time setup: self-hosted runner on local

Do this once in your GitHub repo:

1. Open repository Settings -> Actions -> Runners.
2. Click New self-hosted runner.
3. Choose your local OS/architecture (for your machine: macOS).
4. Follow the exact commands GitHub provides:
   - Download runner package
   - Configure runner with repo URL + registration token
   - Start runner process
5. Keep the runner online whenever you want CD to run.

Important:

- The runner must stay connected to GitHub.
- Docker Desktop must be running on your machine.
- Port 8000 must be free for API health checks.

## 4. Daily operation (step-by-step)

1. Work on feature branch.
2. Open PR and verify CI passes.
3. Merge to `main`.
4. CI runs on GitHub.
5. After CI success, CD Local runs automatically on your local runner.
6. Verify:
   - API docs: `http://localhost:8000/docs`
   - MLflow: `http://localhost:5001`
   - Prometheus: `http://localhost:9090`
   - Grafana: `http://localhost:3000`

## 5. Notes and limitations

- This is local CD, not remote staging deployment.
- Your local machine acts as deploy target.
- If runner is offline, CD job will remain queued and deployment will not happen.
- If you need real team-shared staging, use a dedicated server and SSH-based deploy workflow.

- First-ever deployment can take longer because CD may need to train/register
   an initial model before prediction smoke tests pass.

## 6. Manual recovery commands (if you run outside CI/CD)

CD now bootstraps model loading automatically, but these commands are still
useful for manual operations/troubleshooting.

Run training when needed:

```bash
docker compose run --rm trainer python scripts/run_pipeline.py
docker compose restart api
```

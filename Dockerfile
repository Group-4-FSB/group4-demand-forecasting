# syntax=docker/dockerfile:1

# ---- Builder stage: resolve & install Python deps in isolation ----
FROM python:3.10-slim AS builder
WORKDIR /app
ENV PIP_NO_CACHE_DIR=1 PIP_DISABLE_PIP_VERSION_CHECK=1
COPY requirements.txt .
RUN pip install --user -r requirements.txt

# ---- Runtime stage: slim, non-root, only what's needed to serve ----
FROM python:3.10-slim AS runtime

# libgomp1 is required at runtime by LightGBM's OpenMP-linked shared library.
RUN apt-get update \
    && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*

RUN useradd --create-home --uid 1000 appuser
WORKDIR /app

COPY --from=builder /root/.local /home/appuser/.local
COPY src/ ./src/
COPY scripts/ ./scripts/
COPY pyproject.toml ./

ENV PATH=/home/appuser/.local/bin:$PATH \
    PYTHONPATH=/app/src \
    PYTHONUNBUFFERED=1 \
    API_HOST=0.0.0.0 \
    API_PORT=8000

RUN mkdir -p /app/data/processed /app/data/raw /app/reports && chown -R appuser:appuser /app
USER appuser

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health', timeout=3)" || exit 1

CMD ["uvicorn", "demand_forecast.api.main:app", "--host", "0.0.0.0", "--port", "8000"]

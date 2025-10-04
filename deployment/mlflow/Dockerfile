# syntax=docker/dockerfile:1
FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    MLFLOW_HOST=0.0.0.0 \
    MLFLOW_PORT=5000 \
    MLFLOW_DB_PATH=/app/db/mlflow.db \
    MLFLOW_ARTIFACT_ROOT=/app/artifacts

WORKDIR /app

RUN apt-get update \
    && apt-get install --no-install-recommends -y build-essential curl \
    && pip install --no-cache-dir mlflow==3.4.0 gunicorn \
    && pip install --no-cache-dir google-api-core==2.25.2 gunicorn \
    && pip install --no-cache-dir google-auth==2.41.1 gunicorn \
    && pip install --no-cache-dir google-cloud-core==2.4.3 gunicorn \
    && pip install --no-cache-dir google-cloud-storage==3.4.0 gunicorn \
    && apt-get purge -y --auto-remove build-essential \
    && rm -rf /var/lib/apt/lists/*

RUN mkdir -p /app/db /app/artifacts

COPY entrypoint.sh /app/entrypoint.sh
RUN chmod +x /app/entrypoint.sh

VOLUME ["/app/artifacts"]

ENTRYPOINT ["/app/entrypoint.sh"]

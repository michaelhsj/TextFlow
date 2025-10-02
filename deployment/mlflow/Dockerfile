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
    && pip install --no-cache-dir mlflow==2.17.1 gunicorn \
    && apt-get purge -y --auto-remove build-essential \
    && rm -rf /var/lib/apt/lists/*

RUN mkdir -p /app/db /app/artifacts

COPY entrypoint.sh /app/entrypoint.sh
RUN chmod +x /app/entrypoint.sh

VOLUME ["/app/artifacts"]

ENTRYPOINT ["/app/entrypoint.sh"]

# syntax=docker/dockerfile:1
FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    DAGSTER_PROJECT_DIR=/opt/textflow/app \
    DAGSTER_HOME=/opt/textflow/dagster_home \
    DAGSTER_MODULE=pipeline.definitions

WORKDIR ${DAGSTER_PROJECT_DIR}

RUN apt-get update \
    && apt-get install --no-install-recommends -y build-essential curl \
    && pip install --no-cache-dir --upgrade pip \
    && rm -rf /var/lib/apt/lists/*

# Copy the full project so we can install its Python package and dependencies.
COPY pyproject.toml README.md ${DAGSTER_PROJECT_DIR}/
COPY ml ${DAGSTER_PROJECT_DIR}/ml
COPY pipeline ${DAGSTER_PROJECT_DIR}/pipeline

RUN pip install --no-cache-dir dagster==1.11.13 docker==6.1.3 \
    && pip install --no-cache-dir ${DAGSTER_PROJECT_DIR} \
    && pip install --no-cache-dir ${DAGSTER_PROJECT_DIR}/pipeline \
    && apt-get purge -y --auto-remove build-essential \
    && rm -rf /var/lib/apt/lists/* /root/.cache/pip

ENV PYTHONPATH=${DAGSTER_PROJECT_DIR}:${PYTHONPATH}

VOLUME ["${DAGSTER_HOME}"]

ENTRYPOINT ["/bin/bash"]

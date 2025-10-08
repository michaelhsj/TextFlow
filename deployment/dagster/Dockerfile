# syntax=docker/dockerfile:1
FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    DAGSTER_HOME=/opt/dagster/dagster_home \
    DAGSTER_PROJECT_DIR=/opt/dagster/app \
    DAGSTER_MODULE=pipeline.definitions \
    DAGSTER_PORT=3000

WORKDIR ${DAGSTER_PROJECT_DIR}

# Install system dependencies and Python packages required to run Dagster
RUN apt-get update \
    && apt-get install --no-install-recommends -y build-essential curl \
    && pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir dagster==1.11.13 dagster-webserver==1.11.13 docker==6.1.3 \
    && apt-get purge -y --auto-remove build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy the Dagster project into the image and install it so its dependencies are available.
COPY pipeline ${DAGSTER_PROJECT_DIR}/pipeline
RUN pip install --no-cache-dir ${DAGSTER_PROJECT_DIR}/pipeline \
    && rm -rf /root/.cache/pip

# Include default instance configuration and ensure it is available at runtime.
COPY deployment/dagster/dagster.yaml ${DAGSTER_PROJECT_DIR}/dagster.yaml
COPY deployment/dagster/entrypoint.sh /opt/dagster/entrypoint.sh

RUN mkdir -p ${DAGSTER_HOME} \
    && chmod +x /opt/dagster/entrypoint.sh

VOLUME ["${DAGSTER_HOME}"]

ENTRYPOINT ["/opt/dagster/entrypoint.sh"]

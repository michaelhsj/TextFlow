import os
from typing import Optional

import dagster as dg
from dagster import AssetExecutionContext
import docker

DEFAULT_DOCKER_URL = "tcp://textflow-gpu-runner:2375"
DATASETS_DIR = os.getenv("DAGSTER_DATASETS_DIR", "/opt/textflow/datasets")


@dg.asset
def text_ocr_dataset(context: AssetExecutionContext) -> str:
    """Download the Text OCR dataset onto the GPU runner's local disk."""

    job_image = os.getenv("DAGSTER_JOB_IMAGE")
    docker_url = os.getenv("GPU_RUNNER_DOCKER_URL", DEFAULT_DOCKER_URL)

    if not job_image:
        raise dg.Failure("DAGSTER_JOB_IMAGE is not configured for Dagster.")

    dataset_path = os.path.join(DATASETS_DIR, "TextOCR")

    try:
        client = docker.DockerClient(base_url=docker_url)
    except docker.errors.DockerException as error:
        raise dg.Failure(
            description=(
                "Unable to connect to the GPU runner's Docker daemon. Ensure the "
                "gpu_runner instance is provisioned and exposing its Docker API."
            ),
            metadata={"error": str(error)},
        ) from error

    # Attempt to pull the image upfront so the latest tag is available.
    try:
        client.images.pull(job_image)
    except docker.errors.ImageNotFound:
        raise dg.Failure(
            description=f"Job image {job_image} not found in Artifact Registry."
        )
    except docker.errors.APIError as error:
        context.log.warning(
            f"Failed to pull {job_image} before execution: {error}"
        )

    container: Optional[docker.models.containers.Container] = None
    try:
        container = client.containers.run(
            image=job_image,
            command=[
                "python",
                "-m",
                "ml.ingest.download_textocr",
                f"--dataset-base-dir={dataset_path}",
            ],
            detach=True,
            remove=True,
            volumes={
                DATASETS_DIR: {
                    "bind": DATASETS_DIR,
                    "mode": "rw",
                }
            },
        )

        for raw_line in container.logs(stream=True, follow=True):
            context.log.info(raw_line.decode("utf-8", errors="ignore").rstrip())

        result = container.wait()
        exit_code = result.get("StatusCode", 1)
        if exit_code != 0:
            raise dg.Failure(
                description=f"Remote container exited with status {exit_code}",
                metadata={"status_code": exit_code},
            )
    except docker.errors.DockerException as error:
        raise dg.Failure(
            description=(
                "Failed to execute dataset ingestion on the GPU runner Docker host."
            ),
            metadata={"error": str(error)},
        ) from error
    finally:
        # When remove=True, the container is cleaned up automatically; this guard is for
        # the unlikely case where the container is still present.
        if container is not None:
            try:
                container.remove(force=True)
            except docker.errors.APIError:
                pass
        client.close()

    context.add_output_metadata({"path": dataset_path})
    return dataset_path

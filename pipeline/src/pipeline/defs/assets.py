import subprocess
import os
import dagster as dg
from dagster import AssetExecutionContext

ML_PYTHON = "../.venv-textflow/bin/python"


@dg.asset
def text_ocr_dataset(context: AssetExecutionContext) -> str:
    """Download the Text OCR dataset zip locally and return its path."""

    output_dir = os.path.join(context.instance.storage_directory(), "TextOCR")
    result = subprocess.run(
        [
            "bash",
            "-lc",
            f"{ML_PYTHON} -m ml.ingest.download_textocr --dataset-base-dir {output_dir}",
        ],
    )
    if result.returncode != 0:
        raise dg.Failure(
            description=f"Non-zero return code: {result.returncode}",
        )

    context.add_output_metadata({"path": output_dir})
    context.log.info(result.stdout)
    return output_dir

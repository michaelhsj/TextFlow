import argparse
import os

import matplotlib.pyplot as plt
import mlflow
import torch
from doctr.io import DocumentFile
from doctr.models import ocr_predictor


def load_models_from_mlflow(
    detection_model_name: str,
    recognition_model_name: str,
    detection_stage: str = "Production",
    recognition_stage: str = "Production",
):
    """
    Loads detection and recognition models from the MLflow Model Registry
    and combines them into a single OCR predictor.

    Args:
        detection_model_name: The registered name of the detection model.
        recognition_model_name: The registered name of the recognition model.
        detection_stage: The stage of the detection model to load (e.g., "Production").
        recognition_stage: The stage of the recognition model to load.

    Returns:
        A configured doctr ocr_predictor.
    """
    print("--- Loading models from MLflow ---")

    # Construct MLflow model URIs
    detection_model_uri = f"models:/{detection_model_name}/{detection_stage}"
    recognition_model_uri = f"models:/{recognition_model_name}/{recognition_stage}"

    print(f"Loading detection model from: {detection_model_uri}")
    try:
        detection_model = mlflow.pytorch.load_model(detection_model_uri)
    except mlflow.exceptions.MlflowException as e:
        print(f"Error loading detection model: {e}")
        print("Please ensure the model is registered in MLflow.")
        return None

    print(f"Loading recognition model from: {recognition_model_uri}")
    try:
        recognition_model = mlflow.pytorch.load_model(recognition_model_uri)
    except mlflow.exceptions.MlflowException as e:
        print(f"Error loading recognition model: {e}")
        print("Please ensure the model is registered in MLflow.")
        return None

    print("Models loaded successfully. Creating OCR predictor...")

    # Create the end-to-end predictor
    # The predictor handles all preprocessing and sequencing internally
    predictor = ocr_predictor(
        det_arch=detection_model, reco_arch=recognition_model, pretrained=False
    )

    # Move the predictor to the correct device (GPU if available)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    predictor = predictor.to(device)
    print(f"Predictor moved to device: {device}")

    return predictor


def main():
    parser = argparse.ArgumentParser(
        description="Run OCR inference on a single image or PDF."
    )
    parser.add_argument("file_path", type=str, help="Path to the image or PDF file.")
    args = parser.parse_args()

    # --- 1. Load Models ---
    # Replace these with the actual names you used when registering your models in MLflow
    predictor = load_models_from_mlflow(
        detection_model_name="dbnet_textocr_model",
        recognition_model_name="crnn_textocr_model", # Example name
    )

    if predictor is None:
        print("Exiting due to model loading failure.")
        return

    # --- 2. Load and Preprocess Document ---
    print(f"\n--- Processing document: {args.file_path} ---")
    if not os.path.exists(args.file_path):
        print(f"Error: File not found at '{args.file_path}'")
        return

    # DocumentFile.from_files handles both images and PDFs
    doc = DocumentFile.from_files([args.file_path])

    # --- 3. Run Detection and Recognition ---
    print("Running inference...")
    result = predictor(doc)
    print("Inference complete.")

    # --- 4. Display Results ---
    # The .show() method provides a simple way to visualize the results
    result.show(block=True, interactive=False)


if __name__ == "__main__":
    main()

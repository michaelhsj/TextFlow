from ml.ingest.textocr_to_torch import TextOCRDoctrDetDataset, doctr_detection_collate
from torch.utils.data import DataLoader
from doctr.models import db_resnet50, DBNet
from pathlib import Path
import torch
import os
import mlflow
import getpass

LR = 0.001
BATCH_SIZE = 4
MAX_STEPS = 1
MODEL_DIR = Path("model")
DATASET_DIR = Path("dataset/TextOCR")
NUM_SAMPLES = 32

dataset = TextOCRDoctrDetDataset(
    images_dir=DATASET_DIR / "train_val_images",
    json_path=DATASET_DIR / "TextOCR_0.1_train.json",
    num_samples=NUM_SAMPLES,
)
loader = DataLoader(
    dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=doctr_detection_collate
)

experiment_name = f"db_resnet50_text_ocr_{getpass.getuser()}"
if tracking_uri := os.getenv("MLFLOW_TRACKING_URI"):
    print("MLFlow Tracking URI (default):", tracking_uri)
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(experiment_name)
else:
    raise RuntimeError("No MLFlow Tracking URI in env")

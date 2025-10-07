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
MODEL_DIR = Path("../../model")
DATASET_DIR = Path("../../dataset/TextOCR")
NUM_SAMPLES = 32


mlflow.set_tracking_uri("https://textflowocr.com/mlflow")
mlflow.set_experiment(f"db_resnet50_text_ocr_{getpass.getuser()}")
run_name = os.getenv("MLFLOW_RUN_NAME", "doctr-dbnet-lite")

dataset = TextOCRDoctrDetDataset(
    images_dir=DATASET_DIR / "train_val_images",
    json_path=DATASET_DIR / "TextOCR_0.1_train.json",
    num_samples=NUM_SAMPLES,
)
loader = DataLoader(
    dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=doctr_detection_collate
)

model: DBNet = db_resnet50(pretrained=False).train()
optimizer = torch.optim.Adam([p for p in model.parameters() if p.requires_grad], LR)

with mlflow.start_run(run_name=run_name) as run:
    mlflow.log_params(
        {
            "learning_rate": LR,
            "batch_size": BATCH_SIZE,
            "train_samples": len(dataset),
            "max_steps": MAX_STEPS,
        }
    )
    mlflow.set_tag("model_architecture", "db_resnet50")
    mlflow.set_tag("dataset", f"textocr_subset_{NUM_SAMPLES}")

    for step, (images, targets) in enumerate(loader, start=1):
        if step > MAX_STEPS:
            break

        optimizer.zero_grad(set_to_none=True)
        train_loss = model(images, targets)["loss"]
        train_loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5)
        optimizer.step()

        mlflow.log_metric("train_loss", train_loss.item(), step=step)
        print(step, train_loss.item())

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    model_path = MODEL_DIR / "dbnet_textocr.pt"
    torch.save(model.state_dict(), model_path)
    print("Local checkpoint stored at", MODEL_DIR / "dbnet_textocr.pt")

    mlflow.log_artifact(str(model_path))
    mlflow.pytorch.log_model(model, name="model")
    print("MLflow run logged:", run.info.run_id)

from ml.ingest.textocr_to_torch import TextOCRDoctrDetDataset, doctr_detection_collate
from torch.utils.data import DataLoader
from doctr.models import db_resnet50, DBNet
from pathlib import Path
import torch
import os
import mlflow
import getpass
import yaml
import argparse


def read_config(config_path: str) -> dict:
    with open(config_path, "r") as f:
        cf = yaml.safe_load(f)
    return cf

def train(config: dict):
    # Device selection (CUDA if available)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    DATASET_DIR = Path(config["path"]["dataset_dir"])

    # --- Datasets ---
    train_dataset = TextOCRDoctrDetDataset(
        images_dir= DATASET_DIR / config["path"]["image_dir"],
        json_path=DATASET_DIR / config["path"]["train_json"],
        num_samples=config["data"]["train_num_samples"],
    )
    val_dataset = TextOCRDoctrDetDataset(
        images_dir=DATASET_DIR / config["path"]["image_dir"],
        json_path=DATASET_DIR / config["path"]["val_json"],
        num_samples=config["data"]["val_num_samples"],
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=config["data"]["train_batch_size"],
        shuffle=True,
        collate_fn=doctr_detection_collate,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config["data"]["val_batch_size"],
        shuffle=False,
        collate_fn=doctr_detection_collate,
    )

    # --- MLFlow ---
    experiment_name = config["experiment_name"]
    if tracking_uri := os.getenv("MLFLOW_TRACKING_URI"):
        print("MLFlow Tracking URI (default):", tracking_uri)
        mlflow.set_tracking_uri(tracking_uri)
        mlflow.set_experiment(experiment_name)
    else:
        raise RuntimeError("No MLFlow Tracking URI in env")

    # --- Model ---
    model_config = config["model"]
    pretrained_flag = model_config["pretrain"]  # <--- change here when initializing
    model: DBNet = db_resnet50(pretrained=pretrained_flag).to(device).train()

    # --- Paths depending on pretrained ---
    run_tag = "pretrained" if pretrained_flag else "scratch"
    MODEL_DIR = Path(f"model/{run_tag}")
    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    optimizer = torch.optim.Adam([p for p in model.parameters() if p.requires_grad], model_config["lr"])

    run_name = os.getenv("MLFLOW_RUN_NAME", "doctr-dbnet-lite")

    best_val_loss = float("inf")
    global_step = 0

    with mlflow.start_run(run_name=run_name) as run:
        mlflow.log_params(
            {
                "learning_rate": model_config["lr"],
                "batch_size": config["data"]["train_batch_size"],
                "train_samples": len(train_dataset),
                "epochs": model_config["num_epochs"],
            }
        )
        mlflow.set_tag("model_architecture", "db_resnet50")
        mlflow.set_tag("dataset", f"textocr_subset_{config['data']['train_num_samples']}")

        model_path_best = MODEL_DIR / "dbnet_textocr_best.pt"
        for epoch in range(model_config["num_epochs"]):
            model.train()
            for step, (images, targets) in enumerate(train_loader, start=1):
                images = images.to(device)

                optimizer.zero_grad(set_to_none=True)
                train_loss = model(images, targets)["loss"]
                train_loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 5)
                optimizer.step()

                # print(batch, train_loss.item())
                if step % model_config["train_interval"] == 0:
                    mlflow.log_metric("train_loss", train_loss.item(), step=step)
                    print(
                        f"[Epoch {epoch+1}, Batch {step}] Train Loss: {train_loss.item():.4f}"
                    )

                # --- run validation periodically ---
                if global_step % model_config["val_interval"] == 0 and global_step > 0:
                    model.eval()
                    running_val_loss = 0.0
                    with torch.no_grad():
                        for val_images, val_targets in val_loader:
                            val_images = val_images.to(device)
                            val_loss = model(val_images, val_targets)["loss"]
                            running_val_loss += val_loss.item()

                    avg_val_loss = running_val_loss / len(val_loader)
                    mlflow.log_metric("val_loss", avg_val_loss, step=global_step)
                    print(
                        f"[Epoch {epoch+1}, Step {global_step}] Val Loss: {avg_val_loss:.4f}"
                    )

                    # Save best model
                    if avg_val_loss < best_val_loss:
                        best_val_loss = avg_val_loss
                        torch.save(model.state_dict(), model_path_best)
                        print(f"✅ Saved new best model with val loss {best_val_loss:.4f}")

                    model.train()

                global_step += 1

        model_path_latest = MODEL_DIR / "dbnet_textocr_latest.pt"
        torch.save(model.state_dict(), model_path_latest)
        mlflow.pytorch.log_model(model, name=config["model_uri"])

    print("MLflow run logged:", run.info.run_id)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str,
                        default=f"{os.path.dirname(__file__)}/detection_config.yaml",
                        help="Path to the config file for detecting training.")
    config = read_config(parser.parse_args().config)
    train(config)

if __name__ == "__main__":
    main()
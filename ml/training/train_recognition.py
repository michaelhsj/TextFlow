from ml.ingest.textocr_to_torch import TextOCRDoctrRecDataset, doctr_recognition_collate
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from doctr.datasets.utils import translate
from doctr.models import crnn_vgg16_bn, CRNN
from pathlib import Path
import torch

torch.manual_seed(42)

# Hyperparameters
LR = 0.001
TRAIN_BATCH_SIZE = 512
VALID_BATCH_SIZE = 256
NUM_EPOCHS = 99
VAL_INTERVAL = 20  # batches per validation check

# Device selection (CUDA if available)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# Datasets
train_dataset = TextOCRDoctrRecDataset(num_samples=TRAIN_BATCH_SIZE * 128)
val_dataset = TextOCRDoctrRecDataset(json_path=Path("dataset/TextOCR/TextOCR_0.1_val.json"),
                                     num_samples=VALID_BATCH_SIZE * 10)

train_loader = DataLoader(train_dataset, batch_size=TRAIN_BATCH_SIZE, shuffle=True,
                          collate_fn=doctr_recognition_collate)
val_loader = DataLoader(val_dataset, batch_size=VALID_BATCH_SIZE, shuffle=False,
                        collate_fn=doctr_recognition_collate)

# --- Model ---
pretrained_flag = True  # <--- change here when initializing
model: CRNN = crnn_vgg16_bn(pretrained=pretrained_flag).to(device).train()

# --- Paths depending on pretrained ---
run_tag = "pretrained" if pretrained_flag else "scratch"
MODEL_DIR = Path(f"model/{run_tag}")
MODEL_DIR.mkdir(parents=True, exist_ok=True)
writer = SummaryWriter(log_dir=f"runs/textocr_experiment_{run_tag}")

# Optimizer
optimizer = torch.optim.Adam([p for p in model.parameters() if p.requires_grad], LR)

best_val_loss = float("inf")
global_step = 0

for epoch in range(NUM_EPOCHS):
    model.train()
    for batch, (images, targets) in enumerate(train_loader):
        images = images.to(device)
        targets = [translate(target, "latin", "") for target in targets]

        optimizer.zero_grad(set_to_none=True)
        output = model(images, targets)
        train_loss = output["loss"]
        train_loss.backward()
        optimizer.step()

        # --- log training loss per batch ---
        writer.add_scalar("Loss/train", train_loss.item(), global_step)
        if batch % 50 == 0:
            print(f"[Epoch {epoch+1}, Batch {batch}] Train Loss: {train_loss.item():.4f}")

        # --- run validation periodically ---
        if global_step % VAL_INTERVAL == 0 and global_step > 0:
            model.eval()
            running_val_loss = 0.0
            with torch.no_grad():
                for val_images, val_targets in val_loader:
                    val_images = val_images.to(device)
                    val_targets = [translate(v, "latin", "") for v in val_targets]
                    val_loss = model(val_images, val_targets)["loss"]
                    running_val_loss += val_loss.item()

            avg_val_loss = running_val_loss / len(val_loader)
            writer.add_scalar("Loss/val", avg_val_loss, global_step)
            print(f"[Epoch {epoch+1}, Step {global_step}] Val Loss: {avg_val_loss:.4f}")

            # Save best model
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                torch.save(model.state_dict(), MODEL_DIR / "crnn_best.pt")
                print(f"✅ Saved new best model with val loss {best_val_loss:.4f}")

            model.train()

        global_step += 1

# Save final model
torch.save(model.state_dict(), MODEL_DIR / "crnn_last.pt")
writer.close()
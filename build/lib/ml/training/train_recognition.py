import argparse
from ml.ingest.textocr_to_torch import TextOCRDoctrRecDataset, doctr_recognition_collate
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from doctr.datasets.utils import translate
from doctr.models import crnn_vgg16_bn, CRNN
from pathlib import Path
import torch

def main():
    parser = argparse.ArgumentParser(description="Train CRNN TextOCR recognition model with TensorBoard logging.")

    # --- Training setup ---
    parser.add_argument("--lr", type=float, default=0.001, help="Learning rate.")
    parser.add_argument("--epochs", type=int, default=99, help="Number of training epochs.")
    parser.add_argument("--train-batch-size", type=int, default=512, help="Training batch size.")
    parser.add_argument("--val-batch-size", type=int, default=256, help="Validation batch size.")
    parser.add_argument("--val-interval", type=int, default=20, help="Validation frequency in training steps.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility.")

    # --- Model setup ---
    parser.add_argument("--pretrained", action="store_true", help="Use pretrained weights.")
    parser.add_argument("--model-name", type=str, default="crnn_vgg16_bn", help="Model architecture name.")

    # --- Dataset setup ---
    parser.add_argument("--train-samples", type=int, default=None, help="Number of training samples (optional).")
    parser.add_argument("--val-samples", type=int, default=None, help="Number of validation samples (optional).")
    parser.add_argument("--val-json-path", type=str, default="dataset/TextOCR/TextOCR_0.1_val.json",
                        help="Path to validation dataset JSON.")

    # --- Output setup ---
    parser.add_argument("--run-dir", type=str, default="runs", help="Base directory for TensorBoard runs.")
    parser.add_argument("--model-dir", type=str, default="model", help="Base directory for saved models.")
    parser.add_argument("--tag", type=str, default="", help="Custom tag for experiment name.")

    args = parser.parse_args()

    # --- Setup ---
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    train_samples = args.train_samples or args.train_batch_size * 128
    val_samples = args.val_samples or args.val_batch_size * 10

    # --- Dataset loading ---
    train_dataset = TextOCRDoctrRecDataset(num_samples=train_samples)
    val_dataset = TextOCRDoctrRecDataset(json_path=Path(args.val_json_path), num_samples=val_samples)

    train_loader = DataLoader(train_dataset, batch_size=args.train_batch_size, shuffle=True,
                              collate_fn=doctr_recognition_collate)
    val_loader = DataLoader(val_dataset, batch_size=args.val_batch_size, shuffle=False,
                            collate_fn=doctr_recognition_collate)

    # --- Model selection ---
    model: CRNN = crnn_vgg16_bn(pretrained=args.pretrained).to(device).train()

    # --- Directory setup ---
    run_tag = "pretrained" if args.pretrained else "scratch"
    if args.tag:
        run_tag += f"_{args.tag}"

    model_dir = Path(args.model_dir) / run_tag
    model_dir.mkdir(parents=True, exist_ok=True)
    writer = SummaryWriter(log_dir=f"{args.run-dir}/textocr_{run_tag}")

    optimizer = torch.optim.Adam([p for p in model.parameters() if p.requires_grad], args.lr)

    best_val_loss = float("inf")
    global_step = 0

    # --- Training loop ---
    for epoch in range(args.epochs):
        model.train()
        for batch, (images, targets) in enumerate(train_loader):
            images = images.to(device)
            targets = [translate(target, "latin", "") for target in targets]

            optimizer.zero_grad(set_to_none=True)
            output = model(images, targets)
            train_loss = output["loss"]
            train_loss.backward()
            optimizer.step()

            writer.add_scalar("Loss/train", train_loss.item(), global_step)
            if batch % 50 == 0:
                print(f"[Epoch {epoch+1}, Batch {batch}] Train Loss: {train_loss.item():.4f}")

            # Validation periodically
            if global_step % args.val_interval == 0 and global_step > 0:
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

                if avg_val_loss < best_val_loss:
                    best_val_loss = avg_val_loss
                    torch.save(model.state_dict(), model_dir / "crnn_best.pt")
                    print(f"✅ Saved new best model with val loss {best_val_loss:.4f}")
                model.train()

            global_step += 1

    torch.save(model.state_dict(), model_dir / "crnn_last.pt")
    writer.close()

if __name__ == "__main__":
    main()
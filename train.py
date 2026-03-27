import os
import time

import torch
from torch.optim import Adam
from torch.utils.data import DataLoader, random_split

from data import Flickr8kDataset, BATCH_SIZE
from models import ImageEncoder, TextEncoder, CLIPLoss

# --------------- Config ---------------
CAPTIONS_FILE = "dataset/captions.txt"
IMAGES_DIR = "dataset/Images"
CHECKPOINT_DIR = "checkpoints"
EPOCHS = 10
LR = 1e-4
LOG_EVERY = 50
VAL_SPLIT = 0.1
# --------------------------------------


def train_one_epoch(image_encoder, text_encoder, loss_fn, optimizer, dataloader, device, epoch):
    image_encoder.train()
    text_encoder.train()

    running_loss = 0.0
    for step, (images, input_ids, attention_mask) in enumerate(dataloader):
        images = images.to(device)
        input_ids = input_ids.to(device)
        attention_mask = attention_mask.to(device)

        image_emb = image_encoder(images)
        text_emb = text_encoder(input_ids, attention_mask)
        loss = loss_fn(image_emb, text_emb)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

        if (step + 1) % LOG_EVERY == 0:
            avg = running_loss / LOG_EVERY
            temp = loss_fn.log_temperature.exp().item()
            print(f"  Epoch {epoch+1} | Step {step+1:>4d} | Loss {avg:.4f} | Temp {temp:.4f}")
            running_loss = 0.0

    return running_loss


@torch.no_grad()
def validate(image_encoder, text_encoder, loss_fn, dataloader, device):
    image_encoder.eval()
    text_encoder.eval()

    total_loss = 0.0
    num_batches = 0
    for images, input_ids, attention_mask in dataloader:
        images = images.to(device)
        input_ids = input_ids.to(device)
        attention_mask = attention_mask.to(device)

        image_emb = image_encoder(images)
        text_emb = text_encoder(input_ids, attention_mask)
        loss = loss_fn(image_emb, text_emb)

        total_loss += loss.item()
        num_batches += 1

    return total_loss / num_batches


def main():
    if torch.cuda.is_available():
        device = "cuda"
    elif torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"
    print(f"Device: {device}")

    # --- Dataset & split ---
    print("Loading dataset...")
    dataset = Flickr8kDataset(CAPTIONS_FILE, IMAGES_DIR)
    val_size = int(len(dataset) * VAL_SPLIT)
    train_size = len(dataset) - val_size
    train_dataset, val_dataset = random_split(
        dataset, [train_size, val_size], generator=torch.Generator().manual_seed(42)
    )
    print(f"Train: {train_size}  Val: {val_size}")

    num_workers = 4 if torch.cuda.is_available() else 2
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True,
                              num_workers=num_workers, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False,
                            num_workers=num_workers, pin_memory=True)

    # --- Models & optimizer ---
    image_encoder = ImageEncoder().to(device)
    text_encoder = TextEncoder().to(device)
    loss_fn = CLIPLoss().to(device)

    optimizer = Adam(
        list(image_encoder.parameters()) + list(text_encoder.parameters()) + list(loss_fn.parameters()),
        lr=LR,
    )

    # --- Training ---
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    best_val_loss = float("inf")

    for epoch in range(EPOCHS):
        t0 = time.time()
        print(f"\n{'='*50}")
        print(f"Epoch {epoch+1}/{EPOCHS}")
        print(f"{'='*50}")

        train_one_epoch(image_encoder, text_encoder, loss_fn, optimizer, train_loader, device, epoch)

        val_loss = validate(image_encoder, text_encoder, loss_fn, val_loader, device)
        elapsed = time.time() - t0
        print(f"  Val Loss: {val_loss:.4f} | Time: {elapsed:.1f}s")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            checkpoint = {
                "epoch": epoch + 1,
                "val_loss": val_loss,
                "image_encoder": image_encoder.state_dict(),
                "text_encoder": text_encoder.state_dict(),
                "loss_fn": loss_fn.state_dict(),
                "optimizer": optimizer.state_dict(),
            }
            path = os.path.join(CHECKPOINT_DIR, "best_model.pt")
            torch.save(checkpoint, path)
            print(f"  ** New best model saved (val_loss={val_loss:.4f}) **")

    print(f"\nTraining complete. Best val loss: {best_val_loss:.4f}")


if __name__ == "__main__":
    main()

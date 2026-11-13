import os
import csv
import time
import torch
import torch.nn.functional as F
from torch.optim import Adam
from configurations.configuration import (
    DEVICE,
    SEED,
    LEARNING_RATE,
    EPOCHS_VAE,
    CHECKPOINT_DIR,
    METRIC_DIR
)
from helper.seed import set_seed
from data.dataset import get_data_loaders
from models.vae import VAE

def train_vae():
    set_seed(SEED)

    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    os.makedirs(METRIC_DIR, exist_ok=True)

    train_loader, val_loader, _ = get_data_loaders()
    model = VAE().to(DEVICE)
    optimizer = Adam(model.parameters(),lr=LEARNING_RATE)

    best_val_loss = float("inf")
    history = []

    print("Device:", DEVICE)
    print("Epochs:", EPOCHS_VAE)

    for epoch in range(EPOCHS_VAE):
        start_time = time.perf_counter()

        #train
        model.train()

        train_loss = 0.0
        total_reconstruction_loss = 0.0
        total_kl_loss = 0.0
        train_samples = 0

        for images, _ in train_loader:
            images = images.to(DEVICE)
            reconstructed, mu, logvar = model(images)
            batch_size = images.size(0)
            reconstruction_loss = F.mse_loss(reconstructed,images,reduction="sum") / batch_size

            kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp()) / batch_size

            loss = reconstruction_loss + kl_loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * batch_size
            total_reconstruction_loss += (reconstruction_loss.item() * batch_size)

            total_kl_loss += kl_loss.item() * batch_size
            train_samples += batch_size

        train_loss /= train_samples
        avg_reconstruction_loss = (total_reconstruction_loss / train_samples)
        avg_kl_loss = total_kl_loss / train_samples

        #val
        model.eval()

        val_loss = 0.0
        val_reconstruction_loss = 0.0
        val_kl_loss = 0.0
        val_samples = 0

        with torch.no_grad():
            for images, _ in val_loader:
                images = images.to(DEVICE)
                reconstructed, mu, logvar = model(images)
                batch_size = images.size(0)

                reconstruction_loss = F.mse_loss(reconstructed,images,reduction="sum") / batch_size
                kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp()) / batch_size

                loss = reconstruction_loss + kl_loss

                val_loss += loss.item() * batch_size
                val_reconstruction_loss += (reconstruction_loss.item() * batch_size)
                val_kl_loss += kl_loss.item() * batch_size
                val_samples += batch_size

        val_loss /= val_samples
        val_reconstruction_loss /= val_samples
        val_kl_loss /= val_samples

        epoch_time = time.perf_counter() - start_time

        #save train history
        history.append({
            "epoch": epoch + 1,
            "train_loss": train_loss,
            "train_reconstruction_loss": avg_reconstruction_loss,
            "train_kl_loss": avg_kl_loss,
            "val_loss": val_loss,
            "val_reconstruction_loss": val_reconstruction_loss,
            "val_kl_loss": val_kl_loss,
            "epoch_time_seconds": epoch_time
        })

        print(f"\nEpoch [{epoch + 1}/{EPOCHS_VAE}]")

        print(
            f"Train| Total: {train_loss:.4f} "
            f"| Reconstruction: {avg_reconstruction_loss:.4f} "
            f"| KL: {avg_kl_loss:.4f}"
        )

        print(
            f"Val| Total: {val_loss:.4f} "
            f"| Reconstruction: {val_reconstruction_loss:.4f} "
            f"| KL: {val_kl_loss:.4f}"
        )
        print(f"Time: {epoch_time:.2f} seconds")

        #save best checkpoint
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(
                {
                    "epoch": epoch + 1,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "val_loss": best_val_loss,
                    "val_reconstruction_loss": val_reconstruction_loss,
                    "val_kl_loss": val_kl_loss
                },
                os.path.join(CHECKPOINT_DIR, "vae_best.pth")
            )

    #save
    history_path = os.path.join(METRIC_DIR,"vae_history.csv")

    fieldnames = [
        "epoch",
        "train_loss",
        "train_reconstruction_loss",
        "train_kl_loss",
        "val_loss",
        "val_reconstruction_loss",
        "val_kl_loss",
        "epoch_time_seconds"
    ]

    with open(history_path, "w", newline="") as file:
        writer = csv.DictWriter(file,fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(history)

    print("\nTraining Completed")
    print(f"Best Validation Loss: {best_val_loss:.4f}")
    print("Checkpoint saved:", os.path.join(CHECKPOINT_DIR, "vae_best.pth"))
    print("Training history saved:", history_path)

if __name__ == "__main__":
    train_vae()

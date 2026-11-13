import os
import csv
import time
import torch
import torch.nn as nn
from torch.optim import Adam
from configurations.configuration import (
    DEVICE,
    SEED,
    LEARNING_RATE,
    EPOCHS_SPARSE_AE,
    SPARSITY_LAMBDA,
    CHECKPOINT_DIR,
    METRIC_DIR
)
from helper.seed import set_seed
from data.dataset import get_data_loaders
from models.sparse_ae import SparseAutoencoder

def train_sparse_ae():
    set_seed(SEED)
    #output dir
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    os.makedirs(METRIC_DIR, exist_ok=True)
    #load
    train_loader, val_loader, _ = get_data_loaders()
    #model
    model = SparseAutoencoder().to(DEVICE)
    #loss and optimizer
    criterion = nn.MSELoss()
    optimizer = Adam(model.parameters(), lr=LEARNING_RATE)
    best_val_loss = float("inf")
    history = []

    print("Device:", DEVICE)
    print("Epochs:", EPOCHS_SPARSE_AE)
    print("Sparsity Lambda:", SPARSITY_LAMBDA)

    for epoch in range(EPOCHS_SPARSE_AE):
        start_time = time.perf_counter()
        #train
        model.train()

        train_loss = 0.0
        train_samples = 0

        for images, _ in train_loader:
            images = images.to(DEVICE)
            #forward
            reconstructed, latent = model(
                images,
                return_latent=True
            )
            #calculate recons loss
            reconstruction_loss = criterion(
                reconstructed,
                images
            )
            #calculate sparsity penalty
            sparsity_loss = torch.mean(torch.abs(latent))

            #total loss
            loss = (
                reconstruction_loss
                + SPARSITY_LAMBDA * sparsity_loss
            )

            #backpropagation
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            batch_size = images.size(0)
            train_loss += loss.item() * batch_size
            train_samples += batch_size

        train_loss /= train_samples

        #validation
        model.eval()
        val_loss = 0.0
        val_samples = 0

        with torch.no_grad():
            for images, _ in val_loader:
                images = images.to(DEVICE)
                reconstructed, latent = model(images, return_latent=True)
                reconstruction_loss = criterion(reconstructed,images)

                sparsity_loss = torch.mean(torch.abs(latent))

                loss = (reconstruction_loss + SPARSITY_LAMBDA * sparsity_loss)

                batch_size = images.size(0)

                val_loss += loss.item() * batch_size
                val_samples += batch_size

        val_loss /= val_samples

        #train time
        epoch_time = time.perf_counter() - start_time
        history.append({
            "epoch": epoch + 1,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "epoch_time_seconds": epoch_time
        })
        print(
            f"Epoch [{epoch + 1}/{EPOCHS_SPARSE_AE}] "
            f"| Train Loss: {train_loss:.6f} "
            f"| Val Loss: {val_loss:.6f} "
            f"| Time: {epoch_time:.2f}s"
        )

        #save best model (val loss)
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(
                {
                    "epoch": epoch + 1,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "val_loss": best_val_loss
                },
                os.path.join(CHECKPOINT_DIR, "sparse_ae_best.pth")
            )

    history_path = os.path.join(METRIC_DIR, "sparse_ae_history.csv")

    with open(history_path, "w", newline="") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=[
                "epoch",
                "train_loss",
                "val_loss",
                "epoch_time_seconds"
            ]
        )

        writer.writeheader()
        writer.writerows(history)

    print("\nTraining Completed!")
    print(f"Best Validation Loss: {best_val_loss:.6f}")
    print("Checkpoint saved:", CHECKPOINT_DIR)
    print("Training history saved:", history_path)

if __name__ == "__main__":
    train_sparse_ae()

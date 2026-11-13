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
    EPOCHS_AE,
    CHECKPOINT_DIR,
    METRIC_DIR
)
from helper.seed import set_seed
from data.dataset import get_data_loaders
from models.vanilla_ae import Autoencoder

def train_autoencoder():
    set_seed(SEED)
    #output dir
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    os.makedirs(METRIC_DIR, exist_ok=True)
    #load data
    train_loader, val_loader, _ = get_data_loaders()
    #model
    model = Autoencoder().to(DEVICE)
    #loss and optimizer
    criterion = nn.MSELoss()
    optimizer = Adam(model.parameters(), lr=LEARNING_RATE)
    best_val_loss = float("inf")
    history = []

    print("Device:", DEVICE)
    print("Epochs:", EPOCHS_AE)

    #train loop
    for epoch in range(EPOCHS_AE):
        start_time = time.perf_counter()

        #training
        model.train()
        train_loss = 0.0
        train_samples = 0
        for images, _ in train_loader:
            images = images.to(DEVICE)
            #forward
            reconstructed = model(images)
            #calculate loss
            loss = criterion(reconstructed, images)
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
                reconstructed = model(images)
                loss = criterion(reconstructed, images)
                batch_size = images.size(0)
                val_loss += loss.item() * batch_size
                val_samples += batch_size
        val_loss /= val_samples
        
        #train time
        epoch_time = time.perf_counter() - start_time

        #save history
        history.append({
            "epoch": epoch + 1,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "epoch_time_seconds": epoch_time
        })

        print(
            f"Epoch [{epoch + 1}/{EPOCHS_AE}] "
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
                os.path.join(CHECKPOINT_DIR, "vanilla_ae_best.pth")
            )

    #save training history
    history_path = os.path.join(METRIC_DIR, "vanilla_ae_history.csv")

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

    print("\nTraining Completed")
    print(f"Best Validation Loss: {best_val_loss:.6f}")
    print("Checkpoint saved:", CHECKPOINT_DIR)
    print("Training history saved:", history_path)


if __name__ == "__main__":
    train_autoencoder()


import os
import csv
import time
import torch
import torch.nn as nn
from torch.optim import Adam
from configurations.configuration import (
    DEVICE,
    SEED,
    DDPM_LEARNING_RATE,
    EPOCHS_DDPM,
    DDPM_TIMESTEPS,
    CHECKPOINT_DIR,
    METRIC_DIR
)
from helper.seed import set_seed
from data.dataset import get_data_loaders
from models.ddpm import DDPM

def train_ddpm():
    set_seed(SEED)

    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    os.makedirs(METRIC_DIR, exist_ok=True)

    train_loader, val_loader, _ = get_data_loaders()

    model = DDPM().to(DEVICE)
    criterion = nn.MSELoss()
    optimizer = Adam(model.parameters(),lr=DDPM_LEARNING_RATE)

    best_val_loss = float("inf")
    history = []

    print("Device:", DEVICE)
    print("Epochs:", EPOCHS_DDPM)
    print("Timesteps:", DDPM_TIMESTEPS)

    for epoch in range(EPOCHS_DDPM):
        start_time = time.perf_counter()
        #train
        model.train()
        train_loss = 0.0
        train_samples = 0

        for images, _ in train_loader:
            images = images.to(DEVICE)
            batch_size = images.size(0)
            timesteps = torch.randint(0,DDPM_TIMESTEPS,(batch_size,),device=DEVICE)

            noisy_images, noise = model.q_sample(images,timesteps)
            predicted_noise = model.predict_noise(noisy_images,timesteps)

            loss = criterion(predicted_noise,noise)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

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
                batch_size = images.size(0)
                timesteps = torch.randint(0,DDPM_TIMESTEPS,(batch_size,),device=DEVICE)

                noisy_images, noise = model.q_sample(images,timesteps)
                predicted_noise = model.predict_noise(noisy_images,timesteps)

                loss = criterion(predicted_noise,noise)

                val_loss += loss.item() * batch_size
                val_samples += batch_size

        val_loss /= val_samples

        epoch_time = time.perf_counter() - start_time

        history.append({
            "epoch": epoch + 1,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "epoch_time_seconds": epoch_time
        })

        print(
            f"Epoch [{epoch + 1}/{EPOCHS_DDPM}] "
            f"| Train Loss: {train_loss:.6f} "
            f"| Val Loss: {val_loss:.6f} "
            f"| Time: {epoch_time:.2f}s"
        )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(
                {
                    "epoch": epoch + 1,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "val_loss": best_val_loss
                },
                os.path.join(CHECKPOINT_DIR,"ddpm_best.pth")
            )

    #save history
    history_path = os.path.join(METRIC_DIR,"ddpm_history.csv")

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

if __name__ == "__main__":
    train_ddpm()

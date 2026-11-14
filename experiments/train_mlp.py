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
    EPOCHS_MLP,
    CHECKPOINT_DIR,
    METRIC_DIR
)
from helper.seed import set_seed
from data.dataset import get_data_loaders
from models.mlp_classifier import MLPClassifier

def train_mlp():
    set_seed(SEED)

    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    os.makedirs(METRIC_DIR, exist_ok=True)

    train_loader, val_loader, _ = get_data_loaders()

    model = MLPClassifier().to(DEVICE)
    criterion = nn.CrossEntropyLoss()
    optimizer = Adam(model.parameters(),lr=LEARNING_RATE)

    best_val_accuracy = 0.0
    history = []

    print("Device:", DEVICE)
    print("Epochs:", EPOCHS_MLP)

    for epoch in range(EPOCHS_MLP):
        start_time = time.perf_counter()

        #training
        model.train()

        train_loss = 0.0
        train_correct = 0
        train_samples = 0

        for images, labels in train_loader:
            images = images.to(DEVICE)
            labels = labels.to(DEVICE)

            logits = model(images)
            loss = criterion(logits, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            predictions = torch.argmax(logits,dim=1)

            batch_size = images.size(0)

            train_loss += loss.item() * batch_size
            train_correct += (predictions == labels).sum().item()

            train_samples += batch_size

        train_loss /= train_samples
        train_accuracy = train_correct / train_samples

        #validation
        model.eval()

        val_loss = 0.0
        val_correct = 0
        val_samples = 0

        with torch.no_grad():
            for images, labels in val_loader:

                images = images.to(DEVICE)
                labels = labels.to(DEVICE)

                logits = model(images)
                loss = criterion(logits, labels)

                predictions = torch.argmax(logits,dim=1)
                batch_size = images.size(0)

                val_loss += loss.item() * batch_size
                val_correct += (predictions == labels).sum().item()
                val_samples += batch_size

        val_loss /= val_samples
        val_accuracy = val_correct / val_samples

        epoch_time = time.perf_counter() - start_time

        history.append({
            "epoch": epoch + 1,
            "train_loss": train_loss,
            "train_accuracy": train_accuracy,
            "val_loss": val_loss,
            "val_accuracy": val_accuracy,
            "epoch_time_seconds": epoch_time
        })

        print(
            f"Epoch [{epoch + 1}/{EPOCHS_MLP}] "
            f"| Train Loss: {train_loss:.4f} "
            f"| Train Acc: {train_accuracy * 100:.2f}% "
            f"| Val Loss: {val_loss:.4f} "
            f"| Val Acc: {val_accuracy * 100:.2f}% "
            f"| Time: {epoch_time:.2f}s"
        )

        #save best model (val accuracy)

        if val_accuracy > best_val_accuracy:
            best_val_accuracy = val_accuracy
            torch.save(
                {
                    "epoch": epoch + 1,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "val_accuracy": best_val_accuracy
                },
                os.path.join(CHECKPOINT_DIR,"mlp_best.pth")
            )

    #save
    history_path = os.path.join(METRIC_DIR,"mlp_history.csv")
    with open(history_path, "w", newline="") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=[
                "epoch",
                "train_loss",
                "train_accuracy",
                "val_loss",
                "val_accuracy",
                "epoch_time_seconds"
            ]
        )
        writer.writeheader()
        writer.writerows(history)

    print("\nTraining Completed!")
    print(f"Best Validation Accuracy: {best_val_accuracy * 100:.2f}%")

if __name__ == "__main__":
    train_mlp()